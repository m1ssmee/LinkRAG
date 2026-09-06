#!/usr/bin/env python3
"""Compare baseline vs iterative (P1) vs linkrag retrieval on a labelled question set.

    python scripts/compare_retrieval.py --questions tests/regression/pilot01_questions.jsonl

Input JSONL, one question per line, either:
  {"question": "...", "gold_unit_ids": ["id1", "id2"]}
or the regression format, whose `gold_units` locators (file+page / file+time) are
resolved against the index -- so the same gold file serves both harnesses.

Reports evidence recall@k and precision@k, plus what each method *costs*: LLM calls
and wall-clock. Iterative buys its extra evidence with an LLM call on the critical
path; link-following spends none. A recall win that ignores that is not a fair
comparison.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from linkrag.core import load_config, setup_logging
from linkrag.eval import matches_locator
from linkrag.generate.answer import http_completer
from linkrag.index import Index, default_encoder
from linkrag.link.align import load_links
from linkrag.manifest import MANIFEST_NAME, load_manifest
from linkrag.link.graph import build_graph
from linkrag.retrieve.baseline import retrieve_scored
from linkrag.retrieve.iterative import retrieve_iterative, retrieve_linkrag_iter
from linkrag.retrieve.linkrag import expansion_report, retrieve_linkrag


def gold_ids_for(row: dict, index: Index) -> set[str]:
    if row.get("gold_unit_ids"):
        return set(row["gold_unit_ids"])
    ids = set()
    for locator in row.get("gold_units", []):
        ids |= {u.id for u in index.units if matches_locator(u, locator)}
    return ids


def prf(retrieved_ids: list[str], gold: set[str]) -> tuple[float, float]:
    if not gold:
        return float("nan"), float("nan")
    hit = len(set(retrieved_ids) & gold)
    return hit / len(gold), hit / max(len(retrieved_ids), 1)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default="tests/regression/pilot01_questions.jsonl")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--index", default=None)
    ap.add_argument("--links", default=None)
    ap.add_argument("--k", type=int, default=None, help="k for all methods (default k_final)")
    ap.add_argument("--no-iterative", action="store_true", help="skip the LLM-spending methods")
    ap.add_argument("--normalise-seeds", action="store_true",
                    help="rank-normalise seed scores so expansion can outrank a weak seed")
    ap.add_argument("--out", default=None, help="also append a markdown table here")
    args = ap.parse_args(argv)

    setup_logging()
    cfg = load_config(args.config)
    lcfg = cfg["retrieve"]["linkrag"]
    icfg = cfg["retrieve"]["iterative"]
    k = args.k or lcfg["k_final"]

    index = Index.load(args.index or cfg["index"]["store_dir"])
    encoder = default_encoder(index.embedding_model, cfg["device"], index.normalize)
    encoder([""])
    index_dir = Path(args.index or cfg["index"]["store_dir"])
    manifest_hash = (load_manifest(index_dir.parent / MANIFEST_NAME) or {}).get("hash")
    graph = build_graph(list(index.units),
                        load_links(args.links or cfg["link"]["align"]["links_path"],
                                   expect_manifest=manifest_hash))

    rows = [json.loads(l) for l in Path(args.questions).read_text().splitlines()
            if l.strip() and "_meta" not in l]
    complete = None if args.no_iterative else http_completer(cfg["models"]["llm"])
    norm = args.normalise_seeds or lcfg.get("normalise_seeds", False)

    methods: dict[str, dict] = {}
    expansion: dict[str, tuple[int, int]] = {}

    def record(name, qid, recall, precision, latency, calls, ids):
        m = methods.setdefault(name, {"recall": [], "precision": [], "latency": [],
                                      "calls": 0, "per_q": {}})
        m["recall"].append(recall); m["precision"].append(precision)
        m["latency"].append(latency); m["calls"] += calls
        m["per_q"][qid] = (recall, ids)

    for row in rows:
        gold = gold_ids_for(row, index)
        qid = row.get("qid", row["question"][:24])

        t0 = time.perf_counter()
        base_ids = [u.id for u, _ in retrieve_scored(
            row["question"], index, encoder=encoder, top_k=k,
            candidates=cfg["retrieve"]["candidates"], rrf_k=cfg["retrieve"]["rrf_k"])]
        record("baseline", qid, *prf(base_ids, gold), time.perf_counter() - t0, 0, base_ids)

        t0 = time.perf_counter()
        link_results = retrieve_linkrag(
            row["question"], index, graph, encoder=encoder, mode="linkrag",
            k_seed=lcfg["k_seed"], k_final=k, hops=lcfg["hops"],
            link_types=lcfg["link_types"], min_link_score=lcfg["min_link_score"],
            decay=lcfg["decay"], candidates=cfg["retrieve"]["candidates"],
            rrf_k=cfg["retrieve"]["rrf_k"], normalise_seeds=norm)
        link_ids = [r.id for r in link_results]
        expanded, seeded = expansion_report(link_results, graph)
        expansion[qid] = (expanded, seeded)
        if expanded == 0 and seeded:
            print(f"WARNING: {qid}: linkrag expanded 0 units although {seeded} seed(s) "
                  f"have graph edges -- results are identical to baseline")
        record("linkrag", qid, *prf(link_ids, gold), time.perf_counter() - t0, 0, link_ids)

        if complete is not None:
            it = retrieve_iterative(row["question"], index, encoder=encoder,
                                    complete=complete, rounds=icfg["rounds"],
                                    k_per_round=icfg["k_per_round"], k_final=k,
                                    candidates=cfg["retrieve"]["candidates"],
                                    rrf_k=cfg["retrieve"]["rrf_k"])
            record("iterative (P1)", qid, *prf(it.ids, gold), it.latency_s,
                   it.llm_calls, it.ids)

            t0 = time.perf_counter()
            combo, combo_it = retrieve_linkrag_iter(
                row["question"], index, graph, encoder=encoder, complete=complete,
                rounds=icfg["rounds"], k_seed=lcfg["k_seed"], k_final=k,
                candidates=cfg["retrieve"]["candidates"], rrf_k=cfg["retrieve"]["rrf_k"],
                hops=lcfg["hops"], link_types=lcfg["link_types"],
                min_link_score=lcfg["min_link_score"], decay=lcfg["decay"],
                normalise_seeds=norm)
            combo_ids = [r.id for r in combo]
            expansion[qid] = expansion.get(qid, (0, 0))
            record("linkrag_iter", qid, *prf(combo_ids, gold),
                   time.perf_counter() - t0, combo_it.llm_calls, combo_ids)

    mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")
    order = [m for m in ("baseline", "iterative (P1)", "linkrag", "linkrag_iter")
             if m in methods]

    print(f"\n{len(rows)} questions · k={k} · corpus {len(index)} units · "
          f"normalise_seeds={norm}\n")
    header = f"{'method':<16}{'recall@k':>10}{'prec@k':>9}{'latency':>10}{'LLM calls':>11}"
    print(header); print("-" * len(header))
    lines = ["", f"| method | recall@{k} | precision@{k} | avg latency | LLM calls |",
             "|---|---:|---:|---:|---:|"]
    for name in order:
        m = methods[name]
        print(f"{name:<16}{mean(m['recall']):>9.1%}{mean(m['precision']):>9.1%}"
              f"{mean(m['latency']):>9.2f}s{m['calls']:>11}")
        lines.append(f"| {name} | {mean(m['recall']):.1%} | {mean(m['precision']):.1%} "
                     f"| {mean(m['latency']):.2f}s | {m['calls']} |")

    print(f"\nlinkrag expansion per question (expanded units / seeds with edges):")
    for qid, (exp, seeded) in expansion.items():
        flag = "  <-- NO EXPANSION" if exp == 0 and seeded else ""
        print(f"  {qid:<6} {exp:>2} / {seeded}{flag}")

    print(f"\nper-question recall@{k}:")
    print(f"  {'qid':<6}" + "".join(f"{n:>18}" for n in order))
    for row in rows:
        qid = row.get("qid", row["question"][:24])
        cells = "".join(f"{methods[n]['per_q'][qid][0]:>17.0%} " for n in order)
        print(f"  {qid:<6}{cells}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "a") as fh:
            fh.write("\n".join(lines) + "\n")
        print(f"\nappended table to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
