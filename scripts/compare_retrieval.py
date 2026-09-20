#!/usr/bin/env python3
"""(retrieval mode x rerank method) matrix on a labelled question set.

Modes: baseline, iterative (P1), linkrag, linkrag_iter.
Rerank: none (plain top-k), mmr (text diversity), complementarity.
The greedy selection *replaces* the top-k cut: each mode retrieves `rerank.pool`
candidates and the reranker selects k from them.

**LLM measurement rule.** Cells whose mode calls an LLM are run `--repeats` times and
reported as mean +/- std; a single-run number for such a cell is refused, not printed.
Retrieval happens once per (mode, repeat) and all three rerank methods are applied to
that same candidate pool, so the rerank axis is not confounded by LLM variance.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from linkrag.core import load_config, setup_logging
from linkrag.eval import matches_locator
from linkrag.generate.answer import http_completer
from linkrag.index import Index, default_encoder
from linkrag.link.align import load_links
from linkrag.link.graph import build_graph
from linkrag.manifest import MANIFEST_NAME, load_manifest
from linkrag.retrieve.baseline import retrieve_scored
from linkrag.retrieve.iterative import retrieve_iterative, retrieve_linkrag_iter
from linkrag.retrieve.linkrag import RetrievedUnit, retrieve_linkrag
from linkrag.retrieve.rerank import METHODS, rerank, set_diagnostics

MODES = ("baseline", "iterative", "linkrag", "linkrag_iter")
LLM_MODES = {"iterative", "linkrag_iter"}


def gold_ids_for(row: dict, index: Index) -> set[str]:
    if row.get("gold_unit_ids"):
        return set(row["gold_unit_ids"])
    ids = set()
    for locator in row.get("gold_units", []):
        ids |= {u.id for u in index.units if matches_locator(u, locator)}
    return ids


def prf(retrieved_ids, gold):
    if not gold:
        return float("nan"), float("nan")
    hit = len(set(retrieved_ids) & gold)
    return hit / len(gold), hit / max(len(retrieved_ids), 1)


def retrieve_pool(mode, question, index, graph, *, encoder, cfg, complete, pool):
    """Candidates for one (mode, question). Returns (results, llm_calls, seconds)."""
    lcfg, icfg = cfg["retrieve"]["linkrag"], cfg["retrieve"]["iterative"]
    common = dict(candidates=cfg["retrieve"]["candidates"], rrf_k=cfg["retrieve"]["rrf_k"])
    norm = lcfg.get("normalise_seeds", False)
    t0 = time.perf_counter()

    if mode == "baseline":
        out = [RetrievedUnit(unit=u, score=s, origin="seed")
               for u, s in retrieve_scored(question, index, encoder=encoder,
                                           top_k=pool, **common)]
        return out, 0, time.perf_counter() - t0

    if mode == "linkrag":
        out = retrieve_linkrag(question, index, graph, encoder=encoder, mode="linkrag",
                               k_seed=lcfg["k_seed"], k_final=pool, hops=lcfg["hops"],
                               link_types=lcfg["link_types"],
                               min_link_score=lcfg["min_link_score"],
                               decay=lcfg["decay"], normalise_seeds=norm, **common)
        return out, 0, time.perf_counter() - t0

    if mode == "iterative":
        it = retrieve_iterative(question, index, encoder=encoder, complete=complete,
                                rounds=icfg["rounds"], k_per_round=icfg["k_per_round"],
                                k_final=pool, **common)
        return it.units, it.llm_calls, time.perf_counter() - t0

    out, it = retrieve_linkrag_iter(
        question, index, graph, encoder=encoder, complete=complete,
        rounds=icfg["rounds"], k_seed=lcfg["k_seed"], k_final=pool,
        hops=lcfg["hops"], link_types=lcfg["link_types"],
        min_link_score=lcfg["min_link_score"], decay=lcfg["decay"],
        normalise_seeds=norm, **common)
    return out, it.llm_calls, time.perf_counter() - t0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default="tests/regression/pilot01_questions.jsonl")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--index", default=None)
    ap.add_argument("--links", default=None)
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--repeats", type=int, default=3, help="runs per LLM-touching cell")
    ap.add_argument("--no-llm", action="store_true", help="skip modes that call an LLM")
    ap.add_argument("--rerank", default=",".join(METHODS),
                    help="comma-separated subset of rerank methods (default: all)")
    ap.add_argument("--label", default="", help="heading written above the appended table")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    setup_logging()
    cfg = load_config(args.config)
    lcfg, rcfg = cfg["retrieve"]["linkrag"], cfg["retrieve"]["rerank"]
    llm = cfg["models"]["llm"]
    k = args.k or lcfg["k_final"]
    pool = max(int(rcfg.get("pool", k)), k)

    index = Index.load(args.index or cfg["index"]["store_dir"])
    encoder = default_encoder(index.embedding_model, cfg["device"], index.normalize)
    encoder([""])
    index_dir = Path(args.index or cfg["index"]["store_dir"])
    manifest = load_manifest(index_dir.parent / MANIFEST_NAME) or {}
    graph = build_graph(list(index.units),
                        load_links(args.links or cfg["link"]["align"]["links_path"],
                                   expect_manifest=manifest.get("hash")))

    rows = [json.loads(l) for l in Path(args.questions).read_text().splitlines()
            if l.strip() and "_meta" not in l]
    complete = None if args.no_llm else http_completer(llm)
    modes = [m for m in MODES if not (args.no_llm and m in LLM_MODES)]
    methods = [m for m in METHODS if m in set(args.rerank.split(","))]
    if not methods:
        raise SystemExit(f"--rerank must name at least one of {METHODS}")
    gold_meta = next((json.loads(l)["_meta"] for l in Path(args.questions).read_text().splitlines()
                      if l.strip() and "_meta" in l), {})
    if gold_meta.get("manifest_hash") not in (None, manifest.get("hash")):
        print(f"WARNING: gold stamped {gold_meta.get('manifest_hash')}, corpus is {manifest.get('hash')}")

    # cell -> list of per-run dicts; a run is the mean over all questions
    cells: dict[tuple[str, str], list[dict]] = {}
    id_sets: dict[str, list[tuple]] = {}
    per_q: dict[tuple[str, str, str], list[float]] = {}   # (qid, mode, method) -> recall per run
    calls_total = 0

    for mode in modes:
        repeats = args.repeats if mode in LLM_MODES else 1
        for run in range(repeats):
            per_method = {m: {"recall": [], "prec": [], "mods": [], "red": []}
                          for m in methods}
            latencies, run_ids = [], []
            for row in rows:
                gold = gold_ids_for(row, index)
                results, calls, secs = retrieve_pool(
                    mode, row["question"], index, graph, encoder=encoder, cfg=cfg,
                    complete=complete, pool=pool)
                calls_total += calls
                latencies.append(secs)
                for method in methods:
                    picked = rerank(results, k, method=method, index=index, graph=graph,
                                    alpha=rcfg["alpha"], beta=rcfg["beta"],
                                    gamma=rcfg["gamma"], mmr_lambda=rcfg["mmr_lambda"],
                                    question=row["question"],
                                    cross_encoder=rcfg.get("cross_encoder"),
                                    device=cfg["device"])
                    ids = [r.id for r in picked]
                    r_, p_ = prf(ids, gold)
                    d = set_diagnostics(picked, index)
                    per_method[method]["recall"].append(r_)
                    per_q.setdefault((row["qid"], mode, method), []).append(r_)
                    per_method[method]["prec"].append(p_)
                    per_method[method]["mods"].append(d["modalities"])
                    per_method[method]["red"].append(d["redundancy"])
                    if method == methods[-1]:
                        run_ids.append(tuple(sorted(ids)))
            id_sets.setdefault(mode, []).append(tuple(run_ids))
            mean = lambda xs: sum(xs) / len(xs)
            for method in methods:
                cells.setdefault((mode, method), []).append(
                    {kk: mean(vv) for kk, vv in per_method[method].items()}
                    | {"latency": mean(latencies)})

    def fmt(values):
        if len(values) == 1:
            return f"{values[0]:.1%}"
        return f"{statistics.mean(values):.1%} ± {statistics.stdev(values):.1%}"

    print(f"\n{len(rows)} questions · k={k} · pool={pool} · corpus {len(index)} units "
          f"({manifest.get('hash', '?')})")
    print(f"backend: {llm.get('provider')} · model: {llm.get('model')} · "
          f"temperature={llm.get('temperature')} · seed={llm.get('seed')}")
    print(f"repeats: {args.repeats} for {sorted(LLM_MODES)}, 1 for deterministic modes "
          f"· total LLM calls: {calls_total}\n")

    header = (f"{'mode':<14}{'rerank':<17}{'recall@k':>16}{'prec@k':>16}"
              f"{'modalities':>12}{'redundancy':>12}")
    print(header); print("-" * len(header))
    md = [""] + ([f"## {args.label}", ""] if args.label else []) + [
          f"{len(rows)} questions · k={k} · pool={pool} · corpus `{manifest.get('hash', '?')}` "
          f"({len(index)} units) · gold stamped `{gold_meta.get('manifest_hash', 'unstamped')}` · "
          f"model `{llm.get('model')}` temperature={llm.get('temperature')} seed={llm.get('seed')} · "
          f"repeats {args.repeats} for LLM modes · {calls_total} LLM calls", "",
          f"| mode | rerank | recall@{k} | precision@{k} | distinct modalities "
          f"| redundancy | runs |", "|---|---|---:|---:|---:|---:|---:|"]
    for mode in modes:
        for method in methods:
            runs = cells[(mode, method)]
            if mode in LLM_MODES and len(runs) < 3:
                print(f"{mode:<14}{method:<17}{'REFUSED (n<3)':>16}")
                md.append(f"| {mode} | {method} | REFUSED (n<3) | | | | {len(runs)} |")
                continue
            rec = [r["recall"] for r in runs]
            pre = [r["prec"] for r in runs]
            mods = statistics.mean([r["mods"] for r in runs])
            red = statistics.mean([r["red"] for r in runs])
            print(f"{mode:<14}{method:<17}{fmt(rec):>16}{fmt(pre):>16}"
                  f"{mods:>12.2f}{red:>12.3f}")
            md.append(f"| {mode} | {method} | {fmt(rec)} | {fmt(pre)} | {mods:.2f} "
                      f"| {red:.3f} | {len(runs)} |")

    print(f"\nrepeat determinism ({methods[-1]} cell, identical returned id sets?):")
    md += ["", f"Repeat determinism ({methods[-1]} cell):", ""]
    for mode in modes:
        runs = id_sets[mode]
        if len(runs) == 1:
            verdict = "deterministic by construction (no LLM, single run)"
        else:
            verdict = ("IDENTICAL across runs" if len(set(runs)) == 1
                       else f"NOT identical — {len(set(runs))} distinct outcomes in {len(runs)} runs")
        print(f"  {mode:<14} {verdict}")
        md.append(f"- `{mode}`: {verdict}")

    # per-question recall, mean over runs, for every (mode, rerank) cell
    cols = [(m, r) for m in modes for r in methods]
    md += ["", "Per-question recall@k (mean over runs):", "",
           "| Q | type | " + " | ".join(f"{m}/{r}" for m, r in cols) + " |",
           "|---|---|" + "---:|" * len(cols)]
    for row in rows:
        vals = []
        for m, r in cols:
            xs = per_q.get((row["qid"], m, r), [])
            vals.append(f"{statistics.mean(xs):.0%}" if xs else "—")
        md.append(f"| {row['qid']} | {row.get('type', '')} | " + " | ".join(vals) + " |")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "a") as fh:
            fh.write("\n".join(md) + "\n")
        print(f"\nappended to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
