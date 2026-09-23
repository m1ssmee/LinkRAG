#!/usr/bin/env python3
"""Standing instrument #2: re-run the pilot01 Q1-Q4 set and append to reports/regression.md.

Every phase runs this so the same four questions can be shown improving (or not)
over time. Gold evidence is matched by file+page / file+time-overlap, never by
unit id -- ids are regenerated whenever ASR or chunking settings change.

    python scripts/run_regression.py --phase "phase2 audio-slide alignment"
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from linkrag.core import load_config, set_max_cost, setup_logging
from linkrag.costs import record_run
from linkrag.eval import describe_locator, format_modality_distribution, gold_coverage, gold_hits
from linkrag.generate.answer import answer, answer_json, cited_ids, http_completer
from linkrag.eval.verify_gold import entailment_opts
from linkrag.generate.verify import citation_correctness, hallucination_rate, verify_answer
from linkrag.index import Index, default_encoder
from linkrag.manifest import MANIFEST_NAME, check_gold_manifest, load_manifest
from linkrag.link.align import load_links
from linkrag.link.graph import build_graph
from linkrag.retrieve.baseline import retrieve_scored
from linkrag.retrieve.linkrag import expansion_report, retrieve_linkrag

QUESTIONS = Path("tests/regression/pilot01_questions.jsonl")


def retrieve_for_mode(mode, question, index, *, encoder, graph, cfg):
    """Dispatch on mode. Returns (units, expanded_count, seeds_with_edges).

    This function exists so the mode actually reaches the retriever. It previously
    did not: --mode only changed a prompt suffix while retrieval stayed baseline,
    so a report could label itself `linkrag` over baseline retrieval.
    """
    if mode == "linkrag":
        lcfg = cfg["retrieve"]["linkrag"]
        results = retrieve_linkrag(
            question, index, graph, encoder=encoder, mode="linkrag",
            k_seed=lcfg["k_seed"], k_final=lcfg["k_final"], hops=lcfg["hops"],
            link_types=lcfg["link_types"], min_link_score=lcfg["min_link_score"],
            decay=lcfg["decay"], candidates=cfg["retrieve"]["candidates"],
            rrf_k=cfg["retrieve"]["rrf_k"],
            normalise_seeds=lcfg.get("normalise_seeds", False),
            expansion=cfg["retrieve"].get("expansion", "additive"))
        # additive returns the whole pool; this runner has no reranker, so "by score"
        results = results[:lcfg["k_final"]]
        expanded, seeded = expansion_report(results, graph)
        return [r.unit for r in results], expanded, seeded

    scored = retrieve_scored(question, index, encoder=encoder,
                             top_k=cfg["retrieve"]["top_k"],
                             candidates=cfg["retrieve"]["candidates"],
                             rrf_k=cfg["retrieve"]["rrf_k"])
    return [u for u, _ in scored], 0, 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, help='label for this run, e.g. "phase2 alignment"')
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--max-cost", type=float, default=0.0,
                        help="USD budget for LLM calls this run; 0 = zero-cost mode (refuse anything that bills)")
    ap.add_argument("--index", default=None)
    ap.add_argument("--mode", choices=["baseline", "linkrag"], default="baseline")
    ap.add_argument("--questions", default=str(QUESTIONS))
    ap.add_argument("--links", default=None, help="links.jsonl for --mode linkrag")
    ap.add_argument("--out", default="reports/regression.md")
    ap.add_argument("--strict", action="store_true", help="drop unsupported claims; abstain if none survive")
    ap.add_argument("--prose", action="store_true", help="Phase-1 prose answers instead of verified claims")
    ap.add_argument("--no-llm", action="store_true",
                    help="retrieval instruments only; skips generation and its cost")
    args = ap.parse_args(argv)

    setup_logging()
    cfg = load_config(args.config)
    set_max_cost(cfg, args.max_cost)
    index = Index.load(args.index or cfg["index"]["store_dir"])
    encoder = default_encoder(index.embedding_model, cfg["device"], index.normalize)
    encoder([""])

    raw = [json.loads(line) for line in Path(args.questions).read_text().splitlines() if line.strip()]
    gold_meta = next((r["_meta"] for r in raw if "_meta" in r), {})
    rows = [r for r in raw if "_meta" not in r]
    expansion_warnings: list[str] = []

    index_dir = Path(args.index or cfg["index"]["store_dir"])
    manifest = load_manifest(index_dir.parent / MANIFEST_NAME)
    graph = None
    if args.mode == "linkrag":
        links_path = args.links or cfg["link"]["align"]["links_path"]
        graph = build_graph(list(index.units),
                            load_links(links_path, expect_manifest=(manifest or {}).get("hash")))
    warning = check_gold_manifest(gold_meta.get("manifest_hash"), manifest)
    if warning:
        print(f"WARNING: {warning}")
    complete = None if args.no_llm else http_completer(cfg["models"]["llm"])
    gcfg = cfg.get("generation", {})
    structured = gcfg.get("structured", True) and not args.prose and complete is not None
    judge = None
    if structured and gcfg.get("verify", True):
        jcfg = cfg.get("eval", {}).get("judge") or cfg["models"]["llm"]
        judge = http_completer(jcfg)
    verified: list[dict] = []
    results = []
    for row in rows:
        units, expanded, seeded = retrieve_for_mode(
            args.mode, row["question"], index, encoder=encoder, graph=graph, cfg=cfg)
        if args.mode == "linkrag" and expanded == 0 and seeded:
            msg = (f"{row['qid']}: linkrag expanded 0 units although {seeded} seed(s) "
                   f"have graph edges -- link_types or min_link_score filtered "
                   f"everything; results are identical to baseline")
            print(f"WARNING: {msg}")
            expansion_warnings.append(msg)
        found, missed = gold_hits(units, row["gold_units"])
        verdict = None
        if structured:
            ans = answer_json(row["question"], units, cfg, mode=args.mode, complete=complete)
            if judge is not None and not ans.malformed:
                verdict = verify_answer(ans, units, judge, runs=int(gcfg.get("verify_runs", 3)),
                                        strict=args.strict or gcfg.get("strict", False), **entailment_opts(cfg))
                verdict["gold_units"] = row["gold_units"]
                verified.append(verdict)
                text = verdict["answer"]
            else:
                text = ans.text(strict=args.strict)
        else:
            text = "" if args.no_llm else answer(row["question"], units, cfg, mode=args.mode,
                                                 complete=complete)

        blob = text.lower()
        want = row["gold_terms"]["slide"] + row["gold_terms"]["audio"]
        in_answer = [t for t in want if t in blob]
        results.append({
            "qid": row["qid"], "type": row["type"],
            "modality": format_modality_distribution(units),
            "gold": gold_coverage(units, row["gold_units"]),
            "found": [describe_locator(x) for x in found],
            "missed": [describe_locator(x) for x in missed],
            "terms": "—" if args.no_llm else f"{len(in_answer)}/{len(want)}",
            "answer": text.strip(),
            "cited": (len({i for c in verdict["claims"] for i in c["unit_ids"]}) if verdict
                      else (len(cited_ids(text)) if text else 0)),
            "claims": len(verdict["claims"]) if verdict else 0,
            "unsupported": verdict["unsupported"] if verdict else 0,
            "abstained": bool(verdict and verdict["abstained"]),
            "n": len(units),
            "expanded": expanded,
        })

    def claims_cell(r: dict) -> str:
        if not r["claims"]:
            return r["terms"]
        cell = str(r["claims"])
        if r["unsupported"]:
            cell += f" ({r['unsupported']} unsupported)"
        return cell + (" · ABSTAINED" if r["abstained"] else "")

    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    new = not out.exists()
    lines = []
    if new:
        lines += [
            "# Regression — pilot01 Q1–Q4",
            "",
            "The same four questions, re-run every phase. Gold evidence is matched by",
            "file+page or file+time-overlap, not by unit id. `modality` is the retrieved",
            "set's composition — a gain that only reshuffles within one modality is not",
            "the cross-modal gain LinkRAG claims.",
            "",
        ]
    lines += [
        f"## {stamp} — {args.phase}",
        "",
        f"mode=`{args.mode}` · index=`{args.index or cfg['index']['store_dir']}` · "
        f"embedder=`{index.embedding_model}` · units={len(index)} · "
        f"top_k={cfg['retrieve']['top_k']}" + ("  · retrieval only (no LLM)" if args.no_llm else ""),
        "",
        f"corpus manifest `{(manifest or {}).get('hash', 'none')}` "
        f"({(manifest or {}).get('total_units', '?')} units) · "
        f"gold stamped `{gold_meta.get('manifest_hash', 'unstamped')}`",
        "",
    ] + ([f"> **WARNING** {warning}", ""] if warning else []) + [
        "| Q | type | modality distribution | expanded | gold evidence | gold missed | claims (unsupported) |",
        "|---|---|---|---:|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r['qid']} | `{r['type']}` | {r['modality']} | {r['expanded']} "
            f"| **{r['gold']}** | {', '.join(r['missed']) or '—'} | {claims_cell(r)} |"
        )
    if expansion_warnings:
        lines += [""] + [f"> **WARNING** {w}" for w in expansion_warnings]
    lines.append("")
    if not args.no_llm:
        for r in results:
            lines += [f"<details><summary>{r['qid']} answer ({r['cited']} citations)</summary>",
                      "", r["answer"], "", "</details>", ""]
    if verified:
        all_units = list(index.units)
        hr = hallucination_rate(verified)
        cc = citation_correctness(verified, all_units, [g for v in verified for g in v["gold_units"]])
        lines += ["", f"**Grounding** (`generation.strict={args.strict or gcfg.get('strict', False)}`): "
                  f"hallucination rate **{hr:.1%}** ({sum(v['unsupported'] for v in verified)} unsupported of "
                  f"{sum(len(v['claims']) for v in verified)} claims) · citation correctness **{cc:.1%}** "
                  f"(cited units matching a gold locator, by file+location) · "
                  f"{sum(v['abstained'] for v in verified)} answer(s) abstained · "
                  f"{sum(v['weak'] for v in verified)} claim(s) cited nothing", ""]
        print(f"  grounding: hallucination {hr:.1%} · citation correctness {cc:.1%} · "
              f"{sum(v['abstained'] for v in verified)} abstentions")

    if complete is not None:
        parts = [(str(cfg["models"]["llm"].get("model")), complete.usage)]
        if judge is not None:
            jm = str((cfg.get("eval", {}).get("judge") or cfg["models"]["llm"]).get("model"))
            parts.append((jm, judge.usage))
        footer = record_run("scripts/run_regression.py", args.phase, parts, cfg["models"].get("pricing"))
        lines += footer + [""]
        print("\n".join(footer))
    with out.open("a") as fh:
        fh.write("\n".join(lines) + "\n")

    print(f"appended to {out}")
    for r in results:
        print(f"  {r['qid']} {r['type']:20} {r['modality']:24} exp={r['expanded']:2} "
              f"gold={r['gold']:8} terms={r['terms']}  missed={', '.join(r['missed']) or '-'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
