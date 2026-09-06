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

from linkrag.core import load_config, setup_logging
from linkrag.eval import describe_locator, format_modality_distribution, gold_coverage, gold_hits
from linkrag.generate.answer import answer, cited_ids
from linkrag.index import Index, default_encoder
from linkrag.retrieve.baseline import retrieve_scored

QUESTIONS = Path("tests/regression/pilot01_questions.jsonl")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, help='label for this run, e.g. "phase2 alignment"')
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--index", default=None)
    ap.add_argument("--mode", choices=["baseline", "linkrag"], default="baseline")
    ap.add_argument("--questions", default=str(QUESTIONS))
    ap.add_argument("--out", default="reports/regression.md")
    ap.add_argument("--no-llm", action="store_true",
                    help="retrieval instruments only; skips generation and its cost")
    args = ap.parse_args(argv)

    setup_logging()
    cfg = load_config(args.config)
    index = Index.load(args.index or cfg["index"]["store_dir"])
    encoder = default_encoder(index.embedding_model, cfg["device"], index.normalize)
    encoder([""])

    rows = [json.loads(line) for line in Path(args.questions).read_text().splitlines() if line.strip()]
    results = []
    for row in rows:
        scored = retrieve_scored(row["question"], index, encoder=encoder,
                                 top_k=cfg["retrieve"]["top_k"],
                                 candidates=cfg["retrieve"]["candidates"],
                                 rrf_k=cfg["retrieve"]["rrf_k"])
        units = [u for u, _ in scored]
        found, missed = gold_hits(units, row["gold_units"])
        text = "" if args.no_llm else answer(row["question"], units, cfg, mode=args.mode)

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
            "cited": len(cited_ids(text)) if text else 0,
            "n": len(units),
        })

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
        "| Q | type | modality distribution | gold evidence | gold missed | gold terms in answer |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r['qid']} | `{r['type']}` | {r['modality']} | **{r['gold']}** "
            f"| {', '.join(r['missed']) or '—'} | {r['terms']} |"
        )
    lines.append("")
    if not args.no_llm:
        for r in results:
            lines += [f"<details><summary>{r['qid']} answer ({r['cited']} citations)</summary>",
                      "", r["answer"], "", "</details>", ""]
    with out.open("a") as fh:
        fh.write("\n".join(lines) + "\n")

    print(f"appended to {out}")
    for r in results:
        print(f"  {r['qid']} {r['type']:20} {r['modality']:24} gold={r['gold']:8} "
              f"terms={r['terms']}  missed={', '.join(r['missed']) or '-'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
