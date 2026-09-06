#!/usr/bin/env python3
"""Ask a question against an ingested index.

    python scripts/ask.py "what did the lecturer say about attention?" --mode baseline
"""

from __future__ import annotations

import argparse
from pathlib import Path

from linkrag.core import load_config, setup_logging, stage_timer
from linkrag.eval import format_modality_distribution
from linkrag.generate.answer import answer, cited_ids
from linkrag.index import Index, default_encoder
from linkrag.retrieve.baseline import retrieve_scored


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Question answering over an ingested corpus.")
    parser.add_argument("question")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--mode", choices=["baseline", "linkrag"], default="baseline")
    parser.add_argument("--index", default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--show-evidence", action="store_true")
    args = parser.parse_args(argv)

    setup_logging()
    cfg = load_config(args.config)
    index_dir = args.index or cfg["index"]["store_dir"]
    if not Path(index_dir).exists():
        parser.error(f"no index at {index_dir} -- run scripts/ingest.py first")

    if args.mode == "linkrag":
        parser.error("linkrag mode is not implemented yet; only --mode baseline works")

    with stage_timer("index.load", dir=index_dir) as t:
        index = Index.load(index_dir)
        t["units"] = len(index)

    encoder = default_encoder(index.embedding_model, cfg["device"], index.normalize)
    # Load the embedder outside the retrieval timer, or the first query reports
    # the model load as retrieval latency.
    with stage_timer("encoder.warmup", model=index.embedding_model):
        encoder([""])

    retrieved = retrieve_scored(
        args.question,
        index,
        encoder=encoder,
        top_k=args.top_k or cfg["retrieve"]["top_k"],
        candidates=cfg["retrieve"]["candidates"],
        rrf_k=cfg["retrieve"]["rrf_k"],
    )
    units = [u for u, _ in retrieved]

    # Instrument #1 (DESIGN.md): always report what the retrieved set is made of.
    print(f"retrieved {len(units)} units — modality: {format_modality_distribution(units)}")

    if args.show_evidence:
        print("--- evidence ---")
        for unit, score in retrieved:
            print(f"  {score:.4f}  [{unit.id}, {unit.modality}] {unit.content[:90]}")
        print()

    text = answer(args.question, units, cfg, mode=args.mode)
    print(text)

    valid = {u.id for u in units}
    cited = cited_ids(text)
    unknown = [c for c in cited if c not in valid]
    print(f"\ncited {len(cited)} of {len(units)} units" + (f"; NOT IN EVIDENCE: {unknown}" if unknown else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
