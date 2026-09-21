#!/usr/bin/env python3
"""Modality redundancy of a lecture corpus -- run before adding a lecture to the
extended dataset (DESIGN.md, findings).

    python scripts/ingest.py <deck.pdf> <paper.pdf> <talk.mp3> --index data/processed/cand01
    python scripts/dataset/redundancy.py --index data/processed/cand01 --corpus cand01

Roles are inferred from the index: audio units are the transcript, units flagged
`slide_deck` are the deck, everything else is the paper. Override with --deck / --paper
(file names). Writes reports/redundancy_<corpus>.md and .json. Judge = eval.judge.
"""

from __future__ import annotations

import argparse
import importlib.util
from collections import defaultdict
from pathlib import Path

from linkrag.core import load_config, setup_logging
from linkrag.eval.redundancy import DEFAULT_PAIRS, dump_json, redundancy, role_of, write_report
from linkrag.generate.answer import http_completer
from linkrag.index import Index, default_encoder
from linkrag.manifest import MANIFEST_NAME, load_manifest


def _cached_completer():
    spec = importlib.util.spec_from_file_location(
        "verify_gold_cli", Path(__file__).resolve().parent.parent / "eval" / "verify_gold.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod.cached_completer


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=None)
    ap.add_argument("--corpus", default="pilot01")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--deck", action="append", default=[], help="file name(s) to treat as deck")
    ap.add_argument("--paper", action="append", default=[], help="file name(s) to treat as paper")
    ap.add_argument("--pairs", default=",".join(f"{a}->{b}" for a, b in DEFAULT_PAIRS))
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="sentences per pair (smoke test)")
    ap.add_argument("--cache", default="data/processed/verify_cache")
    args = ap.parse_args(argv)

    setup_logging()
    cfg = load_config(args.config)
    index_dir = Path(args.index or cfg["index"]["store_dir"])
    index = Index.load(index_dir)
    manifest = load_manifest(index_dir.parent / MANIFEST_NAME) or {}
    mhash = manifest.get("hash", "unstamped")
    units = list(index.units)
    for u in units:                          # explicit role overrides
        name = Path(u.source_file).name
        if name in args.deck:
            u.metadata["slide_deck"] = True
        elif name in args.paper:
            u.metadata["slide_deck"] = False

    encoder = default_encoder(index.embedding_model, cfg["device"], index.normalize)
    encoder([""])
    jcfg = cfg.get("eval", {}).get("judge") or cfg["models"]["llm"]
    judge = _cached_completer()(http_completer(jcfg), Path(args.cache) / mhash / str(jcfg.get("model")))
    pairs = [tuple(p.split("->")) for p in args.pairs.split(",")]

    files: dict[str, list[str]] = defaultdict(list)
    for u in units:
        name = Path(u.source_file).name
        if name not in files[role_of(u)]:
            files[role_of(u)].append(name)
    print(f"corpus {mhash} · {len(units)} units · roles: "
          + ", ".join(f"{r}={n}" for r, n in files.items()))
    print(f"judge {jcfg.get('model')} · k={args.k} · runs={args.runs} · pairs {args.pairs}\n")

    verdicts = redundancy(units, encoder, judge, pairs=pairs, k=args.k, runs=args.runs,
                          workers=args.workers, limit=args.limit, progress=print)
    out_md = Path("reports") / f"redundancy_{args.corpus}.md"
    write_report(verdicts, out_md, corpus=args.corpus, manifest_hash=mhash,
                 judge_model=str(jcfg.get("model")), k=args.k, runs=args.runs, files=dict(files))
    dump_json(verdicts, out_md.with_suffix(".json"))
    print(f"\n{sum(judge.calls.values())} judge calls (incl. cached)\nwrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
