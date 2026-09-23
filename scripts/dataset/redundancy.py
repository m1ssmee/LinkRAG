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
from collections import defaultdict
from pathlib import Path

from linkrag.core import load_config, refuse_strong_in_batch, set_max_cost, setup_logging
from linkrag.costs import cached_completer, record_run
from linkrag.eval.redundancy import DEFAULT_PAIRS, dump_json, redundancy, role_of, write_report
from linkrag.eval.verify_gold import entailment_opts, refuse_nli_decisions, verifier_label
from linkrag.generate.answer import http_completer, judge_completer
from linkrag.index import Index, default_encoder
from linkrag.manifest import MANIFEST_NAME, load_manifest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=None)
    ap.add_argument("--corpus", default="pilot01")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--max-cost", type=float, required=True,
                    help="USD budget for this run (required; 0 = cache replays and free backends only)")
    ap.add_argument("--deck", action="append", default=[], help="file name(s) to treat as deck")
    ap.add_argument("--paper", action="append", default=[], help="file name(s) to treat as paper")
    ap.add_argument("--pairs", default=",".join(f"{a}->{b}" for a, b in DEFAULT_PAIRS))
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="sentences per pair (smoke test)")
    ap.add_argument("--sample", type=int, default=0,
                    help="sentences per direction, stratified by position (0 = all); fractions get a Wilson 95%% CI")
    ap.add_argument("--seed", default=None, help="sampling seed (default dataset.intake.sample_seed)")
    ap.add_argument("--cache", default="data/processed/verify_cache")
    ap.add_argument("--entailment", choices=["nli", "llm"], default=None,
                    help="override eval.entailment.backend (nli = local, free, deterministic)")
    ap.add_argument("--reports-dir", default="reports",
                    help="where the .md/.json go (scratch dir for a cost backfill replay)")
    args = ap.parse_args(argv)

    setup_logging()
    cfg = load_config(args.config)
    set_max_cost(cfg, args.max_cost)
    refuse_strong_in_batch(cfg)
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
    ent = entailment_opts(cfg, args.entailment)
    seed = str(args.seed or cfg.get("dataset", {}).get("intake", {}).get("sample_seed", 0))
    judge = cached_completer(judge_completer(cfg, ent["backend"]), Path(args.cache) / mhash / str(jcfg.get("model")))
    pairs = [tuple(p.split("->")) for p in args.pairs.split(",")]

    files: dict[str, list[str]] = defaultdict(list)
    for u in units:
        name = Path(u.source_file).name
        if name not in files[role_of(u)]:
            files[role_of(u)].append(name)
    print(f"corpus {mhash} · {len(units)} units · roles: "
          + ", ".join(f"{r}={n}" for r, n in files.items()))
    print(f"judge {jcfg.get('model')} · k={args.k} · runs={args.runs} · pairs {args.pairs}\n")

    refuse_nli_decisions(ent["backend"], args.reports_dir)
    verifier, vruns = verifier_label(cfg, ent["backend"], args.runs)
    print(f"entailment backend: {ent['backend']}")
    verdicts = redundancy(units, encoder, judge, pairs=pairs, k=args.k, runs=args.runs,
                          workers=1 if ent["backend"] == "nli" else args.workers, limit=args.limit,
                          sample=args.sample, seed=seed, **ent, progress=print)
    out_md = Path(args.reports_dir) / f"redundancy_{args.corpus}.md"
    write_report(verdicts, out_md, corpus=args.corpus, manifest_hash=mhash,
                 judge_model=verifier, k=args.k, runs=vruns, files=dict(files),
                 sampling=f"{args.sample} sentences per direction, stratified by position, seed {seed}" if args.sample else "")
    dump_json(verdicts, out_md.with_suffix(".json"))
    footer = record_run("scripts/dataset/redundancy.py", f"{args.corpus} redundancy",
                        [(str(jcfg.get("model")), judge.usage)], cfg["models"].get("pricing"))
    with out_md.open("a") as fh:
        fh.write("\n".join(footer) + "\n")
    print("\n".join(footer))
    print(f"wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
