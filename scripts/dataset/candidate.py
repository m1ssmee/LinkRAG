#!/usr/bin/env python3
"""Extended-dataset intake: is this lecture worth adding?

    python scripts/dataset/candidate.py --name mit_6006_l03 \
        --audio raw/l03.mp3 --deck raw/l03_slides.pdf [--notes raw/l03_notes.pdf]

Ingests the files to a scratch index under data/processed/candidates/<name>/, runs
the modality-redundancy metric (judge = eval.judge) and prints KEEP / REJECT against
`dataset.intake` in the config. The number is reported either way; the verdict is
advisory -- record it in scripts/dataset/README.md's table when you decide.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from linkrag.core import load_config, setup_logging
from linkrag.costs import cached_completer, record_run
from linkrag.eval.redundancy import DEFAULT_PAIRS, dump_json, redundancy, role_of, summarise, write_report
from linkrag.generate.answer import http_completer
from linkrag.index import Index, default_encoder
from linkrag.manifest import MANIFEST_NAME, load_manifest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="short id, e.g. mit_6006_l03")
    ap.add_argument("--audio", required=True)
    ap.add_argument("--deck", required=True)
    ap.add_argument("--notes", default=None, help="optional paper / lecture notes PDF")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--reingest", action="store_true", help="rebuild the scratch index")
    args = ap.parse_args(argv)

    setup_logging()
    cfg = load_config(args.config)
    intake = cfg.get("dataset", {}).get("intake", {})
    threshold = float(intake.get("max_deck_to_transcript", 0.65))
    root = Path("data/processed/candidates") / args.name
    index_dir = root / "index"

    files = [args.deck, args.audio] + ([args.notes] if args.notes else [])
    for f in files:
        if not Path(f).exists():
            print(f"missing file: {f}", file=sys.stderr)
            return 2
    if args.reingest or not index_dir.exists():
        print(f"ingesting {len(files)} file(s) -> {index_dir}")
        rc = subprocess.call([sys.executable, "scripts/ingest.py", *files, "--config", args.config,
                              "--out", str(index_dir)])
        if rc not in (0, 2):
            return rc

    index = Index.load(index_dir)
    manifest = load_manifest(root / MANIFEST_NAME) or {}
    units = list(index.units)
    encoder = default_encoder(index.embedding_model, cfg["device"], index.normalize)
    encoder([""])
    jcfg = cfg.get("eval", {}).get("judge") or cfg["models"]["llm"]
    judge = cached_completer(http_completer(jcfg),
                             Path("data/processed/verify_cache") / manifest.get("hash", args.name) / str(jcfg.get("model")))
    pairs = [p for p in DEFAULT_PAIRS if args.notes or "paper" not in p]

    roles: dict[str, list[str]] = defaultdict(list)
    for u in units:
        n = Path(u.source_file).name
        if n not in roles[role_of(u)]:
            roles[role_of(u)].append(n)
    print(f"{args.name}: {len(units)} units · " + ", ".join(f"{r}={v}" for r, v in roles.items()))

    verdicts = redundancy(units, encoder, judge, pairs=pairs, k=int(intake.get("redundancy_k", 8)),
                          runs=int(intake.get("redundancy_runs", 3)), workers=args.workers, progress=print)
    summ = summarise(verdicts)
    out_md = Path("reports") / f"redundancy_{args.name}.md"
    write_report(verdicts, out_md, corpus=args.name, manifest_hash=manifest.get("hash", "?"),
                 judge_model=str(jcfg.get("model")), k=int(intake.get("redundancy_k", 8)),
                 runs=int(intake.get("redundancy_runs", 3)), files=dict(roles))
    dump_json(verdicts, out_md.with_suffix(".json"))

    d2t = summ["pairs"].get("deck->transcript", {}).get("fraction")
    keep = d2t is not None and d2t < threshold
    verdict = "KEEP" if keep else "REJECT"
    lines = ["", f"## Intake verdict: **{verdict}**", "",
             f"deck→transcript = **{d2t:.1%}** vs threshold < {threshold:.0%} "
             f"(pilot01: 94.2 %)" if d2t is not None else "deck→transcript could not be computed",
             "", "| pair | redundancy |", "|---|---:|"]
    lines += [f"| {k} | {v['fraction']:.1%} |" for k, v in summ["pairs"].items()]
    footer = record_run("scripts/dataset/candidate.py", f"{args.name} intake",
                        [(str(jcfg.get("model")), judge.usage)], cfg["models"].get("pricing"))
    with out_md.open("a") as fh:
        fh.write("\n".join(lines + footer) + "\n")
    print("\n".join(lines + footer))
    print(f"\nwrote {out_md}")
    return 0 if keep else 1


if __name__ == "__main__":
    raise SystemExit(main())
