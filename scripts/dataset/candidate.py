#!/usr/bin/env python3
"""Extended-dataset intake: is this lecture worth adding?

    python scripts/dataset/candidate.py --name mit_6006_l03 \
        --audio raw/l03.mp3 --deck raw/l03_slides.pdf [--notes raw/l03_notes.pdf]

Ingests the files to a scratch index under data/processed/candidates/<name>/, runs
the modality-redundancy metric (judge = eval.judge) and prints KEEP / REJECT / BORDERLINE against
`dataset.intake` in the config. The number is reported either way; the verdict is
advisory -- record it in scripts/dataset/README.md's table when you decide.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from linkrag.core import load_config, refuse_strong_in_batch, set_max_cost, setup_logging
from linkrag.costs import cached_completer, record_run
from linkrag.eval.redundancy import DEFAULT_PAIRS, dump_json, intake_gate, redundancy, role_of, summarise, write_report
from linkrag.eval.verify_gold import entailment_opts
from linkrag.generate.answer import http_completer, judge_completer
from linkrag.index import Index, default_encoder
from linkrag.manifest import MANIFEST_NAME, load_manifest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="short id, e.g. mit_6006_l03")
    ap.add_argument("--audio", required=True)
    ap.add_argument("--deck", required=True)
    ap.add_argument("--notes", default=None, help="optional paper / lecture notes PDF")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--max-cost", type=float, required=True,
                    help="USD budget for this run (required; 0 = cache replays and free backends only)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cuda",
                    help="ASR/embedder device; intake runs on a Colab GPU by default (scripts/dataset/README.md)")
    ap.add_argument("--asr", choices=["local", "openai"], default="local",
                    help="local = faster-whisper (free, default); openai = whisper-1 ($0.006/min)")
    ap.add_argument("--reingest", action="store_true", help="rebuild the scratch index")
    ap.add_argument("--sample", type=int, default=None,
                    help="redundancy sentences per direction (default dataset.intake.redundancy_sample; 0 = all)")
    args = ap.parse_args(argv)

    setup_logging()
    cfg = load_config(args.config)
    set_max_cost(cfg, args.max_cost)
    refuse_strong_in_batch(cfg)
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
                              "--out", str(index_dir), "--device", args.device, "--asr", args.asr,
                              "--max-cost", str(args.max_cost)])
        if rc not in (0, 2):
            return rc

    index = Index.load(index_dir)
    manifest = load_manifest(root / MANIFEST_NAME) or {}
    units = list(index.units)
    encoder = default_encoder(index.embedding_model, cfg["device"], index.normalize)
    encoder([""])
    jcfg = cfg.get("eval", {}).get("judge") or cfg["models"]["llm"]
    ent = entailment_opts(cfg)
    if ent["backend"] != "llm":
        raise SystemExit("intake is judged by the LLM judge only; eval.entailment.backend is "
                         f"{ent['backend']!r} (DESIGN.md finding 11)")
    judge = cached_completer(judge_completer(cfg, ent["backend"]),
                             Path("data/processed/verify_cache") / manifest.get("hash", args.name) / str(jcfg.get("model")))
    pairs = [p for p in DEFAULT_PAIRS if args.notes or "paper" not in p]

    roles: dict[str, list[str]] = defaultdict(list)
    for u in units:
        n = Path(u.source_file).name
        if n not in roles[role_of(u)]:
            roles[role_of(u)].append(n)
    print(f"{args.name}: {len(units)} units · " + ", ".join(f"{r}={v}" for r, v in roles.items()))

    print(f"entailment backend: {ent['backend']}")
    sample = int(args.sample if args.sample is not None else intake.get("redundancy_sample", 0))
    seed = str(intake.get("sample_seed", 0))
    verdicts = redundancy(units, encoder, judge, pairs=pairs, k=int(intake.get("redundancy_k", 8)),
                          runs=int(intake.get("redundancy_runs", 3)),
                          workers=1 if ent["backend"] == "nli" else args.workers, sample=sample, seed=seed,
                          **ent, progress=print)
    summ = summarise(verdicts)
    out_md = Path("reports") / f"redundancy_{args.name}.md"
    write_report(verdicts, out_md, corpus=args.name, manifest_hash=manifest.get("hash", "?"),
                 judge_model=str(jcfg.get("model")), k=int(intake.get("redundancy_k", 8)),
                 runs=int(intake.get("redundancy_runs", 3)), files=dict(roles),
                 sampling=f"{sample} sentences per direction, stratified by position, seed {seed}" if sample else "")
    dump_json(verdicts, out_md.with_suffix(".json"))

    row = summ["pairs"].get("deck->transcript")
    verdict = intake_gate(row, threshold) if row else "REJECT"
    ci = (f", 95 % CI {row['ci95'][0]:.1%}–{row['ci95'][1]:.1%} from {row['n']} of {row['population']}"
          if row and row["sampled"] else "")
    lines = ["", f"## Intake verdict: **{verdict}**", "",
             f"deck→transcript = **{row['fraction']:.1%}**{ci} vs threshold < {threshold:.0%} "
             f"(pilot01: 94.2 %)" if row else "deck→transcript could not be computed",
             *(["", "The sample's CI straddles the threshold: run the full census (`--sample 0`) to decide."]
               if verdict == "BORDERLINE" else []),
             "", "| pair | redundancy | 95 % CI |", "|---|---:|---|"]
    lines += [f"| {k} | {v['fraction']:.1%} | {v['ci95'][0]:.1%}–{v['ci95'][1]:.1%} |" for k, v in summ["pairs"].items()]
    footer = record_run("scripts/dataset/candidate.py", f"{args.name} intake",
                        [(str(jcfg.get("model")), judge.usage)], cfg["models"].get("pricing"))
    with out_md.open("a") as fh:
        fh.write("\n".join(lines + footer) + "\n")
    print("\n".join(lines + footer))
    print(f"\nwrote {out_md}")
    return {"KEEP": 0, "REJECT": 1, "BORDERLINE": 3}[verdict]


if __name__ == "__main__":
    raise SystemExit(main())
