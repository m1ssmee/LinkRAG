#!/usr/bin/env python3
"""Verify a proposed gold set and apply it -- priority (i).

    python scripts/eval/verify_gold.py --proposed tests/regression/pilot01_proposed.jsonl \
        --out tests/regression/pilot01_questions.jsonl --corpus pilot01

Writes reports/gold_verified_<corpus>.md (+ .json for the audit sampler) and the
stamped gold file. LLM replies are cached on disk per (prompt, nth call) so an
interrupted run resumes without re-paying, while the `runs` repeats stay distinct calls.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from linkrag.core import load_config, set_max_cost, setup_logging
from linkrag.costs import cached_completer, record_run
from linkrag.eval.verify_gold import dump_json, entailment_opts, verified_gold_rows, verify_gold, write_gold, write_report
from linkrag.generate.answer import http_completer, judge_completer
from linkrag.index import Index
from linkrag.manifest import MANIFEST_NAME, load_manifest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--proposed", default="tests/regression/pilot01_proposed.jsonl")
    ap.add_argument("--out", default="tests/regression/pilot01_questions.jsonl")
    ap.add_argument("--corpus", default="pilot01")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--index", default=None)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--cache", default="data/processed/verify_cache")
    ap.add_argument("--entailment", choices=["nli", "llm"], default=None,
                    help="override eval.entailment.backend (nli = local, free, deterministic)")
    ap.add_argument("--reports-dir", default="reports",
                    help="where the .md/.json go (scratch dir for a cost backfill replay)")
    ap.add_argument("--only", default=None, help="comma-separated qids (debugging)")
    ap.add_argument("--max-cost", type=float, default=0.0, help="stop when uncached spend exceeds this")
    args = ap.parse_args(argv)

    setup_logging()
    cfg = load_config(args.config)
    set_max_cost(cfg, args.max_cost)
    index = Index.load(args.index or cfg["index"]["store_dir"])
    manifest = load_manifest(Path(args.index or cfg["index"]["store_dir"]).parent / MANIFEST_NAME) or {}
    mhash = manifest.get("hash", "unstamped")

    raw = [json.loads(l) for l in Path(args.proposed).read_text().splitlines() if l.strip()]
    meta = next((r["_meta"] for r in raw if "_meta" in r), {})
    rows = [r for r in raw if "_meta" not in r]
    if args.only:
        keep = set(args.only.split(","))
        rows = [r for r in rows if r["qid"] in keep]
    if meta.get("corpus_manifest") and meta["corpus_manifest"] != mhash:
        print(f"WARNING: proposal written for corpus {meta['corpus_manifest']}, index is {mhash}")

    llm = cfg["models"]["llm"]
    jcfg = cfg.get("eval", {}).get("judge") or llm
    if jcfg.get("model") == llm.get("model"):
        print("WARNING: eval.judge is the same model as models.llm -- self-grading is lenient")
    from linkrag.costs import price_for
    complete = cached_completer(http_completer(llm), Path(args.cache) / mhash / str(llm.get("model")),
                                max_cost_usd=args.max_cost,
                                price=price_for(str(llm.get("model")), cfg["models"].get("pricing")))
    ent = entailment_opts(cfg, args.entailment)
    judge = cached_completer(judge_completer(cfg, ent["backend"]), Path(args.cache) / mhash / str(jcfg.get("model")),
                             max_cost_usd=args.max_cost,
                             price=price_for(str(jcfg.get("model")), cfg["models"].get("pricing")))
    deck_files = {Path(u.source_file).name for u in index.units if u.metadata.get("slide_deck")}
    print(f"{len(rows)} questions · corpus {mhash} ({len(index)} units) · deck files {sorted(deck_files)}")
    print(f"answerer {llm.get('model')} · judge {jcfg.get('model')} @ temperature={jcfg.get('temperature')} "
          f"seed={jcfg.get('seed')} · {args.runs} runs\n")

    print(f"entailment backend: {ent['backend']}")
    verdicts = verify_gold(rows, list(index.units), complete, judge=judge, deck_files=deck_files,
                           runs=args.runs, workers=1 if ent["backend"] == "nli" else args.workers,
                           **ent, progress=print)

    reports = Path(args.reports_dir)
    md = write_report(verdicts, reports / f"gold_verified_{args.corpus}.md", corpus=args.corpus,
                      manifest_hash=mhash, model=str(jcfg.get("model")), runs=args.runs,
                      n_units=len(index), answerer=str(llm.get("model")))
    js = dump_json(verdicts, reports / f"gold_verified_{args.corpus}.json")
    gold_rows = verified_gold_rows(verdicts, list(index.units))
    gold = write_gold(gold_rows, args.out, manifest_hash=mhash, model=str(jcfg.get("model")),
                      runs=args.runs, source=args.proposed, answerer=str(llm.get("model")))

    kept = sum(v.verified_type is not None for v in verdicts)
    footer = record_run("scripts/eval/verify_gold.py", f"{args.corpus} gold verification",
                        [(str(llm.get("model")), complete.usage), (str(jcfg.get("model")), judge.usage)],
                        cfg["models"].get("pricing"))
    with open(md, "a") as fh:
        fh.write("\n".join(footer) + "\n")
    print("\n".join(footer))
    print(f"\nkept {kept}/{len(verdicts)} questions · "
          f"{sum(len(r['gold_unit_ids']) for r in gold_rows)} gold units")
    print(f"wrote {md}\nwrote {js}\nwrote {gold}  (stamped {mhash})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
