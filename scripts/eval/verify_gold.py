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
import hashlib
import json
import threading
from collections import Counter
from pathlib import Path

from linkrag.core import load_config, setup_logging
from linkrag.eval.verify_gold import dump_json, verified_gold_rows, verify_gold, write_gold, write_report
from linkrag.generate.answer import http_completer
from linkrag.index import Index
from linkrag.manifest import MANIFEST_NAME, load_manifest


def cached_completer(inner, cache_dir: Path):
    """Disk cache keyed by (system, user, n-th identical call). Thread-safe."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    seen: Counter = Counter()
    lock = threading.Lock()

    def complete(system: str, user: str) -> str:
        base = hashlib.sha256((system + "\x00" + user).encode()).hexdigest()
        with lock:
            n = seen[base]
            seen[base] += 1
        path = cache_dir / f"{base}.{n}.txt"
        if path.exists():
            return path.read_text()
        text = inner(system, user)
        path.write_text(text)
        return text

    complete.calls = seen  # type: ignore[attr-defined]
    return complete


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
    ap.add_argument("--only", default=None, help="comma-separated qids (debugging)")
    args = ap.parse_args(argv)

    setup_logging()
    cfg = load_config(args.config)
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
    complete = cached_completer(http_completer(llm), Path(args.cache) / mhash)
    deck_files = {Path(u.source_file).name for u in index.units if u.metadata.get("slide_deck")}
    print(f"{len(rows)} questions · corpus {mhash} ({len(index)} units) · deck files {sorted(deck_files)}")
    print(f"judge {llm.get('model')} @ temperature={llm.get('temperature')} seed={llm.get('seed')} · {args.runs} runs\n")

    verdicts = verify_gold(rows, list(index.units), complete, deck_files=deck_files,
                           runs=args.runs, workers=args.workers, progress=print)

    reports = Path("reports")
    md = write_report(verdicts, reports / f"gold_verified_{args.corpus}.md", corpus=args.corpus,
                      manifest_hash=mhash, model=str(llm.get("model")), runs=args.runs,
                      n_units=len(index))
    js = dump_json(verdicts, reports / f"gold_verified_{args.corpus}.json")
    gold_rows = verified_gold_rows(verdicts, list(index.units))
    gold = write_gold(gold_rows, args.out, manifest_hash=mhash, model=str(llm.get("model")),
                      runs=args.runs, source=args.proposed)

    kept = sum(v.verified_type is not None for v in verdicts)
    calls = sum(complete.calls.values())
    print(f"\nkept {kept}/{len(verdicts)} questions · "
          f"{sum(len(r['gold_unit_ids']) for r in gold_rows)} gold units · {calls} LLM calls (incl. cached)")
    print(f"wrote {md}\nwrote {js}\nwrote {gold}  (stamped {mhash})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
