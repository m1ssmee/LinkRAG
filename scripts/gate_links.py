#!/usr/bin/env python3
"""Relatedness gate over links.jsonl -- priority (ii).

    python scripts/gate_links.py [--links data/processed/links.jsonl] [--corpus pilot01]

Judges every link (eval.judge, temperature 0, 3 runs, majority), writes the verdict
into each link's metadata in place, and reports the pass rate per type. Failed links
are kept in the file, flagged; `load_links()` drops them by default.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from linkrag.core import load_config, setup_logging
from linkrag.costs import cached_completer, record_run
from linkrag.generate.answer import http_completer
from linkrag.index import Index
from linkrag.link.align import load_links, save_links
from linkrag.link.relatedness import apply_verdicts, gate_links, write_report
from linkrag.manifest import MANIFEST_NAME, load_manifest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--index", default=None)
    ap.add_argument("--links", default=None)
    ap.add_argument("--corpus", default="pilot01")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--cache", default="data/processed/verify_cache")
    ap.add_argument("--reports-dir", default="reports")
    args = ap.parse_args(argv)

    setup_logging()
    cfg = load_config(args.config)
    rcfg = cfg["link"].get("relatedness", {})
    index_dir = Path(args.index or cfg["index"]["store_dir"])
    index = Index.load(index_dir)
    manifest = load_manifest(index_dir.parent / MANIFEST_NAME) or {}
    mhash = manifest.get("hash", "unstamped")
    links_path = Path(args.links or cfg["link"]["align"]["links_path"])
    links = load_links(links_path, expect_manifest=manifest.get("hash"), gated=False)

    jcfg = cfg.get("eval", {}).get("judge") or cfg["models"]["llm"]
    judge = cached_completer(http_completer(jcfg), Path(args.cache) / mhash / str(jcfg.get("model")))
    runs = int(rcfg.get("runs", 3))
    print(f"{len(links)} links · corpus {mhash} · judge {jcfg.get('model')} · {runs} runs\n")

    verdicts = gate_links(links, list(index.units), judge, runs=runs, workers=args.workers, progress=print)
    save_links(apply_verdicts(links, verdicts), links_path, manifest_hash=mhash)
    md = write_report(verdicts, links, list(index.units), Path(args.reports_dir) / f"relatedness_{args.corpus}.md",
                      corpus=args.corpus, manifest_hash=mhash, judge_model=str(jcfg.get("model")), runs=runs)
    footer = record_run("scripts/gate_links.py", f"{args.corpus} relatedness gate",
                        [(str(jcfg.get("model")), judge.usage)], cfg["models"].get("pricing"))
    with open(md, "a") as fh:
        fh.write("\n".join(footer) + "\n")
    print("\n".join(footer))
    print(f"\nflagged in place: {links_path}\nwrote {md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
