#!/usr/bin/env python3
"""LectQA-Vid adapter -- target T1 (Shafiq et al., CMC 88(2), 2026). This file is the entry
point; the steps live in `scripts/adapters/lectqa/`:

  common         paths, QA files, video ids, T1's published numbers, strict gold timestamps, prompts
  acquisition    fetch (yt-dlp), coverage
  transcription  prepare: whisper transcript + changed frames with OCR -> per-video index
  metrics        run: T1's metric suite on stored answers; audit (LLM-free)
  localisation   hit@k / IoU against the gold interval; v2 segmentation ablation
  frameslides    frame-derived slide units and links
  mcq            MCQ with options shuffled under a recorded seed
  open_ended     T1 replica vs iterative vs full context; faithfulness

The published dataset (Mendeley, doi:10.17632/yt4nmz9mcv.1, CC BY 4.0) ships the QA
pairs and YouTube links only -- "videos, keyframes and transcripts are not provided
because of copyright issues" -- so the adapter rebuilds the inputs their pipeline
consumed: audio -> Whisper transcript; keyframes -> OCR (they used Gemini captions;
we use tesseract). Their 80/10/10 split and 1,000-pair evaluation subset are not
published; results are reported on every QA pair of the videos processed, with n stated.

    python scripts/adapters/lectqa_vid.py fetch   --videos 10 --max-cost 0   # yt-dlp: audio m4a + video-only mp4
    python scripts/adapters/lectqa_vid.py prepare --videos 10 --max-cost 0   # whisper + frames + OCR -> per-video index
    python scripts/adapters/lectqa_vid.py run --videos 35 --cache-only --answerer-model gpt-5.4-2026-03-05 --max-cost 0

The first-run "ours" metric path (SQuAD-style F1, bge-m3 similarity) was retired on
2026-09-24; `reports/lectqa_vid_first_run.md` is history. `run` reports T1's own suite.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lectqa.acquisition import coverage, fetch  # noqa: E402
from lectqa.common import video_ids  # noqa: E402
from lectqa.frameslides import frameslides  # noqa: E402
from lectqa.localisation import v2  # noqa: E402
from lectqa.mcq import mcq_shuffled  # noqa: E402
from lectqa.metrics import audit, run  # noqa: E402
from lectqa.open_ended import faithfulness, open_modes, open_rerender  # noqa: E402
from lectqa.transcription import prepare  # noqa: E402
from linkrag.core import load_config, refuse_strong_in_batch, set_max_cost, setup_logging  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["fetch", "prepare", "run", "v2", "frameslides", "coverage", "audit", "mcq",
                                     "open", "faith"])
    ap.add_argument("--runs", type=int, default=3, help="faith: judge runs per check (majority)")
    ap.add_argument("--rest", type=int, default=None, help="open: how many non-subset questions (seeded order)")
    ap.add_argument("--report-only", action="store_true", help="open: re-render the report from the stored answers")
    ap.add_argument("--dry-run", action="store_true", help="mcq/open: build prompts and estimate cost only")
    ap.add_argument("--videos", type=int, default=None, help="first N video ids")
    ap.add_argument("--only", default=None, help="comma-separated video ids")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--modes", default="baseline,linkrag")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--out", default="results/external/lectqa_theirs_metrics.md")
    ap.add_argument("--allow-new-calls", action="store_true", help="v2: let iterative modes call the LLM on cache misses")
    ap.add_argument("--metrics", choices=["theirs"], default="theirs",
                    help="run: T1's suite (token P/R/F1, BLEU, METEOR, ROUGE-1 recall, MiniLM sim, MCQ); "
                         "the first-run 'ours' path was retired 2026-09-24")
    ap.add_argument("--cache-only", action="store_true", help="run: replay stored answers only; a miss is skipped")
    ap.add_argument("--answerer-model", default=None, help="override models.llm.model (e.g. the stored gpt-5.4)")
    ap.add_argument("--max-cost", type=float, required=True,
                    help="USD budget for this run (required; 0 = cache replays and free backends only)")
    args = ap.parse_args(argv)
    setup_logging()
    cfg = load_config(args.config)
    set_max_cost(cfg, args.max_cost)
    if args.answerer_model:
        cfg["models"]["llm"]["model"] = args.answerer_model
    refuse_strong_in_batch(cfg, answerer_cache_only=args.cache_only)
    dcfg = cfg.get("datasets", {}).get("lectqa_vid", {"audio_segment_seconds": 15, "frame_interval_s": 5, "top_k": 4})
    ids = video_ids(args.videos, args.only)
    external = Path("results/external")
    if args.step == "fetch":
        fetch(ids)
        return 0
    if args.step == "prepare":
        prepare(ids, cfg, dcfg)
        return 0
    if args.step == "mcq":
        return mcq_shuffled(ids, cfg, dcfg, external / "lectqa_mcq_shuffled.md", max_cost=args.max_cost,
                            dry_run=args.dry_run)
    if args.step == "open" and args.report_only:
        return open_rerender(ids, cfg, external / "lectqa_open_modes.md")
    if args.step == "open":
        return open_modes(ids, cfg, dcfg, external / "lectqa_open_modes.md", max_cost=args.max_cost,
                          dry_run=args.dry_run, rest=args.rest)
    if args.step == "faith":
        return faithfulness(cfg, external / "lectqa_open_modes.jsonl", external / "lectqa_faithfulness.md",
                            max_cost=args.max_cost, dry_run=args.dry_run, runs=args.runs)
    if args.step == "audit":
        return audit(external / "lectqa_audit.md")
    if args.step == "coverage":
        return coverage(external / "lectqa_coverage.md")
    if args.step == "frameslides":
        return frameslides(ids, cfg, dcfg, external / "lectqa_frameslides.md")
    if args.step == "v2":
        return v2(ids, cfg, dcfg, external / "lectqa_v2.md", cache_only=not args.allow_new_calls, max_cost=args.max_cost)
    return run(ids, cfg, dcfg, args.modes.split(","), Path(args.out), args.repeats, cache_only=args.cache_only)


if __name__ == "__main__":
    raise SystemExit(main())
