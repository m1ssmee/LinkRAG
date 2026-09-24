#!/usr/bin/env python3
"""Measure a candidate judge against the stored pilot01 LLM verdicts before it decides anything.

    python scripts/eval/judge_agreement.py --judge colab      # after exporting LINKRAG_COLAB_BASE_URL
    python scripts/eval/judge_agreement.py --judge groq       # needs GROQ_API_KEY
    python scripts/eval/judge_agreement.py --judge default --judge-cache-only   # replay check, $0

Runs gold verification on exactly the pairs of `results/nli_vs_llm_pilot01.md`: the 25
proposed pilot01 questions, their 147 proposed gold units, and the modality-only source
runs answered by the stored answerer (gpt-5.4). That answerer is pinned and **cache-only**,
so its replies are the stored ones and cost nothing. Only the judge changes. The verdicts
go to `results/judge_agreement_<backend>/`, and `scripts/eval/compare_entailment.py`
writes `results/judge_agreement_<backend>.md` in the NLI report's layout: unit-level
kappa vs gpt-4.1-mini and vs gpt-5.4, type-label agreement, and per-question differences.

`--max-cost` is required. With `--max-cost 0`, a judge that bills (`billing: metered`) is
refused unless `--judge-cache-only` is also given. Free backends run at $0.
"""

from __future__ import annotations

import argparse
import os
import sys
from contextlib import contextmanager
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

STORED_ANSWERER = "gpt-5.4-2026-03-05"      # answered every stored pilot01 verification
STORED = [("reports/gold_verified_pilot01.json", "llm:gpt-4.1-mini"),
          ("reports/gold_verified_pilot01.judge-gpt54.json", "llm:gpt-5.4")]


@contextmanager
def judge_env(backend: str):
    """verify_gold resolves its judge from LINKRAG_JUDGE_BACKEND, and the answerer here is the
    stored gpt-5.4, so LINKRAG_LLM_BACKEND is cleared. Both are restored on exit: the caller's
    environment (a test run, a shell session) must not keep the candidate judge."""
    saved = {k: os.environ.get(k) for k in ("LINKRAG_JUDGE_BACKEND", "LINKRAG_LLM_BACKEND")}
    os.environ["LINKRAG_JUDGE_BACKEND"] = backend
    os.environ.pop("LINKRAG_LLM_BACKEND", None)
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def judge_block(config: str, backend: str) -> dict:
    """The judge config this run would use, resolved exactly as verify_gold will."""
    from linkrag.core import load_config
    with judge_env(backend):
        return load_config(config)["eval"]["judge"]


def refuse_billing(judge: dict, max_cost: float, judge_cache_only: bool) -> None:
    if judge_cache_only or max_cost > 0 or judge.get("billing") == "free":
        return
    raise SystemExit(f"judge backend {judge.get('backend')!r} ({judge.get('model')}) bills: pass --max-cost USD "
                     f"to allow spend, or --judge-cache-only to replay stored replies. Nothing was sent.")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--judge", required=True, help="a models.llm_backends name (colab, groq, ...) or 'default'")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--max-cost", type=float, required=True,
                    help="USD budget for this run (required; 0 = cache replays and free backends only)")
    ap.add_argument("--judge-cache-only", action="store_true")
    ap.add_argument("--only", default=None, help="comma-separated qids (a smoke run; kappa is then not comparable)")
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args(argv)
    with judge_env(args.judge):
        return run(args)


def run(args: argparse.Namespace) -> int:
    judge = judge_block(args.config, args.judge)
    refuse_billing(judge, args.max_cost, args.judge_cache_only)
    name = f"{args.judge}:{judge.get('model')}"
    raw = Path(args.out_dir) / f"judge_agreement_{args.judge}"
    raw.mkdir(parents=True, exist_ok=True)

    import verify_gold as vg                           # scripts/eval/verify_gold.py
    argv_vg = ["--config", args.config, "--entailment", "llm", "--answerer-cache-only",
               "--answerer-model", STORED_ANSWERER,
               "--max-cost", str(args.max_cost), "--reports-dir", str(raw), "--out", str(raw / "pilot01_questions.jsonl")]
    if args.judge_cache_only:
        argv_vg.append("--judge-cache-only")
    if args.only:
        argv_vg += ["--only", args.only]
    rc = vg.main(argv_vg)
    if rc:
        return rc

    preamble = raw / "preamble.md"
    preamble.write_text(
        f"Candidate judge **{name}** (backend `{args.judge}`, billing `{judge.get('billing')}`) on the pilot01 pairs of "
        f"`results/nli_vs_llm_pilot01.md`, against the stored LLM verdicts. Answerer gpt-5.4, cache-only (stored replies). "
        f"Judge runs: 3, majority. Raw verdicts: `{raw}/`. "
        + ("Judge replies replayed from cache (`--judge-cache-only`). " if args.judge_cache_only else "")
        + ("**Smoke run on a subset (`--only`): kappa not comparable.** " if args.only else "")
        + "A candidate may decide gold or intake only after this report shows agreement comparable to the "
          "LLM-LLM reference (kappa 0.85, DESIGN.md finding 11).\n")
    import compare_entailment as ce                    # scripts/eval/compare_entailment.py
    return ce.main([*(a for p, n in STORED for a in ("--gold", f"{p}={n}")),
                    "--gold", f"{raw / 'gold_verified_pilot01.json'}={name}",
                    "--title", f"Judge agreement — {name} — pilot01", "--preamble", str(preamble),
                    "--out", str(Path(args.out_dir) / f"judge_agreement_{args.judge}.md")])


if __name__ == "__main__":
    raise SystemExit(main())
