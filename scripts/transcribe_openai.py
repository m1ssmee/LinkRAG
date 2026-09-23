#!/usr/bin/env python3
"""Transcribe with the OpenAI whisper-1 backend and compare against a local transcript.

    python scripts/transcribe_openai.py data/raw/pilot01/hsieh.mp3 \
        --out data/processed/transcripts/hsieh.openai.json \
        --compare data/processed/transcripts/hsieh.frozen.json --max-cost 1

Writes the transcript in the frozen schema (never to `<stem>.frozen.json` unless you
name it explicitly) and reports cost, wall time and the pilot01 term table.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from linkrag.core import load_config, setup_logging
from linkrag.costs import empty_usage, price_for, record_run, usage_cost
from linkrag.ingest.asr_openai import transcribe, write_transcript
from linkrag.ingest.audio import build_asr_prompt, load_frozen_transcript

# The terms the pilot01 ASR work turned on (DESIGN.md, Step 0): `ingest` was heard as
# "interest"; the slide-vocab prompt fixed half of them and broke four "top-K".
TERMS = {"ingest": ("ingest",), "interest (mis-heard ingest)": ("interest",),
         "NoScope": ("noscope", "no scope"), "YOLO": ("yolo",),
         "top-K": ("top k", "topk", "top-k"), "type -k (known regression)": ("type k", "type -k")}


def term_counts(words) -> dict[str, int]:
    text = " ".join(w.text for w in words).lower().replace("-", " ")
    text = " ".join(text.split())
    return {label: sum(text.count(v.replace("-", " ")) for v in variants) for label, variants in TERMS.items()}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--out", required=True)
    ap.add_argument("--compare", default=None, help="local transcript to compare against")
    ap.add_argument("--slides", nargs="*", default=[], help="deck PDFs for the vocabulary prompt")
    ap.add_argument("--no-prompt", action="store_true")
    ap.add_argument("--max-cost", type=float, default=1.0)
    ap.add_argument("--report", default="reports/asr_openai_pilot01.md")
    args = ap.parse_args(argv)

    setup_logging()
    cfg = load_config(args.config)
    audio = Path(args.audio)
    out = Path(args.out)
    if out.name.endswith(".frozen.json") and out.exists():
        raise SystemExit(f"{out} exists and is a frozen transcript; refusing to overwrite it")

    prompt = None
    if not args.no_prompt and cfg["ingest"].get("asr_vocab_from_slides", True):
        slides = args.slides or sorted(str(p) for p in audio.parent.glob("*slides*.pdf"))
        if slides:
            import pymupdf
            texts = []
            for pdf in slides:
                with pymupdf.open(pdf) as doc:
                    texts += [page.get_text() for page in doc]
            prompt = build_asr_prompt(texts)
            print(f"vocabulary prompt from {len(slides)} deck(s), {len(prompt.split(','))} terms")

    minutes = 0.0
    try:
        import av
        with av.open(str(audio)) as c:
            minutes = (float(c.duration) / 1e6) / 60 if c.duration else 0.0
    except Exception:
        pass
    price = price_for(str(cfg["ingest"]["asr_openai"]["model"]), cfg["models"].get("pricing"))
    est = (minutes * float(price.get("per_minute", 0))) if price else None
    print(f"{audio.name}: {minutes:.1f} min · estimated cost {'$%.2f' % est if est is not None else 'unknown'} "
          f"· cap ${args.max_cost:.2f}")
    if est is not None and est > args.max_cost:
        raise SystemExit(f"estimated ${est:.2f} exceeds --max-cost ${args.max_cost:.2f}")

    t0 = time.perf_counter()
    words, meta = transcribe(audio, cfg, prompt=prompt)
    elapsed = time.perf_counter() - t0
    write_transcript(words, out, audio, meta)
    usage = empty_usage() | {"calls": len(meta["chunks"]), "audio_seconds": meta["audio_seconds"]}
    footer = record_run("scripts/transcribe_openai.py", f"{audio.name} whisper-1",
                        [(meta["model"], usage)], cfg["models"].get("pricing"))
    cost = usage_cost(usage, price)

    L = [f"# ASR backend comparison — {audio.name}", "",
         f"`ingest.asr_backend: openai` ({meta['model']}, word timestamps, "
         f"{len(meta['chunks'])} chunk(s)) against the frozen local transcript "
         f"(faster-whisper {cfg['models']['whisper']}, int8, CPU). Vocabulary prompt: "
         f"{'on' if prompt else 'off'}. Requested {meta['requested_utc']}.", "",
         "| | local (frozen) | openai whisper-1 |", "|---|---|---|",
         f"| words | {len(load_frozen_transcript(args.compare)) if args.compare else '—'} | {len(words)} |",
         f"| wall time | 476 s (recorded 2026-09-06, plain run) | {elapsed:.0f} s |",
         f"| cost | $0 (CPU) | {'$%.4f' % cost if cost is not None else 'price unknown'} |",
         f"| audio billed | — | {meta['audio_seconds'] / 60:.1f} min |", ""]
    if args.compare:
        local = load_frozen_transcript(args.compare)
        lc, oc = term_counts(local), term_counts(words)
        L += ["## Term table (the pilot01 ASR terms)", "",
              "| term | local (frozen) | openai |", "|---|---:|---:|"]
        for label in TERMS:
            L.append(f"| {label} | {lc[label]} | {oc[label]} |")
        L += ["", f"Local transcript: `{args.compare}` (unchanged — this run wrote `{out}`).", ""]
        print("\n".join(L[-len(TERMS) - 5:]))
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text("\n".join(L + footer) + "\n")
    print("\n".join(footer))
    print(f"wrote {out} and {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
