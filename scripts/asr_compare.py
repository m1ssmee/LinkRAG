#!/usr/bin/env python3
"""Step 0: slide-guided ASR vs plain ASR.

Transcribes the lecture a second time with whisper's `initial_prompt` set to
vocabulary mined from the slide deck, and writes both transcripts to
data/processed/transcripts/ so neither is lost. The plain side is reconstructed
from the existing index's persisted word stream -- same model, same settings, no
prompt -- so only one new transcription runs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pymupdf

from linkrag.core import load_config, setup_logging, stage_timer
from linkrag.ingest.audio import build_asr_prompt, transcribe_segments


def plain_words_from_index(index_dir: Path, stem: str) -> list[tuple[float, float, str]]:
    units = [
        u for u in json.loads((index_dir / "units.json").read_text())
        if u["modality"] == "audio" and u["id"].startswith(f"{stem}:")
    ]
    units.sort(key=lambda u: u["location"]["start_s"])
    return [tuple(w) for u in units for w in u["metadata"]["words"]]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", default="data/raw/pilot01/hsieh.mp3")
    ap.add_argument("--slides", default="data/raw/pilot01/osdi18_slides_hsieh.pdf")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--index", default="data/processed/index")
    ap.add_argument("--out", default="data/processed/transcripts")
    args = ap.parse_args(argv)

    setup_logging()
    cfg = load_config(args.config)
    audio, out = Path(args.audio), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    deck = pymupdf.open(args.slides)
    prompt = build_asr_prompt([page.get_text() for page in deck])
    deck.close()
    print(f"ASR prompt ({len(prompt)} chars):\n  {prompt}\n")

    plain = plain_words_from_index(Path(args.index), audio.stem)
    print(f"plain transcript: {len(plain)} words (reused from {args.index})")

    with stage_timer("asr.guided", file=audio.name):
        segments = transcribe_segments(
            audio,
            model_size=cfg["models"]["whisper"],
            device=cfg["device"],
            compute_type=cfg["models"]["whisper_compute_type"],
            initial_prompt=prompt,
        )
    guided = [(w.start, w.end, w.text) for s in segments for w in s.words]
    print(f"guided transcript: {len(guided)} words")

    for name, words in (("plain", plain), ("guided", guided)):
        path = out / f"{audio.stem}.{name}.json"
        path.write_text(json.dumps(
            {"audio": str(audio), "model": cfg["models"]["whisper"],
             "initial_prompt": prompt if name == "guided" else None,
             "words": [list(w) for w in words]}, ensure_ascii=False))
        print(f"  wrote {path}")

    print("\n--- target term status ---")
    blobs = {n: " ".join(w[2] for w in ws).lower() for n, ws in (("plain", plain), ("guided", guided))}
    for term in ["ingest", "interest", "noscope", "no scope", "yolo", "resnet", "top-k", "top k"]:
        print(f"  {term:10} plain={blobs['plain'].count(term):3}  guided={blobs['guided'].count(term):3}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
