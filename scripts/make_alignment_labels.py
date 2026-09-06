#!/usr/bin/env python3
"""Ear-labelled slide timeline -> per-segment ground truth for the frozen audio.

A segment's true slide is whichever slide covers the majority of its duration.
Three rules from the labelling brief, all of which exist because the timeline is
approximate rather than frame-accurate:

* **Zero-length slides cannot win.** Slides 9, 11 and 18 have start == end: they were
  skipped or flashed. A segment straddling one is labelled with whichever neighbour
  covers most of it, and marked ambiguous.
* **The Q&A outro is not a slide.** A segment with >50% of its duration after the last
  slide change gets `true_slide=none`, `section=qa`.
* **Boundaries are soft.** A segment starting or ending within +/-3s of any slide
  boundary is marked ambiguous, because the timeline was read by ear.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

BOUNDARY_TOLERANCE_S = 3.0


def parse_mmss(value: str) -> float | None:
    value = (value or "").strip()
    if not value:
        return None
    minutes, _, seconds = value.partition(":")
    return int(minutes) * 60 + float(seconds)


def load_timeline(path: str | Path):
    slides, outro = [], None
    for row in csv.DictReader(open(path, newline="")):
        start, end = parse_mmss(row["start"]), parse_mmss(row["end"])
        if start is None or end is None:
            continue  # slide 27: never shown
        if row["slide"].strip().lower() == "outro":
            outro = (start, end)
            continue
        slides.append({"slide": int(row["slide"]), "start": start, "end": end,
                       "zero_length": end <= start, "note": row.get("note", "")})
    return slides, outro


def overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def label_segment(seg_start: float, seg_end: float, slides, outro):
    duration = max(seg_end - seg_start, 1e-9)
    ambiguous = False

    if outro and overlap(seg_start, seg_end, *outro) / duration > 0.5:
        return None, "qa", True if _near_boundary(seg_start, seg_end, slides, outro) else False

    # Zero-length slides are excluded from the majority vote by construction: their
    # overlap with anything is 0, so they can never win. Straddling one only matters
    # for the ambiguity flag.
    scored = [(overlap(seg_start, seg_end, s["start"], s["end"]), s["slide"])
              for s in slides if not s["zero_length"]]
    best_overlap, best_slide = max(scored, default=(0.0, None))

    straddled = [s for s in slides if s["zero_length"] and seg_start <= s["start"] <= seg_end]
    if straddled:
        ambiguous = True

    if best_overlap <= 0.0:
        return None, "talk", True

    if _near_boundary(seg_start, seg_end, slides, outro):
        ambiguous = True
    return best_slide, "talk", ambiguous


def _near_boundary(seg_start: float, seg_end: float, slides, outro) -> bool:
    edges = {s["start"] for s in slides} | {s["end"] for s in slides}
    if outro:
        edges |= set(outro)
    return any(abs(t - e) <= BOUNDARY_TOLERANCE_S for e in edges for t in (seg_start, seg_end))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeline", default="data/labels/pilot01/pilot01_slide_timeline.csv")
    ap.add_argument("--index", default="data/processed/index")
    ap.add_argument("--audio-source", default="hsieh.mp3")
    ap.add_argument("--out", default="data/labels/pilot01/pilot01_alignment_labels.csv")
    args = ap.parse_args(argv)

    slides, outro = load_timeline(args.timeline)
    units = [u for u in json.loads((Path(args.index) / "units.json").read_text())
             if u["modality"] == "audio" and Path(u["source_file"]).name == args.audio_source]
    units.sort(key=lambda u: u["location"]["start_s"])

    rows = []
    for u in units:
        start, end = u["location"]["start_s"], u["location"]["end_s"]
        slide, section, ambiguous = label_segment(start, end, slides, outro)
        rows.append({
            "segment_id": u["id"],
            "start": f"{start:.2f}",
            "end": f"{end:.2f}",
            "true_slide": "none" if slide is None else slide,
            "section": section,
            "ambiguous": str(bool(ambiguous)).lower(),
        })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["segment_id", "start", "end",
                                                "true_slide", "section", "ambiguous"])
        writer.writeheader()
        writer.writerows(rows)

    talk = [r for r in rows if r["section"] == "talk"]
    qa = [r for r in rows if r["section"] == "qa"]
    amb = [r for r in talk if r["ambiguous"] == "true"]
    showable = sorted({s["slide"] for s in slides if not s["zero_length"]})
    print(f"wrote {out}")
    print(f"  segments      : {len(rows)}  (talk {len(talk)}, qa {len(qa)})")
    print(f"  ambiguous     : {len(amb)}/{len(talk)} talk segments")
    print(f"  showable slides: {len(showable)}  (zero-length excluded: "
          f"{[s['slide'] for s in slides if s['zero_length']]})")
    print(f"  slides labelled: {len(sorted({r['true_slide'] for r in talk if r['true_slide'] != 'none'}))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
