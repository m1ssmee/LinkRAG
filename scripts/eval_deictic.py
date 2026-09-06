#!/usr/bin/env python3
"""Evaluate deictic links against the ear-labelled pointing windows.

Windows mark where the speaker is referring to something on screen. **Outside a
window the label is UNKNOWN, not negative** -- a deictic link there is neither
credited nor penalised, because nobody has said whether the speaker was pointing.

A segment belongs to a window when >=50% of its duration falls inside it, the same
majority rule used to assign slides.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

WINDOWS = "data/labels/pilot01/pilot01_pointing_windows.csv"
LABELS = "data/labels/pilot01/pilot01_alignment_labels_v2.csv"
PAIRS = "reports/deictic_pairs_pilot01.csv"


def mmss(v: str) -> float:
    m, _, s = v.strip().partition(":")
    return int(m) * 60 + float(s)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", default=WINDOWS)
    ap.add_argument("--labels", default=LABELS)
    ap.add_argument("--pairs", default=PAIRS)
    ap.add_argument("--index", default="data/processed/index")
    ap.add_argument("--deck", default="osdi18_slides_hsieh.pdf")
    ap.add_argument("--out", default="reports/deictic_eval_pilot01.md")
    args = ap.parse_args(argv)

    units = json.loads((Path(args.index) / "units.json").read_text())
    fig_pages = {u["location"]["page"] for u in units
                 if u["modality"] == "figure" and Path(u["source_file"]).name == args.deck}
    build = {10: {10, 11}, 11: {10, 11}, 17: {17, 18}, 18: {17, 18}}
    accept = lambda p: build.get(p, {p})

    segs = [r for r in csv.DictReader(open(args.labels, newline=""))]
    span = {r["segment_id"]: (float(r["start"]), float(r["end"])) for r in segs}

    pairs = list(csv.DictReader(open(args.pairs, newline="")))
    by_seg = defaultdict(list)
    for p in pairs:
        by_seg[p["segment_id"]].append(p)

    windows = []
    for r in csv.DictReader(open(args.windows, newline="")):
        lo, hi = mmss(r["start"]), mmss(r["end"])
        slide = int(r["slide"])
        pages = accept(slide)
        if r["target"] == "external_demo":
            kind = "known-negative"
        elif r["target"] == "table":
            kind = "unresolvable (table; no table extraction)"
        elif not (pages & fig_pages):
            kind = "unresolvable (no figure extracted for this slide)"
        else:
            kind = "scorable"
        members = [s["segment_id"] for s in segs
                   if (min(float(s["end"]), hi) - max(float(s["start"]), lo))
                   >= 0.5 * (float(s["end"]) - float(s["start"]))]
        windows.append({**r, "lo": lo, "hi": hi, "slide": slide, "pages": pages,
                        "kind": kind, "segments": members})

    L = []
    w = L.append
    w("# Deictic evaluation against ear-labelled pointing windows — pilot01")
    w("")
    w("Windows: `data/labels/pilot01/pilot01_pointing_windows.csv` (n=10). Slides:")
    w("`pilot01_slide_timeline_v2.csv`. **Outside a pointing window the label is")
    w("UNKNOWN, not negative** — deictic links there are neither credited nor penalised.")
    w("A segment belongs to a window when ≥50% of its duration falls inside it.")
    w("")
    w("## Window inventory")
    w("")
    w("| window | slide | target | segments | status |")
    w("|---|---:|---|---:|---|")
    for x in windows:
        w(f"| {x['start']}–{x['end']} | {x['slide']} | `{x['target']}` "
          f"| {len(x['segments'])} | {x['kind']} |")
    scorable = [x for x in windows if x["kind"] == "scorable"]
    unres = [x for x in windows if x["kind"].startswith("unresolvable")]
    neg = [x for x in windows if x["kind"] == "known-negative"]
    w("")
    w(f"**{len(scorable)} of {len(windows)} windows are scorable.** {len(unres)} are "
      f"unresolvable by the current module and {len(neg)} is a known negative. The deck "
      f"has extracted figures on pages {sorted(p for p in fig_pages if p <= 26)} only — "
      f"figure extraction, not the deictic module, is what caps this evaluation.")
    w("")

    # (a) precision on positives
    w("## (a) Precision on positives")
    w("")
    w("For each deictic pair whose segment sits inside a *scorable* window: is the linked")
    w("figure on the labelled slide (or its build-pair)?")
    w("")
    tiers = defaultdict(lambda: [0, 0])
    rows_a = []
    for x in scorable:
        for sid in x["segments"]:
            for p in by_seg.get(sid, []):
                ok = int(p["figure_page"]) in x["pages"]
                t = int(p["tier"])
                tiers[t][0] += ok
                tiers[t][1] += 1
                rows_a.append((sid, x["slide"], p["figure_page"], t, ok, p["phrase"]))
    total_ok = sum(v[0] for v in tiers.values())
    total_n = sum(v[1] for v in tiers.values())
    w("| tier | correct | pairs | precision |")
    w("|---|---:|---:|---:|")
    for t in sorted(tiers):
        ok, n = tiers[t]
        w(f"| {t} | {ok} | {n} | {100*ok/n:.0f}% |")
    w(f"| **all** | **{total_ok}** | **{total_n}** | "
      f"**{(100*total_ok/total_n) if total_n else float('nan'):.0f}%** |")
    w("")
    if rows_a:
        w("| segment | window slide | linked figure page | tier | correct | phrase |")
        w("|---|---:|---:|---:|---|---|")
        for sid, slide, page, t, ok, phrase in rows_a:
            w(f"| `{sid}` | {slide} | {page} | {t} | {'✓' if ok else '✗'} | `{phrase}` |")
    w("")

    # (b) detection recall
    w("## (b) Detection recall")
    w("")
    w("Per window: what fraction of its segments produced at least one deictic pair?")
    w("")
    w("| window | slide | status | segments | with a pair | recall | tiers seen |")
    w("|---|---:|---|---:|---:|---:|---|")
    hit_all = seg_all = 0
    for x in windows:
        hits = [s for s in x["segments"] if by_seg.get(s)]
        seen = sorted({int(p["tier"]) for s in hits for p in by_seg[s]})
        if x["kind"] == "scorable":
            hit_all += len(hits)
            seg_all += len(x["segments"])
        r = f"{100*len(hits)/len(x['segments']):.0f}%" if x["segments"] else "—"
        w(f"| {x['start']}–{x['end']} | {x['slide']} | {x['kind'].split(' (')[0]} "
          f"| {len(x['segments'])} | {len(hits)} | {r} | {seen or '—'} |")
    w("")
    w(f"**Overall recall over scorable windows: {hit_all}/{seg_all} = "
      f"{(100*hit_all/seg_all) if seg_all else float('nan'):.0f}%.**")
    w("")

    # (c) known negative
    w("## (c) Slide 24 — external demo (known negative)")
    w("")
    fp = [p for p in pairs if int(p["figure_page"]) == 24]
    w(f"Deictic pairs linking to a figure on slide 24: **{len(fp)}**. The demo is not in")
    w("the deck and no figure unit exists on that page, so any such link is a false")
    w("positive by construction.")
    if fp:
        w("")
        for p in fp:
            w(f"- `{p['segment_id']}` → `{p['figure_id']}` tier {p['tier']} `{p['phrase']}`")
    demo_w = neg[0] if neg else None
    if demo_w:
        during = [s for s in demo_w["segments"] if by_seg.get(s)]
        w("")
        w(f"During the demo window itself ({demo_w['start']}–{demo_w['end']}, "
          f"{len(demo_w['segments'])} segments), **{len(during)} of them produced a "
          f"deictic pair**.")
        if during:
            w("")
            w("Each is a false positive: the module pointed at a deck figure while the")
            w("speaker was demonstrating software that is not in the deck.")
            for s in during:
                for p in by_seg[s]:
                    w(f"- `{s}` → page {p['figure_page']} tier {p['tier']} `{p['phrase']}`")
        else:
            w("")
            w("**This is the cleanest negative result in the evaluation.** The speaker is")
            w("talking continuously about something visual for 2m13s, and the module emits")
            w("nothing — because the alignment holds slide 24, and slide 24 has no")
            w("extracted figure to link to. The correct behaviour here is produced by the")
            w("absence of a candidate rather than by a decision, so it should not be read")
            w("as evidence that the module rejects bad referents.")
    w("")

    # (d) unresolvable
    w("## (d) Unresolvable by the current module")
    w("")
    w("Reported separately, **not as misses**: no figure unit exists for the module to")
    w("link to, so a miss here says nothing about deictic resolution.")
    w("")
    w("| window | slide | target | why | segments | pairs emitted |")
    w("|---|---:|---|---|---:|---:|")
    for x in unres:
        n = sum(len(by_seg.get(s, [])) for s in x["segments"])
        w(f"| {x['start']}–{x['end']} | {x['slide']} | `{x['target']}` "
          f"| {x['kind'].split('(')[1].rstrip(')')} | {len(x['segments'])} | {n} |")
    w("")
    w("Any pair emitted in these rows links to a figure on a *different* slide, since the")
    w("labelled slide has none — the alignment placed the segment correctly and the")
    w("deictic module then had nothing valid to choose from.")
    w("")

    w("## Summary and what these numbers can carry")
    w("")
    w(f"| metric | value | n |")
    w("|---|---:|---:|")
    w(f"| precision on positives | {(100*total_ok/total_n) if total_n else 0:.0f}% "
      f"| {total_n} pairs |")
    w(f"| detection recall (scorable windows) | "
      f"{(100*hit_all/seg_all) if seg_all else 0:.0f}% | {seg_all} segments |")
    w(f"| false positives on the known negative | {len(fp)} | slide 24 |")
    w(f"| windows scorable | {len(scorable)}/{len(windows)} | — |")
    w("")
    w("**These are small numbers and should be quoted with n attached.** Precision rests")
    w(f"on {total_n} pairs from {len(scorable)} windows, all of them tier 3 — the")
    w("evaluation contains no tier-1 or tier-2 pair inside a scorable window, so it says")
    w("nothing about whether the tiering helps here. The 82.4% slide-agreement figure")
    w("reported earlier covers all 51 pairs and remains the broader measure; this one is")
    w("narrower and stricter, and only the intersection of both is well evidenced.")
    w("")
    w("**The binding constraint is figure extraction, not deixis.** Seven of ten windows")
    w("cannot be scored because the slide the speaker was pointing at has no extracted")
    w("figure unit. Recovering those slides would do more for this evaluation than any")
    w("change to the deictic module.")
    w("")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n")
    print(f"wrote {out}")
    print(f"  scorable windows {len(scorable)}/{len(windows)}  precision "
          f"{total_ok}/{total_n}  recall {hit_all}/{seg_all}  slide-24 FPs {len(fp)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
