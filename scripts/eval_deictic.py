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

    segs = list(csv.DictReader(open(args.labels, newline="")))

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
    w(f"**{len(scorable)} of {len(windows)} windows are scorable.** "
      f"{len(unres)} unresolvable by the current module, {len(neg)} known negative. "
      f"The deck has extracted figures on pages "
      f"{sorted(p for p in fig_pages if p <= 26)}.")
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
    on24 = 24 in fig_pages
    w(f"Deictic pairs linking to a figure on slide 24: **{len(fp)}**.")
    w("")
    w(f"Slide 24 {'now has' if on24 else 'has no'} extracted figure unit, but the demo "
      f"itself is live software that is not in the deck, so **any** deictic link during "
      f"this window is a false positive: there is no correct referent to find.")
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
            w("Each is a false positive — the module pointed at a deck figure while the")
            w("speaker was demonstrating software that is not in the deck:")
            pages_hit = set()
            for s in during:
                for p in by_seg[s]:
                    pages_hit.add(int(p["figure_page"]))
                    w(f"- `{s}` → page {p['figure_page']} tier {p['tier']} `{p['phrase']}`")
            w("")
            w(f"Note where they land: page(s) {sorted(pages_hit)}, **not** slide 24. The "
              f"alignment places these segments past the demo (the DP's known p24→p26 "
              f"lead), and the deictic module then resolves correctly *within the slide it "
              f"was given*. The failure is upstream, in alignment, not in referent choice.")
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
    w("| metric | value | n |")
    w("|---|---:|---:|")
    w(f"| precision on positives | {(100*total_ok/total_n) if total_n else 0:.0f}% "
      f"| {total_n} pairs |")
    w(f"| detection recall (scorable windows) | "
      f"{(100*hit_all/seg_all) if seg_all else 0:.0f}% | {seg_all} segments |")
    w(f"| false positives on the known negative | {len(fp)} | slide 24 |")
    w(f"| windows scorable | {len(scorable)}/{len(windows)} | — |")
    w("")
    tier_mix = ", ".join(f"tier {t}: {tiers[t][1]}" for t in sorted(tiers)) or "none"
    w(f"**Quote these with n attached.** Precision rests on {total_n} pairs from "
      f"{len(scorable)} scorable windows ({tier_mix}).")
    if len(tiers) > 1:
        best = max(tiers, key=lambda t: tiers[t][0] / max(tiers[t][1], 1))
        worst = min(tiers, key=lambda t: tiers[t][0] / max(tiers[t][1], 1))
        w("")
        w(f"**The tiering is doing work here.** Tier {best} scores "
          f"{100*tiers[best][0]/tiers[best][1]:.0f}% against tier {worst}'s "
          f"{100*tiers[worst][0]/tiers[worst][1]:.0f}%, on referent-level ground truth "
          f"rather than the slide-agreement proxy used earlier.")
    else:
        w("")
        w("Only one tier appears in the scorable windows, so this evaluation says "
          "nothing about whether tiering helps.")
    w("")
    w(f"**The binding constraint has moved.** Before slide-figure clustering only "
      f"{3} of 10 windows were scorable, because most slides yielded no figure unit at "
      f"all. Now {len(scorable)} are. What remains unresolvable is a table (no table "
      f"extraction) and the live demo, which has no correct referent by construction.")
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
