# Deictic evaluation against ear-labelled pointing windows — pilot01

Windows: `data/labels/pilot01/pilot01_pointing_windows.csv` (n=10). Slides:
`pilot01_slide_timeline_v2.csv`. **Outside a pointing window the label is
UNKNOWN, not negative** — deictic links there are neither credited nor penalised.
A segment belongs to a window when ≥50% of its duration falls inside it.

## Window inventory

| window | slide | target | segments | status |
|---|---:|---|---:|---|
| 2:21–2:49 | 4 | `diagram` | 1 | scorable |
| 6:37–7:19 | 12 | `table` | 2 | unresolvable (table; no table extraction) |
| 7:48–9:14 | 13 | `graph` | 3 | unresolvable (no figure extracted for this slide) |
| 9:14–10:19 | 14 | `diagram` | 2 | scorable |
| 11:18–11:34 | 16 | `figure` | 0 | unresolvable (no figure extracted for this slide) |
| 11:34–12:18 | 17 | `diagram` | 2 | scorable |
| 12:44–13:31 | 20 | `figure` | 2 | unresolvable (no figure extracted for this slide) |
| 14:20–15:33 | 22 | `graph_and_figure` | 2 | unresolvable (no figure extracted for this slide) |
| 15:33–16:07 | 23 | `graph` | 2 | unresolvable (no figure extracted for this slide) |
| 16:07–18:20 | 24 | `external_demo` | 4 | known-negative |

**3 of 10 windows are scorable.** 6 are unresolvable by the current module and 1 is a known negative. The deck has extracted figures on pages [1, 2, 3, 4, 5, 7, 12, 14, 17, 18] only — figure extraction, not the deictic module, is what caps this evaluation.

## (a) Precision on positives

For each deictic pair whose segment sits inside a *scorable* window: is the linked
figure on the labelled slide (or its build-pair)?

| tier | correct | pairs | precision |
|---|---:|---:|---:|
| 3 | 6 | 6 | 100% |
| **all** | **6** | **6** | **100%** |

| segment | window slide | linked figure page | tier | correct | phrase |
|---|---:|---:|---:|---|---|
| `hsieh:a6` | 4 | 4 | 3 | ✓ | `that` |
| `hsieh:a6` | 4 | 4 | 3 | ✓ | `that` |
| `hsieh:a22` | 14 | 14 | 3 | ✓ | `that` |
| `hsieh:a22` | 14 | 14 | 3 | ✓ | `that` |
| `hsieh:a22` | 14 | 14 | 3 | ✓ | `that` |
| `hsieh:a26` | 17 | 18 | 3 | ✓ | `here` |

## (b) Detection recall

Per window: what fraction of its segments produced at least one deictic pair?

| window | slide | status | segments | with a pair | recall | tiers seen |
|---|---:|---|---:|---:|---:|---|
| 2:21–2:49 | 4 | scorable | 1 | 1 | 100% | [3] |
| 6:37–7:19 | 12 | unresolvable | 2 | 2 | 100% | [2] |
| 7:48–9:14 | 13 | unresolvable | 3 | 3 | 100% | [2, 3] |
| 9:14–10:19 | 14 | scorable | 2 | 1 | 50% | [3] |
| 11:18–11:34 | 16 | unresolvable | 0 | 0 | — | — |
| 11:34–12:18 | 17 | scorable | 2 | 1 | 50% | [3] |
| 12:44–13:31 | 20 | unresolvable | 2 | 0 | 0% | — |
| 14:20–15:33 | 22 | unresolvable | 2 | 0 | 0% | — |
| 15:33–16:07 | 23 | unresolvable | 2 | 0 | 0% | — |
| 16:07–18:20 | 24 | known-negative | 4 | 0 | 0% | — |

**Overall recall over scorable windows: 3/5 = 60%.**

## (c) Slide 24 — external demo (known negative)

Deictic pairs linking to a figure on slide 24: **0**. The demo is not in
the deck and no figure unit exists on that page, so any such link is a false
positive by construction.

During the demo window itself (16:07–18:20, 4 segments), **0 of them produced a deictic pair**.

**This is the cleanest negative result in the evaluation.** The speaker is
talking continuously about something visual for 2m13s, and the module emits
nothing — because the alignment holds slide 24, and slide 24 has no
extracted figure to link to. The correct behaviour here is produced by the
absence of a candidate rather than by a decision, so it should not be read
as evidence that the module rejects bad referents.

## (d) Unresolvable by the current module

Reported separately, **not as misses**: no figure unit exists for the module to
link to, so a miss here says nothing about deictic resolution.

| window | slide | target | why | segments | pairs emitted |
|---|---:|---|---|---:|---:|
| 6:37–7:19 | 12 | `table` | table; no table extraction | 2 | 6 |
| 7:48–9:14 | 13 | `graph` | no figure extracted for this slide | 3 | 7 |
| 11:18–11:34 | 16 | `figure` | no figure extracted for this slide | 0 | 0 |
| 12:44–13:31 | 20 | `figure` | no figure extracted for this slide | 2 | 0 |
| 14:20–15:33 | 22 | `graph_and_figure` | no figure extracted for this slide | 2 | 0 |
| 15:33–16:07 | 23 | `graph` | no figure extracted for this slide | 2 | 0 |

Any pair emitted in these rows links to a figure on a *different* slide, since the
labelled slide has none — the alignment placed the segment correctly and the
deictic module then had nothing valid to choose from.

## Summary and what these numbers can carry

| metric | value | n |
|---|---:|---:|
| precision on positives | 100% | 6 pairs |
| detection recall (scorable windows) | 60% | 5 segments |
| false positives on the known negative | 0 | slide 24 |
| windows scorable | 3/10 | — |

**These are small numbers and should be quoted with n attached.** Precision rests
on 6 pairs from 3 windows, all of them tier 3 — the
evaluation contains no tier-1 or tier-2 pair inside a scorable window, so it says
nothing about whether the tiering helps here. The 82.4% slide-agreement figure
reported earlier covers all 51 pairs and remains the broader measure; this one is
narrower and stricter, and only the intersection of both is well evidenced.

**The binding constraint is figure extraction, not deixis.** Seven of ten windows
cannot be scored because the slide the speaker was pointing at has no extracted
figure unit. Recovering those slides would do more for this evaluation than any
change to the deictic module.

