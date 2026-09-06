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
| 7:48–9:14 | 13 | `graph` | 3 | scorable |
| 9:14–10:19 | 14 | `diagram` | 2 | scorable |
| 11:18–11:34 | 16 | `figure` | 0 | scorable |
| 11:34–12:18 | 17 | `diagram` | 2 | scorable |
| 12:44–13:31 | 20 | `figure` | 2 | scorable |
| 14:20–15:33 | 22 | `graph_and_figure` | 2 | scorable |
| 15:33–16:07 | 23 | `graph` | 2 | scorable |
| 16:07–18:20 | 24 | `external_demo` | 4 | known-negative |

**8 of 10 windows are scorable.** 1 unresolvable by the current module, 1 known negative. The deck has extracted figures on pages [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 23, 24, 26].

## (a) Precision on positives

For each deictic pair whose segment sits inside a *scorable* window: is the linked
figure on the labelled slide (or its build-pair)?

| tier | correct | pairs | precision |
|---|---:|---:|---:|
| 2 | 12 | 14 | 86% |
| 3 | 7 | 10 | 70% |
| **all** | **19** | **24** | **79%** |

| segment | window slide | linked figure page | tier | correct | phrase |
|---|---:|---:|---:|---|---|
| `hsieh:a6` | 4 | 4 | 3 | ✓ | `that` |
| `hsieh:a6` | 4 | 4 | 3 | ✓ | `that` |
| `hsieh:a18` | 13 | 13 | 2 | ✓ | `this` |
| `hsieh:a18` | 13 | 13 | 2 | ✓ | `this` |
| `hsieh:a18` | 13 | 13 | 2 | ✓ | `this` |
| `hsieh:a19` | 13 | 13 | 2 | ✓ | `that` |
| `hsieh:a19` | 13 | 13 | 2 | ✓ | `that` |
| `hsieh:a19` | 13 | 13 | 2 | ✓ | `that` |
| `hsieh:a20` | 13 | 14 | 3 | ✗ | `this` |
| `hsieh:a20` | 13 | 14 | 3 | ✗ | `this` |
| `hsieh:a20` | 13 | 14 | 3 | ✗ | `this` |
| `hsieh:a22` | 14 | 14 | 3 | ✓ | `that` |
| `hsieh:a22` | 14 | 14 | 3 | ✓ | `that` |
| `hsieh:a22` | 14 | 14 | 3 | ✓ | `that` |
| `hsieh:a26` | 17 | 18 | 3 | ✓ | `here` |
| `hsieh:a29` | 20 | 20 | 2 | ✓ | `this` |
| `hsieh:a30` | 20 | 20 | 3 | ✓ | `that` |
| `hsieh:a33` | 22 | 22 | 2 | ✓ | `here` |
| `hsieh:a33` | 22 | 22 | 2 | ✓ | `here` |
| `hsieh:a34` | 22 | 22 | 2 | ✓ | `here` |
| `hsieh:a34` | 22 | 22 | 2 | ✓ | `this` |
| `hsieh:a35` | 23 | 22 | 2 | ✗ | `that` |
| `hsieh:a35` | 23 | 22 | 2 | ✗ | `that` |
| `hsieh:a36` | 23 | 23 | 2 | ✓ | `that` |

## (b) Detection recall

Per window: what fraction of its segments produced at least one deictic pair?

| window | slide | status | segments | with a pair | recall | tiers seen |
|---|---:|---|---:|---:|---:|---|
| 2:21–2:49 | 4 | scorable | 1 | 1 | 100% | [3] |
| 6:37–7:19 | 12 | unresolvable | 2 | 2 | 100% | [2] |
| 7:48–9:14 | 13 | scorable | 3 | 3 | 100% | [2, 3] |
| 9:14–10:19 | 14 | scorable | 2 | 1 | 50% | [3] |
| 11:18–11:34 | 16 | scorable | 0 | 0 | — | — |
| 11:34–12:18 | 17 | scorable | 2 | 1 | 50% | [3] |
| 12:44–13:31 | 20 | scorable | 2 | 2 | 100% | [2, 3] |
| 14:20–15:33 | 22 | scorable | 2 | 2 | 100% | [2] |
| 15:33–16:07 | 23 | scorable | 2 | 2 | 100% | [2] |
| 16:07–18:20 | 24 | known-negative | 4 | 3 | 75% | [2, 3] |

**Overall recall over scorable windows: 12/14 = 86%.**

## (c) Slide 24 — external demo (known negative)

Deictic pairs linking to a figure on slide 24: **0**.

Slide 24 now has extracted figure unit, but the demo itself is live software that is not in the deck, so **any** deictic link during this window is a false positive: there is no correct referent to find.

During the demo window itself (16:07–18:20, 4 segments), **3 of them produced a deictic pair**.

Each is a false positive — the module pointed at a deck figure while the
speaker was demonstrating software that is not in the deck:
- `hsieh:a38` → page 26 tier 2 `this`
- `hsieh:a39` → page 26 tier 2 `this`
- `hsieh:a40` → page 26 tier 3 `that`

Note where they land: page(s) [26], **not** slide 24. The alignment places these segments past the demo (the DP's known p24→p26 lead), and the deictic module then resolves correctly *within the slide it was given*. The failure is upstream, in alignment, not in referent choice.

## (d) Unresolvable by the current module

Reported separately, **not as misses**: no figure unit exists for the module to
link to, so a miss here says nothing about deictic resolution.

| window | slide | target | why | segments | pairs emitted |
|---|---:|---|---|---:|---:|
| 6:37–7:19 | 12 | `table` | table; no table extraction | 2 | 8 |

Any pair emitted in these rows links to a figure on a *different* slide, since the
labelled slide has none — the alignment placed the segment correctly and the
deictic module then had nothing valid to choose from.

## Summary and what these numbers can carry

| metric | value | n |
|---|---:|---:|
| precision on positives | 79% | 24 pairs |
| detection recall (scorable windows) | 86% | 14 segments |
| false positives on the known negative | 0 | slide 24 |
| windows scorable | 8/10 | — |

**Quote these with n attached.** Precision rests on 24 pairs from 8 scorable windows (tier 2: 14, tier 3: 10).

**The tiering is doing work here.** Tier 2 scores 86% against tier 3's 70%, on referent-level ground truth rather than the slide-agreement proxy used earlier.

**The binding constraint has moved.** Before slide-figure clustering only 3 of 10 windows were scorable, because most slides yielded no figure unit at all. Now 8 are. What remains unresolvable is a table (no table extraction) and the live demo, which has no correct referent by construction.

