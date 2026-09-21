# MaViLS — like-for-like protocol (all: 20 lectures)

Segments: their sentence granularity · slide text: tesseract OCR of every rendered page (their slide-side input) · embedder `BAAI/bge-m3` + BM25 + IDF overlap · DP λ=0.05 σ=0.02 β=0.15 B=2 μ=0.02 · scored with sklearn exactly as their evaluation script.

**Their scoring, verbatim from `evaluation/evaluate_recall_precision.py`:** `ground_truth_labels = ground_truth_column.unique()`; `mask = (ground_truth_column != -1)`; `f1_score(filtered_ground_truth, filtered_result, labels=ground_truth_labels, average='micro')`. One row per transcript sentence (`generate_output_dict_by_sentence`), per lecture, then the unweighted mean over lectures. `labels` comes from the unfiltered column, so -1 is a label: predicting -1 on a labelled sentence is a false positive for -1 and a miss for the true slide, i.e. **abstention cannot raise their F1**; it can only be seen in precision-on-answered and coverage, which we report alongside. Their audio-only similarity is distiluse cosine between the sentence and **tesseract OCR of the rendered slide image** (`matching_algorithm.py`), decoded with their DP (penalty 0.1·|Δslide|, ×2 backwards, no skip penalty).

**Column correspondence.** Their *audio-only* ⇔ our `dp` column **when run with `--slide-text ocr` at sentence granularity** (same input protocol; our similarity and DP). `naive` is the same matrix without sequence structure. Their *all-features* uses video frames we do not consume.

| lecture | n | slides | text layer | **dp** | naive | their audio | their all | Δ dp − their audio |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| ML for health | 489 | 77 | full | **0.29** | 0.25 | 0.57 | 0.95 | -0.28 |
| Decarbonization | 275 | 45 | full | **0.44** | 0.39 | 0.52 | 0.93 | -0.08 |
| Climate & Cities | 143 | 44 | full | **0.60** | 0.48 | 0.49 | 0.86 | +0.11 |
| Climate policies | 274 | 24 | full | **0.76** | 0.42 | 0.79 | 0.93 | -0.03 |
| Cognitive robotics | 735 | 62 | partial | **0.43** | 0.19 | 0.42 | 0.66 | +0.01 |
| Computer Vision | 80 | 18 | full | **0.72** | 0.43 | 0.65 | 1.00 | +0.07 |
| Productdesign | 247 | 27 | full | **0.58** | 0.41 | 0.72 | 0.71 | -0.14 |
| Cryptocurrency | 500 | 52 | full | **0.30** | 0.18 | 0.21 | 0.84 | +0.09 |
| Deep learning | 377 | 39 | full | **0.70** | 0.44 | 0.79 | 0.99 | -0.09 |
| Image processing | 124 | 21 | none | **0.19** | 0.23 | 0.53 | 0.97 | -0.34 |
| Numerics | 713 | 50 | full | **0.50** | 0.21 | 0.46 | 0.81 | +0.04 |
| Phonetics | 221 | 38 | full | **0.35** | 0.28 | 0.05 | 0.65 | +0.30 |
| Physics | 576 | 31 | full | **0.69** | 0.37 | 0.60 | 0.93 | +0.09 |
| Psychology | 893 | 64 | full | **0.48** | 0.24 | 0.58 | 0.80 | -0.10 |
| Reinforcement | 1201 | 57 | full | **0.24** | 0.12 | 0.31 | 0.75 | -0.07 |
| Sensory systems | 421 | 37 | partial | **0.21** | 0.18 | 0.46 | 0.53 | -0.25 |
| Short range | 350 | 21 | full | **0.45** | 0.27 | 0.57 | 0.80 | -0.12 |
| Solar resource | 518 | 60 | full | **0.32** | 0.14 | 0.46 | 0.53 | -0.14 |
| Team dynamics | 55 | 24 | full | **0.67** | 0.49 | 0.89 | 0.96 | -0.22 |
| Computation theory | 554 | 12 | full | **0.30** | 0.22 | 0.56 | 0.84 | -0.26 |
| **mean** | 8746 | | | **0.46** | 0.30 | 0.53 | 0.82 | -0.07 |

`dp` above their audio-only on 7/20 lectures.


## Reading (2026-09-22)

**Like-for-like: 0.46 vs their 0.53 (−0.07).** Under their protocol exactly — sentence
granularity, OCR of rendered pages on the slide side, their sklearn F1 — our DP averages
**0.461**, naive argmax on the same matrix 0.297, their audio-only 0.53. We are above
their audio column on 7/20 lectures (Climate & Cities, Cognitive robotics, Computer
Vision, Cryptocurrency, Numerics, Phonetics, Physics) and below on 13. Two earlier
tables are kept as **protocol deviations**: PDF text layer instead of page OCR
(`mavils_alignment_layer_sentence.md`, 0.41) and 30-second windows on the text layer
(`mavils_alignment_layer_w30.md`, 0.45). The 30-second granularity is LinkRAG's design
choice and is not what they scored; the text layer turned out to be *worse* input than
OCR on this benchmark (see the Decarbonization diagnosis).

**The DP is where our number comes from.** +0.16 over naive on average under their
protocol; on the one deck with no text layer at all (Image processing) it is below
naive (0.19 vs 0.23), and on the partial-layer decks it is roughly level. That is
finding 6 in `DESIGN.md`.

### Decarbonization — diagnosis (before anything was changed)

`reports/mavils_inspect_decarbonization_layer.md` / `_ocr.md`, heatmap
`mavils_inspect_cities_and_decarbonization.png`. The deck is a **Beamer-style build
deck**: 22 of its 45 pages continue the previous page's text (pages 2–5 all begin
"Preparing for class — Introductory reading — short & skimmable…", each adding one
bullet), and 7 pages have a text layer identical to another page's. Their labels
treat every build page as its own slide. On the PDF **text layer**, the last build of
each group is a superset of the earlier ones, so its BM25/IDF/dense score dominates
every sentence in the group and the monotone path parks there: the first ten
mismatches are all "ours p5, label p2/p3/p4" with S(ours) ≥ S(label). That gives
0.21 (naive 0.22 — the DP cannot help when the matrix cannot tell the builds
apart). On **page OCR** — their input — earlier builds render fewer bullets, the
matrix separates them, and the same DP scores 0.44 against their 0.52. So the
Decarbonization gap was an *input* effect (text layer vs rendered page), not a DP
effect, and it is the reason the like-for-like table above uses page OCR. No
parameter was changed for this finding.

### Held-out study — what the designed responses did (`mavils_heldout_study.md`)

Split fixed by seed 20260922 (`results/external/mavils_split.json`); parameters set
on the tune half only (`mavils_tuned.json`), then applied to the test half.

| test half (10 lectures) | naive | dp | dp + abstention | dp + abstention + flatness | their audio |
|---|---:|---:|---:|---:|---:|
| their F1 | 0.281 | **0.452** | 0.406 | 0.406 | 0.51 |

- **Flatness-scaled skip penalty: null effect, inspected (measurement rule 1).** The
  three grid values gave identical F1 to three decimals and *identical paths*. Cause:
  at the pilot σ = 0.02 the skip charge is far too small for its row-scaling to move
  any decision; the mechanism itself works (at a diagnostic σ = 0.2 the paths differ on
  17 of 717 sentences for Sensory systems). A flat row's problem is not the cost of
  moving on but the absence of evidence — which is the abstention's job, not σ's. A σ
  sweep on the tune half would be the next study; not done here.
- **Abstention** at min_segment_sim = 0.5055 (chosen on the tune half: precision-on-
  answered 0.486 vs 0.465 at coverage 0.87): on the test half it raises
  precision-on-answered on 6/10 lectures by 0–4 points for 3–24 points of coverage,
  and lowers their F1 (0.452 → 0.406) exactly as the protocol predicts — under their
  scoring an abstention *is* a wrong slide. It is the right behaviour for a link layer
  (a segment with no recognisable slide should not get a link) and the wrong move for
  this leaderboard.
