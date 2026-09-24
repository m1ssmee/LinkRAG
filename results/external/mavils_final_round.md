# MaViLS — final round: visibility gate and sentence-time frames

Their protocol: sentence granularity, page OCR on the slide side, their sklearn F1. Cells: **their F1 (precision-on-answered / coverage)**. Decoder: our DP at σ = 0.2. Every column is the three-way fusion (image swiftformer / frame OCR bm25 / speech-to-slide text, min-max scaled, `fuse_many`). LLM-free, $0.

- *previous best* — representative frames (one per dHash segment, 320 px), weights 0.5/0.25/0.25, no gate (`mavils_frame_ocr.md`, 0.810).
- *+ gate* — the same with the slide-visibility gate of `mavils_visibility_gate.md` (W = 1, M = 0.02).
- *+ sentence-time* — the frame at each sentence timestamp at native resolution (MaViLS's own choice), OCR'd at ≥ 960 px longest side; weights re-chosen on tune.
- *+ both* — sentence-time frames and a gate re-tuned on them.

Videos: the MaViLS Kaggle zip, found again as `~/Downloads/archive (2).zip` (Kaggle's default name; same 21 files as the earlier `video.zip`); sentence-time frames extracted, then the videos removed. Native resolution varies by lecture from 320×240 to 1280×720.

## Tune half

Sentence-time three-way weights (visual / frame_ocr / theirs), top 5:

| weights | tune |
|---|---:|
| 0.25 / 0.50 / 0.25 | 0.863 ← |
| 0.25 / 0.25 / 0.50 | 0.862 |
| 0.00 / 0.75 / 0.25 | 0.854 |
| 0.50 / 0.25 / 0.25 | 0.850 |
| 0.00 / 0.50 / 0.50 | 0.843 |

Sentence-time gate: no gate 0.863; best W = 1, M = 0.02 at 0.878 → **gate on (W = 1, M = 0.02)**.

| column | tune mean |
|---|---:|
| previous best | 0.793 |
| + gate | 0.824 |
| + sentence-time | 0.863 |
| + both | 0.878 ← |

Chosen on tune: **+ both**.

## Test half — reported once

| lecture | previous best | + gate | + sentence-time | + both | their all-features (T2) |
|---|---:|---:|---:|---:|---:|
| ML for health | 0.93 (0.93 / 1.00) | 0.94 (0.94 / 1.00) | 0.91 (0.91 / 1.00) | 0.91 (0.91 / 1.00) | 0.95 |
| Climate & Cities | 0.84 (0.84 / 1.00) | 0.81 (0.81 / 1.00) | 0.98 (0.97 / 1.00) | 0.98 (0.97 / 1.00) | 0.86 |
| Deep learning | 0.99 (0.99 / 1.00) | 0.99 (0.99 / 1.00) | 0.99 (0.99 / 1.00) | 0.99 (0.99 / 1.00) | 0.99 |
| Numerics | 0.71 (0.71 / 1.00) | 0.71 (0.71 / 1.00) | 0.89 (0.89 / 1.00) | 0.89 (0.89 / 1.00) | 0.81 |
| Phonetics | 0.89 (0.89 / 1.00) | 0.86 (0.86 / 1.00) | 0.99 (0.99 / 1.00) | 0.99 (0.99 / 1.00) | 0.65 |
| Reinforcement | 0.52 (0.52 / 1.00) | 0.46 (0.46 / 1.00) | 0.65 (0.64 / 1.00) | 0.67 (0.66 / 1.00) | 0.75 |
| Short range | 0.84 (0.84 / 1.00) | 0.84 (0.84 / 1.00) | 0.58 (0.58 / 1.00) | 0.58 (0.58 / 1.00) | 0.80 |
| Solar resource | 0.60 (0.59 / 1.00) | 0.72 (0.71 / 1.00) | 0.80 (0.78 / 1.00) | 0.81 (0.79 / 1.00) | 0.53 |
| Computation theory | 0.96 (0.96 / 1.00) | 0.96 (0.96 / 1.00) | 0.87 (0.87 / 1.00) | 0.87 (0.87 / 1.00) | 0.84 |
| **mean** | **0.810** | **0.809** | **0.850** | **0.853** | 0.80 |

## Paired against their all-features (+ both)

| lecture | ours | theirs | difference |
|---|---:|---:|---:|
| ML for health | 0.91 | 0.95 | -0.04 |
| Climate & Cities | 0.98 | 0.86 | +0.12 |
| Deep learning | 0.99 | 0.99 | +0.00 |
| Numerics | 0.89 | 0.81 | +0.08 |
| Phonetics | 0.99 | 0.65 | +0.34 |
| Reinforcement | 0.67 | 0.75 | -0.08 |
| Short range | 0.58 | 0.80 | -0.22 |
| Solar resource | 0.81 | 0.53 | +0.28 |
| Computation theory | 0.87 | 0.84 | +0.03 |

Mean difference **+0.055**, 95 % bootstrap interval over the 9 lectures [-0.050, +0.161] (10,000 resamples, numpy seed 20260924). Wins 6, losses 3 (a tie counts as a win). Our test mean 0.853 vs their 0.80 on the same lectures; their 0.82 is a 20-lecture average and not a paired comparison.

Reinforcement: 0.67 vs their 0.75. Numerics: 0.89 vs their 0.81.

Climate policies (test half) has no video in their Kaggle zip and is outside every mean here.
