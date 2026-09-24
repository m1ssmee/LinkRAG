# MaViLS — visual channel vs their all-features

Their protocol: sentence granularity, page OCR on the slide side, their sklearn F1. Cells: **their F1 (precision-on-answered / coverage)**. Decoder: our DP at the tuned σ = 0.2, λ=0.05 β=0.15 B=2. LLM-free, $0.

Columns: *ours* = bge-m3 + BM25 + IDF hybrid; *theirs* = distiluse cosine (their audio feature); *fused* = 0.5·theirs + 0.5·ours (tuned 2026-09-22). *visual* = each sentence takes the frame→page row of the representative frame on screen at its timestamp (`mavils_frames.md`), min-max scaled. *visual+text* / *visual+theirs* = w·visual + (1−w)·text, each min-max scaled. Their published columns: audio-only (Table 1) and all features, i.e. speech + frame OCR + SwiftFormer image features, merged, λ = 0.1 (Table 2).

Split `results/external/mavils_split.json`: tune = Decarbonization, Cognitive robotics, Computer Vision, Productdesign, Cryptocurrency, Image processing, Physics, Psychology, Sensory systems, Team dynamics; test = ML for health, Climate & Cities, Climate policies, Deep learning, Numerics, Phonetics, Reinforcement, Short range, Solar resource, Computation theory. Every choice below is made on the tune half; the test half is run once. Chosen values are recorded in `results/external/mavils_tuned.json`.

## Tune half

| frame→page score | visual-only tune |
|---|---:|
| dhash | 0.315 (0.300 / 1.00) |
| swiftformer | 0.691 (0.681 / 1.00) ← |

| w (share of visual, swiftformer) | visual+text | visual+theirs |
|---:|---:|---:|
| 0.0 | 0.468 (0.462 / 1.00) | 0.522 (0.515 / 1.00) |
| 0.25 | 0.575 (0.569 / 1.00) | 0.685 (0.678 / 1.00) |
| 0.5 | 0.685 (0.679 / 1.00) | 0.758 (0.754 / 1.00) ← |
| 0.75 | 0.725 (0.717 / 1.00) ← | 0.741 (0.735 / 1.00) |
| 1.0 | 0.691 (0.681 / 1.00) | 0.691 (0.681 / 1.00) |

Chosen on tune: frame→page score **swiftformer**, visual+text w = **0.75**, visual+theirs w = **0.5**. No confidence floor was needed or used.

## Test half — reported once

| lecture | text layer | jumpiness | no-slide ratio | ours | theirs | fused | visual | visual+text | visual+theirs | their audio (T1) | their all-features (T2) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ML for health | full | 1.64 | 0.68 | 0.34 (0.34 / 1.00) | 0.64 (0.64 / 1.00) | 0.62 (0.62 / 1.00) | 0.95 (0.95 / 1.00) | 0.95 (0.95 / 1.00) | 0.93 (0.93 / 1.00) | 0.57 | 0.95 |
| Climate & Cities | full | 1.52 | 0.34 | 0.60 (0.59 / 1.00) | 0.57 (0.57 / 1.00) | 0.66 (0.66 / 1.00) | 0.79 (0.78 / 1.00) | 0.81 (0.80 / 1.00) | 0.83 (0.83 / 1.00) | 0.49 | 0.86 |
| Deep learning | full | 1.34 | 0.12 | 0.70 (0.70 / 1.00) | 0.71 (0.71 / 1.00) | 0.69 (0.69 / 1.00) | 0.99 (0.99 / 1.00) | 0.98 (0.98 / 1.00) | 0.98 (0.98 / 1.00) | 0.79 | 0.99 |
| Numerics | full | 1.49 | 0.03 | 0.50 (0.50 / 1.00) | 0.48 (0.48 / 1.00) | 0.55 (0.55 / 1.00) | 0.66 (0.66 / 1.00) | 0.70 (0.70 / 1.00) | 0.69 (0.69 / 1.00) | 0.46 | 0.81 |
| Phonetics | full | 4.78 | 3.01 | 0.36 (0.36 / 1.00) | 0.16 (0.16 / 1.00) | 0.33 (0.33 / 1.00) | 0.88 (0.88 / 1.00) | 0.94 (0.94 / 1.00) | 0.84 (0.84 / 1.00) | 0.05 | 0.65 |
| Reinforcement | full | 1.45 | 0.06 | 0.20 (0.20 / 1.00) | 0.33 (0.32 / 1.00) | 0.30 (0.29 / 1.00) | 0.13 (0.12 / 1.00) | 0.37 (0.36 / 1.00) | 0.45 (0.45 / 1.00) | 0.31 | 0.75 |
| Short range | full | 1.27 | 0.06 | 0.49 (0.49 / 1.00) | 0.57 (0.57 / 1.00) | 0.54 (0.54 / 1.00) | 0.76 (0.76 / 1.00) | 0.79 (0.79 / 1.00) | 0.77 (0.77 / 1.00) | 0.57 | 0.80 |
| Solar resource | full | 1.80 | 0.63 | 0.35 (0.34 / 1.00) | 0.43 (0.41 / 1.00) | 0.39 (0.38 / 1.00) | 0.42 (0.41 / 1.00) | 0.51 (0.51 / 1.00) | 0.55 (0.54 / 1.00) | 0.46 | 0.53 |
| Computation theory | full | 1.18 | 0.08 | 0.28 (0.28 / 1.00) | 0.51 (0.51 / 1.00) | 0.35 (0.35 / 1.00) | 0.81 (0.78 / 1.00) | 0.83 (0.81 / 1.00) | 0.85 (0.84 / 1.00) | 0.56 | 0.84 |
| **mean** | | | | **0.426 (0.423 / 1.00)** | **0.489 (0.485 / 1.00)** | **0.493 (0.490 / 1.00)** | **0.710 (0.704 / 1.00)** | **0.764 (0.760 / 1.00)** | **0.765 (0.763 / 1.00)** | 0.47 | 0.80 |
| Climate policies (no video) | full | 3.11 | 2.07 | 0.78 (0.75 / 1.00) | 0.76 (0.70 / 1.00) | 0.76 (0.73 / 1.00) | — | — | — | 0.79 | 0.93 |

**No video for Climate policies**: the MaViLS Kaggle zip has none for it, so the test mean is over 9 lectures (paired: every column on the same lectures) and the no-video row is outside it.

Best visual column on tune: **visual+theirs**. On the test half it beats their all-features F1 on 3 of 9 lectures (loses 6) and our text-only *ours* on 9 (loses 0); a tie counts as a win.

All 19 lectures with video (contains the tune half, so the tuned columns are optimistic): ours 0.456 (0.452 / 1.00), theirs 0.509 (0.503 / 1.00), fused 0.525 (0.520 / 1.00), visual 0.700 (0.691 / 1.00), visual+text 0.743 (0.738 / 1.00), visual+theirs 0.762 (0.758 / 1.00).

