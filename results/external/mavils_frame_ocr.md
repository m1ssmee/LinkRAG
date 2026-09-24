# MaViLS — frame OCR and three-feature fusion vs their all-features

Their protocol: sentence granularity, page OCR on the slide side, their sklearn F1. Cells: **their F1 (precision-on-answered / coverage)**. Decoder: our DP at the tuned σ = 0.2, λ=0.05 β=0.15 B=2. LLM-free, $0.

Columns: *ours* = our text-only hybrid. *visual+theirs* = the previous best (`mavils_visual.md`: swiftformer image score, w = 0.5). *frame_ocr* = each sentence takes the frame→page **text** row of the frame on screen at its timestamp (`mavils_frame_ocr_sanity.md`), min-max scaled. *visual+frame_ocr* = w·frame_ocr + (1−w)·visual. *visual+frame_ocr+theirs* = weighted sum of the three, each min-max scaled (`linkrag.link.align.fuse_many`): all three of their feature types (image, frame OCR, speech-to-slide text). Their published columns: audio-only (Table 1) and all features (Table 2, λ = 0.1).

**Caveat carried from the frames:** frames are one representative per dHash segment, kept at 320 px; the text on them is read after a 3× upscale. MaViLS reads the full-resolution frame at each sentence timestamp. Re-extracting sentence-time or full-resolution frames was not possible: the videos are no longer available.

## Tune half

| frame-OCR text score | frame_ocr-only tune |
|---|---:|
| tfidf | 0.700 (0.687 / 1.00) |
| bm25 | 0.702 (0.689 / 1.00) ← |

| w (share of frame_ocr, bm25) | visual+frame_ocr tune |
|---:|---:|
| 0.0 | 0.691 (0.681 / 1.00) |
| 0.25 | 0.777 (0.768 / 1.00) ← |
| 0.5 | 0.771 (0.761 / 1.00) |
| 0.75 | 0.740 (0.729 / 1.00) |
| 1.0 | 0.702 (0.689 / 1.00) |

| weights (visual, frame_ocr, theirs) | visual+frame_ocr+theirs tune |
|---|---:|
| 0.50, 0.25, 0.25 | 0.793 (0.786 / 1.00) ← |
| 0.25, 0.25, 0.50 | 0.792 (0.782 / 1.00) |
| 0.25, 0.50, 0.25 | 0.790 (0.780 / 1.00) |
| 0.75, 0.25, 0.00 | 0.777 (0.768 / 1.00) |
| 0.50, 0.50, 0.00 | 0.771 (0.761 / 1.00) |
| 0.50, 0.00, 0.50 | 0.758 (0.754 / 1.00) |
| 0.00, 0.50, 0.50 | 0.753 (0.743 / 1.00) |
| 0.00, 0.25, 0.75 | 0.742 (0.731 / 1.00) |
| 0.75, 0.00, 0.25 | 0.741 (0.735 / 1.00) |
| 0.25, 0.75, 0.00 | 0.740 (0.729 / 1.00) |
| 0.00, 0.75, 0.25 | 0.732 (0.721 / 1.00) |
| 0.00, 1.00, 0.00 | 0.702 (0.689 / 1.00) |
| 1.00, 0.00, 0.00 | 0.691 (0.681 / 1.00) |
| 0.25, 0.00, 0.75 | 0.685 (0.678 / 1.00) |
| 0.00, 0.00, 1.00 | 0.522 (0.515 / 1.00) |

Chosen on tune: text score **bm25**, visual+frame_ocr w = **0.25**, three-way weights **0.50 / 0.25 / 0.25** (visual / frame_ocr / theirs). Best new column on tune: **visual+frame_ocr+theirs**.

## Test half — reported once

| lecture | ours | visual+theirs | frame_ocr | visual+frame_ocr | visual+frame_ocr+theirs | their all-features (T2) |
|---|---:|---:|---:|---:|---:|---:|
| ML for health | 0.34 (0.34 / 1.00) | 0.93 (0.93 / 1.00) | 0.06 (0.06 / 1.00) | 0.94 (0.94 / 1.00) | 0.93 (0.93 / 1.00) | 0.95 |
| Climate & Cities | 0.60 (0.59 / 1.00) | 0.83 (0.83 / 1.00) | 0.68 (0.67 / 1.00) | 0.79 (0.78 / 1.00) | 0.84 (0.84 / 1.00) | 0.86 |
| Deep learning | 0.70 (0.70 / 1.00) | 0.98 (0.98 / 1.00) | 0.99 (0.99 / 1.00) | 0.99 (0.99 / 1.00) | 0.99 (0.99 / 1.00) | 0.99 |
| Numerics | 0.50 (0.50 / 1.00) | 0.69 (0.69 / 1.00) | 0.29 (0.29 / 1.00) | 0.71 (0.71 / 1.00) | 0.71 (0.71 / 1.00) | 0.81 |
| Phonetics | 0.36 (0.36 / 1.00) | 0.84 (0.84 / 1.00) | 0.13 (0.13 / 1.00) | 0.87 (0.87 / 1.00) | 0.89 (0.89 / 1.00) | 0.65 |
| Reinforcement | 0.20 (0.20 / 1.00) | 0.45 (0.45 / 1.00) | 0.23 (0.22 / 1.00) | 0.27 (0.27 / 1.00) | 0.52 (0.52 / 1.00) | 0.75 |
| Short range | 0.49 (0.49 / 1.00) | 0.77 (0.77 / 1.00) | 0.77 (0.77 / 1.00) | 0.78 (0.78 / 1.00) | 0.84 (0.84 / 1.00) | 0.80 |
| Solar resource | 0.35 (0.34 / 1.00) | 0.55 (0.54 / 1.00) | 0.45 (0.44 / 1.00) | 0.58 (0.57 / 1.00) | 0.60 (0.59 / 1.00) | 0.53 |
| Computation theory | 0.28 (0.28 / 1.00) | 0.85 (0.84 / 1.00) | 0.86 (0.86 / 1.00) | 0.96 (0.96 / 1.00) | 0.96 (0.96 / 1.00) | 0.84 |
| **mean** | **0.426 (0.423 / 1.00)** | **0.765 (0.763 / 1.00)** | **0.495 (0.492 / 1.00)** | **0.765 (0.763 / 1.00)** | **0.810 (0.808 / 1.00)** | 0.80 |
| Climate policies (no video) | 0.78 (0.75 / 1.00) | — | — | — | — | 0.93 |

Best new column on tune: **visual+frame_ocr+theirs**. On the 9 test lectures with video it beats their all-features F1 on 5 and loses 4; against the previous best (visual+theirs) it wins 8 and loses 1 (a tie counts as a win). Their all-features mean on these lectures is 0.80; 0.82 is their 20-lecture average.

All 19 lectures with video (contains the tune half, so the tuned columns are optimistic): ours 0.456 (0.452 / 1.00), visual+theirs 0.762 (0.758 / 1.00), frame_ocr 0.604 (0.596 / 1.00), visual+frame_ocr 0.771 (0.765 / 1.00), visual+frame_ocr+theirs 0.801 (0.797 / 1.00).

