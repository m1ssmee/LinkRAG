# MaViLS — fused similarity study

Their protocol (sentence granularity, page OCR, their sklearn F1). Cells: **their F1 (precision-on-answered / coverage)**. Decoder: our DP, σ = 0.2 (tuned earlier), λ=0.05 β=0.15 B=2 μ=0.02. Matrices: *theirs* = distiluse-base-multilingual-cased cosine vs page OCR; *ours* = bge-m3 + BM25 + IDF hybrid vs page OCR. Each matrix is min-max scaled over the lecture before fusion (`linkrag.link.align.fuse_similarity`). Split `results/external/mavils_split.json`; tune-half choices only; test reported once.

## Tune half — weighted fusion, weight on *theirs*

| weight | tune |
|---:|---:|
| 0.0 | 0.468 (0.462 / 1.00) |
| 0.25 | 0.501 (0.494 / 1.00) |
| 0.5 | 0.554 (0.547 / 1.00) ← |
| 0.75 | 0.543 (0.537 / 1.00) |
| 1.0 | 0.522 (0.515 / 1.00) |

## Tune half — all similarities

| align.similarity | tune |
|---|---:|
| ours | 0.484 (0.477 / 1.00) |
| theirs | 0.527 (0.519 / 1.00) |
| fused_max | 0.501 (0.494 / 1.00) |
| fused_weighted (w=0.5) | 0.554 (0.547 / 1.00) ← |

Chosen on tune: similarity = **fused_weighted**, fusion_weight = 0.5.

## Test half — reported once

| lecture | text layer | ours | theirs | fused_max | fused_weighted (w=0.5) | their audio (paper) |
|---|---|---:|---:|---:|---:|---:|
| ML for health | full | 0.34 (0.34 / 1.00) | 0.64 (0.64 / 1.00) | 0.39 (0.39 / 1.00) | 0.62 (0.62 / 1.00) | 0.57 |
| Climate & Cities | full | 0.60 (0.59 / 1.00) | 0.57 (0.57 / 1.00) | 0.67 (0.66 / 1.00) | 0.66 (0.66 / 1.00) | 0.49 |
| Climate policies | full | 0.78 (0.75 / 1.00) | 0.76 (0.70 / 1.00) | 0.74 (0.69 / 1.00) | 0.76 (0.73 / 1.00) | 0.79 |
| Deep learning | full | 0.70 (0.70 / 1.00) | 0.71 (0.71 / 1.00) | 0.67 (0.67 / 1.00) | 0.69 (0.69 / 1.00) | 0.79 |
| Numerics | full | 0.50 (0.50 / 1.00) | 0.48 (0.48 / 1.00) | 0.54 (0.54 / 1.00) | 0.55 (0.55 / 1.00) | 0.46 |
| Phonetics | full | 0.36 (0.36 / 1.00) | 0.16 (0.16 / 1.00) | 0.29 (0.29 / 1.00) | 0.33 (0.33 / 1.00) | 0.05 |
| Reinforcement | full | 0.20 (0.20 / 1.00) | 0.33 (0.32 / 1.00) | 0.25 (0.24 / 1.00) | 0.30 (0.29 / 1.00) | 0.31 |
| Short range | full | 0.49 (0.49 / 1.00) | 0.57 (0.57 / 1.00) | 0.49 (0.49 / 1.00) | 0.54 (0.54 / 1.00) | 0.57 |
| Solar resource | full | 0.35 (0.34 / 1.00) | 0.43 (0.41 / 1.00) | 0.37 (0.36 / 1.00) | 0.39 (0.38 / 1.00) | 0.46 |
| Computation theory | full | 0.28 (0.28 / 1.00) | 0.51 (0.51 / 1.00) | 0.29 (0.29 / 1.00) | 0.35 (0.35 / 1.00) | 0.56 |
| **mean** | | **0.461 (0.455 / 1.00)** | **0.515 (0.507 / 1.00)** | **0.471 (0.463 / 1.00)** | **0.520 (0.515 / 1.00)** | 0.51 |

All 20 lectures (for the comparison against the paper's 0.53 and our 0.46; contains the tune half, so the fused columns are optimistic): ours 0.472 (0.466 / 1.00), theirs 0.521 (0.513 / 1.00), fused_max 0.486 (0.479 / 1.00), fused_weighted 0.537 (0.531 / 1.00).

