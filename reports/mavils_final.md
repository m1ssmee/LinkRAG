# MaViLS — final round

Their protocol throughout (sentence granularity, page OCR, their sklearn F1; see `mavils_alignment.md`). Every cell shows **their F1 (precision-on-answered / coverage)**. Split `results/external/mavils_split.json` (seed 20260922); tune = Decarbonization, Cognitive robotics, Computer Vision, Productdesign, Cryptocurrency, Image processing, Physics, Psychology, Sensory systems, Team dynamics; test = ML for health, Climate & Cities, Climate policies, Deep learning, Numerics, Phonetics, Reinforcement, Short range, Solar resource, Computation theory. Tune-half changes only; the test half is reported once, at the end.

## 1. Skip-penalty (σ) sweep — tune half

λ=0.05 β=0.15 B=2 μ=0.02 fixed; σ varied. Criterion: mean their-F1 on the tune half. Pilot value σ = 0.02.

| σ | tune |
|---:|---:|
| 0.0 | 0.468 (0.462 / 1.00) |
| 0.01 | 0.471 (0.465 / 1.00) |
| 0.02 | 0.471 (0.465 / 1.00) |
| 0.05 | 0.481 (0.475 / 1.00) |
| 0.1 | 0.483 (0.476 / 1.00) |
| 0.2 | 0.484 (0.477 / 1.00) ← |
| 0.3 | 0.484 (0.477 / 1.00) |

Chosen on tune: σ = 0.2 (pilot σ = 0.02, tune 0.471 (0.465 / 1.00)).

## 2. Decomposition — similarity matrix × decoder, all 20 lectures

Their cells run **their public code**: `calculate_dp_with_jumps` (helpers/utils.py, verbatim, λ_jump = 0.1) over distiluse-base-multilingual-cased cosine between each sentence and the page OCR (matching_algorithm.py). Deviation: tesseract `eng` only (their `eng+ell+equ+deu` traineddata are not installed here); rendering 150 dpi vs their 2.0× (144 dpi). Our decoder uses the pilot σ = 0.02 here (not the tuned value), so this table has no tuned quantity in it.

| similarity \ decoder | their DP | our DP |
|---|---:|---:|
| theirs (distiluse · page OCR) | 0.513 (0.505 / 1.00) | 0.520 (0.513 / 1.00) |
| ours (bge-m3 + BM25 + IDF · page OCR) | 0.425 (0.419 / 1.00) | 0.461 (0.454 / 1.00) |

Replication check: their similarity + their decoder = **0.513** against the paper's 0.53 audio-only average (difference = OCR language pack + rendering + transcript-file drift, not algorithm). Swapping only the matrix moves the mean by +0.074, swapping only the decoder by -0.022: **the gap lives in the similarity matrix** (their distiluse-on-OCR matrix is the better input; our DP is at least as good a decoder).

## 3. Build-deck handling — tune half

Detection: consecutive pages whose token multiset is a superset of the previous page's form a build group (`build_groups`, page OCR text, ≥ 5 tokens). Alignment runs the DP over groups (column = max over the group's builds); within a group each sentence is scored against each build's *incremental* text (IDF overlap over the delta) and decoded monotonically. Config `align.build_groups` (off by default; evaluated here on the tune half at the tuned σ).

| variant (σ = 0.2) | tune |
|---|---:|
| dp, build_groups off | 0.484 (0.477 / 1.00) |
| dp, build_groups on | 0.478 (0.471 / 1.00) |

Decision on tune: build_groups = **off**.

| tune lecture | build decks | dp off | dp on |
|---|---|---:|---:|
| Decarbonization | 7 groups / 22 pages | 0.55 (0.55 / 1.00) | 0.53 (0.53 / 1.00) |
| Cognitive robotics | — | 0.43 (0.42 / 1.00) | 0.43 (0.42 / 1.00) |
| Computer Vision | — | 0.69 (0.69 / 1.00) | 0.69 (0.69 / 1.00) |
| Productdesign | — | 0.62 (0.60 / 1.00) | 0.62 (0.60 / 1.00) |
| Cryptocurrency | 4 groups / 9 pages | 0.31 (0.31 / 1.00) | 0.30 (0.30 / 1.00) |
| Image processing | — | 0.19 (0.19 / 1.00) | 0.19 (0.19 / 1.00) |
| Physics | — | 0.69 (0.69 / 1.00) | 0.69 (0.69 / 1.00) |
| Psychology | 4 groups / 8 pages | 0.48 (0.44 / 1.00) | 0.45 (0.41 / 1.00) |
| Sensory systems | — | 0.21 (0.20 / 1.00) | 0.21 (0.20 / 1.00) |
| Team dynamics | — | 0.67 (0.67 / 1.00) | 0.67 (0.67 / 1.00) |

## 4. Test half — reported once

| lecture | text layer | build decks | naive | dp (pilot σ) | dp (σ = 0.2) | dp + builds (σ = 0.2) | their audio |
|---|---|---|---:|---:|---:|---:|---:|
| ML for health | full | — | 0.25 (0.25 / 1.00) | 0.29 (0.29 / 1.00) | 0.34 (0.34 / 1.00) | 0.34 (0.34 / 1.00) | 0.57 |
| Climate & Cities | full | 10 groups / 23 pages | 0.48 (0.47 / 1.00) | 0.60 (0.59 / 1.00) | 0.60 (0.59 / 1.00) | 0.61 (0.61 / 1.00) | 0.49 |
| Climate policies | full | — | 0.42 (0.36 / 1.00) | 0.76 (0.71 / 1.00) | 0.78 (0.75 / 1.00) | 0.78 (0.75 / 1.00) | 0.79 |
| Deep learning | full | — | 0.44 (0.44 / 1.00) | 0.70 (0.70 / 1.00) | 0.70 (0.70 / 1.00) | 0.70 (0.70 / 1.00) | 0.79 |
| Numerics | full | 1 groups / 2 pages | 0.21 (0.21 / 1.00) | 0.50 (0.50 / 1.00) | 0.50 (0.50 / 1.00) | 0.50 (0.50 / 1.00) | 0.46 |
| Phonetics | full | 2 groups / 5 pages | 0.28 (0.28 / 1.00) | 0.35 (0.35 / 1.00) | 0.36 (0.36 / 1.00) | 0.38 (0.38 / 1.00) | 0.05 |
| Reinforcement | full | — | 0.12 (0.11 / 1.00) | 0.24 (0.22 / 1.00) | 0.20 (0.20 / 1.00) | 0.20 (0.20 / 1.00) | 0.31 |
| Short range | full | — | 0.27 (0.27 / 1.00) | 0.45 (0.45 / 1.00) | 0.49 (0.49 / 1.00) | 0.49 (0.49 / 1.00) | 0.57 |
| Solar resource | full | 3 groups / 6 pages | 0.14 (0.14 / 1.00) | 0.32 (0.32 / 1.00) | 0.35 (0.34 / 1.00) | 0.34 (0.33 / 1.00) | 0.46 |
| Computation theory | full | — | 0.22 (0.20 / 1.00) | 0.30 (0.30 / 1.00) | 0.28 (0.28 / 1.00) | 0.28 (0.28 / 1.00) | 0.56 |
| **mean** | | | **0.281 (0.272 / 1.00)** | **0.452 (0.443 / 1.00)** | **0.461 (0.455 / 1.00)** | **0.462 (0.457 / 1.00)** | 0.51 |

