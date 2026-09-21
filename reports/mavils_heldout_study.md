# MaViLS — held-out abstention / flatness study

Split `results/external/mavils_split.json` (seed 20260922): tune = Decarbonization, Cognitive robotics, Computer Vision, Productdesign, Cryptocurrency, Image processing, Physics, Psychology, Sensory systems, Team dynamics; test = ML for health, Climate & Cities, Climate policies, Deep learning, Numerics, Phonetics, Reinforcement, Short range, Solar resource, Computation theory. Segments: sentence · slide text: ocr · metric: their F1 (protocol note in `mavils_alignment.md`) plus precision-on-answered and coverage.

**Parameters were set on the tune half only** and are recorded in `results/external/mavils_tuned.json`, not in `configs/default.yaml` (no parameter changes on the full set / pilot01).

## Tune half — flatness grid (DP, no abstention)

| flatness_scaling | their F1 | prec@answered | coverage |
|---:|---:|---:|---:|
| 0.0 | 0.471 ← | 0.465 | 1.00 |
| 0.5 | 0.471 | 0.465 | 1.00 |
| 1.0 | 0.471 | 0.465 | 1.00 |

## Tune half — abstention grid (flatness = 0.0)

| min_segment_sim | their F1 | prec@answered | coverage |
|---:|---:|---:|---:|
| off | 0.471 | 0.465 | 1.00 |
| 0.455 | 0.465 | 0.471 | 0.97 |
| 0.4816 | 0.453 | 0.474 | 0.94 |
| 0.5055 | 0.432 | 0.486 ← | 0.87 |
| 0.5239 | 0.404 | 0.503 | 0.79 |

Chosen on the tune half: flatness_scaling = 0.0, min_segment_sim = 0.5055.

## Tune half — their F1 per lecture (prec@answered / coverage for abstaining variants)

| lecture | text layer | naive | dp | dp+abstain | dp+abstain+flat | their audio |
|---|---|---:|---:|---:|---:|---:|
| Decarbonization | full | 0.39 | 0.44 | 0.42 (0.45 / 0.93) | 0.42 (0.45 / 0.93) | 0.52 |
| Cognitive robotics | partial | 0.19 | 0.43 | 0.43 (0.46 / 0.86) | 0.43 (0.46 / 0.86) | 0.42 |
| Computer Vision | full | 0.43 | 0.72 | 0.72 (0.74 / 0.95) | 0.72 (0.74 / 0.95) | 0.65 |
| Productdesign | full | 0.41 | 0.58 | 0.48 (0.62 / 0.76) | 0.48 (0.62 / 0.76) | 0.72 |
| Cryptocurrency | full | 0.18 | 0.30 | 0.29 (0.32 / 0.91) | 0.29 (0.32 / 0.91) | 0.21 |
| Image processing | none | 0.23 | 0.19 | 0.19 (0.19 / 0.96) | 0.19 (0.19 / 0.96) | 0.53 |
| Physics | full | 0.37 | 0.69 | 0.59 (0.71 / 0.83) | 0.59 (0.71 / 0.83) | 0.60 |
| Psychology | full | 0.24 | 0.48 | 0.42 (0.46 / 0.84) | 0.42 (0.46 / 0.84) | 0.58 |
| Sensory systems | partial | 0.18 | 0.21 | 0.17 (0.24 / 0.71) | 0.17 (0.24 / 0.71) | 0.46 |
| Team dynamics | full | 0.49 | 0.67 | 0.62 (0.67 / 0.93) | 0.62 (0.67 / 0.93) | 0.89 |
| **mean** | | **0.312** | **0.471** | **0.432** | **0.432** | 0.56 |

## Test half — their F1 per lecture (prec@answered / coverage for abstaining variants)

| lecture | text layer | naive | dp | dp+abstain | dp+abstain+flat | their audio |
|---|---|---:|---:|---:|---:|---:|
| ML for health | full | 0.25 | 0.29 | 0.28 (0.36 / 0.79) | 0.28 (0.36 / 0.79) | 0.57 |
| Climate & Cities | full | 0.48 | 0.60 | 0.57 (0.59 / 0.96) | 0.57 (0.59 / 0.96) | 0.49 |
| Climate policies | full | 0.42 | 0.76 | 0.66 (0.75 / 0.83) | 0.66 (0.75 / 0.83) | 0.79 |
| Deep learning | full | 0.44 | 0.70 | 0.68 (0.70 / 0.97) | 0.68 (0.70 / 0.97) | 0.79 |
| Numerics | full | 0.21 | 0.50 | 0.43 (0.50 / 0.86) | 0.43 (0.50 / 0.86) | 0.46 |
| Phonetics | full | 0.28 | 0.35 | 0.33 (0.39 / 0.86) | 0.33 (0.39 / 0.86) | 0.05 |
| Reinforcement | full | 0.12 | 0.24 | 0.20 (0.24 / 0.78) | 0.20 (0.24 / 0.78) | 0.31 |
| Short range | full | 0.27 | 0.45 | 0.38 (0.49 / 0.78) | 0.38 (0.49 / 0.78) | 0.57 |
| Solar resource | full | 0.14 | 0.32 | 0.26 (0.33 / 0.79) | 0.26 (0.33 / 0.79) | 0.46 |
| Computation theory | full | 0.22 | 0.30 | 0.26 (0.31 / 0.84) | 0.26 (0.31 / 0.84) | 0.56 |
| **mean** | | **0.281** | **0.452** | **0.406** | **0.406** | 0.51 |

