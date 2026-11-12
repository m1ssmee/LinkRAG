# MaViLS — free re-checks of the final result

From the cached sentence × page matrices of `mavils_final_round.md`. Nothing is tuned and no feature is recomputed. LLM-free, $0.

## Caveats first

- **This is a fourth look at the test half** (findings 14–16 were the first three). No weight, threshold or design choice is set from anything below; the tables are diagnostics of the system fixed in finding 16.
- **Leave-one-lecture-out is not the registered protocol.** The registered result is the fixed split of `mavils_split.json`. In the leave-one-out table a tune-half lecture's weights are chosen with test-half lectures among the others, and σ, the image score and the text score stay as tuned on the tune half, so tune-half lectures remain optimistic. It is exploratory and does not replace the fixed-split number.
- **Their numbers are published per-lecture figures at two decimals**, decoded by their DP on their features; they are not re-run. A difference below 0.005 is inside their rounding.
- Intervals: 10,000-resample bootstrap over lectures, numpy seed 20260924, as in `mavils_final_round.md`. An interval that contains 0 is marked *(crosses 0)*: no measurable difference.

Sanity: the final system reproduces 0.853 on the test half and 0.878 on the tune half (weights 0.25/0.5/0.25, gate {'min_words': 1, 'min_margin': 0.02}).

## Floors: position only, no content

Sentence i of n, or time t of T, mapped linearly onto the deck's pages.

| lectures | linear by index | linear by time | their audio-only | their all-features |
|---|---:|---:|---:|---:|
| all 20 | 0.206 | 0.214 | 0.532 | 0.822 |
| test, 9 with video | 0.233 | 0.209 | 0.473 | 0.798 |
| tune, 10 | 0.202 | 0.239 | 0.558 | 0.833 |

## The decoder, on our final fused matrix

| decoder | test | tune | test, paired against their all-features | all 19, paired against our DP |
|---|---:|---:|---:|---:|
| row-wise argmax, no DP | 0.789 | 0.790 | -0.009 [-0.142, +0.113] (crosses 0) | -0.077 [-0.112, -0.044] |
| their DP (λ_jump 0.1), their code | 0.849 | 0.874 | +0.052 [-0.053, +0.158] (crosses 0) | -0.004 [-0.015, +0.007] (crosses 0) |
| our DP (σ = 0.2) | 0.853 | 0.878 | +0.055 [-0.050, +0.161] (crosses 0) |  |

## Channels, through our DP

Left-out rows renormalise the final weights. Differences are paired against the final system.

| channels | test | tune | test − final | tune − final |
|---|---:|---:|---:|---:|
| image only | 0.730 | 0.685 | -0.123 [-0.244, -0.014] | -0.194 [-0.309, -0.084] |
| frame OCR only | 0.818 | 0.833 | -0.035 [-0.067, -0.007] | -0.045 [-0.077, -0.014] |
| speech-to-slide text only | 0.483 | 0.522 | -0.370 [-0.506, -0.250] | -0.356 [-0.468, -0.250] |
| without image | 0.825 | 0.854 | -0.028 [-0.048, -0.011] | -0.024 [-0.041, -0.007] |
| without frame OCR | 0.792 | 0.788 | -0.061 [-0.141, +0.027] (crosses 0) | -0.090 [-0.170, -0.020] |
| without speech-to-slide text | 0.842 | 0.848 | -0.011 [-0.034, +0.005] (crosses 0) | -0.030 [-0.056, -0.007] |
| final (all three) | 0.853 | 0.878 | | |

## Wins, ties and losses at their published precision (fixed split, test half)

| lecture | ours | theirs | difference |
|---|---:|---:|---:|
| ML for health | 0.9141 | 0.95 | -0.0359 |
| Climate & Cities | 0.9754 | 0.86 | +0.1154 |
| Deep learning | 0.9947 | 0.99 | +0.0047 |
| Numerics | 0.8878 | 0.81 | +0.0778 |
| Phonetics | 0.9864 | 0.65 | +0.3364 |
| Reinforcement | 0.6681 | 0.75 | -0.0819 |
| Short range | 0.5771 | 0.80 | -0.2229 |
| Solar resource | 0.8051 | 0.53 | +0.2751 |
| Computation theory | 0.8682 | 0.84 | +0.0282 |

Rounded to two decimals: **5 wins, 1 tied, 3 losses**. Paired mean +0.055 [-0.050, +0.161] (crosses 0); Wilcoxon signed-rank p = 0.43.

## Leave one lecture out, all 19 lectures with video (exploratory)

For each lecture the three-way weights, then the gate, are chosen on the other 18 as `final-round` chooses them on the tune half.

| lectures | ours | theirs | paired difference |
|---|---:|---:|---:|
| all 19 | 0.866 | 0.816 | +0.050 [+0.002, +0.103] |
| test half, 9 | 0.868 | 0.798 | +0.070 [-0.014, +0.160] (crosses 0) |
| tune half, 10 | 0.865 | 0.833 | +0.032 [-0.014, +0.095] (crosses 0) |

All 19, rounded to two decimals: 9 wins, 2 tied, 8 losses. Wilcoxon signed-rank p = 0.20. The three largest differences are Phonetics +0.30, Solar resource +0.29, Sensory systems +0.27; without them the mean difference is +0.006.

| lecture | half | ours, left out | theirs | difference |
|---|---|---:|---:|---:|
| ML for health | test | 0.92 | 0.95 | -0.03 |
| Decarbonization | tune | 0.99 | 0.93 | +0.06 |
| Climate & Cities | test | 0.95 | 0.86 | +0.09 |
| Cognitive robotics | tune | 0.63 | 0.66 | -0.03 |
| Computer Vision | tune | 1.00 | 1.00 | +0.00 |
| Productdesign | tune | 0.79 | 0.71 | +0.08 |
| Cryptocurrency | tune | 0.82 | 0.84 | -0.02 |
| Deep learning | test | 0.99 | 0.99 | +0.00 |
| Image processing | tune | 0.96 | 0.97 | -0.01 |
| Numerics | test | 0.88 | 0.81 | +0.07 |
| Phonetics | test | 0.95 | 0.65 | +0.30 |
| Physics | tune | 0.97 | 0.93 | +0.04 |
| Psychology | tune | 0.75 | 0.80 | -0.05 |
| Reinforcement | test | 0.67 | 0.75 | -0.08 |
| Sensory systems | tune | 0.80 | 0.53 | +0.27 |
| Short range | test | 0.71 | 0.80 | -0.09 |
| Solar resource | test | 0.82 | 0.53 | +0.29 |
| Team dynamics | tune | 0.95 | 0.96 | -0.01 |
| Computation theory | test | 0.92 | 0.84 | +0.08 |

## Where the errors are (tune half only, final system)

| lecture | F1 | errors / labelled | within 2 pages | on frames judged not a slide | such frames overall |
|---|---:|---:|---:|---:|---:|
| Decarbonization | 0.99 | 2 / 275 | 100% | 0% | 0% |
| Cognitive robotics | 0.66 | 254 / 735 | 98% | 82% | 52% |
| Computer Vision | 0.99 | 1 / 80 | 0% | 0% | 0% |
| Productdesign | 0.85 | 38 / 247 | 100% | 74% | 38% |
| Cryptocurrency | 0.88 | 65 / 500 | 77% | 8% | 3% |
| Image processing | 0.96 | 5 / 124 | 100% | 20% | 9% |
| Physics | 0.97 | 17 / 576 | 100% | 0% | 0% |
| Psychology | 0.70 | 300 / 893 | 73% | 47% | 25% |
| Sensory systems | 0.83 | 88 / 421 | 88% | 67% | 31% |
| Team dynamics | 0.95 | 3 / 55 | 33% | 0% | 0% |

## Provenance

`python scripts/adapters/mavils.py recheck` · inputs: the caches under `data/processed/mavils`, `mavils_tuned.json`, `mavils_split.json` · their DP executed from their source.
