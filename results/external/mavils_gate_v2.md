# Relatedness gate — second pass (MaViLS, 30-second windows)

Gate statistic: penalised DP objective per segment vs the mean of **30** shuffled slide orders, z = (true − null mean) / max(null std, 0.002); our DP at pilot parameters; page-OCR slide text; 30-second windows of their transcript sentences. Default config unchanged (`align.relatedness_z` = 2.0, `align.null_shuffles` = 5); the values chosen here are recorded in `results/external/mavils_tuned.json`.

## 1. Related pairs (20), 30-shuffle null

| lecture | text layer | windows × slides | column contrast (mean pairwise slide cosine) | objective | null mean ± std | z | F1 (their protocol) |
|---|---|---:|---:|---:|---:|---:|---:|
| ML for health | full | 145×77 | 0.477 | 0.5324 | 0.5310 ± 0.0031 | 0.5 | 0.56 |
| Decarbonization | full | 67×45 | 0.530 | 0.6622 | 0.5888 ± 0.0141 | 5.2 | 0.53 |
| Climate & Cities | full | 39×44 | 0.533 | 0.6745 | 0.5791 ± 0.0139 | 6.9 | 0.61 |
| Climate policies | full | 131×24 | 0.550 | 0.6211 | 0.5764 ± 0.0073 | 6.1 | 0.79 |
| Cognitive robotics | partial | 124×62 | 0.493 | 0.6294 | 0.5648 ± 0.0080 | 8.0 | 0.50 |
| Computer Vision | full | 35×18 | 0.572 | 0.6923 | 0.6540 ± 0.0070 | 5.5 | 0.64 |
| Productdesign | full | 95×27 | 0.477 | 0.5794 | 0.5342 ± 0.0077 | 5.9 | 0.80 |
| Cryptocurrency | full | 106×52 | 0.467 | 0.6047 | 0.5707 ± 0.0070 | 4.8 | 0.29 |
| Deep learning | full | 94×39 | 0.487 | 0.6445 | 0.5602 ± 0.0099 | 8.5 | 0.79 |
| Image processing | none | 47×21 | 0.614 | 0.6012 | 0.5993 ± 0.0021 | 0.9 | 0.04 |
| Numerics | full | 146×50 | 0.527 | 0.5998 | 0.5485 ± 0.0053 | 9.7 | 0.57 |
| Phonetics | full | 126×38 | 0.545 | 0.6266 | 0.6135 ± 0.0052 | 2.5 | 0.24 |
| Physics | full | 135×31 | 0.555 | 0.6403 | 0.5747 ± 0.0059 | 11.2 | 0.77 |
| Psychology | full | 127×64 | 0.460 | 0.6323 | 0.5447 ± 0.0059 | 14.8 | 0.56 |
| Reinforcement | full | 182×57 | 0.480 | 0.5873 | 0.5596 ± 0.0062 | 4.5 | 0.36 |
| Sensory systems | partial | 138×37 | 0.494 | 0.5304 | 0.5344 ± 0.0085 | -0.5 | 0.17 |
| Short range | full | 97×21 | 0.598 | 0.6085 | 0.5702 ± 0.0081 | 4.7 | 0.65 |
| Solar resource | full | 134×60 | 0.541 | 0.5846 | 0.5711 ± 0.0067 | 2.0 | 0.36 |
| Team dynamics | full | 77×24 | 0.667 | 0.6040 | 0.5921 ± 0.0054 | 2.2 | 0.78 |
| Computation theory | full | 110×12 | 0.577 | 0.5890 | 0.5824 ± 0.0041 | 1.6 | 0.29 |

## 2. Negative set: 380 cross pairs (each audio × every other deck)

z distribution of unrelated pairs: min -3.70 · median -0.09 · 95th percentile 1.26 · max 3.51.

| threshold z | false acceptance (380 unrelated) | false rejection (20 related) |
|---:|---:|---:|
| 0.00 | 35.5% (135/380) | 5% (1/20) |
| 1.00 | 8.4% (32/380) | 15% (3/20) |
| 1.50 | 3.7% (14/380) | 15% (3/20) |
| 2.00 | 2.9% (11/380) | 20% (4/20) |
| 1.27 ← | 4.7% (18/380) | 15% (3/20) |
| 3.00 | 1.3% (5/380) | 35% (7/20) |
| 4.00 | 0.0% (0/380) | 35% (7/20) |

**Chosen threshold: z = 1.27** — the smallest z with ≤ 5 % false acceptance on the 380 pairs; false rejection on the 20 related pairs at that threshold: **15%** (3/20: ML for health, Image processing, Sensory systems).

Highest-z unrelated pairs (what a false acceptance looks like):

- Solar resource audio × Climate policies deck: z = 3.51
- Physics audio × Climate policies deck: z = 3.30
- Team dynamics audio × Cognitive robotics deck: z = 3.13
- Deep learning audio × Cognitive robotics deck: z = 3.12
- Image processing audio × Deep learning deck: z = 3.04
- Solar resource audio × Cognitive robotics deck: z = 2.95
- Decarbonization audio × Cognitive robotics deck: z = 2.65
- Sensory systems audio × Cognitive robotics deck: z = 2.62

## 3. z vs alignment quality

Spearman ρ between a related pair's z and its their-protocol F1 (20 lectures): **0.56** (p = 0.010). Median F1 of pairs rejected at z = 1.27: 0.17; of accepted pairs: 0.57. **The rejected pairs are the low-F1 pairs**: the gate declines the alignments the DP gets wrong anyway.

| lecture | z | F1 | column contrast |
|---|---:|---:|---:|
| Sensory systems | -0.5 | 0.17 | 0.494 |
| ML for health | 0.5 | 0.56 | 0.477 |
| Image processing | 0.9 | 0.04 | 0.614 |
| Computation theory | 1.6 | 0.29 | 0.577 |
| Solar resource | 2.0 | 0.36 | 0.541 |
| Team dynamics | 2.2 | 0.78 | 0.667 |
| Phonetics | 2.5 | 0.24 | 0.545 |
| Reinforcement | 4.5 | 0.36 | 0.480 |
| Short range | 4.7 | 0.65 | 0.598 |
| Cryptocurrency | 4.8 | 0.29 | 0.467 |
| Decarbonization | 5.2 | 0.53 | 0.530 |
| Computer Vision | 5.5 | 0.64 | 0.572 |
| Productdesign | 5.9 | 0.80 | 0.477 |
| Climate policies | 6.1 | 0.79 | 0.550 |
| Climate & Cities | 6.9 | 0.61 | 0.533 |
| Cognitive robotics | 8.0 | 0.50 | 0.493 |
| Deep learning | 8.5 | 0.79 | 0.487 |
| Numerics | 9.7 | 0.57 | 0.527 |
| Physics | 11.2 | 0.77 | 0.555 |
| Psychology | 14.8 | 0.56 | 0.460 |

## 4. z vs deck column contrast

Spearman ρ between z and mean pairwise slide cosine: **-0.27** (p = 0.257; negative = slides that look alike give low z). Median mean-pairwise-cosine of rejected decks: 0.494; of accepted decks: 0.533. Rejected and accepted decks do not separate on column contrast.


## 5. ML for health — inspected before being counted as a false rejection

`reports/mavils_inspect_ml_for_health.md`, heatmap `reports/mavils_inspect_ML_for_health_MIT.png` (30-second windows, page OCR). The pair is **aligned correctly**: our DP reaches their-protocol F1 **0.56** against their audio-only 0.57, and the first ten mismatches are all one slide *early* (ours p2 where the label is p1, p3 for p2, p4 for p3) — the speaker-runs-ahead pattern seen on pilot01, not a wrong deck. The labels are monotone (77 forward transitions, 2 backward). Yet z = 0.5: the true objective (0.469) barely beats the 30-shuffle null (0.463 ± 0.011). Three explanations were tested across all 20 lectures and none holds: deck column contrast (mean pairwise slide cosine 0.477, near the median; ρ(z, contrast) = −0.27, p = 0.26 — see §4), off-slide fraction (41 % of sentences are −1 here, but Team dynamics is 91 % off-slide and passes; ρ = −0.33, p = 0.15), and near-empty slides (ρ = 0.07). What is specific to this pair is its size — 145 windows over 77 slides, under two windows per slide — so the true path has to advance almost every window and the penalty structure buys little over a shuffled order. **Counted as a false rejection; cause not confirmed.** The other two rejects (Image processing, F1 0.04; Sensory systems, F1 0.17) are pairs the aligner cannot align either.

## 6. Sentence-level numbers — kept once, as a labelled deviation

The gate runs at 30-second windows in the pipeline. The earlier table at their sentence granularity (5-shuffle null, z = 2.0, `results/external/mavils_gate.md`) is reproduced here for the record and is not the operating point:

| lecture | text layer | z (sentence, 5 shuffles) | verdict at 2.0 | z (30 s, 30 shuffles) |
|---|---|---:|---|---:|
| ML for health | full | 0.0 | rejected | 0.5 |
| Decarbonization | full | 19.4 | accepted | 5.2 |
| Climate & Cities | full | 15.5 | accepted | 6.9 |
| Climate policies | full | 2.2 | accepted | 6.1 |
| Cognitive robotics | partial | 4.2 | accepted | 8.0 |
| Computer Vision | full | 3.7 | accepted | 5.5 |
| Productdesign | full | 1.3 | rejected | 5.9 |
| Cryptocurrency | full | 4.7 | accepted | 4.8 |
| Deep learning | full | 7.9 | accepted | 8.5 |
| Image processing | none | 1.6 | rejected | 0.9 |
| Numerics | full | 7.7 | accepted | 9.7 |
| Phonetics | full | 1.8 | rejected | 2.5 |
| Physics | full | 4.8 | accepted | 11.2 |
| Psychology | full | 5.5 | accepted | 14.8 |
| Reinforcement | full | 2.6 | accepted | 4.5 |
| Sensory systems | partial | -0.0 | rejected | -0.5 |
| Short range | full | 2.7 | accepted | 4.7 |
| Solar resource | full | 1.6 | rejected | 2.0 |
| Team dynamics | full | 1.3 | rejected | 2.2 |
| Computation theory | full | 1.3 | rejected | 1.6 |

Sentence level, 5 shuffles, z = 2.0: 8/20 false rejections (40 %). Operating point (30 s, 30 shuffles, z = 1.27): 3/20 (15 %) at 4.7 % false acceptance.

