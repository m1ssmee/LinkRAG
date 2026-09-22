# Relatedness gate on MaViLS — all 20 pairs, both granularities

Gate: penalised DP objective per segment vs 5 shuffled slide orders; z = (true − null mean) / max(null std, floor); threshold z = 2.0; `align.null_std_floor` = 0.002. Matrices: page OCR on the slide side, our hybrid similarity, our DP at pilot parameters (λ=0.05 σ=0.02 β=0.15 B=2 μ=0.02). All 20 pairs are related by construction, so every rejection is a false rejection. Text layer: `full` = every page has a PDF text layer, `partial` / `none` = pages that had to be OCR'd from the rendered image.

| lecture | text layer | sentences × slides | z (sentence) | verdict | z (30 s) | verdict | segments (30 s) |
|---|---|---:|---:|---|---:|---|---:|
| ML for health | full | 822×77 | 0.0 | **rejected** | 0.1 | **rejected** | 145 |
| Decarbonization | full | 312×45 | 19.4 | accepted | 9.3 | accepted | 67 |
| Climate & Cities | full | 192×44 | 15.5 | accepted | 6.4 | accepted | 39 |
| Climate policies | full | 841×24 | 2.2 | accepted | 8.7 | accepted | 131 |
| Cognitive robotics | partial | 735×62 | 4.2 | accepted | 7.7 | accepted | 124 |
| Computer Vision | full | 80×18 | 3.7 | accepted | 8.2 | accepted | 35 |
| Productdesign | full | 715×27 | 1.3 | **rejected** | 7.9 | accepted | 95 |
| Cryptocurrency | full | 684×52 | 4.7 | accepted | 5.3 | accepted | 106 |
| Deep learning | full | 421×39 | 7.9 | accepted | 8.0 | accepted | 94 |
| Image processing | none | 127×21 | 1.6 | **rejected** | 1.3 | **rejected** | 47 |
| Numerics | full | 736×50 | 7.7 | accepted | 15.5 | accepted | 146 |
| Phonetics | full | 886×38 | 1.8 | **rejected** | 2.7 | accepted | 126 |
| Physics | full | 809×31 | 4.8 | accepted | 22.2 | accepted | 135 |
| Psychology | full | 897×64 | 5.5 | accepted | 43.4 | accepted | 127 |
| Reinforcement | full | 1270×57 | 2.6 | accepted | 3.2 | accepted | 182 |
| Sensory systems | partial | 717×37 | -0.0 | **rejected** | -1.2 | **rejected** | 138 |
| Short range | full | 370×21 | 2.7 | accepted | 5.8 | accepted | 97 |
| Solar resource | full | 843×60 | 1.6 | **rejected** | 2.1 | accepted | 134 |
| Team dynamics | full | 630×24 | 1.3 | **rejected** | 1.7 | **rejected** | 77 |
| Computation theory | full | 601×12 | 1.3 | **rejected** | 2.3 | accepted | 110 |

**False rejections at z = 2.0: sentence level 8/20 = 40%; 30-second windows 4/20 = 20%.** The std floor bound the z in 1 (sentence) / 2 (30 s) of the 40 cells.

Per-lecture detail (objective, shuffled mean ± std as estimated, std used):

| lecture | granularity | objective | null mean | null std | std used |
|---|---|---:|---:|---:|---:|
| ML for health | sentence | 0.4140 | 0.4139 | 0.0040 | 0.0040 |
| ML for health | 30 s | 0.5324 | 0.5321 | 0.0027 | 0.0027 |
| Decarbonization | sentence | 0.5357 | 0.4970 | 0.0019 | 0.0020 |
| Decarbonization | 30 s | 0.6622 | 0.5858 | 0.0082 | 0.0082 |
| Climate & Cities | sentence | 0.5438 | 0.4894 | 0.0035 | 0.0035 |
| Climate & Cities | 30 s | 0.6745 | 0.5785 | 0.0149 | 0.0149 |
| Climate policies | sentence | 0.4611 | 0.4533 | 0.0035 | 0.0035 |
| Climate policies | 30 s | 0.6211 | 0.5743 | 0.0054 | 0.0054 |
| Cognitive robotics | sentence | 0.4624 | 0.4348 | 0.0066 | 0.0066 |
| Cognitive robotics | 30 s | 0.6294 | 0.5626 | 0.0087 | 0.0087 |
| Computer Vision | sentence | 0.5904 | 0.5727 | 0.0047 | 0.0047 |
| Computer Vision | 30 s | 0.6923 | 0.6585 | 0.0041 | 0.0041 |
| Productdesign | sentence | 0.4129 | 0.4079 | 0.0040 | 0.0040 |
| Productdesign | 30 s | 0.5794 | 0.5372 | 0.0053 | 0.0053 |
| Cryptocurrency | sentence | 0.4626 | 0.4481 | 0.0031 | 0.0031 |
| Cryptocurrency | 30 s | 0.6047 | 0.5734 | 0.0058 | 0.0058 |
| Deep learning | sentence | 0.5184 | 0.4739 | 0.0056 | 0.0056 |
| Deep learning | 30 s | 0.6445 | 0.5545 | 0.0113 | 0.0113 |
| Image processing | sentence | 0.5378 | 0.5345 | 0.0021 | 0.0021 |
| Image processing | 30 s | 0.6012 | 0.5986 | 0.0001 | 0.0020 |
| Numerics | sentence | 0.4629 | 0.4422 | 0.0027 | 0.0027 |
| Numerics | 30 s | 0.5998 | 0.5471 | 0.0034 | 0.0034 |
| Phonetics | sentence | 0.4435 | 0.4379 | 0.0031 | 0.0031 |
| Phonetics | 30 s | 0.6266 | 0.6161 | 0.0040 | 0.0040 |
| Physics | sentence | 0.4752 | 0.4597 | 0.0033 | 0.0033 |
| Physics | 30 s | 0.6403 | 0.5775 | 0.0028 | 0.0028 |
| Psychology | sentence | 0.4593 | 0.4353 | 0.0044 | 0.0044 |
| Psychology | 30 s | 0.6323 | 0.5454 | 0.0019 | 0.0020 |
| Reinforcement | sentence | 0.4353 | 0.4255 | 0.0037 | 0.0037 |
| Reinforcement | 30 s | 0.5873 | 0.5630 | 0.0075 | 0.0075 |
| Sensory systems | sentence | 0.4211 | 0.4213 | 0.0049 | 0.0049 |
| Sensory systems | 30 s | 0.5304 | 0.5388 | 0.0069 | 0.0069 |
| Short range | sentence | 0.4659 | 0.4562 | 0.0036 | 0.0036 |
| Short range | 30 s | 0.6085 | 0.5716 | 0.0064 | 0.0064 |
| Solar resource | sentence | 0.4455 | 0.4379 | 0.0046 | 0.0046 |
| Solar resource | 30 s | 0.5846 | 0.5727 | 0.0057 | 0.0057 |
| Team dynamics | sentence | 0.4537 | 0.4468 | 0.0054 | 0.0054 |
| Team dynamics | 30 s | 0.6040 | 0.5924 | 0.0068 | 0.0068 |
| Computation theory | sentence | 0.4785 | 0.4720 | 0.0050 | 0.0050 |
| Computation theory | 30 s | 0.5890 | 0.5794 | 0.0041 | 0.0041 |
