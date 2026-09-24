# MaViLS — slide-visibility gate (representative frames)

Rule: a frame shows a slide if its OCR reads at least *W* words (3× upscale, `mavils_frame_ocr_sanity.md`) or its image channel (swiftformer) picks a page by more than *M* (best minus runner-up); otherwise it is a speaker or room shot, and every sentence taking that frame gets constant image and frame-OCR rows, so its page is decided by the speech-to-slide text alone (`three_way`, `linkrag.link.visual.slide_visible`). The optional high-contrast-text-area signal was not used: the two signals above were already computed.

Fusion: the tuned three-way (visual / frame_ocr / theirs = 0.5 / 0.25 / 0.25, bm25), our DP at σ = 0.2. **Tune half only**; the test half is run once in `mavils_final_round.md`. LLM-free, $0.

## Tune half: W × M grid (their F1, mean of 10 lectures)

No gate: **0.793**.

| W (words) \ M (margin) | 0.01 | 0.02 | 0.05 | 0.1 |
|---|---:|---:|---:|---:|
| 1 | 0.818 | 0.824 ← | 0.807 | 0.798 |
| 3 | 0.813 | 0.810 | 0.796 | 0.785 |
| 5 | 0.814 | 0.810 | 0.790 | 0.782 |
| 10 | 0.817 | 0.812 | 0.787 | 0.774 |

Chosen: **W = 1, M = 0.02** (tune 0.824 vs 0.793 without the gate); recorded in `results/external/mavils_tuned.json`.

## Frames classified speaker, and the tune-half effect

| lecture | half | frames | speaker frames | tune F1 no gate | tune F1 gate | predictions changed |
|---|---|---:|---:|---:|---:|---:|
| Climate & Cities | test | 98 | 51% | — | — | — |
| Cognitive robotics | tune | 296 | 52% | 0.605 | 0.632 | 139 of 735 |
| Computation theory | test | 96 | 6% | — | — | — |
| Computer Vision | tune | 29 | 0% | 0.981 | 0.981 | 0 of 80 |
| Cryptocurrency | tune | 470 | 13% | 0.811 | 0.811 | 6 of 684 |
| Decarbonization | tune | 93 | 33% | 0.924 | 0.924 | 2 of 312 |
| Deep learning | test | 94 | 24% | — | — | — |
| Image processing | tune | 59 | 39% | 0.694 | 0.629 | 34 of 127 |
| ML for health | test | 322 | 38% | — | — | — |
| Numerics | test | 174 | 26% | — | — | — |
| Phonetics | test | 441 | 76% | — | — | — |
| Physics | tune | 187 | 47% | 0.958 | 0.958 | 182 of 809 |
| Productdesign | tune | 267 | 46% | 0.690 | 0.818 | 114 of 715 |
| Psychology | tune | 367 | 45% | 0.687 | 0.685 | 129 of 897 |
| Reinforcement | test | 1141 | 72% | — | — | — |
| Sensory systems | tune | 447 | 34% | 0.600 | 0.824 | 229 of 717 |
| Short range | test | 360 | 77% | — | — | — |
| Solar resource | test | 474 | 49% | — | — | — |
| Team dynamics | tune | 403 | 49% | 0.982 | 0.982 | 31 of 630 |

Tune lectures with < 10 % speaker frames: Computer Vision; predictions unchanged by the gate on 1 of 1.
