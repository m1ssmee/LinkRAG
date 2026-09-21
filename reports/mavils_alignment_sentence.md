# MaViLS alignment — target T2 — sentence level

20 of 20 lectures · transcript + slide PDF only (no video frames) · segments: their sentence granularity · embedder `BAAI/bge-m3` · align weights {'dense': 0.6, 'bm25': 0.25, 'keyword': 0.15} · λ=0.05 σ=0.02 β=0.15 B=2 μ=0.02 · metric: their micro-F1 over labelled sentences (−1 excluded) · sentences and transcript are theirs verbatim · wall time 2736s (embedding included, not a latency claim)

Their numbers are copied from Anderer et al. 2024, Table 1 (*Audio* column: transcript only, λ_jump = 0.1) and Table 2 (all three features, λ_jump = 0.1). **Audio is the like-for-like column**; *All* uses frame OCR and image features this adapter does not consume.

| lecture | sentences | slides | **ours (monotonic)** | naive argmax | their audio | their all | Δ vs their audio |
|---|---:|---:|---:|---:|---:|---:|---:|
| Deep learning | 377 | 39 | **0.62** | 0.42 | 0.79 | 0.99 | -0.17 |
| Short range | 350 | 21 | **0.42** | 0.28 | 0.57 | 0.80 | -0.15 |
| Numerics | 713 | 50 | **0.51** | 0.24 | 0.46 | 0.81 | +0.05 |
| Reinforcement | 1201 | 57 | **0.23** | 0.12 | 0.31 | 0.75 | -0.08 |
| Computer Vision | 80 | 18 | **0.73** | 0.49 | 0.65 | 1.00 | +0.08 |
| Climate & Cities | 143 | 44 | **0.43** | 0.32 | 0.49 | 0.86 | -0.06 |
| Decarbonization | 275 | 45 | **0.21** | 0.22 | 0.52 | 0.93 | -0.31 |
| Cryptocurrency | 500 | 52 | **0.31** | 0.17 | 0.21 | 0.84 | +0.10 |
| Solar resource | 518 | 60 | **0.29** | 0.16 | 0.46 | 0.53 | -0.17 |
| Psychology | 893 | 64 | **0.44** | 0.23 | 0.58 | 0.80 | -0.14 |
| Productdesign | 247 | 27 | **0.55** | 0.42 | 0.72 | 0.71 | -0.17 |
| Image processing | 124 | 21 | **0.19** | 0.23 | 0.53 | 0.97 | -0.34 |
| Sensory systems | 421 | 37 | **0.15** | 0.20 | 0.46 | 0.53 | -0.31 |
| ML for health | 489 | 77 | **0.20** | 0.21 | 0.57 | 0.95 | -0.37 |
| Climate policies | 274 | 24 | **0.64** | 0.40 | 0.79 | 0.93 | -0.15 |
| Computation theory | 554 | 12 | **0.36** | 0.29 | 0.56 | 0.84 | -0.20 |
| Physics | 576 | 31 | **0.75** | 0.40 | 0.60 | 0.93 | +0.15 |
| Phonetics | 221 | 38 | **0.39** | 0.26 | 0.05 | 0.65 | +0.34 |
| Team dynamics | 55 | 24 | **0.62** | 0.50 | 0.89 | 0.96 | -0.27 |
| Cognitive robotics | 735 | 62 | **0.23** | 0.15 | 0.42 | 0.66 | -0.19 |
| **average** | 8746 | | **0.41** | 0.29 | 0.53 | 0.82 | -0.12 |

Ours beats their audio-only F1 on **5/20** lectures; beats their all-features F1 on 0/20. Back-jumps in our paths: 835 total (B=2).

Caveats. Their audio column was produced with distiluse-base-multilingual-cased and their own DP; ours uses bge-m3 + BM25 + IDF overlap and our DP, so the delta mixes embedder and algorithm — the naive column isolates the DP's share on *our* similarity. Slide numbering: we assume their `Slidenumber` is the 1-based PDF page; a lecture where `gt_max_slide` exceeds the page count would break that assumption and is flagged below.

Image-only pages OCR'd (no text layer): Image processing (21), Sensory systems (7), Cognitive robotics (5)

