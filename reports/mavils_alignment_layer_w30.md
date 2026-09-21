# MaViLS alignment — target T2 — 30s windows

20 of 20 lectures · transcript + slide PDF only (no video frames) · segments: 30-second windows of their sentences (our segment granularity), scored per sentence · embedder `BAAI/bge-m3` · align weights {'dense': 0.6, 'bm25': 0.25, 'keyword': 0.15} · λ=0.05 σ=0.02 β=0.15 B=2 μ=0.02 · metric: their micro-F1 over labelled sentences (−1 excluded) · sentences and transcript are theirs verbatim · wall time 1126s (embedding included, not a latency claim)

Their numbers are copied from Anderer et al. 2024, Table 1 (*Audio* column: transcript only, λ_jump = 0.1) and Table 2 (all three features, λ_jump = 0.1). **Audio is the like-for-like column**; *All* uses frame OCR and image features this adapter does not consume.

| lecture | sentences | slides | **ours (monotonic)** | naive argmax | their audio | their all | Δ vs their audio |
|---|---:|---:|---:|---:|---:|---:|---:|
| Deep learning | 377 | 39 | **0.78** | 0.56 | 0.79 | 0.99 | -0.01 |
| Short range | 350 | 21 | **0.65** | 0.43 | 0.57 | 0.80 | +0.08 |
| Numerics | 713 | 50 | **0.56** | 0.40 | 0.46 | 0.81 | +0.10 |
| Reinforcement | 1201 | 57 | **0.25** | 0.25 | 0.31 | 0.75 | -0.06 |
| Computer Vision | 80 | 18 | **0.61** | 0.58 | 0.65 | 1.00 | -0.04 |
| Climate & Cities | 143 | 44 | **0.39** | 0.25 | 0.49 | 0.86 | -0.10 |
| Decarbonization | 275 | 45 | **0.21** | 0.20 | 0.52 | 0.93 | -0.31 |
| Cryptocurrency | 500 | 52 | **0.31** | 0.25 | 0.21 | 0.84 | +0.10 |
| Solar resource | 518 | 60 | **0.38** | 0.25 | 0.46 | 0.53 | -0.08 |
| Psychology | 893 | 64 | **0.56** | 0.40 | 0.58 | 0.80 | -0.02 |
| Productdesign | 247 | 27 | **0.77** | 0.52 | 0.72 | 0.71 | +0.05 |
| Image processing | 124 | 21 | **0.04** | 0.28 | 0.53 | 0.97 | -0.49 |
| Sensory systems | 421 | 37 | **0.08** | 0.23 | 0.46 | 0.53 | -0.38 |
| ML for health | 489 | 77 | **0.34** | 0.28 | 0.57 | 0.95 | -0.23 |
| Climate policies | 274 | 24 | **0.62** | 0.63 | 0.79 | 0.93 | -0.17 |
| Computation theory | 554 | 12 | **0.41** | 0.39 | 0.56 | 0.84 | -0.15 |
| Physics | 576 | 31 | **0.82** | 0.68 | 0.60 | 0.93 | +0.22 |
| Phonetics | 221 | 38 | **0.24** | 0.36 | 0.05 | 0.65 | +0.19 |
| Team dynamics | 55 | 24 | **0.76** | 0.65 | 0.89 | 0.96 | -0.13 |
| Cognitive robotics | 735 | 62 | **0.27** | 0.23 | 0.42 | 0.66 | -0.15 |
| **average** | 8746 | | **0.45** | 0.39 | 0.53 | 0.82 | -0.08 |

Ours beats their audio-only F1 on **6/20** lectures; beats their all-features F1 on 1/20. Back-jumps in our paths: 38 total (B=2).

Caveats. Their audio column was produced with distiluse-base-multilingual-cased and their own DP; ours uses bge-m3 + BM25 + IDF overlap and our DP, so the delta mixes embedder and algorithm — the naive column isolates the DP's share on *our* similarity. Slide numbering: we assume their `Slidenumber` is the 1-based PDF page; a lecture where `gt_max_slide` exceeds the page count would break that assumption and is flagged below.

Image-only pages OCR'd (no text layer): Image processing (21), Sensory systems (7), Cognitive robotics (5)


## Reading the first run (2026-09-21)

- **Average 0.45 (30 s windows) / 0.41 (sentence level) vs their audio-only 0.53.** We
  are below the like-for-like column on 14 of 20 lectures and above on 6 (Physics
  +0.22, Phonetics +0.19, Numerics and Cryptocurrency +0.10, Short range +0.08,
  Productdesign +0.05). `reports/mavils_alignment_sentence.md` is the same run at
  their sentence granularity.
- **The DP earns its keep only when the similarity carries signal.** Over 20
  lectures the monotone path adds +0.06 over the naive argmax on the same matrix
  (+0.13 at sentence level), but on the three decks that had to be page-OCR'd
  (Image processing, Sensory systems, Cognitive robotics) it is *worse* than naive
  (0.04 vs 0.28, 0.08 vs 0.23): with a near-flat matrix the skip penalty makes the
  path stay put, and it stays on the wrong slide. Diagnosed on Sensory systems:
  the ground-truth slide's median rank under our S is 9 of 37 (top-1 19 %); dense-only
  is no better. These MIT OCW decks are full of "Image removed due to copyright"
  placeholders, so the text side is thin.
- **Decarbonization (0.21 vs 0.52) is unexplained** — no OCR fallback, no page-count
  flag. To inspect before anything else is changed.
- What this does and does not say. Our alignment was built and measured on one
  27-slide deck with 30 s segments; on 20 decks of 12–77 pages it does not match a
  distiluse-cosine + jump-penalty DP on transcript alone. The number that would move
  it is the similarity, not the DP, and that is a design question (which signals to
  fuse for a deck with little text), not a threshold. No parameter was changed for
  this run.
