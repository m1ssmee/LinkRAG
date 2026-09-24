# MaViLS — frame OCR and frame→page text similarity

Every representative frame of `mavils_frames.md` (320 px on the longest side, the only size kept) OCR'd with tesseract after a 3× LANCZOS upscale (`linkrag.link.visual.ocr_frame`; at native size slide text is a few pixels tall and OCR reads almost nothing: a label-free check on frames alone, before any F1). Cached per frame by the hash of the frame. Frame→page scores (`text_similarity`), tokenised as everywhere else (`linkrag.index.tokenize`), against the page OCR text: **TF-IDF** cosine (fit on the pages) and **BM25** (min-max scaled per frame). A frame with no text gets a zero row. LLM-free, $0.

*Confident* = best page beats the runner-up by more than 0.05 (TF-IDF) / 0.1 (BM25), set before scoring. A sanity check, not an accuracy: no label is used.

| lecture | frames | with any text | median words | confident TF-IDF | confident BM25 | same best page |
|---|---:|---:|---:|---:|---:|---:|
| ML for health | 322 | 0.41 | 0 | 0.17 | 0.24 | 0.80 |
| Decarbonization | 93 | 0.58 | 11 | 0.43 | 0.35 | 0.92 |
| Climate & Cities | 98 | 0.46 | 0 | 0.30 | 0.26 | 0.99 |
| Cognitive robotics | 296 | 0.48 | 0 | 0.30 | 0.33 | 0.91 |
| Computer Vision | 29 | 0.93 | 63 | 0.90 | 0.90 | 0.97 |
| Productdesign | 267 | 0.38 | 0 | 0.35 | 0.36 | 0.99 |
| Cryptocurrency | 470 | 0.87 | 17 | 0.64 | 0.55 | 0.98 |
| Deep learning | 94 | 0.76 | 7 | 0.60 | 0.57 | 0.98 |
| Image processing | 59 | 0.41 | 0 | 0.29 | 0.36 | 0.90 |
| Numerics | 174 | 0.56 | 4 | 0.33 | 0.42 | 0.95 |
| Phonetics | 441 | 0.08 | 0 | 0.04 | 0.04 | 0.98 |
| Physics | 187 | 0.52 | 1 | 0.47 | 0.47 | 0.96 |
| Psychology | 367 | 0.55 | 4 | 0.40 | 0.40 | 0.92 |
| Reinforcement | 1141 | 0.28 | 0 | 0.12 | 0.15 | 0.89 |
| Sensory systems | 447 | 0.57 | 2 | 0.40 | 0.44 | 0.93 |
| Short range | 360 | 0.23 | 0 | 0.22 | 0.22 | 1.00 |
| Solar resource | 474 | 0.38 | 0 | 0.14 | 0.21 | 0.94 |
| Team dynamics | 403 | 0.48 | 0 | 0.19 | 0.18 | 0.97 |
| Computation theory | 96 | 0.94 | 31 | 0.75 | 0.88 | 0.94 |
| **total / mean** | 5818 | 0.52 | 0 | 0.37 | 0.38 | 0.94 |

Climate policies has no video (see `mavils_frames.md`), so no frames.
