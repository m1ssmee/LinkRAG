# MaViLS — representative frames and frame→page similarity

Videos: the MaViLS Kaggle dataset (their README; the GitHub repo ships none). Frames: 1 fps, shrunk to 320 px on the longest side, segmented by 64-bit dHash (`linkrag.ingest.video_slides.segment_slides`: a new segment when consecutive frames differ by > 2 bits, segments < 2 s merged, revisits **not** merged: at the LectQA default of 10 bits with revisit merging, slides sharing a template collapsed, e.g. Decarbonization kept 28 frames for a 45-page deck). One representative frame per segment is kept (`data/processed/mavils/frames/`, not tracked), with its intervals; the unzipped videos are removed after extraction (the Kaggle zip is the source). Pages: rendered from their PDFs and shrunk to the same size. LLM-free, $0.

Frame→page scores (`linkrag.link.visual`): **dHash** 1 − Hamming/64; **SwiftFormer-xs** (`MBZUAI/swiftformer-xs`, MaViLS's own image model) cosine of the flattened last hidden state. *Confident* = the best page beats the runner-up by more than 0.0625 (dHash) / 0.05 (SwiftFormer), margins set before any frame was scored. This is a sanity check, not an accuracy: no frame label is used.

| lecture | video min | frames kept | frame disk | pages | confident dHash | confident SwiftFormer | same best page |
|---|---:|---:|---:|---:|---:|---:|---:|
| ML for health | 83 | 322 | 19.3 MB | 77 | 0.05 | 0.22 | 0.04 |
| Decarbonization | 40 | 93 | 2.9 MB | 45 | 0.06 | 0.41 | 0.24 |
| Climate & Cities | 23 | 98 | 3.1 MB | 44 | 0.32 | 0.17 | 0.21 |
| Cognitive robotics | 72 | 296 | 14.4 MB | 62 | 0.08 | 0.18 | 0.10 |
| Computer Vision | 35 | 29 | 0.9 MB | 18 | 0.79 | 0.83 | 0.97 |
| Productdesign | 53 | 267 | 20.7 MB | 27 | 0.09 | 0.03 | 0.11 |
| Cryptocurrency | 60 | 470 | 23.3 MB | 52 | 0.09 | 0.11 | 0.21 |
| Deep learning | 54 | 94 | 2.5 MB | 39 | 0.16 | 0.31 | 0.40 |
| Image processing | 31 | 59 | 3.7 MB | 21 | 0.05 | 0.10 | 0.10 |
| Numerics | 85 | 174 | 5.4 MB | 50 | 0.00 | 0.12 | 0.06 |
| Phonetics | 70 | 441 | 26.9 MB | 38 | 0.00 | 0.16 | 0.03 |
| Physics | 79 | 187 | 9.1 MB | 31 | 0.10 | 0.29 | 0.27 |
| Psychology | 71 | 367 | 18.1 MB | 64 | 0.04 | 0.11 | 0.07 |
| Reinforcement | 102 | 1141 | 58.7 MB | 57 | 0.46 | 0.00 | 0.00 |
| Sensory systems | 82 | 447 | 21.2 MB | 37 | 0.18 | 0.23 | 0.08 |
| Short range | 64 | 360 | 24.3 MB | 21 | 0.08 | 0.14 | 0.04 |
| Solar resource | 75 | 474 | 23.4 MB | 60 | 0.26 | 0.12 | 0.03 |
| Team dynamics | 43 | 403 | 34.0 MB | 24 | 0.10 | 0.00 | 0.05 |
| Computation theory | 63 | 96 | 2.8 MB | 12 | 0.08 | 0.27 | 0.38 |
| **total / mean** | 1187 | 5818 | 314.7 MB | | 0.16 | 0.20 | 0.18 |

**No video found for 1 lecture(s):** climate_science_policy_MIT2.
