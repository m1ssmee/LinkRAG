# Relatedness gate — negative control and false-rejection check

The gate (`linkrag.link.align.relatedness_gate`): the penalised DP objective per segment must exceed the mean of 5 shuffled-slide-order alignments by `align.relatedness_z` shuffled standard deviations; otherwise no audio_slide links are emitted and the pair is recorded as unrelated in `links.jsonl` `_meta`. Cross-document semantic figure_text links use the same z over a word-shuffle null (`figure_text.document_pair_gate`). Per-segment abstention (`align.min_segment_sim`) runs after the gate on surviving pairs.

## Negative control — pilot01 audio × unrelated deck

Audio: `hsieh.mp3` (frozen transcript, 51 segments). Deck: `unrelated_psychology_mit.pdf`. Gate: `align.relatedness_z` = 2.0.

| link type | total | cross-file |
|---|---:|---:|
| audio_slide | 0 | 0 |
| deictic | 0 | 0 |
| figure_text | 13 | 0 |
| same_slide | 13 | 0 |
| **all** | 26 | **0** |

Pairs recorded as unrelated in `links.jsonl` `_meta`: [{"audio": "hsieh.mp3", "deck": "unrelated_psychology_mit.pdf", "link_type": "audio_slide", "score": 0.5274, "null_mean": 0.528, "null_std": 0.0003, "z": -1.7653, "threshold_z": 2.0}]

Expected zero cross-file links: **PASS**.

## False-rejection check — 20 MaViLS lectures, sentence granularity (their protocol) (all related by construction)

Gate on the cached matrices (sentence granularity (their protocol), page OCR, our hybrid similarity), our DP at pilot parameters, 5 shuffled slide orders, threshold z = 2.0.

| lecture | segments × slides | path score | shuffled mean ± std | z | verdict |
|---|---:|---:|---:|---:|---|
| ML for health | 822×77 | 0.414 | 0.414 ± 0.004 | 0.0 | **REJECTED** |
| Decarbonization | 312×45 | 0.536 | 0.497 ± 0.002 | 19.9 | related |
| Climate & Cities | 192×44 | 0.544 | 0.489 ± 0.004 | 15.5 | related |
| Climate policies | 841×24 | 0.461 | 0.453 ± 0.004 | 2.2 | related |
| Cognitive robotics | 735×62 | 0.462 | 0.435 ± 0.007 | 4.2 | related |
| Computer Vision | 80×18 | 0.590 | 0.573 ± 0.005 | 3.7 | related |
| Productdesign | 715×27 | 0.413 | 0.408 ± 0.004 | 1.3 | **REJECTED** |
| Cryptocurrency | 684×52 | 0.463 | 0.448 ± 0.003 | 4.7 | related |
| Deep learning | 421×39 | 0.518 | 0.474 ± 0.006 | 7.9 | related |
| Image processing | 127×21 | 0.538 | 0.534 ± 0.002 | 1.6 | **REJECTED** |
| Numerics | 736×50 | 0.463 | 0.442 ± 0.003 | 7.7 | related |
| Phonetics | 886×38 | 0.443 | 0.438 ± 0.003 | 1.8 | **REJECTED** |
| Physics | 809×31 | 0.475 | 0.460 ± 0.003 | 4.8 | related |
| Psychology | 897×64 | 0.459 | 0.435 ± 0.004 | 5.5 | related |
| Reinforcement | 1270×57 | 0.435 | 0.425 ± 0.004 | 2.6 | related |
| Sensory systems | 717×37 | 0.421 | 0.421 ± 0.005 | -0.0 | **REJECTED** |
| Short range | 370×21 | 0.466 | 0.456 ± 0.004 | 2.7 | related |
| Solar resource | 843×60 | 0.445 | 0.438 ± 0.005 | 1.6 | **REJECTED** |
| Team dynamics | 630×24 | 0.454 | 0.447 ± 0.005 | 1.3 | **REJECTED** |
| Computation theory | 601×12 | 0.479 | 0.472 ± 0.005 | 1.3 | **REJECTED** |

**False-rejection rate at z = 2.0: 8/20 = 40%.** z-scores: min -0.0, median 2.7, max 19.9. Rejected: ML for health, Productdesign, Image processing, Phonetics, Sensory systems, Solar resource, Team dynamics, Computation theory.

## False-rejection check — 20 MaViLS lectures, 30-second windows (LinkRAG's segment granularity) (all related by construction)

Gate on the cached matrices (30-second windows (LinkRAG's segment granularity), page OCR, our hybrid similarity), our DP at pilot parameters, 5 shuffled slide orders, threshold z = 2.0.

| lecture | segments × slides | path score | shuffled mean ± std | z | verdict |
|---|---:|---:|---:|---:|---|
| ML for health | 145×77 | 0.532 | 0.532 ± 0.003 | 0.1 | **REJECTED** |
| Decarbonization | 67×45 | 0.662 | 0.586 ± 0.008 | 9.3 | related |
| Climate & Cities | 39×44 | 0.674 | 0.578 ± 0.015 | 6.4 | related |
| Climate policies | 131×24 | 0.621 | 0.574 ± 0.005 | 8.7 | related |
| Cognitive robotics | 124×62 | 0.629 | 0.563 ± 0.009 | 7.7 | related |
| Computer Vision | 35×18 | 0.692 | 0.658 ± 0.004 | 8.2 | related |
| Productdesign | 95×27 | 0.579 | 0.537 ± 0.005 | 7.9 | related |
| Cryptocurrency | 106×52 | 0.605 | 0.573 ± 0.006 | 5.3 | related |
| Deep learning | 94×39 | 0.644 | 0.554 ± 0.011 | 8.0 | related |
| Image processing | 47×21 | 0.601 | 0.599 ± 0.000 | 18.7 | related |
| Numerics | 146×50 | 0.600 | 0.547 ± 0.003 | 15.5 | related |
| Phonetics | 126×38 | 0.627 | 0.616 ± 0.004 | 2.7 | related |
| Physics | 135×31 | 0.640 | 0.577 ± 0.003 | 22.2 | related |
| Psychology | 127×64 | 0.632 | 0.545 ± 0.002 | 45.8 | related |
| Reinforcement | 182×57 | 0.587 | 0.563 ± 0.008 | 3.2 | related |
| Sensory systems | 138×37 | 0.530 | 0.539 ± 0.007 | -1.2 | **REJECTED** |
| Short range | 97×21 | 0.609 | 0.572 ± 0.006 | 5.8 | related |
| Solar resource | 134×60 | 0.585 | 0.573 ± 0.006 | 2.1 | related |
| Team dynamics | 77×24 | 0.604 | 0.592 ± 0.007 | 1.7 | **REJECTED** |
| Computation theory | 110×12 | 0.589 | 0.579 ± 0.004 | 2.3 | related |

**False-rejection rate at z = 2.0: 3/20 = 15%.** z-scores: min -1.2, median 7.0, max 45.8. Rejected: ML for health, Sensory systems, Team dynamics.


## Reading

- Negative control passes: pilot01's talk against an unrelated MIT psychology deck emits **0 cross-file links** (audio_slide gate z = −1.8; the 13 figure_text and 13 same_slide links are all inside the deck). Retrieval on that corpus is plain hybrid search with no expansion (tested).
- The price is false rejections on related pairs. At z = 2.0 the gate wrongly rejects **8/20 MaViLS pairs at sentence granularity (40 %)** and **3/20 at LinkRAG's 30-second granularity (15 %)** — the granularity the gate runs at in the pipeline. The rejected pairs are the ones whose alignment F1 is lowest anyway (ML for health, Sensory systems, Team dynamics — 55 sentences): where the similarity carries no monotone signal, the gate cannot see it either.
- False-rejection rate vs threshold, from the per-lecture z-scores above (the control sits at z ≈ −2, so any z ≥ 0 rejects it):

| z threshold | sentence | 30 s windows |
|---:|---:|---:|
| 0.5 | 2/20 | 2/20 |
| 1.0 | 2/20 | 2/20 |
| 1.5 | 5/20 | 2/20 |
| 2.0 | 8/20 | 3/20 |
| 3.0 | 11/20 | 6/20 |

z = 2.0 stays the default: on the extended dataset a wrongly *admitted* unrelated pair poisons retrieval for every question on that lecture, while a wrongly rejected related pair degrades to plain hybrid search. A lower z is a one-line config change (`align.relatedness_z`) if the extended set shows related pairs being lost.

Aside from the gate: the 30-second-window matrices built for this check also give the DP **0.515** on their protocol (vs 0.461 at sentence level, their 0.53) — LinkRAG's own segment granularity is worth 5 points on MaViLS (`data/processed/mavils/S/*.w30.ocr.npz`; table in the scratch run, not a tuned quantity).
