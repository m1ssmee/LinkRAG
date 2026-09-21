# MaViLS — like-for-like protocol (all: 20 lectures)

Segments: 30-second windows of their sentences, broadcast back and scored per sentence · slide text: tesseract OCR of every rendered page (their slide-side input) · embedder `BAAI/bge-m3` + BM25 + IDF overlap · DP λ=0.05 σ=0.02 β=0.15 B=2 μ=0.02 · scored with sklearn exactly as their evaluation script.

**Their scoring, verbatim from `evaluation/evaluate_recall_precision.py`:** `ground_truth_labels = ground_truth_column.unique()`; `mask = (ground_truth_column != -1)`; `f1_score(filtered_ground_truth, filtered_result, labels=ground_truth_labels, average='micro')`. One row per transcript sentence (`generate_output_dict_by_sentence`), per lecture, then the unweighted mean over lectures. `labels` comes from the unfiltered column, so -1 is a label: predicting -1 on a labelled sentence is a false positive for -1 and a miss for the true slide, i.e. **abstention cannot raise their F1** -- it scores like a wrong slide that is itself a label, and *worse* than a wrong slide the ground truth never uses (that one costs recall only). Abstention can only be seen in precision-on-answered and coverage, which we report alongside. Their audio-only similarity is distiluse cosine between the sentence and **tesseract OCR of the rendered slide image** (`matching_algorithm.py`), decoded with their DP (penalty 0.1·|Δslide|, ×2 backwards, no skip penalty).

**Column correspondence.** Their *audio-only* ⇔ our `dp` column **when run with `--slide-text ocr` at sentence granularity** (same input protocol; our similarity and DP). `naive` is the same matrix without sequence structure. Their *all-features* uses video frames we do not consume.

Cells: **their F1 (precision-on-answered / coverage)**.

| lecture | n | slides | text layer | **dp** | naive | their audio | their all | Δ dp − their audio |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| ML for health | 489 | 77 | full | **0.56 (0.56 / 1.00)** | 0.32 (0.32 / 1.00) | 0.57 | 0.95 | -0.01 |
| Decarbonization | 275 | 45 | full | **0.53 (0.53 / 1.00)** | 0.39 (0.39 / 1.00) | 0.52 | 0.93 | +0.01 |
| Climate & Cities | 143 | 44 | full | **0.61 (0.59 / 1.00)** | 0.49 (0.49 / 1.00) | 0.49 | 0.86 | +0.12 |
| Climate policies | 274 | 24 | full | **0.79 (0.77 / 1.00)** | 0.67 (0.62 / 1.00) | 0.79 | 0.93 | +0.00 |
| Cognitive robotics | 735 | 62 | partial | **0.50 (0.50 / 1.00)** | 0.31 (0.31 / 1.00) | 0.42 | 0.66 | +0.08 |
| Computer Vision | 80 | 18 | full | **0.64 (0.64 / 1.00)** | 0.53 (0.51 / 1.00) | 0.65 | 1.00 | -0.01 |
| Productdesign | 247 | 27 | full | **0.80 (0.79 / 1.00)** | 0.54 (0.52 / 1.00) | 0.72 | 0.71 | +0.08 |
| Cryptocurrency | 500 | 52 | full | **0.29 (0.29 / 1.00)** | 0.25 (0.25 / 1.00) | 0.21 | 0.84 | +0.08 |
| Deep learning | 377 | 39 | full | **0.79 (0.79 / 1.00)** | 0.56 (0.56 / 1.00) | 0.79 | 0.99 | -0.00 |
| Image processing | 124 | 21 | none | **0.04 (0.04 / 1.00)** | 0.28 (0.28 / 1.00) | 0.53 | 0.97 | -0.49 |
| Numerics | 713 | 50 | full | **0.57 (0.57 / 1.00)** | 0.35 (0.35 / 1.00) | 0.46 | 0.81 | +0.11 |
| Phonetics | 221 | 38 | full | **0.24 (0.24 / 1.00)** | 0.37 (0.37 / 1.00) | 0.05 | 0.65 | +0.19 |
| Physics | 576 | 31 | full | **0.77 (0.77 / 1.00)** | 0.64 (0.64 / 1.00) | 0.60 | 0.93 | +0.17 |
| Psychology | 893 | 64 | full | **0.56 (0.52 / 1.00)** | 0.41 (0.38 / 1.00) | 0.58 | 0.80 | -0.02 |
| Reinforcement | 1201 | 57 | full | **0.36 (0.36 / 1.00)** | 0.27 (0.26 / 1.00) | 0.31 | 0.75 | +0.05 |
| Sensory systems | 421 | 37 | partial | **0.17 (0.17 / 1.00)** | 0.26 (0.26 / 1.00) | 0.46 | 0.53 | -0.29 |
| Short range | 350 | 21 | full | **0.65 (0.65 / 1.00)** | 0.43 (0.43 / 1.00) | 0.57 | 0.80 | +0.08 |
| Solar resource | 518 | 60 | full | **0.36 (0.36 / 1.00)** | 0.25 (0.24 / 1.00) | 0.46 | 0.53 | -0.10 |
| Team dynamics | 55 | 24 | full | **0.78 (0.78 / 1.00)** | 0.67 (0.67 / 1.00) | 0.89 | 0.96 | -0.11 |
| Computation theory | 554 | 12 | full | **0.29 (0.29 / 1.00)** | 0.29 (0.28 / 1.00) | 0.56 | 0.84 | -0.27 |
| **mean** | 8746 | | | **0.51** | 0.41 | 0.53 | 0.82 | -0.02 |

`dp` above their audio-only on 11/20 lectures.

