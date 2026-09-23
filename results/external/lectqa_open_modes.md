# LectQA-Vid — open-ended: T1 replica vs iterative vs full context

Three ways to give the same answerer its context, scored with T1's own metric suite:

- **baseline**: T1's pipeline (§4.2-4.4) replicated. Chunks: whisper sentences merged while cosine > 0.75 and gap < 5.0 s, near-duplicates (> 0.9) dropped, frame OCR overlapping speech by > 2.0 s suppressed; embedder `BAAI/bge-base-en-v1.5` mean-pooled; top-10 above cosine 0.6; temporal filter (eq. 22) on the question's annotated interval ± 3.0 s; merge of consecutive results with gap < 5.0 s; cross-encoder `cross-encoder/ms-marco-MiniLM-L-6-v2` over the top-8; top-4 in temporal order. **Eq. 22 reads the gold timestamps (an oracle).** It is applied only where the stamp is usable (`gold_interval`); elsewhere the pipeline runs without it. Both are reported separately below.
- **iterative**: ours: RRF (bge-m3 dense + BM25) over our sentence-packed transcript and frame-OCR units, one LLM follow-up query, top-4. No timestamps are read.
- **full_context**: no retrieval; every transcript unit of the video, in time order. No timestamps and no frame OCR.

T1 leaves open: M, L and the merge gap (ours above); the cross-encoder (ours: MS MARCO MiniLM); one FAISS index over all videos then a video-id filter (ours: each video's own chunks, i.e. that filter without other videos competing for the top-K); Gemini captions (ours: tesseract OCR, mostly suppressed by their θt rule).

**Scope.** The single run on the other 1118 open-ended questions was not made: the dry run estimated $7.43 for subset + rest against the item's $3.50 cap, so only the subset ran.

## Subset: 300 questions (100 per difficulty, seed `lectqa-open-subset-20260923`), 93 videos, 3 repeats, mean ± std over repeats

answerer `gpt-5.4-mini-2026-03-17` temperature 0.0 (JSON answer with claims, the same prompt for every mode) · metrics T1 eqs. 29-35 (`linkrag.eval.lectqa_metrics`), in %.

| level | mode | n | F1 | Sim | BLEU | METEOR | R1 | abstained |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| simple | baseline | 100 | 41.34 ± 0.08 | 64.38 ± 0.16 | 13.09 ± 0.51 | 41.50 ± 0.44 | 46.01 ± 0.41 | 12 |
| simple | iterative | 100 | 46.30 ± 0.27 | 72.58 ± 0.30 | 13.88 ± 0.27 | 47.02 ± 0.27 | 54.09 ± 0.41 | 1 |
| simple | full_context | 100 | 48.88 ± 0.40 | 74.48 ± 0.21 | 16.18 ± 0.54 | 50.80 ± 0.22 | 57.31 ± 0.23 | 1 |
| simple | **T1 Table 4** | — | 29.47 | 77.23 | 8.61 | 38.15 | 36.82 | — |
| hard | baseline | 100 | 28.06 ± 0.30 | 52.18 ± 0.22 | 4.83 ± 0.26 | 27.22 ± 0.26 | 33.41 ± 0.43 | 24 |
| hard | iterative | 100 | 36.26 ± 0.44 | 64.89 ± 0.62 | 7.72 ± 0.17 | 35.84 ± 0.67 | 42.67 ± 0.34 | 6 |
| hard | full_context | 100 | 38.39 ± 0.41 | 65.38 ± 0.33 | 8.49 ± 0.33 | 38.66 ± 0.23 | 46.29 ± 0.21 | 6 |
| hard | **T1 Table 4** | — | 24.38 | 74.56 | 5.29 | 34.71 | 30.94 | — |
| very hard | baseline | 100 | 24.28 ± 0.20 | 55.41 ± 0.20 | 1.24 ± 0.17 | 21.13 ± 0.18 | 28.49 ± 0.22 | 15 |
| very hard | iterative | 100 | 27.53 ± 0.13 | 59.91 ± 0.40 | 1.76 ± 0.10 | 25.36 ± 0.31 | 32.58 ± 0.32 | 7 |
| very hard | full_context | 100 | 28.94 ± 0.16 | 60.79 ± 0.18 | 1.92 ± 0.10 | 27.76 ± 0.22 | 35.50 ± 0.33 | 6 |
| very hard | **T1 Table 4** | — | 16.72 | 71.48 | 2.83 | 24.36 | 22.51 | — |
| overall | baseline | 300 | 31.23 ± 0.06 | 57.32 ± 0.01 | 6.39 ± 0.27 | 29.95 ± 0.20 | 35.97 ± 0.09 | 51 |
| overall | iterative | 300 | 36.70 ± 0.16 | 65.79 ± 0.39 | 7.79 ± 0.16 | 36.07 ± 0.18 | 43.11 ± 0.06 | 14 |
| overall | full_context | 300 | 38.74 ± 0.23 | 66.88 ± 0.09 | 8.86 ± 0.29 | 39.07 ± 0.15 | 46.37 ± 0.25 | 13 |
| overall | **T1 Table 4** | — | 23.52 | 74.42 | 5.58 | 32.41 | 29.76 | — |

## Paired differences (subset, token F1 points, 95 % bootstrap CI over questions)

Each question's F1 averaged over its repeats; the mean paired difference and a 10,000-resample bootstrap over questions (numpy seed 20260923). This is the question-sampling uncertainty; the ± above is run-to-run noise only.

| difference | simple | hard | very hard | overall |
|---|---:|---:|---:|---:|
| full_context − iterative | +2.58 [+0.70, +4.59] | +2.13 [+0.61, +3.66] | +1.41 [+0.25, +2.58] | +2.04 [+1.14, +2.99] |
| full_context − baseline | +7.54 [+4.03, +11.35] | +10.33 [+7.11, +13.82] | +4.66 [+2.80, +6.61] | +7.51 [+5.72, +9.40] |
| iterative − baseline | +4.95 [+1.68, +8.45] | +8.20 [+4.98, +11.62] | +3.25 [+1.46, +5.18] | +5.47 [+3.80, +7.24] |

## Baseline with and without its oracle filter (repeat 0, subset)

answerer `gpt-5.4-mini-2026-03-17` temperature 0.0 (JSON answer with claims, the same prompt for every mode) · metrics T1 eqs. 29-35 (`linkrag.eval.lectqa_metrics`), in %.

The two groups are different questions, so the other modes' F1 on the same groups is the control: a gap they share is the questions, not the filter.

| questions | n | baseline F1 | baseline Sim | baseline empty context | iterative F1 | full_context F1 |
|---|---:|---:|---:|---:|---:|---:|
| eq. 22 applied (usable gold stamp) | 160 | 27.54 | 50.91 | 34 | 35.71 | 36.76 |
| eq. 22 not applicable | 140 | 35.45 | 64.67 | 1 | 37.92 | 41.49 |

Malformed JSON replies (answer scored as the raw text): 0 of 2700. Answers run: 2700 (subset 2700, rest 0).

LLM cost (this run):

- `gpt-5.4-mini-2026-03-17`: 3496 calls (0 cached) · 2,357,374 in / 349,798 out · $3.3421
- run total: $3.3421
- cumulative (all recorded runs, `reports/llm_ledger.jsonl`): 31,223 calls · 27,901,195 in / 1,246,756 out · $21.54 · 6 row(s) unpriced · 14,412,524 tokens estimated
