# LectQA-Vid — attribution ablation: re-query vs pipeline vs eq. 22

Open-ended subset: 300 questions (100 per difficulty, seed `lectqa-open-subset-20260923`), 93 videos, 3 repeats · answerer `gpt-5.4-mini-2026-03-17` temperature 0.0, the same JSON answer prompt for every mode · metrics T1 eqs. 29-35 (`linkrag.eval.lectqa_metrics`), in %, mean ± std over repeats.

Modes (context handed to the same answerer, top-4 units unless stated):

- **T1 replica (eq. 22 on)** (stored mode `baseline`, `lectqa_open_modes.jsonl`): T1's pipeline replicated (bge-base, semantic chunking, top-10 above cosine 0.6, merge, MS MARCO MiniLM cross-encoder). Eq. 22 keeps only chunks overlapping the question's **annotated (gold) interval ± 3 s, read at test time**, where the stamp is usable.
- **T1 replica, eq. 22 off** (new): the same pipeline; the gold interval is never read.
- **ours: RRF, no re-query** (new): our plain baseline, RRF of bge-m3 dense + BM25 over our sentence-packed transcript and frame-OCR units; no LLM call before answering.
- **ours: RRF + one re-query (iterative)** (stored): the same RRF, one LLM follow-up query, top-4 of the merged rounds.
- **full transcript** (stored): every transcript unit, no retrieval.

The replica, iterative and full-context rows are the stored answers of `lectqa_open_modes.jsonl` (not re-run). New answers used the same reply cache: where a new mode's evidence equals a stored run's, the stored reply was replayed.

| level | mode | n | F1 | Sim | BLEU | METEOR | R1 | abstained |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| simple | T1 replica (eq. 22 on) | 100 | 41.34 ± 0.08 | 64.38 ± 0.16 | 13.09 ± 0.51 | 41.50 ± 0.44 | 46.01 ± 0.41 | 12 |
| simple | T1 replica, eq. 22 off | 100 | 43.39 ± 0.16 | 68.28 ± 0.36 | 13.38 ± 0.10 | 43.53 ± 0.30 | 48.66 ± 0.39 | 6 |
| simple | ours: RRF, no re-query | 100 | 42.64 ± 0.37 | 70.02 ± 0.22 | 12.43 ± 0.22 | 43.06 ± 0.31 | 49.60 ± 0.07 | 2 |
| simple | ours: RRF + one re-query (iterative) | 100 | 46.30 ± 0.27 | 72.58 ± 0.30 | 13.88 ± 0.27 | 47.02 ± 0.27 | 54.09 ± 0.41 | 1 |
| simple | full transcript | 100 | 48.88 ± 0.40 | 74.48 ± 0.21 | 16.18 ± 0.54 | 50.80 ± 0.22 | 57.31 ± 0.23 | 1 |
| simple | *T1 Table 4 (published; unnamed answerer)* | — | 29.47 | 77.23 | 8.61 | 38.15 | 36.82 | — |
| hard | T1 replica (eq. 22 on) | 100 | 28.06 ± 0.30 | 52.18 ± 0.22 | 4.83 ± 0.26 | 27.22 ± 0.26 | 33.41 ± 0.43 | 24 |
| hard | T1 replica, eq. 22 off | 100 | 35.21 ± 0.14 | 63.06 ± 0.19 | 6.45 ± 0.19 | 34.89 ± 0.23 | 42.10 ± 0.14 | 6 |
| hard | ours: RRF, no re-query | 100 | 34.66 ± 0.14 | 62.87 ± 0.23 | 6.35 ± 0.09 | 33.65 ± 0.29 | 41.06 ± 0.23 | 7 |
| hard | ours: RRF + one re-query (iterative) | 100 | 36.26 ± 0.44 | 64.89 ± 0.62 | 7.72 ± 0.17 | 35.84 ± 0.67 | 42.67 ± 0.34 | 6 |
| hard | full transcript | 100 | 38.39 ± 0.41 | 65.38 ± 0.33 | 8.49 ± 0.33 | 38.66 ± 0.23 | 46.29 ± 0.21 | 6 |
| hard | *T1 Table 4 (published; unnamed answerer)* | — | 24.38 | 74.56 | 5.29 | 34.71 | 30.94 | — |
| very hard | T1 replica (eq. 22 on) | 100 | 24.28 ± 0.20 | 55.41 ± 0.20 | 1.24 ± 0.17 | 21.13 ± 0.18 | 28.49 ± 0.22 | 15 |
| very hard | T1 replica, eq. 22 off | 100 | 25.48 ± 0.09 | 57.77 ± 0.49 | 1.37 ± 0.09 | 22.52 ± 0.22 | 30.26 ± 0.34 | 9 |
| very hard | ours: RRF, no re-query | 100 | 27.58 ± 0.51 | 58.85 ± 0.06 | 1.60 ± 0.02 | 24.60 ± 0.31 | 32.34 ± 0.47 | 6 |
| very hard | ours: RRF + one re-query (iterative) | 100 | 27.53 ± 0.13 | 59.91 ± 0.40 | 1.76 ± 0.10 | 25.36 ± 0.31 | 32.58 ± 0.32 | 7 |
| very hard | full transcript | 100 | 28.94 ± 0.16 | 60.79 ± 0.18 | 1.92 ± 0.10 | 27.76 ± 0.22 | 35.50 ± 0.33 | 6 |
| very hard | *T1 Table 4 (published; unnamed answerer)* | — | 16.72 | 71.48 | 2.83 | 24.36 | 22.51 | — |
| overall | T1 replica (eq. 22 on) | 300 | 31.23 ± 0.06 | 57.32 ± 0.01 | 6.39 ± 0.27 | 29.95 ± 0.20 | 35.97 ± 0.09 | 51 |
| overall | T1 replica, eq. 22 off | 300 | 34.69 ± 0.04 | 63.03 ± 0.18 | 7.07 ± 0.05 | 33.64 ± 0.25 | 40.34 ± 0.03 | 21 |
| overall | ours: RRF, no re-query | 300 | 34.96 ± 0.31 | 63.92 ± 0.11 | 6.79 ± 0.07 | 33.77 ± 0.25 | 41.00 ± 0.13 | 15 |
| overall | ours: RRF + one re-query (iterative) | 300 | 36.70 ± 0.16 | 65.79 ± 0.39 | 7.79 ± 0.16 | 36.07 ± 0.18 | 43.11 ± 0.06 | 14 |
| overall | full transcript | 300 | 38.74 ± 0.23 | 66.88 ± 0.09 | 8.86 ± 0.29 | 39.07 ± 0.15 | 46.37 ± 0.25 | 13 |
| overall | *T1 Table 4 (published; unnamed answerer)* | — | 23.52 | 74.42 | 5.58 | 32.41 | 29.76 | — |

## Paired differences (token F1 points, 95 % bootstrap CI over questions)

Each question's F1 averaged over its repeats; mean paired difference, 10,000-resample bootstrap over questions (numpy seed 20260923), as in `lectqa_open_modes.md`.

| difference | what it isolates | simple | hard | very hard | overall |
|---|---|---:|---:|---:|---:|
| T1 replica (eq. 22 on) − T1 replica, eq. 22 off | the eq. 22 oracle filter | -2.05 [-4.31, -0.03] | -7.15 [-10.48, -4.17] | -1.19 [-2.54, -0.00] | -3.46 [-4.91, -2.12] |
| ours: RRF + one re-query (iterative) − ours: RRF, no re-query | the re-query (same index, same top-4) | +3.66 [+1.69, +5.92] | +1.60 [-0.02, +3.41] | -0.04 [-1.29, +1.15] | +1.74 [+0.76, +2.75] |
| ours: RRF, no re-query − T1 replica (eq. 22 on) | the rest of the pipeline: embedder, index, chunking, reranker, eq. 22 | +1.30 [-2.33, +4.90] | +6.60 [+3.51, +9.92] | +3.29 [+1.49, +5.14] | +3.73 [+2.03, +5.50] |
| ours: RRF, no re-query − T1 replica, eq. 22 off | the two retrieval pipelines, neither reading gold timestamps | -0.75 [-3.96, +2.40] | -0.55 [-2.75, +1.52] | +2.10 [+0.48, +3.72] | +0.27 [-1.13, +1.63] |
| ours: RRF + one re-query (iterative) − T1 replica, eq. 22 off | ours (with re-query) vs T1's pipeline without its oracle filter | +2.91 [+0.48, +5.58] | +1.05 [-0.83, +2.95] | +2.06 [+0.70, +3.49] | +2.00 [+0.88, +3.21] |

## Eq. 22 at test time

Eq. 22 read the gold interval on 160 of the 300 questions (the others have no usable stamp, so the replica ran without it and its evidence there is identical with eq. 22 on or off). On 34 questions it left the replica **no context at all**; the answer is then the fixed "No evidence was retrieved" reply, with no model call.

| questions | n | mode | F1 (repeat 0) | empty context |
|---|---:|---|---:|---:|
| eq. 22 applied | 160 | T1 replica (eq. 22 on) | 27.54 | 34 |
| eq. 22 applied | 160 | T1 replica, eq. 22 off | 33.93 | 6 |
| eq. 22 applied | 160 | ours: RRF, no re-query | 34.05 | 0 |
| of which eq. 22 left no context | 34 | T1 replica (eq. 22 on) | 1.73 | 34 |
| of which eq. 22 left no context | 34 | T1 replica, eq. 22 off | 27.63 | 6 |
| of which eq. 22 left no context | 34 | ours: RRF, no re-query | 28.95 | 0 |

Malformed JSON replies among the new answers: 0 of 1800.

LLM cost (this run):

- `gpt-5.4-mini-2026-03-17`: 1781 calls (547 cached) · 857,504 in / 222,638 out · $1.1281 (cache replays free; would have been $1.6450)
- run total: $1.1281
- cumulative (all recorded runs, `reports/llm_ledger.jsonl`): 35,264 calls · 29,426,382 in / 1,619,400 out · $23.18 · 6 row(s) unpriced · 14,412,524 tokens estimated
