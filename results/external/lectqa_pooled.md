# LectQA-Vid — pooled corpus: retrieval when the video is not given

**Exploratory. This setting is not in DESIGN.md's priority order and has not been agreed with the mentor; do not cite it until it has.** It stays on T1's dataset (dataset policy), but it is not T1's task.

## What changed

T1 retrieves inside the one video a question belongs to. Here all 95 prepared videos are one corpus (3753 units) and the question is asked without its video. The per-video rows are computed in the same pass on the same questions, so every difference below is paired. No answerer, no LLM call, no re-embedding (the stored per-video vectors are stacked): **$0, deterministic, one run.**

## Caveats first

- **The questions were written for one video.** 2102 of 2832 name "the video", "the speaker" or similar, which identifies nothing in a pooled corpus; such questions bound video@k from above for every method alike. The split is in the last table.
- **Interval hits are scored on the 1598 of 2832 questions with a usable gold interval** (known issue (h)); video hits on all of them.
- **Retrieval only.** A hit says a unit was retrieved, not that an answer would be right. Answer quality needs LLM calls.
- **The iterative modes are not run** (one LLM call per question).
- **`linkrag` with no reranker equals plain top-8 by construction** (seed normalisation, DESIGN.md *Design changes*), so links are tested only through the reranker: *links + reranker* against *reranker, no links*.
- **"reranker, no links" hides the graph from the reranker.** `lectqa_frameslides.md` and `compare_retrieval.py` pass it the graph in baseline mode too, so their `baseline / complementarity` rows are not this cell.
- **The @1 columns barely differ between methods by construction:** the reranker's first pick is the top-scored unit, and a method can differ at @1 only where two units tie on score and the tie is broken differently. Only @3 and @8 separate the methods (measurement rule 1: inspected, not a wiring fault).
- **The reranker is not scale-free.** In baseline mode it sees raw RRF scores (about 0.03) next to alpha 0.3 and gamma 0.4, so relevance barely counts: that is the *raw scores* row, and it is how `baseline / complementarity` runs everywhere in the repo. The control for the links is *reranker, no links*: the full system's own rank-normalised scores, nothing followed, graph hidden. That row was added after the first full run showed the raw-score row confounded links with score scale; it is a correction of the comparison, not a tuned setting.
- **Nothing was tuned.** Config defaults; pool 20, k = 8, one hop, as in `lectqa_frameslides.md`. The method list was fixed after a 4-video smoke run of the code (120 questions) and before the full run. *links + reranker* is the system's default and the primary row. *gated* is the recorded modality gate (default off). *links only* switches the coverage and redundancy terms off (alpha = gamma = 0, not tuned values) to show the links alone.
- Frames are OCR'd with tesseract; many videos are animated explainers, not slide talks (finding 12).

## Corpora

| corpus | videos | units | by modality | links in the graph |
|---|---:|---:|---|---:|
| frames | 95 | 3753 | audio 1870, figure 1883 | 3459 |
| frame-derived slides | 95 | 6957 | audio 1870, figure 2912, text 2175 | 5000 |

*frames* = transcript segments + one OCR unit per changed frame, joined by temporal co-occurrence. *frame-derived slides* = the same transcript segments + slide page and figure units recovered from the frames, joined by the linking layer (`link_corpus`).

## Results, all difficulties

video@k: a top-k unit comes from the question's video. interval@k: it also overlaps the gold interval.

| corpus | setting | method | video@1 | video@3 | video@8 | interval@1 | interval@3 | interval@8 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| frames | single video | plain top-8 | given | given | given | 45.6% | 68.7% | 85.1% |
| frames | single video | reranker on raw scores, no links | given | given | given | 45.8% | 61.8% | 81.5% |
| frames | single video | reranker, no links | given | given | given | 45.8% | 66.1% | 84.5% |
| frames | single video | links + reranker | given | given | given | 45.8% | 63.7% | 82.4% |
| frames | single video | links + reranker, gated | given | given | given | 45.8% | 64.3% | 82.7% |
| frames | single video | links only | given | given | given | 45.8% | 67.4% | 83.8% |
| frames | pooled | plain top-8 | 63.5% | 75.9% | 83.8% | 25.8% | 41.2% | 52.8% |
| frames | pooled | reranker on raw scores, no links | 63.5% | 69.3% | 75.8% | 25.9% | 29.5% | 34.2% |
| frames | pooled | reranker, no links | 63.5% | 73.2% | 81.6% | 25.9% | 35.4% | 47.3% |
| frames | pooled | links + reranker | 63.5% | 72.5% | 79.3% | 25.9% | 35.2% | 45.9% |
| frames | pooled | links + reranker, gated | 63.5% | 72.6% | 79.3% | 25.9% | 35.1% | 45.9% |
| frames | pooled | links only | 63.5% | 75.3% | 83.2% | 25.9% | 40.6% | 52.4% |
| frame-derived slides | single video | plain top-8 | given | given | given | 43.2% | 67.0% | 84.3% |
| frame-derived slides | single video | reranker on raw scores, no links | given | given | given | 43.2% | 60.2% | 80.4% |
| frame-derived slides | single video | reranker, no links | given | given | given | 43.2% | 62.1% | 82.4% |
| frame-derived slides | single video | links + reranker | given | given | given | 43.2% | 57.7% | 78.8% |
| frame-derived slides | single video | links + reranker, gated | given | given | given | 43.2% | 60.7% | 78.8% |
| frame-derived slides | single video | links only | given | given | given | 43.2% | 66.1% | 82.7% |
| frame-derived slides | pooled | plain top-8 | 60.5% | 74.9% | 83.3% | 22.6% | 38.5% | 51.4% |
| frame-derived slides | pooled | reranker on raw scores, no links | 60.6% | 69.9% | 77.3% | 22.7% | 29.8% | 34.7% |
| frame-derived slides | pooled | reranker, no links | 60.6% | 72.0% | 81.8% | 22.7% | 32.2% | 46.3% |
| frame-derived slides | pooled | links + reranker | 60.6% | 68.9% | 79.0% | 22.7% | 29.2% | 42.2% |
| frame-derived slides | pooled | links + reranker, gated | 60.6% | 70.5% | 79.0% | 22.7% | 30.9% | 42.4% |
| frame-derived slides | pooled | links only | 60.6% | 74.3% | 82.3% | 22.7% | 38.0% | 49.8% |

n = 2832 questions for video@k, 1598 for interval@k, in every row.

## Paired differences, pooled setting (percentage points, 95 % bootstrap CI over questions)

10,000 resamples, numpy seed 20260923, as in `lectqa_open_modes.md`.

| difference | corpus | video@1 | video@8 | interval@1 | interval@3 | interval@8 |
|---|---|---:|---:|---:|---:|---:|
| links + reranker − reranker, no links (what the links add) | frames | +0.00 [+0.00, +0.00] | -2.30 [-2.86, -1.77] | +0.00 [+0.00, +0.00] | -0.25 [-0.94, +0.38] | -1.44 [-2.50, -0.38] |
| reranker, no links − plain top-8 (what set-level reranking adds) | frames | +0.00 [-0.14, +0.14] | -2.19 [-2.93, -1.52] | +0.06 [+0.00, +0.19] | -5.82 [-7.13, -4.57] | -5.44 [-6.70, -4.26] |
| reranker on raw scores, no links − plain top-8 | frames | +0.00 [-0.14, +0.14] | -7.91 [-9.11, -6.74] | +0.06 [+0.00, +0.19] | -11.76 [-13.45, -10.08] | -18.52 [-20.65, -16.46] |
| links + reranker − plain top-8 (the full system, primary) | frames | +0.00 [-0.14, +0.14] | -4.48 [-5.30, -3.71] | +0.06 [+0.00, +0.19] | -6.07 [-7.51, -4.69] | -6.88 [-8.39, -5.44] |
| links + reranker, gated − plain top-8 | frames | +0.00 [-0.14, +0.14] | -4.48 [-5.30, -3.71] | +0.06 [+0.00, +0.19] | -6.13 [-7.57, -4.76] | -6.88 [-8.39, -5.44] |
| links only − plain top-8 (link-following alone) | frames | +0.00 [-0.14, +0.14] | -0.56 [-0.85, -0.32] | +0.06 [+0.00, +0.19] | -0.63 [-1.19, -0.13] | -0.38 [-1.06, +0.25] |
| links + reranker − reranker, no links (what the links add) | frame-derived slides | +0.00 [+0.00, +0.00] | -2.79 [-3.43, -2.19] | +0.00 [+0.00, +0.00] | -3.00 [-3.94, -2.13] | -4.07 [-5.19, -3.00] |
| reranker, no links − plain top-8 (what set-level reranking adds) | frame-derived slides | +0.14 [+0.04, +0.28] | -1.52 [-2.22, -0.81] | +0.06 [+0.00, +0.19] | -6.26 [-7.63, -4.88] | -5.07 [-6.38, -3.82] |
| reranker on raw scores, no links − plain top-8 | frame-derived slides | +0.14 [+0.04, +0.28] | -6.04 [-7.20, -4.94] | +0.06 [+0.00, +0.19] | -8.64 [-10.20, -7.07] | -16.71 [-18.71, -14.77] |
| links + reranker − plain top-8 (the full system, primary) | frame-derived slides | +0.14 [+0.04, +0.28] | -4.31 [-5.12, -3.50] | +0.06 [+0.00, +0.19] | -9.26 [-10.89, -7.70] | -9.14 [-10.70, -7.63] |
| links + reranker, gated − plain top-8 | frame-derived slides | +0.14 [+0.04, +0.28] | -4.31 [-5.12, -3.50] | +0.06 [+0.00, +0.19] | -7.63 [-9.14, -6.20] | -9.01 [-10.51, -7.51] |
| links only − plain top-8 (link-following alone) | frame-derived slides | +0.14 [+0.04, +0.28] | -1.02 [-1.41, -0.67] | +0.06 [+0.00, +0.19] | -0.50 [-0.94, -0.13] | -1.56 [-2.32, -0.81] |
| frame-derived slides − frames, plain top-8 | both | -3.00 [-4.06, -1.94] | -0.46 [-1.24, +0.28] | -3.25 [-4.57, -1.94] | -2.75 [-4.26, -1.31] | -1.38 [-2.69, -0.06] |

## What not knowing the video costs (frames corpus, plain top-8)

| difference | interval@1 | interval@3 | interval@8 |
|---|---:|---:|---:|
| pooled − single video | -19.71 [-21.96, -17.52] | -27.47 [-29.72, -25.16] | -32.35 [-34.67, -30.04] |

## Per difficulty, pooled, frames corpus

| level | n (video) | n (interval) | method | video@1 | video@8 | interval@1 | interval@8 |
|---|---:|---:|---|---:|---:|---:|---:|
| simple | 951 | 601 | plain top-8 | 65.5% | 85.3% | 34.9% | 63.9% |
| simple | 951 | 601 | reranker on raw scores, no links | 65.5% | 77.8% | 34.9% | 42.6% |
| simple | 951 | 601 | reranker, no links | 65.5% | 83.3% | 34.9% | 58.4% |
| simple | 951 | 601 | links + reranker | 65.5% | 80.7% | 34.9% | 56.7% |
| simple | 951 | 601 | links + reranker, gated | 65.5% | 80.7% | 34.9% | 56.7% |
| simple | 951 | 601 | links only | 65.5% | 84.8% | 34.9% | 63.9% |
| hard | 944 | 515 | plain top-8 | 67.3% | 86.3% | 22.9% | 46.4% |
| hard | 944 | 515 | reranker on raw scores, no links | 67.2% | 77.8% | 22.9% | 29.5% |
| hard | 944 | 515 | reranker, no links | 67.2% | 83.5% | 22.9% | 40.2% |
| hard | 944 | 515 | links + reranker | 67.2% | 81.7% | 22.9% | 40.0% |
| hard | 944 | 515 | links + reranker, gated | 67.2% | 81.7% | 22.9% | 40.0% |
| hard | 944 | 515 | links only | 67.2% | 85.6% | 22.9% | 46.2% |
| very hard | 937 | 482 | plain top-8 | 57.5% | 79.6% | 17.6% | 45.6% |
| very hard | 937 | 482 | reranker on raw scores, no links | 57.6% | 71.9% | 17.8% | 28.8% |
| very hard | 937 | 482 | reranker, no links | 57.6% | 77.9% | 17.8% | 41.1% |
| very hard | 937 | 482 | links + reranker | 57.6% | 75.5% | 17.8% | 38.6% |
| very hard | 937 | 482 | links + reranker, gated | 57.6% | 75.5% | 17.8% | 38.6% |
| very hard | 937 | 482 | links only | 57.6% | 79.2% | 17.8% | 44.6% |

## Question wording (pooled, frames corpus, plain top-8)

| question names "the video / the speaker / …" | n | video@1 | video@8 |
|---|---:|---:|---:|
| yes | 2102 | 61.5% | 82.7% |
| no | 730 | 69.0% | 86.8% |

## Reading

- **Pooling makes retrieval matter.** Finding the interval is far harder once the video is not given (the *pooled − single video* row), so this setting can separate retrieval methods where T1's cannot.
- **Plain top-8 is the best row.** No component of the system improves localisation here; each interval of the *− plain top-8* differences is at or below zero.
- **The links add nothing measurable.** Against the proper control (*reranker, no links*) the full system is level or slightly lower, and *links only* is level with plain top-8.
- **Most of the reranker's cost in baseline mode is a score-scale artefact.** Compare the two *no links* rows: the same objective on rank-normalised scores loses about a third as much as on raw RRF scores. Every `baseline / complementarity` number in the repo runs on raw scores.
- **The metric cannot reward what the system is for.** A gold interval is one stretch of speech. A slide, a figure or a linked neighbour that completes the answer takes a slot and can only lower interval@k. Whether such evidence improves answers is an answer-quality question and needs LLM calls.

## Provenance

- Dataset: LectQA-Vid (T1), 95 of 100 videos, MCQ and open-ended questions together (question text only, no options).
- Embedder `BAAI/bge-m3`; retrieval RRF of dense + BM25, candidates 50; reranker alpha 0.3, beta 0.2, gamma 0.4.
- Config hash (retrieve, link, index) `aee9d9be8eb94a46`; commit `83396c0` with uncommitted changes.
- Rows: `lectqa_pooled.jsonl`. Reproduce: `python scripts/adapters/lectqa_vid.py pooled --max-cost 0`.
