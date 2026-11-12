# LectQA-Vid — free re-checks of the stored results

Computed from stored answers and stored retrieval rows. No answer was generated and no report was re-run. $0.

## Caveats first

- **The fallback is our construction.** T1 does not say what its pipeline does when eq. 22 leaves nothing, so both the fixed reply in the stored replica and the fallback here are our choices. The fallback swaps in the stored answer of the unfiltered replica; nothing was re-ranked or re-answered.
- **Where the gold window sits is judged by proxies**: whether our own top-4 overlaps it, and whether the full-transcript answer cites a unit inside it. Neither is a human reading. The hand-check sheet is for that.
- **The echo floor depends on how the references are written.** It says how much token F1 a reply earns by restating the question; it is not a system.
- By-question intervals are the stored ones (10,000 resamples, numpy seed 20260923). By-video intervals resample whole videos with the same seed. An interval that contains 0 is marked *(crosses 0)* and is to be read as no measurable difference.

## Open-ended answers: paired differences, by question and by video

300 questions on 93 videos, token F1 points, each question's F1 averaged over its 3 repeats.

| difference | level | by question | by video |
|---|---|---:|---:|
| T1 replica (eq. 22 on) − T1 replica, eq. 22 off | overall | -3.46 [-4.91, -2.12] | -3.46 [-5.30, -1.71] |
| T1 replica (eq. 22 on) − T1 replica, eq. 22 off | simple | -2.05 [-4.31, -0.03] | -2.05 [-4.67, +0.20] (crosses 0) |
| T1 replica (eq. 22 on) − T1 replica, eq. 22 off | hard | -7.15 [-10.48, -4.17] | -7.15 [-11.02, -3.41] |
| T1 replica (eq. 22 on) − T1 replica, eq. 22 off | very hard | -1.19 [-2.54, -0.00] | -1.19 [-2.83, +0.20] (crosses 0) |
| T1 replica, eq. 22 on, unfiltered when the window is empty − T1 replica, eq. 22 off | overall | -0.53 [-1.31, +0.17] | -0.53 [-1.30, +0.20] (crosses 0) |
| T1 replica, eq. 22 on, unfiltered when the window is empty − T1 replica, eq. 22 off | simple | +0.02 [-1.21, +1.35] | +0.02 [-1.17, +1.23] (crosses 0) |
| T1 replica, eq. 22 on, unfiltered when the window is empty − T1 replica, eq. 22 off | hard | -1.60 [-3.33, -0.19] | -1.60 [-3.22, -0.20] |
| T1 replica, eq. 22 on, unfiltered when the window is empty − T1 replica, eq. 22 off | very hard | -0.03 [-0.86, +0.77] | -0.03 [-0.84, +0.75] (crosses 0) |
| ours: RRF, no re-query − T1 replica, eq. 22 off | overall | +0.27 [-1.13, +1.63] | +0.27 [-0.90, +1.48] (crosses 0) |
| ours: RRF, no re-query − T1 replica, eq. 22 off | simple | -0.75 [-3.96, +2.40] | -0.75 [-3.66, +1.89] (crosses 0) |
| ours: RRF, no re-query − T1 replica, eq. 22 off | hard | -0.55 [-2.75, +1.52] | -0.55 [-2.52, +1.51] (crosses 0) |
| ours: RRF, no re-query − T1 replica, eq. 22 off | very hard | +2.10 [+0.48, +3.72] | +2.10 [+0.64, +3.60] |
| ours: RRF + one re-query (iterative) − ours: RRF, no re-query | overall | +1.74 [+0.76, +2.75] | +1.74 [+0.77, +2.75] |
| ours: RRF + one re-query (iterative) − ours: RRF, no re-query | simple | +3.66 [+1.69, +5.92] | +3.66 [+1.64, +5.77] |
| ours: RRF + one re-query (iterative) − ours: RRF, no re-query | hard | +1.60 [-0.02, +3.41] | +1.60 [+0.03, +3.11] |
| ours: RRF + one re-query (iterative) − ours: RRF, no re-query | very hard | -0.04 [-1.29, +1.15] | -0.04 [-1.27, +1.16] (crosses 0) |
| ours: RRF + one re-query (iterative) − T1 replica, eq. 22 off | overall | +2.00 [+0.88, +3.21] | +2.00 [+0.96, +3.11] |
| ours: RRF + one re-query (iterative) − T1 replica, eq. 22 off | simple | +2.91 [+0.48, +5.58] | +2.91 [+0.51, +5.36] |
| ours: RRF + one re-query (iterative) − T1 replica, eq. 22 off | hard | +1.05 [-0.83, +2.95] | +1.05 [-0.46, +2.72] (crosses 0) |
| ours: RRF + one re-query (iterative) − T1 replica, eq. 22 off | very hard | +2.06 [+0.70, +3.49] | +2.06 [+0.67, +3.51] |
| ours: RRF + one re-query (iterative) − T1 replica, eq. 22 on, unfiltered when the window is empty | overall | +2.54 [+1.24, +3.91] | +2.54 [+1.23, +3.82] |
| ours: RRF + one re-query (iterative) − T1 replica, eq. 22 on, unfiltered when the window is empty | simple | +2.89 [+0.21, +5.76] | +2.89 [+0.25, +5.53] |
| ours: RRF + one re-query (iterative) − T1 replica, eq. 22 on, unfiltered when the window is empty | hard | +2.65 [+0.23, +5.27] | +2.65 [+0.75, +4.68] |
| ours: RRF + one re-query (iterative) − T1 replica, eq. 22 on, unfiltered when the window is empty | very hard | +2.08 [+0.60, +3.69] | +2.08 [+0.58, +3.66] |
| ours: RRF + one re-query (iterative) − T1 replica (eq. 22 on) | overall | +5.47 [+3.80, +7.24] | +5.47 [+3.34, +7.68] |
| ours: RRF + one re-query (iterative) − T1 replica (eq. 22 on) | simple | +4.95 [+1.68, +8.45] | +4.95 [+1.42, +8.54] |
| ours: RRF + one re-query (iterative) − T1 replica (eq. 22 on) | hard | +8.20 [+4.98, +11.62] | +8.20 [+4.61, +11.89] |
| ours: RRF + one re-query (iterative) − T1 replica (eq. 22 on) | very hard | +3.25 [+1.46, +5.18] | +3.25 [+1.27, +5.41] |
| full transcript − ours: RRF + one re-query (iterative) | overall | +2.04 [+1.14, +2.99] | +2.04 [+1.08, +3.06] |
| full transcript − ours: RRF + one re-query (iterative) | simple | +2.58 [+0.70, +4.59] | +2.58 [+0.60, +4.56] |
| full transcript − ours: RRF + one re-query (iterative) | hard | +2.13 [+0.61, +3.66] | +2.13 [+0.76, +3.52] |
| full transcript − ours: RRF + one re-query (iterative) | very hard | +1.41 [+0.25, +2.58] | +1.41 [+0.20, +2.65] |

## The replica-to-re-query gap, step by step

| step | F1 after | change, by video |
|---|---:|---:|
| T1 replica (eq. 22 on) | 31.23 | |
| unfiltered answer on the empty windows | 34.16 | +2.93 [+1.45, +4.56] |
| eq. 22 off everywhere | 34.69 | +0.53 [-0.20, +1.30] (crosses 0) |
| our RRF index in place of T1's pipeline | 34.96 | +0.27 [-0.90, +1.48] (crosses 0) |
| one re-query | 36.70 | +1.74 [+0.77, +2.75] |

## Why eq. 22 leaves windows empty

Eq. 22 was applied on 160 of the 300 questions and left 34 with no context. The replica applies it to the top-10 above cosine 0.6, as T1's §4.4.2 does.

| what happened to the chunks inside the window | questions |
|---|---:|
| a window chunk is in the top-10 but at or below cosine 0.6 | 18 |
| every window chunk ranks below the top-10 | 16 |

| questions | speech exists in the window | our RRF top-4 overlaps it | full-transcript answer's first claim cites a unit in it | RRF F1 | full-transcript F1 |
|---|---:|---:|---:|---:|---:|
| eq. 22 kept something (n = 126) | 126 of 126 (100%) | 109 of 126 (87%) | 93 of 121 (77%) | 35.23 | 37.48 |
| eq. 22 left nothing (n = 34) | 34 of 34 (100%) | 11 of 34 (32%) | 8 of 29 (28%) | 28.60 | 33.34 |

## Question-echo floor

The question text scored as if it were the answer, on the same 300 questions.

| level | echo F1 | echo similarity | T1 Table 4 F1 | T1 replica F1 | full transcript F1 |
|---|---:|---:|---:|---:|---:|
| simple | 23.70 | 59.08 | 29.47 | 41.34 | 48.88 |
| hard | 22.62 | 58.66 | 24.38 | 28.06 | 38.39 |
| very hard | 22.28 | 59.23 | 16.72 | 24.28 | 28.94 |
| overall | 22.87 | 58.99 | 23.52 | 31.23 | 38.74 |

## Localisation: paired differences, by question and by video

Frames corpus, questions whose gold interval passes the format and range checks, percentage points, from the stored rows of `lectqa_pooled.jsonl` (its single-video rows are the setting of `lectqa_frameslides.md`).

| setting | difference | metric | by question | by video |
|---|---|---|---:|---:|
| single video | links + reranker − plain top-8 | interval@3 | -5.01 [-6.82, -3.25] | -5.01 [-7.60, -2.24] |
| single video | links + reranker − plain top-8 | interval@8 | -2.69 [-4.32, -1.06] | -2.69 [-5.51, +0.53] (crosses 0) |
| single video | reranker, no links − plain top-8 | interval@3 | -2.57 [-4.13, -1.00] | -2.57 [-4.30, -0.95] |
| single video | reranker, no links − plain top-8 | interval@8 | -0.56 [-1.81, +0.69] | -0.56 [-1.79, +0.67] (crosses 0) |
| pooled | links + reranker − reranker, no links | interval@3 | -0.25 [-0.94, +0.38] | -0.25 [-1.29, +0.80] (crosses 0) |
| pooled | links + reranker − reranker, no links | interval@8 | -1.44 [-2.50, -0.38] | -1.44 [-2.79, +0.00] (crosses 0) |
| pooled | reranker, no links − plain top-8 | interval@3 | -5.82 [-7.13, -4.57] | -5.82 [-8.47, -3.57] |
| pooled | reranker, no links − plain top-8 | interval@8 | -5.44 [-6.70, -4.26] | -5.44 [-8.06, -3.40] |
| pooled | links + reranker − plain top-8 | interval@3 | -6.07 [-7.51, -4.69] | -6.07 [-8.75, -3.71] |
| pooled | links + reranker − plain top-8 | interval@8 | -6.88 [-8.39, -5.44] | -6.88 [-10.12, -4.03] |
| pooled | links only − plain top-8 | interval@3 | -0.63 [-1.19, -0.13] | -0.63 [-1.21, -0.06] |
| pooled | links only − plain top-8 | interval@8 | -0.38 [-1.06, +0.25] | -0.38 [-1.13, +0.54] (crosses 0) |

## All transcripts together

95 videos: **72,926 o200k tokens** in total (per video 142–1544, median 780). The pooled corpus of `lectqa_pooled.md` also fits in one context window. `lectqa_audit.md` gives the per-video minimum as 140; the 2-token difference was not traced.

## Timestamp hand-check sheet

`results/external/lectqa_timestamp_check.csv`: 34 empty-window questions, 16 drawn from the other 1564 stamps that pass the checks, 25 drawn from the 878 with an ambiguous format (seed `lectqa-timestamp-check-20261112`). The last five columns are for the checker.

## Provenance

`python scripts/adapters/lectqa_vid.py recheck --max-cost 0` · commit `348507c` · inputs `lectqa_open_modes.jsonl`, `lectqa_attribution.jsonl`, `lectqa_pooled.jsonl` · embedder for the chunk check: T1 replica's (`open_ended.T1_EMBEDDER`).
