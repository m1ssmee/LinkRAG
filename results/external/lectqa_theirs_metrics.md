# LectQA-Vid — their metric suite (T1 eqs. 29–36)

28 videos · answerer `gpt-5.4-2026-03-05` (cache-only replay of stored answers; 0 cache misses skipped) · metrics `linkrag.eval.lectqa_metrics`: set-based token P/R/F1, nltk BLEU-4 (no smoothing) and METEOR (alpha 0.9, beta 3, gamma 0.5), ROUGE-1 = unigram recall, similarity = all-MiniLM-L6-v2 cosine, all in %.

**Not a like-for-like comparison.** Their answering LLM is unnamed (§4.4.4). Ours is far stronger, and their split and 1,000-pair evaluation subset are unpublished, so these are all QA pairs of the videos processed. The baseline-vs-linkrag rows are the retrieval comparison.

## Open-ended (their Table 4)

| level | mode | n | F1 | Sim | BLEU | METEOR | R1 | token P | token R |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| simple | baseline | 141 | 39.88 | 65.10 | 11.45 | 38.35 | 43.12 | 38.48 | 43.12 |
| simple | linkrag | 141 | 37.66 | 62.70 | 10.09 | 35.17 | 39.95 | 37.22 | 39.95 |
| simple | **theirs (Table 4)** | — | 29.47 | 77.23 | 8.61 | 38.15 | 36.82 | — | — |
| hard | baseline | 140 | 32.38 | 61.14 | 7.14 | 31.48 | 37.42 | 30.24 | 37.42 |
| hard | linkrag | 140 | 29.29 | 56.81 | 5.56 | 27.70 | 33.68 | 27.46 | 33.68 |
| hard | **theirs (Table 4)** | — | 24.38 | 74.56 | 5.29 | 34.71 | 30.94 | — | — |
| very hard | baseline | 139 | 20.95 | 51.10 | 0.53 | 20.14 | 27.86 | 17.47 | 27.86 |
| very hard | linkrag | 139 | 18.98 | 48.08 | 0.63 | 17.80 | 25.24 | 15.75 | 25.24 |
| very hard | **theirs (Table 4)** | — | 16.72 | 71.48 | 2.83 | 24.36 | 22.51 | — | — |
| overall | baseline | 420 | 31.12 | 59.14 | 6.40 | 30.03 | 36.17 | 28.78 | 36.17 |
| overall | linkrag | 420 | 28.69 | 55.90 | 5.45 | 26.93 | 32.99 | 26.86 | 32.99 |
| overall | **theirs (Table 4)** | — | 23.52 | 74.42 | 5.58 | 32.41 | 29.76 | — | — |

## MCQ (their Table 5)

| level | mode | n | ACC | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|---:|
| simple | baseline | 140 | 93.57 | 25.00 | 23.39 | 24.17 |
| simple | linkrag | 140 | 93.57 | 25.00 | 23.39 | 24.17 |
| simple | **theirs (Table 5)** | — | 57.29 | — | — | — |
| hard | baseline | 140 | 95.71 | 33.33 | 31.90 | 32.60 |
| hard | linkrag | 140 | 95.00 | 25.00 | 23.75 | 24.36 |
| hard | **theirs (Table 5)** | — | 52.34 | — | — | — |
| very hard | baseline | 140 | 99.29 | 50.00 | 49.64 | 49.82 |
| very hard | linkrag | 140 | 99.29 | 50.00 | 49.64 | 49.82 |
| very hard | **theirs (Table 5)** | — | 50.67 | — | — | — |
| overall | baseline | 420 | 96.19 | 25.00 | 24.05 | 24.51 |
| overall | linkrag | 420 | 95.95 | 25.00 | 23.99 | 24.48 |
| overall | **theirs (Table 5)** | — | 53.43 | — | — | — |

**MCQ caveat: not reportable in this form.** The gold option is `A` for 840 of 840 questions. The published file lists the correct answer first for 1,484 of 1,489 MCQs, and options were presented in that stored order, so accuracy rewards any preference for the first option. Macro P/R/F1 are degenerate with one gold class. Their Table 5 P/R/F1 ≈ ACC suggests shuffled options there; the paper does not say. The valid measurement is a re-run with options shuffled under a fixed seed, which needs new LLM calls.

Counts: mcq baseline 420, mcq linkrag 420, open baseline 420, open linkrag 420; cache misses skipped: 0.

LLM cost (this run):

- `gpt-5.4-2026-03-05`: 1680 calls (1680 cached) · 436,118 in / 39,474 out · $0.0000 (cache replays free; would have been $1.6824)
- run total: $0.0000
- cumulative (all recorded runs, `reports/llm_ledger.jsonl`): 23,112 calls · 24,171,542 in / 878,369 out · $17.81 · 6 row(s) unpriced · 14,412,524 tokens estimated
