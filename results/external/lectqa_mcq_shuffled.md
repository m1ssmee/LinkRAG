# LectQA-Vid — MCQ with shuffled options

1414 MCQs · 95 videos · answerer `gpt-5.4-mini-2026-03-17` temperature 0.0 · baseline retrieval (RRF dense + BM25, top-4) · options shuffled with seed `lectqa-mcq-shuffle-20260923` (per video and question) · one run.

Gold position after shuffling: A 357, B 365, C 356, D 336 (before: A in 1,484 of 1,489).

| level | n | ACC | Precision | Recall | F1 | theirs ACC (Table 5) |
|---|---:|---:|---:|---:|---:|---:|
| simple | 475 | 90.11 | 90.15 | 90.11 | 90.09 | 57.29 |
| hard | 470 | 96.38 | 96.40 | 96.40 | 96.39 | 52.34 |
| very hard | 469 | 99.57 | 99.58 | 99.57 | 99.57 | 50.67 |
| overall | 1414 | 95.33 | 95.34 | 95.33 | 95.33 | 53.43 |

**Answerable from the options alone.** A no-video answerer that picks the unique longest option scores simple 88.2, hard 95.1, very hard 100.0, overall 94.4 on these 1414 questions (audit (a)). The answerer picked the longest option for 1280/1414. The per-difficulty accuracy above follows the longest-option share, so this accuracy does not measure retrieval either.

**Run-to-run agreement** on a seeded 100-question subset, 3 repeats: identical answer in all repeats for 100/100; subset accuracy per repeat 91.0, 91.0, 91.0 (mean ± std 91.0 ± 0.0).


LLM cost (this run):

- `gpt-5.4-mini-2026-03-17`: 1614 calls (1614 cached) · 482,685 in / 6,504 out · $0.0000 (cache replays free; would have been $0.3913)
- run total: $0.0000
- cumulative (all recorded runs, `reports/llm_ledger.jsonl`): 27,727 calls · 25,543,821 in / 896,958 out · $18.20 · 6 row(s) unpriced · 14,412,524 tokens estimated
