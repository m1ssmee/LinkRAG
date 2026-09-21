# LectQA-Vid — target T1 — first run

3 of 100 videos · 90 QA pairs per mode · modes ['baseline', 'linkrag'] · k=4 · answerer `gpt-5.4-2026-03-05` temperature=0.0 · repeats 1 · transcript: whisper-small · frames every 5s, OCR tesseract (their pipeline used Whisper large-v3 and Gemini captions)

Their split and 1,000-pair evaluation subset are unpublished; this is every QA pair of the videos processed. Semantic similarity here is bge-m3 cosine; they do not name their embedder.

**Read with care.** The answerer here (`gpt-5.4-2026-03-05`) is far stronger than the open models T1 evaluated with, so a gap to their numbers is mostly the LLM, not retrieval; the baseline-vs-linkrag rows are the retrieval comparison. Videos are 2–5 minutes, so top-4 of ~15–40 units already covers much of each video. MCQ distractors are weak (see accuracy).

## Open-ended (their Table 4 / 6)

| level | n | mode | token-F1 | ROUGE-1 | sim (bge-m3) | their F1 | their ROUGE-1 | their sim |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| simple | 15 | baseline | 55.1% | 61.0% | 0.87 | 31.15% | 39.80% | 0.75 |
| simple | 15 | linkrag | 46.1% | 52.5% | 0.81 | 31.15% | 39.80% | 0.75 |
| hard | 15 | baseline | 41.1% | 43.9% | 0.82 | 19.35% | 24.60% | 0.68 |
| hard | 15 | linkrag | 34.6% | 37.1% | 0.78 | 19.35% | 24.60% | 0.68 |
| very hard | 15 | baseline | 25.5% | 28.5% | 0.75 | 10.24% | 15.32% | 0.51 |
| very hard | 15 | linkrag | 24.5% | 27.3% | 0.74 | 10.24% | 15.32% | 0.51 |
| overall | 45 | baseline | 40.6% | 44.5% | 0.81 | 23.52% | 29.76% | 0.71 |
| overall | 45 | linkrag | 35.1% | 39.0% | 0.78 | 23.52% | 29.76% | 0.71 |

## MCQ (their Table 5)

| level | n | mode | accuracy | their accuracy |
|---|---:|---|---:|---:|
| simple | 15 | baseline | 100.0% | 68.40% |
| simple | 15 | linkrag | 100.0% | 68.40% |
| hard | 15 | baseline | 100.0% | 56.30% |
| hard | 15 | linkrag | 100.0% | 56.30% |
| very hard | 15 | baseline | 100.0% | 44.20% |
| very hard | 15 | linkrag | 100.0% | 44.20% |
| overall | 45 | baseline | 100.0% | 56.30% |
| overall | 45 | linkrag | 100.0% | 56.30% |

LLM cost (this run):

- `gpt-5.4-2026-03-05`: 180 calls (0 cached) · 43,075 in / 3,738 out · $0.1638
- run total: $0.1638
- cumulative (all recorded runs, `reports/llm_ledger.jsonl`): 6,508 calls · 5,253,381 in / 308,718 out · $8.07 · 5 row(s) unpriced · 4,719,273 tokens estimated
