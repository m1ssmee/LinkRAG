# LectQA-Vid — target T1 — first run

28 of 100 videos · 840 QA pairs per mode · modes ['baseline', 'linkrag'] · k=4 · answerer `gpt-5.4-2026-03-05` temperature=0.0 · repeats 1 · transcript: whisper-small · frames every 5s, OCR tesseract (their pipeline used Whisper large-v3 and Gemini captions)

Their split and 1,000-pair evaluation subset are unpublished; this is every QA pair of the videos processed. Semantic similarity here is bge-m3 cosine; they do not name their embedder.

**Read with care.** The answerer here (`gpt-5.4-2026-03-05`) is far stronger than the open models T1 evaluated with, so a gap to their numbers is mostly the LLM, not retrieval; the baseline-vs-linkrag rows are the retrieval comparison. Videos are 2–5 minutes, so top-4 of ~15–40 units already covers much of each video. MCQ distractors are weak (see accuracy).


> *Corrected 2026-09-23:* the "their" columns below previously used values that do not match the published paper (Tables 4–5, read as images from the CMC full-text HTML). Old → published: open-ended F1 simple 31.15→29.47, hard 19.35→24.38, very hard 10.24→16.72; ROUGE-1 39.80→36.82, 24.60→30.94, 15.32→22.51; similarity 0.75/0.68/0.51/0.71 → 77.23/74.56/71.48/74.42 %; MCQ accuracy 68.40/56.30/44.20/56.30 → 57.29/52.34/50.67/53.43 %. Overall F1 23.52 and ROUGE-1 29.76 were right. Our own numbers are unchanged.

## Open-ended (their Table 4 / 6)

| level | n | mode | token-F1 | ROUGE-1 | sim (bge-m3) | their F1 | their ROUGE-1 | their sim |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| simple | 141 | baseline | 33.9% | 38.7% | 0.76 | 29.47% | 36.82% | 0.77 |
| simple | 141 | linkrag | 31.8% | 36.6% | 0.75 | 29.47% | 36.82% | 0.77 |
| hard | 140 | baseline | 28.3% | 31.4% | 0.73 | 24.38% | 30.94% | 0.75 |
| hard | 140 | linkrag | 25.5% | 28.5% | 0.70 | 24.38% | 30.94% | 0.75 |
| very hard | 139 | baseline | 16.8% | 20.1% | 0.65 | 16.72% | 22.51% | 0.71 |
| very hard | 139 | linkrag | 15.1% | 18.1% | 0.63 | 16.72% | 22.51% | 0.71 |
| overall | 420 | baseline | 26.4% | 30.1% | 0.71 | 23.52% | 29.76% | 0.74 |
| overall | 420 | linkrag | 24.2% | 27.8% | 0.69 | 23.52% | 29.76% | 0.74 |

## MCQ (their Table 5)

| level | n | mode | accuracy | their accuracy |
|---|---:|---|---:|---:|
| simple | 140 | baseline | 93.6% | 57.29% |
| simple | 140 | linkrag | 93.6% | 57.29% |
| hard | 140 | baseline | 95.7% | 52.34% |
| hard | 140 | linkrag | 95.0% | 52.34% |
| very hard | 140 | baseline | 99.3% | 50.67% |
| very hard | 140 | linkrag | 99.3% | 50.67% |
| overall | 420 | baseline | 96.2% | 53.43% |
| overall | 420 | linkrag | 96.0% | 53.43% |

LLM cost (this run):

- `gpt-5.4-2026-03-05`: 1680 calls (1318 cached) · 436,118 in / 39,474 out · $0.3443 (cache replays free; would have been $1.6824)
- run total: $0.3443
- cumulative (all recorded runs, `reports/llm_ledger.jsonl`): 8,188 calls · 5,689,499 in / 348,192 out · $8.42 · 5 row(s) unpriced · 4,719,273 tokens estimated

## Reading (28/100 videos, 840 QA pairs)

- **Open-ended**: baseline token-F1 **26.4 %** / ROUGE-1 30.1 % / sim 0.71 vs their 23.52 % / 29.76 % / 74.42 % (their sim is all-MiniLM-L6-v2, ours bge-m3: not comparable) — level with
  their number, not above it, once n grows from 3 videos (40.6 %) to 28. The 3-video figure was a small-sample artefact.
- **MCQ**: **96.2 %** vs their 53.43 % — mostly the answerer (gpt-5.4 vs their open model) and weak distractors; this column
  says little about retrieval.
- **linkrag is 2 points *below* baseline on open-ended** (24.2 % vs 26.4 %) at every level. On a single 2–5-minute video the only
  link is temporal co-occurrence, and expansion + complementarity admit frame-OCR units that displace transcript units the
  reference answers are worded from. This is the setting LinkRAG was not built for (no cross-file structure), reported as such.
- Every number here is on 28 of 100 videos (7 of the first 35 links are unavailable; `data/raw/lectqa_vid/fetch_failures.txt`),
  one run, their split unpublished. Repeats and the remaining 65 videos are a cost/time decision (~$1 per 28 videos in
  answerer calls; whisper on CPU is the bottleneck at ~4 min per video).
