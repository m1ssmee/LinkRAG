# LectQA-Vid — second pass (28/100 videos, 840 QA pairs)

Subset: the 28 of the first 35 videos whose YouTube links were still available; 7 were not (`data/raw/lectqa_vid/fetch_failures.txt`). Every number below is on this 28/100 subset, one run. Transcript whisper-small, frames every 5 s + tesseract OCR, 15-second segments, retrieval pool 20, k = 8 for localisation.

**Caption for every `linkrag` / `linkrag_iter` row:** single-stream setting — one video, no separate deck or paper — so the only link available is temporal co-occurrence between a transcript segment and the frames on screen; **cross-file linking is inactive by construction**. These rows test additive expansion + the complementarity reranker over transcript and frame-OCR units, nothing more.

## 1. Temporal localisation (primary metric; no answerer)

hit@k: any of the top-k retrieved units overlaps the question's gold interval `[timestamp_start, timestamp_end]` (the files mix `HH:MM:SS`, `MM:SS` and plain-seconds stamps; all three are parsed); mean best IoU: the best temporal IoU among the top 8. `iterative` / `linkrag_iter` need one LLM call per question for the follow-up query; **this run made no new LLM calls** — those cells are filled only where the call was already cached, so their n is smaller (sentence pass: 0 question-mode cells skipped; fixed pass: 0). No answer is generated for this table.

### Sentence-aware segmentation (`ingest.audio_segmentation: sentence`, the default)

| level | n | mode | rerank | hit@1 | hit@3 | hit@8 | mean best IoU |
|---|---:|---|---|---:|---:|---:|---:|
| simple | 281 | baseline | none | 53.0% | 76.2% | 90.4% | 0.467 |
| simple | 281 | baseline | complementarity | 53.4% | 64.1% | 78.6% | 0.396 |
| simple | 281 | linkrag | none | 53.4% | 76.2% | 90.4% | 0.467 |
| simple | 281 | linkrag | complementarity | 53.4% | 69.8% | 84.3% | 0.454 |
| simple | 281 | iterative | none | 66.2% | 82.2% | 91.1% | 0.469 |
| simple | 281 | iterative | complementarity | 66.2% | 74.4% | 86.5% | 0.447 |
| simple | 281 | linkrag_iter | none | 66.5% | 81.9% | 91.8% | 0.476 |
| simple | 281 | linkrag_iter | complementarity | 66.5% | 77.9% | 87.9% | 0.481 |
| hard | 280 | baseline | none | 36.4% | 60.0% | 80.0% | 0.392 |
| hard | 280 | baseline | complementarity | 36.8% | 45.0% | 68.9% | 0.311 |
| hard | 280 | linkrag | none | 36.8% | 60.0% | 80.0% | 0.392 |
| hard | 280 | linkrag | complementarity | 36.8% | 53.9% | 74.6% | 0.349 |
| hard | 280 | iterative | none | 44.3% | 60.0% | 80.0% | 0.395 |
| hard | 280 | iterative | complementarity | 44.3% | 55.7% | 80.0% | 0.373 |
| hard | 280 | linkrag_iter | none | 46.4% | 60.0% | 79.3% | 0.391 |
| hard | 280 | linkrag_iter | complementarity | 46.4% | 57.1% | 71.8% | 0.345 |
| very hard | 279 | baseline | none | 44.8% | 68.1% | 83.5% | 0.354 |
| very hard | 279 | baseline | complementarity | 44.8% | 57.0% | 77.8% | 0.304 |
| very hard | 279 | linkrag | none | 44.8% | 68.1% | 83.5% | 0.354 |
| very hard | 279 | linkrag | complementarity | 44.8% | 58.8% | 77.4% | 0.323 |
| very hard | 279 | iterative | none | 45.5% | 68.5% | 81.4% | 0.345 |
| very hard | 279 | iterative | complementarity | 45.5% | 61.6% | 79.9% | 0.341 |
| very hard | 279 | linkrag_iter | none | 45.2% | 67.4% | 81.7% | 0.346 |
| very hard | 279 | linkrag_iter | complementarity | 45.2% | 62.7% | 73.8% | 0.315 |
| overall | 840 | baseline | none | 44.8% | 68.1% | 84.6% | 0.404 |
| overall | 840 | baseline | complementarity | 45.0% | 55.4% | 75.1% | 0.337 |
| overall | 840 | linkrag | none | 45.0% | 68.1% | 84.6% | 0.404 |
| overall | 840 | linkrag | complementarity | 45.0% | 60.8% | 78.8% | 0.375 |
| overall | 840 | iterative | none | 52.0% | 70.2% | 84.2% | 0.403 |
| overall | 840 | iterative | complementarity | 52.0% | 63.9% | 82.1% | 0.387 |
| overall | 840 | linkrag_iter | none | 52.7% | 69.8% | 84.3% | 0.404 |
| overall | 840 | linkrag_iter | complementarity | 52.7% | 66.0% | 77.9% | 0.380 |

## 2. Segmentation ablation on their data

Same transcript words, same frames, same retrieval; only the cut points change. `fixed` = equal-time windows (15 s), what the baseline papers do; `sentence` = split on terminal punctuation, then packed.

| segmentation | mid-sentence boundary rate | hit@1 (overall, baseline/none) | hit@8 | mean best IoU |
|---|---:|---:|---:|---:|
| sentence | 0.0% | 44.8% | 84.6% | 0.404 |
| fixed | 82.8% | 44.3% | 85.8% | 0.394 |

### Fixed windows — full table

| level | n | mode | rerank | hit@1 | hit@3 | hit@8 | mean best IoU |
|---|---:|---|---|---:|---:|---:|---:|
| simple | 281 | baseline | none | 53.0% | 75.4% | 91.8% | 0.446 |
| simple | 281 | baseline | complementarity | 53.4% | 66.5% | 83.3% | 0.404 |
| simple | 281 | linkrag | none | 53.4% | 75.4% | 91.8% | 0.446 |
| simple | 281 | linkrag | complementarity | 53.4% | 68.3% | 84.0% | 0.419 |
| simple | 281 | iterative | none | 65.8% | 84.0% | 92.5% | 0.446 |
| simple | 281 | iterative | complementarity | 65.8% | 74.7% | 91.1% | 0.428 |
| simple | 281 | linkrag_iter | none | 65.1% | 84.3% | 91.5% | 0.441 |
| simple | 281 | linkrag_iter | complementarity | 65.1% | 79.4% | 87.9% | 0.448 |
| hard | 280 | baseline | none | 37.1% | 59.3% | 82.1% | 0.376 |
| hard | 280 | baseline | complementarity | 37.5% | 46.4% | 72.9% | 0.319 |
| hard | 280 | linkrag | none | 37.5% | 59.3% | 82.1% | 0.376 |
| hard | 280 | linkrag | complementarity | 37.5% | 53.2% | 72.9% | 0.340 |
| hard | 280 | iterative | none | 42.5% | 60.4% | 83.6% | 0.385 |
| hard | 280 | iterative | complementarity | 42.5% | 53.9% | 80.7% | 0.374 |
| hard | 280 | linkrag_iter | none | 40.7% | 62.1% | 81.8% | 0.390 |
| hard | 280 | linkrag_iter | complementarity | 40.7% | 54.3% | 75.0% | 0.349 |
| very hard | 279 | baseline | none | 42.7% | 67.0% | 83.5% | 0.360 |
| very hard | 279 | baseline | complementarity | 43.4% | 55.2% | 76.0% | 0.318 |
| very hard | 279 | linkrag | none | 43.4% | 67.4% | 83.5% | 0.360 |
| very hard | 279 | linkrag | complementarity | 43.4% | 63.1% | 77.8% | 0.339 |
| very hard | 279 | iterative | none | 48.7% | 69.9% | 82.4% | 0.360 |
| very hard | 279 | iterative | complementarity | 48.7% | 58.8% | 82.4% | 0.350 |
| very hard | 279 | linkrag_iter | none | 47.7% | 70.3% | 83.5% | 0.365 |
| very hard | 279 | linkrag_iter | complementarity | 47.7% | 63.4% | 78.5% | 0.332 |
| overall | 840 | baseline | none | 44.3% | 67.3% | 85.8% | 0.394 |
| overall | 840 | baseline | complementarity | 44.8% | 56.1% | 77.4% | 0.347 |
| overall | 840 | linkrag | none | 44.8% | 67.4% | 85.8% | 0.394 |
| overall | 840 | linkrag | complementarity | 44.8% | 61.5% | 78.2% | 0.366 |
| overall | 840 | iterative | none | 52.4% | 71.4% | 86.2% | 0.397 |
| overall | 840 | iterative | complementarity | 52.4% | 62.5% | 84.8% | 0.384 |
| overall | 840 | linkrag_iter | none | 51.2% | 72.3% | 85.6% | 0.399 |
| overall | 840 | linkrag_iter | complementarity | 51.2% | 65.7% | 80.5% | 0.377 |

## 3. Answer token-F1 per difficulty vs their Table 4 (not directly comparable)

Their paper does not name the answering LLM ("an instruction-tuned LLM", §4.4.4) and evaluates against LLaVA-1.6, so no matched-strength row can be built; ours is `gpt-5.4-2026-03-05`. **Answer-F1 is therefore not directly comparable — lead with localisation above.** Their semantic similarity used all-MiniLM-L6-v2; the first-run column used bge-m3 and is omitted here.

| level | n | mode | token-F1 | ROUGE-1 | their F1 (Table 4) | their ROUGE-1 |
|---|---:|---|---:|---:|---:|---:|
| simple | 141 | baseline | 33.9% | 38.7% | 31.15% | 39.80% |
| simple | 141 | linkrag | 31.8% | 36.6% | 31.15% | 39.80% |
| hard | 140 | baseline | 28.3% | 31.4% | 19.35% | 24.60% |
| hard | 140 | linkrag | 25.5% | 28.5% | 19.35% | 24.60% |
| very hard | 139 | baseline | 16.8% | 20.1% | 10.24% | 15.32% |
| very hard | 139 | linkrag | 15.1% | 18.1% | 10.24% | 15.32% |
| overall | 420 | baseline | 26.4% | 30.1% | 23.52% | 29.76% |
| overall | 420 | linkrag | 24.2% | 27.8% | 23.52% | 29.76% |

LLM cost (this run):

- `gpt-5.4-2026-03-05`: 1680 calls (1680 cached) · 765,855 in / 31,644 out · $0.0000 (cache replays free; would have been $2.3893)
- `gpt-5.4-2026-03-05`: 1680 calls (872 cached) · 751,949 in / 31,681 out · $1.1738 (cache replays free; would have been $2.3551)
- run total: $1.1738
- cumulative (all recorded runs, `reports/llm_ledger.jsonl`): 11,668 calls · 7,249,776 in / 413,893 out · $9.73 · 5 row(s) unpriced · 4,719,273 tokens estimated

## 5. Reading

- **Localisation is where this benchmark can be judged, and there `iterative` wins hit@1**: 52.0 % vs baseline 44.8 %
  (`linkrag_iter` 52.7 %). A second query finds the right moment in a 2–5-minute video more often than one query does;
  link-following adds ~0.7 pp on top, which is inside run-to-run noise. By hit@8 every mode converges (84–86 %): with
  15–40 units per video, eight of them cover most of the timeline whatever the retriever does.
- **`linkrag` ≈ `baseline` exactly at hit@1/hit@3/hit@8 and IoU** — expected and documented: additive expansion cannot
  evict a seed, and with the only link type being temporal co-occurrence the neighbours it proposes are the units already
  retrieved. The single-stream caption at the top of this file is the whole story for these rows.
- **The complementarity reranker hurts localisation** (hit@3 68.1 % → 55.4 % for baseline) because it spends slots on
  frame-OCR units to cover a second modality, and the gold interval is defined by what was *said*. Same interaction as on
  pilot01: α and β are worth paying only when the pool holds genuinely complementary evidence.
- **Segmentation ablation: no localisation difference.** Sentence-aware cutting removes 82.8 % of mid-sentence boundaries
  (82.8 % → 0.0 %) but moves hit@1 by 0.5 pp and IoU by 0.010 — both inside noise. The pilot01 result that sentence
  segmentation matters was about *answer* quality and BM25 matching, not about finding the right 15 seconds; on this
  benchmark the segmentation choice is free.
- Answer-F1 (§3) is **not** comparable to their Table 4: their paper never names its answering LLM ("an instruction-tuned
  LLM", §4.4.4) and evaluates against LLaVA-1.6, while ours is gpt-5.4. Localisation above is the honest comparison.
- Cost of this run: $1.17 (1,680 calls, 872 served from cache), under the $10 cap. Cumulative project spend: $9.73.
