# ASR backend comparison — hsieh.mp3

`ingest.asr_backend: openai` (whisper-1, word timestamps, 2 chunk(s)) against the frozen local transcript (faster-whisper small, int8, CPU). Vocabulary prompt: on. Requested 2026-09-22T17:32:37Z.

| | local (frozen) | openai whisper-1 |
|---|---|---|
| words | 3247 | 3260 |
| wall time | 476 s (recorded 2026-09-06, plain run) | 70 s |
| cost | $0 (CPU) | $0.1371 |
| audio billed | — | 22.9 min |

## Term table (the pilot01 ASR terms)

| term | local (frozen) | openai |
|---|---:|---:|
| ingest | 17 | 27 |
| interest (mis-heard ingest) | 16 | 6 |
| NoScope | 5 | 5 |
| YOLO | 3 | 3 |
| top-K | 5 | 16 |
| type -k (known regression) | 4 | 0 |

Local transcript: `data/processed/transcripts/hsieh.frozen.json` (unchanged — this run wrote `data/processed/transcripts/hsieh.openai.json`).


LLM cost (this run):

- `whisper-1`: 2 calls (0 cached) · 22.9 audio minutes · $0.1371
- run total: $0.1371
- cumulative (all recorded runs, `reports/llm_ledger.jsonl`): 15,811 calls · 9,225,900 in / 544,345 out · $10.82 · 5 row(s) unpriced · 4,719,273 tokens estimated
