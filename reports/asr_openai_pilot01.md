# ASR backend comparison — hsieh.mp3

`ingest.asr_backend: openai` (whisper-1, word timestamps, 2 chunk(s)) against the frozen local transcript (faster-whisper small, int8, CPU). Vocabulary prompt: on. Requested 2026-09-22T17:06:54Z.

| | local (frozen) | openai whisper-1 |
|---|---|---|
| words | 3247 | 3253 |
| wall time | 476 s (recorded 2026-09-06, plain run) | 94 s |
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
- cumulative (all recorded runs, `reports/llm_ledger.jsonl`): 15,809 calls · 9,225,900 in / 544,345 out · $10.68 · 5 row(s) unpriced · 4,719,273 tokens estimated

## Reading

**whisper-1 is materially better on the terms this project cares about, and the
`type -k` regression disappears.**

| term | local (faster-whisper small, frozen) | openai whisper-1 |
|---|---:|---:|
| `ingest` (correct) | 17 | **27** |
| `interest` (mis-heard *ingest*) | 16 | **6** |
| `top-K` | 5 | **16** |
| `type -k` (the open regression) | 4 | **0** |
| NoScope | 5 | 5 |
| YOLO | 3 | 3 |

- The open ASR issue recorded in `DESIGN.md` ("Open: `type -k` regression", cause
  unknown since 2026-09-06) **does not occur with whisper-1**, with the same slide
  vocabulary prompt. It was a property of the local model, not of the prompt.
- Speed: **94 s** wall for 22.9 minutes of audio (14.6x realtime) against 476 s for
  the local CPU run — and most of the 94 s is upload.
- Cost: **$0.1371** for the talk (22.9 billed minutes at $0.006/min), inside the
  $1.00 cap the script enforces before it starts.
- Chunking was exercised for real: the 21.9 MB mp3 exceeds the 20 MB upload limit, so
  it was split into 2 pieces at a silence point (694.0 s). The merged transcript spans
  0.9-1368.7 s with no gap at the seam — the words around 694 s read as continuous
  prose ("...instead of all the objects | So how do we integrate clustering...").
- 3,253 words against the local transcript's 3,247.

**The frozen pilot01 transcript was not replaced.** This run wrote
`data/processed/transcripts/hsieh.openai.json`; `hsieh.frozen.json` is untouched, so
every recorded pilot01 number still refers to the transcript it was measured on.
Switching pilot01 to whisper-1 would invalidate the alignment, deictic and regression
history and is a separate decision.

**For the extended dataset** (`scripts/dataset/candidate.py`), `ingest.asr_backend:
openai` is the right default: ~$0.36 per hour-long lecture and 15x realtime instead of
~20 minutes of CPU per lecture, with better terminology out of the box.
