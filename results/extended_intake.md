# Extended-dataset intake log

One entry per candidate lecture put through `scripts/dataset/` — ingest, links, the
relatedness gate, modality redundancy, automated question proposal, machine gold
verification, audit sheet. The verdict is advisory (`dataset.intake.max_deck_to_transcript`);
the numbers are the record.

| # | corpus | deck→transcript | verdict | questions kept | cross-modal surviving | cost |
|---:|---|---:|---|---:|---:|---:|
| 0 | pilot01-w1 | 86.5 % | **REJECT** | 25/30 | **0** | $7.12 |

---

## Entry 0 — `pilot01-w1` (2026-09-23)

The pilot corpus re-ingested as a *new candidate*: same three sources
(`osdi18_slides_hsieh.pdf`, `osdi18-hsieh.pdf`, `hsieh.mp3`) with the **whisper-1
transcript** instead of the frozen faster-whisper one. Run end to end through the
intake to exercise the pipeline on a corpus whose answer is already known.

Config `configs/pilot01-w1.yaml`, everything under `data/processed/pilot01-w1/`.
**The frozen pilot01 corpus, links, gold and history are untouched** — verified: its
index, `links.jsonl` and `transcripts/hsieh.frozen.json` are unchanged, and its
manifest is still `2f3b35f27e86caf8`.

### Pipeline

| step | command | result |
|---|---|---|
| ingest | `scripts/ingest.py --config configs/pilot01-w1.yaml` | **209 units** — 111 text, 47 figure, 51 audio · manifest `f59cbae3b8757659` |
| links | `scripts/build_links.py` | **308 links** — audio_slide 51, deictic 123, figure_text 103, same_slide 31 |
| relatedness gate | `scripts/gate_links.py` | audio_slide **96.1 %**, deictic **80.5 %**, figure_text 100 %, same_slide 100 % |
| redundancy | `scripts/dataset/redundancy.py` | see below |
| question proposal | `scripts/dataset/propose_questions.py` (new) | **30 questions** by modality diff — 15 single-source, 15 cross-source |
| gold verification | `scripts/eval/verify_gold.py` | **25/30 kept**, 49 gold units, **0 cross-modal** |
| audit sheet | `scripts/eval/audit_sample.py` | 31 of 131 unit verdicts sampled → `reports/audit_sheet_pilot01-w1.csv` (unfilled) |

### Redundancy and the verdict

| pair | pilot01-w1 | pilot01 (local transcript) |
|---|---:|---:|
| deck→transcript | **86.5 %** | 94.2 % |
| transcript→deck | 64.9 % | 61.1 % |
| transcript→paper | 42.7 % | 38.9 % |
| deck→paper | 48.1 % | 46.2 % |

**Verdict: REJECT** — deck→transcript 86.5 % against the 65 % bar. The better
transcript moves the number by 7.7 points (the speaker's words now match the slides
less literally, because fewer of them are mis-transcribed), but the lecture is still
one where the speaker narrates the deck. A better ASR does not turn a redundant
lecture into a complementary one.

### The result that matters: 0 of 15 proposed cross-modal questions survived

The proposer built 15 questions designed to need a fact from two different sources
(topically close units, each carrying vocabulary exclusive to its own source).
Verification kept 25 of 30 questions overall but **relabelled every cross-modal one**:

| outcome | n | what the modality-only runs found |
|---|---:|---|
| `single_modality` | 8 | two or three sources each answered it alone |
| `audio_only` / `paper_only` / `slides_only` | 4 | one source answered it alone |
| dropped | 3 | not answerable even from all three sources (P17, P18, P26) |
| **stayed cross-modal** | **0** | — |

This is the same finding as pilot01 (4/16 survived there, with hand-written
questions) only sharper, and it is a property of the corpus, not of the proposer: on a
lecture where the deck, the talk and the paper all state the same facts, a question
needing two of them cannot be constructed. It is the quantitative case for the intake
criterion, and the reason pilot01-family corpora cannot carry the cross-modal claim.

The three dropped questions are a proposer weakness worth recording: each asks about a
**figure's content** ("In the Figure 10 discussion…", "what speedup over
state-of-the-art…") where the exclusive term came from figure OCR that no source
states in prose. The proposer should not build questions from OCR-only terms; not
fixed here.

### Steps that needed intervention

Three, all found by running it and all fixed in code — none by hand-editing data:

1. **1 audio unit instead of 51.** whisper-1's word timestamps carry no punctuation
   (3 of 3,260 words ended in `.?!`), and the sentence splitter cuts on punctuation, so
   the entire talk became one segment. Fixed: request `timestamp_granularities[]` for
   **word and segment**, and re-attach punctuation from the segment texts onto the word
   stream (`asr_openai.restore_punctuation`) — 209 of 3,260 words now end a sentence.
   4 tests, including one asserting both granularities are requested.
2. **Manifest hash collided with pilot01's.** Same source files and unit counts gave
   `2f3b35f27e86caf8` for both corpora despite different transcripts — a gold set
   verified on one would have silently passed its manifest check against the other.
   Fixed: `build_manifest(..., derived=...)` hashes the frozen transcript that was
   actually read or written; pilot01-w1 is `f59cbae3b8757659`. Test added.
3. **No question-proposal script existed** — pilot01's 24 questions were written by
   hand. Wrote `scripts/dataset/propose_questions.py` implementing the documented
   modality-diff method. This was the last manual step in the intake; the chain is now
   unattended end to end.

### Cost

| step | model | cost |
|---|---|---:|
| transcription (whisper-1, 22.9 min, 2 chunks) | whisper-1 | $0.137 |
| relatedness gate | gpt-4.1-mini | $0.158 |
| redundancy (622 sentence judgements) | gpt-4.1-mini | $0.952 |
| question proposal (30 calls) | gpt-5.4 | $0.058 |
| gold verification | gpt-5.4 (answerer) + gpt-4.1-mini (judge) | $5.821 |
| **total** | | **$7.13** |

Gold verification is 82 % of it, and 97 % of *that* is the answerer doing four
full-context runs per question (~18k tokens each, 120 calls, 2.2M input tokens).
Cheaper options for the next candidate, in order of preference: run the modality-only
checks with the judge model instead of the answerer (~10× cheaper, but changes the
protocol and should be validated against one corpus verified both ways); or cut the
proposal to ~15 questions. **A full-size candidate costs about $7 at the current
protocol** — budget for it before starting one.
