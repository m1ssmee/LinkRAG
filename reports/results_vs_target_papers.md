# Results against the two target papers (draft)

Draft of 2026-09-25, for the mentor report now and the paper's results section later.

**Where the numbers come from.** Every number below is copied from one of four sources: `results/external/*.md`, `results/external/*.json`, `DESIGN.md` findings 12–16, and the target rows of `README.md`. Each is cited where it is used. The one exception is the total API spend at the end, which comes from `reports/llm_ledger.jsonl`. Where a number is not in those sources, the text says **TODO** and states what is missing; nothing is filled in by estimate.

**Two targets.**
- **T1:** Intra-Video Temporal-Aware RAG (Shafiq et al., *CMC* 88(2):96, 2026), evaluated on **LectQA-Vid**.
- **T2:** MaViLS (Anderer, Reich, Wölfel, *Interspeech 2024*), evaluated on the **MaViLS** alignment benchmark.

## Summary

*Table S. Headline comparison. The detailed tables and their sources are in §1 and §2.*
- *LectQA-Vid:* 300-question stratified open-ended subset (100 per difficulty), 93 videos, 3 repeats; answerer gpt-5.4-mini-2026-03-17 at temperature 0; T1's metric suite (eqs. 29–35); spend $3.34.
- *MaViLS:* the 9 test-half lectures with video; their F1 protocol; LLM-free, $0.
- *Their numbers:* as published.

| paper | dataset | their number | our number | protocol | verdict |
|---|---|---|---|---|---|
| T1, Intra-Video Temporal-Aware RAG | LectQA-Vid, open-ended | Table 4 overall F1 23.52 (their unnamed answerer, not comparable with ours) | replica of their pipeline **31.23 ± 0.06**; ours (iterative) **36.70 ± 0.16**; full transcript **38.74 ± 0.23** | one answerer for all three modes; their metrics | **Ours exceeds the replica of their pipeline**: +5.47 F1, 95 % CI [+3.80, +7.24], and the CI excludes zero at every difficulty. **The benchmark cannot measure retrieval value**: the full transcript beats both (+2.04 [+1.14, +2.99] over ours), and every transcript fits a 2,048-token window. |
| T1 | LectQA-Vid, MCQ | Table 5 accuracy 53.43 | 95.33 with shuffled options | baseline retrieval, one run | **Not a retrieval measure.** A no-video "pick the longest option" rule scores 94.4. |
| T2, MaViLS | MaViLS, 9 test lectures with video | all-features **0.80** on the same lectures (0.82 is their 20-lecture average) | **0.853** (image + frame OCR + speech-to-slide text, frames at each sentence's timestamp) | their F1, sentence granularity; test half run once per round | **Parity**: +0.055, 95 % CI [−0.050, +0.161] |

**Total project API spend:** $22.05 recorded across 55 priced ledger rows (2026-09-21 to 2026-09-23).
- Six more rows carry no price: five gpt-5.4 rows and one free-tier Groq row. So the true total is higher than $22.05 by an unrecorded amount.
- The ledger also does not credit OpenAI's automatic prompt-cache discount, so each priced row is an upper bound.
- All MaViLS work and all re-renders since 2026-09-24 cost $0.

---

## 1. LectQA-Vid: Intra-Video Temporal-Aware RAG (CMC 2026)

### Caveats first

- **Different answerer.** Their answering LLM is unnamed (§4.4.4; `lectqa_theirs_metrics.md`). Ours is gpt-5.4-mini-2026-03-17 in every mode. Our absolute numbers are therefore not comparable with their Table 4. Only the comparisons among our three modes, which share one answerer, are.
- **Different evaluation set.** Their 80/10/10 split and their 1,000-pair evaluation subset are not published (README target row).
  - We use the 95 of 100 videos still on YouTube (`lectqa_audit.md` §(c)), with our own faster-whisper transcripts (`lectqa_coverage.md`).
  - The open-ended comparison runs on a stratified 300-question subset.
  - The single run on the other 1,118 open-ended questions was not made: the dry run estimated $7.43 against the item's $3.50 cap (`lectqa_open_modes.md`, "Scope").
- **The baseline is our replica of their pipeline, not their code**, which is unpublished. The choices the paper leaves open are ours (§1b).
- **Their eq. 22 filter reads the gold timestamps**, an oracle. It is usable on only 160 of the 300 subset questions (§1e).
- **Faithfulness uses one judge run per check**, so it is *direction only*. It was corrected twice on 2026-09-24 (§1f).

### 1a. Their published numbers

*Table 1a. T1's published results, from the paper's tables (read as images from the CMC full-text HTML).*
- *Dataset and split:* LectQA-Vid, their 80/10/10 split, 1,000-pair evaluation subset (500 MCQ + 500 open-ended; README row); not published.
- *Answerer:* unnamed (§4.4.4). Retrieval k = 10 (their Table 3). The re-rank pool M and the context size L are not stated.
- *Protocol:* their eqs. 29–36, all in %. *Spend:* n/a (published figures).
- *Sources:* per-difficulty open-ended rows from `lectqa_open_modes.md` (T1 Table 4 rows); MCQ from `lectqa_mcq_shuffled.md` (Table 5); Tables 6 and 7 from the README T1 row.

| level | F1 | Sim | BLEU | METEOR | ROUGE-1 | MCQ accuracy |
|---|---:|---:|---:|---:|---:|---:|
| simple | 29.47 | 77.23 | 8.61 | 38.15 | 36.82 | 57.29 |
| hard | 24.38 | 74.56 | 5.29 | 34.71 | 30.94 | 52.34 |
| very hard | 16.72 | 71.48 | 2.83 | 24.36 | 22.51 | 50.67 |
| **overall** | **23.52** | **74.42** | **5.58** | **32.41** | **29.76** | **53.43** |

Two more of their numbers, both overall:
- **Table 6:** their multimodal RAG without timestamps scores F1 14.80 and similarity 61.00.
- **Table 7 ablation:** without the temporal filter, F1 falls to 11.40.

On what they leave unnamed: the answering LLM is not named. The top-k is given (k = 10, Table 3), but M and L, the merge gap, the cross-encoder and the index layout are not (`lectqa_open_modes.md`). An older sentence in `lectqa_v2.md` says the paper "never states its top-K". That was written before Table 3 was read, and this draft supersedes it.

### 1b. Our replication of their pipeline

**What was implemented** (`lectqa_open_modes.md`, "baseline"). Values marked "Table 3" are theirs; the rest are our choices where the paper is silent.

- **Chunking (§4.2):**
  - whisper sentences merged while adjacent cosine > 0.75 (θsem) and the time gap < 5.0 s (δgap);
  - near-duplicates above cosine 0.9 (δdup) dropped;
  - frame OCR overlapping speech by more than 2.0 s (θt) suppressed.
- **Embedder:** BAAI/bge-base-en-v1.5, mean-pooled (their "BGE base").
- **Retrieval:** top-10 (k) above cosine 0.6 (θr).
- **Temporal filter, eq. 22:** keep chunks overlapping the question's annotated interval ± 3.0 s (Δt).
- **Merge of consecutive results:** gap < 5.0 s.
- **Cross-encoder re-rank:** cross-encoder/ms-marco-MiniLM-L-6-v2 over the top-8 (M), then the top-4 (L) in temporal order.

**Where we filled gaps:**
- the choice of cross-encoder;
- M = 8 and L = 4;
- the merge gap;
- searching each video's own chunks instead of one index over all videos followed by a video-id filter;
- tesseract OCR instead of their Gemini captions. Most OCR units are suppressed by their θt rule anyway.

**How it was validated.** Against the equations, not against their outputs, since no code or per-question outputs were released:
- A unit test (`tests/test_lectqa_protocol.py::test_t1_retrieve_threshold_oracle_filter_merge_and_top_l`) checks:
  - the θr cut;
  - the eq. 22 overlap filter;
  - the merging of temporally adjacent chunks;
  - the top-L cut in temporal order;
  - the empty context when nothing survives the filter.
- The parameter values are copied from their Table 3.
- The replica's own scores are not a validation. With a different answerer they are not comparable with Table 4.

### 1c. Corrected-protocol comparison

*Table 1c-i. Replica vs ours vs full context.*
- *Dataset:* LectQA-Vid open-ended; 300-question stratified subset (100 per difficulty, seed `lectqa-open-subset-20260923`), n = 100 / 100 / 100 / 300, 93 videos.
- *Answerer:* gpt-5.4-mini-2026-03-17 at temperature 0; the same JSON answer prompt for all three modes. No judge.
- *Protocol:* T1's metric suite (eqs. 29–35), in %. Each cell is the mean ± std over **3 repeats**, i.e. run-to-run noise only.
- *Spend:* $3.3421 for all 2,700 answers (`lectqa_open_modes.md`). T1 rows for reference.

| level | mode | F1 | Sim | BLEU | METEOR | ROUGE-1 |
|---|---|---:|---:|---:|---:|---:|
| simple | replica of T1 | 41.34 ± 0.08 | 64.38 ± 0.16 | 13.09 ± 0.51 | 41.50 ± 0.44 | 46.01 ± 0.41 |
| simple | ours (iterative) | 46.30 ± 0.27 | 72.58 ± 0.30 | 13.88 ± 0.27 | 47.02 ± 0.27 | 54.09 ± 0.41 |
| simple | full context | 48.88 ± 0.40 | 74.48 ± 0.21 | 16.18 ± 0.54 | 50.80 ± 0.22 | 57.31 ± 0.23 |
| simple | *T1 Table 4* | 29.47 | 77.23 | 8.61 | 38.15 | 36.82 |
| hard | replica of T1 | 28.06 ± 0.30 | 52.18 ± 0.22 | 4.83 ± 0.26 | 27.22 ± 0.26 | 33.41 ± 0.43 |
| hard | ours (iterative) | 36.26 ± 0.44 | 64.89 ± 0.62 | 7.72 ± 0.17 | 35.84 ± 0.67 | 42.67 ± 0.34 |
| hard | full context | 38.39 ± 0.41 | 65.38 ± 0.33 | 8.49 ± 0.33 | 38.66 ± 0.23 | 46.29 ± 0.21 |
| hard | *T1 Table 4* | 24.38 | 74.56 | 5.29 | 34.71 | 30.94 |
| very hard | replica of T1 | 24.28 ± 0.20 | 55.41 ± 0.20 | 1.24 ± 0.17 | 21.13 ± 0.18 | 28.49 ± 0.22 |
| very hard | ours (iterative) | 27.53 ± 0.13 | 59.91 ± 0.40 | 1.76 ± 0.10 | 25.36 ± 0.31 | 32.58 ± 0.32 |
| very hard | full context | 28.94 ± 0.16 | 60.79 ± 0.18 | 1.92 ± 0.10 | 27.76 ± 0.22 | 35.50 ± 0.33 |
| very hard | *T1 Table 4* | 16.72 | 71.48 | 2.83 | 24.36 | 22.51 |
| **overall** | replica of T1 | **31.23 ± 0.06** | 57.32 ± 0.01 | 6.39 ± 0.27 | 29.95 ± 0.20 | 35.97 ± 0.09 |
| **overall** | ours (iterative) | **36.70 ± 0.16** | 65.79 ± 0.39 | 7.79 ± 0.16 | 36.07 ± 0.18 | 43.11 ± 0.06 |
| **overall** | full context | **38.74 ± 0.23** | 66.88 ± 0.09 | 8.86 ± 0.29 | 39.07 ± 0.15 | 46.37 ± 0.25 |
| overall | *T1 Table 4* | 23.52 | 74.42 | 5.58 | 32.41 | 29.76 |

The modes abstained (answered "not found") on 51 (replica), 14 (ours) and 13 (full context) of the 300 questions.

*Table 1c-ii. Paired differences in token F1 points, with 95 % bootstrap CIs over questions.*
- *Same subset, answerer and spend as Table 1c-i.*
- *Method:* each question's F1 is averaged over its 3 repeats; the mean paired difference is then bootstrapped over questions (10,000 resamples, numpy seed 20260923). This interval is question-sampling uncertainty, unlike the run-to-run ± in Table 1c-i. Source: `lectqa_open_modes.md`.

| difference | simple | hard | very hard | overall |
|---|---:|---:|---:|---:|
| **ours − replica** | +4.95 [+1.68, +8.45] | +8.20 [+4.98, +11.62] | +3.25 [+1.46, +5.18] | **+5.47 [+3.80, +7.24]** |
| full context − ours | +2.58 [+0.70, +4.59] | +2.13 [+0.61, +3.66] | +1.41 [+0.25, +2.58] | +2.04 [+1.14, +2.99] |
| full context − replica | +7.54 [+4.03, +11.35] | +10.33 [+7.11, +13.82] | +4.66 [+2.80, +6.61] | +7.51 [+5.72, +9.40] |

**Full-set single-run means: TODO.** They were not produced. The single run on the 1,118 non-subset open-ended questions did not fit the item's $3.50 cap (estimated $7.43 with the subset), so every open-ended number here comes from the 300-question subset.

### 1d. Verdicts

The caveats above apply to both sentences.

1. **On the 300-question stratified subset of LectQA-Vid, our retrieval (iterative) beats our replication of T1's pipeline by +5.47 token-F1 points (95 % CI +3.80 to +7.24), and the interval excludes zero at every difficulty** (simple +4.95, hard +8.20, very hard +3.25; Table 1c-ii). It does so without reading any timestamp, while the replica's eq. 22 filter reads the gold timestamps.
2. **Giving the answerer the whole transcript beats both retrieval pipelines** (+2.04 [+1.14, +2.99] over ours, +7.51 [+5.72, +9.40] over the replica), **so LectQA-Vid cannot measure the value of retrieval.**
   - The reason is context fit (`lectqa_audit.md` §(e)). Full transcripts run from 140 to 1,544 o200k tokens, median 780, n = 95, and all 95 fit in a 2,048-token window.
   - The answerer's own context window is not reported by the API. Every current OpenAI window is far larger than the longest transcript.
   - On lectures this short, retrieval can only lose information.

### 1e. Benchmark audit

All LLM-free, $0 (`lectqa_audit.md`, except where another file is named).

**MCQ answer position and length.**
- In the published file, the correct answer is option A in 1,484 of 1,489 MCQs (99.7 %). So a constant-"A" answerer scores 99.7 %, against their reported 53.43 % (§(a)).
- Shuffling the options does not fix it. The correct answer is also the unique longest option in 1,409 of 1,489 (94.6 %): 88.6 % simple, 95.4 % hard, 100.0 % very hard.
- A no-video picker of the longest option scores 94.4 % on the 1,414 MCQs of the obtained videos: simple 88.2, hard 95.1, very hard 100.0 (`lectqa_mcq_shuffled.md`).

**MCQ with shuffled options.** We re-ran the MCQs with options shuffled under a recorded seed (`lectqa-mcq-shuffle-20260923`); the correct answer then sits at A 357, B 365, C 356, D 336. Setup: 1,414 MCQs, 95 videos, gpt-5.4-mini-2026-03-17 at temperature 0, baseline retrieval top-4, one run, spend $0.39.
- Accuracy is **95.33** (simple 90.11, hard 96.38, very hard 99.57), against their 53.43 (57.29 / 52.34 / 50.67).
- A seeded 100-question subset answered identically in all 3 repeats.
- Accuracy rises with difficulty, tracking the longest-option share. So this number measures neither retrieval nor the video (`lectqa_mcq_shuffled.md`).

**Timestamps.** The published start/end stamps mix eight digit shapes: `HH:MM:SS`, `MM:SS`, `SS:cc` (`00:12:70` = 12.70 s), `SSS:cc`, `M:SSS:cc`, plain seconds and others. Their meaning differs between videos (§(b)).
- Only 4,532 of 5,964 stamps are unambiguous.
- Of the 2,832 QA pairs of the 95 obtained videos, 1,598 have a usable gold interval. The rest: 878 ambiguous format, 355 past the video's end, 1 ending before it starts.
- By video range, 828 are usable in videos 1–35, 595 in 36–63, and 175 in 64–100.

**Unavailable videos.** 95 of 100 videos were obtained. Mendeley ships no media or transcripts, so everything comes from YouTube. Five are dead (§(c)): videos 1, 10 and 12 ("This video is not available") and videos 38 and 75 (HTTP 403).

**Genre.** Classified by a rule stated before any answer was scored (§(d)), using slide changes per minute from the frame-derived deck:
- ≤ 3 per minute = slide talk: 19 videos;
- > 6 = animated explainer: 47;
- otherwise mixed: 29.

The relatedness gate rejected the speech-to-deck alignment on **49 of 95** videos: 16 of 19 slide talks, 11 of 29 mixed, 22 of 47 animated explainers. Frame-derived slides gave no localisation gain: hit@1 43.2 % with slides vs 45.6 % without (DESIGN finding 12).

**Eq. 22 hurts their own pipeline.** Setup: repeat 0 of the subset; the two groups are different questions, so the other modes are the control (`lectqa_open_modes.md`).
- *Where eq. 22 applies* (usable gold stamp, n = 160): the replica scores 27.54 F1, against 35.71 for ours and 36.76 for full context on the same questions.
- *Where it doesn't* (n = 140): 35.45, against 37.92 and 41.49.
- So the replica loses about five points more than the controls where its oracle filter is on, and the filter leaves 34 of those 160 questions with no context at all.
- Their Table 7 reports the opposite: F1 falls to 11.40 without the filter.

### 1f. Faithfulness per mode

*Table 1f. Unsupported claims per mode.*
- *Data:* the 900 repeat-0 answers of the open-ended subset (300 questions × 3 modes, 93 videos).
- *Models:* answerer gpt-5.4-mini-2026-03-17; judge gpt-4.1-mini-2025-04-14 at temperature 0, **one run per check** (direction only).
- *Protocol:* `linkrag.generate.verify.verify_answer`. Each claim is checked against the text of the units it cites, and a quotable span is required.
- *Spend:* $0.5071 for the original run; the two corrections re-rendered from cache at $0. Source: `lectqa_faithfulness.md`.

| level | replica of T1 | ours (iterative) | full context |
|---|---:|---:|---:|
| simple | 3.1 % (6/195) | 7.1 % (17/239) | 7.0 % (16/228) |
| hard | 5.4 % (11/205) | 0.8 % (2/257) | 4.3 % (11/257) |
| very hard | 2.8 % (7/247) | 2.0 % (6/300) | 2.3 % (7/301) |
| **overall** | **3.7 % (24/647)** | **3.1 % (25/796)** | **4.3 % (34/786)** |

> *Corrected 2026-09-24, twice* (dated notes in `lectqa_faithfulness.md`; DESIGN finding 13):
> - *First,* a judge-cache replay artefact moved one iterative claim (overall 5.4 → 5.3 %).
> - *Second,* `parse_json` had rejected judge replies that were valid JSON followed by a stray `}` and scored them as "no". After the fix, 33 claims moved from unsupported to supported.
> - *Overall rates before both corrections:* replica 4.6 %, ours 5.4 %, full context 5.6 %.

**The modes are indistinguishable.** The largest gap, ours vs full context, is 1.2 points (z ≈ 1.25, p ≈ 0.2). That is inside the ~1-point standard error of a difference (DESIGN finding 13). So the full transcript's F1 gain is not paid for in unsupported claims.

---

## 2. MaViLS (Interspeech 2024)

### Caveats first

- **n = 9.** The benchmark has 20 lectures, split once into 10 for tuning and 10 for testing (`mavils_split.json`, seed 20260922). The MaViLS Kaggle video set has no video for Climate policies, a test-half lecture. So every visual number is a paired mean over **9** test lectures. Climate policies appears with text-only numbers, outside every mean (findings 14–16).
- **Large per-lecture spread.** Paired differences against their per-lecture numbers run from −0.22 to +0.34 (finding 16).
- **The test half was examined in three rounds** (findings 14, 15, 16). Every threshold and weight was set on the tune half only. But the decision to run further rounds, and what to try in them, was informed by earlier test results.
- **Their numbers are published per-lecture figures, decoded by their DP; ours use our DP.**
- **Frames.**
  - Findings 14–15 use one representative frame per dHash segment, kept at 320 px.
  - Finding 16 uses the frame at each sentence's timestamp at native resolution (320×240 to 1280×720), stored as JPEG quality 90.
  - Frame OCR reads 320 px frames after a 3× upscale.

### 2a. Their published numbers and protocol

*Table 2a. MaViLS per lecture on the 9 test lectures with video.*
- *Dataset:* MaViLS; our test half; n = 9 lectures.
- *Numbers:* their published per-lecture F1, audio-only (their Table 1) and all features (Table 2, λ_jump = 0.1).
- *Protocol:* theirs, below. *Spend:* n/a (published).
- *Source:* the "their audio (T1)" and "their all-features (T2)" columns of `mavils_visual.md` and `mavils_final_round.md`.

| lecture | their audio-only | their all-features |
|---|---:|---:|
| ML for health | 0.57 | 0.95 |
| Climate & Cities | 0.49 | 0.86 |
| Deep learning | 0.79 | 0.99 |
| Numerics | 0.46 | 0.81 |
| Phonetics | 0.05 | 0.65 |
| Reinforcement | 0.31 | 0.75 |
| Short range | 0.57 | 0.80 |
| Solar resource | 0.46 | 0.53 |
| Computation theory | 0.56 | 0.84 |
| **mean (9)** | **0.47** | **0.80** |

Their 20-lecture averages (README T2 row):
- audio transcript only: **0.53**
- OCR text only: 0.76
- image only: 0.64
- SIFT baseline: 0.56
- all three features combined: **0.82**

Climate policies, which has no video, scores 0.79 audio-only and 0.93 all-features in their tables.

**Protocol, as read from their code.** This is their `evaluation/evaluate_recall_precision.py` and `mavils/matching_algorithm.py`, as documented in `scripts/adapters/mavils.py`.
- There is one prediction per transcript sentence. Labels are 1-based slide numbers, and −1 means no slide.
- Per lecture, F1 is scikit-learn's micro-averaged `f1_score` over the rows whose gold label is not −1. The label set is taken from the unfiltered column, so it includes −1, and predicting −1 for a labelled sentence counts as a wrong label.
- The paper's averages are the unweighted mean of the per-lecture F1s.
- The DESIGN Targets row's phrase "ignoring −1 labels" refers to the row mask.

### 2b. Where the text-only gap is: features, not the decoder

*Table 2b. Text-only alignment, similarity matrix × decoder.*
- *Dataset:* MaViLS. **TODO:** the allowed sources don't state the lecture set these three numbers were computed on (the README gives no n); `reports/mavils_final.md` has it.
- *Protocol:* their F1; sentence granularity; slide text is page OCR.
- *Matrices:* "theirs" is distiluse-base-multilingual-cased cosine; "ours" is the bge-m3 + BM25 + IDF hybrid.
- *LLM-free, $0. Source:* README T2 row.

| similarity matrix ↓ \ decoder → | their DP | our DP |
|---|---:|---:|
| theirs (distiluse) | 0.513 | **0.520** |
| ours (hybrid) | **TODO:** not in the allowed sources (it is in `reports/mavils_final.md`) | 0.46 |

**Conclusion.** On their own similarity matrix, our decoder scores at least as well as theirs (0.520 vs 0.513). So the gap to their numbers lies in the similarity features, not in the decoder (README T2 row). Fusing the two text matrices, with the weight set on the tune half, reaches 0.520 on the test half against their 0.51 there (README).

### 2c. Adding feature channels, test half

*Table 2c. From text only to all three of their feature types.*
- *Dataset:* MaViLS test half, the 9 lectures with video, paired (every column on the same lectures). Test half run once per round.
- *Protocol:* their F1; our DP at σ = 0.2 (λ = 0.05, β = 0.15, B = 2).
- *Tuned on the tune half only* (`mavils_tuned.json`):
  - image score: SwiftFormer-xs;
  - frame-OCR score: BM25;
  - three-way weights, image / frame OCR / speech text: 0.50 / 0.25 / 0.25 with representative frames, and 0.25 / 0.50 / 0.25 with sentence-time frames;
  - slide-visibility gate: at least 1 OCR word, or an image margin above 0.02.
- *LLM-free, $0. Sources:* `mavils_visual.md` (columns 2–4), `mavils_frame_ocr.md` (column 5), `mavils_final_round.md` (columns 6–7).

| lecture | ours, text only | fused text (ours + theirs) | + image (visual + theirs) | + frame OCR (three-way, representative frames) | + sentence-time frames | + sentence-time frames + gate | their all-features |
|---|---:|---:|---:|---:|---:|---:|---:|
| ML for health | 0.34 | 0.62 | 0.93 | 0.93 | 0.91 | 0.91 | 0.95 |
| Climate & Cities | 0.60 | 0.66 | 0.83 | 0.84 | 0.98 | 0.98 | 0.86 |
| Deep learning | 0.70 | 0.69 | 0.98 | 0.99 | 0.99 | 0.99 | 0.99 |
| **Numerics** | 0.50 | 0.55 | 0.69 | 0.71 | 0.89 | **0.89** | 0.81 |
| Phonetics | 0.36 | 0.33 | 0.84 | 0.89 | 0.99 | 0.99 | 0.65 |
| **Reinforcement** | 0.20 | 0.30 | 0.45 | 0.52 | 0.65 | **0.67** | 0.75 |
| **Short range** | 0.49 | 0.54 | 0.77 | 0.84 | 0.58 | **0.58** | 0.80 |
| Solar resource | 0.35 | 0.39 | 0.55 | 0.60 | 0.80 | 0.81 | 0.53 |
| Computation theory | 0.28 | 0.35 | 0.85 | 0.96 | 0.87 | 0.87 | 0.84 |
| **mean (9)** | **0.426** | **0.493** | **0.765** | **0.810** | **0.850** | **0.853** | **0.80** |
| wins / losses against their all-features | — | — | 3 / 6 | 5 / 4 | TODO: not stated in the sources | **6 / 3** | |

A tie counts as a win. Precision-on-answered and coverage for every cell are in the source files; coverage is 1.00 throughout, since the DP always answers.

**Three lectures to call out.**
- **Numerics** moves from 0.71 to 0.89 with sentence-time frames and now beats their 0.81. The share of its frames carrying any OCR text went from 56 % to 99 % (finding 16).
- **Reinforcement** improves at every step, 0.20 → 0.45 → 0.52 → 0.67, but still loses to their 0.75. It is a camera-heavy lecture; image alone reached 0.13 (finding 14).
- **Short range** falls from 0.84 to 0.58 when sentence-time frames replace the representative ones, against their 0.80. This was not investigated on the test half, so as not to tune on it (finding 16).

The visibility gate gained +0.031 on the tune half but nothing on the test half: 0.810 → 0.809 with representative frames, 0.850 → 0.853 with sentence-time frames (`mavils_final_round.md`). The sentence-time frames did the work.

### 2d. Verdict

Finding 16's verdict sentence, verbatim:

> On MaViLS, combining speech-to-slide text, frame OCR and image features, taken from the frame at each sentence's timestamp, our alignment reaches a mean F1 of 0.853 on the 9 held-out lectures with video, against 0.80 for MaViLS's published all-features result on the same lectures (6 wins, 3 losses). The paired difference (+0.055, 95 % CI −0.050 to +0.161) is within lecture-to-lecture variation, so we report parity with their result, not an improvement.

**Caveats that bound it:**
- one test lecture (Climate policies) has no video, so n = 9;
- the test half has been examined in three rounds;
- the per-lecture differences span −0.22 to +0.34.

0.853 is numerically above their 20-lecture average of 0.82, but that is not a paired comparison. On the same 9 lectures their mean is 0.80, and the +0.055 lies inside a bootstrap interval that includes zero. **Parity** is the label that holds; "exceeds" would not.
