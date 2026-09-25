# LinkRAG — project review, 2026-09-25

A self-contained review of the repository `m1ssmee/LinkRAG` at commit `10c7322` (2026-09-25), written for an external reviewer who has not seen the code. Every number below is quoted, with its source. Numbers taken from DESIGN.md, README.md, `results/` or `reports/` name the file. Numbers produced for this review say how they were produced. This review cost **$0** in API calls.

Section order follows the request. Two changes: the code-health and performance review is §9, and the research-integrity reviewer's verdict on this document comes last, as §10, so that it can be appended verbatim.

---

## 1. One-page summary

**What the project is.** LinkRAG is a question-answering system over a lecture's mixed materials: slide PDF, paper PDF/DOCX, images and the audio recording. It is built as retrieval-augmented generation (RAG). Standard multimodal RAG indexes each modality independently and answers from a plain top-k. LinkRAG adds an **Evidence Linking Layer**: at indexing time it builds typed, scored links across modalities and across files. There are four link types:
- `audio_slide`: speech segment ↔ slide;
- `figure_text`: figure ↔ explaining paragraph;
- `deictic`: "as you see here" ↔ the visual element;
- `same_slide`: figure ↔ the text of its own slide.

Retrieval then follows these links out from its top-k seeds, and a reranker scores the evidence *set* for complementarity. Everything is CPU-first. Every stage has a `baseline` and a `linkrag` mode, so every claim is an ablation (DESIGN.md, *What this is*; *The two-mode rule*).

**Contribution statement, as currently written** (DESIGN.md, *Our three novel components*, verbatim):

> 1. **Evidence Linking Layer** (`src/linkrag/link/`) — typed, scored links at index time, across modalities *and* across files. T1 links only by timestamp inside one video; T2 aligns but does not retrieve.
> 2. **Link-following retrieval** (`src/linkrag/retrieve/`) — top-k gives seeds; traverse links to pull in the evidence a seed *depends on*, at zero LLM calls.
> 3. **Complementarity-aware reranking** (`src/linkrag/retrieve/rerank.py`) — score the evidence *set*: reward an uncovered modality or a linked unit, penalise a restatement within a modality.

DESIGN.md also states what is *not* ours:
- T2 already uses dynamic programming over slides with a jump penalty.
- T1 already does timestamp-constrained retrieval with a cross-encoder rerank.

**The two target papers** (DESIGN.md, *Targets*; dataset policy: results are reported only on these two papers' datasets):

| | Target | Benchmark | Their headline number |
|---|---|---|---|
| T1 | Intra-Video Temporal-Aware RAG (Shafiq et al., *CMC* 88(2):96, 2026) | **LectQA-Vid**: 100 CS lecture videos of 2–5 min, 3,000 QA pairs (½ MCQ, ½ open-ended) | Open-ended overall token F1 **23.52**, semantic similarity 74.42 (Table 4); MCQ accuracy **53.43** (Table 5) |
| T2 | MaViLS (Anderer, Reich, Wölfel, *Interspeech 2024*) | **MaViLS**: 20 lectures, every spoken sentence labelled with its slide | Audio-transcript-only F1 **0.53**; all features combined **0.82** (20-lecture averages) |

**What the verdicts do and do not support** (read these before the numbers):
- On LectQA-Vid, "ours" is the *iterative re-querying* mode. That is our approximation of MI-RAG, which DESIGN.md calls a component, not a contribution. The linking layer is inactive on a single-video benchmark (finding 8).
- On MaViLS, what is compared is the `audio_slide` alignment, one link type of the layer. There, our contribution is the similarity features and their fusion; the DP is not ours.
- **No reported target-benchmark result yet exercises link-following retrieval or complementarity reranking.** Their only measurements are on the development lecture pilot01, which is never reported (n = 25, 4 cross-modal, direction only). §7 ranks this as the top threat.
- The T1 comparison is between **two whole pipelines**, not one component. The replica uses T1's choices: bge-base, semantic chunking, top-10 above 0.6, the eq. 22 oracle filter (which leaves 34 of 160 questions with no context) and a cross-encoder. Ours uses bge-m3 + BM25 with RRF, and one LLM follow-up query. The +5.47 cannot be attributed to re-querying or to any single part. That would need our plain `baseline` on the same 300 questions and the replica run without eq. 22; neither has been run.
- The MaViLS test half has been examined in **three rounds** (findings 14–16). Knobs were set on the tune half, but what to try next was informed by test results (§7 #3).
- Every open-ended LectQA-Vid number is on a 300-question stratified subset, not the full set.

**Headline verdicts.**

| Target | Verdict | Numbers |
|---|---|---|
| T1, open-ended | **Our iterative pipeline beats our replica of T1's pipeline (a whole-pipeline difference), but the benchmark cannot measure the value of retrieval.** | 300-question stratified subset, 93 videos, 3 repeats, answerer gpt-5.4-mini-2026-03-17 for all modes. Token F1: T1 replica **31.23 ± 0.06**, ours (iterative) **36.70 ± 0.16**, full transcript **38.74 ± 0.23**. Ours − replica **+5.47, 95 % CI [+3.80, +7.24]**. Full transcript − ours **+2.04 [+1.14, +2.99]**. Every transcript fits a 2,048-token window. T1's own 23.52 is not comparable: their answerer is unnamed. |
| T1, MCQ | **Not reportable as a retrieval or video result.** | The correct option is A in 99.7 % of published MCQs. With options shuffled we score 95.33, but a no-video "longest option" rule scores 94.4. |
| T2 | **Parity with MaViLS's all-features result.** | 9 test-half lectures with video, their F1 protocol: **ours 0.853 vs their 0.80**. Paired difference **+0.055, 95 % CI [−0.050, +0.161]**; 6 wins, 3 losses. Test half examined in three rounds. LLM-free, $0. |

**Spend.**
- **Project ledger** (`reports/llm_ledger.jsonl`, now kept local): 61 rows dated 2026-09-10 → 2026-09-23 (UTC); the 55 priced rows (2026-09-21 → 2026-09-23) total **$22.05**. By model: gpt-5.4 $14.67, gpt-5.4-mini $3.73, gpt-4.1-mini $3.37, whisper-1 $0.27, Groq gpt-oss-120b $0.
- **Unpriced rows:** 6 rows carry no price (5 gpt-5.4, 1 free Groq).
- **Estimated rows:** 5 priced rows ($7.89) were backfilled from the reply cache after the fact; the rows carrying estimated token counts total $7.56.
- **Upper bound:** the ledger does not credit OpenAI's prompt-cache discount, so each priced row is an upper bound.
- **Billing dashboard** (reported by the account holder, not verifiable from the repository): **$19 in total across keys**.
- **The discrepancy (~$3) is unexplained** (§7).
- All MaViLS work and everything since 2026-09-24, this review included, cost $0.

**Tests.** 354 passed, 1 deselected (the `slow` marker, which downloads a model) on `10c7322`: `pytest -q`, with an invalid API key so that nothing can bill.

---

## 2. Repository map

**Size.** `git log --oneline | wc -l` = **87** commits at `10c7322` (88 with this review). History runs from 2026-09-06 (initial commit) to 2026-09-25.

The tables below are equivalent to `tree -L 3` (the `tree` binary is not installed). One line per module, taken from its own docstring; *lines* is the file length.

### src/linkrag/ (the library)

| module | lines | what it does |
|---|---:|---|
| `core.py` | 191 | Core data model shared by every stage: `EvidenceUnit`, `Link`, `Location`, `Mode`, the closed `LINK_TYPES` set, config loading |
| `costs.py` | 314 | Token and cost accounting for every LLM-touching run, the spend guard (`--max-cost`), and the reply cache |
| `manifest.py` | 122 | Corpus manifest: what was indexed, and a hash that identifies it (stamped on links and gold) |
| `ingest/__init__.py` | 177 | Parse PDF/DOCX, images and audio into EvidenceUnits; id de-duplication |
| `ingest/pdf.py` | 469 | PDF → units via PyMuPDF: text blocks with bboxes, caption-anchored paper figures, vector/raster clustering of deck figures |
| `ingest/docx.py` | 53 | DOCX → text units via python-docx |
| `ingest/image.py` | 54 | Standalone images → figure units via OCR (pytesseract) |
| `ingest/audio.py` | 284 | Audio → units via faster-whisper (int8 CPU), word timestamps, sentence packing, frozen transcripts |
| `ingest/asr_openai.py` | 250 | whisper-1 ASR backend: word timestamps, silence-aware chunking |
| `ingest/video_slides.py` | 203 | Slide units derived from a lecture video's frames (dHash segmentation, revisits, OCR, figure boxes) |
| `ingest/vlm_caption.py` | 107 | Optional VLM figure descriptions via a local vision model (off by default) |
| `index/__init__.py` | 186 | Embed units (bge-m3) and store them; exact numpy dense search + BM25 |
| `link/__init__.py` | 16 | The Evidence Linking Layer (novel component 1) |
| `link/align.py` | 584 | Audio→slide alignment: penalised DP, naive ablation, relatedness z-gate, abstention, similarity options and fusion |
| `link/figure_text.py` | 327 | Figure ↔ paragraph links (reference, layout, page, dense, overlap terms) |
| `link/deictic.py` | 399 | Deictic speech ↔ visual element, with cue tiers |
| `link/same_slide.py` | 67 | Figure ↔ the text of the deck page it sits on |
| `link/graph.py` | 162 | The links as a networkx graph; traversal, filtering, GraphML export |
| `link/pipeline.py` | 95 | Every link type in one call, shared by `build_links.py` and the LectQA adapter |
| `link/relatedness.py` | 167 | Per-link relatedness gate (LLM judge verdicts in link metadata) |
| `link/visual.py` | 144 | Visual channel for alignment: frame→page image, frame-OCR and visibility scores |
| `retrieve/__init__.py` | 10 | Retrieval (novel components 2 and 3) |
| `retrieve/baseline.py` | 54 | Hybrid dense + BM25 fused by reciprocal rank, plain top-k |
| `retrieve/iterative.py` | 173 | Iterative re-querying (the MI-RAG approximation), `linkrag_iter`, and the shared `retrieve_pool` |
| `retrieve/linkrag.py` | 193 | Link-following retrieval: seeds, 1-hop expansion (additive or evict), seed normalisation |
| `retrieve/rerank.py` | 267 | Complementarity reranking and its ablations (`none`, `mmr`), optional cross-encoder, modality gate |
| `generate/__init__.py` | 6 | Answer synthesis package |
| `generate/answer.py` | 370 | Cited answers (JSON claims or prose), OpenAI-compatible completers, backends |
| `generate/citations.py` | 146 | Citations a reader can check: page crop, figure image, audio clip |
| `generate/verify.py` | 109 | Claim-level verification: entailment of each claim against the units it cites |
| `eval/__init__.py` | 33 | Metrics and ablations |
| `eval/metrics.py` | 81 | Standing instruments (modality distribution, locator-matched recall) |
| `eval/lectqa_metrics.py` | 130 | T1's answer metrics (eqs. 29–36) |
| `eval/verify_gold.py` | 528 | Automated gold verification (unit entailment, modality-only answering, relabel rules, sampler) |
| `eval/redundancy.py` | 316 | Modality redundancy: how much of one source another already states |
| `eval/nli.py` | 131 | Local NLI entailment: an ablation backend, never the verifier |
| `ui/__init__.py` | 5 | Placeholder for the Streamlit front-end (not started) |

### scripts/ (entry points)

| script | lines | what it does |
|---|---:|---|
| `ingest.py` | 91 | Ingest files into a searchable index |
| `build_links.py` | 150 | Build every link type over an index and persist `links.jsonl` |
| `ask.py` | 166 | Ask one question; prints the modality distribution, the evidence, the claims with verdicts |
| `compare_retrieval.py` | 328 | Retrieval mode × rerank method matrix on a labelled question set (per-type, per-question tables) |
| `run_regression.py` | 238 | Standing instrument #2: re-run the pilot01 regression set and append to `reports/regression.md` |
| `gate_links.py` | 69 | Per-link relatedness gate over `links.jsonl` |
| `negative_control.py` | 137 | Negative control and false-rejection check for the file-pair gate |
| `eval_alignment.py` | 137 | DP vs naive alignment against the ear-labelled slide timeline |
| `eval_deictic.py` | 259 | Deictic links against the ear-labelled pointing windows |
| `make_alignment_labels.py` | 153 | Ear-labelled slide timeline → per-segment ground truth |
| `plot_alignment.py`, `plot_graph.py` | 104, 103 | Alignment heatmap; link neighbourhood of one unit |
| `transcribe_openai.py` | 117 | whisper-1 transcription and comparison with the local transcript |
| `check_link_types.py` | 40 | Every stored links file against the closed link-type set |
| `adapters/lectqa_vid.py` | 109 | LectQA-Vid (T1) entry point; the steps live in `adapters/lectqa/` |
| `adapters/lectqa/acquisition.py` | 71 | Fetch the videos (yt-dlp) and report coverage |
| `adapters/lectqa/transcription.py` | 79 | Whisper transcript + changed frames with OCR → per-video index |
| `adapters/lectqa/common.py` | 155 | Paths, QA files, T1's published numbers, the strict gold-timestamp parser, prompts |
| `adapters/lectqa/localisation.py` | 206 | Temporal localisation (hit@k, IoU) and the segmentation ablation |
| `adapters/lectqa/frameslides.py` | 126 | Frame-derived slide units, their links, localisation with and without them |
| `adapters/lectqa/metrics.py` | 279 | T1's metric suite on stored answers; the LLM-free benchmark audit |
| `adapters/lectqa/mcq.py` | 131 | MCQ with options shuffled under a recorded seed |
| `adapters/lectqa/open_ended.py` | 437 | Open-ended: T1 replica vs iterative vs full context; faithfulness |
| `adapters/mavils.py` | 1,788 | MaViLS (T2) adapter: their scorer and DP, our alignment, frames, visual/OCR channels, gate, every study command |
| `dataset/candidate.py`, `redundancy.py`, `propose_questions.py` | 128, 106, 213 | Extended-dataset intake tooling (development only since the dataset policy) |
| `eval/verify_gold.py`, `audit_sample.py` | 118, 113 | Verify a proposed gold set; sampled human-audit sheet |
| `eval/judge_agreement.py`, `compare_entailment.py` | 118, 155 | Measure a candidate judge / the NLI switch against stored verdicts |

### tests/ (31 test modules + 4 regression `.jsonl` files; 355 tests, 1 of them `slow`)

| test file | what it pins |
|---|---|
| `test_align.py` | the monotonic DP and its naive ablation |
| `test_alignment_labels.py` | ear-labelled timeline → per-segment truth |
| `test_asr_openai.py` | whisper-1 chunking, timestamp offsets, freezing |
| `test_costs.py` | cache usage, estimated flags, longest-prefix pricing, ledger accumulation |
| `test_deictic_tiers.py` | cue tiers, pair collapsing, `same_slide` |
| `test_generate.py`, `test_phase6.py` | answers, structured claims, verification, citations |
| `test_graph.py` | traversal, filtering, GraphML export |
| `test_index.py`, `test_retrieve.py` | exact dense search, BM25, RRF |
| `test_ingest.py`, `test_video_slides.py`, `test_vlm_caption.py` | PDF/DOCX/image/audio ingestion, frame slides, VLM captions (never called) |
| `test_judge_agreement.py` | judge-agreement tool with a mocked judge; billing refusal |
| `test_lectqa_metrics.py`, `test_lectqa_protocol.py`, `test_lectqa_timestamps.py` | T1's metrics against hand-computed cases; shuffled options and the T1 replica; strict timestamp parsing |
| `test_link_phase3.py`, `test_link_types.py` | figure_text and deictic resolution; the closed link-type set |
| `test_manifest.py` | manifest hash and gold staleness |
| `test_mavils_adapter.py`, `test_visual_channel.py` | their scorer, build groups, their DP from their source; visual channel and fusion |
| `test_redundancy.py`, `test_verify_gold.py` | redundancy metric; gold majority, span check, relabel rules |
| `test_relatedness.py`, `test_relatedness_gate.py` | per-link gate; file-pair z-gate, abstention composition |
| `test_rerank.py`, `test_retrieve_linkrag.py` | complementarity and its ablations; link-following and the iterative baseline |
| `test_run_regression_mode.py` | the regression runner dispatches on `--mode` |
| `test_skeleton.py`, `conftest.py` | imports, config keys, generated fixtures |

### results/ (external benchmarks; the reportable numbers)

| file | content |
|---|---|
| `external/lectqa_coverage.md` | acquisition: 95/100 videos |
| `external/lectqa_theirs_metrics.md/.jsonl` | T1's metric suite on stored first-run answers |
| `external/lectqa_v2.md/.jsonl` | localisation, 28/100 videos (finding 8) |
| `external/lectqa_frameslides.md/.json` | frame-derived slides (finding 12) |
| `external/lectqa_audit.md` | LLM-free benchmark audit (finding 13) |
| `external/lectqa_mcq_shuffled.md/.jsonl` | MCQ with shuffled options |
| `external/lectqa_open_modes.md/.jsonl` | open-ended replica vs iterative vs full context |
| `external/lectqa_faithfulness.md/.jsonl` | unsupported claims per mode |
| `external/mavils_gate*.md/.json` | file-pair gate on MaViLS (v1–v3, finding 7) |
| `external/mavils_split.json`, `mavils_tuned.json` | tune/test split and tuned values |
| `external/mavils_frames.md`, `mavils_visual.md` | visual channel (finding 14) |
| `external/mavils_frame_ocr_sanity.md`, `mavils_frame_ocr.md` | frame OCR, three-feature fusion (finding 15) |
| `external/mavils_visibility_gate.md`, `mavils_final_round.md` | visibility gate, sentence-time frames (finding 16) |
| `nli_vs_llm_pilot01.md`, `nli_pilot01/`, `sampled_redundancy_pilot01/`, `extended_intake.md` | verifier and redundancy studies (development, finding 11) |

### reports/ (development reports, pilot01 and first MaViLS rounds)

`phase1_pilot.md`, `phase2_step0_asr.md`, `asr_openai_pilot01.md`, `regression.md`, `deictic_eval_pilot01.md` (+ `deictic_pairs_pilot01.csv`), `gold_proposal_pilot01.md`, `question_proposal_pilot01.md`, `gold_verified_pilot01*.md/.json` (runs and judges), `audit_sheet_pilot01*.csv` (unfilled human audit), `redundancy_pilot01*.md/.json`, `relatedness_pilot01*.md`, `relatedness_gate.md`, `lectqa_vid_first_run.md/.jsonl`, `mavils_alignment*.md/.json`, `mavils_heldout_study.md`, `mavils_final.md`, `mavils_fused.md`, `mavils_inspect_*.md/.png`, `audit_phase4.md`, `audit_ponytail_2026-09-24.md`, `results_vs_target_papers.md` (the mentor draft), `alignment_pilot01.png`, `graph_hsieh_a22.png`. Kept local, not published: the university progress sheets, the progress-report PDF and the spend ledger (untracked in `10c7322`).

### data/labels/

`pilot01/pilot01_slide_timeline_v2.csv` (ear-labelled slide timeline), `pilot01_alignment_labels_v2.csv` (per-segment truth), `pilot01_pointing_windows.csv` (deictic pointing windows), `alignment_labels.README.md` (their provenance). `data/raw` is not redistributed; `data/processed` is regenerable except the frozen transcripts.

### Ablation switches in `configs/default.yaml`

| switch | default | alternative(s) | finding / decision it belongs to |
|---|---|---|---|
| `mode` (every stage) | `linkrag` | `baseline` | the two-mode rule |
| `ingest.audio_segmentation` | `sentence` | `fixed` | finding 9 (free on localisation, not claimed as a gain) |
| `ingest.ocr_figures` | `true` | `false` | pilot01: 13/18 deck figures had empty content without it |
| `ingest.figures_from_captions` | `null` (on for papers) | `true`/`false` | caption-anchored paper figures (13 captioned vs 2 raster fragments) |
| `ingest.cluster_deck_figures` | `true` | `false` | vector/raster deck-figure clustering |
| `ingest.vlm_captions.enabled` | `false` | `true` | optional; not measured |
| `ingest.asr_backend` | `local` | `openai` | known issue (c) |
| `link.align.method` | `monotonic` | `naive` | finding 6 (DP vs argmax) |
| `link.align.start_prior_mu` | `0.02` | `0.0` (every earlier measurement) | alignment start prior |
| `link.align.flatness_scaling` | `0.0` | > 0 | finding 6 (inert) |
| `link.align.min_segment_sim` | `null` | a threshold (0.5055 tuned) | finding 6 (abstention) |
| `link.align.relatedness_z` | `1.27` | `null` disables | finding 7 (file-pair gate) |
| `link.align.build_groups` | `false` | `true` | finding 6 (lowers tune F1 → off) |
| `link.align.similarity` | `ours` | `theirs`, `fused_max`, `fused_weighted`, `visual`, `visual+text`, `visual+theirs`, `frame_ocr`, `visual+frame_ocr`, `visual+frame_ocr+theirs` | findings 6, 14, 15, 16 |
| `link.align.frame_source` | `representative` | `sentence_time` | finding 16 |
| `link.same_slide.enabled` | `true` | `false` | same_slide link type |
| `link.relatedness.enabled` | `true` | `false` | per-link gate (priority ii) |
| `retrieve.expansion` | `additive` | `evict` | design change 2026-09-21 (A1 100 % → 0 % under evict) |
| `retrieve.linkrag.normalise_seeds` | `true` | `false` | design change; `linkrag/none ≡ baseline/none` by construction |
| `retrieve.rerank.method` | `complementarity` | `none`, `mmr` | novel component 3; finding 8 |
| `retrieve.rerank.cross_encoder` | `null` | a model name | optional |
| `retrieve.rerank.modality_gate` | `false` | `true` | finding 8 (fires on 79 % of cells, changes nothing measurable) |
| `generation.structured` / `verify` / `strict` | `true` / `true` / `false` | — | Phase 6 |
| `eval.entailment.backend` | `llm` | `nli` | finding 11 (NLI κ 0.31/0.36 → ablation only) |
| `models.llm.allow_strong_in_batch` | `false` | `true` | cost policy |
| `dataset.intake.redundancy_sample` | `0` (census) | N | sampled redundancy design change |

---

## 3. Pipeline, end to end

**Every pilot01 number in this section is development evidence.** By the dataset policy, pilot01 "never appears in a reported table". This review shows the table only so that an external reviewer can see what each stage does. Whether a project review may include such a table is not settled in DESIGN.md (flagged in §7 #16), and none of these numbers is a result.

pilot01 is one lecture (Hsieh et al., *Focus*, OSDI '18). It is made of:
- a 22.9-minute talk recording;
- a 27-slide deck;
- the 19-page paper.

Corpus manifest `2f3b35f27e86caf8` holds 209 units:
- 111 text: 84 from the paper, 27 from the deck;
- 47 figure: 16 from the paper, 31 from the deck;
- 51 audio.

The numbers come from the local corpus manifest `data/processed/manifest.json`, and 111 + 47 + 51 = 209. Where a stage's number is not from pilot01, the table says so.

| stage | file(s) (`src/linkrag/`) | key config keys | the characterising number |
|---|---|---|---|
| **Ingestion: PDF** (text blocks with bboxes, chunking; deck vs paper by page aspect) | `ingest/pdf.py`, `ingest/__init__.py` | `ingest.chunk_tokens` 300, `chunk_overlap_tokens` 60, `slide_deck_landscape_ratio` 0.6 | 111 text units (84 paper + 27 deck) |
| **Ingestion: DOCX** | `ingest/docx.py` | none | pilot01 has no DOCX; covered by tests only |
| **Ingestion: images** (OCR) | `ingest/image.py`; figures are OCR'd via `pdf.py` | `ingest.ocr_figures` true | "13/18 pilot01 figures had empty content" before figure OCR (`configs/default.yaml`, comment on `ocr_figures`) |
| **Ingestion: audio** (faster-whisper `small` int8, word timestamps, sentence packing, frozen transcript) | `ingest/audio.py`, `ingest/asr_openai.py` | `models.whisper` small, `ingest.audio_segment_seconds` 30, `audio_segmentation` sentence, `asr_vocab_from_slides`, `frozen_transcript_dir` | Mid-sentence boundaries: fixed windows cut **32/46 (70 %)**, sentence packing **0/53 (0 %)** (`reports/phase1_pilot.md`) |
| **Ingestion: deck-figure clustering** (vector drawing ops + raster boxes) | `ingest/pdf.py` `cluster_figure_regions` | `ingest.cluster_deck_figures` true, `min_figure_area_px` 10000 | Deck figures **18 → 31**; slides carrying a figure unit **11 → 24 of 27** (`reports/regression.md`) |
| **Ingestion: caption-anchored paper figures** | `ingest/pdf.py` `caption_regions` | `ingest.figures_from_captions` null (= on for papers) | The paper has **13 captioned figures**; raster extraction had produced **2**, both logo fragments (`docs/pilot01_history.md`) |
| **Ingestion: frame-derived slides** (1 fps dHash segmentation, revisits merged, OCR, figure boxes) | `ingest/video_slides.py` | function arguments (`fps` 1.0, `max_dist` 10, `min_dur` 2.0); `datasets.lectqa_vid.*` | *LectQA-Vid, not pilot01:* **2,175 slides** over 95 videos (~23 per video); 5,000 links (audio_slide 992, deictic 995, figure_text 101, same_slide 2,912) (DESIGN.md finding 12) |
| **Indexing** (bge-m3 embeddings, exact numpy inner product, BM25, reciprocal-rank fusion) | `index/__init__.py`, `retrieve/baseline.py` | `models.embedding` BAAI/bge-m3, `index.backend` numpy, `normalize_embeddings`, `retrieve.candidates` 50, `rrf_k` 60 | **209 units** indexed. DESIGN.md's "8.8 ms/query over 100k × 1024d" states no measurement protocol, so it is **not reportable** under the latency rule (see §9 for a rule-compliant timing) |
| **Linking: audio_slide** (penalised DP with bounded backtracking, O(nm) prefix max, start prior, file-pair relatedness z-gate, per-segment abstention) | `link/align.py` | `link.align.method` monotonic, `jump_penalty` 0.05, `skip_penalty` 0.02, `back_penalty` 0.15, `max_back` 2, `start_prior_mu` 0.02, `relatedness_z` 1.27, `null_shuffles` 30, `min_segment_sim` null | Segment accuracy vs the ear-labelled timeline: **DP 32/42 = 76.2 %** vs naive argmax 24/42 = 57.1 % (`reports/regression.md`). Optimistic: the start prior was fitted on the same 42 labels. The negative control (pilot01 audio × unrelated deck) gives **0 cross-file links** |
| **Linking: figure_text** (reference, layout, page, dense and overlap terms) | `link/figure_text.py` | `link.figure_text.weights`, `page_decay` 2.0, `layout_max_gap_pt` 220, `figure_text_threshold` 0.50 | **103 links**; relatedness-gate pass **100 %**, partly tautological, since shared slide titles appear in the OCR (`reports/relatedness_pilot01.md`) |
| **Linking: deictic** (cue tiers: 1 explicit object deixis, 2 pronoun + visual word, 3 bare pronoun) | `link/deictic.py` | `link.deictic.weights`, `tier_weights` {1: 1.0, 2: 0.6, 3: 0.3}, `deictic_threshold` 0.45 | Against ear-labelled pointing windows: **precision 12/15 = 80 %, recall 12/14 = 86 %** (`reports/deictic_eval_pilot01.md`). Relatedness gate: 95/126 = 75.4 %. Tier 1 is empty on pilot01 |
| **Linking: same_slide** | `link/same_slide.py` | `link.same_slide.enabled`, `score` 1.0 | **31 links**, gate pass 100 %. All types together: 311 links, gate pass 279/311 = 89.7 % (audio_slide 51, deictic 126, figure_text 103, same_slide 31; 51 + 126 + 103 + 31 = 311) |
| **Retrieval modes**: `baseline` (RRF top-k), `iterative` (MI-RAG approximation, 2 rounds), `linkrag` (seeds + 1-hop expansion), `linkrag_iter` (both) | `retrieve/baseline.py`, `retrieve/iterative.py`, `retrieve/linkrag.py` | `retrieve.expansion` additive, `linkrag.k_seed` 5, `k_final` 8, `hops` 1, `decay` 0.5, `normalise_seeds` true, `iterative.rounds` 2 | **Additive expansion (design change):** under `evict`, question A1 went from baseline 100 % to linkrag 0 %; under `additive`, linkrag/none recovers to 100 % (`reports/regression.md`). **Seed normalisation:** `linkrag/none` equals `baseline/none` exactly, by construction (inspected per measurement rule 1) |
| **Reranking**: `complementarity` (modality bonus α, link bonus β, redundancy penalty γ), `mmr`, `none`; modality-need gate | `retrieve/rerank.py` | `retrieve.rerank.method` complementarity, `alpha` 0.3, `beta` 0.2, `gamma` 0.4, `pool` 20, `modality_gate` false | Cross-modal questions (**n = 4, direction only**): linkrag/none 45.8 % → linkrag/complementarity 60.4 % gold recall. Modality gate: calls 12 of 25 pools concentrated, and the selected 8-unit sets are identical (`results/external/lectqa_v2.md`) |
| **Grounded generation and verification** (JSON claims citing unit ids; citations rendered as page crops, figure images or audio clips; claim entailment with 3 judge runs, majority and a quotable span; `strict` drops unsupported claims and abstains if none survive) | `generate/answer.py`, `generate/citations.py`, `generate/verify.py` | `generation.structured`, `verify`, `strict`, `verify_runs` 3, `citations.media`; `eval.entailment.backend` llm | Hallucination rate (unsupported / claims), one generation run, gpt-5.4 answerer, gpt-4.1-mini judge: **baseline 6.7 % (5/75), linkrag 4.0 % (3/75), linkrag + strict 1.4 % (1/73)**; citation correctness 88–91 % (87.6, 89.0, 90.6; `reports/regression.md`, Phase 6). Direction only |
| **Evaluation**: gold groups matched by file+location, `verify_gold` (unit entailment + modality-only answering), separate judge tiers, cost module, measurement rules | `eval/verify_gold.py`, `eval/metrics.py`, `eval/redundancy.py`, `eval/nli.py`, `costs.py` | `eval.judge.model` gpt-4.1-mini-2025-04-14, `eval.entailment.backend` llm, `models.cheap_tier`, `allow_strong_in_batch` false, `cost.max_usd` 0.0 | **25 of 25** proposed questions kept; **69 of 147** proposed gold units kept; **4 cross-modal**. Judge vs answerer agreement on unit verdicts: 92.5 %, **Cohen's κ = 0.85** (`reports/gold_verified_pilot01.md`) |

---

## 4. Findings 1–16

The findings are DESIGN.md's binding record. Each gets three lines here: what was tried, what was measured, and what was decided. *Recorded* is the commit that first wrote the finding. *Revised* lists every later commit that corrected, retracted, superseded or re-framed it. All hashes are verified.

Findings 1–5 and 11 are on pilot01, so they are development evidence and never reported. Findings 6, 7 and 14–16 are on MaViLS; 8, 9, 12 and 13 are on LectQA-Vid.

| # | tried | measured | decided | recorded → revised |
|---|---|---|---|---|
| 1 | Link-following on the machine-verified pilot01 gold | The only clear win is C5, the one deictic question whose referent is visual-only; where a fact is restated in another modality, plain retrieval reaches it | Link-following helps only where modalities are complementary | `187fc27`. Re-run on gated links without change (`9198596`); pilot01 made development-only (`af921aa`) |
| 2 | Measure how redundant pilot01 is | Only 4–5 of 16 proposed cross-modal questions survive verification; deck→transcript redundancy 94.2 % | pilot01 cannot carry a cross-modal claim; redundancy became an intake metric | `187fc27`. An NLI verifier would have made it 59.6 %; the LLM verifier was kept (`af42c3c`) |
| 3 | Additive link-following vs baseline, by question type | Single-source n = 21: 50.5 % vs 49.7 %. Cross-modal n = 4: 60.4 % vs 45.8 % | Neutral on single-modality, gain on cross-modal; **n = 4, direction only** | `a49ce4b`. Unchanged on gated links (`9198596`) |
| 4 | Compose iterative + link-following (`linkrag_iter`) | Cross-modal: 45.8 % vs `linkrag` 60.4 % on the same four | Hypothesis: re-querying re-textualises the seeds. To be tested on an extended set | `a49ce4b`. Became untestable when the extended dataset was dropped (`af921aa`) |
| 5 | Same comparison, single-modality questions | `linkrag_iter/complementarity` 71.3 % vs iterative 55.8 % and linkrag 50.5 % (n = 21) | **Unexplained; flagged; not to be cited** until it has a cause | `a49ce4b`. Still open |
| 6 | Our DP and features on MaViLS, like-for-like (transcript + PDF, their scorer) | 0.46 vs their audio-only 0.53. Decomposition: their matrix × our DP 0.520 > theirs × theirs 0.513; ours × theirs 0.425; ours × ours 0.461. Fused similarity 0.520 on the test half (paper 0.51). σ sweep +0.009 (noise); flatness scaling inert; abstention trades coverage for precision; build groups lower tune F1 | The gap is in the similarity features, not the decoder. All knobs stay off by default | `fd7beb3` (as "0.45 vs 0.53") → reframed `e1ad25c`; DP margin +0.06 → +0.16 `6471635`; decomposition and σ sweep `c39f51b`; fusion `b4a0b31` |
| 7 | File-pair relatedness gate: penalised-DP objective vs a 30-shuffle null | 380 unrelated pairs: z median −0.09, 95th percentile 1.26. At z = 1.27, 4.7 % false acceptance and 15 % false rejection (the low-F1 lectures plus ML for health, cause unconfirmed); ρ(z, F1) = 0.56. The segments-per-slide hypothesis was rejected (pooled ρ(z, n/m) = −0.31) | z = 1.27; 30-second windows. Limitation: it cannot separate topical neighbours (z = 3.5), so pairing must come from metadata | threshold `71af96e`; finding and rejection `241b299`. DESIGN.md's *Design changes* table still says "(2.0)" |
| 8 | LinkRAG on LectQA-Vid localisation (28/100 videos, 840 QA) | linkrag ≡ baseline (one video, no cross-file links); iterative +7 pp hit@1 (44.8 → 52.0); complementarity −13 pp hit@3 (68.1 → 55.4); modality gate fires on 79 % of cells and changes nothing | Single-stream benchmark: linking inactive by construction; gate off | `442875c`. Its numbers were later declared understated because unusable timestamps were scored (`4225b5c`), and re-measured on the filtered set (`8bef0cc`). The finding's text was not edited |
| 9 | Sentence-aware vs fixed segmentation on LectQA-Vid | Mid-sentence boundaries 82.8 % → 0.0 %; hit@1 moves 0.5 pp, IoU 0.010 (noise) | Keep `sentence` (free); not claimed as a retrieval gain | `442875c`. Same understatement note (`4225b5c`) |
| 10 | Extended-dataset selection criterion (low redundancy, diagram-heavy decks, a speaker who points) | — | **Superseded** by the dataset policy (no extended dataset) | `187fc27` (then numbered 3; renumbered 6, 7, 8 and 10) → superseded `af921aa` |
| 11 | Local NLI cross-encoder as the gold/intake verifier | κ 0.31 vs gpt-4.1-mini and 0.36 vs gpt-5.4 (the LLM judges agree at 0.85); drops 14/25 questions; flips pilot01 redundancy 94.2 % → 59.6 %; the large model reaches κ 0.35 | Verification stays LLM-based; NLI is an ablation only | `af42c3c`. Runtime ratio dropped (`0758fe1`); caveat first + "retune" label (`9f8bbb9`); artefacts' verifier name corrected (`d28685b`); judge moved from Groq to gpt-4.1-mini (`af921aa`) |
| 12 | Frame-derived slides on LectQA-Vid (95 videos, retrieval only, $0) | hit@1 43.2 % with slides vs 45.6 % without; the DP picks the on-screen slide for 22.5 % of 1,870 segments (argmax 21.7 %); the gate rejects 49/95 | No localisation gain; the cause is the speech↔animation-frame alignment | `8bef0cc`. None |
| 13 | Corrected T1 comparison + LLM-free audit (95 videos; answerer gpt-5.4-mini; judge gpt-4.1-mini; $4.24) | See §5.1–5.3: full context 38.74 > ours 36.70 > replica 31.23 F1; MCQ confounded; faithfulness 3.7 / 3.1 / 4.3 % | LectQA-Vid can support the audit, the replication and the full-context ceiling, not a retrieval claim | `181f171`. It builds on the corrections `bbf3570`, `7b52252` and `4225b5c`; corrected twice after (`5c8c976`, `9ac40f8`), conclusion unchanged |
| 14 | Visual channel on MaViLS (SwiftFormer-xs frame→page; 19/20 lectures have video) | Test half, 9 lectures: visual 0.710, visual+text 0.764, visual+theirs 0.765 vs their all-features 0.80; wins 3 of 9 | The frames carry the alignment; frame OCR is next | `970a28a`. Followed by `7adce5f`; headline superseded by `95354ea` |
| 15 | + frame OCR (BM25, 3× upscale), three-way fusion | Frame OCR alone 0.495; three-way **0.810** vs 0.80; 5 wins, 4 losses | Parity, not a win | `7adce5f`. Superseded by `95354ea`. Its caveat that sentence-time frames were skipped "because the videos are no longer available" became stale when `0a049ee` recovered them; not edited |
| 16 | + slide-visibility gate + sentence-time frames (native resolution) | 0.810 → gate 0.809 → sentence-time 0.850 → both **0.853** (tune 0.878); 6 wins, 3 losses; CI [−0.050, +0.161] | Report parity; the gate does not generalise | `95354ea`. None |

**Findings that are not in the numbered list but changed a design** (DESIGN.md, *Design changes recorded against the verified-gold result*, from 2026-09-21):
- additive expansion;
- seed normalisation;
- a separate judge;
- per-type reporting;
- cost accounting;
- the per-link and file-pair relatedness gates.

---

## 5. Results vs the target papers

### 5.1 LectQA-Vid (T1) — corrected-protocol comparison

**Caveats first** (DESIGN.md finding 13):
- **Different answerer.** T1's answering LLM is unnamed. Ours is gpt-5.4-mini-2026-03-17 in every mode, so only comparisons among our modes are valid.
- **Evaluation set.** Their split and 1,000-pair subset are unpublished. We use the 95 of 100 videos still online, with our own faster-whisper transcripts, and a stratified 300-question open-ended subset (100 per difficulty, seed `lectqa-open-subset-20260923`) on 93 videos.
- **The baseline is our replica of T1's pipeline** (§4.2–4.4), not their code. M = 8, L = 4, the merge gap, the MS MARCO MiniLM cross-encoder, per-video search and tesseract OCR are our choices where the paper is silent.
- **Their eq. 22 filter reads the gold timestamps** (an oracle), and is usable on 160 of the 300 questions.
- **The full set was not run.** The single run on the other 1,118 open-ended questions was estimated at $7.43 against a $3.50 cap.
- **Whole pipelines.** Replica and ours differ in embedder, index, chunking, filter and re-query, so Table 5.1b's differences are pipeline differences (§1).
- **Provenance gap.** The source files for every table in §5 (5.1–5.5, including `reports/mavils_final.md` and the audit files) record dataset, n, models and runs but no corpus/config hash, and neither does this review.

*Table 5.1a.*
- *Data:* token F1 and the other T1 metrics (eqs. 29–35, in %), mean ± std over 3 repeats (run-to-run noise); n = 100 / 100 / 100 / 300.
- *Answerer:* gpt-5.4-mini-2026-03-17 at temperature 0; no judge.
- *Spend:* $3.3421 for 2,700 answers. *Source:* `results/external/lectqa_open_modes.md`.

| level | mode | F1 | Sim | BLEU | METEOR | ROUGE-1 |
|---|---|---:|---:|---:|---:|---:|
| simple | replica of T1 | 41.34 ± 0.08 | 64.38 ± 0.16 | 13.09 ± 0.51 | 41.50 ± 0.44 | 46.01 ± 0.41 |
| simple | ours (iterative) | 46.30 ± 0.27 | 72.58 ± 0.30 | 13.88 ± 0.27 | 47.02 ± 0.27 | 54.09 ± 0.41 |
| simple | full context | 48.88 ± 0.40 | 74.48 ± 0.21 | 16.18 ± 0.54 | 50.80 ± 0.22 | 57.31 ± 0.23 |
| simple | *T1 Table 4 (published)* | 29.47 | 77.23 | 8.61 | 38.15 | 36.82 |
| hard | replica of T1 | 28.06 ± 0.30 | 52.18 ± 0.22 | 4.83 ± 0.26 | 27.22 ± 0.26 | 33.41 ± 0.43 |
| hard | ours (iterative) | 36.26 ± 0.44 | 64.89 ± 0.62 | 7.72 ± 0.17 | 35.84 ± 0.67 | 42.67 ± 0.34 |
| hard | full context | 38.39 ± 0.41 | 65.38 ± 0.33 | 8.49 ± 0.33 | 38.66 ± 0.23 | 46.29 ± 0.21 |
| hard | *T1 Table 4 (published)* | 24.38 | 74.56 | 5.29 | 34.71 | 30.94 |
| very hard | replica of T1 | 24.28 ± 0.20 | 55.41 ± 0.20 | 1.24 ± 0.17 | 21.13 ± 0.18 | 28.49 ± 0.22 |
| very hard | ours (iterative) | 27.53 ± 0.13 | 59.91 ± 0.40 | 1.76 ± 0.10 | 25.36 ± 0.31 | 32.58 ± 0.32 |
| very hard | full context | 28.94 ± 0.16 | 60.79 ± 0.18 | 1.92 ± 0.10 | 27.76 ± 0.22 | 35.50 ± 0.33 |
| very hard | *T1 Table 4 (published)* | 16.72 | 71.48 | 2.83 | 24.36 | 22.51 |
| **overall** | replica of T1 | **31.23 ± 0.06** | 57.32 ± 0.01 | 6.39 ± 0.27 | 29.95 ± 0.20 | 35.97 ± 0.09 |
| **overall** | ours (iterative) | **36.70 ± 0.16** | 65.79 ± 0.39 | 7.79 ± 0.16 | 36.07 ± 0.18 | 43.11 ± 0.06 |
| **overall** | full context | **38.74 ± 0.23** | 66.88 ± 0.09 | 8.86 ± 0.29 | 39.07 ± 0.15 | 46.37 ± 0.25 |
| overall | *T1 Table 4 (published)* | 23.52 | 74.42 | 5.58 | 32.41 | 29.76 |

Abstentions ("not found") out of 300: replica 51, ours 14, full context 13.

*Table 5.1b.* Paired differences in token F1 points, 95 % bootstrap CI over questions:
- *Method:* per-question F1 averaged over the 3 repeats; 10,000 resamples, numpy seed 20260923.
- *Data:* the same subset and answerer as Table 5.1a.

| difference | simple | hard | very hard | overall |
|---|---:|---:|---:|---:|
| **ours − replica** | +4.95 [+1.68, +8.45] | +8.20 [+4.98, +11.62] | +3.25 [+1.46, +5.18] | **+5.47 [+3.80, +7.24]** |
| full context − ours | +2.58 [+0.70, +4.59] | +2.13 [+0.61, +3.66] | +1.41 [+0.25, +2.58] | +2.04 [+1.14, +2.99] |
| full context − replica | +7.54 [+4.03, +11.35] | +10.33 [+7.11, +13.82] | +4.66 [+2.80, +6.61] | +7.51 [+5.72, +9.40] |

### 5.2 LectQA-Vid — benchmark audit (LLM-free, $0; `results/external/lectqa_audit.md`, `lectqa_mcq_shuffled.md`)

| check | result |
|---|---|
| Correct MCQ option position (published file) | A in **1,484 of 1,489** (99.7 %). A constant-"A" answerer scores 99.7 against their 53.43 |
| Correct option = unique longest option | **1,409 of 1,489** (94.6 %): simple 88.6 %, hard 95.4 %, very hard 100.0 % |
| No-video "pick the longest option" rule | **94.4 %** on 1,414 MCQs (simple 88.2, hard 95.1, very hard 100.0) |
| MCQ with shuffled options (seed `lectqa-mcq-shuffle-20260923`; gold at A 357, B 365, C 356, D 336; gpt-5.4-mini, our RRF baseline retrieval (dense + BM25) top-4, one run, $0.39) | **95.33** (simple 90.11, hard 96.38, very hard 99.57) vs their 53.43 (57.29 / 52.34 / 50.67). 100 of 100 seeded questions answered identically in 3 repeats |
| Timestamp formats | 8 digit shapes (`HH:MM:SS`, `MM:SS`, `SS:cc` as in `00:12:70` = 12.70 s, `SSS:cc`, `M:SSS:cc`, plain seconds, others); 4,532 of 5,964 stamps unambiguous |
| Usable gold intervals | **1,598 of 2,832** QA pairs (878 ambiguous format, 355 past the video's end, 1 ending before it starts; 1,598 + 878 + 355 + 1 = 2,832). By video range: 1–35: 828, 36–63: 595, 64–100: 175 |
| Videos obtained | 95 of 100 (dead: 1, 10, 12 "not available"; 38, 75 HTTP 403) |
| Genre, by slide changes/min (rule fixed before scoring) | 19 slide talks (≤ 3/min), 29 mixed, 47 animated explainers (> 6/min); 19 + 29 + 47 = 95 |
| Relatedness gate on frame-derived decks | rejected the speech-to-deck alignment on **49 of 95**: 16/19 slide talks, 11/29 mixed, 22/47 animated |
| Context fit | full transcripts 140–1,544 o200k tokens, median 780, n = 95. **All 95 fit a 2,048-token window** |
| T1's eq. 22 filter, repeat 0 | where it applies (n = 160): replica 27.54 F1 vs ours 35.71, full 36.76. Where it does not (n = 140): 35.45 vs 37.92, 41.49. It leaves 34 of 160 questions with no context. Their Table 7 reports the opposite (11.40 without the filter) |

### 5.3 LectQA-Vid — faithfulness after correction

*Table 5.3.* Unsupported claims / claims:
- *Data:* the 900 repeat-0 answers (300 questions × 3 modes, 93 videos).
- *Models:* answerer gpt-5.4-mini-2026-03-17; judge gpt-4.1-mini-2025-04-14 at temperature 0, **one run per check** (direction only).
- *Spend:* $0.5071; the corrections were re-rendered from cache at $0. *Source:* `results/external/lectqa_faithfulness.md`.

| level | replica of T1 | ours (iterative) | full context |
|---|---:|---:|---:|
| simple | 3.1 % (6/195) | 7.1 % (17/239) | 7.0 % (16/228) |
| hard | 5.4 % (11/205) | 0.8 % (2/257) | 4.3 % (11/257) |
| very hard | 2.8 % (7/247) | 2.0 % (6/300) | 2.3 % (7/301) |
| **overall** | **3.7 % (24/647)** | **3.1 % (25/796)** | **4.3 % (34/786)** |

Per-level claim counts sum to the overall ones (195 + 205 + 247 = 647; 239 + 257 + 300 = 796; 228 + 257 + 301 = 786).

**Corrected twice on 2026-09-24**, each in its own commit. Before the corrections the overall rates were 4.6 / 5.4 / 5.6 %.
- `5c8c976`: a judge-cache replay artefact moved one iterative claim (5.4 → 5.3 %).
- `9ac40f8`: `parse_json` had rejected valid JSON followed by a stray `}`; 33 claims moved from unsupported to supported.

The largest gap (ours vs full context, 1.2 points; z ≈ 1.25, p ≈ 0.2) is inside the ~1-point standard error of a difference, so the modes are indistinguishable.

**Finding 13 has no separate verdict sentence. Its headline and its consequence, verbatim:**

> On LectQA-Vid the whole transcript beats retrieval, T1's and ours, and the MCQ half measures neither retrieval nor the video.

> LectQA-Vid lectures are 2–5 minutes long, and every transcript fits in the answerer's context. Retrieval can only lose information here, and no retrieval method can show a benefit over reading everything. LinkRAG's cross-file linking is also inactive on one video (the v2 caption). What LectQA-Vid can support is the audit, the replication (our retrieval beats a T1 replica by 5.5 F1 without reading timestamps), and the full-context ceiling. A retrieval claim needs material that does not fit in the context.

*Review note:* the "our retrieval beats a T1 replica by 5.5 F1" in this quote is a whole-pipeline difference (embedder, index, chunking, filter and re-query all differ; §1), not a gain attributable to retrieval alone. DESIGN.md's wording should be corrected with a dated note.

### 5.4 MaViLS (T2) — decomposition: similarity matrix × decoder

*Table 5.4.*
- *Data:* text-only alignment, **all 20 lectures**; their F1 (sentence granularity, −1 in the label set, micro F1 per lecture, unweighted mean); slide text = page OCR.
- *Cells:* "their DP" is their public code; "theirs" is distiluse-base-multilingual-cased cosine; "ours" is the bge-m3 + BM25 + IDF hybrid.
- *LLM-free, $0. Source:* `reports/mavils_final.md` §2 (F1, with precision-on-answered / coverage in brackets); DESIGN.md finding 6.

| similarity matrix ↓ / decoder → | their DP | our DP |
|---|---:|---:|
| theirs (distiluse, page OCR) | 0.513 (0.505 / 1.00) | **0.520** (0.513 / 1.00) |
| ours (bge-m3 + BM25 + IDF, page OCR) | 0.425 (0.419 / 1.00) | 0.461 (0.454 / 1.00) |

- **Replication check:** their matrix × their DP gives 0.513, against their paper's 0.53.
- **Where the gap is:** swapping the matrix moves the mean +0.074; swapping the decoder, −0.022. The gap is in the similarity features, and our decoder is at least as good as theirs.
- **Fusion:** the fused text similarity (weight 0.5, set on the tune half) reaches 0.520 on the test half, against the paper's 0.51 there (`reports/mavils_fused.md`).
- *Resolves two TODOs in `reports/results_vs_target_papers.md` §2b:* the lecture set is all 20, and the ours × their-DP cell is 0.425.

### 5.5 MaViLS — feature channels, per lecture

*Table 5.5.*
- *Data:* MaViLS test half, the 9 lectures with video (Climate policies has none), paired; their F1 protocol.
- *Protocol:* our DP at σ = 0.2 (λ = 0.05, β = 0.15, B = 2).
- *Tuned on the tune half only* (`mavils_tuned.json`):
  - image score SwiftFormer-xs; frame-OCR score BM25;
  - three-way weights, image / frame OCR / text: 0.50 / 0.25 / 0.25 on representative frames, 0.25 / 0.50 / 0.25 on sentence-time frames;
  - visibility gate: ≥ 1 OCR word or image margin > 0.02.
- *Test half run once per round (three rounds). What to try in rounds 2 and 3 was informed by round-1 and round-2 test results (Reinforcement, Numerics): a high threat (§7 #3). LLM-free, $0. Sources:* `mavils_visual.md`, `mavils_frame_ocr.md`, `mavils_final_round.md`.

| lecture | our text (hybrid) | fused text (ours + theirs) | image + their text (visual+theirs) | image + frame OCR + their text (three-way, repr. frames) | three-way, sentence-time frames | three-way, sentence-time frames + gate (final) | their all-features | final − theirs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ML for health | 0.34 | 0.62 | 0.93 | 0.93 | 0.91 | 0.91 | 0.95 | −0.04 |
| Climate & Cities | 0.60 | 0.66 | 0.83 | 0.84 | 0.98 | 0.98 | 0.86 | +0.12 |
| Deep learning | 0.70 | 0.69 | 0.98 | 0.99 | 0.99 | 0.99 | 0.99 | +0.00 |
| Numerics | 0.50 | 0.55 | 0.69 | 0.71 | 0.89 | 0.89 | 0.81 | +0.08 |
| Phonetics | 0.36 | 0.33 | 0.84 | 0.89 | 0.99 | 0.99 | 0.65 | +0.34 |
| Reinforcement | 0.20 | 0.30 | 0.45 | 0.52 | 0.65 | 0.67 | 0.75 | −0.08 |
| Short range | 0.49 | 0.54 | 0.77 | 0.84 | 0.58 | 0.58 | 0.80 | −0.22 |
| Solar resource | 0.35 | 0.39 | 0.55 | 0.60 | 0.80 | 0.81 | 0.53 | +0.28 |
| Computation theory | 0.28 | 0.35 | 0.85 | 0.96 | 0.87 | 0.87 | 0.84 | +0.03 |
| **mean (9)** | **0.426** | **0.493** | **0.765** | **0.810** | **0.850** | **0.853** | **0.80** | **+0.055** |
| wins / losses vs their all-features | 0 / 9 | 0 / 9 | 3 / 6 | 5 / 4 | 6 / 3 | **6 / 3** | | |

Notes on Table 5.5:
- **Wins and losses.** A tie counts as a win. The 3/6, 5/4 and 6/3 for the image, frame-OCR and final columns are from the source files. I recounted the text-only, fused-text and sentence-time columns from the two-decimal values in this table.
- **Their published audio-only mean** on the same 9 lectures is 0.47. Their 20-lecture averages are audio 0.53, OCR 0.76, image 0.64, SIFT 0.56, all features 0.82.
- **The +0.055 mean difference** is computed from unrounded F1. The two-decimal differences in the last column average +0.057.
- **The visibility gate generalises poorly.** It gained +0.031 on the tune half but −0.001 / +0.003 on test.

**Finding 16's verdict sentence, verbatim:**

> On MaViLS, combining speech-to-slide text, frame OCR and image features, taken from the frame at each sentence's timestamp, our alignment reaches a mean F1 of 0.853 on the 9 held-out lectures with video, against 0.80 for MaViLS's published all-features result on the same lectures (6 wins, 3 losses). The paired difference (+0.055, 95 % CI −0.050 to +0.161) is within lecture-to-lecture variation, so we report parity with their result, not an improvement.

---

## 6. Measurement rules, verbatim, with the case each caught

The rules are quoted verbatim from DESIGN.md. Each is followed by a case from the project history where it caught something; commit hashes were checked with `git cat-file -t`. "Before the first commit" means the case happened before commit `36068d7` and is recorded there.

### 6.1 *Measurement rules (binding)*

> **Every latency number must be:** measured on an **idle machine**, in a **single
> process**, with a **warm model**, **repeated at least 3 times**, and reported as the
> **median**. A number not taken this way is **not reportable** — do not put it in the
> paper, in a report, or in this file.

**Cases.**
- **Slide-guided ASR.** An early draft claimed slide-guided ASR "costs 2.1× decode time". That run was competing for cores with a bge-m3 embedding job. A clean re-run finished in 231 s against the plain run's 476 s. This happened before the first commit and is recorded in `36068d7`.
- **NLI model runtime.** A "~17×" runtime ratio for the large NLI model came from one run on a busy machine. It was removed from the NLI report in its own correction commit, `0758fe1`.
- **Still-open case.** DESIGN.md's "8.8 ms/query" states no protocol, so this review treats it as not reportable (§3).

> **Two results are presumed to be bugs until proven otherwise. Neither may be
> reported until it has been inspected.**
>
> 1. **An ablation condition whose metrics exactly equal the full system, or are
>    exactly zero.** Identical means the conditions are probably not separated;
>    zero usually means the weaker condition is handicapped by construction rather
>    than beaten on merit. Both have already happened here: `deictic` baseline
>    returned exactly the same 18 links as linkrag because it still added
>    `w_slide * on_slide` (it was *using* the alignment it was meant to be measured
>    against), and once that was fixed it returned exactly 0 because its score
>    ceiling sat below the shared threshold. Check the wiring, then check whether
>    the losing condition could reach the threshold at all, before believing either.

**Cases for rule 1.**
- **Deictic baseline.** The deictic baseline returned the identical 18 links, then exactly 0. The code fix is in `36068d7`, and the rule text was added in `0f38ebe`.
- **Later "identical" results, each inspected and explained rather than reported as effects:**
  - `linkrag/none ≡ baseline/none`: seed normalisation makes it so by construction (`187fc27`);
  - the modality gate producing identical 8-unit sets (`442875c`);
  - `linkrag ≡ baseline` on LectQA-Vid localisation, because one video gives no cross-file links (finding 8).

> 2. **A table whose per-category counts do not sum to the stated total.** Reconcile
>    before reporting. The link table once showed 51+22+5 against a stated 97 —
>    `build_graph` was keying edges by `link_type` and silently dropping 19 of 41
>    deictic links. The discrepancy was the only visible symptom.

**Cases for rule 2.**
- **`build_graph` dropping links.** It dropped 19 of 41 deictic links. This happened before the first commit, is recorded in `36068d7`, and the rule text came in `0f38ebe`.
- **Gold counts.** DESIGN.md said 24 / 71 / 5 while the stored gold said 25 / 69 / 4. Reconciled in `49be2d5` (known issue (g)).
- **Tables in this review.** Every count table was re-summed; the sums are shown next to each table.

> Correctness numbers (accuracy, WER, recall) are not covered by this rule, but their provenance still has to be stated — see the ASR and alignment sections.

### 6.2 *Reporting convention (binding, from 2026-09-23)*

> - **Every number carries its provenance:** dataset, split, n, corpus/config hash, the
>   models involved (answerer, judge), and how many runs. LLM-dependent cells run 3
>   times and report mean ± std. A single run, or a small n (e.g. the n = 4 cross-modal
>   cell), is labelled *direction only* / *not reportable*, never stated as a result.

**Case.** The NLI artefacts' headers said "judge gpt-4.1-mini, 3 runs". The verdicts were in fact one deterministic run of `cross-encoder/nli-deberta-v3-base`. The code fix is `9f8bbb9` and the report correction `d28685b`. Relatedly, the 24 / 71 / 5 gold counts came from a run in which the answerer judged itself (`49be2d5`).

> - **Thresholds and knobs are set only on a tune half or on external data,** then
>   applied once to the test half. Choosing a value after seeing evaluation numbers is a
>   **retune** and is reported as one, never as a result. A **design change** is a
>   mechanism with a config switch that reproduces the old behaviour, applied once
>   before re-measuring (see *Design changes*); a retune is not a design change.

**Cases.**
- **The visibility gate** was tuned on the MaViLS tune half. It is reported as not generalising (+0.031 on tune, ~0 on test) rather than being re-tuned (`95354ea`).
- **The Short range drop** (0.84 → 0.58) was deliberately left uninvestigated on the test half (`95354ea`).
- **The NLI formulation** (κ 0.22 → 0.27 → 0.31) is labelled a retune on the evaluation pairs (`9f8bbb9`).
- **pilot01's start prior** is labelled as fitted on the same 42 labels it is scored on.

> - **A report says what changed, then what was found, with caveats before headline
>   numbers.** Per-category counts sum to the stated total (*Measurement rules* §2).

**Case.** Findings 11–16 all open with "Caveats first". Finding 11 gained its "Caveat first" paragraph, and its retune label, in `9f8bbb9`.

> - **Corrections go in their own commit,** naming the number, what was wrong and the fix;
>   the corrected report keeps a dated note of the old value.

**Cases.** Correction commits:

| commit | correction |
|---|---|
| `bbf3570` | T1 reference numbers: similarity 74.42 (was 0.71), MCQ 53.43 (was 56.30), Table 6 F1 14.80 (was 19.62) |
| `7b52252` | LectQA-Vid MCQ accuracy (96.2 % on the first run) declared not reportable: option A in 1,484 of 1,489 |
| `4225b5c` | v2 localisation had scored 119 of 840 QA pairs with no usable gold interval, so its hit@k and IoU are understated |
| `5c8c976` | Faithfulness: iterative 5.4 → 5.3 % (a judge-cache replay artefact) |
| `9ac40f8` | Faithfulness: `parse_json` trailing-content bug; 33 verdicts moved; 4.6 / 5.3 / 5.6 → 3.7 / 3.1 / 4.3 % |
| `0758fe1` | The uncontrolled NLI runtime ratio was dropped |
| `d28685b` | NLI artefacts named the wrong verifier |
| `49be2d5` | pilot01 gold counts reconciled |
| `f5713d2` | A test assertion that passed only through an environment leak; both the assertion and the leak were fixed |

### 6.3 *Execution rule (binding)*

> **Every new script or code path must be executed once for real — not mocked — before it
> is considered complete.** Unit tests with stubs do not discharge this.

**Cases, as named in DESIGN.md.** In all three the test suite was green.
- **`scripts/ingest.py`** raised `NameError: Path` on its manifest write. The write was introduced in `6506c2e` without the import, and fixed in `ee04859`.
- **`run_regression.py --mode linkrag`** silently used baseline retrieval: `reports/audit_phase4.md` found it always called `retrieve_scored`. The dispatch was fixed in `6506c2e`, and `tests/test_run_regression_mode.py` pins it.
- **The faiss/torch OpenMP abort** (`OMP: Error #15`) was invisible to a stub encoder. It happened before the first commit, is recorded in `36068d7`, and led to the "No faiss" decision.

### 6.4 *Cost policy (binding from 2026-09-23)* — its enforcing line, verbatim

> - **`--max-cost` is required** on every batch script (…). `0` allows only cache replays and free backends. The spend guard prices each request before sending it and refuses anything that could exceed the budget.

**Cases.** The LectQA-Vid open-ended dry run estimated $7.43 against the item's $3.50 cap, so only the 300-question subset ran (`e90a3da`). Faithfulness ran one judge pass instead of three because three did not fit the cap (`181f171`). No report logs a mid-run refusal by the spend guard.

---

## 7. Known issues and limitations, ranked by threat to a claim

"Threat" means how much the issue could change or invalidate a stated result. High: it bounds or undermines a headline claim. Medium: it limits generality or precision. Low: hygiene, or documentation drift that could mislead a reader.

| # | issue | claim it threatens | threat | status |
|---|---|---|---|---|
| 1 | **The contribution statement has no reported target-benchmark support for components 2 and 3.** On LectQA-Vid, "ours" is iterative re-querying (the MI-RAG approximation) inside a different pipeline from the replica, so even the +5.47 is not attributable to one component; linking is inactive on a single video (finding 8). On MaViLS, only the `audio_slide` alignment is compared. Link-following and complementarity reranking are measured only on pilot01 (never reported; n = 25, 4 cross-modal). On LectQA-Vid localisation, complementarity cost 13 pp of hit@3 (68.1 → 55.4). | the contribution statement itself | **high** | open: needs a benchmark whose material does not fit in the context and spans files |
| 2 | **LectQA-Vid cannot measure retrieval value.** The full transcript beats every retrieval mode (+2.04 [+1.14, +2.99] over ours), and all 95 transcripts fit 2,048 tokens. | any retrieval claim on T1 | **high** | stated in finding 13 |
| 3 | **MaViLS test half examined in three rounds** (findings 14, 15, 16). Knobs were set on the tune half, but the decision to run further rounds, and what to try, was informed by test results. The tune half set: the image score, the text score, four weight sets and two gate settings. | the 0.853 | **high** | disclosed; a fresh held-out set would be needed to remove it |
| 4 | **n = 9 MaViLS lectures; the CI includes zero** ([−0.050, +0.161]); per-lecture differences span −0.22 to +0.34. | "exceeds" (hence "parity") | **high** | parity is the stated label |
| 5 | **T1's answerer is unnamed; their split and 1,000-pair subset are unpublished; the baseline is our replica.** The replica was validated against the equations (unit test), not against their outputs. Our choices fill M, L, merge gap, cross-encoder, index layout, and tesseract OCR instead of Gemini captions. | comparability with T1's numbers; fairness of the replica | **high** | only within-our-modes comparisons are claimed |
| 6 | **Open-ended results are on a 300-question subset**; the other 1,118 questions were not run (cost cap). | generality of Table 5.1 | medium | full-set means are TODO |
| 7 | **MCQ confounds** (99.7 % option A; 94.6 % longest option; a no-video picker scores 94.4 %). | any MCQ number | medium (claim withdrawn) | known issue (i) |
| 8 | **Timestamps usable for only 1,598 of 2,832 QA pairs**, 175 in videos 64–100. Earlier localisation scored everything (corrected `4225b5c`). | localisation numbers; the eq. 22 control | medium | known issue (h) |
| 9 | **The judge is the same vendor family as the answerer** (gpt-4.1-mini judging gpt-5.4-mini; on pilot01, gpt-4.1-mini judging gpt-5.4). Faithfulness used one judge run per check and was corrected twice. Gold is machine-verified; the human audit sheet is unfilled. | faithfulness rates; pilot01 gold | medium | direction only; known issues (d), (f) |
| 10 | **95 of 100 videos**; our own whisper transcripts, not theirs; 93 videos in the open-ended subset. | coverage | medium | stated |
| 11 | **One MaViLS test lecture (Climate policies) has no video**, so every visual number is on 9 lectures, not 10. | the 0.853 | medium | stated |
| 12 | **Unexplained cells.** Short range falls 0.84 → 0.58 with sentence-time frames (not investigated, to avoid tuning on test). ML for health is aligned correctly (F1 0.56) but rejected by the file-pair gate for an unconfirmed cause (finding 7). pilot01's `linkrag_iter` scores 71.3 % on single-modality questions with no mechanism (finding 5, flagged, not cited). | per-lecture stability; the gate's false rejections | medium | open |
| 13 | **Gates generalise poorly or have known blind spots.** The visibility gate gained +0.031 on tune and ~0 on test. The file-pair gate accepts topical neighbours (Solar resource audio × Climate policies deck, z = 3.5) and falsely rejects 15 % of related pairs. The modality-need gate changes nothing measurable. | cross-file linking on real multi-lecture courses | medium | finding 7 limitation |
| 14 | **Correctness defects found by this review (§9.1)**: LectQA-Vid transcripts are never frozen (a re-run re-transcribes; whisper is not deterministic across settings); `compare_retrieval.py` crashes after any model run *before* writing the ledger row or the report; `transcribe_openai.py` ignores `--max-cost`; `link.align.build_groups` in the config is read by no code. | reproducibility of every LectQA-Vid number; completeness of the ledger | medium | found 2026-09-25, not fixed (read-only review) |
| 15 | **Ledger vs dashboard.** Ledger $22.05 priced (plus 6 unpriced rows); the dashboard shows $19 across keys. The ledger over-counts by design (no cache discount; backfilled rows with estimated tokens, $7.56) and under-counts the unpriced rows. The net ~$3 gap is not reconciled row by row. | cost statements | low | open |
| 16 | **pilot01 is never reported** (dataset policy). Every pilot01 number in §3 is development evidence only. DESIGN.md does not say whether a project review may show a pilot01 table at all; this review does (§3), labelled development only, and the question is flagged for a decision. n = 25 with 4 cross-modal questions, highly redundant (only 4–5 of 16 proposed cross-modal questions survive verification). | nothing reported, but it is the only evidence for components 2–3 | low for reported claims (high for item 1) | policy |
| 17 | **Non-determinism.** Temperature 0 plus a seed is best-effort; run-to-run std is small (≤ 0.44 F1 points, ≤ 0.67 on any metric in Table 5.1a) but non-zero. | exact reproduction | low | 3 repeats reported |
| 18 | **Documentation drift.** DESIGN.md *Pipeline status* still says claim-level entailment is "not started" and "No target benchmark has been run yet"; both are done. DESIGN.md priority (i) still says the answer metrics are "on the first 28" videos. README says the default config points at Ollama, but it is OpenAI gpt-5.4-mini-2026-03-17. README lists `--mode baseline\|linkrag\|iterative\|linkrag_iter` beside `ask.py`, which accepts only `baseline` and `linkrag` (all four exist in `compare_retrieval.py`). `run_regression.py` and `reports/regression.md` are still titled "Q1–Q4" though the set is 25 questions. `reports/results_vs_target_papers.md` §2b has two TODOs this review resolves (§5.4). DESIGN.md's *Design changes* table gives `align.relatedness_z` as 2.0; the config and finding 7 say 1.27. Findings 8 and 9 were not edited after `4225b5c` declared their localisation numbers understated. Finding 15 still says sentence-time frames were skipped "because the videos are no longer available"; `0a049ee` recovered them. `docs/pilot01_history.md` still says "209 units, 354 links" (the count before the deictic fix; now 311) and gives deictic precision 6/6 and recall 3/5, superseded by 80 % and 86 %. `reports/deictic_eval_pilot01.md` has a template slip ("Tier 3 scores 80% against tier 3's 80%"). The 69 pilot01 gold items are called "units" in the gold report and "locators" in DESIGN.md. | a reader's understanding | low | open, listed so it is not mistaken for current state |
| 19 | **No reportable latency number exists.** DESIGN.md's 8.8 ms/query has no stated protocol, and this review's profiling ran on a loaded machine, so its wall times were discarded (§9.2). | any efficiency claim, e.g. "CPU-first, runs on a laptop" | low | re-time on an idle machine before any latency claim |
| 20 | **Older known issues** (DESIGN.md): (a) evidence units can span a speaker change (a false author with a valid citation on pilot01); (c) local faster-whisper turns "top-K" into `type -k`; faiss was dropped (OpenMP conflict) for exact numpy search, which needs an approximate index above ~200k units. | pilot01 answers; scale | low | not fixed |

---

## 8. State of deliverables

| deliverable | state | evidence |
|---|---|---|
| Paper | **not started** | `paper/` contains only `bibliography.bib`. `reports/results_vs_target_papers.md` (commit `7381b67`) is a draft of the results section |
| UI | **not started** | `src/linkrag/ui/__init__.py` is a 5-line placeholder; DESIGN.md *Pipeline status* lists `ui` as not started |
| Progress reports | **delivered** | Technical progress report (PDF, dated 2026-09-11) and the weekly supervisor annexure (DOCX/PDF), committed in `2999d8a` (2026-09-23). Since `10c7322` they are kept local and untracked, but they remain in the public history. Results-vs-targets draft: `.md` pushed, PDF version kept local |
| Code, results, reports | **pushed** | `origin/main` = `10c7322` on github.com/m1ssmee/LinkRAG (public), 87 commits. The tag `v0-pilot` exists locally only |
| Target benchmarks | T1: audit, replication and full-context ceiling done; full-set run not done. T2: parity result done | §5 |
| Priority list (DESIGN.md) | (i) full set: 95/100 acquired, open-ended on a 300-question subset. (ii) metrics: done. (iii) frame slides: done, no gain. (iv) faithfulness: done, direction only. (v) MaViLS: done, extended to video | DESIGN.md *Priority order*, findings 12–16 |

---

## 9. Code health and performance (read-only)

### 9.1 Code health: YAGNI audit at the strictest level (read-only; nothing was edited)

**Scope and method.**
- **Scope:** `src/`, `scripts/` and `tests/` at `10c7322`: 18,008 lines of Python.
- **Method:** every file read. `ruff --select F,ARG,SIM,PERF,C4,RET` and `vulture --min-confidence 60` were run as aids. Every candidate's line and callers were checked against the current file.
- **Earlier audit:** items 1–4 and 6 of `reports/audit_ponytail_2026-09-24.md` are applied (`85cd84a`, `6d0ab2e`, `75dacec`, `95f6772`, `9e6789e`), and nothing of them remains. Its item 5, the UI placeholder, is kept for the UI task.
- **Categories:** *delete* dead code; *stdlib*/*native* for something the standard library, numpy/sklearn/networkx or an in-repo helper already provides; *yagni* for single-use abstraction; *shrink* for the same logic in fewer lines.
- **KEEP rule:** recorded ablation switches, standing instruments, measurement tooling, and anything a recorded number depends on are marked **KEEP**.

**Candidates, biggest cut first.**

| file:line | category | cut → replacement | Δ lines | risk to a recorded result | verdict |
|---|---|---|---:|---|---|
| `scripts/check_link_types.py:1` (+ `tests/test_link_types.py:25-32`) | delete | One-off survey from before link-type validation was switched on (`76a79a6`); `Link`/`load_links` now raise on retired types | −53 | none | CUT |
| `scripts/adapters/mavils.py:1545-1749` | shrink | One shared tune→test evaluator for the fused / visual / visual-ocr / visibility-gate / final-round commands | ≈ −60 | **high**: re-renders `mavils_tuned.json` and six recorded MaViLS reports, which would have to be byte-identical | **KEEP** (recorded numbers depend on it) |
| `src/linkrag/generate/citations.py:82-117` | shrink | `clip_audio`'s decode/resample/mux duplicates `ingest/asr_openai.py:90 cut`; call `cut(src, start, end, out)` | −22 | none: clips are not measured; the end boundary moves from `>` to `>=` | CUT |
| `scripts/adapters/mavils.py:406,481,539,691,765,987,1241,1295,1461,1539,1639,1746` | shrink | 12 commands repeat the "write report, print, return" tail; return the lines and write once in `main` | −20 | none: report text unchanged | CUT |
| `src/linkrag/eval/__init__.py:17-33` | yagni | Re-export layer used by 3 scripts; import from `linkrag.eval.metrics` | −19 | none | CUT |
| `src/linkrag/ingest/audio.py:27-35,166-186,262` | delete | `Segment` start/end/text are never read, only `.words`; return `list[Word]` | −18 | none | CUT |
| `src/linkrag/ingest/vlm_caption.py:91-107` (caller `pdf.py:390-392`) | yagni | `vlm_config` repeats `describe_image`'s defaults; call `describe_image` with the config section directly | −15 | none: off by default, never run for a result | CUT |
| `src/linkrag/link/graph.py:101-127` | native | Hand-rolled BFS in `subgraph_around` → `nx.ego_graph` over a filtered `subgraph_view` | −14 | none: plotting only | CUT |
| `scripts/ask.py:69-91` | shrink | Re-implements `retrieve.iterative.retrieve_pool`; call it | −14 | none: same arguments | CUT |
| `tests/test_link_phase3.py:22-36` vs `tests/test_deictic_tiers.py:24-38` | shrink | Duplicate unit factories → one set in `conftest.py` | −14 | none: tests only | CUT |
| `scripts/run_regression.py:34-60` | shrink | `retrieve_for_mode` re-implements `retrieve_pool` | −10 | low: a standing instrument; the next phase run must match | CUT |
| `src/linkrag/eval/verify_gold.py:183-190` | delete | `locator_str` ≡ `generate.citations.locator`; import it | −10 | none: judge prompt text byte-identical (cache keys still hit) | CUT |
| `src/linkrag/ingest/docx.py:32-51` | shrink | Copy of `pdf.chunk_page`'s sliding window | −10 | none: no DOCX in any recorded corpus | CUT |
| `scripts/transcribe_openai.py:67-79` | shrink | Hand-rolled cost pre-check → `set_max_cost(cfg, args.max_cost)` (also fixes note 2 below) | −10 | none | CUT |
| `audio.py:188`, `baseline.py:53`, `retrieve/linkrag.py:178`, `mavils.py:211`, `lectqa/common.py:14`, `lectqa/acquisition.py:8` | shrink | Runs of ≥ 3 blank lines, trailing blanks | −9 | none | CUT |
| `scripts/eval/judge_agreement.py:36-51` | stdlib | Hand-rolled env save/restore → `unittest.mock.patch.dict(os.environ, …)` | −8 | none | CUT |
| `scripts/adapters/lectqa/metrics.py:49-62` | shrink | Inline baseline/linkrag retrieval → `retrieve_pool(…)` then `rerank` | −8 | low: `lectqa_theirs_metrics.md` is a cache replay; prompts must stay identical | CUT |
| `src/linkrag/link/align.py:107-113` | delete | `Alignment.n` / `.m` read only by one test | −8 | none | CUT |
| `src/linkrag/retrieve/iterative.py:42,44-46,74,110` | delete | `IterativeResult.ids` and `latency_s` read only by tests; the latency is not taken per the measurement rules anyway | −8 | none | CUT |
| `src/linkrag/eval/lectqa_metrics.py:113-130` | native | Hand-rolled macro P/R/F1 → `sklearn.metrics.precision_recall_fscore_support(average="macro", …)` | −7 | low: float summation order in the MCQ macro-F1 (reported at 2 dp) | CUT |
| `scripts/eval/compare_entailment.py:32-39` | native | `kappa` → `sklearn.metrics.cohen_kappa_score` | −6 | low: identical at 2 dp; on all-equal labels sklearn returns nan where the current code returns 1.0 | CUT |
| `src/linkrag/link/graph.py:152-162` + `link/deictic.py:357-367` | shrink | Two copies of "count and mean score per key" | −6 | none | CUT |
| `src/linkrag/ingest/__init__.py:118-121` | yagni | `transcripts_used()` wraps one `getattr`, one caller | −6 | none | CUT |
| `tests/test_ingest.py:160-164` | delete | Test that only checks the fixture | −6 | none | CUT |
| `pipeline.py:45-49`, `negative_control.py:76-78`, `mavils.py:781-783, 925-927` | shrink | The same `align_monotonic` partial copied 4 times → one `monotonic_decoder(a)` | −5 | low: the gate copies omit `flatness_scaling` (0.0 today) | CUT |
| `scripts/make_alignment_labels.py:67-91` | shrink | Flag juggling for `ambiguous` → one expression | −5 | none: labels regenerate identically | CUT |
| `src/linkrag/generate/answer.py:272-274` | delete | `Answer.unsupported` never read | −4 | none | CUT |
| `eval/redundancy.py:92-95` vs `eval/verify_gold.py:205-208` | shrink | Identical reading-order key → one function | −4 | none | CUT |
| `scripts/adapters/mavils.py:255-257` | delete | `_tokens` ≡ `lectqa_metrics.tokens` | −4 | none | CUT |
| `scripts/adapters/mavils.py:1118-1123` vs `1182-1186` | shrink | Two copies of the sha1-keyed OCR disk cache → one helper with `scale` | −4 | none: same cache files | CUT |
| `tests/conftest.py:147-148` | delete | Re-registers the `slow` marker already in `pyproject.toml` | −4 | none | CUT |
| `scripts/negative_control.py:51-55`, `ingest/pdf.py:340-341` | shrink | Append loops → `Counter` / `list.extend` | −4 | none | CUT |
| `scripts/compare_retrieval.py:202-205` vs `269-274` | shrink | `fmt` duplicates `fmt_runs(pct=True)` | −4 | none | CUT |
| `src/linkrag/link/deictic.py:67-70, 237-240` | shrink | Loops → literal / comprehension | −4 | none | CUT |
| `src/linkrag/manifest.py:27-33` | stdlib | Hand-streamed sha256 → `hashlib.file_digest` (Python 3.11) | −3 | none: same digest | CUT |
| `src/linkrag/link/align.py:125-132` | stdlib | Document frequency by hand → `Counter` | −3 | none | CUT |
| `src/linkrag/ingest/pdf.py:410-413` | stdlib | Dict count → `Counter` | −3 | none | CUT |
| `src/linkrag/ingest/pdf.py:256,273-274` | delete | No-op `try/finally: pix = None` | −3 | none | CUT |
| `src/linkrag/ingest/__init__.py:129,148-149` | yagni | `skip_failures` parameter no caller sets | −3 | none | CUT |
| `src/linkrag/eval/verify_gold.py:350-354` | shrink | if/else → conditional expression | −3 | none | CUT |
| `src/linkrag/link/figure_text.py:58-59,171-173` | yagni | Single-element set constants → string compares | −3 | none | CUT |
| `src/linkrag/link/graph.py:139-146` | shrink | Two identical stringify loops → one | −3 | none | CUT |
| `scripts/adapters/mavils.py:96-102` | yagni | `their_prf` callers read only F1; return F1 | −3 | none | CUT |
| `scripts/dataset/candidate.py:85-89` + `redundancy.py:76-80` | shrink | Identical "files per role" block → one helper | −3 | none | CUT |
| `src/linkrag/eval/lectqa_metrics.py:59-61,109` | delete | `rouge1` equals `token_prf[1]`, already averaged | −3 | none: identical floats, same order | CUT |
| 5 test files (`test_rerank.py:5`, `test_costs.py:6`, `test_link_phase3.py:9`, `test_ingest.py:8`, `test_verify_gold.py:94`) | delete | Unused imports | −3 | none | CUT |
| `scripts/adapters/mavils.py:679-684, 751-757` | delete | Accumulated values never read | −3 | none | CUT |
| `src/linkrag/generate/answer.py:105,148` | in-repo helper | Hand-written usage dicts → `costs.empty_usage()` | −2 | none | CUT |
| `src/linkrag/link/relatedness.py:108-118` | shrink | `apply_verdicts` rebuilds the list it mutates | −2 | none | CUT |
| `src/linkrag/link/visual.py:139-144` | shrink | `confident_rate` recomputes the margin → `np.mean(frame_margin(fp) > margin)` | −2 | none | CUT |
| `scripts/ingest.py:76-78` | stdlib | Modality count → `Counter` | −2 | none | CUT |
| `scripts/adapters/mavils.py:629-635` | shrink | Decoder repeats `decode(…, variant="dp")` | −2 | none | CUT |
| `scripts/adapters/lectqa/common.py:52-62` | shrink | Two identical loops in `load_qa` → one | −2 | none | CUT |
| `scripts/adapters/lectqa/mcq.py:48-50` | shrink | Wrapper → `retrieve_pool("baseline", …)` | −2 | low: MCQ prompts must stay identical | CUT |
| `src/linkrag/link/deictic.py:87` | delete | `TERMINAL_PUNCT` ≡ `ingest.audio.TERMINAL` | −1 | none | CUT |
| `src/linkrag/core.py:174` | shrink | Redundant copy of a fresh kwargs dict | −1 | none | CUT |
| `src/linkrag/eval/nli.py:74-75` | yagni | `margin` / `model` kwargs no caller passes | −1 | none | CUT |
| `src/linkrag/eval/redundancy.py:70` | delete | Whitespace normalisation repeated per chunk | −1 | none | CUT |
| `src/linkrag/index/__init__.py:101` | delete | Stale comment (argpartition) above a stable argsort | −1 | none | CUT |
| `src/linkrag/ingest/pdf.py:198-199` | shrink | Unpack bbox in one line | −1 | none | CUT |
| `scripts/adapters/mavils.py:319` | delete | `seg_texts = sentence_texts` alias | −1 | none | CUT |
| `src/linkrag/generate/verify.py:73` | shrink | `t.update({k: v for …})` → `t.update(counts)` | 0 | none | CUT |
| `src/linkrag/ingest/pdf.py:214` | delete | Unused `page` parameter of `caption_regions` | 0 | none | CUT |
| `src/linkrag/link/relatedness.py:65-73` | shrink | `_where` is a third locator formatter (truncates where `citations.locator` rounds) | −8 | **high**: changes the judge prompt, so cached gate verdicts miss and the recorded pass rates stop reproducing | **KEEP** |
| `src/linkrag/link/align.py:105, 334` | delete | `Alignment.total_score`, `path_score` reached only by tests | −6 | none | **KEEP** (gate tests; earlier audit) |
| `src/linkrag/costs.py:206,216` | delete | `cumulative()["audio_seconds"]` never read | −1 | none | **KEEP** (cost ledger) |
| `src/linkrag/ingest/__init__.py:37` | delete | Unused `mode` argument | 0 | none | **KEEP** (two-mode rule) |
| `scripts/adapters/mavils.py:306` `decode_with_builds` | delete | Not on any default path | — | **high**: `reports/mavils_final.md` | **KEEP** (recorded build-groups switch) |
| `scripts/adapters/lectqa/localisation.py:183-196` | delete | Reads the retired first-run jsonl | −14 | **high**: `lectqa_v2.md` §3 | **KEEP** (recorded report) |
| Non-default branches: `retrieve/linkrag.py:117` (evict), `:132` (normalise_seeds), `ingest/audio.py:265` (fixed), `retrieve/rerank.py:213,230` (none/mmr), `align.py:237` (naive), `:250` (start prior), `:251` (flatness), `:395` (min_segment_sim), `:437` (similarity options), `rerank.py:197` (modality gate), `verify_gold.py:259` (nli), `mavils.py:1309` (frame_source) | delete | — | — | **high** | **KEEP** (recorded ablation switches; two-mode rule) |
| `eval/metrics.py:23`, `scripts/run_regression.py` | — | — | — | — | **KEEP** (standing instruments) |

**Totals.** The CUT candidates add up to about **−403 lines** (2.2 % of 18,008), with no dependency added. Two rows (MCQ macro-F1, κ) would use scikit-learn. It is already imported by `mavils.py` and `link/visual.py` but is only installed transitively, so it should be pinned in `requirements.txt` first.

**Checked and found lean:**
- `core.py`, `costs.py`, `manifest.py` (one row each at most);
- `generate/citations.py` (apart from the `cut` reuse), `generate/verify.py`;
- `eval/metrics.py`, `eval/nli.py`;
- `index/__init__.py`;
- `ingest/image.py`, `ingest/video_slides.py`, `ingest/asr_openai.py`;
- `link/same_slide.py`, `link/pipeline.py`;
- `retrieve/baseline.py`, `retrieve/linkrag.py`, `retrieve/rerank.py`;
- `scripts/build_links.py`, `gate_links.py`, `eval_alignment.py`, `eval_deictic.py`, `plot_alignment.py`, `eval/verify_gold.py`, `eval/audit_sample.py`, `dataset/propose_questions.py`;
- `adapters/lectqa_vid.py` and `lectqa/{acquisition, transcription, frameslides, open_ended}.py`.

**Correctness defects found in passing.** These are not cuts. Each was confirmed by reading the code; none was executed, because executing 1 or 2 needs a paid call.

1. **`scripts/compare_retrieval.py:286`** rebinds `llm` (the model config, line 113) to a float (`n, per_run, llm = got`). Line 311, `llm.get("model")`, then raises `AttributeError` at the end of any run that called a model. That is *before* `record_run` writes the ledger row and before `--out` is appended, so such a run would spend money with no ledger row and no report. Introduced in `187fc27`. The review has not established whether any recorded LLM run of this script hit it. `--no-llm` runs skip line 311.
2. **`scripts/transcribe_openai.py`** never calls `set_max_cost`. The spend guard inside `asr_openai.transcribe` therefore sees the $0 default and refuses every file, whatever `--max-cost` is given.
3. **`scripts/adapters/lectqa/transcription.py:62`** passes `freeze_to` with the local ASR backend. `ingest_audio` writes a frozen transcript only on the hosted-backend branch (`src/linkrag/ingest/audio.py:249-256`), so no LectQA-Vid transcript is frozen. A re-run would re-transcribe, and whisper is not deterministic across settings. This is a reproducibility risk for every LectQA-Vid number.
4. **`scripts/plot_graph.py:22`** `EDGE_STYLE` has no `same_slide` entry, so those edges are not drawn.
5. **`link.align.build_groups`** in `configs/default.yaml` is read by no library code. The switch is the MaViLS adapter's own argument, so the config key does nothing.

### 9.2 Performance: three hot paths, profiled (nothing applied)

**No latency number appears in this section.** DESIGN.md's latency rule requires an idle machine and forbids non-compliant numbers "in the paper, in a report, or in this file".

The profiling was set up as the rule requires:
- single process, models warm;
- one untimed warm-up, then 3 timed repeats with the median taken;
- a 4th repeat under cProfile.

But the machine was **not idle**. The load average was 3.1–3.6 during path 1 and 3.1–4.5 during paths 2–3, and a system media-analysis daemon held about 104–113 % CPU throughout paths 2–3, even after a 3-minute wait. The wall times were therefore discarded. What remains is **where the time goes**: shares of one profiled run and call counts. Those shares also come from a loaded machine, so they need one idle-machine profile before anything relies on them.

- **Setup:** Apple M2, 8 cores (4P + 4E), 8 GB RAM, Python 3.11.15, bge-m3 on the CPU.
- **$0:** invalid API key; `requests.post` replaced by a function that raises, which never fired; spend budget $0.
- **Read-only:** outputs went to a scratch directory. A before/after snapshot of `data/processed/`, `results/`, `reports/` and `configs/` showed no change.

**Encoder determinism probe.** This is the basis for every "would it change output" answer below.
- The same single query encoded twice gives bitwise-identical vectors, and so does the same 47-figure list.
- A query alone vs inside a 25-query batch differs by up to 1.5e-7. So do a subset vs the same rows of a full-list call, and the stored index vs a fresh encode.
- So repeating an identical encoder call is output-neutral, while changing batch composition moves embeddings by ~1e-7 and needs an equivalence check.

#### Path 1 — `build_links` on pilot01 (`link.pipeline.link_corpus`)

The corpus is 209 units: 51 audio, 111 text (27 of them deck slides), 47 figure.
- **Output check.** 312 lines each (311 links + `_meta`). The output is **identical to the local on-disk `data/processed/links.jsonl` once `metadata.relatedness` is removed**. That field is the LLM verdict that `gate_links.py` adds afterwards. `_meta` is identical (`manifest_hash 2f3b35f27e86caf8`). `data/processed` is not tracked, so this compares against the local file, not a committed one.
- **Excluded:** model and index load; writing links, graph and the deictic CSV. `build_links.py` writes into `reports/`, so `link_corpus` was called in-process.
- **Where the time goes** (shares of the profiled run):
  - encoding is **99.9 %**;
  - **`figure_text.document_pair_gate` is 77.7 %**: 2 document pairs × (1 true statistic + 5 word-shuffled re-embeddings) = 12 encoder calls;
  - `figure_text_scores` 18.0 %, `resolve_deictic` 2.4 %, alignment `similarity_matrix` 1.8 %;
  - repo code's own self time is below 0.1 % (`align_monotonic`, 32 calls).

Top 5 by cumulative share:
- `link/pipeline.py:31 link_corpus` 100 % (1 call);
- `index/__init__.py:39 encode` 99.9 % (20);
- `link/figure_text.py:239 link_figures_to_text` 95.8 % (1);
- `figure_text.py:199 document_pair_gate` 77.7 % (2);
- `figure_text.py:215 stat` 76.5 % (12).

| top 5 by self time | share | ncalls | proposed change | changes output? |
|---|---:|---:|---|---|
| torch `linear` (bge-m3 forward) | 77.7 % | 5,945 | Cache the `document_pair_gate` raw score and null list, keyed by a hash of figure and text contents, seed, shuffle count, model id and library versions; keep z and the threshold outside the key | **No**: deterministic in those inputs; repeat encodes are bitwise identical |
| torch `scaled_dot_product_attention` | 14.7 % | 984 | Take the gate's true statistic from the dense matrix `figure_text_scores` already computed, instead of re-encoding per-document subsets | **Possibly**: subset vs full-list embeddings differ by ≤ 1.5e-7. This moves only the gate z; links change only if a pair crosses z = 1.27 |
| torch `gelu` | 4.1 % | 984 | Memoise identical encoder calls inside `link_corpus`: `resolve_deictic` (`deictic.py:234`) re-encodes the 47-figure list that `figure_text_scores` (`figure_text.py:128`) already encoded | **No**: identical list, bitwise identical |
| torch `layer_norm` | 0.9 % | 2,009 | bf16/int8 inference | **Yes**: every embedding changes. Rejected |
| torch `embedding` | 0.8 % | 123 | Tune torch threads (4) or encode batch size | Batch size **yes** (padding changes floats); threads **unknown** (GEMM reduction split). Needs the byte-compare |

#### Path 2 — `compare_retrieval --no-llm --max-cost 0`, 25 regression questions (2 modes × 3 rerank methods)

- **Output check.** The in-process run and a real CLI run produce identical `--out` files (`diff`).
- **Excluded:** model and index load; `iterative` and `linkrag_iter`, which need an LLM.
- **Where the time goes:**
  - 50 query encodes (each question encoded once by baseline and again by linkrag), plus 1 warm-up: **96.1 %**;
  - all 150 reranks together: 1.1 %.

Top 5 by cumulative share:
- `compare_retrieval.py:85 main` 100 %;
- `index/__init__.py:39 encode` 96.1 % (51 calls);
- `retrieve/iterative.py:150 retrieve_pool` 95.8 % (50);
- `retrieve/baseline.py:35 retrieve_scored` 95.6 % (50);
- `retrieve/linkrag.py:71 retrieve_linkrag` 47.9 % (25).

| top 5 by self time | share | ncalls | proposed change | changes output? |
|---|---:|---:|---|---|
| torch `linear` | 79.9 % | 7,395 | Encode each question once per run and reuse the vector across modes (memo on the exact string; still a single-text call). Removes half of the query encodes | **No**: bitwise identical on repeat |
| torch `Module._apply` | 1.5 % | 22,848 | The embedding library re-applies `.to(device)` over every submodule on each encode; the memo halves it | **No** |
| torch `gelu` | 1.5 % | 1,224 | Same memo. Batching all 25 questions would do more | memo **no**; batching **possibly** (~1e-7 shifts can flip rank ties) |
| torch `scaled_dot_product_attention` | 1.3 % | 1,224 | Same memo | **No** |
| torch `layer_norm` | 1.1 % | 2,499 | Same memo; lower precision rejected | memo **no**; precision **yes** |

The reranker's own self time is 0.3 % over 150 calls. Its greedy loop re-sums over the chosen set each round; an incremental sum is not worth the change.

#### Path 3 — one `ask.py` query, warm, up to the generation call (Q1, complementarity, k = 8, pool = 20; baseline and linkrag)

- **Timed:** `Index.load` (for linkrag also `load_links` and `build_graph`), then query encode, retrieval, rerank and the modality report.
- **Excluded:**
  - model load;
  - **generation and claim verification**: both call paid hosted models, and `ask.py`'s completer has no reply cache, so nothing can be answered at $0 without a network call;
  - citation rendering and the ledger write, which follow generation.
- **Modality of the retrieved 8** (Instrument #1): baseline text 3 / figure 2 / audio 3; linkrag text 3 / figure 3 / audio 2.
- **Where the time goes** (both modes, same order): the single query encode is the largest item. Then come `Index.load`, which rebuilds BM25 by tokenising all 209 units on every load, and `tokenize` (210 calls). `build_graph` and `load_links` are negligible.

| top 5 by self time (baseline; linkrag has the same order) | ncalls | proposed change | changes output? |
|---|---:|---|---|
| torch `linear` (query encode) | 145 | None for a fresh question (model floor); a per-string memo only helps repeats | memo **no**; lower precision **yes**, rejected |
| `index/__init__.py:65` genexpr in `tokenize` | 137,342 | `re.findall(r"[^\W_]+", text.lower())` instead of a per-character generator | **No**: 0 mismatches over all 209 units, the 25 questions and every printable BMP character |
| `str.join` (same line) | 217 | Removed by the same change | **No** |
| `rank_bm25 _initialize` (BM25 refit in `Index.load`) | 1 | Keep the index resident in a long-lived process (the planned UI) | **No** |
| `str.isalnum` | 137,132 | Removed by the tokenize change | **No** |

### 9.3 The three changes I would make first

Gains are stated as shares of the profiled path, not as times (§9.2).

| # | change | expected gain | equivalence check it needs |
|---|---|---|---|
| 1 | **Cache the `document_pair_gate` statistics** (raw score + null list), keyed by content hash, model id, seed, shuffle count and library versions | Removes the gate's 77.7 % of profiled `build_links` time on a warm cache | Byte-compare `links.jsonl` on pilot01 (with `metadata.relatedness` removed, `_meta` included) on **both** a cold-cache and a warm-cache run; the full test suite |
| 2 | **Memoise the query embedding per exact string** within a process, shared by baseline and linkrag (no batching) | Removes half of the query encodes that make up 96.1 % of `compare_retrieval --no-llm` | `compare_retrieval --no-llm --max-cost 0` `--out` files identical on the 25 questions (`diff`), as done here for CLI vs in-process |
| 3 | **`tokenize` via `re.findall`** | Removes ~137k generator calls per corpus tokenisation; this is on `Index.load` (every ask) and on the BM25, keyword and deictic paths of `build_links` | Token-list equality on all units, questions and every printable BMP character (0 mismatches already found); then `compare_retrieval --no-llm` identical and `links.jsonl` byte-identical on pilot01 |

**Runner-up:** the identical-list encoder memo inside `link_corpus` (output-neutral). Before any of these is claimed as a speed-up, time it under the latency rule on an idle machine.

**Correctness fixes before any of this** (from §9.1):
- `compare_retrieval.py:286`: a model run crashes before the ledger row is written;
- LectQA-Vid transcripts are never frozen;
- `transcribe_openai.py` ignores `--max-cost`.

The first two threaten recorded results more than any speed-up would.

---

## 10. Research-integrity review of this document

This document was checked by the project's read-only research-integrity reviewer. The reviewer checks a change and the latest report against DESIGN.md's measurement, reporting, tuning, priority and cost rules, and never edits anything. It ran twice. The first pass reviewed the draft. The second pass reviewed the draft after the first pass's fixes. Both verdicts are reproduced **verbatim** below. Line numbers in them refer to the version each pass read, so they have since shifted. After the second pass, every fix it listed under "Fix first" and "Numbers that don't reconcile" was applied (listed at the end). No third pass was run.

### 10.1 First pass (on the draft) — verbatim

**Verdict: Commit with fixes.** I found no fabricated or unreconcilable headline number, but a few factual slips and one latency-rule problem need fixing first.

I recomputed these and they match their sources: Tables 5.1a/b, 5.2, 5.3, 5.4 and 5.5, including the win/loss recounts; the ledger ($22.05 over 55 priced rows, 6 unpriced, split by model); 87 commits; 18,008 lines; the −403-line total; all 41 cited commit hashes; and the defect at `compare_retrieval.py:286` → `:311`, which I confirmed by reading the code.

**2. Numbers that don't reconcile**
- `project_review_2026-09-25.md:351`: the review says the MCQ run used "replica retrieval top-4". It actually used our RRF baseline (dense + BM25), per `results/external/lectqa_mcq_shuffled.md:3` and `scripts/adapters/lectqa/mcq.py:46` (`retrieve_scored`).
- `:52`: "5 priced rows ($7.89) … estimated token counts". Only 4 rows carry estimated tokens: ledger rows 1–4, $4.586 + 0.226 + 1.811 + 0.939 = **$7.56**. The 5th is the MCQ backfill ($0.33), and its `estimated_tokens` is 0. The correct wording is "5 backfilled rows ($7.89), 4 of them estimated ($7.56)".
- `:257`: citation correctness is given as "89–91 %". The three rows are 90.6, 89.0 and 87.6 (`reports/regression.md:1693-1695`), so the range is 88–91 %.
- `:716`: 51 audio + 27 slides + 111 text + 47 figure = 236, but line 234 says 209 units. The 27 deck units are inside the 111 text units, so the wording should say "of which".
- `:722`: "committed-state `data/processed/links.jsonl`" is wrong. `data/processed/*` is gitignored, so the comparison was against the local on-disk file.

**3. Rules violated**
- *Measurement rules*: a non-compliant latency number must not go "in the paper, in a report, or in this file". §9.2 (`:720, :739, :760`) and §9.3 (`:785-787`) put absolute timings from a loaded machine into a report. They are clearly labelled NOT REPORTABLE, but DESIGN.md forbids including them at all, and the label does not override that. `:776` ("5.9 → 3.1 ms") has neither that label nor a repeat count or median. Fix: keep the shares and call counts and drop the seconds, or add an explicit profiling exception to DESIGN.md.
- *Reporting convention*, "caveats before headline numbers": the §1 verdict table (`:36-42`) comes before the caveat block (`:44-47`). The T2 row (`:42`) and §5.5 also leave out the three-round test-half exposure, which §7 #3 itself ranks as a **high** threat.
- *Reporting convention*, provenance: Tables 5.1a, 5.3 and 5.5 carry no corpus/config hash. The gap is inherited from the source files, but it should be stated.
- *Pilot corpus*: "it never appears in a reported table". The §3 table is a table of pilot01 numbers in a document handed to an external reviewer. Every cell is labelled development/direction only, so this is a tension rather than a breach, but DESIGN.md should say whether a project review is allowed to do this.
- `:802`: the `<<VERDICT>>` placeholder must be filled before commit.

**4. Claims that outrun the evidence**
- "Our retrieval beats our replica of T1's pipeline" (`:40`). The evidence is +5.47 [+3.80, +7.24], but it compares two whole pipelines that differ in:
  - embedder (bge-base vs bge-m3);
  - BM25;
  - chunking;
  - an extra LLM follow-up query;
  - the eq. 22 oracle filter, which leaves 34 of 160 questions with an empty context.

  So the gap cannot be attributed to re-querying or to "our retrieval". Attributing it needs our `baseline` (RRF, no re-query) on the same 300 questions, and the replica run without eq. 22.
- The latency-share argument at `:709`: "shares … large enough (77 %, 96 %) not to depend on contention". This is an argument, not a measurement: the shares come from one cProfile run under load. It needs one idle-machine profile.

**5. Record in DESIGN.md**
- New known issues:
  - `compare_retrieval.py:286` crashes before `record_run`;
  - `transcribe_openai.py` ignores `--max-cost`;
  - LectQA-Vid transcripts are never frozen, which is a reproducibility risk for findings 12 and 13;
  - `link.align.build_groups` is a config key that no code reads;
  - the ledger ($22.05) vs dashboard ($19) gap is not reconciled.
- *Cost accounting* design change: `reports/llm_ledger.jsonl` has been untracked since `10c7322`. Say where spend provenance now lives (the per-report cost footers).
- Fix documentation drift:
  - `relatedness_z` in the *Design changes* table (2.0 → 1.27);
  - *Pipeline status*;
  - the "first 28" text in priority (i);
  - the stale "videos no longer available" caveat in finding 15;
  - a dated note on findings 8 and 9 pointing to `4225b5c`.
- Remove or re-measure the "8.8 ms/query" in *Environment decisions*. It breaks the latency rule's "or in this file".

**6. Fix first**
1. Correct `:351` (it was RRF baseline retrieval, not the replica) and state that the T1 +5.47 is a whole-pipeline difference, not a gain from re-querying.
2. Take the absolute latency numbers out of §9.2 and §9.3 (keep the shares), or get DESIGN.md to allow them explicitly. Label `:776`.
3. Move the §1 caveats above the verdict table, add the three-round test-exposure caveat to the T2 headline and §5.5, fix the `:52`, `:257`, `:716` and `:722` slips, and fill §10.

### 10.2 Second pass (after the first pass's fixes) — verbatim

1. **Verdict:** Commit with fixes. Of the nine first-pass fixes, eight are made correctly. The pilot01-tension fix is only half done (see item 3 below). The remaining faults are small wording and number slips; none changes a headline.

**What I checked (all in `/Users/main/Desktop/Major project 1/reports/project_review_2026-09-25.md`)**
- MCQ retrieval: line 356 now matches `results/external/lectqa_mcq_shuffled.md:3` (RRF dense + BM25, top-4). The $0.39 reconciles: 0.061 + 0.330 in the ledger.
- Whole-pipeline wording is at lines 40, 48, 310 and §7 #1. The absolute latencies are gone from §9.2/§9.3, which now show only shares and call counts. The caveats (lines 36–42) sit above the verdict table. Three-round exposure is stated at lines 41, 50 and 419.
- Ledger, recomputed: 61 rows, 55 priced, $22.0513. The per-model sums match. 5 backfilled rows = $7.892 (the fifth is the MCQ interrupted-attempts row). The 4 rows with estimated tokens = $7.562. No row is dated after 2026-09-23T18:39, which is consistent with "$0 since 2026-09-24".
- Citation correctness 88–91 % matches 87.6 / 89.0 / 90.6 in `reports/regression.md:1693-1695`. The 209-unit wording and "local, not committed" (line 729) are fixed. The provenance gap is stated (line 311). §10 is filled apart from the placeholder.
- Recomputed and all reconcile: every Table 5.1a overall value (mean of the three levels), all of Table 5.1b, the abstentions (12+24+15 = 51, 1+6+7 = 14, 1+6+6 = 13), Table 5.3, the Table 5.5 means and win/loss counts, the per-lecture mean difference +0.057, the §5.2 sums, the §3 link counts (311, 279/311) and the 87 commits.

2. **Numbers that don't reconcile**
- Line 578: "≤ 0.62 on any metric in Table 5.1a" is wrong. The largest std is METEOR, hard, iterative, 35.84 ± 0.67 (line 325; `lectqa_open_modes.md:24`). It should say ≤ 0.67.
- Line 144: "35 files, 354 tests". The 35 are 31 `.py` files plus 4 regression `.jsonl` files. Line 61 gives 354 passed + 1 deselected, which is 355 collected. Say "31 test modules, 355 tests (1 slow)".
- Line 53: "61 rows … over 2026-09-21 → 2026-09-23". The 61 rows run from 2026-09-10 (unpriced rows); only the priced rows start on 09-21.
- Lines 746 and 741: torch `linear` self time (77.7 %) exactly equals `document_pair_gate`'s cumulative share (77.7 %). This could be a coincidence or a copy error; check it against the profile.

3. **Rules violated**
- *Reporting convention*, "Every number carries its provenance … corpus/config hash": line 311 names the missing hash only for Tables 5.1a, 5.3 and 5.5. Table 5.4 (`reports/mavils_final.md`) and the §5.2 sources carry no hash either. Make the statement cover all of §5.
- *Pilot corpus* / *Dataset policy* ("never appears in a reported table"): line 230 says the tension is "flagged in §7 #16", but #16 (line 577) does not say DESIGN.md leaves it unsettled. The cross-reference points at nothing.

4. **Claims that outrun the evidence**
- Line 391 quotes finding 13 verbatim: "our retrieval beats a T1 replica by 5.5 F1". Line 40 says the +5.47 cannot be attributed to retrieval. The quote needs a bracketed note pointing to line 40.
- Table 5.5, lines 421–432: the "+ image" column is visual+theirs, i.e. distiluse text + image (`mavils_visual.md:28-39`), not fused text + image as the chained headers suggest. The same applies to "+ frame OCR" (visual+frame_ocr+theirs). Name the actual channel combination in each header.

5. **Record in DESIGN.md**
- Finding 13 *Consequence*: change "our retrieval beats a T1 replica by 5.5 F1" to a whole-pipeline difference, with a dated note of the old wording.
- Known issues: the §9.1 defects (`compare_retrieval.py:286` crashes before the ledger row is written; LectQA-Vid transcripts are never frozen; `transcribe_openai.py` ignores `--max-cost`; `build_groups` config key is dead). Also the ledger-vs-dashboard gap (~$3), and 456 unpriced gpt-5.4 calls.
- Remove or qualify the "8.8 ms/query" (*Environment decisions*), which has no protocol.
- Fix the §7 #18 drift items, including `relatedness_z` 2.0 vs 1.27 and the *Pipeline status* lines.
- Decide whether a project review may carry the pilot01 development table.

6. **Fix first**
1. Line 391: annotate the finding-13 quote as a whole-pipeline difference. At the same time, relabel the Table 5.5 "+ image" / "+ frame OCR" headers.
2. Line 578: change 0.62 to 0.67.
3. Line 311: extend the provenance gap to Table 5.4 and §5.2. Make §7 #16 actually carry the pilot01-table flag that line 230 points to. Fix the file/test count (line 144) and the ledger date range (line 53).

### 10.3 Applied after the second pass

- **Std bound:** 0.62 → 0.67.
- **Tests heading:** now "31 test modules + 4 regression files; 355 tests, 1 slow".
- **Ledger:** date range split into all rows vs priced rows.
- **Provenance gap:** extended to all of §5.
- **pilot01 flag:** §7 #16 now carries it.
- **Finding-13 quote:** annotated as a whole-pipeline difference.
- **Table 5.5:** headers name the actual channel combinations.
- **The two 77.7 % shares:** checked against the profile. They are both genuine (linear self time 77.68 %, gate cumulative 77.73 %), not a copy error.
- **Not done here:** the DESIGN.md records under "Record in DESIGN.md" (both passes). This review changes no file but itself, so they remain open follow-ups, and are listed in §7 #14 and #18.
