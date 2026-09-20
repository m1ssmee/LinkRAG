# LinkRAG — design notes

## What this is (10 lines)

1. A question-answering chatbot over a mixed corpus: documents (PDF/DOCX), images
   (diagrams, scanned notes) and audio (lecture recordings).
2. Architecture is Retrieval-Augmented Generation: index the corpus, retrieve
   evidence for a question, synthesise a cited answer.
3. Standard multimodal RAG indexes each modality **independently** and does plain
   top-k. Evidence arrives as a bag of unrelated chunks.
4. LinkRAG's novelty is an **Evidence Linking Layer**: at indexing time we build
   explicit, typed, scored links *across* modalities **and across files**.
5. Link types: audio segment ↔ slide, figure ↔ explaining paragraph, deictic
   speech ("as you see here") ↔ the visual element it points at, figure ↔ own slide.
6. Retrieval then **follows** those links out from the top-k seeds instead of
   stopping at them.
7. The final evidence set is reranked for **complementarity** — the set is scored
   for coverage, not each unit for similarity.
8. So an answer can be grounded in "what the lecturer said" *and* "the figure they
   were pointing at" *and* "the paragraph that defines the term" — jointly.
9. Everything is CPU-first so it runs on a laptop; GPU is an optional speedup.
10. Every stage runs in `baseline` or `linkrag` mode, so every claim is an ablation.

## Targets (repositioned 2026-09-20)

The project is measured against **two published systems on their own benchmarks**.
Everything else is a component or related work.

| | Target | Benchmark | Their number we must beat | Our metric |
|---|---|---|---|---|
| **T1** | Intra-Video Temporal-Aware RAG — Shafiq, Ejaz, Shah, Kamal, Sohail, Aslam. *CMC* 88(2), art. 96, 2026. doi:10.32604/cmc.2026.081534 | **LectQA-Vid**: 100 CS lecture videos (2–5 min), 3,000 QA pairs (1,500 MCQ + 1,500 open-ended), 80/10/10 split; evaluation subset 1,000 (500 MCQ + 500 open) | Open-ended, *Overall* row of their Table 4/6: **F1 23.52 %, semantic similarity 0.71** (ROUGE-1 29.76 %). MCQ, *Overall* row of Table 5: **accuracy 56.30 %**. Their multimodal-RAG-without-timestamps baseline (Table 6): F1 19.62 %, sim 0.61. | Same metrics, same split, same difficulty breakdown (Simple / Hard / Very Hard / Overall), plus evidence recall and modality coverage which they do not report. |
| **T2** | MaViLS — Anderer, Reich, Wölfel. *Interspeech 2024*, pp. 1375–1379. doi:10.21437/Interspeech.2024-978. arXiv:2409.16765 | **MaViLS**: 20 lectures (MIT OCW, Tübingen, DeepMind), >22 h video, 12,830 segments; every spoken sentence hand-labelled with a slide index (−1 = no slide). github.com/andererka/MaViLS | Per-frame F1 against ground truth, ignoring −1 labels. *Average* row of their Table 1: **audio-transcript-only 0.53**, OCR-text-only 0.76, image-only 0.64, SIFT 0.56; Table 2 / §4.2: **all three features combined, λ_jump = 0.1: 0.82**. | Same F1 definition, per lecture and average. **Our setting is their audio column** — we align a transcript to a slide PDF with no video frames — so 0.53 is the like-for-like number and 0.82 is the ceiling that uses information we do not have. |

What the targets already have, stated plainly so it is not re-claimed as ours:

- T2 already uses **dynamic programming over slides with a jump penalty** (their
  §3.2, λ_jump). A monotone/penalised DP for slide alignment is therefore *not* a
  contribution of this project. What is ours in `link/align.py`: the hybrid
  dense + BM25 + IDF-overlap similarity, the skip/back/start-prior penalty
  structure, the O(nm) prefix-max reduction, and — above all — that the alignment
  is one *link type* in a layer that retrieval and reranking consume, rather than
  an end in itself.
- T1 already does **timestamp-constrained retrieval** and a cross-encoder rerank,
  and reports an ablation. What it cannot do: link to a slide deck or paper that
  is *not the video*, follow a relation from one retrieved unit to another, or
  select for modality coverage. Its own Table 7 shows the temporal filter is worth
  12 F1 points — that is the size of the structure signal we generalise.

### Components and related work (not targets)

- **MI-RAG** — Choi, Lee, Ko, Rhee, arXiv:2509.00798 (v1 title "Multimodal
  Iterative RAG for Knowledge Visual Question Answering"; ICLR 2026 submission).
  **A component**: `retrieve/iterative.py` is our approximation of its
  re-querying loop and `linkrag_iter` composes it with link-following. It is not
  a comparison target.
- **MARA** — Wu et al., ACM MM 2025, pp. 4329–4338, doi:10.1145/3746027.3755390.
  Related work: page-independent document QA; motivates the cross-page
  `figure_text` link, not a target.
- **Literature-review set** (related work; cite, do not benchmark against):
  Evidence-grounded multimodal KG for multi-lecture reasoning (arXiv:2608.03161);
  MaViLS (also T2); LPM — Lecture Presentations Multimodal dataset (Lee et al.,
  ICCV 2023); Script-to-Slide alignment; AutoLectures (Holmberg, arXiv:2505.02966);
  MDKAG (Zhao, Wang, Lu, *Applied Sciences* 15(16):9095, 2025); Constrained
  Dominant Sets; MultiHaystack (arXiv:2603.05697). Bib entries in
  `paper/bibliography.bib`.
- **Concept-graph baseline** (an LLM-extracted entity/relation graph in the
  GraphRAG style, cf. arXiv:2608.03161, MDKAG): **deferred to Phase 8, optional.**
  Do not build it before priorities (i)–(vi) below are closed.

The labels `P1`/`P2`/`P3` survive in code docstrings, config comments and tests
from the original framing. Mapping: **P1 = MI-RAG (component), P2 = T1
Intra-Video Temporal-Aware RAG (target), P3 = MARA (related work).** They are not
renamed in code without a separate instruction.

## Priority order (binding from 2026-09-20)

Work is done in this order. Nothing outside this list is started without an
explicit instruction.

1. **(i) Automated gold verification.** Replace the ear-verified gold workflow
   with a checkable one: every gold locator must be reproducible from the corpus
   by a script, and a gold set that does not match its manifest stamp fails the
   run instead of warning.
2. **(ii) Relatedness gate.** A test that a linked pair is actually about the same
   thing (not merely co-located), applied to every link type before it enters the
   graph; report the pass rate per type.
3. **(iii) LectQA-Vid adapter and first run.** Load T1's videos (ASR + frame
   captions as their pipeline does), run baseline and linkrag, report their Table
   4/5 metrics on their split, with n stated.
4. **(iv) MaViLS adapter and first run.** Load the 20 lectures' transcripts + slide
   PDFs + ground truth, run `align_monotonic` and `align_naive`, report per-lecture
   F1 with their definition, alongside their audio column and combined row.
5. **(v) Phase 6 — citations + entailment.** Claim-level faithfulness: each
   answer sentence must be entailed by a cited unit; id-level citation checks are
   known to pass wrong answers.
6. **(vi) Extended cross-file dataset.** MaViLS / NPTEL lectures with decks + notes,
   the setting neither target covers, built only after (iii) and (iv) have numbers.

## Our three novel components

1. **Evidence Linking Layer** (`src/linkrag/link/`) — typed, scored links at
   index time, across modalities *and* across files. T1 links only by timestamp
   inside one video; T2 aligns but does not retrieve.
2. **Link-following retrieval** (`src/linkrag/retrieve/`) — top-k gives seeds;
   traverse links to pull in the evidence a seed *depends on*, at zero LLM calls.
3. **Complementarity-aware reranking** (`src/linkrag/retrieve/rerank.py`) — score
   the evidence *set*: reward an uncovered modality or a linked unit, penalise a
   restatement within a modality.

## Coding conventions

- **Type hints everywhere.** `from __future__ import annotations` at the top; use
  `X | None`, not `Optional[X]`.
- **Dataclasses for evidence units.** `EvidenceUnit`, `Link`, `Location` live in
  `src/linkrag/core.py` and are the only currency between stages. Do not invent a
  parallel dict schema; do not add a second "chunk" type.
- **pytest**, in `tests/`. Every non-trivial function gets one test that fails if
  the logic breaks. No fixtures-for-the-sake-of-fixtures.
- **No notebooks in `src/`.** `notebooks/` is for exploration only; anything that
  earns its keep gets moved into a module with a test.
- Config comes from `configs/*.yaml` via `core.load_config`. No magic numbers in
  code — thresholds and k values are config, because they get tuned.
- Deliberate simplifications are marked with a `Note:` comment naming the
  ceiling and the upgrade path.

## The two-mode rule (non-negotiable)

**Every module exposes a `baseline` mode and a `linkrag` mode.**

- Signature: entry points take `mode: Mode = "linkrag"` (`Mode` is in `core.py`).
- `baseline` must be a faithful reproduction of what the papers do — independent
  per-modality indexing, plain top-k, similarity-only ranking, no links. Do not
  sandbag it; a weak baseline invalidates the result.
- `linkrag` is ours.
- Both modes run from the same code path and the same config, so the only
  difference is the mechanism under test.
- `eval` reports both and the delta *is* the contribution. If a component can't be
  ablated this way, it isn't finished.

## Pipeline status (at tag `v0-pilot`)

| Stage | baseline | linkrag |
|---|---|---|
| `ingest` | **done** — PDF (PyMuPDF blocks + bboxes; caption-anchored figures for papers, vector/raster clustering for decks), DOCX, images (pytesseract OCR), audio (faster-whisper, word timestamps, sentence packing, frozen transcripts) | same extraction; provenance recorded |
| `index` | **done** — bge-m3 embeddings, exact numpy dense search + BM25, persisted to `data/processed/index/`, corpus manifest hash | same |
| `link` | n/a by definition | **done** — `audio_slide` (monotonic DP), `figure_text`, `deictic` (tiered cues), `same_slide`; networkx graph; manifest-stamped `links.jsonl` |
| `retrieve` | **done** — RRF of dense + BM25, plain top-k; `iterative` (MI-RAG approximation) | **done** — seed + 1-hop expansion, `normalise_seeds`, `linkrag_iter` |
| `rerank` | **done** — `none`, `mmr` | **done** — `complementarity`, optional cross-encoder |
| `generate` | **done** — cited answers, OpenAI-compatible endpoint, temperature 0 + seed sent | same, modality tags |
| `eval` | partial — regression runner, `compare_retrieval` (mode × rerank, repeats), alignment and deictic evaluators against ear labels | **claim-level entailment not started** (priority v) |
| `ui` | **not started** | **not started** |

Datasets: pilot01 only (`docs/pilot01_history.md`). **No target benchmark has
been run yet** — that is priorities (iii) and (iv).

`scripts/ask.py --mode linkrag` deliberately errors out rather than silently
falling back to baseline; a silent fallback would quietly fake the ablation.

## Measurement rules (binding)

**Every latency number must be:** measured on an **idle machine**, in a **single
process**, with a **warm model**, **repeated at least 3 times**, and reported as the
**median**. A number not taken this way is **not reportable** — do not put it in the
paper, in a report, or in this file.

This exists because an earlier draft claimed slide-guided ASR "costs 2.1× decode
time". It did not: that run was competing for cores with a bge-m3 embedding job, and a
clean re-run of the same audio finished in 231s against the plain run's 476s. Three
uncontrolled timings produced a confident, wrong, and paper-bound conclusion.

Correctness numbers (accuracy, WER, recall) are not covered by this rule, but their
provenance still has to be stated — see the ASR and alignment sections.

**Two results are presumed to be bugs until proven otherwise. Neither may be
reported until it has been inspected.**

1. **An ablation condition whose metrics exactly equal the full system, or are
   exactly zero.** Identical means the conditions are probably not separated;
   zero usually means the weaker condition is handicapped by construction rather
   than beaten on merit. Both have already happened here: `deictic` baseline
   returned exactly the same 18 links as linkrag because it still added
   `w_slide * on_slide` (it was *using* the alignment it was meant to be measured
   against), and once that was fixed it returned exactly 0 because its score
   ceiling sat below the shared threshold. Check the wiring, then check whether
   the losing condition could reach the threshold at all, before believing either.
2. **A table whose per-category counts do not sum to the stated total.** Reconcile
   before reporting. The link table once showed 51+22+5 against a stated 97 —
   `build_graph` was keying edges by `link_type` and silently dropping 19 of 41
   deictic links. The discrepancy was the only visible symptom.

## Execution rule (binding)

**Every new script or code path must be executed once for real — not mocked — before it
is considered complete.** Unit tests with stubs do not discharge this.

Three defects reached the repository because a path was only ever exercised through
mocks or never at all: `scripts/ingest.py` raised `NameError: Path` on its manifest
write, which had never run once (earlier manifests came from a separate script);
`run_regression.py --mode linkrag` silently used baseline retrieval while every test
passed; and the faiss/torch OpenMP crash was invisible to a test suite whose stub
encoder never loaded torch. In all three the test suite was green.

"For real" means against the actual corpus, with the actual models and files, and
reading the output — not just checking the exit code.

## Rejected approaches (do not re-propose)

- **`ingest ↔ interest` query-time alias.** Rejected 2026-09-06. Lecture-specific;
  would not generalise; ASR quality is not the contribution. This overrides the
  recommendation in `reports/phase2_step0_asr.md`, which predates the decision.
- **De-hyphenating the ASR vocabulary prompt.** Tried and disproved — a full re-run
  produced identical output, `type -k` included. See known issue (c).

## Pilot corpus (pilot01) — pinned

`data/processed/transcripts/hsieh.frozen.json` (guided-v2 transcript) is the
default for every pilot01 run; `ingest.frozen_transcript_dir` makes whisper skip.
**Do not re-transcribe pilot01 unless explicitly instructed.** Regression gold is
matched by locator, not unit id. Full pilot narrative, all measured numbers and
the bugs they surfaced: `docs/pilot01_history.md`.

## Standing instruments (report these every phase)

1. **Modality distribution of the retrieved set, per question.** `format_modality_
   distribution` in `linkrag.eval`; printed by `scripts/ask.py` and by the regression
   runner. A retrieval gain that only reshuffles within one modality is not the
   cross-modal gain this project claims — so the composition is reported, not just a score.
2. **The pilot01 regression set.** `tests/regression/pilot01_questions.jsonl` holds
   Q1–Q4 with gold evidence and types. Re-run every phase with
   `python scripts/run_regression.py --phase "<label>"`, which appends to
   `reports/regression.md`. Gold evidence is matched by **file+page / file+time
   overlap, never by unit id** — audio ids are regenerated whenever ASR or chunking
   settings change, so an id-keyed gold set would go stale silently.

## Known issues, not fixed

- **(a) Discourse boundaries.** Units are cut by sentence and duration only, with no
  notion of *discourse role*. `hsieh:a43` runs the talk's closing line straight into an
  audience member's Q&A self-introduction ("Saurabh Bakshi from Purdue"), which is how
  Q3 produced a false author with a *valid* citation. Segment-level speaker/discourse
  turn detection would fix it; not attempted. Any evidence unit may therefore span a
  speaker change.
- **(b) Modality imbalance — now known to be a corpus effect, not a retriever one.**
  On the 2-document corpus audio was 53 of 96 units (55%) and dominated every ranking
  (Q1/Q3 7-8/8 audio). Adding the OSDI paper moved audio to 51 of 196 (26%) and it now
  dominates nothing — **with no change to retrieval code, weights or thresholds**. See
  the standing observation in `reports/regression.md`. Consequence: a future
  modality-balance improvement cannot be credited to a retrieval change unless the
  corpus manifest is identical across the compared runs. Complementarity-aware
  reranking is still the intended fix for the underlying blindness.

- **(c) `type -k` ASR regression.** The slide-vocabulary prompt turns four spoken
  "top-K" into `type -k`. Hyphenation ruled out (identical output with and
  without). Cause unknown, unfixed, accepted as future work. A prompt omitting
  `Top K` entirely is the next thing to try.
- **(d) Gold set stale.** pilot01 gold is stamped for the 2-document corpus
  (`dded007ab45f5b9f`) against the live `2f3b35f27e86caf8`; every run warns.
  Resolved by priority (i), not by hand-editing.
- **(e) n = 4 questions.** Nothing retrieval-side on pilot01 is a result.

## Environment decisions worth not re-litigating

**No faiss.** `faiss-cpu` and `torch` each bundle their own OpenMP runtime. Loading
both into one process on macOS arm64 aborts with `OMP: Error #15`, and its only
documented workaround (`KMP_DUPLICATE_LIB_OK=TRUE`) is described upstream as
possibly producing *silently incorrect results* — unusable for a paper. faiss's
`IndexFlatIP` is exact brute force anyway, i.e. the same matmul numpy does, so it
bought nothing here. Measured 8.8 ms/query over 100k x 1024d vectors. Above roughly
200k units, add an approximate index (hnswlib, or faiss from conda-forge, which
links a single shared libomp) — and swap `test_dense_search_is_exact` for a
recall@k check rather than deleting it.

**OCR is pytesseract**, which needs a system binary: `brew install tesseract`
(macOS) or `apt-get install tesseract-ocr`. `linkrag.ingest.image` raises with that
hint if it is missing, and the OCR test skips rather than fails.

**Unit ids are citations.** They are built from the filename stem
(`notes:p2:t0`, `lecture03:a5`), so `a/notes.pdf` and `b/notes.pdf` mint colliding
ids. `ingest_files`/`dedupe_ids` suffixes collisions (`#2`) and `build_index`
refuses a duplicate outright — a collision would silently mis-attribute evidence.
