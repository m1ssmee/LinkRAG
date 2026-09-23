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

## Dataset policy (binding from 2026-09-23)

Mentor's rule: **results are reported only on the datasets of the papers being improved on.**

- **LectQA-Vid** (T1, *Intra-Video Temporal-Aware RAG*, CMC 2026) is the **primary**
  benchmark.
- **MaViLS** (T2, Interspeech 2024) is the benchmark for the **alignment** component.
- **No own-collected data** and **no third-party datasets outside those two papers**. The
  extended cross-file dataset plan (NPTEL / MIT lectures + notes) and the M3AV adapter are
  dropped. The intake tooling (`scripts/dataset/`) stays, for development only.
- **pilot01 is a development lecture only.** It is used to build and debug, and it never
  appears in a reported table.

## Priority order (binding from 2026-09-23)

Work is done in this order. Nothing outside this list is started without an
explicit instruction.

1. **(i) LectQA-Vid, full set.** Run on all videos that can be fetched (the first runs used
   28/100), with n and the skipped videos stated.
2. **(ii) LectQA-Vid protocol-matched metrics.** Their F1, semantic similarity and MCQ
   accuracy, with the per-difficulty tables (Simple / Hard / Very Hard / Overall).
3. **(iii) Frame-derived slide units on LectQA-Vid.** Slide text and figures from video
   frames, so that linking has a second stream on a single-video benchmark (finding 8).
4. **(iv) Verification / faithfulness on LectQA-Vid.** Claim-level entailment and
   citation checks on its answers.
5. **(v) MaViLS alignment rows.** As already done (finding 6); no new work planned.

### Previous priority order (2026-09-20 → 2026-09-23), for the record

(ii) relatedness gate, (iv) MaViLS and (v) Phase 6 were completed. (iii) LectQA-Vid was
started (28/100 videos) and continues as the new (i)–(ii). (vi) the extended dataset was
dropped by the dataset policy. Details of each item as it stood:



Work is done in this order. Nothing outside this list is started without an
explicit instruction.

- **(i) Automated gold verification.** ✅ **Done 2026-09-20** —
   `src/linkrag/eval/verify_gold.py`, `scripts/eval/verify_gold.py`,
   `scripts/eval/audit_sample.py`. Unit entailment (3 judge runs, majority,
   quotable span) + modality-only full-context answering decide gold and type
   labels; the only human step is the sampled audit sheet. pilot01 (judge
   gpt-4.1-mini, 2026-09-21): **25 of 25** proposed questions kept, **69** gold
   locators, **4 cross-modal**. The first run (2026-09-20, gpt-5.4 judging its own
   answers) kept 24 / 71 / 5 and is superseded; see known issue (g). See
   `reports/gold_verified_pilot01.md`, including the run history. Rules learned:
   reference answers list only the asked facts; a unit stating *one* required
   fact is gold.
- **(ii) Relatedness gate.** ✅ **Done 2026-09-21** — `src/linkrag/link/relatedness.py`,
   `scripts/gate_links.py`; verdicts stored in link metadata, `load_links(gated=True)`
   drops failures, `reports/relatedness_pilot01.md`. Pass rates: audio_slide 98 %,
   deictic 75 % (55.6 % before it exposed the page-number bug below), figure_text and
   same_slide 100 % (partly tautological — shared slide titles in OCR).
- **(iii) LectQA-Vid adapter and first run.** 🔶 adapter built
   (`scripts/adapters/lectqa_vid.py`: fetch / prepare / run). The published dataset
   ships QA + YouTube links only; videos are re-fetched (yt-dlp), transcribed
   (whisper) and frame-OCR'd here. Their split and 1,000-pair eval subset are
   unpublished — results are on the videos processed, n stated. First run:
   `reports/lectqa_vid_first_run.md`.
- **(iv) MaViLS adapter and first run.** ✅ **Run 2026-09-21** (`scripts/adapters/mavils.py`,
   their micro-F1 reproduced exactly, all 20 lectures, transcript + PDF only):
   **ours 0.45 vs their audio-only 0.53** (their all-features 0.82); above them on
   6/20 lectures. The DP adds +0.06 over naive argmax on average but *hurts* on
   the three page-OCR'd image decks — the similarity, not the DP, is the limit.
   `reports/mavils_alignment.md`.
- **(v) Phase 6 — citations + entailment.** Claim-level faithfulness: each
   answer sentence must be entailed by a cited unit; id-level citation checks are
   known to pass wrong answers.
- **(vi) Extended cross-file dataset.** *Dropped 2026-09-23 (dataset policy).*

## Findings (binding for dataset and design decisions)

1. **Link-following benefits are concentrated on questions where modalities are
   complementary.** On machine-verified pilot01 gold, the only clear win for
   link-following is C5 — the one deictic question whose referent is visual-only.
   Where a fact is restated in another modality, plain retrieval already reaches it
   and expansion can only add coverage, not recall (`reports/regression.md`,
   2026-09-20/21).
2. **pilot01 is highly redundant.** Only 4–5 of 16 proposed cross-modal questions
   survive verification under either judge; the paper restates the deck and the
   transcript narrates the slides. Quantified in `reports/redundancy_pilot01.md`
   (`scripts/dataset/redundancy.py`).
3. **Link-following is neutral on single-modality questions and gains on
   cross-modal ones** (2026-09-21, additive expansion, verified gold): single-source
   n = 21, `linkrag/complementarity` 50.5 % vs baseline 49.7 %; cross-modal n = 4,
   60.4 % vs 45.8 %. **n = 4 — direction only.**
4. **`linkrag_iter` underperforms `linkrag` on cross-modal questions** (45.8 % vs
   60.4 % on the same four). Hypothesis: iterative re-querying *re-textualises* the
   candidate pool — the follow-up query is text and pulls text units, so the
   seeds link-following expands from are less cross-modal than the single-shot
   seeds. **Test when the extended set exists**; not testable at n = 4.
5. **Unexplained: the composed mode's 71.3 % on single-modality questions**
   (`linkrag_iter/complementarity`, n = 21, vs 55.8 % for iterative alone and 50.5 %
   for linkrag alone). Neither mechanism explains a +15 pp gain on questions whose
   answer sits in one source. Flagged; do not cite it until it has a cause.
6. **The DP improves alignment only when the similarity matrix carries signal; on
   text-poor decks it underperforms naive argmax; abstention and flatness-scaling
   are the designed responses, evaluated on a held-out split.** MaViLS, 20
   lectures, their protocol (sentence granularity, page OCR, their sklearn F1 —
   `reports/mavils_alignment.md`). Every alignment table now carries the two paired
   numbers: their F1, and precision-on-answered / coverage.
   **Outcomes** (`reports/mavils_heldout_study.md`, `reports/mavils_final.md`;
   tune/test split `results/external/mavils_split.json`, chosen values
   `mavils_tuned.json`, all knobs OFF in `configs/default.yaml`):
   - like-for-like **0.46 vs their audio-only 0.53**; DP +0.16 over naive argmax on
     the same matrix, below naive only on the deck with no text layer.
   - **Decomposition (their public code for their cells):** their matrix × their DP
     0.513 (paper: 0.53); their matrix × our DP **0.520**; our matrix × their DP
     0.425; our matrix × our DP 0.461. Swapping the matrix moves the mean +0.074,
     swapping the decoder −0.022: **the gap is in the similarity features; our
     decoder ≥ theirs.** Distiluse cosine on page OCR beats our bge-m3 + BM25 + IDF
     hybrid at sentence granularity.
   - **Fused similarity** (`align.similarity`, default `ours`; `reports/mavils_fused.md`):
     weight chosen on the tune half (w = 0.5 on theirs, matrices min-max scaled).
     Test half, our DP at σ = 0.2: ours 0.461, theirs 0.515, fused_max 0.471,
     **fused_weighted 0.520** vs the paper's 0.51 on that half. Fusion closes the gap
     and edges their features by half a point; all 20 lectures 0.537 vs paper 0.53
     (optimistic — contains the tune half). The remaining lever is the similarity
     for short, OCR-noisy inputs; the DP is not the bottleneck.
   - σ sweep on the tune half: 0.2 chosen (tune 0.484 vs 0.471 at the pilot 0.02);
     test 0.461 vs 0.452 — a +0.009 that is inside lecture-to-lecture noise.
   - flatness scaling: inert at σ = 0.02 (identical paths; inspected).
   - abstention (min_sim 0.5055): +0–4 points precision-on-answered for 3–24 points
     of coverage; their F1 falls by construction (−1 is a label in their scorer).
   - build-deck grouping (`align.build_groups`): detects 7 groups / 22 pages on
     Decarbonization, but on the tune half it lowers F1 (0.478 vs 0.484) → off; test
     0.462 vs 0.461. The Decarbonization gap itself was an input effect (text layer
     vs page OCR: 0.21 → 0.44, 0.55 at σ = 0.2), not a DP effect.
7. **The file-pair relatedness gate works on unrelated pairs and fails on
   topical neighbours.** (`align.relatedness_z`, `results/external/mavils_gate_v2.md`.)
   Penalised DP objective vs a 30-shuffle slide-order null, 30-second windows:
   on 380 unrelated MaViLS audio × deck pairs the z distribution is median −0.09,
   95th percentile 1.26; at z = 1.27 (the default, set as the smallest z with
   ≤ 5 % false acceptance) it accepts 4.7 % of unrelated pairs and rejects 15 % of
   related ones — the low-F1 ones (ρ(z, F1) = 0.56), plus ML for health, which is
   aligned correctly (F1 0.56) and rejected for a cause not yet confirmed. The
   segments-per-slide hypothesis was tested and **rejected**
   (`results/external/mavils_gate_v3.md`): z does not rise with finer transcript
   granularity — it falls (pooled ρ(z, n/m) = −0.31), and the three rejects stay
   rejected at 30 s, 15 s and sentence level — so no adaptive re-windowing was added.
   30-second windows are the gate's best operating granularity.
   **Limitation:** the false acceptances are *topical neighbours* — Solar resource
   audio × Climate policies deck scores z = 3.5, Physics × Climate policies 3.3 —
   because a talk on a neighbouring subject does produce a weakly monotone match
   against a related-topic deck. The gate separates "about this deck" from "about
   nothing here"; it cannot separate two decks on the same subject. On an extended
   dataset drawn from one course (several lectures, several decks) this is exactly
   the confusion that will occur, so audio → deck pairing there must come from
   metadata (which lecture the deck belongs to), with the gate as a check, not as
   the pairing mechanism. Negative control (pilot01 audio × unrelated deck): 0
   cross-file links (`reports/relatedness_gate.md`).
8. **LectQA-Vid (target T1) is a single-stream benchmark: linking is inactive by
   construction, re-querying is what helps, and the reranker costs recall on
   speech-defined gold.** (`results/external/lectqa_v2.md`, 28/100 videos, 840 QA.)
   Their paper reports **no localisation metric** (§5.2 is answer quality only), so
   the localisation table is ours-on-their-data with their fixed-window configuration
   replicated as the baseline row. Findings: `linkrag` ≡ `baseline` to three decimals
   on hit@1/3/8 and IoU — one video, no deck or paper, so the only link type is
   temporal co-occurrence and expansion proposes units already retrieved; **iterative
   re-querying gains +7 pp hit@1** (44.8 → 52.0, `linkrag_iter` 52.7) while all modes
   converge by hit@8 (84–86 %); and **complementarity costs 13 pp of hit@3** at every
   difficulty (68.1 → 55.4) because it spends slots on frame OCR while the gold
   interval is defined by what was *said*. The modality-need gate (α = 0 on
   concentrated pools) was built and evaluated: it fires on 79 % of cells and changes
   nothing measurable on either benchmark — α decides which unit is taken first, not
   which eight are taken. Default off; gating β and γ is the next thing to test.
9. **Sentence-aware segmentation is free on localisation, not on answers.** On
   LectQA-Vid, sentence cutting removes 82.8 % of mid-sentence boundaries
   (82.8 % → 0.0 %) and moves hit@1 by 0.5 pp and IoU by 0.010 — inside noise, at
   every difficulty. The pilot01 result that segmentation matters was about answer
   quality and BM25 matching, not about finding the right 15 seconds. Keep
   `ingest.audio_segmentation: sentence` (it costs nothing), but do not claim it as a
   retrieval gain.
10. *Superseded 2026-09-23 by the dataset policy (no extended dataset).* **Extended-dataset selection criterion** (priority vi): **low redundancy**
   (measure it with `scripts/dataset/redundancy.py` before ingesting; a candidate
   with transcript→deck above pilot01's number is rejected), **diagram-heavy decks**
   (figures that carry facts the text does not), and **a speaker who points**
   (pointing windows exist; see `data/labels/pilot01/pilot01_pointing_windows.csv`
   for the labelling format).
11. **Local NLI cross-encoders are unsuitable as the gold / intake verifier.**
   (`results/nli_vs_llm_pilot01.md`, 2026-09-23; pilot01, corpus `2f3b35f27e86caf8`,
   `cross-encoder/nli-deberta-v3-base`, deterministic single run vs stored 3-run-majority
   LLM verdicts.)
   - **Caveat first:** the NLI formulation (initials-safe splitting, a minimum premise
     length, question conditioning: κ 0.22 → 0.27 → 0.31) is a **retune on the evaluation
     pairs**, with no held-out set. The bias favours NLI, so the negative conclusion stands;
     read 0.31 / 0.36 as an upper estimate. The large-model and formulation numbers come
     from exploratory scratch runs with no committed raw verdicts, so they are *indicative,
     not reportable*.
   - **Gold, unit level:** κ **0.31** vs gpt-4.1-mini and **0.36** vs gpt-5.4 over 147 unit
     pairs; the two LLM judges agree with each other at **0.85**.
   - **Question level:** type labels agree on 10/25; NLI drops **14/25** questions where
     the LLM judges drop 0–1.
   - **Redundancy:** overall κ 0.17. deck→transcript falls from **94.2 % to 59.6 %**
     (n = 52 sentences), below the 0.65 intake bar, so pilot01 would flip from REJECT to KEEP.
   - **A larger model does not help.** DeBERTa-v3-large (mnli-fever-anli-ling-wanli)
     reaches κ **0.35 / 0.35**. It is slower, but its timing fails the measurement rules
     (one run on a busy machine), so no speed ratio is claimed.
   - **Mechanism.** 13 of the 15 type differences are the NLI *grader* failing a correct
     full-corpus answer. The grader hypothesis is *question + reference fact*, and a
     correct answer does not restate the question: A5's "The baseline query took 4
     minutes, and the video was 6 hours long" is rejected. Deck bullets are fragments,
     not propositions, which is why the deck redundancy pairs sit at chance.
   - **Decision:** verification stays LLM-based (`eval.entailment.backend: llm`). The
     judge is gpt-4.1-mini, the cheap OpenAI tier and the model behind the stored verdicts
     (see *Cost policy*). A free judge (Groq, Colab) may replace it only after
     `scripts/eval/judge_agreement.py` has measured it against the stored verdicts. NLI
     remains selectable as an ablation (`--entailment nli`) and never decides gold or intake.

### Design changes recorded against the verified-gold result (2026-09-21)

Recorded as design changes, not retunes: each has a config switch that reproduces
the previous behaviour, and each was applied once, before re-measuring.

| change | switch | why |
|---|---|---|
| **Additive expansion.** Retrieve `k_final` seeds, add 1-hop neighbours to the pool, let the reranker (or score, for `rerank=none`) select `k_final`. Expansion never evicts a seed on its own. | `retrieve.expansion: additive` (old: `evict`) | `evict` with `k_seed 5 < k_final 8` threw away seeds ranked 6–8 unconditionally; A1 went 100 % → 0 %. |
| **Seed normalisation on** | `retrieve.linkrag.normalise_seeds: true` (old: `false`) | Under additive expansion "select by score" is degenerate on raw RRF scores (~0.03 vs `seed×link×decay`). Consequence, inspected per measurement rule 1: `linkrag/none` is identical to `baseline/none` by construction — neighbours enter the final set only through the reranker. |
| **Separate judge** | `eval.judge` (model ≠ `models.llm`) | Self-grading is lenient. Judge/answerer agreement measured: κ = 0.85 on unit verdicts, 20/25 type labels (`reports/gold_verified_pilot01.md`). The intended default is a local model; this machine (8 GB, no Ollama) uses a different hosted family instead. The judge is gpt-4.1-mini (cheap tier; see *Cost policy*), which is also where the stored verdicts above came from. The batch answerer is a different cheap model. |
| **Per-type reporting** | `compare_retrieval.py` always emits it | The all-questions mean hides that 21 of 25 questions are single-source. From now on the by-type table is the one that matters. |
| **Cost accounting** | `linkrag.costs`, `reports/llm_ledger.jsonl` | Every LLM-touching run prints per-run and cumulative spend; exact tokens from `usage`, cache replays free, backfilled rows flagged *estimated*. |
| **Colab backend for Phase 8** | backend `colab` (`LINKRAG_LLM_BACKEND` / `LINKRAG_JUDGE_BACKEND`), URL from `LINKRAG_COLAB_BASE_URL` | Reported Phase 8 numbers come from an open model served from Colab (Ollama/vLLM) on the same code path. `notebooks/colab_serve.ipynb` serves it; as judge only after `scripts/eval/judge_agreement.py --judge colab` has measured it. Setup in `scripts/README.md`. |
| **Dataset intake** | `scripts/dataset/candidate.py`, `dataset.intake` | A candidate lecture is measured before it is ingested for real; deck→transcript ≥ 0.65 rejects. Procedure in `scripts/dataset/README.md`. |
| **Relatedness gate (links)** | `link.relatedness.enabled`, `load_links(gated=...)` | Structural links are judged for shared content before they enter the graph; the per-type pass rate is the signal-strength number. Found the deictic file/page bug. |
| **Relatedness gate (file pairs)** | `align.relatedness_z` (2.0) | Before any cross-file link is emitted, the penalised DP objective must beat 5 shuffled-slide-order alignments by z std devs; cross-document semantic figure_text uses a word-shuffle null. Negative control (pilot01 audio × unrelated deck): 0 cross-file links; false-rejection on 20 related MaViLS pairs: 15 % at 30 s windows, 40 % at sentence level (`reports/relatedness_gate.md`). Unrelated pairs fall back to plain hybrid retrieval. |
| **Sampled redundancy (2026-09-23)** | `redundancy.py --sample N`, `dataset.intake.redundancy_sample` (0 = census), `sample_seed` | Redundancy is an estimate for a gate. N sentences per direction, stratified by reading position, fixed recorded seed; each fraction gets a Wilson 95 % CI. **Rule:** KEEP if the CI's upper bound of deck→transcript < 0.65, REJECT if its lower bound > 0.65, otherwise BORDERLINE (exit 3) and the census decides. On pilot01's stored LLM verdicts all four full-run fractions lie inside the sampled CIs at N = 150 and at N = 50; a real N = 150 run (cache replay, $0) reproduced the simulated counts exactly. N = 150 saves only 8 % of judge calls on pilot01 (15 % on pilot01-w1) because a deck has ~52 sentences; N = 50 saves 54–58 % and still decides both (deck→transcript CI lower bound 83.8 % / 76.2 %). Evidence is two lectures, replayed — direction only. |

## Cost policy (binding from 2026-09-23)

Hosted credit is limited, about $10 for the rest of the project. Spend goes to the cheap
tier, and nothing bills without an explicit budget.

- **`--max-cost` is required** on every batch script (`verify_gold`, `compare_retrieval`,
  `run_regression`, `redundancy`, `candidate`, `propose_questions`, `lectqa_vid`,
  `gate_links`, `judge_agreement`). `0` allows only cache replays and free backends. The
  spend guard prices each request before sending it and refuses anything that could
  exceed the budget.
- **Judge:** gpt-4.1-mini (`eval.judge`), the model behind the stored verdicts.
- **Batch answerer:** a different cheap model (gpt-5.4-mini). It must not be the judge
  (*Separate judge*).
- **Strong model** (gpt-5.4, the answerer of every stored pilot01 run): refused in batch
  runs unless `models.llm.allow_strong_in_batch: true`. It is for a final reference row
  and the demo only. A cache-only replay of its stored answers is allowed, because it
  cannot spend.
- **Free backends** (`billing: free`): Colab-served open models (`notebooks/colab_serve.ipynb`,
  backend `colab`), Groq and Google AI Studio. Always allowed. As judge, only after
  `scripts/eval/judge_agreement.py` has measured them.
- **Free by construction:** reply-cache replays, local faster-whisper ASR, and the NLI
  ablation.
- **Comparability.** A number is comparable only to numbers produced by the same judge
  and answerer. Moving the batch answerer from gpt-5.4 to gpt-5.4-mini makes new runs a
  new condition against the stored gpt-5.4 rows; say which answerer produced each row.

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
| `eval` | partial — regression runner, `compare_retrieval` (mode × rerank, repeats, per-type and per-question tables), alignment and deictic evaluators against ear labels, automated gold verification + sampled audit, separate judge (`eval.judge`), **modality redundancy metric** (`scripts/dataset/redundancy.py`) | **claim-level entailment not started** (priority v) |
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

## Reporting convention (binding, from 2026-09-23)

- **Every number carries its provenance:** dataset, split, n, corpus/config hash, the
  models involved (answerer, judge), and how many runs. LLM-dependent cells run 3
  times and report mean ± std. A single run, or a small n (e.g. the n = 4 cross-modal
  cell), is labelled *direction only* / *not reportable*, never stated as a result.
- **Thresholds and knobs are set only on a tune half or on external data,** then
  applied once to the test half. Choosing a value after seeing evaluation numbers is a
  **retune** and is reported as one, never as a result. A **design change** is a
  mechanism with a config switch that reproduces the old behaviour, applied once
  before re-measuring (see *Design changes*); a retune is not a design change.
- **A report says what changed, then what was found, with caveats before headline
  numbers.** Per-category counts sum to the stated total (*Measurement rules* §2).
- **Corrections go in their own commit,** naming the number, what was wrong and the fix;
  the corrected report keeps a dated note of the old value.

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

**Development lecture only (dataset policy, 2026-09-23): it never appears in a reported table.**

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
   the 25 machine-verified questions (all 25 proposed in `pilot01_proposed.jsonl`,
   judge gpt-4.1-mini) with 69 gold locators,
   verified unit ids and types, stamped `2f3b35f27e86caf8`. Regenerate with
   `scripts/eval/verify_gold.py` whenever the corpus or the proposals change; never
   hand-edit. Re-run every phase with
   `python scripts/run_regression.py --max-cost <USD> --phase "<label>"`, which appends to
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

- **(c) `type -k` ASR regression — cause identified 2026-09-23: the local model.**
  faster-whisper `small` turns four spoken "top-K" into `type -k` with the slide
  vocabulary prompt. With the **same prompt**, OpenAI `whisper-1` produces 0
  occurrences and 16 correct `top-K` (`reports/asr_openai_pilot01.md`), so it was a
  property of the local model, not of the prompt. pilot01 stays on the frozen local
  transcript -- switching it would invalidate every recorded alignment, deictic and
  regression number. For the extended dataset, zero-cost mode (2026-09-23) makes local
  faster-whisper on a Colab GPU the default; whisper-1 remains `--asr openai`, and the
  `type -k` risk comes back with the local model, so check new transcripts for it.
- **(d) Gold is machine-verified, not human-verified.** Judge (gpt-4.1-mini) and
  answerer (gpt-5.4) are different models and agree at κ = 0.85 on unit verdicts;
  the sampled audit (`reports/audit_sheet_pilot01.csv`) is unfilled until someone
  scores it, so human agreement is unknown.
- **(e) n = 25 questions, 4 cross-modal.** Enough to run the harness, not enough to
  claim a cross-modal result; see Findings and `reports/redundancy_pilot01.md`.
- **(f) Free judges are not validated.** Groq and Colab-served models have made no real
  judge call yet, and their agreement with the stored judges is unmeasured
  (`scripts/eval/judge_agreement.py` measures it). Until then they may not decide gold or
  intake; the NLI result shows that a judge swap alone can flip a verdict. The Groq free
  tier caps requests per day. `gpt-oss-120b` is a reasoning model, and `max_tokens: 1024`
  may truncate its JSON reply (unverified). The batch answerer gpt-5.4-mini is not yet
  verified against this account's model list.
- **(g) Gold counts — reconciled 2026-09-23.** The stored gold (`reports/gold_verified_pilot01.md`,
  `tests/regression/pilot01_questions.jsonl`, judge gpt-4.1-mini, verified 2026-09-21) is
  the reference: 25 questions, 69 gold locators, 4 cross-modal (3 split + 1 deictic).
  DESIGN.md had said 24 / 71 / 5. Those figures come from the first verification (commit
  03213ed, 2026-09-20), judged by gpt-5.4, the answerer itself. That run kept 76 gold
  *units*, which collapse to 71 *locators* in the regression file because several units
  share a page. The separate-judge change (commit 187fc27) regenerated the gold, but
  *Priority order* (i) and *Standing instruments* were not updated. That is the drift.
  Both are now corrected. Unit counts (report) and locator counts (regression file) are
  different quantities; name which one a number is.

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
