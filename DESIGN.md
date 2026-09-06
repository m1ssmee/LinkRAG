# LinkRAG — design notes

## What this is (10 lines)

1. A question-answering chatbot over a mixed corpus: documents (PDF/DOCX), images
   (diagrams, scanned notes) and audio (lecture recordings).
2. Architecture is Retrieval-Augmented Generation: index the corpus, retrieve
   evidence for a question, synthesise a cited answer.
3. Standard multimodal RAG indexes each modality **independently** and does plain
   top-k. Evidence arrives as a bag of unrelated chunks.
4. LinkRAG's novelty is an **Evidence Linking Layer**: at indexing time we build
   explicit, typed, scored links *across* modalities.
5. Link types: audio segment ↔ slide, figure ↔ explaining paragraph, deictic
   speech ("as you see here") ↔ the visual element it points at.
6. Retrieval then **follows** those links out from the top-k seeds instead of
   stopping at them.
7. The final evidence set is reranked for **complementarity** — the set is scored
   for coverage, not each unit for similarity.
8. So an answer can be grounded in "what the lecturer said" *and* "the figure they
   were pointing at" *and* "the paragraph that defines the term" — jointly.
9. Everything is CPU-first so it runs on a laptop; GPU is an optional speedup.
10. Every stage runs in `baseline` or `linkrag` mode, so every claim is an ablation.

## Baselines and their limitations

| | Paper | What it does | Why it isn't enough |
|---|---|---|---|
| **P1** | MI-RAG (ICLR 2026) | Iterative re-querying: reformulate the question and retrieve again, several rounds | Recall goes up, precision goes down. More rounds pile up loosely related chunks; nothing models *how* two chunks relate, so the LLM gets a bigger haystack, not a better one. |
| **P2** | Intra-Video Temporal-Aware RAG (CMC 2026) | Fuses audio and visual streams **within a single video** by timestamp | Timestamp co-occurrence is the only link, and it stops at the video boundary — a lecture recording is never connected to the separate slide deck or textbook PDF. No hallucination check on the generated answer. |
| **P3** | MARA (ACM MM 2025) | Document-only multimodal retrieval | Pages are treated as independent units — a figure and the paragraph explaining it three pages later never get connected. No audio at all. |

Common gap: **no explicit cross-modal, cross-file relation is ever materialised.**
Each paper either re-queries harder (P1), links only within one video by clock
time (P2), or links nothing (P3).

## Our three novel components

1. **Evidence Linking Layer** (`src/linkrag/link/`) — build typed, scored links at
   index time, across modalities *and* across files. This is what P2 does only
   inside one video and P3 does not do at all.
2. **Link-following retrieval** (`src/linkrag/retrieve/`) — top-k gives seeds;
   traverse links to pull in the evidence a seed *depends on*. Fixes P1's
   precision loss: expansion is along known relations, not another fuzzy query.
3. **Complementarity-aware reranking** (`src/linkrag/retrieve/`) — score the
   evidence *set*: reward a unit that adds an uncovered aspect or an unrepresented
   modality, penalise one that restates what is already in the set.

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

## Pipeline status

| Stage | baseline | linkrag |
|---|---|---|
| `ingest` | **done** — PDF (PyMuPDF blocks + bboxes, embedded figures, naive captions), DOCX, images (pytesseract OCR), audio (faster-whisper, word timestamps) | same extraction; provenance already recorded |
| `index` | **done** — bge-m3 embeddings, exact numpy dense search + BM25, persisted to `data/processed/index/` | link table: not built |
| `link` | n/a by definition | **not started** |
| `retrieve` | **done** — RRF of dense + BM25, plain top-k | **not started** |
| `generate` | **done** — cited answers via Ollama / any OpenAI-compatible endpoint | prompt suffix in place, needs the link set |
| `eval` | **not started** | **not started** |
| `ui` | **not started** | **not started** |

`scripts/ask.py --mode linkrag` deliberately errors out rather than silently
falling back to baseline; a silent fallback would quietly fake the ablation.

## Phase-1 pilot (`data/raw/pilot01`, run 2026-09-06)

Corpus: Hsieh et al., *Focus*, OSDI '18 — 22.9 min talk audio + 27-slide deck.
Full run in `reports/phase1_pilot.md`.

| | |
|---|---|
| Units indexed | **98** — 27 text, 18 figure, 53 audio |
| Whisper | `small`, int8, CPU — **476s (7.9 min) for 22.9 min audio ≈ 2.9× realtime** |
| Transcript | 3,243 words, 201 whisper segments, 172 sentences (mean 6.9s) |
| Embedding | bge-m3, dim 1024, 36s for 98 units |
| Retrieval | 0.15-0.5s/query after warmup; embedder load is ~20s, timed separately |
| LLM | `gpt-5.4-2026-03-05` @ temp 0.0, ~2.5-3.3s/answer |

### Segmentation: the spec'd fix was wrong, measure before believing

| approach | audio units | boundaries mid-sentence |
|---|---:|---:|
| fixed-window (what P1/P2/P3 do) | 46 | 32/46 (**70%**) |
| packing whole whisper segments | 50 | 37/50 (**74%**) |
| splitting the word stream on terminal punctuation | 53 | 0/53 (**0%**) |

Whisper's `segments` are decoder chunks, not sentences — only ~26% end on
punctuation, so packing them is *worse than doing nothing*. Punctuation is
attached to individual words, so cut there instead. `ingest.audio_segmentation`
selects `sentence` (default) or `fixed` (the papers' behaviour, kept for ablation).

### Baseline results — the failure modes Phase 2 must fix

| Q | type | modalities returned | verdict |
|---|---|---|---|
| Q1 | slides_only | 6 audio / 2 text | partial — honest refusal, retrieval failure |
| Q2 | audio_only | 8 audio | **complete** |
| Q3 | cross_modal_split | 7 audio / 1 text | **WRONG — citation valid, answer false** |
| Q4 | cross_modal_deictic | 8 audio / 0 slide | partial — honest refusal, retrieval failure |

1. **Modality collapse.** Audio is 54% of units and dominated every ranking. The
   slide holding the answer was never retrieved for Q1, Q3 or Q4. Both rankers are
   modality-blind; RRF does not rebalance them, so the bigger modality wins on volume.
2. **A valid citation is not a correct answer.** Q3 asserted "Saurabh Bakshi from
   Purdue" as a presenter, citing a real unit — that string is an *audience member
   introducing themselves in the Q&A*, sitting in the same unit as the closing line.
   Our id-level citation check passes. P2 has no hallucination check at all and would
   also pass. **`eval` therefore needs a claim-level check, not just an id-level one.**
3. **Deixis is unresolvable without links.** Q4's segment says "in this figure" and
   "here we have"; nothing in the baseline turns either into a pointer to slide p20.
   This is the Evidence Linking Layer's reason to exist, measured.
4. **ASR noise hits the load-bearing term.** *ingest* → **"interest"** throughout, so
   BM25 cannot match the talk's central concept. Consider an alias/glossary at query
   time, and do not attribute all sparse-side failure to BM25 itself.

Caveat: the two questions the baseline handled acceptably are the two whose gold
evidence is single-modality. Every question needing two modalities degraded. n=4 on
one lecture — indicative, not a result.

## Step 0 — slide-guided ASR (`ingest.asr_vocab_from_slides`)

`build_asr_prompt` mines ~75 terms from the deck into whisper's `initial_prompt`;
`ingest_files` runs documents before audio so the vocabulary exists in time. Both
transcripts are kept under `data/processed/transcripts/`. Full write-up:
`reports/phase2_step0_asr.md`.

| | plain | guided |
|---|---:|---:|
| `ingest` (correct) | 7 | **16** |
| `interest` (mis-heard *ingest*) | 20 | **10** |
| `NoScope` | 4 | 4 — was never broken |
| `YOLO` | 3 | 3 — was never broken |
| `top-K` (any spelling) | 8 | **4** — 4 became `type -k` |

**Three things worth remembering.**

1. The premise was one-third right. Plain whisper already got `NoScope` and `YOLO v2`
   correct; only `ingest` was broken, and it is now *half*-fixed.
2. The prompt **introduces** a new error: four spoken "top-K" become `type -k`. I
   hypothesised the hyphen in `Top-K` caused it, de-hyphenated, and re-ran the full
   transcription — **identical output**. Hypothesis rejected, workaround reverted,
   cause still unknown. Don't re-try de-hyphenation without new evidence.
3. **Do not trust the decode timings** (plain 476s, v1 990s, v2 231s). v1 ran while a
   bge-m3 embedding job was competing for cores. An earlier draft of this file claimed
   "2.1× slower" from that number; it was an artifact. Re-measure on an idle machine
   before quoting any speed claim.

Net: **+9 `ingest`, −4 `top-K`** — a trade, not a free win. The cheaper fix for the
original problem is a **query-time alias** (`ingest ↔ interest`) in BM25 tokenisation:
no decode cost, no new ASR errors, and it reaches the ~10 residual cases the prompt
misses.

WER on a 50-word window (613.6–636.3s) is **2.0% for both** — the guided run did not
help there (the window's only difference is a `the` I cannot adjudicate by ear). The two transcripts differ in only 2.6% of word positions, so a 50-word
sample is statistically thin; **the term table is the result, not the WER**. The WER
reference is text-adjudicated (slides settle *ingest*), not audio-verified — nobody
on this project has checked the recording by ear.

Residual: ~10 `interest`-for-*ingest* errors remain, so BM25 recall on the corpus's
central term is still degraded. A query-time alias would recover them for free.

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

## Rejected approaches (do not re-propose)

- **`ingest ↔ interest` query-time alias.** Rejected 2026-09-06. Lecture-specific;
  would not generalise; ASR quality is not the contribution. This overrides the
  recommendation in `reports/phase2_step0_asr.md`, which predates the decision.
- **De-hyphenating the ASR vocabulary prompt.** Tried and disproved — a full re-run
  produced identical output, `type -k` included. See below.

## Frozen corpus — pilot01

`data/processed/transcripts/hsieh.frozen.json` (the guided-v2 transcript) is the
**default for all subsequent phases**. `ingest.frozen_transcript_dir` makes
`ingest_file` use `<stem>.frozen.json` verbatim whenever it exists; whisper does not
run. **Do not re-transcribe pilot01 unless explicitly instructed.**

Kept alongside, for the ASR ablation only:
`hsieh.plain.json` (no prompt), `hsieh.guided_v1_hyphenated.json`,
`hsieh.guided_v2_ablation.json` (identical content to the frozen file).

Frozen index: 96 units — 27 text, 18 figure, **51 audio**. Audio unit ids changed
again in this rebuild, which is why regression gold is matched by locator, not id.

## Open: `type -k` regression

The vocabulary prompt turns four spoken "top-K" into `type -k`. Hyphenation is ruled
out (identical output with and without). Cause unknown, unfixed, accepted as future
work. A prompt omitting `Top K` entirely is the next thing to try.

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

## Phase 2 — audio-to-slide alignment (`src/linkrag/link/align.py`)

First component of the Evidence Linking Layer. Full recurrence in the module
docstring, in LaTeX, ready to paste into the paper.

`S_ij = w_d·cos(e_i,e_j) + w_b·BM25norm + w_k·IDF-overlap`, then a Viterbi-style DP
over slide assignments with a forward-jump penalty λ, a per-unassigned-slide skip
penalty σ, and a back-jump penalty β limited to B slides (`B=0` ⇒ strictly
monotonic). **O(nm), not O(nm²)**: the forward-jump penalty is affine in `j−j'`, so
the inner maximum collapses to a running prefix maximum. `tests/test_align.py`
checks the shipped DP against a brute-force O(nm²) reference on random matrices —
keep that test, an off-by-one in the prefix max would return subtly wrong paths in
silence.

Ablation is `link.align.method: naive` — per-segment argmax, no sequence structure,
roughly what P2 does.

### Measured on pilot01 (51 audio segments × 27 slides)

Evaluated against an **ear-labelled slide timeline, approximate (±3 s), n=42 talk
segments** (`data/labels/pilot01/`), plus 9 Q&A segments with no true slide.

| | monotonic DP | naive argmax |
|---|---|---|
| exact (talk, n=42) | **29/42 = 69.0%** | 22/42 = 52.4% |
| ±1 slide | **90.5%** | 71.4% |
| exact (non-ambiguous, n=27) | **63.0%** | 51.9% |
| backward steps | **0** | 9 |
| slides covered (23 showable) | **21/23** | 15/23 |
| Q&A (n=9): distinct slides assigned | **1 — holds slide 26** | 5 — wanders |

The naive baseline's errors are *catastrophic*; the DP's are adjacent slides, and
**11 of 13 are off-by-one ahead of truth** (p5→p6, p8→p9, p13→p14, p17→p18) — the path
runs slightly early, consistent with a speaker discussing the next slide before
advancing. A lag term is the obvious next parameter; not attempted.

`start_prior_mu` was added and tuned 0.0 → **0.02**, fixing the head-of-sequence error
(first segment p3 → p1). Two caveats, recorded in full in `reports/regression.md`: it
was fitted on the same labels it is scored on (no held-out split, so 69.0% is
optimistic), and the entire gain sits on *ambiguous* segments — the non-ambiguous
subset is 63.0% at every value of mu.

**Deictic precision proxy** against the same labels: 42/51 = 82.4% of deictic pairs
link a figure sitting on the segment's true slide — **tier 2 scores 16/16 = 100%**,
tier 3 26/35 = 74.3%. Direct evidence that the cue tiering separates reliable deixis
from noise. It checks the slide, not the referent; referent labels do not exist.

Artifacts: `data/processed/links.jsonl` (51 audio_slide Links), `links.npz`
(similarity + path, so plotting and eval skip re-embedding),
`reports/alignment_pilot01.png` (heatmap with DP, naive and ear-labelled truth).

## Phase 3 — figure_text, deictic, and the link graph

Modules: `link/figure_text.py`, `link/deictic.py`, `link/graph.py`,
`ingest/vlm_caption.py`. Scoring formulae in LaTeX in each module docstring.
Link types are `figure_text` and `deictic` (renamed from the earlier
`figure_paragraph` / `deictic_visual`), and `Link` now carries a `metadata` dict.

### What each part does

- **figure_text**: numbered references (`Figure 3`, `Table 2` — kind-aware, so
  "Table 2" cannot match "Figure 2") restricted to the same or an adjacent page;
  unnumbered descriptive references (`the diagram below`) resolved by bbox direction;
  **bbox vertical-gap layout proximity** on the same page; page-distance prior; and
  semantic similarity over caption + OCR + optional VLM text. Every link records
  *why* it was made in `metadata`.
- **deictic**: ⚠️ the emitted set is **unfiltered**. `mass` includes `w_slide`, and
  every scored candidate is on the aligned slide, so all of them start at
  `w_slide/mass = 0.45` — exactly `deictic_threshold`. Measured floor over 98 pilot
  links: 0.5622, none below 0.50. "Links above threshold" is a vacuous phrase here;
  the set is "every cue on a slide that has a figure", capped by `max_links_per_unit`.
  Deliberately not retuned — retuning changes every recorded deictic number.
  Cue phrases are generated as determiner × noun × direction
  (`this arrow here`, `the box on the right`, `that step below`) plus a base list —
  266 phrases from two short config lists. Candidates are restricted to the aligned
  slide, and ties between figures on one slide are broken by **IDF keyword overlap**
  against the figure's OCR/caption. The phrase, its timestamps, and the slide are
  stored in `metadata`.
- **graph**: `networkx.MultiDiGraph` over all units and links, with
  `neighbors(unit_id, link_types, min_score, direction)`, `subgraph_around`, and
  GraphML export. `scripts/plot_graph.py` renders a neighbourhood to `reports/`.
- **vlm_caption**: optional Ollama VLM descriptions, **off by default** (a 7B VLM on
  CPU is tens of seconds per image), cached by image content hash, and degrading to
  `""` rather than aborting a corpus build when Ollama is absent.

### Measured on pilot01

| link type | count | avg score | baseline |
|---|---:|---:|---:|
| audio_slide | 51 | 0.6749 | 51 |
| deictic (raw cue hits) | 41 | 0.5956 | 14 |
| **deictic (distinct segment–figure pairs)** | **22** | — | **11** |
| figure_text | 5 | 0.5786 | 5 |
| same_slide | 18 | 1.0000 | 0 |

Total 115, and the per-type counts sum to it (measurement rule 2).

Both suspicious readings were inspected per measurement rule 1: `figure_text` is
identical in both conditions because all 5 links are same-page, so the same-page-only
baseline keeps every one — this corpus has no cross-page figure references at all.
`same_slide` is 0 in baseline *by construction*: baseline is the ablation that removes
that link type, not a condition competing and losing.

### Deictic cue tiers

Cues are tiered, and the tier weight multiplies `weights.cue` — so tier changes
ranking without moving any threshold:

| tier | meaning | weight | pilot01 pairs | avg score |
|---|---|---:|---:|---:|
| 1 | explicit object deixis ("this arrow here") | 1.0 | **0** | — |
| 2 | bare pronoun + visual word in the same sentence | 0.6 | 4 | 0.6183 |
| 3 | bare pronoun alone | 0.3 | 18 | 0.5947 |

**Tier 1 is empty.** This speaker never produces explicit object deixis, so 18 of 22
pairs rest on a bare "this"/"that" with no visual word anywhere in the sentence. The
tiering exists to make that visible in the output rather than averaged away — the
earlier report of "41 deictic links" was two illusions at once: raw cue hits
double-counting 22 real pairs, and no indication that almost all of them were the
weakest possible trigger.

Reporting therefore leads with **distinct (segment, figure) pairs at max score**, with
the raw cue count as a secondary statistic. Every pair is listed in
`reports/deictic_pairs_pilot01.csv` (segment start, phrase, tier, figure page, score,
sentence) for checking against ear labels.

### same_slide

Each figure on a deck page links to that page's text at a fixed score of 1.0. Deck-ness
is detected at ingest from page geometry (pilot01: 27/27 landscape) and overridable per
file via `ingest.slide_deck_files`. This relation is *certain*, not scored — the figure
was rendered on that page — so it does not compete inside `figure_text`, where it would
be judged on caption and reference evidence that a slide structurally cannot produce.
Ablatable via `link.same_slide.enabled` or `--link-mode baseline`.

### Two bugs found by the counts disagreeing — keep the guards

1. **19 of 41 deictic links were silently dropped.** `build_graph` keyed edges by
   `link_type`, so two cues in one segment pointing at the same figure collapsed into
   one edge. Caught only because the per-type table disagreed with `links.jsonl`.
   Edges are auto-keyed now; `neighbors` dedupes to the best score for traversal.
2. **The deictic ablation was invalid** (Phase 3 first pass): `baseline` still added
   `w_slide * on_slide`, so the "no alignment" condition was using the alignment.
   Guarded by `test_baseline_does_not_use_the_alignment`.

### Honest read on deictic quality

**All 41 deictic links come from bare one-word pronouns** — `that` (23), `this` (17),
`here` (1). Not one came from the multi-word object deixis the spec asked for; this
speaker simply does not say "the arrow above". The unit tests prove the mechanism
fires on `this arrow here`, but on *this* corpus it is driven by the weakest cues.
Compounding that, `w_slide` (0.45) dominates and is constant across all figures on the
aligned slide, so the cue type barely affects ranking — in practice *any* cue on a
slide that has a figure produces a link. 41 links cover only 22 distinct
(audio, figure) pairs, one pair carrying 6 cues.

Two tuning levers if this matters: drop bare `this`/`that` from `link.deictic_cues`,
or raise `weights.cue`. **Not changed unilaterally** — it trades recall for precision
and there are no labelled referents to decide it on.

### figure_text still under-exercised by pilot01

5 links, still **0 cross-page**, because the deck has no captions and never writes
"Figure N" (see the corpus analysis below). The fixture PDF *does* contain a
cross-page `As Figure 1 shows` reference and `test_figure_text_links_the_cross_page_
reference` proves the mechanism reaches across pages and that baseline does not.
Evaluating it on real data still needs a paper-style PDF.

## Phase 3b — the OSDI paper added (3-document corpus)

`data/raw/pilot01/osdi18-hsieh.pdf`, 19 pages, 612x792 **portrait → deck=False**.

| source | text | figure | audio |
|---|---:|---:|---:|
| `osdi18_slides_hsieh.pdf` (deck) | 27 | 18 | — |
| `osdi18-hsieh.pdf` (paper) | 84 | 16 | — |
| `hsieh.mp3` (frozen transcript) | — | — | 51 |
| **total 196** | 111 | 34 | 51 |

### Figure extraction had to be fixed first

The paper has **13 captioned figures**; the extractor produced **2**, both logo
fragments. `page.get_images()` sees only *raster* images and a LaTeX paper draws its
plots with vector operators — 239 draw-ops on p4, invisible to the extractor. Of 30
raster images, 28 were sub-fragments below `min_figure_area_px`.

Fixed by anchoring figures to their captions (`ingest.figures_from_captions`, on for
non-decks): find blocks matching `^(Figure|Table) N[:.]`, take the region above,
render it. The first attempt bounded that region at the nearest block above and
collapsed 9 of 14 regions to 6–16pt — because a vector plot's own axis labels are text
blocks sitting just above its caption. Bounding at the nearest *prose* block
(`BODY_TEXT_WORDS`) fixed it: 14 regions, 72–238pt. **This is an extraction change,
not threshold tuning.**

### figure_text links in the paper, by signal

| signal | links | avg score |
|---|---:|---:|
| explicit (numbered `Figure N` / `Table N`) | 31 | 0.8355 |
| explicit (descriptive, "the diagram below") | 0 | — |
| proximity (bbox layout) | 21 | 0.5394 |
| semantic only | 0 | — |
| **total (paper)** | **52** | |

Counts sum to the total (measurement rule 2). Corpus-wide `figure_text` went 5 → 66;
10 of those are cross-document (9 deck-figure → paper-text, 1 paper-figure →
deck-text). The paper is the corpus this mechanism was designed for: explicit numbered
references dominate and score 0.84, which the deck could never produce.

### Cross-document figure↔figure: nothing real

**A first pass looked like a strong result and was an artifact.** Top pairs sat at
0.92 — all of them *empty-content* figures, because `embeddable_text` falls back to a
provenance descriptor for captionless figures, so they all embed near-identically and
match each other.

Restricted to content-bearing figures (15/16 paper, 10/18 deck): max cosine **0.546**,
9 pairs ≥ 0.5, **none ≥ 0.6**, and the top pairs match a real paper caption against
deck OCR crumbs ("trucks Object clusters"). **There are no meaningful cross-document
figure↔figure semantic links in this corpus** — the paper has captions, the deck has
fragments, and there is nothing to match on. No links were emitted.

### same_slide correctly skipped the paper

18 links, all from deck figures; **0 from paper figures**, via portrait geometry.

### Regression moved — and the gold set is now incomplete

Adding the paper **broke the modality collapse**: Q1 went from 8/8 audio to 6/8 text,
Q4 from 8/8 audio to 5/8 text. Q3 now scores **4/4 gold terms** while its gold
*locators* are both missed — the paper supplies the authors and affiliations that
previously only the title slide and opening audio carried. The Q1–Q4 gold locators
were written for a 2-document corpus and now under-credit a legitimate third source.
They need extending before the next phase's numbers mean anything.

## Phase 4 — link-following retrieval (`retrieve/linkrag.py`)

Top-k becomes **seeds**; the Evidence Linking Layer is then traversed outward, so a
unit enters the set because something already retrieved depends on it. Expanded score
is `seed_score * link_score * decay`, and every unit records whether it was a seed or
expanded and via which seed and link (`RetrievedUnit.explain()`).

`retrieve/iterative.py` is a third, fairer baseline: our approximation of P1 (MI-RAG) —
retrieve, ask the LLM for a follow-up query, retrieve again, merge. It exists so
link-following is not credited merely for returning more units than single-shot top-k.

### Measured on the 4-question pilot set (k=8, 196 units)

| method | recall@8 | precision@8 | LLM calls |
|---|---:|---:|---:|
| baseline (top-k) | 37.5% | 9.4% | 0 |
| **iterative (P1)** | **62.5%** | **18.8%** | **4** |
| linkrag | 50.0% | 12.5% | **0** |

Per question: Q1 0/0/0, Q2 100/100/100, Q3 0/**50**/0, Q4 50/**100**/**100**.

**P1 currently beats link-following on recall.** It wins Q3, which link-following
cannot reach because no link runs from anything retrieved to the title slide. It pays
one LLM call per question on the critical path; link-following pays none. State it that
way — recall alone favours P1 on this set.

What link-following does buy, concretely: **Q4 went 50% → 100%**, and
`osdi18_slides_hsieh:p20:t0` entered an evidence set **for the first time in any run**.
The chain is `hsieh:a28` → `audio_slide` (0.681) → slide p20, whose text carries
`Optimize for Ingest Cost / Balance / Optimize for Query Latency`. That slide has been
the missing gold unit in every regression since Phase 1, and no amount of re-querying
had found it. The generated answer then names the three configurations and cites p20.

Caveats, in order of size: **n=4 questions**; the gold set is still stamped for the
2-document corpus and known to under-credit Q1/Q3; and expansion currently cannot
displace a weak seed, because RRF seed scores (~0.03) dwarf `seed x link x decay`
(~0.01) — with `k_seed < k_final` expansion only fills the remaining slots. Phase 5's
complementarity reranker is what changes that.

**Latency was measured but is not reportable** under the measurement rules above:
single run, non-idle machine, no repeats. The LLM-call counts (0 vs 4) are exact and
are the cost comparison that stands.

### Bug found while testing: BM25 tie-break

`sparse_search` used `np.argsort(scores)[::-1]`, which breaks ties in **reverse corpus
order** — so for a query matching nothing, every unit ties at BM25 0.0 and the
*last-ingested* unit ranked first. It surfaced because a deliberately unreachable test
unit kept appearing in baseline top-5. Now a stable descending sort (ties break by
ascending position); `test_sparse_ties_break_by_position_not_reverse_position` guards it.

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
