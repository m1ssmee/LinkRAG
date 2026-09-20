# pilot01 history (v0-pilot)

Narrative record of the pilot corpus work, moved out of `DESIGN.md` on 2026-09-20
when the project's comparison targets were repositioned (see `DESIGN.md`,
"Targets"). Nothing here was edited in the move; section order is preserved.
Numbers refer to the corpus and gold set as they stood at tag `v0-pilot`
(commit d89cd5f). Binding rules that grew out of these episodes -- measurement
rules, execution rule, rejected approaches -- stay in `DESIGN.md`.

Corpus: Hsieh et al., *Focus*, OSDI '18 -- 22.9-min talk audio, 27-slide deck,
19-page paper. Manifest `2f3b35f27e86caf8`, 209 units, 354 links.

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

Evaluated against **ear-labelled v2, approximate (±3 s), n=42 talk segments, pointing windows n=10** (`data/labels/pilot01/`), plus 9 Q&A segments with no true
slide. **Labels v2 replaced v1 entirely**; v1 is deleted and must not be used.

| | monotonic DP | naive argmax |
|---|---|---|
| exact (talk, n=42) | **32/42 = 76.2%** | 24/42 = 57.1% |
| ±1 slide | **92.9%** | 73.8% |
| exact (non-ambiguous, n=33) | **72.7%** | 57.6% |
| backward steps | **0** | 9 |
| pages covered (26 showable = 24 distinct slides) | **21/26** | 15/26 |
| Q&A (n=9): distinct slides assigned | **1 — holds slide 26** | 5 — wanders |

Build slides (10/11 and 17/18) are one slide with two page numbers; either is scored
correct. v1's three "zero-length" slides were an artifact of inferred ends — v2 has
none, and its corrected boundaries cut ambiguous segments from 15 to 9.

The naive baseline's errors are *catastrophic*; the DP's are adjacent slides, and
**11 of 13 are off-by-one ahead of truth** (p5→p6, p8→p9, p13→p14, p17→p18) — the path
runs slightly early, consistent with a speaker discussing the next slide before
advancing. A lag term is the obvious next parameter; not attempted.

`start_prior_mu` is **0.02**, re-checked against v2 and unchanged (same plateau).
Unlike v1, the gain now also shows on the non-ambiguous subset (69.7% → 72.7%), which
retires the v1 caveat that it only moved unreliable labels. It is still fitted on the
same labels it is scored on — no held-out split, so 76.2% is optimistic.

**Deictic evaluation** now has referent-ish ground truth: `pilot01_pointing_windows.csv`
marks 10 windows where the speaker is referring to something on screen (outside a
window the label is UNKNOWN, not negative). Full report:
`reports/deictic_eval_pilot01.md`.

Only **3 of 10 windows are scorable** — seven point at a slide with no extracted figure
unit, one is a table, one is the external demo. Within those: precision **6/6 = 100%**,
detection recall **3/5 = 60%**, and **0 false positives** on the slide-24 known
negative. All six scorable pairs are tier 3, so this evaluation says nothing about
whether tiering helps. **The binding constraint is figure extraction, not deixis.**

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

**Composing the two beats either alone.** `linkrag_iter` (iterative seeds + link
expansion) reaches 75.0% recall / 28.1% precision with `normalise_seeds`, against 56.2%
for P1 and 50.0% for link-following. Q4 needs an *edge*, Q3 needs a *seed*; each
mechanism supplies one. See `reports/regression.md`.

On the plain 4-question set without composition, **P1 beats link-following on recall.** It wins Q3, which link-following
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

## Phase 5 — complementarity-aware reranking (`retrieve/rerank.py`)

Scores the **set**, not the unit. Objective in LaTeX in the module docstring:

```
F(R) = Σ rel(u)  +  α·|distinct modalities|  +  β·|link edges inside R|
                 −  γ·Σ cos(u,v) over same-modality pairs
```

Greedy MMR-style selection (exact maximisation is NP-hard). **Redundancy is charged
only within a modality** — an audio segment and the slide it describes are *supposed*
to be similar, and penalising that would defeat the objective.

Two ablations share the same code path so they cannot diverge by accident:
`none` (plain top-k) and `mmr` (standard MMR, text-diversity only — modality-blind and
link-blind, which isolates what α and β add).

Optional cross-encoder (`retrieve.rerank.cross_encoder`, e.g. `BAAI/bge-reranker-base`)
supplies `rel(u)` from a modality-tagged string — `[AUDIO 18:40-19:10] …`,
`[FIGURE p.12] …`. A missing model degrades to retrieval scores with a warning rather
than aborting the query.

`--rerank none|mmr|complementarity` on `scripts/ask.py`. Candidates are retrieved to
`rerank.pool` (20) then selected down to `k_final`, so reranking filters rather than
merely reorders.

### Measured: modality coverage and redundancy (4 questions, k=8)

| method | recall@8 | prec@8 | distinct modalities | redundancy |
|---|---:|---:|---:|---:|
| baseline | 33.3% | 9.4% | 1.75 | 0.711 |
| iterative (P1) | 39.6% | 12.5% | 2.25 | 0.689 |
| linkrag | 50.0% | 15.6% | 2.75 | 0.710 |
| **linkrag_iter** | **75.0%** | **28.1%** | **3.00** | **0.678** |

`linkrag_iter` reaches **3.00 — every question's set spans all three modalities**.
These figures are from `compare_retrieval.py`, which reports the diagnostics but does
**not** itself apply the reranker; the reranker runs in `ask.py`. Wiring it into the
comparison is the obvious next step and would let α/β/γ be ablated on recall.

⚠️ `iterative` scored 56.2% in an earlier run and 39.6% here with no config change —
the follow-up query is LLM-generated and not deterministic. Only `baseline` and
`linkrag` repeat exactly.

