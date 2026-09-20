# Regression — pilot01 Q1–Q4

The same four questions, re-run every phase. Gold evidence is matched by
file+page or file+time-overlap, not by unit id. `modality` is the retrieved
set's composition — a gain that only reshuffles within one modality is not
the cross-modal gain LinkRAG claims.

## Ear-labelled alignment evaluation (2026-09-06)

Ground truth is a hand-made slide timeline read from the recording. **v2
(`pilot01_slide_timeline_v2.csv`) replaces v1 entirely — v1 is deleted.** Converted to
per-segment labels for all 51 frozen audio segments: **ear-labelled v2, approximate (±3 s), n=42 talk segments, pointing windows n=10** (9 ambiguous), plus 9 Q&A
segments with no true slide. 26 of 27 pages are showable (27 never shown), forming 24
distinct slides — 10/11 and 17/18 are build slides where either page scores correct.

*The numbers below were regenerated against v2; the v1 figures they replace are given
in the v1→v2 table further down.*

The labelled slide sequence is **monotonic non-decreasing**, independently confirming
the assumption `align_monotonic` is built on.

### Talk section — all 42 labelled segments

| method | exact | ±1 | backward steps | slides covered |
|---|---:|---:|---:|---:|
| **monotonic DP (ours)** | **32/42 = 76.2%** | **92.9%** | **0** | **21/26** |
| naive argmax (P2-style) | 24/42 = 57.1% | 73.8% | 9 | 15/26 |

### Talk section — non-ambiguous subset (n=33)

| method | exact | ±1 | backward steps | slides covered |
|---|---:|---:|---:|---:|
| **monotonic DP (ours)** | **24/33 = 72.7%** | **90.9%** | **0** | 17/26 |
| naive argmax (P2-style) | 19/33 = 57.6% | 78.8% | 6 | 14/26 |

### Q&A section (n=9) — no slide changes after 19:05

| method | assigned slides |
|---|---|
| **monotonic DP (ours)** | `26` × 9 — **one distinct slide, held throughout** |
| naive argmax (P2-style) | `26, 26, 16, 26, 13, 12, 12, 10, 12` — **5 distinct slides** |

The DP holds the last slide through 3¾ minutes of questions, which is what actually
happened. Naive wanders back to slides 10–16 because nothing constrains it.

### start_prior_mu tuned (0.0 → 0.02)

Exact accuracy changed, so per the tuning rule it was tuned; the smallest value
reaching the plateau was taken.

| mu | exact (all talk) | ±1 | first segment | slides covered |
|---:|---:|---:|---|---:|
| 0.0 (previous default) | 27/42 = 64.3% | 88.1% | p3 ✗ | 19/23 |
| **0.02 (new default)** | **29/42 = 69.0%** | **90.5%** | **p1 ✓** | **21/23** |
| 0.05 – 0.5 | 29/42 = 69.0% | 90.5% | p1 ✓ | 21/23 |

⚠️ **Two caveats on this tuning.** It was fitted on the same 42 labels it is evaluated
on — there is no held-out split, so 69.0% is optimistic. And the entire gain sits on
*ambiguous* segments: the non-ambiguous subset is unchanged at 17/27 = 63.0% for every
value of mu. The honest reading is that mu fixes the head-of-sequence error (first
segment p3 → p1, which is certain) and otherwise moves only labels that are least
reliable.

### Residual error is a systematic lead, not noise

11 of 13 DP misses are off-by-one **ahead** of truth (p5→p6, p6→p7, p8→p9, p10→p11,
p13→p14, p17→p18, p24→p25). The path runs slightly early rather than drifting randomly
— consistent with a lecturer talking about the next slide before advancing to it. A
lag term is the obvious next parameter; not attempted.

### Deictic precision proxy

Cross-checking `reports/deictic_pairs_pilot01.csv` against the ear labels: **is the
linked figure on the segment's true slide?**

| subset | agreement |
|---|---:|
| all talk-section pairs | **42/51 = 82.4%** |
| **tier 2 (pronoun + visual word)** | **16/16 = 100%** |
| tier 3 (bare pronoun) | 26/35 = 74.3% |
| non-ambiguous pairs only | 25/32 = 78.1% |

**Tier 2 is perfect and tier 3 is not**, which is direct evidence that the cue tiering
introduced earlier separates reliable deixis from noise. Note the set being scored is
**unfiltered**: `deictic_threshold` cannot reject an on-slide candidate (see DESIGN.md),
so this is precision over every cue on a slide that has a figure, not over a
threshold-selected subset. Every disagreement is
off-by-one, inherited from the alignment's lead rather than from cue selection. This is
a proxy: it checks the *slide*, not the *referent*. Referent-level labels do not exist.

## Retrieval comparison — baseline vs P1 vs link-following (2026-09-06)

`scripts/compare_retrieval.py`, 4 pilot questions, k=8, 196-unit corpus.

| method | recall@8 | precision@8 | LLM calls |
|---|---:|---:|---:|
| baseline (top-k) | 37.5% | 9.4% | 0 |
| **iterative (P1)** | **62.5%** | **18.8%** | 4 |
| linkrag | 50.0% | 12.5% | **0** |

| qid | baseline | iterative (P1) | linkrag |
|---|---:|---:|---:|
| Q1 | 0% | 0% | 0% |
| Q2 | 100% | 100% | 100% |
| Q3 | 0% | **50%** | 0% |
| Q4 | 50% | **100%** | **100%** |

**P1 beats link-following on recall on this set (62.5% vs 50.0%)** and should be
reported that way. It wins Q3; link-following cannot, because no link connects anything
retrieved to the title slide. P1 spends 1 LLM call per question, link-following 0.

The specific thing link-following buys: **`osdi18_slides_hsieh:p20:t0` entered an
evidence set for the first time in any run**, via `hsieh:a28` → `audio_slide` (0.681).
That slide has been the missing gold unit for Q4 since Phase 1 and no re-querying had
ever found it — it shares almost no vocabulary with the question.

Caveats: n=4; the gold set is still stamped `dded007ab45f5b9f` (2-document corpus) and
is known to under-credit Q1/Q3; latency was measured but is **not reportable** under
the measurement rules (single run, non-idle machine, no repeats).

## Q3 — design intent no longer holds

**`Q3 cross_modal_split` is retained unchanged and must not be rewritten.** Its design
intent — force composition of author names in one modality with affiliations in another,
across two files — **no longer holds after slide figure extraction: it is answerable
from one page.** Slide 1 now yields both `osdi18_slides_hsieh:p1:t0` (names, text) and
`osdi18_slides_hsieh:p1:g1` (affiliations, OCR'd from the logo band, figure). The OSDI
paper's title block had already collapsed it once; the deck now collapses it again.

Consequence for reading this file: **a Q3 pass is no longer evidence of cross-modal
composition.** Treat its gold-term column as a single-source retrieval measure until the
question is scoped or replaced. Every historical Q3 row stays valid for what it measured
at the time.

## Standing observation — modality dominance is a corpus-composition effect

Adding the OSDI paper (2 documents → 3, 96 → 196 units) inverted retrieved-set
dominance **without any change to retrieval code, weights or thresholds**:

| Q | 2-document corpus | 3-document corpus |
|---|---|---|
| Q1 | 8/8 audio | text 6/8, audio 2/8 |
| Q2 | 8/8 audio | text 3/8, audio 5/8 |
| Q3 | text 1/8, audio 7/8 | **8/8 text** |
| Q4 | 8/8 audio | text 5/8, audio 3/8 |

Audio was 53 of 96 units (55%) and dominated every ranking; it is now 51 of 196 (26%)
and dominates none. The retriever did not change between these two runs — only the
corpus did.

**This is evidence that the "audio dominance" recorded as known issue (b) is a property
of corpus composition, not of the retriever.** Two consequences for how results here
are read:

1. A future modality-balance improvement **cannot be credited to a retrieval change**
   unless the corpus manifest is identical across the compared runs. That is what the
   manifest stamp in each run header is for.
2. Conversely, the pilot's original cross-modal failures were partly an artifact of a
   corpus where one modality held the majority of units. The baseline was being judged
   on a corpus that flattered audio. Neither reading is safe without the manifest.

## 2026-09-06 16:02 — phase1 close (baseline, plain ASR)

mode=`baseline` · index=`data/processed/index` · embedder=`BAAI/bge-m3` · units=98 · top_k=8  · retrieval only (no LLM)

| Q | type | modality distribution | gold evidence | gold missed | gold terms in answer |
|---|---|---|---|---|---|
| Q1 | `slides_only` | text 1/8, audio 7/8 | **no** | osdi18_slides_hsieh.pdf p.5 | 0/2 |
| Q2 | `audio_only` | 8/8 audio | **yes** | — | 0/3 |
| Q3 | `cross_modal_split` | text 1/8, audio 7/8 | **partial** | osdi18_slides_hsieh.pdf p.1 | 0/4 |
| Q4 | `cross_modal_deictic` | 8/8 audio | **partial** | osdi18_slides_hsieh.pdf p.20 | 0/3 |

## 2026-09-06 16:13 — phase2 audio-slide alignment (links built; retrieval not yet link-aware)

mode=`baseline` · index=`data/processed/index` · embedder=`BAAI/bge-m3` · units=98 · top_k=8

| Q | type | modality distribution | gold evidence | gold missed | gold terms in answer |
|---|---|---|---|---|---|
| Q1 | `slides_only` | text 1/8, audio 7/8 | **no** | osdi18_slides_hsieh.pdf p.5 | 0/2 |
| Q2 | `audio_only` | 8/8 audio | **yes** | — | 3/3 |
| Q3 | `cross_modal_split` | text 1/8, audio 7/8 | **partial** | osdi18_slides_hsieh.pdf p.1 | 2/4 |
| Q4 | `cross_modal_deictic` | 8/8 audio | **partial** | osdi18_slides_hsieh.pdf p.20 | 1/3 |

<details><summary>Q1 answer (2 citations)</summary>

The talk cites **NoScope** as the state-of-the-art **query-time-only** approach [hsieh:a32] [hsieh:a8].

The **venue and year** are **not stated in the provided evidence**.

</details>

<details><summary>Q2 answer (1 citations)</summary>

In the Q&A, the speaker says they currently make cheap CNNs by starting with a simple CNN, specifically ResNet-18, and then removing layers one by one to create a set of cheap CNNs. [hsieh:a52]

He also says this process is not automatic right now: they “don’t have the automatic way to generate the cheap CNN based on the architecture of the expensive ones.” [hsieh:a52]

</details>

<details><summary>Q3 answer (2 citations)</summary>

The evidence only identifies one named presenter, **Saurabh Bakshi from Purdue** [hsieh:a43]. It also says the work is **joint work with collaborators in Carnegie Mellon University, Microsoft, University of Wisconsin, and ETH Zurich** [hsieh:a0].

The provided evidence does **not** list the full authors of Focus or map each author to an institution. [hsieh:a0] [hsieh:a43]

</details>

<details><summary>Q4 answer (2 citations)</summary>

The evidence says the plot contains “different set of configurations,” and the speaker says they can choose “the option that balances these two metrics,” or alternatively a low-latency option or an ingest-cost-optimized option depending on the application [hsieh:a30] [hsieh:a29].

However, the evidence provided does **not** list the actual configuration labels shown on the plot, so I can’t name them from the supplied material [hsieh:a29] [hsieh:a30].

As for which one they would select, the speaker says they could select:
- “the option that balances these two metrics,” or
- for low-latency, frequent-query applications, an option that pays more ingest cost for lower query latency, or
- for home cameras, an option optimized for ingest cost with higher latency [hsieh:a30].

</details>

## 2026-09-06 20:29 — phase3 figure+deictic links (frozen guided-v2 transcript; retrieval still not link-aware)

mode=`baseline` · index=`data/processed/index` · embedder=`BAAI/bge-m3` · units=96 · top_k=8

| Q | type | modality distribution | gold evidence | gold missed | gold terms in answer |
|---|---|---|---|---|---|
| Q1 | `slides_only` | 8/8 audio | **no** | osdi18_slides_hsieh.pdf p.5 | 0/2 |
| Q2 | `audio_only` | 8/8 audio | **yes** | — | 3/3 |
| Q3 | `cross_modal_split` | text 1/8, audio 7/8 | **partial** | osdi18_slides_hsieh.pdf p.1 | 2/4 |
| Q4 | `cross_modal_deictic` | 8/8 audio | **partial** | osdi18_slides_hsieh.pdf p.20 | 2/3 |

<details><summary>Q1 answer (2 citations)</summary>

The talk cites **NoScope** as the state-of-the-art **query-time** approach [hsieh:a32] [hsieh:a8].

The **venue and year** are **not provided in the evidence**.

</details>

<details><summary>Q2 answer (2 citations)</summary>

The speaker says they currently use a simple manual approach: start with a simple CNN, specifically ResNet18, and remove layers one by one to generate a set of cheap CNNs for exploration [hsieh:a49]. He also says they do **not** have an automatic way to generate cheap CNNs from expensive architectures; it is “not fundamental,” and there is no automatic method right now [hsieh:a50].

</details>

<details><summary>Q3 answer (1 citations)</summary>

The evidence only identifies one speaker and the collaborating institutions, not a full author list. It says the work is “joint work with my collaborators in Carnegie Mellon University, Microsoft, University of Wisconsin, and ETH Zurich.” [hsieh:a0]

So, based on the provided evidence:
- Presenter/author named in the evidence: Irene Hsieh (implied by the source labeling, but not explicitly stated in the transcript, so I cannot confirm from the evidence alone).
- Institutions involved: Carnegie Mellon University, Microsoft, University of Wisconsin, and ETH Zurich. [hsieh:a0]

The provided evidence does not list all authors by name.

</details>

<details><summary>Q4 answer (2 citations)</summary>

The evidence says the plot contains “different set of configurations” and that one can choose among options like:
- “the option that balances these two metrics” [hsieh:a30]
- an option for applications needing “low latency” that “pay[s] a little bit more cost … to achieve lower query latency” [hsieh:a30]
- an option for applications such as “home cameras” that “optimize for ingest cost with higher latency” [hsieh:a30]

The evidence does **not** provide the exact labels shown on the plot. It only describes these option types verbally [hsieh:a30].

As for which one the speaker says they would select: the speaker says **“we can select different options”** depending on the application, rather than naming one single universally chosen option [hsieh:a30].

</details>

## 2026-09-06 20:49 — phase3 complete (figure_text + deictic + graph)

mode=`baseline` · index=`data/processed/index` · embedder=`BAAI/bge-m3` · units=96 · top_k=8  · retrieval only (no LLM)

| Q | type | modality distribution | gold evidence | gold missed | gold terms in answer |
|---|---|---|---|---|---|
| Q1 | `slides_only` | 8/8 audio | **no** | osdi18_slides_hsieh.pdf p.5 | — |
| Q2 | `audio_only` | 8/8 audio | **yes** | — | — |
| Q3 | `cross_modal_split` | text 1/8, audio 7/8 | **partial** | osdi18_slides_hsieh.pdf p.1 | — |
| Q4 | `cross_modal_deictic` | 8/8 audio | **partial** | osdi18_slides_hsieh.pdf p.20 | — |

## 2026-09-06 21:16 — phase3 rev2: deictic tiering + same_slide (retrieval still not link-aware)

mode=`baseline` · index=`data/processed/index` · embedder=`BAAI/bge-m3` · units=96 · top_k=8

| Q | type | modality distribution | gold evidence | gold missed | gold terms in answer |
|---|---|---|---|---|---|
| Q1 | `slides_only` | 8/8 audio | **no** | osdi18_slides_hsieh.pdf p.5 | 0/2 |
| Q2 | `audio_only` | 8/8 audio | **yes** | — | 3/3 |
| Q3 | `cross_modal_split` | text 1/8, audio 7/8 | **partial** | osdi18_slides_hsieh.pdf p.1 | 2/4 |
| Q4 | `cross_modal_deictic` | 8/8 audio | **partial** | osdi18_slides_hsieh.pdf p.20 | 1/3 |

<details><summary>Q1 answer (2 citations)</summary>

The talk cites **NoScope** as the **state-of-the-art query-time** approach [hsieh:a32] [hsieh:a8].

The **venue and year** are **not provided in the evidence**.

</details>

<details><summary>Q2 answer (2 citations)</summary>

In the Q&A, the speaker says they currently produce cheap CNNs with a simple manual approach: they start with a simple CNN, specifically ResNet18, and remove layers one by one to create a set of cheaper CNNs for exploration [hsieh:a49].  

He also says this is **not automatic**: they “don’t have the automatic way to generate the cheap CNNs based on the architecture of the expensive ones” [hsieh:a50].

</details>

<details><summary>Q3 answer (1 citations)</summary>

The evidence only states that *Focus* is “a joint work with my collaborators in Carnegie Mellon University, Microsoft, University of Wisconsin, and ETH Zurich,” but it does not name the authors. [hsieh:a0]

So, based on the provided evidence:
- Institutions: Carnegie Mellon University, Microsoft, University of Wisconsin, and ETH Zurich. [hsieh:a0]
- Authors: not specified in the provided evidence. [hsieh:a0]

</details>

<details><summary>Q4 answer (2 citations)</summary>

The evidence says the plot contains “different set of configurations” and that they can choose among options based on goals like balancing cost and latency, favoring lower latency, or favoring lower ingest cost [hsieh:a30]. It also says the tunable configuration parameters include “which cheap CNN we use,” “what’s the top-k results that we are selecting,” and “what’s the threshold for clustering” [hsieh:a28].

However, the provided evidence does **not** list the exact labels shown on the plot, and it does **not** say a single specific option the speaker personally selects from that plot. It only says they could select:
- an option that balances the two metrics,
- an option that pays a bit more ingest cost for lower query latency,
- or an option that optimizes ingest cost with higher latency [hsieh:a30].

</details>

## 2026-09-06 22:36 — phase3b: OSDI paper added (196 units, 3 documents)

mode=`baseline` · index=`data/processed/index` · embedder=`BAAI/bge-m3` · units=196 · top_k=8

| Q | type | modality distribution | gold evidence | gold missed | gold terms in answer |
|---|---|---|---|---|---|
| Q1 | `slides_only` | text 6/8, audio 2/8 | **no** | osdi18_slides_hsieh.pdf p.5 | 0/2 |
| Q2 | `audio_only` | text 3/8, audio 5/8 | **yes** | — | 3/3 |
| Q3 | `cross_modal_split` | 8/8 text | **no** | osdi18_slides_hsieh.pdf p.1, hsieh.mp3 0.6-30.0s | 4/4 |
| Q4 | `cross_modal_deictic` | text 5/8, audio 3/8 | **partial** | osdi18_slides_hsieh.pdf p.20 | 2/3 |

<details><summary>Q1 answer (2 citations)</summary>

The talk/paper cites **NoScope** as the state-of-the-art **query-time** approach [hsieh:a41] [osdi18-hsieh:p11:t3]. The provided evidence does **not** state the **venue and year** where NoScope was published [osdi18-hsieh:p11:t3].

</details>

<details><summary>Q2 answer (2 citations)</summary>

In the Q&A, the speaker says they produce cheap CNNs by starting with a simple CNN, specifically ResNet18, and then removing layers one by one to create a set of cheaper CNNs for exploration [hsieh:a49]. The speaker also says this process is not automatic: “we don't have the automatic way to generate the cheap CNNs based on the architecture of the expensive ones” [hsieh:a50].

</details>

<details><summary>Q3 answer (1 citations)</summary>

The authors of *Focus* are Kevin Hsieh, Ganesh Ananthanarayanan, Peter Bodik, Shivaram Venkataraman, Paramvir Bahl, Matthai Philipose, Phillip B. Gibbons, and Onur Mutlu. Their listed institutions are Carnegie Mellon University, Microsoft, University of Wisconsin, and ETH Zürich. [osdi18-hsieh:p2:t0]

</details>

<details><summary>Q4 answer (3 citations)</summary>

The plot’s labeled configuration options are **Balance**, **Opt-Query**, and **Opt-Ingest** [osdi18-hsieh:p9:t2].

The speaker says they could select different options, including **the option that balances these two metrics**; for low-latency, frequently queried applications they would select the option that pays more ingest cost for lower query latency, and for applications like home cameras they would optimize for ingest cost with higher latency [hsieh:a30] [osdi18-hsieh:p9:t3].

</details>

## 2026-09-07 00:43 — post-audit: baseline (stable sorts, dead config removed)

mode=`baseline` · index=`data/processed/index` · embedder=`BAAI/bge-m3` · units=196 · top_k=8

corpus manifest `6ee13080f9ad2084` (196 units) · gold stamped `dded007ab45f5b9f`

> **WARNING** gold set was written against corpus dded007ab45f5b9f, but the current corpus is 6ee13080f9ad2084 (196 units: hsieh.mp3, osdi18-hsieh.pdf, osdi18_slides_hsieh.pdf). Gold coverage may be understated -- a newly added document can answer a question its gold locators do not mention.

| Q | type | modality distribution | expanded | gold evidence | gold missed | gold terms in answer |
|---|---|---|---:|---|---|---|
| Q1 | `slides_only` | text 6/8, audio 2/8 | 0 | **no** | osdi18_slides_hsieh.pdf p.5 | 0/2 |
| Q2 | `audio_only` | text 3/8, audio 5/8 | 0 | **yes** | — | 3/3 |
| Q3 | `cross_modal_split` | 8/8 text | 0 | **no** | osdi18_slides_hsieh.pdf p.1, hsieh.mp3 0.6-30.0s | 4/4 |
| Q4 | `cross_modal_deictic` | text 5/8, audio 3/8 | 0 | **partial** | osdi18_slides_hsieh.pdf p.20 | 2/3 |

<details><summary>Q1 answer (2 citations)</summary>

The talk cites **NoScope** as the state-of-the-art **query-time** approach [hsieh:a41][osdi18-hsieh:p11:t3]. The provided evidence gives the name and says it is a “recent state-of-the-art querying system,” but it does **not** state the publication venue or year [osdi18-hsieh:p11:t3].

</details>

<details><summary>Q2 answer (2 citations)</summary>

In the Q&A, the speaker says they produce cheap CNNs with a simple manual approach: they start with a simple CNN, ResNet18, and remove layers one by one to generate a set of cheap CNNs for exploration [hsieh:a49]. The speaker also says this process is not automatic: “we don't have the automatic way to generate the cheap CNNs based on the architecture of the expensive ones” [hsieh:a50].

</details>

<details><summary>Q3 answer (1 citations)</summary>

The authors of *Focus* are Kevin Hsieh, Ganesh Ananthanarayanan, Peter Bodik, Shivaram Venkataraman, Paramvir Bahl, Matthai Philipose, Phillip B. Gibbons, and Onur Mutlu. Their listed institutions are Carnegie Mellon University, Microsoft, University of Wisconsin, and ETH Zürich. [osdi18-hsieh:p2:t0]

</details>

<details><summary>Q4 answer (3 citations)</summary>

The plot’s labeled configuration options are **Balance**, **Opt-Query**, and **Opt-Ingest** [osdi18-hsieh:p9:t2].

The speaker says they can select different options, including **the option that balances these two metrics**; for low-latency, frequently queried applications they would select the option that pays more ingest cost for lower query latency (**Opt-Query**), while for applications like home cameras they would optimize for ingest cost with higher latency (**Opt-Ingest**) [hsieh:a30] [osdi18-hsieh:p9:t3].

</details>

## 2026-09-07 00:44 — post-audit: FIRST TRUE LINKRAG ENTRY (mode dispatch fixed)

mode=`linkrag` · index=`data/processed/index` · embedder=`BAAI/bge-m3` · units=196 · top_k=8

corpus manifest `6ee13080f9ad2084` (196 units) · gold stamped `dded007ab45f5b9f`

> **WARNING** gold set was written against corpus dded007ab45f5b9f, but the current corpus is 6ee13080f9ad2084 (196 units: hsieh.mp3, osdi18-hsieh.pdf, osdi18_slides_hsieh.pdf). Gold coverage may be understated -- a newly added document can answer a question its gold locators do not mention.

| Q | type | modality distribution | expanded | gold evidence | gold missed | gold terms in answer |
|---|---|---|---:|---|---|---|
| Q1 | `slides_only` | text 5/8, figure 2/8, audio 1/8 | 3 | **no** | osdi18_slides_hsieh.pdf p.5 | 0/2 |
| Q2 | `audio_only` | text 2/8, figure 1/8, audio 5/8 | 3 | **yes** | — | 3/3 |
| Q3 | `cross_modal_split` | text 5/8, figure 3/8 | 3 | **no** | osdi18_slides_hsieh.pdf p.1, hsieh.mp3 0.6-30.0s | 0/4 |
| Q4 | `cross_modal_deictic` | text 4/8, figure 2/8, audio 2/8 | 3 | **yes** | — | 3/3 |

<details><summary>Q1 answer (3 citations)</summary>

The talk cites **NoScope** as the state-of-the-art **query-time-only** approach [hsieh:a41] [osdi18-hsieh:p11:t3] [osdi18_slides_hsieh:p26:t0].

The provided evidence does **not** state the **venue and year** where NoScope was published, so I can’t answer that part from the evidence alone [osdi18-hsieh:p11:t3].

</details>

<details><summary>Q2 answer (2 citations)</summary>

In the Q&A, the speaker says they generate cheap CNNs with a simple manual approach: start from a ResNet18 CNN and remove layers one by one to create a set of cheaper CNNs for exploration [hsieh:a49]. The speaker also says this is **not automatic**: “we don't have the automatic way to generate the cheap CNNs based on the architecture of the expensive ones” [hsieh:a50].

</details>

<details><summary>Q3 answer (0 citations)</summary>

The provided evidence does not include the paper’s author list or affiliations, so I can’t determine who the authors are or their institutions from this evidence alone.

</details>

<details><summary>Q4 answer (2 citations)</summary>

The plot’s labeled configuration options are:

- **Optimize for Ingest Cost** [osdi18_slides_hsieh:p20:t0]
- **Balance** [osdi18_slides_hsieh:p20:t0]
- **Optimize for Query Latency** [osdi18_slides_hsieh:p20:t0]

The speaker says they can **select different options depending on the application**: the **balanced** option to balance the two metrics, the **low-latency / optimize-for-query-latency** option for applications that query a lot, and the **optimize-for-ingest-cost** option for cases like home cameras where queries are rare and higher latency is acceptable [hsieh:a30]

</details>


### Reading the first true linkrag entry

The two entries above are the first pair produced after `--mode linkrag` was fixed to
call `retrieve_linkrag`. **Every linkrag entry in this file dated before them used
baseline retrieval** regardless of its label; only entries stamped `mode=baseline`
were ever accurate, and they remain so.

| Q | baseline | linkrag | change |
|---|---|---|---|
| Q1 | gold no, terms 0/2 | gold no, terms 0/2 | modality only: figures enter the set |
| Q2 | gold **yes**, terms 3/3 | gold **yes**, terms 3/3 | unchanged |
| Q3 | gold no, terms **4/4** | gold no, terms **0/4** | **regression** |
| Q4 | gold **partial**, terms 2/3 | gold **yes**, terms **3/3** | **improvement** |

**Q4 is the win.** `osdi18_slides_hsieh.pdf p.20` — the gold unit missed by every run
since Phase 1 — is now retrieved, via `hsieh:a28 → audio_slide → p20`. Gold coverage
reaches `yes` for the first time and the answer carries all three gold terms.

**Q3 is a loss, and it is caused by expansion.** Baseline returns 8 seeds and scores
4/4 gold terms because the OSDI paper's title block answers the question outright.
linkrag returns 5 seeds + 3 expanded, so three baseline units are displaced — including
the one that was answering it. Gold locator coverage was already `no` in both modes, so
this shows up only in the terms column, which is why the terms column matters.

**Expansion is exactly 3 units on every question**, because `k_final - k_seed = 8 - 5`
and RRF seed scores (~0.03) always exceed `seed × link × decay` (~0.01). Expansion can
only fill the tail; it can never displace a *weak* seed, only the *last* ones. On Q3
those last three were the useful ones. This is the structural limitation recorded in
the Phase 4 notes, now with a measured cost attached.

Net on this set: link-following converts one partial to a pass and one pass to a fail.
It is not yet a win, and the complementarity reranker (Phase 5) is the component meant
to decide *which* units get displaced rather than dropping whatever ranked last.

## Labels v1 → v2: every number that moved

v2 corrects inferred slide ends, removes three phantom zero-length slides, identifies
two build-slide pairs, and moves the outro start 19:05 → 19:00. **No retrieval, link or
alignment code changed between these two columns** — only the ground truth did.

| measure | v1 | v2 |
|---|---:|---:|
| talk segments | 42 | 42 |
| ambiguous talk segments | 15 | **9** |
| showable slides | 23 | 26 pages / 24 distinct |
| DP exact | 69.0% | **76.2%** |
| DP ±1 | 90.5% | **92.9%** |
| DP exact, non-ambiguous | 63.0% (n=27) | **72.7%** (n=33) |
| naive exact | 52.4% | **57.1%** |
| naive ±1 | 71.4% | **73.8%** |
| DP pages covered | 21/23 | 21/26 |
| DP backward steps | 0 | 0 |
| Q&A: DP distinct slides | 1 | 1 |
| Q&A: naive distinct slides | 5 | 5 |
| tuned `start_prior_mu` | 0.02 | **0.02 (unchanged)** |

**The aligner did not improve; the measurement did.** Roughly seven points of the DP's
apparent gain is v1 mislabelling. The DP-over-naive margin is stable (+16.7pp v1,
+19.1pp v2), which is the part that was never dependent on label quality.

One v1 caveat is retired: the mu gain no longer sits only on ambiguous segments — the
non-ambiguous subset moves 69.7% → 72.7% with mu, so the head-of-sequence fix is real.
## 2026-09-07 01:37 — post figure-clustering: baseline (OLD gold, pending re-approval)

mode=`baseline` · index=`data/processed/index` · embedder=`BAAI/bge-m3` · units=209 · top_k=8

corpus manifest `2f3b35f27e86caf8` (209 units) · gold stamped `dded007ab45f5b9f`

> **WARNING** gold set was written against corpus dded007ab45f5b9f, but the current corpus is 2f3b35f27e86caf8 (209 units: hsieh.mp3, osdi18-hsieh.pdf, osdi18_slides_hsieh.pdf). Gold coverage may be understated -- a newly added document can answer a question its gold locators do not mention.

| Q | type | modality distribution | expanded | gold evidence | gold missed | gold terms in answer |
|---|---|---|---:|---|---|---|
| Q1 | `slides_only` | text 6/8, audio 2/8 | 0 | **no** | osdi18_slides_hsieh.pdf p.5 | 0/2 |
| Q2 | `audio_only` | text 2/8, audio 6/8 | 0 | **yes** | — | 3/3 |
| Q3 | `cross_modal_split` | 8/8 text | 0 | **no** | osdi18_slides_hsieh.pdf p.1, hsieh.mp3 0.6-30.0s | 4/4 |
| Q4 | `cross_modal_deictic` | text 4/8, audio 4/8 | 0 | **partial** | osdi18_slides_hsieh.pdf p.20 | 1/3 |

<details><summary>Q1 answer (3 citations)</summary>

The talk/paper cites **NoScope** as the **state-of-the-art query-time** approach [hsieh:a41][osdi18-hsieh:p11:t3].

The provided evidence does **not state the venue or year** of NoScope’s publication, beyond calling it “a recent state-of-the-art querying system [51]” [osdi18-hsieh:p11:t3].

</details>

<details><summary>Q2 answer (2 citations)</summary>

In the Q&A, the speaker says they currently make cheap CNNs by starting with a simple CNN, specifically ResNet18, and then removing layers one by one to create a set of cheaper CNNs to explore [hsieh:a49]. He also says this is a relatively simple current approach and that they do **not** have an automatic way to generate cheap CNNs from the architecture of the expensive ones [hsieh:a49][hsieh:a50].

</details>

<details><summary>Q3 answer (1 citations)</summary>

The authors of *Focus* are Kevin Hsieh, Ganesh Ananthanarayanan, Peter Bodik, Shivaram Venkataraman, Paramvir Bahl, Matthai Philipose, Phillip B. Gibbons, and Onur Mutlu [osdi18-hsieh:p2:t0].

Their institutions are Carnegie Mellon University, Microsoft, University of Wisconsin, and ETH Zürich [osdi18-hsieh:p2:t0].

</details>

<details><summary>Q4 answer (2 citations)</summary>

The plot’s labeled configuration options are:
- **Balance** [osdi18-hsieh:p9:t2]
- **Opt-Query** [osdi18-hsieh:p9:t2]
- **Opt-Ingest** [osdi18-hsieh:p9:t2]

The speaker says they could select different ones depending on the application, including:
- the option that **balances** the two metrics [hsieh:a30]
- an option that pays more ingest cost to get **lower query latency** for query-heavy, low-latency applications [hsieh:a30]
- an option that optimizes for **ingest cost** with higher latency for cases like home cameras [hsieh:a30]

So the speaker does **not** name a single universally chosen option; they say the selection depends on the application [hsieh:a30].

</details>

## 2026-09-07 01:38 — post figure-clustering: linkrag (OLD gold, pending re-approval)

mode=`linkrag` · index=`data/processed/index` · embedder=`BAAI/bge-m3` · units=209 · top_k=8

corpus manifest `2f3b35f27e86caf8` (209 units) · gold stamped `dded007ab45f5b9f`

> **WARNING** gold set was written against corpus dded007ab45f5b9f, but the current corpus is 2f3b35f27e86caf8 (209 units: hsieh.mp3, osdi18-hsieh.pdf, osdi18_slides_hsieh.pdf). Gold coverage may be understated -- a newly added document can answer a question its gold locators do not mention.

| Q | type | modality distribution | expanded | gold evidence | gold missed | gold terms in answer |
|---|---|---|---:|---|---|---|
| Q1 | `slides_only` | text 5/8, figure 2/8, audio 1/8 | 3 | **no** | osdi18_slides_hsieh.pdf p.5 | 0/2 |
| Q2 | `audio_only` | text 2/8, figure 2/8, audio 4/8 | 3 | **yes** | — | 3/3 |
| Q3 | `cross_modal_split` | text 5/7, figure 2/7 | 2 | **no** | osdi18_slides_hsieh.pdf p.1, hsieh.mp3 0.6-30.0s | 4/4 |
| Q4 | `cross_modal_deictic` | text 3/8, figure 3/8, audio 2/8 | 3 | **yes** | — | 2/3 |

<details><summary>Q1 answer (3 citations)</summary>

The talk cites **NoScope** as the state-of-the-art **query-time-only** approach [hsieh:a41][osdi18-hsieh:p11:t3][osdi18_slides_hsieh:p26:t0].

The provided evidence does **not** state the **venue** or **year** in which NoScope was published, so I can’t answer that part from the evidence alone [osdi18-hsieh:p11:t3].

</details>

<details><summary>Q2 answer (2 citations)</summary>

In the Q&A, the speaker says they currently make cheap CNNs by starting with a simple CNN, specifically ResNet18, and then removing layers one by one to create a set of cheaper CNNs for exploration [hsieh:a49]. The speaker also says this process is not automatic: they “don’t have the automatic way” to generate cheap CNNs from the architecture of the expensive ones [hsieh:a50].

</details>

<details><summary>Q3 answer (1 citations)</summary>

The authors of **Focus** are **Kevin Hsieh, Ganesh Ananthanarayanan, Peter Bodik, Shivaram Venkataraman, Paramvir Bahl, Matthai Philipose, Phillip B. Gibbons, and Onur Mutlu** [osdi18-hsieh:p2:t0].

Their institutions are:
- **Carnegie Mellon University** — Kevin Hsieh, Phillip B. Gibbons [osdi18-hsieh:p2:t0]
- **Microsoft** — Ganesh Ananthanarayanan, Peter Bodik, Shivaram Venkataraman, Paramvir Bahl, Matthai Philipose [osdi18-hsieh:p2:t0]
- **University of Wisconsin** — Shivaram Venkataraman [osdi18-hsieh:p2:t0]
- **ETH Zürich** — Onur Mutlu [osdi18-hsieh:p2:t0]

</details>

<details><summary>Q4 answer (2 citations)</summary>

The plot’s labeled configuration options are:

- **Balance** [osdi18_slides_hsieh:p20:g0]
- **Optimize for Ingest Cost** [osdi18_slides_hsieh:p20:g0]
- **Query Latency** / effectively optimizing for low query latency [osdi18_slides_hsieh:p20:g0]

The speaker says they would select different ones depending on the application:

- For a default trade-off, select the option that **balances** the two metrics [hsieh:a30].
- For applications that require **low latency** and are queried a lot, select the option that pays a bit more ingest cost to get **lower query latency** [hsieh:a30].
- For applications such as **home cameras**, select the option that **optimizes for ingest cost** and accepts higher latency [hsieh:a30].

</details>


### Effect of slide-figure clustering alone (gold unchanged)

Both runs above use the **old gold**, stamped `dded007ab45f5b9f`, deliberately: the only
variable between these and the previous pair is figure extraction. **Gold is pending
re-approval** — see `reports/gold_proposal_pilot01.md`, which now proposes two more deck
figure units and argues Q3 has degraded again.

Corpus 196 → 209 units; deck figures 18 → 31; 24 of 27 slides now carry a figure unit
(was 11).

| Q | baseline before | baseline after | linkrag before | linkrag after |
|---|---|---|---|---|
| Q1 | no, 0/2 | no, 0/2 | no, 0/2 | no, 0/2 |
| Q2 | yes, 3/3 | yes, 3/3 | yes, 3/3 | yes, 3/3 |
| Q3 | no, **4/4** | no, **4/4** | no, **0/4** | no, **4/4** |
| Q4 | partial, **2/3** | partial, **1/3** | **yes**, **3/3** | **yes**, **2/3** |

**Q3's linkrag regression disappeared — but not because anything was fixed.** In the
previous run expansion displaced three seeds, one of which was the paper title block
answering the question. Here expansion added only two units (one of its targets was
already a seed), the set came back at 7 units instead of 8, and the answering seed
survived. That is a corpus-composition accident, not a repair. The structural cause
recorded earlier still stands: expansion drops whatever ranked last, not whatever is
least useful, and the complementarity reranker is what addresses it.

**Q4 lost a gold term in both modes** (2/3 → 1/3 baseline, 3/3 → 2/3 linkrag) while
linkrag retains gold coverage `yes`. Figures now compete for the same eight slots, so
some text that carried a gold term was displaced by a figure unit whose OCR does not
contain it. Retrieval got more cross-modal and slightly less lexically complete.

**Figures now appear in every linkrag evidence set** (2–3 of 8), where before the
clustering they were largely absent. That is the intended effect of the extraction fix,
and it is visible in the modality column rather than in the gold column.

## Four-mode retrieval comparison (2026-09-07)

`scripts/compare_retrieval.py`, 4 pilot questions, k=8, corpus `2f3b35f27e86caf8`
(209 units). `linkrag_iter` seeds link-following from the iterative retriever.

| method | recall@8 | prec@8 | LLM calls | recall@8 (normalise_seeds) | prec@8 |
|---|---:|---:|---:|---:|---:|
| baseline | 33.3% | 9.4% | 0 | 33.3% | 9.4% |
| iterative (P1) | 52.1% | 18.8% | 4 | 56.2% | 18.8% |
| linkrag | 41.7% | 12.5% | **0** | **50.0%** | 15.6% |
| **linkrag_iter** | **60.4%** | **21.9%** | 4 | **75.0%** | **28.1%** |

Per-question recall@8 with `normalise_seeds=true`:

| qid | baseline | iterative | linkrag | linkrag_iter |
|---|---:|---:|---:|---:|
| Q1 | 0% | 0% | 0% | 0% |
| Q2 | 100% | 100% | 100% | 100% |
| Q3 | 0% | 25% | 0% | **100%** |
| Q4 | 33% | 100% | 100% | **100%** |

**The two mechanisms are complementary, and the per-question table shows why.**
Q4 needs an *edge* — `hsieh:a28 → audio_slide → p20` — and link-following finds it while
plain iterative does not reliably. Q3 needs a *seed*: the `hsieh:a0 → audio_slide →
p1:t0` edge exists, but `hsieh:a0` never enters the top-8 (Q3's seeds are all paper text
units), so traversal never starts there. A second query supplies the missing entry
point. Neither alone solves both; composed, they solve both.

**This is the diagnosis for Q3 requested as a Phase-4 follow-up: link-following inherits
the seed retriever's recall and cannot reach a region of the graph that has no seed.**

`normalise_seeds` rank-normalises seeds to (0,1] so `seed × link × decay` is
commensurable with a seed score. Without it, RRF seeds (~0.03) dwarf expanded scores
(~0.01) and expansion can only fill the tail. It is **off by default** — turning it on
changes every recorded linkrag number.

### Caveats

- **n=4.** Nothing here is a result; it is a direction.
- **Gold is the old 2-document set** (`dded007ab45f5b9f`), pending re-approval. Q1 scores
  0% in every mode partly because its gold locator is a slide the retriever never reaches.
- **The two LLM-using modes are not deterministic.** `iterative` moved 52.1% → 56.2%
  between two runs whose only difference was a flag it does not read; that is follow-up
  query variation, not the flag. Only `baseline` and `linkrag` repeat exactly.
- Latencies were recorded but are **not reportable** under the measurement rules
  (single run, non-idle machine).

## (mode × rerank) matrix — SMOKE TEST, n=4, NOT REPORTABLE (2026-09-10)

`scripts/compare_retrieval.py --repeats 3`. Corpus `2f3b35f27e86caf8` (209 units),
k=8, pool=20. Backend **openai**, model **gpt-5.4-2026-03-05**, temperature **0.0**,
seed **20260910**. 24 LLM calls. Gold is still the stale 2-document set.

**n=4 questions. This is a smoke test of the harness, not a result.** No number here
belongs in the paper.

| mode | rerank | recall@8 | precision@8 | distinct modalities | redundancy |
|---|---|---:|---:|---:|---:|
| baseline | none | 33.3% | 9.4% | 1.75 | 0.711 |
| baseline | mmr | 25.0% | 6.2% | 2.00 | 0.655 |
| baseline | complementarity | **12.5%** | 3.1% | 2.25 | 0.682 |
| iterative | none | 47.9% ± 7.2% | 16.7% ± 3.6% | 2.50 | 0.709 |
| iterative | mmr | 32.6% ± 12.0% | 10.4% ± 3.6% | 2.33 | 0.642 |
| iterative | complementarity | 29.9% ± 2.4% | 11.5% ± 1.8% | 2.50 | 0.649 |
| linkrag | none | 41.7% | 12.5% | 2.75 | 0.688 |
| linkrag | mmr | 29.2% | 9.4% | 2.75 | 0.614 |
| linkrag | complementarity | 37.5% | 12.5% | 2.75 | 0.635 |
| **linkrag_iter** | none | 63.2% ± 4.8% | 22.9% ± 1.8% | 3.00 | 0.716 |
| **linkrag_iter** | mmr | 54.2% ± 0.0% | 21.9% ± 0.0% | 3.00 | 0.568 |
| **linkrag_iter** | **complementarity** | **70.8% ± 7.2%** | **27.1% ± 1.8%** | **3.00** | 0.599 |

### The reranker needs a pool worth reranking

Complementarity **helps only in the one cell whose candidate pool already spans all
three modalities**, and hurts everywhere else:

| mode | pool modalities | none → complementarity |
|---|---:|---|
| baseline | 1.75 | 33.3% → **12.5%** (−20.8pp) |
| iterative | 2.50 | 47.9% → 29.9% (−18.0pp) |
| linkrag | 2.75 | 41.7% → 37.5% (−4.2pp) |
| linkrag_iter | 3.00 | 63.2% → **70.8%** (+7.6pp) |

The objective does exactly what it was written to do — coverage rises and redundancy
falls in every mode. But on a pool that is mostly text, "add an unrepresented modality"
means evicting a gold text unit to admit a figure, and the gold for these four
questions is predominantly text. **α and β are only worth paying when the pool contains
cross-modal evidence to select; otherwise they buy diversity with recall.** That is a
statement about the interaction, not about the reranker being wrong.

The monotone trend across the four pools is the substantive observation here, and it is
the thing to re-test on a larger question set.

### seed and temperature=0 do NOT make the LLM modes reproducible

Both LLM-touching modes produced **3 distinct outcomes in 3 runs** despite
`temperature: 0.0` and `seed: 20260910` in the outgoing request body (asserted by
`test_request_body_carries_temperature_and_seed`). The API accepts `seed` but returns
`system_fingerprint: null` for this account, so there is no backend-stability signal to
check against — and empirically it is not stable.

Consequences, now enforced in the harness:

- Every LLM-touching cell is run 3× and reported as **mean ± std**; a single-run number
  for such a cell is **refused, not printed**.
- `iterative + mmr` has a std of **12.0pp** on recall — larger than most of the
  differences anyone would want to claim between modes.
- `baseline` and `linkrag` remain deterministic by construction and are run once.


## 2026-09-20 — verified gold (24 questions, 71 units), four modes × rerank none/complementarity

24 questions · k=8 · pool=20 · corpus `2f3b35f27e86caf8` (209 units) · gold stamped `2f3b35f27e86caf8` · model `gpt-5.4-2026-03-05` temperature=0.0 seed=20260910 · repeats 3 for LLM modes · 144 LLM calls

| mode | rerank | recall@8 | precision@8 | distinct modalities | redundancy | runs |
|---|---|---:|---:|---:|---:|---:|
| baseline | none | 45.1% | 15.1% | 2.04 | 0.716 | 1 |
| baseline | complementarity | 42.4% | 13.0% | 2.33 | 0.651 | 1 |
| iterative | none | 58.3% ± 0.3% | 19.3% ± 0.0% | 2.40 | 0.725 | 3 |
| iterative | complementarity | 58.9% ± 1.6% | 18.6% ± 0.3% | 2.51 | 0.670 | 3 |
| linkrag | none | 39.4% | 14.1% | 2.79 | 0.689 | 1 |
| linkrag | complementarity | 43.2% | 15.7% | 2.83 | 0.602 | 1 |
| linkrag_iter | none | 45.8% ± 0.8% | 16.3% ± 0.3% | 2.78 | 0.694 | 3 |
| linkrag_iter | complementarity | 45.1% ± 3.6% | 16.3% ± 1.2% | 2.88 | 0.605 | 3 |

Repeat determinism (complementarity cell):

- `baseline`: deterministic by construction (no LLM, single run)
- `iterative`: NOT identical — 3 distinct outcomes in 3 runs
- `linkrag`: deterministic by construction (no LLM, single run)
- `linkrag_iter`: NOT identical — 3 distinct outcomes in 3 runs

Per-question recall@k (mean over runs):

| Q | type | baseline/none | baseline/complementarity | iterative/none | iterative/complementarity | linkrag/none | linkrag/complementarity | linkrag_iter/none | linkrag_iter/complementarity |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Q1 | single_modality | 0% | 0% | 22% | 22% | 0% | 0% | 11% | 11% |
| Q2 | audio_only | 50% | 50% | 50% | 25% | 50% | 25% | 50% | 25% |
| Q3 | paper_only | 17% | 0% | 67% | 50% | 17% | 17% | 67% | 67% |
| Q4 | single_modality | 50% | 25% | 50% | 33% | 38% | 38% | 42% | 42% |
| A1 | slides_only | 100% | 100% | 100% | 100% | 0% | 0% | 33% | 0% |
| A2 | slides_only | 0% | 50% | 67% | 83% | 0% | 0% | 0% | 0% |
| A4 | audio_only | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| A5 | audio_only | 0% | 0% | 100% | 100% | 0% | 0% | 0% | 0% |
| A7 | paper_only | 0% | 0% | 0% | 17% | 0% | 0% | 0% | 0% |
| A8 | single_modality | 100% | 50% | 100% | 50% | 100% | 100% | 100% | 100% |
| B1 | paper_only | 50% | 100% | 100% | 100% | 50% | 50% | 100% | 100% |
| B2 | cross_modal_split | 50% | 50% | 50% | 33% | 50% | 25% | 25% | 17% |
| B3 | cross_modal_split | 100% | 50% | 100% | 100% | 50% | 50% | 67% | 67% |
| B4 | cross_modal_split | 50% | 50% | 50% | 50% | 50% | 50% | 50% | 50% |
| B5 | single_modality | 33% | 67% | 78% | 78% | 33% | 100% | 89% | 100% |
| B7 | audio_only | 33% | 33% | 33% | 33% | 33% | 33% | 33% | 33% |
| B8 | cross_modal_split | 100% | 100% | 50% | 100% | 100% | 50% | 67% | 50% |
| C1 | single_modality | 33% | 33% | 78% | 100% | 33% | 67% | 67% | 78% |
| C2 | audio_only | 25% | 0% | 0% | 25% | 25% | 0% | 0% | 0% |
| C3 | audio_only | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% |
| C4 | single_modality | 33% | 33% | 56% | 56% | 33% | 33% | 44% | 44% |
| C5 | cross_modal_deictic | 25% | 25% | 17% | 25% | 50% | 100% | 0% | 0% |
| C6 | slides_only | 33% | 0% | 33% | 33% | 33% | 100% | 56% | 100% |
| C7 | slides_only | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |

### Reading the verified-gold run

This is the first run on gold that was **not** written by hand, and it reverses the
n=4 picture:

- **Link-following alone loses recall** (39.4 % vs baseline 45.1 %). Cause, confirmed
  by a diagnostic with `k_seed: 8` (not appended, scratch config): with `k_seed: 5`
  the mode keeps only the top-5 seeds and fills the remaining three slots with
  expanded units, so any gold unit the hybrid retriever ranked 6–8 is evicted
  (A1 100 % → 0 %, A2, A5). With `k_seed: 8` and `normalise_seeds: false`,
  `linkrag/none` is byte-identical to `baseline/none` — expansions never outrank a
  seed — and `linkrag/complementarity` reaches 42.9 % at 2.88 modalities against the
  baseline's 42.4 % at 2.33. **Link-following buys modality coverage, not recall, on
  this set.**
- **P1-style re-querying is the strongest single mechanism** (58.3 % ± 0.3), and
  composing it with link-following (`linkrag_iter`, 45.8 %) is *worse* than
  re-querying alone — the same seed-eviction effect applied to the iterative pool.
- **The reranker is roughly neutral** (−2.7 pp baseline, +3.8 pp linkrag, +0.6 pp
  iterative) while raising modality coverage in every mode, which matches the
  pool-coverage interaction recorded on 2026-09-10.
- **Only 5 of 24 questions are cross-modal after verification** (B2, B3, B4, B8, C5).
  C5, the strongest deictic question, is the one place link-following clearly wins
  (25 % → 50 %, 100 % with complementarity) — and it is one question.

What this does and does not say: the mechanism's benefit is concentrated on
genuinely cross-modal questions, and this corpus has almost none once a machine
checks. The 2026-09-07 headline (75 % for `linkrag_iter`) rested on four hand-picked
questions and page-level gold; it does not survive unit-level, verified gold. The
next numbers that matter are on LectQA-Vid and MaViLS (priorities iii, iv), not here.
The `k_seed < k_final` eviction is a design defect to fix before those runs, with a
recorded ablation, not a silent retune.
