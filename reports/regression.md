# Regression — pilot01 Q1–Q4

The same four questions, re-run every phase. Gold evidence is matched by
file+page or file+time-overlap, not by unit id. `modality` is the retrieved
set's composition — a gain that only reshuffles within one modality is not
the cross-modal gain LinkRAG claims.

## Ear-labelled alignment evaluation (2026-09-06)

Ground truth is now a hand-made slide timeline read from the recording
(`data/labels/pilot01/pilot01_slide_timeline.csv`), converted to per-segment labels
for all 51 frozen audio segments. **Ear-labelled, approximate (±3 s), n=42 talk
segments** (15 of them ambiguous), plus 9 Q&A segments with no true slide.
23 of 27 slides are showable — slide 27 was never shown; 9, 11 and 18 are zero-length.

The labelled slide sequence is **monotonic non-decreasing**, independently confirming
the assumption `align_monotonic` is built on.

### Talk section — all 42 labelled segments

| method | exact | ±1 | backward steps | slides covered |
|---|---:|---:|---:|---:|
| **monotonic DP (ours)** | **29/42 = 69.0%** | **90.5%** | **0** | **21/23** |
| naive argmax (P2-style) | 22/42 = 52.4% | 71.4% | 9 | 15/23 |

### Talk section — non-ambiguous subset (n=27)

| method | exact | ±1 | backward steps | slides covered |
|---|---:|---:|---:|---:|
| **monotonic DP (ours)** | **17/27 = 63.0%** | **85.2%** | **0** | 15/23 |
| naive argmax (P2-style) | 14/27 = 51.9% | 74.1% | 5 | 13/23 |

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
