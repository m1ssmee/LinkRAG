# Regression — pilot01 Q1–Q4

The same four questions, re-run every phase. Gold evidence is matched by
file+page or file+time-overlap, not by unit id. `modality` is the retrieved
set's composition — a gain that only reshuffles within one modality is not
the cross-modal gain LinkRAG claims.

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

