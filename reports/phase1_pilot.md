# Phase 1 pilot — baseline multimodal RAG on `pilot01`

Corpus: `hsieh.mp3` (22.9 min talk) + `osdi18_slides_hsieh.pdf` (27 slides) —
Hsieh et al., *Focus: Querying Large Video Datasets with Low Latency and Low Cost*, OSDI '18.

Mode: **baseline** (RRF of dense + BM25, plain top-k, no links). 
Embedder `BAAI/bge-m3`; LLM `gpt-5.4-2026-03-05` @ temperature 0.0; top_k=8.

## Segmentation

Step 2 asked for sentence-aware segmentation *if* fixed windows cut mid-sentence.
They did (70%). The first fix -- packing whole whisper segments, as specified --
**did not work**: whisper's segments are decoder chunks, not sentences, so only
~26% of them end on punctuation and packing them left boundaries 74% mid-sentence,
marginally worse than doing nothing. Splitting the word stream on terminal
punctuation (whisper attaches it to words) and packing those sentences fixes it.

| segmentation | audio units | boundaries mid-sentence | mean len | max len |
|---|---:|---:|---:|---:|
| fixed-window (baseline papers' approach) | 46 | **32/46 (70%)** | 29.1s | 30.0s |
| packed whisper segments (first attempt) | 50 | **37/50 (74%)** | 26.7s | 30.0s |
| sentence-split + packed (shipped) | 53 | **0/53 (0%)** | 24.7s | 32.3s |

Transcript content itself is good throughout: technical terms survive
(`ResNet 18`, `YOLO V2`, `NoScope`, `162 times`). Recurring ASR error: *ingest*
is transcribed as **"interest"** almost everywhere -- worth noting, since it
degrades BM25 recall on the single most important term in this talk.

## Questions

### Q1 · `slides_only`

**Q.** Which prior system does the talk cite as the state-of-the-art query-time approach, and in which venue and year was it published?

**Gold modalities:** figure, text — retrieved modalities: audio, text → **partial**

| rank | score | id | modality | location | content (truncated) |
|---:|---:|---|---|---|---|
| 1 | 0.0307 | `hsieh:a43` | audio | 1118.9s-1147.4s | Our result shows that focus is 57 times cheaper than ingest time only solution and it's 162 times faster than  |
| 2 | 0.0298 | `hsieh:a32` | audio | 837.5s-863.5s | The first baseline is interest store which I explained earlier that analyzed all the frames using the expensiv |
| 3 | 0.0297 | `hsieh:a11` | audio | 285.1s-314.7s | And secondly, the layer before the final output of these probabilities is often referred to as a feature layer |
| 4 | 0.0294 | `hsieh:a0` | audio | 0.6s-30.0s | Good afternoon. I'm sure everyone wants to end. This is the last session, so I'm sure I won't go over time. So |
| 5 | 0.0290 | `hsieh:a8` | audio | 205.9s-229.5s | However, even with all these optimizations, there's a state of the art system, no scope, that's exactly that.  |
| 6 | 0.0267 | `hsieh:a5` | audio | 119.8s-149.7s | So then when the user want to query a particular object class, let's say a truck, then we can just return the  |
| 7 | 0.0262 | `osdi18_slides_hsieh:p26:t0` | text | p.26 | Key Takeaways • Problem: Querying objects in massive videos is challenging • Our Approach: Low-latency query w |
| 8 | 0.0260 | `hsieh:a22` | audio | 580.7s-608.5s | And then at query time, when we want to say, want to find a truck, we will find the objects that match this cl |

- slide-only gold terms ['kang', 'pvldb']: in retrieval NONE; in answer NONE

**Answer.** The talk cites **NoScope** as the **state-of-the-art query-time-only** approach [hsieh:a32] [hsieh:a43].

The provided evidence does **not state the venue or year** where NoScope was published [hsieh:a32] [hsieh:a43].

**Citations:** 2 of 8 units

**Verdict (mechanical): wrong — required evidence never retrieved**

**Verdict (reviewed): partial (honest refusal; retrieval failure)** — Named NoScope correctly and explicitly declined the venue/year rather than inventing it. The failure is retrieval: slide p5 carries `Kang et al., NoScope, PVLDB'17` and was never returned -- 6 of 8 slots went to audio.

### Q2 · `audio_only`

**Q.** In the Q&A, how does the speaker say the cheap CNNs are produced, and is that process automatic?

**Gold modalities:** audio — retrieved modalities: audio → **yes**

| rank | score | id | modality | location | content (truncated) |
|---:|---:|---|---|---|---|
| 1 | 0.0323 | `hsieh:a14` | audio | 369.5s-394.7s | Those compressed, especially CNNs, reduce, remove layers, reduce weights. And they specialize the CNN for each |
| 2 | 0.0318 | `hsieh:a50` | audio | 1285.4s-1311.3s | So in our experiments the CNNs are about five to 40, 50 times faster than the expensive CNNs and because we ev |
| 3 | 0.0315 | `hsieh:a52` | audio | 1338.8s-1368.5s | So right now we actually use a relatively simple approach. We use a simple CNN Resnet 18 and then we remove la |
| 4 | 0.0311 | `hsieh:a51` | audio | 1311.4s-1337.6s | So there's a training phase that we have to do to select parameters and choose the CNNs and that takes a few h |
| 5 | 0.0310 | `hsieh:a25` | audio | 646.3s-661.9s | Since we already processed those objects using CNN, the cheap CNN, now we can extract the features that we get |
| 6 | 0.0277 | `hsieh:a4` | audio | 95.9s-119.3s | So a natural approach to enable such query with short latency is we can interest and process those live videos |
| 7 | 0.0271 | `hsieh:a16` | audio | 421.0s-446.6s | That is, we found that the best results in the expensive CNN usually show up in the top K results of the cheap |
| 8 | 0.0271 | `hsieh:a20` | audio | 537.2s-563.2s | In another word, we can use cheap CNNs to achieve high recall when we are using the small top k results. But w |

- audio-only gold terms ['resnet', 'layer', 'automatic']: in retrieval ['resnet', 'layer', 'automatic']; in answer ['resnet', 'layer', 'automatic']

**Answer.** In the Q&A, the speaker says they currently make cheap CNNs by starting with a simple CNN, specifically ResNet-18, and then removing layers one by one to create a set of cheap CNNs. [hsieh:a52]

He also says this is not automatic: they “don’t have the automatic way to generate the cheap CNN based on the architecture of the expensive ones.” [hsieh:a52]

**Citations:** 1 of 8 units

**Verdict (mechanical): complete**

**Verdict (reviewed): complete** — Both halves correct and cited to hsieh:a52, the exact Q&A segment. The one question whose gold evidence is single-modality is the one the baseline gets right.

### Q3 · `cross_modal_split`

**Q.** Who are the authors of Focus, and which institutions are they from?

**Gold modalities:** audio, figure, text — retrieved modalities: audio, text → **partial**

| rank | score | id | modality | location | content (truncated) |
|---:|---:|---|---|---|---|
| 1 | 0.0315 | `hsieh:a43` | audio | 1118.9s-1147.4s | Our result shows that focus is 57 times cheaper than ingest time only solution and it's 162 times faster than  |
| 2 | 0.0293 | `hsieh:a29` | audio | 745.8s-773.2s | So in focus, we have several parameters that we can select. We can use that to enable flexible tradeoff betwee |
| 3 | 0.0286 | `hsieh:a34` | audio | 893.1s-921.1s | So to plot focus in this figure the focus will be at here and we also have the zoom in figures with different  |
| 4 | 0.0277 | `hsieh:a39` | audio | 1022.0s-1049.8s | We can see that the query here is the baseline and we going to check more than 10 ,000 frames and obviously we |
| 5 | 0.0272 | `osdi18_slides_hsieh:p23:t0` | text | p.23 | Effect of Different Components 23 89X 162X 1 10 100 1000 auburn_c bellevue_d bend jackson_ts coral lausanne ox |
| 6 | 0.0270 | `hsieh:a31` | audio | 807.4s-837.3s | So we can enable different tradeoffs based on the application's need. Now let's see how we evaluate our system |
| 7 | 0.0267 | `hsieh:a0` | audio | 0.6s-30.0s | Good afternoon. I'm sure everyone wants to end. This is the last session, so I'm sure I won't go over time. So |
| 8 | 0.0261 | `hsieh:a11` | audio | 285.1s-314.7s | And secondly, the layer before the final output of these probabilities is often referred to as a feature layer |

- slide-only gold terms ['ananthanarayanan', 'bodik']: in retrieval NONE; in answer NONE
- audio-only gold terms ['microsoft', 'carnegie']: in retrieval ['microsoft', 'carnegie']; in answer ['microsoft', 'carnegie']

**Answer.** The evidence only names the presenter and the institutions involved in the joint work; it does not list all authors. Specifically, the talk says it is “joint work with my collaborators in Carnegie Mellon University, Microsoft, University of Wisconsin, and ETH Zurich,” and identifies the speaker as “Saurabh Bakshi from Purdue.” [hsieh:a0] [hsieh:a43]

So, based on the provided evidence:
- Named person: Saurabh Bakshi [hsieh:a43]
- Institutions mentioned: Purdue, Carnegie Mellon University, Microsoft, University of Wisconsin, and ETH Zurich [hsieh:a43] [hsieh:a0]

The evidence does not provide the full author list.

**Citations:** 2 of 8 units

**Verdict (mechanical): partial (explicit refusal on the missing half)**

**Verdict (reviewed): WRONG (citation valid, answer false)** — Asserted 'Saurabh Bakshi from Purdue' as the named presenter, citing hsieh:a43. That string really is in a43 -- it is an *audience member introducing themselves to ask a question*, captured in the same unit as the talk's closing line. The citation check passes and the answer is still wrong. The title slide p1, which carries the real author list, was never retrieved. This is precisely the failure P2 cannot detect: it has no hallucination check, and a citation-validity test like ours passes here too.

### Q4 · `cross_modal_deictic`

**Q.** In the figure the speaker introduces by saying they plotted a set of different configurations, what are the configuration options labelled on the plot, and which one does the speaker say they would select?

**Gold modalities:** audio, figure, text — retrieved modalities: audio → **partial**

| rank | score | id | modality | location | content (truncated) |
|---:|---:|---|---|---|---|
| 1 | 0.0325 | `hsieh:a29` | audio | 745.8s-773.2s | So in focus, we have several parameters that we can select. We can use that to enable flexible tradeoff betwee |
| 2 | 0.0320 | `hsieh:a30` | audio | 773.9s-806.3s | And here we have different set of configurations and then we can select different options such as the option t |
| 3 | 0.0262 | `hsieh:a38` | audio | 1003.0s-1021.3s | So as we can see, let's say we want to query different videos and so in this video this is a video on the stre |
| 4 | 0.0255 | `hsieh:a34` | audio | 893.1s-921.1s | So to plot focus in this figure the focus will be at here and we also have the zoom in figures with different  |
| 5 | 0.0245 | `hsieh:a26` | audio | 662.3s-686.5s | This is a well established study so that we can now cluster those features based on the distance and so that w |
| 6 | 0.0244 | `hsieh:a18` | audio | 476.1s-505.6s | We use an expensive CNN YOLO V2 as our ground truth. And then we use three different RESTnet 18 architecture,  |
| 7 | 0.0244 | `hsieh:a21` | audio | 563.9s-580.1s | So when we process those objects using the compressed and specialized CNN, we will generate a top k index wher |
| 8 | 0.0240 | `hsieh:a17` | audio | 447.0s-476.1s | Recall is a fraction of the objects that return to the user among the other populations that we can find. And  |

- slide-only gold terms ['optimize for ingest cost', 'optimize for query latency']: in retrieval NONE; in answer ['optimize for ingest cost']
- audio-only gold terms ['balance']: in retrieval ['balance']; in answer ['balance']

**Answer.** The evidence says the plot contains “different set of configurations,” and the speaker gives example choices among them:  
- “the option that balances these two metrics”  
- an option for “applications that requires low latency and they query a lot”  
- an option for “applications such as home cameras” that “optimize for ingest cost with a higher latency” [hsieh:a30].

However, the provided evidence does **not** give the exact labels shown on the plot, so I can’t name the configuration labels precisely from this evidence alone [hsieh:a29] [hsieh:a30].

As for which one the speaker says they would select: the speaker says they can select different ones depending on the application, including the balanced option, the low-latency option, or the low-ingest-cost/higher-latency option; they do **not** state a single universally chosen one in the provided evidence [hsieh:a30].

**Citations:** 2 of 8 units

**Verdict (mechanical): complete**

**Verdict (reviewed): partial (honest refusal; retrieval failure)** — Recovered the speaker's spoken tradeoff options and correctly declined the plotted labels. All 8 retrieved units were audio -- slide p20, which carries `Optimize for Ingest Cost` / `Optimize for Query Latency`, never surfaced. A deictic phrase ('in this figure', 'here we have') has no way to reach the figure it points at.

## Summary

| Q | type | gold modalities retrieved | modalities returned | reviewed verdict |
|---|---|---|---|---|
| Q1 | `slides_only` | partial | audio, text | partial (honest refusal; retrieval failure) |
| Q2 | `audio_only` | yes | audio | complete |
| Q3 | `cross_modal_split` | partial | audio, text | WRONG (citation valid, answer false) |
| Q4 | `cross_modal_deictic` | partial | audio | partial (honest refusal; retrieval failure) |

## What the baseline actually failed at

1. **Modality collapse.** 53 of 98 units are audio, and audio dominated every
   ranking: Q4 returned 8/8 audio, Q1 and Q3 returned 6/8. The slide carrying the
   answer was never retrieved for Q1, Q3 or Q4. Both rankers are modality-blind and
   RRF does nothing to rebalance them, so the larger modality wins on volume alone.
2. **A valid citation is not a correct answer.** Q3 cited a real unit and still
   asserted a false fact, because a Q&A self-introduction and an author list are
   indistinguishable to a flat index. Citation validity is necessary, not sufficient
   -- the eval needs a claim-level check, not just an id-level one.
3. **Deixis is unresolvable without links.** Q4's segment says "in this figure" and
   "here we have"; nothing in the baseline can turn either into a pointer to slide
   p20. This is the gap the Evidence Linking Layer exists to close.
4. **ASR noise hits the load-bearing term.** *ingest* is transcribed **"interest"**
   throughout, so BM25 cannot match the talk's central concept, and the dense side
   carries retrieval alone on exactly the queries that need it most.

Honest caveat: the two questions the baseline handled acceptably (Q1's system name,
Q2 in full) are the ones whose evidence sits in a single modality. Every question
needing two modalities degraded. n=4 on one lecture -- indicative, not a result.

