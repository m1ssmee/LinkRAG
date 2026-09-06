# Proposed pilot01 question set — 24 candidates

**Proposal only. Not added to the regression set.** Every question is to be verified by
ear before approval. Corpus: manifest `2f3b35f27e86caf8`, 209 units
(111 text, 47 figure, 51 audio) across deck, OSDI paper and frozen transcript.

Method: exclusive-vocabulary term-diff over the four sources
(deck-text / deck-figure / paper / audio). A term marked *only* under one source was
verified absent from the other three. Where a type requires a fact to be unavailable
from one modality, the evidence column names the term and the sources that lack it.

Locators follow the regression format: `file + page` or `file + time span`.

⚠️ **Two design cautions carried from the existing set.** Slide-figure clustering put
affiliations and plot labels into deck *figure* units, so a "slides-only" fact can now
live in two units of one page; and the paper duplicates most deck content. Both are
noted per question where they bite.

---

## A. Single-modality (8) — 3 slides, 3 audio, 2 paper

### A1 · `slides_only` — What does the deck say the paper covers that the talk does not?
**Expected:** Characterization of real-world videos, implementation details, other applications.
**Gold:** `osdi18_slides_hsieh.pdf` p.25.
**Evidence:** `characterization` and `implementation details` appear in **deck-text only**
— absent from deck-figures, paper and audio. The "More in the Paper" slide is never read aloud.

### A2 · `slides_only` — Which object classes appear in the top-K example?
**Expected:** The example classes shown on the object-class slide (incl. passenger/recreational categories).
**Gold:** `osdi18_slides_hsieh.pdf` p.12.
**Evidence:** `passenger`, `recreational` occur in **deck only** (text + figure); absent
from paper and audio. `airplane` is *not* usable — it also appears in the paper.

### A3 · `slides_only` — What speedup does approximate indexing alone give, as printed on the components chart?
**Expected:** 89× (rising to 162× with clustering).
**Gold:** `osdi18_slides_hsieh.pdf` p.23.
**Evidence:** the token `89X` appears in **deck-text and deck-figure only**. The paper
states the same quantity in different notation, so the *printed form* is deck-exclusive
— a genuine slides-only fact, though a paraphrasing model could answer from the paper.
**Flag for approval:** weaker than A1/A2 for that reason.

### A4 · `audio_only` — What poster number does the speaker direct the audience to?
**Expected:** Poster number 47.
**Gold:** `hsieh.mp3` 1118.9–1147.4s.
**Evidence:** `number 47` occurs in **audio only**. The deck says "poster" but never the number.

### A5 · `audio_only` — In the live demo, how long did the baseline take and how long was the video?
**Expected:** About four minutes for the baseline; the video is six hours.
**Gold:** `hsieh.mp3` ≈1022–1082s.
**Evidence:** `four minutes`, `six hours`, `month -long` all **audio only**. Slide 24 is
a live demo not present in the deck, so no document records these numbers.

### A6 · `audio_only` — Which institutions does the speaker credit at the opening?
**Expected:** Carnegie Mellon, Microsoft, University of Wisconsin, ETH Zurich.
**Gold:** `hsieh.mp3` 0.6–30.0s.
**Evidence:** `collaborators` is **audio only**. ⚠️ **Degraded**: the institution *names*
now also appear in deck-figure OCR (`p1:g1`, `p27:g1`) and in the paper. Only the
spoken framing is exclusive. **Recommend rejecting or rewording** — this is the same
collapse that broke Q3.

### A7 · `paper_only` — Which deep-learning framework and version implements Focus?
**Expected:** Microsoft Cognitive Toolkit 2.4.
**Gold:** `osdi18-hsieh.pdf` p.11.
**Evidence:** `cognitive toolkit` occurs in **paper only**. Never on a slide, never spoken.

### A8 · `paper_only` — What does the paper mean by a Pareto-optimal parameter setting?
**Expected:** A setting where neither ingest cost nor query latency can improve without worsening the other; the dashed line in Figure 6 is the Pareto boundary.
**Gold:** `osdi18-hsieh.pdf` p.9.
**Evidence:** `pareto` occurs in **paper only**. The deck shows the same plot without the term.

---

## B. Cross-modal split-fact (8)

Each requires two facts that verifiably do not co-occur in one source.

### B1 · `cross_modal_split` — Which video datasets are evaluated, and what kind of scene is `auburn_c`?
**Expected:** Named datasets (auburn_c, bellevue_d, bend, jackson_ts, coral, lausanne, oxford, sittard); auburn_c is a commercial-area intersection in the City of Auburn.
**Gold:** `osdi18_slides_hsieh.pdf` p.23 **+** `osdi18-hsieh.pdf` p.11.
**Evidence:** dataset names appear in deck **and** paper, but `commercial area intersection` is **paper only** — the descriptions exist nowhere in the deck or audio.

### B2 · `cross_modal_split` — What poster number is given, and which sections does the paper say cover specialization?
**Expected:** Poster 47; §4.2 and §4.3.
**Gold:** `hsieh.mp3` 1118.9–1147.4s **+** `osdi18-hsieh.pdf` p.7.
**Evidence:** `number 47` **audio only**; `§4.2` **paper only**.

### B3 · `cross_modal_split` — What does the aquarium dataset test, and what does the speaker say about querying month-long video?
**Expected:** The aquarium/coral dataset is a surveillance-style static camera; the speaker says NoScope needs five hours to query a month-long video.
**Gold:** `osdi18-hsieh.pdf` p.11 **+** `hsieh.mp3` 864–892s.
**Evidence:** `aquarium` **paper only**; `five hours` and `month -long` **audio only**.

### B4 · `cross_modal_split` — Which earlier cascade work does the paper cite, and how does the speaker describe the same idea?
**Expected:** Viola et al. (cascaded classifiers); the speaker describes cheap classifiers filtering frames before expensive CNNs.
**Gold:** `osdi18-hsieh.pdf` p.15 **+** `hsieh.mp3` ≈176–206s.
**Evidence:** `viola` **paper only**; the spoken paraphrase carries no citation.

### B5 · `cross_modal_split` — What are the three trade-off configurations, and which does the speaker say suits home cameras?
**Expected:** Optimize-for-Ingest-Cost / Balance / Optimize-for-Query-Latency; for home cameras, optimize for ingest cost and accept higher latency.
**Gold:** `osdi18_slides_hsieh.pdf` p.20 **+** `hsieh.mp3` 793.7–821.5s.
**Evidence:** the printed labels are deck (text + figure); the home-camera application is spoken only.

### B6 · `cross_modal_split` — What does the deck list as remaining paper content, and what does the paper actually report for moving cameras?
**Expected:** Deck lists characterization/implementation/other applications; the paper reports Focus on moving cameras (cnn, foxnews, msnbc).
**Gold:** `osdi18_slides_hsieh.pdf` p.25 **+** `osdi18-hsieh.pdf` p.12.
**Evidence:** `characterization` **deck-text only**; `moving cameras` **paper only**.

### B7 · `cross_modal_split` — What ground-truth CNN is used, and why does the speaker say that choice was made?
**Expected:** YOLOv2 as GT-CNN; chosen because the NoScope baseline ships with YOLOv2 code.
**Gold:** `osdi18-hsieh.pdf` p.11 **+** `hsieh.mp3` ≈476–506s.
**Evidence:** `gt-cnn` **paper only**; the spoken reason is in audio. Note `yolov2` (paper/deck)
vs `yolo v2` (audio) are different tokens — a lexical retriever must bridge them.

### B8 · `cross_modal_split` — What does the takeaways slide claim, and what does the speaker add about the Q&A poster?
**Expected:** Takeaways slide's three key results; the speaker adds the poster number and invites questions.
**Gold:** `osdi18_slides_hsieh.pdf` p.26 **+** `hsieh.mp3` 1118.9–1147.4s.
**Evidence:** `takeaways` **deck only**; `number 47` **audio only**.

---

## C. Deictic (8) — drawn from the pointing windows

All eight sit inside a labelled window in `pilot01_pointing_windows.csv`. Slides 13, 14,
22 and 23 are emphasised as requested; each now has an extracted figure unit (they had
none before slide-figure clustering).

### C1 · `deictic` — slide 13, window 7:48–9:14
**Q:** When the speaker says the recall "becomes even lower", which curves on screen is he comparing?
**Expected:** The ResNet18 variants — full, 4 fewer layers, 6 fewer layers.
**Gold:** `hsieh.mp3` 506–536s **+** `osdi18_slides_hsieh.pdf` p.13 (figure).
**Evidence:** the curve labels live in the deck; the comparison is spoken. Referent is visual-only.

### C2 · `deictic` — slide 13, window 7:48–9:14
**Q:** What does "the small top-k results" refer to on the chart being shown?
**Expected:** The low-K region of the x-axis (number of selected results).
**Gold:** `hsieh.mp3` 537–563s **+** `osdi18_slides_hsieh.pdf` p.13 (figure).
**Evidence:** `top -k` as spoken never appears in any document; the axis label does.

### C3 · `deictic` — slide 14, window 9:14–10:19
**Q:** In the architecture diagram on screen, what is generated at ingest time?
**Expected:** A top-K index mapping object classes to objects, from the specialized compressed CNN.
**Gold:** `hsieh.mp3` 564–580s **+** `osdi18_slides_hsieh.pdf` p.14 (figure).
**Evidence:** `specialized, compressed cnn` is deck-text and paper; the diagram is the referent.

### C4 · `deictic` — slide 14, window 9:14–10:19
**Q:** When the speaker says query-time work is "only done when the user wants to query", which part of the diagram is he pointing at?
**Expected:** The query-time half of the split diagram.
**Gold:** `hsieh.mp3` 609–636s **+** `osdi18_slides_hsieh.pdf` p.14 (figure).
**Evidence:** the split is drawn, not stated in prose anywhere.

### C5 · `deictic` — slide 22, window 14:20–15:33
**Q:** When the speaker says "the focus will be at here", where on the plot does Focus sit?
**Expected:** Low on both axes — low ingest cost and low query latency, toward the "Better" corner.
**Gold:** `hsieh.mp3` 893–921s **+** `osdi18_slides_hsieh.pdf` p.22 (figure).
**Evidence:** `zoom -in` is **audio only**; the position is purely visual. **The strongest
deictic candidate in the set** — "at here" has no textual referent at all.

### C6 · `deictic` — slide 22, window 14:20–15:33
**Q:** Which two baselines are marked on the plot, and what are their plotted costs?
**Expected:** Ingest-heavy ($380/month/stream) and NoScope (5 hours/month/stream).
**Gold:** `hsieh.mp3` 864–892s **+** `osdi18_slides_hsieh.pdf` p.22.
**Evidence:** the plotted annotations are deck; the speaker names them while pointing.

### C7 · `deictic` — slide 23, window 15:33–16:07
**Q:** When the speaker says "both techniques are very important", which two series on the chart does he mean?
**Expected:** Approximate indexing alone (89×) and +Clustering (162×).
**Gold:** `hsieh.mp3` 950–970s **+** `osdi18_slides_hsieh.pdf` p.23 (figure).
**Evidence:** `89X` printed form is **deck only**; the series names are chart labels.

### C8 · `deictic` — slide 20, window 12:44–13:31
**Q:** Which configuration on the plot does the speaker say balances the two metrics?
**Expected:** The "Balance" configuration, between Optimize-for-Ingest-Cost and Optimize-for-Query-Latency.
**Gold:** `hsieh.mp3` 773.9–806.3s **+** `osdi18_slides_hsieh.pdf` p.20.
**Evidence:** the three labels are printed on the slide (text and figure OCR); the selection is spoken.
⚠️ Overlaps the existing Q4 — approve only one of the two.

---

## Verification checklist before approval

1. **Listen to each window** — every C-question assumes the speaker is pointing at what
   the timeline says was on screen. The alignment's known p24→p26 lead shows that
   assumption can fail.
2. **A3, A6 and C8 are flagged above** as weaker or overlapping; decide before adding.
3. **Type labels assume the current corpus.** Adding a document can collapse a
   cross-modal question into a single-source one — that has already happened twice to Q3.
4. Gold units are given as locators, never ids: unit ids regenerate on every
   re-transcription or re-chunk.
