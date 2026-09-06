# Phase 4 read-only audit — `src/linkrag/**`, `scripts/**`

4,779 lines across 37 modules. Nothing was modified. Every claim below was checked
against the code and, where the claim is numeric, against `data/processed/links.jsonl`
or `links.npz`.

Ranked by risk to the paper's claims: **could produce a false result** > **could crash
or silently degrade** > **cosmetic**.

---

## A. Could produce a false result

### A1. `--mode linkrag` does not change retrieval in the standing regression instrument
`scripts/run_regression.py:33` declares `--mode {baseline,linkrag}`; line **57** always
calls `retrieve_scored` (baseline top-k). The flag reaches only `answer(..., mode=)` at
line **63**, which changes a *prompt suffix*. Line **98** then stamps the report
`mode=linkrag`.

Running `run_regression.py --mode linkrag` today emits a report labelled linkrag whose
retrieval is entirely baseline. This is the `w_slide` pattern exactly: a mode flag that
does not reach the path it names. No published number is wrong — every existing entry
in `regression.md` says `mode=baseline` — but the trap is armed and the instrument is
the one the project uses to show improvement over time.

**Fix:** branch on `args.mode` at line 57 to call `retrieve_linkrag` (loading links and
building the graph as `ask.py:50-62` does), or reject `--mode linkrag` until it does.

### A2. The deictic threshold filters nothing in linkrag mode
`link/deictic.py:206` sets `mass = w_slide + w_cue + w_dense + w_overlap = 1.00` for
linkrag. Line **250** discards every off-slide figure, so for every *scored* candidate
`on_slide == 1.0` and line **257** adds a constant `w_slide/mass = 0.45`. The
configured threshold is `deictic_threshold: 0.45` (`configs/default.yaml:132`).

So every on-slide candidate begins exactly at the threshold and can only rise.
Confirmed empirically: 98 deictic links, **min score 0.5622, zero links below 0.50**.
The threshold has never rejected anything. Two consequences: (a) "links above
threshold" is not a quality statement; (b) `w_slide` cannot affect *ranking* between
candidates — it is a constant offset — so the config presents as a weight something
that only moves scores relative to an inert threshold.

**Fix:** exclude `w_slide` from `mass` (or from the score) in linkrag mode so the
threshold discriminates on evidence that actually varies.

### A3. Seven config keys are never read; one of them looks like a live filter
Never read by any code in `src/` or `scripts/`:
`audio_slide_threshold` (`configs/default.yaml:130`), `same_topic_threshold` (:133),
`link_hops` (:143), `max_evidence_units` (:145), `rerank` + `redundancy_penalty` +
`modality_bonus` (:159-162).

`audio_slide_threshold: 0.55` is the dangerous one: the actual audio_slide filter is
`link.align.min_score: 0.0` (:85). Observed audio_slide scores run **0.4766–0.8422**,
so a reader assuming 0.55 filtered the set would be wrong about 51 links. Anyone
tuning it would see no effect and conclude the signal is insensitive.

**Fix:** delete the six dead keys; either wire `audio_slide_threshold` into
`build_links.py:78` or delete it.

### A4. Link scores from different types are not commensurable, and are multiplied anyway
`retrieve/linkrag.py:122` computes `seed_score * link_score * decay`. Measured ranges:

| link type | range | source |
|---|---|---|
| `audio_slide` | 0.4766 – 0.8422 | weighted cos+BM25+IDF |
| `figure_text` | 0.5077 – 0.9623 | weighted ref+layout+page+cos+overlap |
| `deictic` | 0.5622 – 0.8226 | normalised, floored at 0.45 (A2) |
| `same_slide` | **exactly 1.0** | fixed constant, `link/same_slide.py:60` |
| seed score | ~0.016 – 0.033 | RRF, `retrieve/baseline.py:32` |

`same_slide` is a fixed 1.0 by design ("co-location is certain, not scored"), so if it
is ever added to `retrieve.linkrag.link_types` (`configs/default.yaml:150`, currently
`[audio_slide, figure_text, deictic]`) it will outrank every other link type
unconditionally. Latent, not live.

**Fix:** document that `link_score` must be a comparable [0,1] confidence, and give
`same_slide` a score below the top of the scored types' range, or exclude it from
expansion explicitly.

### A5. IDF and BM25 normalisation are corpus-relative, so link scores are not comparable across corpora
`link/figure_text.py:136` computes IDF over the *text units passed in*;
`link/deictic.py:236` over the *figures passed in*; `link/align.py:_rescale_rows` (line
~123) min-max scales BM25 **per row** of the candidate set.

Adding the OSDI paper changed the text set from 27 to 111 units, so every deck figure's
`figure_text` score moved for reasons unrelated to the figure. This compounds the
already-recorded finding that modality dominance is a corpus-composition effect: link
scores are also corpus-composition-dependent, and `links.jsonl` carries no manifest
stamp.

**Fix:** stamp `links.jsonl` with the corpus manifest hash (as gold files are), and
never compare link scores across manifests.

### A6. `index.build` timing includes model load
`scripts/ingest.py:40` constructs `default_encoder(...)` and passes it straight to
`build_index`, which wraps the first `encoder(texts)` call inside
`stage_timer("index.build")` (`index/__init__.py:171-179`). No warm-up call precedes
it, unlike `ask.py:46`, `build_links.py:85`, `run_regression.py:44`,
`compare_retrieval.py:70` and `pilot_report.py:131`, which all warm first.

Every `index.build` figure recorded so far (155.69s, 225.78s) therefore includes a
bge-m3 load. Under the project's own measurement rules these are not reportable.

**Fix:** call `encoder([""])` inside its own `stage_timer("encoder.warmup")` before
`build_index`, matching every other script.

### A7. Measurement rule 2 has no mechanical enforcement
DESIGN.md requires that any table whose per-category counts do not sum to the stated
total be reconciled. **No `assert` anywhere in `src/` or `scripts/` checks this.**
`scripts/build_links.py:110-117` prints `links={len(links)}` and a per-type table from
`summarise(graph)` — two different sources. They agree today only because the
edge-collapse bug was fixed; nothing prevents regression.

**Fix:** `assert sum(stats[t]["count"] for t in stats) == len(links)` in
`build_links.py` before printing.

### A8. Hardcoded reviewed verdicts in `pilot_report.py` are stale
`scripts/pilot_report.py:36` holds a `REVIEWED` dict of hand-checked verdicts written
against the 96-unit, 2-document corpus. The current corpus is 196 units. Re-running the
script prints those verdicts beside freshly computed retrieval, labelled "reviewed".

**Fix:** key `REVIEWED` by manifest hash and refuse to print when it does not match.

---

## B. Could crash or silently degrade

### B1. A stale or empty link graph makes linkrag silently identical to baseline
`retrieve/linkrag.py:118` drops links whose endpoint is absent from the index with
`continue` and no counter or log. If `links.jsonl` is stale, empty, or built against a
different index, expansion contributes nothing and `retrieve_linkrag` returns exactly
the seeds — a silent degradation to baseline. Only `t["expanded"]=0` in a log line
would reveal it. This is precisely the condition measurement rule 1 exists to catch.

**Fix:** count skipped neighbours and `log.warning` when any were dropped or when
`expanded == 0` in linkrag mode.

### B2. Bare `except Exception: continue` with no log
`ingest/pdf.py:238` swallows any failure while rendering a caption-anchored figure
region. A systematically failing render (bad clip rect, memory) silently produces fewer
figures, and figure counts are a reported quantity.

**Fix:** `log.warning("figure region render failed on p%s: %s", page_no, exc)`.

### B3. Unstable sorts break ties arbitrarily in three ranking paths
Same class as the BM25 tie-break already fixed at `index/__init__.py:114`:

- `link/figure_text.py:236` — `np.argsort(-scores[i])`, default quicksort. Ties decide
  which `figure_text` links are emitted.
- `index/__init__.py:102-103` — `argpartition` then `argsort(-scores[top])`; both
  unstable. Affects dense retrieval order.
- `link/deictic.py:261` — `scored.sort(reverse=True)` on `(score, j, overlap)` tuples;
  ties break by **figure index descending**, favouring later figures.

**Fix:** `kind="stable"` on the numpy sorts; sort on `(-score, j)` at deictic.py:261.

### B4. `gold_unit_ids` path re-introduces the stale-id hazard
`scripts/compare_retrieval.py:36-37` accepts raw `gold_unit_ids` and uses them
verbatim. Unit ids are regenerated on every re-transcription or re-chunk — the reason
`eval/metrics.py:41 matches_locator` exists. The locator path (line 40) is what the
shipped gold file uses, so this is latent.

**Fix:** warn when `gold_unit_ids` is used and no locators are present.

---

## C. Cosmetic, dead, and stale from the Phase 3 renames

| item | location | note |
|---|---|---|
| `deictic_visual`, `figure_paragraph` | `scripts/demo.py:42-43,53` | Link types that no longer exist in `LinkType`; `Link` does not validate, so demo emits invalid types silently |
| stale docstring | `src/linkrag/link/__init__.py:3-4` | still says "figure_paragraph, deictic_visual" |
| stage name `link.figure` | `link/figure_text.py:221` | module renamed; log key was not |
| `same_topic` | `core.py:32`, `configs:133`, `plot_graph.py:25` | declared, thresholded and styled; **never emitted** |
| unused import `align_monotonic` | `scripts/plot_alignment.py:17` | |
| unused import `stage_timer` | `src/linkrag/link/graph.py:19` | |

---

## D. Config flags and where each takes effect

| flag | config line | takes effect | non-default tested? |
|---|---|---|---|
| `ingest.audio_segmentation` | :33 | `ingest/audio.py:258` | code path yes, config key no |
| `ingest.ocr_figures` | :39 | `ingest/pdf.py:245` | **no** |
| `ingest.slide_deck_landscape_ratio` | :42 | `ingest/pdf.py:210` | **no** |
| `ingest.figures_from_captions` | :47 | `ingest/pdf.py:213` | yes |
| `ingest.vlm_captions.enabled` | :52 | `ingest/vlm_caption.py:104` | disabled path only |
| `ingest.asr_vocab_from_slides` | :61 | `ingest/__init__.py:145` | **no** |
| `ingest.slide_deck_files` | — | `ingest/__init__.py:97` | **no** |
| `index.normalize_embeddings` | :68 | `scripts/ingest.py:43` | **no** |
| `link.align.method` | :74 | `link/align.py:280` | yes |
| `link.align.max_back` | :82 | `link/align.py:246` | yes |
| `link.align.start_prior_mu` | :83 | `link/align.py:224` | yes |
| `link.align.min_score` | :85 | `link/align.py:305` | yes |
| `link.figure_text.*` | :90-99 | `link/figure_text.py:224-231` | yes |
| `link.same_slide.enabled` | :102 | `link/same_slide.py:45` | yes |
| `link.deictic.tier_weights` | :123 | `link/deictic.py:256` | yes |
| `link.figure_text_threshold` | :131 | `link/figure_text.py:247` | yes |
| `link.deictic_threshold` | :132 | `link/deictic.py:263` | yes — **but inert, see A2** |
| `link.audio_slide_threshold` | :130 | **nowhere** | dead |
| `link.same_topic_threshold` | :133 | **nowhere** | dead |
| `retrieve.link_hops` | :143 | **nowhere** | dead |
| `retrieve.linkrag.k_seed` | :147 | `retrieve/linkrag.py:100` | yes |
| `retrieve.linkrag.hops` | :149 | `retrieve/linkrag.py:110` | yes |
| `retrieve.linkrag.link_types` | :150 | `retrieve/linkrag.py:97` | yes |
| `retrieve.linkrag.min_link_score` | :151 | `retrieve/linkrag.py:116` | yes |
| `retrieve.linkrag.decay` | :152 | `retrieve/linkrag.py:122` | yes |
| `retrieve.iterative.rounds` | :156 | `retrieve/iterative.py:82` | yes |
| `retrieve.max_evidence_units`, `retrieve.rerank.*` | :145,:159-162 | **nowhere** | dead (Phase 5) |

**Mode-sharing check.** Every `baseline`/`linkrag` branch was traced:
`same_slide.py:45`, `figure_text.py:244`, `deictic.py:206/246/250`,
`retrieve/linkrag.py:100/110`, `generate/answer.py:157`. Only one leak remains and it
is A1. `deictic.py:246` correctly nulls `slide_of_audio` in baseline mode — the fix for
the original `w_slide` bug is intact and guarded by
`test_baseline_does_not_use_the_alignment`.

---

## The three I would fix first

**1. A1 — `run_regression.py --mode linkrag` uses baseline retrieval.**
It is the only finding that can put a false number into the paper through the
project's own standing instrument, it is silent, and the report labels itself with the
mode it did not use. Everything else is either latent or visible in the logs.

**2. A2 — the deictic threshold is inert.**
98 links are described as passing a 0.45 threshold that nothing can fail. Any claim of
the form "deictic links above threshold" is currently vacuous, and the tier analysis
(tier 2 = 100% precision) is reported against a set that was never filtered. Fixing it
will change link counts, so it should happen before any deictic number is published.

**3. B1 — a stale link graph degrades linkrag to baseline silently.**
Phase 5 will iterate on links repeatedly. The first time `links.jsonl` is out of date
with the index, linkrag and baseline will produce identical results and the only signal
will be `expanded=0` in a log line. Measurement rule 1 says identical ablation results
must be inspected; this makes the inspection automatic instead of hoping someone reads
the log.

A3, A6 and A7 are quick and worth doing in the same pass — deleting six dead config
keys, adding one warm-up call, and adding one `assert` are each a one-line change.
