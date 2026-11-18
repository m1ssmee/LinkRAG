# Plan: XFile, a cross-file benchmark for the Evidence Linking Layer

## Context

The comparison against T1 (LectQA-Vid) and T2 (MaViLS) is done and reported. Neither benchmark
can test LinkRAG's own idea. Both are single-file, and every LectQA-Vid transcript fits in
context. As a result, the README lists the Evidence Linking Layer, link-following and
complementarity reranking as "system components, not results".

This plan builds a small benchmark of our own (working name **XFile**). Its questions need
evidence from two or more separate files on the same topic, so the core idea gets a test it
can pass or fail. Decisions already taken with the user:

- Corpus: MaViLS lectures (transcript and slide PDF) plus the course's lecture notes from
  MIT OCW.
- Questions: Claude drafts them, and the user and their friend check them.
- Models: free backends, plus Claude subagents where the free tiers' caps stop a run. Cost $0.

Work happens on branch `cross-file-benchmark`.

## Step 0: Record the policy change

- `DESIGN.md` → *Dataset policy*: add a dated note (2026-11-xx). The comparison phase is
  closed. XFile is an own-collected benchmark for testing the novelty, and is reported
  separately from the T1/T2 tables. Mark the "extended cross-file dataset plan … dropped" line
  as superseded rather than deleting it. Add XFile to *Priority order*.
- The user tells the mentor in one line. Nothing waits on a reply unless the mentor objects.

## Step 1: Corpus (about 8–10 topics)

- Clone MaViLS (`git clone --depth 1 https://github.com/andererka/MaViLS data/raw/mavils`),
  which is not present locally. It ships the SRT transcripts and slide PDFs.
- Choose lectures that have a matching MIT OCW lecture-notes PDF (CC BY-NC-SA, kept local and
  never committed). Prefer diagram-heavy decks, as `scripts/dataset/README.md` advises.
- Layout: `data/raw/xfile/<topic>/{<topic>_talk.srt, <topic>_slides.pdf, <topic>_notes.pdf}`.
  File basenames must be unique across the corpus, because gold matching is by basename.
- **New, small:** `scripts/adapters/xfile.py prepare` converts each MaViLS SRT into the
  frozen-transcript JSON that `linkrag.ingest.audio.load_frozen_transcript`
  (`src/linkrag/ingest/audio.py:170`) reads. It spreads word times evenly within each
  sentence. It also writes the stub audio path that `frozen_transcript_for` resolves by stem,
  so ingest skips Whisper. Before writing it, check the exact JSON schema against
  `data/processed/transcripts/hsieh.frozen.json`.
- **New:** `configs/xfile.yaml`, following the pattern of `configs/pilot01-w1.yaml`. It gives
  separate `store_dir`, `links_path`, `frozen_transcript_dir`, **and** `figures_dir`,
  `vlm_captions` and `citations` under `data/processed/xfile/`. The last three are still
  shared in pilot01-w1.yaml, so they must not be left as they are. It also sets
  `slide_deck_files` so that notes PDFs are never treated as decks.
- Ingest the whole corpus as **one pooled index**, so retrieval has to find the right file, and
  then run `scripts/build_links.py`. This exercises the existing cross-file links: audio_slide
  for every recording–deck pair with the relatedness gate, figure_text between deck and notes
  through `document_pair_gate`, and deictic links.

## Step 2: Judge and intake gate

- The cost policy says a free backend may judge only after it has been measured. Run
  `scripts/eval/judge_agreement.py` for `groq` and for `google_ai_studio` against the stored
  pilot01 verdicts. This costs $0. Use whichever agrees best with the stored verdicts (κ vs
  gpt-4.1-mini). If neither reaches an acceptable κ (set the bar before looking, e.g. ≥ 0.7),
  the judge is Claude subagents, scored through a `from_agents`-style path like
  `scripts/adapters/lectqa/closedbook.py`.
- Run `scripts/dataset/candidate.py`'s redundancy gate (`src/linkrag/eval/redundancy.py`,
  `--sample 50`) for each topic. Drop topics that are REJECTED, where the deck mostly repeats
  the talk. Record the decision log in `scripts/dataset/README.md`.

## Step 3: Questions (Claude drafts, humans check, the gold checker filters)

- Source material: each topic's redundancy report ends with "what each source says that the
  others do not". From it, Claude drafts per topic:
  - about 8 `cross_modal_split` questions, each needing a fact from file A **and** a fact
    from file B;
  - about 2 `cross_modal_deictic` questions, where the referent exists only on a slide or
    figure;
  - about 4 single-source controls. These are needed to show that links do no harm.
  Rows use the existing `*_proposed.jsonl` format (`gold_units` as `{source, page}` /
  `{source, start_s, end_s}`). Reference answers contain only the asked-for facts (the
  pilot01 run-1 lesson).
- Review sheet `reports/xfile_review.csv` with the columns qid, question, answer, gold
  locators, accept/edit/reject and note. The user and friend split topics between them, and
  rejected rows are deleted.
- Run `scripts/eval/verify_gold.py` on the accepted rows (3 runs). Its per-source check (whole
  transcript / whole deck / whole notes / all) already decides whether a question really
  needs more than one file. Questions that need more are kept as cross-modal; the rest are
  relabelled or dropped. Target: **≥ 60 verified cross-file questions**. If too few survive,
  draft another round on the topics that kept the most.
- Human audit: `scripts/eval/audit_sample.py` (10%, at least 5 per type), and the user and
  friend actually fill in the sheet. This also closes known issue (d) for XFile.
- Output: `tests/regression/xfile_questions.jsonl` with `_meta.manifest_hash`.

## Step 4: Evaluation

- **Split by topic** into a tune half and a test half, recorded before any run. Knobs
  (`decay`, α/β/γ, `normalise_seeds`, `expansion`) may be set on the tune half only and are
  applied once to the test half.
- **Fix the control first.** `scripts/compare_retrieval.py` passes the link graph to the
  reranker even in baseline mode, so "baseline + complementarity" is not a no-links control.
  Pass an empty graph when the mode is baseline, add a test for it, and record the fix as a
  correction.
- **Retrieval run** ($0): `compare_retrieval.py` over baseline / linkrag / iterative /
  linkrag_iter × none / mmr / complementarity, with `--repeats 3`. It reports evidence
  recall@k per type (cross-modal vs single) and modality coverage.
- **Answer quality.** Retrieval-only metrics cannot reward a linked slide, as
  `lectqa_pooled.md` showed. Score answers against `expected_answer` with the
  same judge pass `verify_gold.source_run` uses (reuse it, don't write a new judge). Also run
  `--grounding` for hallucination rate and citation correctness. Use 3 runs.
- **Statistics:** paired bootstrap that resamples **topics**, the sampled unit (per the
  reporting convention). An interval containing 0 is reported as "no measurable difference".
- **Headline comparison:** linkrag + complementarity vs. the fixed no-links control, on
  cross-file questions in the test half, with single-source questions as the no-harm check.

## Step 5: Report

- `results/xfile/xfile_benchmark.md` contains the corpus table (topics, files, redundancy),
  question counts per type that sum to the total, and the retrieval and answer tables with
  provenance (models, n, runs, manifest hash).
- Update the README *Contributions* and *Limitation* sections **only according to what the
  test half shows**, including a null result if that is what comes out.

## Files

- New: `scripts/adapters/xfile.py`, `configs/xfile.yaml`, `tests/test_xfile_adapter.py`,
  `tests/regression/xfile_questions.jsonl`, `reports/xfile_review.csv`,
  `results/xfile/xfile_benchmark.md`.
- Changed: `DESIGN.md` (policy note, priority list, findings), `scripts/compare_retrieval.py`
  (empty graph for the baseline control) and its test, `scripts/dataset/README.md` (decision
  log), `README.md` (at the end, evidence-led).
- Reused unchanged: `candidate.py`, `redundancy.py`, `verify_gold.py`, `audit_sample.py`,
  `judge_agreement.py`, `build_links.py`, `link/pipeline.py`, `retrieve/linkrag.py`,
  `retrieve/rerank.py`, `SpendGuard` (`src/linkrag/costs.py`).

## Verification

- `make test` passes, including the new adapter test and the baseline-control test.
- Execution rule: run `xfile.py prepare`, ingest and `build_links.py` once for real on the
  corpus and read the output. Check unit counts per file, check that `links.jsonl` contains
  deck↔notes figure_text links and at least one audio_slide pair per topic, and check that
  `unrelated_pairs` makes sense.
- Sanity: on one topic, `scripts/ask.py --mode linkrag --show-evidence` on a verified
  cross-file question shows provenance (`via_seed`/`via_link`) for an expanded unit.
- Measurement rules: if a linkrag row exactly equals baseline, or is exactly 0, inspect it
  before believing it. Per-type counts must reconcile to the total.
- All runs use `--max-cost 0`, and `reports/llm_ledger.jsonl` shows $0.

## Tune / test split (fixed 2026-11-18, before any evaluation run)

`random.Random(20261118).shuffle(sorted(topics))`, first 3 = tune, rest = test:

- **tune:** nfa_closure, psych_methods, visual_system
- **test:** gp_scaling, lightning, mimic_data, rl_mdp

Knobs (`decay`, alpha/beta/gamma, `normalise_seeds`, `expansion`, any gate change from DESIGN.md
issue (n)) are set on tune only and applied once to test. Runs over all 7 topics with default
knobs and no tuning are labelled *exploratory*.

## Intake gate outcome (2026-11-18)

The deck→transcript redundancy gate (DESIGN.md *Sampled redundancy*: keep below 65 %, census
decides borderline) gave **KEEP** for 6 topics and **REJECT** for `lightning` (census 67.3 %).
The rule is applied as written: `lightning` leaves the reported XFile tables. Its questions stay
in the gold file and are reported as a separate, labelled row. The test half is therefore
**gp_scaling, mimic_data, rl_mdp**; tune is unchanged. The gate is a fixed rule and was not
chosen by looking at any evaluation number. Reports: `reports/redundancy_xfile_*.md`.

## Knobs frozen at defaults (2026-11-18, before any tune answer-correctness number)

Deadline decision (Rakshit and m1ssmee): no knob is tuned. The `configs/xfile.yaml` defaults are
final for both halves, and no gate change from issue (n) is made. The tune half is still run and
reported as a check, and the test half may start before the tune half finishes, because nothing
from tune can change it. At the time of writing, no tune answer-correctness number had been
computed: the tune run stopped at the usage limit with grading incomplete. The only numbers seen
before this point are the exploratory retrieval-only ones (`retrieval_exploratory.md`, all topics,
default knobs), which the frozen defaults do not depend on. The report states that no tuning was
done.

## When the plain top-k comparison was added (confirmed 2026-11-18)

Asked in the paper review; confirmed by m1ssmee's session and checked against git by Rakshit's.
No test-topic answer score existed when the plain top-k comparison was added:

| Time (18 Nov, IST) | Event | Evidence |
|---|---|---|
| 10:54 | Exploratory retrieval-only comparison, all topics, unreviewed gold, no answer scores | a94c1a9 |
| 18:16 | Knobs frozen at defaults | e6f0cbf |
| 19:33 | Paper Table 3 includes plain top-k (`baseline|none`) as a row | 17e5de4, `paper/xfile_fill.py` |
| 21:05 | Tune check commits the rule: read the test headline against plain top-8 as well | a8e520a, `eval_tune.md` |
| 21:08 | First test-topic prompt queued (file time on m1ssmee's machine) | `test1.jsonl` |
| 22:40 | Test results and paired rows, including LinkRAG vs plain top-8 | 703adef |

The only test-topic numbers before 21:05 are the exploratory retrieval-only ones (a94c1a9); the paper
discloses them in §3.1.1.

*Hashes:* `main` was rewritten on 18 Nov at about 23:17 IST to drop Claude co-author lines (content and author
dates unchanged; the tip's file tree is identical). Hashes in this file and in the XFile results refer to the
rewritten history; for example, the freeze commit, formerly `bf53fd7`, is now `e6f0cbf`.
