# XFile: a cross-file benchmark for the Evidence Linking Layer

Own-collected benchmark (DESIGN.md *Dataset policy*, amended 2026-11-17). Reported on its own, never mixed
into a T1/T2 table. Corpus manifest **`a086ca1c22dc1d93`** (2,210 units, one pooled index).

## Caveats first

- **Single answering model family.** Answers come from `claude-haiku-5-5` and grades from
  `claude-sonnet-5-5`, both run as Claude Code subagents through the reply cache. The questions were
  drafted by Claude Opus subagents. Subagent runs can be re-scored but not re-generated. Treat every
  number as **direction only**.
- **No tuning.** Every setting is the `configs/xfile.yaml` default, frozen before any tune answer score
  existed (PLAN.md, e6f0cbf). The tune topics were run as a check only.
- **The default complementarity reranker is unstable as a control** (known issue (j)). On the tune topics
  it halved the no-links control (63.9 % -> 33.3 % correct); on the test topics it did not (55.3 % ->
  58.8 %). The headline is therefore also shown against plain top-8 retrieval.
- **Three test topics.** The paired bootstrap resamples topics, the sampled unit. With 3 topics there are
  only 10 distinct resamples, so the interval is coarse; an interval containing 0 is reported as "no
  measurable difference".
- **The audio↔slide gate leaks cross-lecture links** (known issue (n)): 4 of 9 kept recording-deck
  pairs are false, and the gate failed on mimic_data and visual_system's own decks. XFile measures the
  layer as configured.
- **Gold.** The gold check ran 3 times on the reviewed set (`gold_check_3runs.md`); 53 cross-file
  questions survived, below the plan's target of 60. Review: Rakshit read all 98 rows individually;
  m1ssmee's half was first a blanket acceptance of Claude's recommendations. Human audit of the judge's unit verdicts
  (`reports/xfile_audit_sheet.csv`, 50 rows, 10 % per type, at least 5 per topic, each person auditing
  the other's topics): **50/50 agree** (`audit_score.md`). Three rows were borderline (S018, S027,
  S031); 8 rows belong to questions the gold check dropped.
- **Read coverage.** 133 of 360 gold-check answers were first made by subagents that had not read their
  whole context (DESIGN.md (o)); they were redone, and every evaluation answer round was audited from the
  subagent transcripts before import.

## Result (test topics: gp_scaling, mimic_data, rl_mdp; 38 questions)

**No measurable gain from links.** On the pre-registered headline, linkrag + complementarity against the
no-links control baseline + complementarity, cross-file answer correctness is **38.1 % vs 47.6 %,
−9.5 points [−16.7, +0.0]**, an interval that touches 0: no measurable difference, with the point
estimate against links. Over all 38 questions: −3.5 [−9.1, +5.1]. On single-source questions (the
no-harm check): +3.9 [+0.0, +11.1], no measurable difference.

linkrag + complementarity answers **exactly** as well as plain top-8 retrieval (baseline + none) on
every question group (55.3 % overall), and link expansion without the reranker (linkrag + none) scores
identically to baseline + none on every question (recall and correctness): links change nothing unless the
reranker pulls linked units in. The only
measurable effect is on retrieval: the headline pair's recall@8 over all 38 questions is +4.3 points
[+1.5, +7.3] for links (cross-file +6.1 [−2.8, +13.6]), and that recall does not turn into correct
answers. The best cell overall is iterative re-querying without links (61.4 %, complementarity).

Models: answerer `claude-haiku-5-5` (temperature 0), judge `claude-sonnet-5-5` (3 grades per answer,
majority), both Claude Code subagents, one answer prompt per subagent. 3 runs per cell. k = 8, pool = 20.
Corpus `a086ca1c22dc1d93`, gold stamped `a086ca1c22dc1d93`. Final numbers are a cache-only replay
(`--answerer-cache-only --judge-cache-only`). Full tables: `eval_test.md`; tune check: `eval_tune.md`.

### Paired differences, 95 % interval from resampling the 3 test topics (10,000 resamples, seed 20261118)

Answer correctness unless marked; per question averaged over 3 runs; `scripts/adapters/xfile.py paired`.

### Headline (pre-registered): linkrag + complementarity vs baseline + complementarity

| questions | n | topics | linkrag + complementarity | baseline + complementarity | difference [95 % CI, topics resampled] | |
|---|---:|---:|---:|---:|---|---|
| cross-file | 21 | 3 | 38.1% | 47.6% | -9.5% [-16.7%, +0.0%] | no measurable difference |
| single-source | 17 | 3 | 76.5% | 72.5% | +3.9% [+0.0%, +11.1%] | no measurable difference |
| all | 38 | 3 | 55.3% | 58.8% | -3.5% [-9.1%, +5.1%] | no measurable difference |

### linkrag + complementarity vs plain top-8 (baseline + none)

| questions | n | topics | linkrag + complementarity | baseline + none | difference [95 % CI, topics resampled] | |
|---|---:|---:|---:|---:|---|---|
| cross-file | 21 | 3 | 38.1% | 38.1% | +0.0% [+0.0%, +0.0%] | no measurable difference |
| single-source | 17 | 3 | 76.5% | 76.5% | +0.0% [+0.0%, +0.0%] | no measurable difference |
| all | 38 | 3 | 55.3% | 55.3% | +0.0% [+0.0%, +0.0%] | no measurable difference |

### linkrag + none vs baseline + none (link expansion alone)

| questions | n | topics | linkrag + none | baseline + none | difference [95 % CI, topics resampled] | |
|---|---:|---:|---:|---:|---|---|
| cross-file | 21 | 3 | 38.1% | 38.1% | +0.0% [+0.0%, +0.0%] | no measurable difference |
| single-source | 17 | 3 | 76.5% | 76.5% | +0.0% [+0.0%, +0.0%] | no measurable difference |
| all | 38 | 3 | 55.3% | 55.3% | +0.0% [+0.0%, +0.0%] | no measurable difference |

### linkrag_iter + complementarity vs iterative + complementarity

| questions | n | topics | linkrag_iter + complementarity | iterative + complementarity | difference [95 % CI, topics resampled] | |
|---|---:|---:|---:|---:|---|---|
| cross-file | 21 | 3 | 44.4% | 50.8% | -6.3% [-11.1%, -0.0%] | no measurable difference |
| single-source | 17 | 3 | 74.5% | 74.5% | -0.0% [-5.6%, +5.6%] | no measurable difference |
| all | 38 | 3 | 57.9% | 61.4% | -3.5% [-7.7%, +2.4%] | no measurable difference |

### Headline pair, recall@8 (retrieval only)

| questions | n | topics | linkrag + complementarity | baseline + complementarity | difference [95 % CI, topics resampled] | |
|---|---:|---:|---:|---:|---|---|
| cross-file | 21 | 3 | 60.2% | 54.0% | +6.1% [-2.8%, +13.6%] | no measurable difference |
| single-source | 17 | 3 | 76.5% | 74.5% | +2.0% [+0.0%, +6.7%] | no measurable difference |
| all | 38 | 3 | 67.5% | 63.2% | +4.3% [+1.5%, +7.3%] | a higher |

### Retrieval (test, all 38 questions)

| mode | rerank | recall@8 | precision@8 | distinct modalities | redundancy | runs |
|---|---|---:|---:|---:|---:|---:|
| baseline | none | 66.1% ± 0.0% | 17.8% ± 0.0% | 2.03 | 0.675 | 3 |
| baseline | mmr | 50.6% ± 0.0% | 13.2% ± 0.0% | 2.13 | 0.590 | 3 |
| baseline | complementarity | 63.2% ± 0.0% | 16.8% ± 0.0% | 2.18 | 0.588 | 3 |
| iterative | none | 68.1% ± 2.5% | 18.9% ± 0.8% | 2.23 | 0.678 | 3 |
| iterative | mmr | 60.0% ± 1.3% | 15.6% ± 0.7% | 2.34 | 0.632 | 3 |
| iterative | complementarity | 66.1% ± 3.1% | 17.5% ± 1.0% | 2.42 | 0.625 | 3 |
| linkrag | none | 66.1% ± 0.0% | 17.8% ± 0.0% | 2.03 | 0.675 | 3 |
| linkrag | mmr | 66.1% ± 0.0% | 17.8% ± 0.0% | 2.05 | 0.668 | 3 |
| linkrag | complementarity | 67.5% ± 0.0% | 18.1% ± 0.0% | 2.95 | 0.624 | 3 |
| linkrag_iter | none | 68.3% ± 2.2% | 18.9% ± 0.5% | 2.25 | 0.680 | 3 |
| linkrag_iter | mmr | 71.0% ± 2.2% | 19.5% ± 0.5% | 2.27 | 0.665 | 3 |
| linkrag_iter | complementarity | 68.0% ± 2.3% | 18.8% ± 0.7% | 2.96 | 0.619 | 3 |


### Answer correctness by type and kind (test)

| group | n | mode | rerank | correct | runs |
|---|---:|---|---|---:|---:|
| single_modality | 17 | baseline | none | 76.5% ± 0.0% | 3 |
| single_modality | 17 | baseline | mmr | 41.2% ± 0.0% | 3 |
| single_modality | 17 | baseline | complementarity | 72.5% ± 3.4% | 3 |
| single_modality | 17 | iterative | none | 72.5% ± 3.4% | 3 |
| single_modality | 17 | iterative | mmr | 64.7% ± 5.9% | 3 |
| single_modality | 17 | iterative | complementarity | 74.5% ± 3.4% | 3 |
| single_modality | 17 | linkrag | none | 76.5% ± 0.0% | 3 |
| single_modality | 17 | linkrag | mmr | 72.5% ± 3.4% | 3 |
| single_modality | 17 | linkrag | complementarity | 76.5% ± 0.0% | 3 |
| single_modality | 17 | linkrag_iter | none | 72.5% ± 3.4% | 3 |
| single_modality | 17 | linkrag_iter | mmr | 74.5% ± 6.8% | 3 |
| single_modality | 17 | linkrag_iter | complementarity | 74.5% ± 9.0% | 3 |
| cross_modal | 21 | baseline | none | 38.1% ± 0.0% | 3 |
| cross_modal | 21 | baseline | mmr | 38.1% ± 0.0% | 3 |
| cross_modal | 21 | baseline | complementarity | 47.6% ± 0.0% | 3 |
| cross_modal | 21 | iterative | none | 46.0% ± 5.5% | 3 |
| cross_modal | 21 | iterative | mmr | 41.3% ± 9.9% | 3 |
| cross_modal | 21 | iterative | complementarity | 50.8% ± 2.7% | 3 |
| cross_modal | 21 | linkrag | none | 38.1% ± 0.0% | 3 |
| cross_modal | 21 | linkrag | mmr | 38.1% ± 0.0% | 3 |
| cross_modal | 21 | linkrag | complementarity | 38.1% ± 0.0% | 3 |
| cross_modal | 21 | linkrag_iter | none | 44.4% ± 2.7% | 3 |
| cross_modal | 21 | linkrag_iter | mmr | 47.6% ± 0.0% | 3 |
| cross_modal | 21 | linkrag_iter | complementarity | 44.4% ± 2.7% | 3 |
| deictic | 1 | baseline | none | 100.0% ± 0.0% | 3 |
| deictic | 1 | baseline | mmr | 0.0% ± 0.0% | 3 |
| deictic | 1 | baseline | complementarity | 100.0% ± 0.0% | 3 |
| deictic | 1 | iterative | none | 100.0% ± 0.0% | 3 |
| deictic | 1 | iterative | mmr | 66.7% ± 57.7% | 3 |
| deictic | 1 | iterative | complementarity | 100.0% ± 0.0% | 3 |
| deictic | 1 | linkrag | none | 100.0% ± 0.0% | 3 |
| deictic | 1 | linkrag | mmr | 100.0% ± 0.0% | 3 |
| deictic | 1 | linkrag | complementarity | 100.0% ± 0.0% | 3 |
| deictic | 1 | linkrag_iter | none | 100.0% ± 0.0% | 3 |
| deictic | 1 | linkrag_iter | mmr | 100.0% ± 0.0% | 3 |
| deictic | 1 | linkrag_iter | complementarity | 100.0% ± 0.0% | 3 |
| kind: bridge | 20 | baseline | none | 40.0% ± 0.0% | 3 |
| kind: bridge | 20 | baseline | mmr | 35.0% ± 0.0% | 3 |
| kind: bridge | 20 | baseline | complementarity | 45.0% ± 0.0% | 3 |
| kind: bridge | 20 | iterative | none | 46.7% ± 2.9% | 3 |
| kind: bridge | 20 | iterative | mmr | 41.7% ± 7.6% | 3 |
| kind: bridge | 20 | iterative | complementarity | 51.7% ± 2.9% | 3 |
| kind: bridge | 20 | linkrag | none | 40.0% ± 0.0% | 3 |
| kind: bridge | 20 | linkrag | mmr | 40.0% ± 0.0% | 3 |
| kind: bridge | 20 | linkrag | complementarity | 40.0% ± 0.0% | 3 |
| kind: bridge | 20 | linkrag_iter | none | 45.0% ± 0.0% | 3 |
| kind: bridge | 20 | linkrag_iter | mmr | 48.3% ± 2.9% | 3 |
| kind: bridge | 20 | linkrag_iter | complementarity | 45.0% ± 0.0% | 3 |
| kind: conjunction | 1 | baseline | none | 0.0% ± 0.0% | 3 |
| kind: conjunction | 1 | baseline | mmr | 100.0% ± 0.0% | 3 |
| kind: conjunction | 1 | baseline | complementarity | 100.0% ± 0.0% | 3 |
| kind: conjunction | 1 | iterative | none | 33.3% ± 57.7% | 3 |
| kind: conjunction | 1 | iterative | mmr | 33.3% ± 57.7% | 3 |
| kind: conjunction | 1 | iterative | complementarity | 33.3% ± 57.7% | 3 |
| kind: conjunction | 1 | linkrag | none | 0.0% ± 0.0% | 3 |
| kind: conjunction | 1 | linkrag | mmr | 0.0% ± 0.0% | 3 |
| kind: conjunction | 1 | linkrag | complementarity | 0.0% ± 0.0% | 3 |
| kind: conjunction | 1 | linkrag_iter | none | 33.3% ± 57.7% | 3 |
| kind: conjunction | 1 | linkrag_iter | mmr | 33.3% ± 57.7% | 3 |
| kind: conjunction | 1 | linkrag_iter | complementarity | 33.3% ± 57.7% | 3 |
| kind: control | 12 | baseline | none | 83.3% ± 0.0% | 3 |
| kind: control | 12 | baseline | mmr | 50.0% ± 0.0% | 3 |
| kind: control | 12 | baseline | complementarity | 83.3% ± 0.0% | 3 |
| kind: control | 12 | iterative | none | 77.8% ± 4.8% | 3 |
| kind: control | 12 | iterative | mmr | 72.2% ± 4.8% | 3 |
| kind: control | 12 | iterative | complementarity | 80.6% ± 4.8% | 3 |
| kind: control | 12 | linkrag | none | 83.3% ± 0.0% | 3 |
| kind: control | 12 | linkrag | mmr | 83.3% ± 0.0% | 3 |
| kind: control | 12 | linkrag | complementarity | 83.3% ± 0.0% | 3 |
| kind: control | 12 | linkrag_iter | none | 77.8% ± 4.8% | 3 |
| kind: control | 12 | linkrag_iter | mmr | 80.6% ± 9.6% | 3 |
| kind: control | 12 | linkrag_iter | complementarity | 80.6% ± 12.7% | 3 |
| kind: deictic | 5 | baseline | none | 60.0% ± 0.0% | 3 |
| kind: deictic | 5 | baseline | mmr | 20.0% ± 0.0% | 3 |
| kind: deictic | 5 | baseline | complementarity | 46.7% ± 11.5% | 3 |
| kind: deictic | 5 | iterative | none | 60.0% ± 0.0% | 3 |
| kind: deictic | 5 | iterative | mmr | 46.7% ± 11.5% | 3 |
| kind: deictic | 5 | iterative | complementarity | 60.0% ± 0.0% | 3 |
| kind: deictic | 5 | linkrag | none | 60.0% ± 0.0% | 3 |
| kind: deictic | 5 | linkrag | mmr | 46.7% ± 11.5% | 3 |
| kind: deictic | 5 | linkrag | complementarity | 60.0% ± 0.0% | 3 |
| kind: deictic | 5 | linkrag_iter | none | 60.0% ± 0.0% | 3 |
| kind: deictic | 5 | linkrag_iter | mmr | 60.0% ± 0.0% | 3 |
| kind: deictic | 5 | linkrag_iter | complementarity | 60.0% ± 0.0% | 3 |
| all | 38 | baseline | none | 55.3% ± 0.0% | 3 |
| all | 38 | baseline | mmr | 39.5% ± 0.0% | 3 |
| all | 38 | baseline | complementarity | 58.8% ± 1.5% | 3 |
| all | 38 | iterative | none | 57.9% ± 4.6% | 3 |
| all | 38 | iterative | mmr | 51.8% ± 4.0% | 3 |
| all | 38 | iterative | complementarity | 61.4% ± 1.5% | 3 |
| all | 38 | linkrag | none | 55.3% ± 0.0% | 3 |
| all | 38 | linkrag | mmr | 53.5% ± 1.5% | 3 |
| all | 38 | linkrag | complementarity | 55.3% ± 0.0% | 3 |
| all | 38 | linkrag_iter | none | 57.0% ± 1.5% | 3 |
| all | 38 | linkrag_iter | mmr | 59.6% ± 3.0% | 3 |
| all | 38 | linkrag_iter | complementarity | 57.9% ± 2.6% | 3 |

## Corpus

One pooled index: every file of every topic, so retrieval has to find the right file. Units per file
from the manifest. Intake gate: deck→transcript redundancy (`scripts/dataset/README.md`).

| topic | half | lecture (MaViLS) + separate source | talk | deck | notes | redundancy (deck→talk) | gate |
|---|---|---|---:|---:|---:|---|---|
| nfa_closure | tune | MIT 18.404 L2 + Erickson notes | 142 | 12 | 56 | 32.0 % | KEEP |
| psych_methods | tune | MIT 9.00 L2 + OpenStax Psych ch. 2 | 154 | 77 | 102 | census 57.1 % | KEEP |
| visual_system | tune | MIT 9.04 + OpenStax Psych ch. 5 | 189 | 42 | 76 | 40.0 % | KEEP |
| gp_scaling | test | Numerics of ML L03 + GPML ch. 8 | 191 | 84 | 50 | 30.0 % | KEEP |
| mimic_data | test | MIT ML-for-health + MIMIC-III paper | 188 | 129 | 30 | 16.0 % | KEEP |
| rl_mdp | test | Silver RL L2 + Sutton & Barto ch. 3 | 222 | 80 | 81 | 32.0 % | KEEP |
| lightning | (row only) | MIT MAS.S62 L14 + Lightning paper | 132 | 64 | 109 | census 67.3 % | REJECT |
| **total** | | 21 files | 1,218 | 488 | 504 | | |

## Questions

98 drafted by Claude Opus subagents, 8 rejected in review, 90 checked by `verify_gold` (3 runs):
86 kept, 4 dropped (RL03, RL09, NF01, LN12), 6 relabelled to single-source. Counts by verified type and
question kind (they sum to 86):

| type | bridge | conjunction | deictic | control | total | tune | test | lightning |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cross_modal_split | 40 | 4 | 0 | 0 | **44** | 17 | 20 | 7 |
| cross_modal_deictic | 0 | 0 | 9 | 0 | **9** | 6 | 1 | 2 |
| audio_only | 1 | 0 | 1 | 10 | **12** | 3 | 8 | 1 |
| slides_only | 1 | 0 | 3 | 7 | **11** | 4 | 6 | 1 |
| paper_only | 0 | 0 | 0 | 10 | **10** | 6 | 3 | 1 |
| **total** | 42 | 4 | 13 | 27 | **86** | 36 | 38 | 12 |

Lightning was rejected by the intake gate; its 12 questions stay in the gold file and are not in any
table above. Its separate row was not run (deadline).

## What was not done

- No knob tuning; no change to the audio↔slide gate (known issue (n)).
- Lightning row not run. `cut_pages` reproducibility and the 3 informational redundancy directions are
  left for later (HANDOFF.md).
