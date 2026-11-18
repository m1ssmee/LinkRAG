# XFile retrieval-only comparison (EXPLORATORY, 2026-11-18)

**Not a result yet.** It covers all 7 topics, including the 3 tune topics, with default knobs and no
tuning. The gold is the 1-run gold check on **drafts nobody has reviewed yet** (97 questions). It is
retrieval only (recall@8 against gold units); no answers are scored. Per DESIGN.md issue (n), the
link file carries 699 audio_slide links between unrelated lectures. `baseline/none` and
`linkrag/none` are identical by construction: `expansion: additive` returns the baseline top-k when
nothing reranks. The no-links control is `baseline/complementarity`, which since 2026-11-17 runs
without the graph.

**Reading.** Against the no-links control, links add +9.7 recall@8 on the 64 cross-file questions
(95% CI by topic [+3.4, +15.2]). Against plain top-8, no contrast shows a measurable difference.
Links mainly recover what the complementarity reranker loses, while raising modality coverage
(2.82 vs 2.02 distinct modalities per set). Whether that helps answers is the correctness run's
question.

## XFile retrieval-only, EXPLORATORY: all 7 topics, default knobs, gold = 1-run gold check on unreviewed drafts

97 questions · k=8 · pool=20 · corpus `a086ca1c22dc1d93` (2210 units) · gold stamped `a086ca1c22dc1d93` · model `gpt-5.4-mini-2026-03-17` temperature=0.0 seed=20260910 · repeats 3 for LLM modes · 0 LLM calls · expansion `additive` · normalise_seeds `True`

| mode | rerank | recall@8 | precision@8 | distinct modalities | redundancy | runs |
|---|---|---:|---:|---:|---:|---:|
| baseline | none | 69.6% | 19.2% | 2.02 | 0.661 | 1 |
| baseline | mmr | 47.7% | 12.5% | 2.12 | 0.573 | 1 |
| baseline | complementarity | 58.0% | 15.6% | 2.24 | 0.566 | 1 |
| linkrag | none | 69.6% | 19.2% | 2.02 | 0.661 | 1 |
| linkrag | mmr | 70.3% | 19.2% | 2.05 | 0.649 | 1 |
| linkrag | complementarity | 68.1% | 18.6% | 2.82 | 0.618 | 1 |

Repeat determinism (complementarity cell):

- `baseline`: deterministic by construction (no LLM, single run)
- `linkrag`: deterministic by construction (no LLM, single run)

### By verified question type

| type | n | mode | rerank | recall@k | precision@k | modalities | LLM calls / q |
|---|---:|---|---|---:|---:|---:|---:|
| single_modality | 33 | baseline | none | 81.8% | 17.0% | 1.82 | 0.0 |
| single_modality | 33 | baseline | mmr | 56.1% | 11.0% | 2.03 | 0.0 |
| single_modality | 33 | baseline | complementarity | 70.2% | 13.6% | 2.15 | 0.0 |
| single_modality | 33 | linkrag | none | 81.8% | 17.0% | 1.82 | 0.0 |
| single_modality | 33 | linkrag | mmr | 83.8% | 17.0% | 1.82 | 0.0 |
| single_modality | 33 | linkrag | complementarity | 80.8% | 16.7% | 2.79 | 0.0 |
| cross_modal | 64 | baseline | none | 63.3% | 20.3% | 2.12 | 0.0 |
| cross_modal | 64 | baseline | mmr | 43.4% | 13.3% | 2.17 | 0.0 |
| cross_modal | 64 | baseline | complementarity | 51.7% | 16.6% | 2.28 | 0.0 |
| cross_modal | 64 | linkrag | none | 63.3% | 20.3% | 2.12 | 0.0 |
| cross_modal | 64 | linkrag | mmr | 63.3% | 20.3% | 2.17 | 0.0 |
| cross_modal | 64 | linkrag | complementarity | 61.5% | 19.5% | 2.84 | 0.0 |
| deictic | 10 | baseline | none | 53.5% | 21.2% | 2.20 | 0.0 |
| deictic | 10 | baseline | mmr | 27.8% | 10.0% | 2.30 | 0.0 |
| deictic | 10 | baseline | complementarity | 44.8% | 17.5% | 2.60 | 0.0 |
| deictic | 10 | linkrag | none | 53.5% | 21.2% | 2.20 | 0.0 |
| deictic | 10 | linkrag | mmr | 50.2% | 20.0% | 2.30 | 0.0 |
| deictic | 10 | linkrag | complementarity | 51.5% | 20.0% | 2.80 | 0.0 |

Bucket sizes: single_modality 33, cross_modal 64, deictic 10 (deictic ⊂ cross_modal; single + cross = 97 questions).

Per-question recall@k (mean over runs):

| Q | type | baseline/none | baseline/mmr | baseline/complementarity | linkrag/none | linkrag/mmr | linkrag/complementarity |
|---|---|---:|---:|---:|---:|---:|---:|
| RL01 | cross_modal_split | 67% | 67% | 67% | 67% | 67% | 33% |
| RL02 | cross_modal_split | 0% | 0% | 0% | 0% | 0% | 0% |
| RL03 | cross_modal_split | 100% | 67% | 100% | 100% | 100% | 100% |
| RL04 | cross_modal_split | 100% | 100% | 100% | 100% | 100% | 100% |
| RL05 | cross_modal_split | 50% | 50% | 50% | 50% | 50% | 50% |
| RL06 | cross_modal_split | 100% | 100% | 100% | 100% | 100% | 100% |
| RL07 | cross_modal_split | 33% | 0% | 0% | 33% | 33% | 33% |
| RL08 | cross_modal_split | 50% | 100% | 100% | 50% | 50% | 50% |
| RL09 | cross_modal_deictic | 33% | 33% | 33% | 33% | 33% | 33% |
| RL10 | slides_only | 33% | 0% | 0% | 33% | 33% | 33% |
| RL11 | audio_only | 100% | 100% | 100% | 100% | 100% | 100% |
| RL12 | audio_only | 100% | 0% | 100% | 100% | 100% | 100% |
| RL13 | slides_only | 100% | 100% | 100% | 100% | 100% | 100% |
| RL14 | paper_only | 100% | 50% | 100% | 100% | 100% | 100% |
| GP01 | cross_modal_split | 100% | 50% | 50% | 100% | 100% | 100% |
| GP02 | cross_modal_split | 100% | 50% | 50% | 100% | 100% | 100% |
| GP03 | cross_modal_split | 0% | 50% | 50% | 0% | 0% | 50% |
| GP04 | cross_modal_split | 33% | 0% | 0% | 33% | 33% | 33% |
| GP05 | cross_modal_split | 67% | 33% | 33% | 67% | 67% | 67% |
| GP06 | cross_modal_split | 80% | 40% | 60% | 80% | 80% | 80% |
| GP07 | cross_modal_split | 0% | 33% | 33% | 0% | 33% | 0% |
| GP08 | cross_modal_split | 25% | 25% | 50% | 25% | 25% | 25% |
| GP09 | audio_only | 67% | 67% | 67% | 67% | 67% | 33% |
| GP10 | slides_only | 100% | 50% | 100% | 100% | 100% | 100% |
| GP11 | audio_only | 0% | 0% | 0% | 0% | 0% | 0% |
| GP12 | audio_only | 100% | 0% | 100% | 100% | 100% | 100% |
| GP13 | slides_only | 100% | 100% | 100% | 100% | 100% | 100% |
| GP14 | paper_only | 50% | 0% | 50% | 50% | 50% | 50% |
| NF01 | cross_modal_split | 25% | 25% | 25% | 25% | 25% | 25% |
| NF02 | cross_modal_split | 67% | 0% | 33% | 67% | 67% | 67% |
| NF03 | cross_modal_split | 0% | 67% | 67% | 0% | 0% | 0% |
| NF04 | cross_modal_split | 100% | 100% | 100% | 100% | 100% | 100% |
| NF05 | cross_modal_split | 50% | 50% | 50% | 50% | 50% | 50% |
| NF06 | cross_modal_split | 100% | 50% | 50% | 100% | 100% | 100% |
| NF07 | cross_modal_split | 67% | 33% | 33% | 67% | 67% | 67% |
| NF08 | cross_modal_split | 100% | 50% | 100% | 100% | 100% | 100% |
| NF09 | cross_modal_deictic | 33% | 0% | 33% | 33% | 33% | 33% |
| NF10 | cross_modal_deictic | 67% | 33% | 67% | 67% | 67% | 67% |
| NF11 | audio_only | 100% | 100% | 100% | 100% | 100% | 100% |
| NF12 | slides_only | 100% | 100% | 100% | 100% | 100% | 100% |
| NF13 | paper_only | 100% | 0% | 0% | 100% | 100% | 100% |
| NF14 | paper_only | 100% | 33% | 33% | 100% | 100% | 100% |
| MD01 | audio_only | 67% | 33% | 67% | 67% | 67% | 67% |
| MD02 | cross_modal_split | 33% | 33% | 33% | 33% | 33% | 33% |
| MD03 | cross_modal_split | 50% | 50% | 50% | 50% | 50% | 50% |
| MD04 | cross_modal_split | 33% | 33% | 67% | 33% | 33% | 33% |
| MD05 | cross_modal_split | 67% | 33% | 33% | 67% | 67% | 67% |
| MD06 | cross_modal_split | 50% | 50% | 0% | 50% | 50% | 50% |
| MD07 | cross_modal_split | 100% | 100% | 100% | 100% | 100% | 100% |
| MD08 | cross_modal_split | 100% | 100% | 100% | 100% | 100% | 100% |
| MD09 | slides_only | 67% | 33% | 67% | 67% | 33% | 67% |
| MD10 | cross_modal_deictic | 75% | 25% | 75% | 75% | 75% | 75% |
| MD11 | audio_only | 50% | 50% | 50% | 50% | 50% | 50% |
| MD12 | audio_only | 100% | 100% | 100% | 100% | 100% | 100% |
| MD13 | slides_only | 100% | 100% | 100% | 100% | 100% | 100% |
| MD14 | paper_only | 100% | 100% | 100% | 100% | 100% | 100% |
| PS01 | slides_only | 67% | 33% | 33% | 67% | 67% | 67% |
| PS02 | cross_modal_split | 67% | 33% | 67% | 67% | 67% | 67% |
| PS03 | cross_modal_split | 100% | 50% | 100% | 100% | 100% | 100% |
| PS04 | cross_modal_split | 100% | 50% | 100% | 100% | 100% | 100% |
| PS05 | cross_modal_split | 100% | 50% | 50% | 100% | 100% | 100% |
| PS06 | cross_modal_split | 67% | 0% | 33% | 67% | 67% | 33% |
| PS07 | cross_modal_split | 25% | 25% | 25% | 25% | 25% | 25% |
| PS08 | cross_modal_split | 67% | 33% | 33% | 67% | 67% | 33% |
| PS09 | cross_modal_deictic | 67% | 33% | 67% | 67% | 67% | 100% |
| PS10 | cross_modal_deictic | 60% | 20% | 40% | 60% | 60% | 40% |
| PS11 | audio_only | 100% | 50% | 50% | 100% | 100% | 100% |
| PS12 | slides_only | 100% | 100% | 100% | 100% | 100% | 100% |
| PS13 | paper_only | 100% | 50% | 50% | 100% | 100% | 100% |
| PS14 | paper_only | 100% | 50% | 50% | 100% | 100% | 100% |
| VS01 | cross_modal_split | 100% | 50% | 50% | 100% | 100% | 100% |
| VS02 | cross_modal_split | 100% | 100% | 100% | 100% | 100% | 100% |
| VS03 | cross_modal_split | 25% | 25% | 25% | 25% | 25% | 50% |
| VS04 | cross_modal_split | 50% | 50% | 50% | 50% | 50% | 50% |
| VS05 | cross_modal_split | 100% | 50% | 50% | 100% | 100% | 100% |
| VS06 | cross_modal_split | 60% | 20% | 20% | 60% | 60% | 40% |
| VS07 | cross_modal_split | 100% | 33% | 67% | 100% | 100% | 100% |
| VS08 | cross_modal_split | 67% | 33% | 67% | 67% | 67% | 67% |
| VS09 | cross_modal_deictic | 0% | 0% | 0% | 0% | 0% | 0% |
| VS10 | cross_modal_deictic | 100% | 100% | 100% | 100% | 100% | 100% |
| VS11 | audio_only | 100% | 100% | 50% | 100% | 100% | 100% |
| VS12 | slides_only | 100% | 0% | 100% | 100% | 100% | 100% |
| VS13 | paper_only | 100% | 100% | 100% | 100% | 100% | 100% |
| VS14 | paper_only | 100% | 50% | 50% | 100% | 100% | 100% |
| LN01 | cross_modal_split | 50% | 50% | 50% | 50% | 50% | 50% |
| LN02 | cross_modal_split | 100% | 50% | 50% | 100% | 100% | 100% |
| LN03 | cross_modal_split | 100% | 50% | 50% | 100% | 100% | 50% |
| LN04 | cross_modal_split | 100% | 100% | 100% | 100% | 100% | 100% |
| LN05 | cross_modal_split | 67% | 33% | 67% | 67% | 67% | 67% |
| LN06 | cross_modal_split | 50% | 0% | 0% | 50% | 50% | 50% |
| LN07 | cross_modal_split | 25% | 25% | 25% | 25% | 25% | 25% |
| LN08 | cross_modal_split | 50% | 50% | 0% | 50% | 50% | 50% |
| LN09 | cross_modal_deictic | 33% | 33% | 33% | 33% | 33% | 33% |
| LN10 | cross_modal_deictic | 67% | 0% | 0% | 67% | 33% | 33% |
| LN11 | audio_only | 0% | 0% | 0% | 0% | 0% | 0% |
| LN13 | slides_only | 0% | 100% | 100% | 0% | 100% | 0% |
| LN14 | paper_only | 100% | 100% | 100% | 100% | 100% | 100% |

### Paired differences, 95% CI resampling whole topics (5,000 draws, seed 20261118)

| contrast | questions | Δ recall@8 | 95% CI |
|---|---:|---:|---|
| links vs no-links control (complementarity), cross-file | 64 | +9.7% | [+3.4%, +15.2%] |
| links vs no-links control (complementarity), single-source | 33 | +10.6% | [-4.9%, +26.8%] (no measurable difference) |
| links vs no-links control (complementarity), bridge | 44 | +10.4% | [+4.6%, +14.9%] |
| links + complementarity vs plain top-8, cross-file | 64 | -1.8% | [-5.1%, +1.6%] (no measurable difference) |
| links + complementarity vs plain top-8, single-source | 33 | -1.0% | [-2.9%, +0.0%] (no measurable difference) |
| links + complementarity vs plain top-8, bridge | 44 | +0.5% | [-2.2%, +3.4%] (no measurable difference) |
| links + MMR vs plain top-8, cross-file | 64 | -0.0% | [-1.5%, +1.6%] (no measurable difference) |
| links + MMR vs plain top-8, single-source | 33 | +2.0% | [-2.7%, +10.3%] (no measurable difference) |
| links + MMR vs plain top-8, bridge | 44 | +0.8% | [+0.0%, +2.3%] (no measurable difference) |
