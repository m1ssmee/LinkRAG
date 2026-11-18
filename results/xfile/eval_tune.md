# XFile evaluation, tune half (check only)

Tune topics: nfa_closure, psych_methods, visual_system (36 questions: 23 cross-file, 13 single-source).
Default knobs (`configs/xfile.yaml`), frozen before any tune answer score existed (PLAN.md, bf53fd7);
**nothing here changed any setting.** Corpus `a086ca1c22dc1d93`. Answerer `claude-haiku-5-5`, judge
`claude-sonnet-5-5`, both Claude Code subagents through the reply cache (one answer prompt per subagent,
read coverage audited, DESIGN.md (o)). 3 runs per cell; each answer graded 3 times (majority).
Cache-only replay (`--answerer-cache-only --judge-cache-only`).

## What the check shows

With the default weights the complementarity reranker roughly halves the no-links control:
baseline + complementarity answers 33.3 % of tune questions correctly against 63.9 % for plain top-8
(baseline + none), and 59.0 % against 100 % on single-source questions. This is known issue (j):
retrieval scores near 0.03 are swamped by the coverage and redundancy terms. linkrag + complementarity
(65.7 %) escapes it because the link bonus mostly picks linked units that are relevant. So the
pre-registered headline (linkrag + complementarity vs baseline + complementarity) compares links against a
broken control; any gap it shows on test must be read against plain top-8 as well. linkrag + none equals
baseline + none exactly: without the reranker no linked unit reaches the top 8.

After this run, a retrieval-only grid (no LLM calls) scaled alpha/beta/gamma together by c in
{1, 0.3, 0.1, 0.03, 0.01}: baseline + complementarity recall@8 59.0 / 61.3 / 68.0 / 78.4 / 78.7 %,
linkrag + complementarity 77.5 / 79.4 / 79.9 / 79.0 / 79.0 % (plain top-8: 79.0 %). **Not used**: the
knobs stay frozen at the defaults; this is recorded only as context for the caveat above.

## Tables
36 questions · k=8 · pool=20 · corpus `a086ca1c22dc1d93` (2210 units) · gold stamped `a086ca1c22dc1d93` · model `claude-haiku-5-5` temperature=0.0 seed=20260910 · repeats 3 for LLM modes · 216 LLM calls · expansion `additive` · normalise_seeds `True`

| mode | rerank | recall@8 | precision@8 | distinct modalities | redundancy | runs |
|---|---|---:|---:|---:|---:|---:|
| baseline | none | 79.0% ± 0.0% | 21.9% ± 0.0% | 1.97 | 0.643 | 3 |
| baseline | mmr | 45.5% ± 0.0% | 11.8% ± 0.0% | 2.14 | 0.548 | 3 |
| baseline | complementarity | 59.0% ± 0.0% | 16.0% ± 0.0% | 2.25 | 0.542 | 3 |
| iterative | none | 83.7% ± 1.0% | 23.8% ± 0.4% | 2.17 | 0.645 | 3 |
| iterative | mmr | 67.9% ± 2.1% | 19.2% ± 0.2% | 2.19 | 0.595 | 3 |
| iterative | complementarity | 68.8% ± 1.1% | 19.9% ± 0.5% | 2.31 | 0.589 | 3 |
| linkrag | none | 79.0% ± 0.0% | 21.9% ± 0.0% | 1.97 | 0.643 | 3 |
| linkrag | mmr | 79.0% ± 0.0% | 21.9% ± 0.0% | 2.03 | 0.630 | 3 |
| linkrag | complementarity | 77.5% ± 0.0% | 21.2% ± 0.0% | 2.69 | 0.616 | 3 |
| linkrag_iter | none | 84.0% ± 1.4% | 23.8% ± 0.5% | 2.17 | 0.641 | 3 |
| linkrag_iter | mmr | 84.0% ± 0.9% | 24.0% ± 0.3% | 2.21 | 0.625 | 3 |
| linkrag_iter | complementarity | 80.1% ± 1.5% | 22.5% ± 0.5% | 2.75 | 0.607 | 3 |

Repeat determinism (complementarity cell):

- `baseline`: IDENTICAL across runs
- `iterative`: NOT identical — 3 distinct outcomes in 3 runs
- `linkrag`: IDENTICAL across runs
- `linkrag_iter`: NOT identical — 3 distinct outcomes in 3 runs

### By verified question type

| type | n | mode | rerank | recall@k | precision@k | modalities | LLM calls / q |
|---|---:|---|---|---:|---:|---:|---:|
| single_modality | 13 | baseline | none | 97.4% ± 0.0% | 20.2% ± 0.0% | 1.62 | 0.0 |
| single_modality | 13 | baseline | mmr | 59.0% ± 0.0% | 11.5% ± 0.0% | 1.85 | 0.0 |
| single_modality | 13 | baseline | complementarity | 62.8% ± 0.0% | 11.5% ± 0.0% | 1.92 | 0.0 |
| single_modality | 13 | iterative | none | 97.4% ± 0.0% | 20.2% ± 0.0% | 1.77 | 1.0 |
| single_modality | 13 | iterative | mmr | 79.1% ± 8.5% | 15.7% ± 1.1% | 2.03 | 1.0 |
| single_modality | 13 | iterative | complementarity | 70.9% ± 3.0% | 14.1% ± 1.1% | 2.13 | 1.0 |
| single_modality | 13 | linkrag | none | 97.4% ± 0.0% | 20.2% ± 0.0% | 1.62 | 0.0 |
| single_modality | 13 | linkrag | mmr | 97.4% ± 0.0% | 20.2% ± 0.0% | 1.62 | 0.0 |
| single_modality | 13 | linkrag | complementarity | 97.4% ± 0.0% | 20.2% ± 0.0% | 2.62 | 0.0 |
| single_modality | 13 | linkrag_iter | none | 97.4% ± 0.0% | 20.2% ± 0.0% | 1.82 | 1.0 |
| single_modality | 13 | linkrag_iter | mmr | 97.4% ± 0.0% | 20.2% ± 0.0% | 1.97 | 1.0 |
| single_modality | 13 | linkrag_iter | complementarity | 95.3% ± 2.0% | 19.6% ± 0.6% | 2.74 | 1.0 |
| cross_modal | 23 | baseline | none | 68.6% ± 0.0% | 22.8% ± 0.0% | 2.17 | 0.0 |
| cross_modal | 23 | baseline | mmr | 37.8% ± 0.0% | 12.0% ± 0.0% | 2.30 | 0.0 |
| cross_modal | 23 | baseline | complementarity | 56.8% ± 0.0% | 18.5% ± 0.0% | 2.43 | 0.0 |
| cross_modal | 23 | iterative | none | 75.9% ± 1.6% | 25.9% ± 0.6% | 2.39 | 1.0 |
| cross_modal | 23 | iterative | mmr | 61.5% ± 1.6% | 21.2% ± 0.5% | 2.29 | 1.0 |
| cross_modal | 23 | iterative | complementarity | 67.5% ± 0.7% | 23.2% ± 0.3% | 2.42 | 1.0 |
| cross_modal | 23 | linkrag | none | 68.6% ± 0.0% | 22.8% ± 0.0% | 2.17 | 0.0 |
| cross_modal | 23 | linkrag | mmr | 68.6% ± 0.0% | 22.8% ± 0.0% | 2.26 | 0.0 |
| cross_modal | 23 | linkrag | complementarity | 66.2% ± 0.0% | 21.7% ± 0.0% | 2.74 | 0.0 |
| cross_modal | 23 | linkrag_iter | none | 76.3% ± 2.2% | 25.9% ± 0.8% | 2.36 | 1.0 |
| cross_modal | 23 | linkrag_iter | mmr | 76.4% ± 1.5% | 26.1% ± 0.5% | 2.35 | 1.0 |
| cross_modal | 23 | linkrag_iter | complementarity | 71.5% ± 1.8% | 24.1% ± 0.6% | 2.75 | 1.0 |
| deictic | 6 | baseline | none | 54.4% ± 0.0% | 20.8% ± 0.0% | 2.17 | 0.0 |
| deictic | 6 | baseline | mmr | 31.1% ± 0.0% | 10.4% ± 0.0% | 2.50 | 0.0 |
| deictic | 6 | baseline | complementarity | 51.1% ± 0.0% | 18.8% ± 0.0% | 2.67 | 0.0 |
| deictic | 6 | iterative | none | 67.0% ± 3.2% | 26.4% ± 1.2% | 2.50 | 1.0 |
| deictic | 6 | iterative | mmr | 58.5% ± 1.3% | 22.9% ± 0.0% | 2.22 | 1.0 |
| deictic | 6 | iterative | complementarity | 71.9% ± 5.1% | 28.5% ± 2.4% | 2.56 | 1.0 |
| deictic | 6 | linkrag | none | 54.4% ± 0.0% | 20.8% ± 0.0% | 2.17 | 0.0 |
| deictic | 6 | linkrag | mmr | 54.4% ± 0.0% | 20.8% ± 0.0% | 2.33 | 0.0 |
| deictic | 6 | linkrag | complementarity | 56.7% ± 0.0% | 20.8% ± 0.0% | 2.67 | 0.0 |
| deictic | 6 | linkrag_iter | none | 68.5% ± 2.8% | 26.4% ± 1.2% | 2.50 | 1.0 |
| deictic | 6 | linkrag_iter | mmr | 70.7% ± 3.2% | 27.8% ± 1.2% | 2.50 | 1.0 |
| deictic | 6 | linkrag_iter | complementarity | 68.9% ± 5.6% | 27.1% ± 2.1% | 2.72 | 1.0 |

Bucket sizes: single_modality 13, cross_modal 23, deictic 6 (deictic ⊂ cross_modal; single + cross = 36 questions).

### Answer correctness (judge PASS vs expected answer)

| group | n | mode | rerank | correct | runs |
|---|---:|---|---|---:|---:|
| single_modality | 13 | baseline | none | 100.0% ± 0.0% | 3 |
| single_modality | 13 | baseline | mmr | 61.5% ± 0.0% | 3 |
| single_modality | 13 | baseline | complementarity | 59.0% ± 4.4% | 3 |
| single_modality | 13 | iterative | none | 92.3% ± 7.7% | 3 |
| single_modality | 13 | iterative | mmr | 69.2% ± 7.7% | 3 |
| single_modality | 13 | iterative | complementarity | 71.8% ± 8.9% | 3 |
| single_modality | 13 | linkrag | none | 100.0% ± 0.0% | 3 |
| single_modality | 13 | linkrag | mmr | 97.4% ± 4.4% | 3 |
| single_modality | 13 | linkrag | complementarity | 97.4% ± 4.4% | 3 |
| single_modality | 13 | linkrag_iter | none | 97.4% ± 4.4% | 3 |
| single_modality | 13 | linkrag_iter | mmr | 94.9% ± 4.4% | 3 |
| single_modality | 13 | linkrag_iter | complementarity | 87.2% ± 4.4% | 3 |
| cross_modal | 23 | baseline | none | 43.5% ± 0.0% | 3 |
| cross_modal | 23 | baseline | mmr | 8.7% ± 0.0% | 3 |
| cross_modal | 23 | baseline | complementarity | 18.8% ± 2.5% | 3 |
| cross_modal | 23 | iterative | none | 59.4% ± 2.5% | 3 |
| cross_modal | 23 | iterative | mmr | 34.8% ± 4.3% | 3 |
| cross_modal | 23 | iterative | complementarity | 44.9% ± 6.6% | 3 |
| cross_modal | 23 | linkrag | none | 43.5% ± 0.0% | 3 |
| cross_modal | 23 | linkrag | mmr | 39.1% ± 4.3% | 3 |
| cross_modal | 23 | linkrag | complementarity | 47.8% ± 4.3% | 3 |
| cross_modal | 23 | linkrag_iter | none | 63.8% ± 5.0% | 3 |
| cross_modal | 23 | linkrag_iter | mmr | 60.9% ± 4.3% | 3 |
| cross_modal | 23 | linkrag_iter | complementarity | 55.1% ± 2.5% | 3 |
| deictic | 6 | baseline | none | 33.3% ± 0.0% | 3 |
| deictic | 6 | baseline | mmr | 16.7% ± 0.0% | 3 |
| deictic | 6 | baseline | complementarity | 16.7% ± 0.0% | 3 |
| deictic | 6 | iterative | none | 72.2% ± 9.6% | 3 |
| deictic | 6 | iterative | mmr | 44.4% ± 9.6% | 3 |
| deictic | 6 | iterative | complementarity | 55.6% ± 9.6% | 3 |
| deictic | 6 | linkrag | none | 33.3% ± 0.0% | 3 |
| deictic | 6 | linkrag | mmr | 27.8% ± 9.6% | 3 |
| deictic | 6 | linkrag | complementarity | 50.0% ± 0.0% | 3 |
| deictic | 6 | linkrag_iter | none | 77.8% ± 9.6% | 3 |
| deictic | 6 | linkrag_iter | mmr | 72.2% ± 19.2% | 3 |
| deictic | 6 | linkrag_iter | complementarity | 72.2% ± 9.6% | 3 |
| kind: bridge | 16 | baseline | none | 56.2% ± 0.0% | 3 |
| kind: bridge | 16 | baseline | mmr | 8.3% ± 3.6% | 3 |
| kind: bridge | 16 | baseline | complementarity | 20.8% ± 3.6% | 3 |
| kind: bridge | 16 | iterative | none | 54.2% ± 3.6% | 3 |
| kind: bridge | 16 | iterative | mmr | 33.3% ± 9.5% | 3 |
| kind: bridge | 16 | iterative | complementarity | 50.0% ± 10.8% | 3 |
| kind: bridge | 16 | linkrag | none | 56.2% ± 0.0% | 3 |
| kind: bridge | 16 | linkrag | mmr | 52.1% ± 3.6% | 3 |
| kind: bridge | 16 | linkrag | complementarity | 54.2% ± 7.2% | 3 |
| kind: bridge | 16 | linkrag_iter | none | 62.5% ± 6.2% | 3 |
| kind: bridge | 16 | linkrag_iter | mmr | 60.4% ± 3.6% | 3 |
| kind: bridge | 16 | linkrag_iter | complementarity | 58.3% ± 3.6% | 3 |
| kind: conjunction | 2 | baseline | none | 0.0% ± 0.0% | 3 |
| kind: conjunction | 2 | baseline | mmr | 0.0% ± 0.0% | 3 |
| kind: conjunction | 2 | baseline | complementarity | 0.0% ± 0.0% | 3 |
| kind: conjunction | 2 | iterative | none | 83.3% ± 28.9% | 3 |
| kind: conjunction | 2 | iterative | mmr | 50.0% ± 50.0% | 3 |
| kind: conjunction | 2 | iterative | complementarity | 0.0% ± 0.0% | 3 |
| kind: conjunction | 2 | linkrag | none | 0.0% ± 0.0% | 3 |
| kind: conjunction | 2 | linkrag | mmr | 0.0% ± 0.0% | 3 |
| kind: conjunction | 2 | linkrag | complementarity | 0.0% ± 0.0% | 3 |
| kind: conjunction | 2 | linkrag_iter | none | 50.0% ± 50.0% | 3 |
| kind: conjunction | 2 | linkrag_iter | mmr | 50.0% ± 50.0% | 3 |
| kind: conjunction | 2 | linkrag_iter | complementarity | 0.0% ± 0.0% | 3 |
| kind: control | 12 | baseline | none | 100.0% ± 0.0% | 3 |
| kind: control | 12 | baseline | mmr | 63.9% ± 4.8% | 3 |
| kind: control | 12 | baseline | complementarity | 63.9% ± 4.8% | 3 |
| kind: control | 12 | iterative | none | 91.7% ± 8.3% | 3 |
| kind: control | 12 | iterative | mmr | 66.7% ± 8.3% | 3 |
| kind: control | 12 | iterative | complementarity | 69.4% ± 9.6% | 3 |
| kind: control | 12 | linkrag | none | 100.0% ± 0.0% | 3 |
| kind: control | 12 | linkrag | mmr | 97.2% ± 4.8% | 3 |
| kind: control | 12 | linkrag | complementarity | 100.0% ± 0.0% | 3 |
| kind: control | 12 | linkrag_iter | none | 97.2% ± 4.8% | 3 |
| kind: control | 12 | linkrag_iter | mmr | 94.4% ± 4.8% | 3 |
| kind: control | 12 | linkrag_iter | complementarity | 86.1% ± 4.8% | 3 |
| kind: deictic | 6 | baseline | none | 33.3% ± 0.0% | 3 |
| kind: deictic | 6 | baseline | mmr | 16.7% ± 0.0% | 3 |
| kind: deictic | 6 | baseline | complementarity | 16.7% ± 0.0% | 3 |
| kind: deictic | 6 | iterative | none | 72.2% ± 9.6% | 3 |
| kind: deictic | 6 | iterative | mmr | 44.4% ± 9.6% | 3 |
| kind: deictic | 6 | iterative | complementarity | 55.6% ± 9.6% | 3 |
| kind: deictic | 6 | linkrag | none | 33.3% ± 0.0% | 3 |
| kind: deictic | 6 | linkrag | mmr | 27.8% ± 9.6% | 3 |
| kind: deictic | 6 | linkrag | complementarity | 50.0% ± 0.0% | 3 |
| kind: deictic | 6 | linkrag_iter | none | 77.8% ± 9.6% | 3 |
| kind: deictic | 6 | linkrag_iter | mmr | 72.2% ± 19.2% | 3 |
| kind: deictic | 6 | linkrag_iter | complementarity | 72.2% ± 9.6% | 3 |
| all | 36 | baseline | none | 63.9% ± 0.0% | 3 |
| all | 36 | baseline | mmr | 27.8% ± 0.0% | 3 |
| all | 36 | baseline | complementarity | 33.3% ± 2.8% | 3 |
| all | 36 | iterative | none | 71.3% ± 4.2% | 3 |
| all | 36 | iterative | mmr | 47.2% ± 4.8% | 3 |
| all | 36 | iterative | complementarity | 54.6% ± 7.0% | 3 |
| all | 36 | linkrag | none | 63.9% ± 0.0% | 3 |
| all | 36 | linkrag | mmr | 60.2% ± 3.2% | 3 |
| all | 36 | linkrag | complementarity | 65.7% ± 3.2% | 3 |
| all | 36 | linkrag_iter | none | 75.9% ± 4.2% | 3 |
| all | 36 | linkrag_iter | mmr | 73.1% ± 3.2% | 3 |
| all | 36 | linkrag_iter | complementarity | 66.7% ± 2.8% | 3 |

Per-question recall@k (mean over runs):

| Q | type | baseline/none | baseline/mmr | baseline/complementarity | iterative/none | iterative/mmr | iterative/complementarity | linkrag/none | linkrag/mmr | linkrag/complementarity | linkrag_iter/none | linkrag_iter/mmr | linkrag_iter/complementarity |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| NF02 | cross_modal_split | 67% | 0% | 33% | 89% | 56% | 33% | 67% | 67% | 67% | 78% | 78% | 67% |
| NF03 | cross_modal_split | 0% | 67% | 67% | 0% | 33% | 33% | 0% | 0% | 0% | 22% | 22% | 22% |
| NF07 | cross_modal_split | 67% | 33% | 33% | 56% | 67% | 67% | 67% | 67% | 67% | 56% | 56% | 67% |
| NF08 | cross_modal_split | 100% | 50% | 100% | 100% | 67% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| NF09 | cross_modal_deictic | 33% | 0% | 33% | 33% | 33% | 33% | 33% | 33% | 33% | 56% | 56% | 44% |
| NF10 | cross_modal_deictic | 67% | 33% | 67% | 33% | 56% | 67% | 67% | 67% | 67% | 33% | 33% | 33% |
| NF11 | audio_only | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| NF12 | slides_only | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| NF13 | paper_only | 100% | 0% | 0% | 100% | 67% | 0% | 100% | 100% | 100% | 100% | 100% | 100% |
| NF14 | paper_only | 100% | 33% | 33% | 100% | 56% | 78% | 100% | 100% | 100% | 100% | 100% | 89% |
| PS01 | slides_only | 67% | 33% | 33% | 67% | 56% | 44% | 67% | 67% | 67% | 67% | 67% | 67% |
| PS02 | cross_modal_split | 67% | 33% | 67% | 100% | 67% | 67% | 67% | 67% | 67% | 100% | 89% | 89% |
| PS03 | cross_modal_split | 100% | 50% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| PS04 | cross_modal_split | 100% | 50% | 100% | 100% | 50% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| PS05 | cross_modal_split | 100% | 50% | 50% | 100% | 50% | 50% | 100% | 100% | 100% | 100% | 100% | 100% |
| PS06 | cross_modal_split | 67% | 0% | 33% | 67% | 33% | 33% | 67% | 67% | 33% | 67% | 67% | 44% |
| PS08 | cross_modal_split | 67% | 33% | 33% | 100% | 67% | 33% | 67% | 67% | 33% | 89% | 89% | 33% |
| PS09 | cross_modal_deictic | 67% | 33% | 67% | 100% | 33% | 100% | 67% | 67% | 100% | 100% | 100% | 100% |
| PS10 | cross_modal_deictic | 60% | 20% | 40% | 80% | 73% | 87% | 60% | 60% | 40% | 67% | 80% | 80% |
| PS11 | audio_only | 100% | 50% | 50% | 100% | 50% | 83% | 100% | 100% | 100% | 100% | 100% | 100% |
| PS12 | slides_only | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| PS13 | paper_only | 100% | 50% | 50% | 100% | 100% | 50% | 100% | 100% | 100% | 100% | 100% | 83% |
| PS14 | paper_only | 100% | 50% | 50% | 100% | 100% | 50% | 100% | 100% | 100% | 100% | 100% | 100% |
| VS02 | cross_modal_split | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| VS03 | cross_modal_split | 25% | 25% | 25% | 42% | 25% | 42% | 25% | 25% | 50% | 42% | 42% | 50% |
| VS04 | cross_modal_split | 50% | 50% | 50% | 50% | 50% | 50% | 50% | 50% | 50% | 50% | 50% | 50% |
| VS05 | cross_modal_split | 100% | 50% | 50% | 100% | 83% | 83% | 100% | 100% | 100% | 100% | 100% | 100% |
| VS06 | cross_modal_split | 75% | 25% | 25% | 75% | 83% | 75% | 75% | 75% | 50% | 75% | 75% | 42% |
| VS07 | cross_modal_split | 100% | 33% | 67% | 100% | 67% | 89% | 100% | 100% | 100% | 100% | 100% | 100% |
| VS08 | cross_modal_split | 67% | 33% | 67% | 67% | 67% | 67% | 67% | 67% | 67% | 67% | 67% | 67% |
| VS09 | cross_modal_deictic | 0% | 0% | 0% | 56% | 56% | 44% | 0% | 0% | 0% | 56% | 56% | 56% |
| VS10 | cross_modal_deictic | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| VS11 | audio_only | 100% | 100% | 50% | 100% | 83% | 67% | 100% | 100% | 100% | 100% | 100% | 100% |
| VS12 | slides_only | 100% | 0% | 100% | 100% | 67% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| VS13 | paper_only | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| VS14 | paper_only | 100% | 50% | 50% | 100% | 50% | 50% | 100% | 100% | 100% | 100% | 100% | 100% |

LLM cost (this run):

- `claude-haiku-5-5`: 1512 calls (1512 cached) · 2,364,716 in / 218,121 out (of which 2,582,837 tokens estimated from cache) · price unknown
- `claude-sonnet-5-5`: 2772 calls (2772 cached) · 843,234 in / 223,040 out (of which 1,066,274 tokens estimated from cache) · price unknown
- run total: price unknown
- cumulative (all recorded runs, `reports/llm_ledger.jsonl`): 55,062 calls · 85,795,769 in / 3,322,582 out · $23.21 · 24 row(s) unpriced · 72,446,335 tokens estimated
