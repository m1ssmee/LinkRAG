
## XFile test half (gp_scaling, mimic_data, rl_mdp), frozen default knobs, cache-only replay

38 questions · k=8 · pool=20 · corpus `a086ca1c22dc1d93` (2210 units) · gold stamped `a086ca1c22dc1d93` · model `claude-haiku-5-5` temperature=0.0 seed=20260910 · repeats 3 for LLM modes · 228 LLM calls · expansion `additive` · normalise_seeds `True`

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

Repeat determinism (complementarity cell):

- `baseline`: IDENTICAL across runs
- `iterative`: NOT identical — 3 distinct outcomes in 3 runs
- `linkrag`: IDENTICAL across runs
- `linkrag_iter`: NOT identical — 3 distinct outcomes in 3 runs

### By verified question type

| type | n | mode | rerank | recall@k | precision@k | modalities | LLM calls / q |
|---|---:|---|---|---:|---:|---:|---:|
| single_modality | 17 | baseline | none | 76.5% ± 0.0% | 16.9% ± 0.0% | 1.94 | 0.0 |
| single_modality | 17 | baseline | mmr | 51.0% ± 0.0% | 11.0% ± 0.0% | 2.18 | 0.0 |
| single_modality | 17 | baseline | complementarity | 74.5% ± 0.0% | 16.2% ± 0.0% | 2.24 | 0.0 |
| single_modality | 17 | iterative | none | 75.2% ± 4.1% | 16.9% ± 0.7% | 2.14 | 1.0 |
| single_modality | 17 | iterative | mmr | 65.0% ± 4.9% | 13.5% ± 1.1% | 2.29 | 1.0 |
| single_modality | 17 | iterative | complementarity | 72.2% ± 2.5% | 15.4% ± 0.7% | 2.35 | 1.0 |
| single_modality | 17 | linkrag | none | 76.5% ± 0.0% | 16.9% ± 0.0% | 1.94 | 0.0 |
| single_modality | 17 | linkrag | mmr | 74.5% ± 0.0% | 16.2% ± 0.0% | 1.94 | 0.0 |
| single_modality | 17 | linkrag | complementarity | 76.5% ± 0.0% | 16.9% ± 0.0% | 2.88 | 0.0 |
| single_modality | 17 | linkrag_iter | none | 78.4% ± 4.3% | 17.9% ± 0.8% | 2.27 | 1.0 |
| single_modality | 17 | linkrag_iter | mmr | 80.1% ± 5.7% | 18.4% ± 1.3% | 2.24 | 1.0 |
| single_modality | 17 | linkrag_iter | complementarity | 76.5% ± 4.9% | 17.4% ± 0.8% | 2.92 | 1.0 |
| cross_modal | 21 | baseline | none | 57.8% ± 0.0% | 18.5% ± 0.0% | 2.10 | 0.0 |
| cross_modal | 21 | baseline | mmr | 50.3% ± 0.0% | 14.9% ± 0.0% | 2.10 | 0.0 |
| cross_modal | 21 | baseline | complementarity | 54.0% ± 0.0% | 17.3% ± 0.0% | 2.14 | 0.0 |
| cross_modal | 21 | iterative | none | 62.4% ± 2.6% | 20.4% ± 0.9% | 2.30 | 1.0 |
| cross_modal | 21 | iterative | mmr | 55.9% ± 4.8% | 17.3% ± 1.6% | 2.38 | 1.0 |
| cross_modal | 21 | iterative | complementarity | 61.1% ± 6.8% | 19.2% ± 2.3% | 2.48 | 1.0 |
| cross_modal | 21 | linkrag | none | 57.8% ± 0.0% | 18.5% ± 0.0% | 2.10 | 0.0 |
| cross_modal | 21 | linkrag | mmr | 59.4% ± 0.0% | 19.0% ± 0.0% | 2.14 | 0.0 |
| cross_modal | 21 | linkrag | complementarity | 60.2% ± 0.0% | 19.0% ± 0.0% | 3.00 | 0.0 |
| cross_modal | 21 | linkrag_iter | none | 60.2% ± 1.8% | 19.6% ± 0.6% | 2.24 | 1.0 |
| cross_modal | 21 | linkrag_iter | mmr | 63.7% ± 1.4% | 20.4% ± 0.7% | 2.30 | 1.0 |
| cross_modal | 21 | linkrag_iter | complementarity | 61.1% ± 1.5% | 19.8% ± 0.7% | 3.00 | 1.0 |
| deictic | 1 | baseline | none | 75.0% ± 0.0% | 37.5% ± 0.0% | 3.00 | 0.0 |
| deictic | 1 | baseline | mmr | 25.0% ± 0.0% | 12.5% ± 0.0% | 2.00 | 0.0 |
| deictic | 1 | baseline | complementarity | 75.0% ± 0.0% | 37.5% ± 0.0% | 3.00 | 0.0 |
| deictic | 1 | iterative | none | 75.0% ± 0.0% | 37.5% ± 0.0% | 3.00 | 1.0 |
| deictic | 1 | iterative | mmr | 50.0% ± 0.0% | 25.0% ± 0.0% | 3.00 | 1.0 |
| deictic | 1 | iterative | complementarity | 66.7% ± 14.4% | 33.3% ± 7.2% | 3.00 | 1.0 |
| deictic | 1 | linkrag | none | 75.0% ± 0.0% | 37.5% ± 0.0% | 3.00 | 0.0 |
| deictic | 1 | linkrag | mmr | 75.0% ± 0.0% | 37.5% ± 0.0% | 3.00 | 0.0 |
| deictic | 1 | linkrag | complementarity | 75.0% ± 0.0% | 37.5% ± 0.0% | 3.00 | 0.0 |
| deictic | 1 | linkrag_iter | none | 75.0% ± 0.0% | 37.5% ± 0.0% | 3.00 | 1.0 |
| deictic | 1 | linkrag_iter | mmr | 66.7% ± 14.4% | 33.3% ± 7.2% | 3.00 | 1.0 |
| deictic | 1 | linkrag_iter | complementarity | 75.0% ± 0.0% | 37.5% ± 0.0% | 3.00 | 1.0 |

Bucket sizes: single_modality 17, cross_modal 21, deictic 1 (deictic ⊂ cross_modal; single + cross = 38 questions).

### Answer correctness (judge PASS vs expected answer)

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

Per-question recall@k (mean over runs):

| Q | type | baseline/none | baseline/mmr | baseline/complementarity | iterative/none | iterative/mmr | iterative/complementarity | linkrag/none | linkrag/mmr | linkrag/complementarity | linkrag_iter/none | linkrag_iter/mmr | linkrag_iter/complementarity |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| RL02 | cross_modal_split | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% |
| RL04 | cross_modal_split | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| RL05 | cross_modal_split | 50% | 50% | 50% | 50% | 50% | 50% | 50% | 50% | 50% | 50% | 50% | 50% |
| RL06 | cross_modal_split | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 50% | 83% | 83% |
| RL07 | cross_modal_split | 33% | 0% | 0% | 44% | 44% | 33% | 33% | 33% | 33% | 33% | 33% | 22% |
| RL08 | cross_modal_split | 50% | 100% | 100% | 67% | 67% | 67% | 50% | 50% | 50% | 67% | 67% | 67% |
| RL10 | slides_only | 33% | 0% | 0% | 44% | 11% | 11% | 33% | 33% | 33% | 33% | 44% | 33% |
| RL11 | audio_only | 100% | 100% | 100% | 67% | 100% | 83% | 100% | 100% | 100% | 67% | 83% | 83% |
| RL12 | audio_only | 100% | 0% | 100% | 100% | 67% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| RL13 | slides_only | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| RL14 | paper_only | 100% | 50% | 100% | 100% | 50% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| GP01 | cross_modal_split | 100% | 50% | 50% | 100% | 50% | 83% | 100% | 100% | 100% | 100% | 100% | 100% |
| GP02 | cross_modal_split | 100% | 50% | 50% | 50% | 100% | 100% | 100% | 100% | 100% | 83% | 83% | 33% |
| GP03 | cross_modal_split | 0% | 50% | 50% | 17% | 33% | 33% | 0% | 0% | 50% | 0% | 50% | 50% |
| GP05 | cross_modal_split | 67% | 33% | 33% | 67% | 44% | 44% | 67% | 67% | 67% | 78% | 78% | 78% |
| GP06 | cross_modal_split | 80% | 40% | 60% | 80% | 40% | 60% | 80% | 80% | 80% | 80% | 80% | 93% |
| GP07 | cross_modal_split | 0% | 33% | 33% | 11% | 11% | 11% | 0% | 33% | 0% | 0% | 0% | 0% |
| GP08 | cross_modal_split | 25% | 25% | 50% | 50% | 33% | 33% | 25% | 25% | 25% | 42% | 42% | 25% |
| GP09 | audio_only | 67% | 67% | 67% | 67% | 33% | 33% | 67% | 67% | 33% | 56% | 56% | 56% |
| GP10 | slides_only | 67% | 33% | 67% | 67% | 56% | 67% | 67% | 67% | 100% | 100% | 100% | 100% |
| GP11 | audio_only | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% |
| GP12 | audio_only | 100% | 0% | 100% | 67% | 100% | 100% | 100% | 100% | 100% | 67% | 67% | 67% |
| GP13 | slides_only | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| GP14 | paper_only | 50% | 0% | 50% | 67% | 50% | 50% | 50% | 50% | 50% | 100% | 100% | 50% |
| MD01 | audio_only | 67% | 33% | 67% | 67% | 56% | 67% | 67% | 67% | 67% | 67% | 67% | 67% |
| MD02 | cross_modal_split | 33% | 33% | 33% | 33% | 33% | 33% | 33% | 33% | 33% | 33% | 33% | 33% |
| MD03 | cross_modal_split | 50% | 50% | 50% | 50% | 50% | 50% | 50% | 50% | 50% | 50% | 50% | 50% |
| MD04 | cross_modal_split | 33% | 33% | 67% | 67% | 67% | 67% | 33% | 33% | 33% | 67% | 67% | 67% |
| MD05 | cross_modal_split | 67% | 33% | 33% | 100% | 67% | 100% | 67% | 67% | 67% | 89% | 89% | 89% |
| MD06 | cross_modal_split | 50% | 50% | 0% | 50% | 50% | 50% | 50% | 50% | 50% | 67% | 67% | 67% |
| MD07 | cross_modal_split | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| MD08 | cross_modal_split | 100% | 100% | 100% | 100% | 83% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| MD09 | slides_only | 67% | 33% | 67% | 67% | 33% | 67% | 67% | 33% | 67% | 78% | 78% | 78% |
| MD10 | cross_modal_deictic | 75% | 25% | 75% | 75% | 50% | 67% | 75% | 75% | 75% | 75% | 67% | 75% |
| MD11 | audio_only | 50% | 50% | 50% | 67% | 50% | 50% | 50% | 50% | 50% | 67% | 67% | 67% |
| MD12 | audio_only | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| MD13 | slides_only | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| MD14 | paper_only | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |

LLM cost (this run):

- `claude-haiku-5-5`: 1596 calls (1596 cached) · 2,698,044 in / 239,485 out (of which 2,937,529 tokens estimated from cache) · price unknown
- `claude-sonnet-5-5`: 2829 calls (2829 cached) · 888,708 in / 250,885 out (of which 1,139,593 tokens estimated from cache) · price unknown
- run total: price unknown
- cumulative (all recorded runs, `reports/llm_ledger.jsonl`): 59,487 calls · 89,382,521 in / 3,812,952 out · $23.21 · 26 row(s) unpriced · 76,523,457 tokens estimated

## Paired differences, 95 % interval from resampling the 3 test topics (10,000 resamples, seed 20261118)

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
