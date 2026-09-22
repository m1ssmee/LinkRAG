# Gate v3 — segments-per-slide hypothesis (20 related MaViLS pairs)

z at three transcript granularities, 30-shuffle null, threshold z = 1.27 (`align.relatedness_z`). n = transcript segments, m = slides. Hypothesis under test: low n/m (few segments per slide) → low z.

| lecture | z 30 s | n/m | z 15 s | n/m | z sentence | n/m | rejected at (30 s / 15 s / sentence) |
|---|---:|---:|---:|---:|---:|---:|---|
| ML for health | 0.5 | 1.9 | 0.8 | 3.3 | 0.3 | 10.7 | ✗ / ✗ / ✗ |
| Decarbonization | 5.2 | 1.5 | 5.6 | 2.6 | 6.6 | 6.9 | ✓ / ✓ / ✓ |
| Climate & Cities | 6.9 | 0.9 | 9.3 | 1.6 | 9.1 | 4.4 | ✓ / ✓ / ✓ |
| Climate policies | 6.1 | 5.5 | 5.5 | 9.8 | 3.0 | 35.0 | ✓ / ✓ / ✓ |
| Cognitive robotics | 8.0 | 2.0 | 6.8 | 3.6 | 5.5 | 11.9 | ✓ / ✓ / ✓ |
| Computer Vision | 5.5 | 1.9 | 3.9 | 2.7 | 3.0 | 4.4 | ✓ / ✓ / ✓ |
| Productdesign | 5.9 | 3.5 | 3.9 | 6.4 | 1.8 | 26.5 | ✓ / ✓ / ✓ |
| Cryptocurrency | 4.8 | 2.0 | 4.8 | 3.7 | 2.8 | 13.2 | ✓ / ✓ / ✓ |
| Deep learning | 8.5 | 2.4 | 10.0 | 4.3 | 7.6 | 10.8 | ✓ / ✓ / ✓ |
| Image processing | 0.9 | 2.2 | 1.1 | 3.2 | 0.6 | 6.0 | ✗ / ✗ / ✗ |
| Numerics | 9.7 | 2.9 | 9.5 | 5.1 | 5.1 | 14.7 | ✓ / ✓ / ✓ |
| Phonetics | 2.5 | 3.3 | 3.2 | 5.9 | 2.0 | 23.3 | ✓ / ✓ / ✓ |
| Physics | 11.2 | 4.4 | 9.8 | 7.8 | 5.3 | 26.1 | ✓ / ✓ / ✓ |
| Psychology | 14.8 | 2.0 | 15.3 | 3.4 | 5.8 | 14.0 | ✓ / ✓ / ✓ |
| Reinforcement | 4.5 | 3.2 | 4.6 | 5.8 | 2.0 | 22.3 | ✓ / ✓ / ✓ |
| Sensory systems | -0.5 | 3.7 | -0.1 | 6.5 | 0.3 | 19.4 | ✗ / ✗ / ✗ |
| Short range | 4.7 | 4.6 | 5.1 | 7.7 | 2.7 | 17.6 | ✓ / ✓ / ✓ |
| Solar resource | 2.0 | 2.2 | 2.2 | 3.8 | 1.3 | 14.1 | ✓ / ✓ / ✓ |
| Team dynamics | 2.2 | 3.2 | 2.9 | 5.8 | 1.9 | 26.2 | ✓ / ✓ / ✓ |
| Computation theory | 1.6 | 9.2 | 1.3 | 15.9 | 0.6 | 50.1 | ✓ / ✓ / ✗ |

False rejections at z = 1.27: 30 s 3/20 · 15 s 3/20 · sentence 4/20.

Spearman ρ(z, n/m) pooled over the 60 (lecture, granularity) points: **-0.31** (p = 0.015). Within a granularity: w30 ρ = -0.15 (p = 0.53); w15 ρ = -0.14 (p = 0.57); sentence ρ = -0.35 (p = 0.13).

The three 30-second rejects:

- ML for health: w30 z = 0.5 (n/m 1.9), w15 z = 0.8 (n/m 3.3), sentence z = 0.3 (n/m 10.7)
- Image processing: w30 z = 0.9 (n/m 2.2), w15 z = 1.1 (n/m 3.2), sentence z = 0.6 (n/m 6.0)
- Sensory systems: w30 z = -0.5 (n/m 3.7), w15 z = -0.1 (n/m 6.5), sentence z = 0.3 (n/m 19.4)

**Hypothesis not confirmed** — finer granularity does not raise z for the rejected pairs (and the pooled correlation is driven by the granularity change itself, not by per-lecture n/m). No adaptive re-windowing is added; `align.gate_min_windows_per_slide` is not introduced.

