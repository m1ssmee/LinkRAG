# LectQA-Vid — timestamp check of 75 gold intervals

Each row's reference answer was located in the video's timestamped transcript, and the location was compared with the annotated stamp ± 3 s. Rows and locations: `lectqa_timestamp_check.csv`.

## Caveats first

- **The checker is a language model, not a person.** Claude Sonnet agents located each answer in the transcript without being shown the stamp; the comparison with the stamp is computed here. Each location carries the agent's confidence. Three rows were spot-checked by reading the transcript; all three agreed.
- **Speech only.** An answer that describes something shown on screen but never said counts as not spoken, even if the stamp points at the right frame.
- **Small samples:** 34, 16 and 25 rows. The control group (16 passing stamps) is the comparison for the 34.

## Stamps that pass the format and range checks

| group | rows | stamp points at the spoken answer | stamp misses it | answer never spoken |
|---|---:|---:|---:|---:|
| the 34 where eq. 22 left no context | 34 | 7 | 16 | 11 |
| … high-confidence locations only | 16 | 4 | 12 | — |
| 16 other passing stamps (control) | 16 | 12 | 3 | 1 |
| … high-confidence locations only | 11 | 10 | 1 | — |

## Stamps with an ambiguous format

For each of the 25, the stamp was read under every candidate convention (M:S with seconds above 59, H:M:S with overflow, S:cc, M:S:cc) and compared with the located answer.

| reading that matches the spoken answer | rows |
|---|---:|
| n under every reading | 9 |
| H:M:S overflow | 5 |
| M:S:cc | 5 |
| S:cc | 4 |
| M:S overflow | 2 |

## Reading

On the empty-window questions the stamp misses the spoken answer or the answer is never said in 27 of 34, against 4 of 16 among other passing stamps. So most of the 34 empty windows point at a moment where the answer is not spoken: eq. 22 removed the evidence because of the interval, not because retrieval failed. Where the answer is never said, it may be shown on screen; this check cannot tell. Ambiguous stamps follow no single convention: no reading recovers them all, which is why they are excluded rather than guessed.

`python scripts/adapters/lectqa_vid.py tscheck --max-cost 0` rebuilds this file from the sheet.
