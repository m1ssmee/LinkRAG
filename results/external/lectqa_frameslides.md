# LectQA-Vid — frame-derived slide units

95 prepared videos · slides from frames at 1 fps (dHash segmentation, transitions merged, revisits deduplicated), OCR tesseract, figures by the deck clustering rules on raster frames (`linkrag.ingest.video_slides`) · links from `linkrag.link.pipeline.link_corpus` (the pilot01 path). **Retrieval only, no LLM, $0.** Iterative modes need a follow-up LLM call and are not run.

*Without* = the prepared index: audio + one OCR unit per changed frame, joined by temporal co-occurrence links (T1's own signal). *With* = the same audio units + frame-derived slide page and figure units, joined by the linking layer (alignment, figure_text, same_slide, deictic); no temporal links, so the alignment has to find the slides from text. A revisited slide counts a hit on any of its on-screen intervals.

## Slides, figures and links

- slides 2175, figure units 2912, audio units 1870
- relatedness gate rejected the audio↔slide alignment on 49 of 95 videos

| link type | count |
|---|---:|
| audio_slide | 992 |
| deictic | 995 |
| figure_text | 101 |
| same_slide | 2912 |
| **total** | 5000 |

## Alignment against the known on-screen interval (free sanity check)

Audio segments whose midpoint falls inside a slide's interval: n = 1870. DP alignment (config `link.align`) picks the on-screen slide for **421/1870 = 22.5%**; naive argmax on the same matrix 406/1870 = 21.7%. (Their alignment component is T2, MaViLS; this is not their metric.)

## Modality mix of the retrieved top-8 (rerank none; mean units per set)

| variant | mode | audio | text (slide) | figure |
|---|---|---:|---:|---:|
| without | baseline | 6.43 | 0.00 | 1.55 |
| without | linkrag | 6.43 | 0.00 | 1.55 |
| with | baseline | 5.77 | 1.71 | 0.52 |
| with | linkrag | 5.77 | 1.71 | 0.52 |

## Localisation (hit@k: a retrieved unit overlaps the gold interval)

Questions with a usable gold interval: **1598 of 2832**. Excluded: ambiguous format 878, beyond the fetched video's end 355, end before start 1. The published annotations mix timestamp conventions (HH:MM:SS, SS:cc, SSS:cc, M:SSS:cc) whose meaning differs between videos, and some stamps end past the fetched video, so only unambiguous, in-range intervals are scored (`strict_seconds`, `gold_interval`).

### without frame-derived slides

| level | n | mode | rerank | hit@1 | hit@3 | hit@8 | mean best IoU |
|---|---:|---|---|---:|---:|---:|---:|
| simple | 601 | baseline | none | 54.1% | 78.2% | 90.2% | 0.469 |
| simple | 601 | baseline | complementarity | 54.6% | 66.6% | 80.5% | 0.397 |
| simple | 601 | linkrag | none | 54.6% | 78.2% | 90.2% | 0.469 |
| simple | 601 | linkrag | complementarity | 54.6% | 73.2% | 87.4% | 0.467 |
| hard | 515 | baseline | none | 39.4% | 59.6% | 79.0% | 0.373 |
| hard | 515 | baseline | complementarity | 39.4% | 49.7% | 70.5% | 0.309 |
| hard | 515 | linkrag | none | 39.4% | 59.6% | 79.0% | 0.373 |
| hard | 515 | linkrag | complementarity | 39.4% | 55.0% | 76.3% | 0.330 |
| very hard | 482 | baseline | none | 41.5% | 66.6% | 85.3% | 0.359 |
| very hard | 482 | baseline | complementarity | 41.7% | 55.4% | 76.8% | 0.309 |
| very hard | 482 | linkrag | none | 41.7% | 66.4% | 85.3% | 0.359 |
| very hard | 482 | linkrag | complementarity | 41.7% | 61.2% | 82.8% | 0.334 |
| overall | 1598 | baseline | none | 45.6% | 68.7% | 85.1% | 0.405 |
| overall | 1598 | baseline | complementarity | 45.8% | 57.8% | 76.2% | 0.342 |
| overall | 1598 | linkrag | none | 45.8% | 68.6% | 85.1% | 0.405 |
| overall | 1598 | linkrag | complementarity | 45.8% | 63.7% | 82.4% | 0.383 |

### with frame-derived slides

| level | n | mode | rerank | hit@1 | hit@3 | hit@8 | mean best IoU |
|---|---:|---|---|---:|---:|---:|---:|
| simple | 601 | baseline | none | 51.7% | 74.5% | 89.5% | 0.457 |
| simple | 601 | baseline | complementarity | 51.9% | 64.9% | 80.0% | 0.386 |
| simple | 601 | linkrag | none | 51.9% | 74.5% | 89.5% | 0.457 |
| simple | 601 | linkrag | complementarity | 51.9% | 64.4% | 84.2% | 0.419 |
| hard | 515 | baseline | none | 37.5% | 59.2% | 77.7% | 0.363 |
| hard | 515 | baseline | complementarity | 37.5% | 51.7% | 72.6% | 0.309 |
| hard | 515 | linkrag | none | 37.5% | 59.2% | 77.7% | 0.363 |
| hard | 515 | linkrag | complementarity | 37.5% | 50.9% | 72.4% | 0.327 |
| very hard | 482 | baseline | none | 38.8% | 65.8% | 84.9% | 0.361 |
| very hard | 482 | baseline | complementarity | 38.6% | 57.5% | 78.8% | 0.291 |
| very hard | 482 | linkrag | none | 38.6% | 65.8% | 84.9% | 0.361 |
| very hard | 482 | linkrag | complementarity | 38.6% | 56.6% | 78.8% | 0.323 |
| overall | 1598 | baseline | none | 43.2% | 67.0% | 84.3% | 0.397 |
| overall | 1598 | baseline | complementarity | 43.2% | 58.4% | 77.3% | 0.332 |
| overall | 1598 | linkrag | none | 43.2% | 67.0% | 84.3% | 0.397 |
| overall | 1598 | linkrag | complementarity | 43.2% | 57.7% | 78.8% | 0.360 |

