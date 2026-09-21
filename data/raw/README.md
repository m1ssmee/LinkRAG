# Corpus — sources and licences

**Nothing in this directory is redistributed with the repository.** `.gitignore`
excludes every media file here; only this README is tracked. Fetch the sources
yourself from the links below and place them as shown.

## pilot01 — Focus (OSDI '18)

| file | what it is | where it comes from |
|---|---|---|
| `pilot01/hsieh.mp3` | Audio of the conference talk (22.9 min) | USENIX OSDI '18 presentation recording |
| `pilot01/osdi18_slides_hsieh.pdf` | Speaker's slide deck (27 slides) | USENIX OSDI '18 presentation materials |
| `pilot01/<paper>.pdf` | The paper itself (not yet added) | USENIX OSDI '18 proceedings |

Work: Kevin Hsieh, Ganesh Ananthanarayanan, Peter Bodik, Shivaram Venkataraman,
Paramvir Bahl, Matthai Philipose, Phillip B. Gibbons, Onur Mutlu — *Focus: Querying
Large Video Datasets with Low Latency and Low Cost*, 13th USENIX Symposium on
Operating Systems Design and Implementation (OSDI '18).

### Licence status — confirm before redistributing

USENIX publishes OSDI proceedings as open access, and presentation materials are
posted on the conference programme page. **The exact licence covering the audio
recording and the slide deck has not been verified by this project.** Treat all three
files as third-party copyrighted material: use them locally for research, and do not
commit, mirror or redistribute them. Check the terms on the USENIX OSDI '18 programme
page before doing anything else with them.

Derived text (`data/processed/transcripts/*.json`) *is* tracked, because the pilot
results are pinned to an exact transcript and cannot otherwise be reproduced. If you
consider a verbatim transcript of a talk to be a derivative work you would rather not
publish, remove the `!data/processed/transcripts/` exception from `.gitignore` and
re-run the ingest from your own copy of the audio.

## Adding your own corpus

Drop files into `data/raw/<corpus>/` and run:

    python scripts/ingest.py data/raw/<corpus>/*

Supported: `.pdf`, `.docx`, `.wav/.mp3/.m4a`, `.png/.jpg/.tif`. Slide decks are
detected by landscape page geometry; override per file with
`ingest.slide_deck_files` in `configs/default.yaml`.

## mavils — target T2 benchmark

```
git clone --depth 1 https://github.com/andererka/MaViLS data/raw/mavils
```

Anderer, Reich, Wölfel — *MaViLS, a Benchmark Dataset for Video-to-Slide Alignment…*,
Interspeech 2024. Repository licence: **Apache-2.0** (code and the files it ships:
faster-whisper transcripts `data/audioscripts/*.srt`, ground-truth `data/ground_truth_files/*.xlsx`,
slide PDFs `data/lectures/*.pdf`). The lecture *videos* are not in the repository
(Kaggle link in their README) and are not needed by `scripts/adapters/mavils.py`. The
slide PDFs themselves are the lecturers' / institutions' material (mostly MIT OCW,
CC BY-NC-SA 4.0) redistributed by the benchmark authors; keep them local.

## lectqa_vid — target T1 benchmark

Shafiq et al. — *Intra-Video Temporal-Aware RAG*, CMC 88(2), 2026. Annotations from
Mendeley Data, doi:10.17632/yt4nmz9mcv.1, **CC BY 4.0**: `mcq_questions.json`,
`open_ended_questions.json`, `video_links.docx` (YouTube links only). The dataset
page states that videos, keyframes and transcripts are *not* provided for copyright
reasons; `scripts/adapters/lectqa_vid.py fetch` downloads audio + low-res video per
link with yt-dlp for local research use only — do not commit or redistribute them.
`fetch_failures.txt` records links that were unavailable at fetch time (video_1 on
2026-09-21).
