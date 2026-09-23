# Extended-dataset intake (priority vi)

Who this is for: the teammate collecting lectures. You need Python set up
(`make setup`) and a GPU for the transcription (a free Colab T4 is the standard route,
below). Redundancy is judged by the LLM judge on Groq's free tier: export
`GROQ_API_KEY` (free account) in `~/.zshenv`. Do not switch the intake to the local NLI
backend: on pilot01 it moves deck→transcript from 94.2 % to 59.6 % and would flip a
REJECT to KEEP (DESIGN.md finding 11). Until the Groq judge's agreement with the
stored judges has been measured, treat its intake verdicts as provisional (DESIGN.md
known issue (f)). The free tier caps requests per day and one lecture needs more than
a thousand judge calls, so intake takes more than one day. Re-running resumes from the
reply cache.

## What we are looking for

From `DESIGN.md` → Findings: link-following helps only where the modalities are
**complementary**. pilot01 failed that test — the speaker narrated 94 % of the deck.
A candidate lecture is kept when:

1. **Low redundancy** — deck→transcript **< 0.65** (`dataset.intake.max_deck_to_transcript`).
   The other pairs are reported for information.
2. **Diagram-heavy deck** — plots, architectures, tables that carry facts the slide
   text does not. Eyeball the deck: if every slide is bullet text, skip it.
3. **A speaker who points** — "as you can see here", "this curve", "the box on the
   left". Listen to five minutes; if the speaker never refers to the screen, skip it.

Sources that fit: MaViLS lectures (20, with slides and human slide labels —
`github.com/andererka/MaViLS`), NPTEL courses that publish both slides and notes,
MIT OCW courses with lecture notes PDFs. Prefer a deck **plus** a notes/paper PDF:
that is the cross-file setting no target benchmark covers.

## Procedure

```bash
# 1. Put the files somewhere under data/raw/<name>/ (raw media is gitignored)
mkdir -p data/raw/mit_6006_l03 && cp l03.mp3 l03_slides.pdf l03_notes.pdf data/raw/mit_6006_l03/

# 2. Run intake. Ingests to data/processed/candidates/<name>/ and measures redundancy.
python scripts/dataset/candidate.py --name mit_6006_l03 \
    --audio data/raw/mit_6006_l03/l03.mp3 \
    --deck  data/raw/mit_6006_l03/l03_slides.pdf \
    --notes data/raw/mit_6006_l03/l03_notes.pdf        # optional

# 3. Read reports/redundancy_<name>.md: the verdict, the per-pair numbers, and the
#    "what each source says that the others do not" samples -- those are the
#    sentences future cross-modal questions will be built from.
```

Exit code 0 = KEEP, 1 = REJECT, 2 = a file was missing. The number is written to the
report either way; a REJECT that is close to the bar with a diagram-heavy deck is
worth flagging rather than discarding.

### Transcription — the Colab batch is the standard route

Intake transcribes with **local faster-whisper on a GPU** (`candidate.py --device cuda
--asr local`, the defaults). It costs nothing and does not touch the frozen pilot01
transcript. A laptop CPU runs at ~3× realtime (a 60-minute lecture takes ~20 minutes);
a free Colab T4 is much faster, so batch intake runs there:

```python
# Colab, Runtime → Change runtime type → T4 GPU
!git clone https://github.com/<you>/LinkRAG.git && cd LinkRAG && pip -q install -e .
from google.colab import drive; drive.mount("/content/drive")   # lectures live in Drive
!mkdir -p data/raw && ln -s /content/drive/MyDrive/linkrag_raw/* data/raw/
# one line per lecture; --device cuda switches whisper to float16
!python scripts/dataset/candidate.py --name mit_6006_l03 --device cuda \
    --audio data/raw/mit_6006_l03/l03.mp3 --deck data/raw/mit_6006_l03/l03_slides.pdf
# keep the frozen transcripts and reports: copy them back to Drive, commit from there
!cp -r data/processed/transcripts reports /content/drive/MyDrive/linkrag_out/
```

The frozen transcript (`data/processed/transcripts/<stem>.frozen.json`) is written on
the first run and reused after that, so the GPU is needed once per lecture; the
redundancy step afterwards runs anywhere (`--device cpu`). Re-runs reuse the scratch
index unless `--reingest`.

**whisper-1 is kept as an option** (`--asr openai`): ~15× realtime, better
terminology (`reports/asr_openai_pilot01.md`), **$0.36 per hour of audio**. It bills,
so under zero-cost mode it is refused before the first upload unless the budget is
raised (`cost.max_usd` in the config).

## Record every decision

Add a row here whenever a candidate is run, kept or not — the rejects are the
evidence that the criterion means something.

| name | source | deck→transcript | transcript→deck | deck→notes | diagram-heavy? | points? | verdict | who / date |
|---|---|---:|---:|---:|---|---|---|---|
| pilot01 | OSDI '18 Focus talk | 94.2 % | 61.1 % | 46.2 % | partly | yes (10 windows) | reference corpus, would be REJECTED | — / 2026-09-21 |

## What happens after KEEP

1. Ear-label nothing. Write 20–30 proposed questions with reference answers that
   list only the asked facts (`tests/regression/<name>_proposed.jsonl`, same format as
   `pilot01_proposed.jsonl`).
2. `python scripts/eval/verify_gold.py --proposed tests/regression/<name>_proposed.jsonl --out tests/regression/<name>_questions.jsonl --corpus <name> --index data/processed/candidates/<name>/index`
3. `python scripts/eval/audit_sample.py --verdicts reports/gold_verified_<name>.json --out reports/audit_sheet_<name>.csv` and fill the sheet.
4. `python scripts/build_links.py` and `compare_retrieval.py` against the candidate index.
