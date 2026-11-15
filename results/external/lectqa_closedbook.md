# LectQA-Vid — closed-book control: how much is answered without the video

The open-ended subset (300 questions, seed `lectqa-open-subset-20260923`), answered by Claude Haiku 5.5 (`claude-haiku-5-5`) run as 12 Claude Code subagents, under one prompt, 1 run(s): once with no context and once with the video's full transcript. Metrics as in `lectqa_open_modes.md`, in % (mean ± std over repeats when there is more than one).

## Caveats first

- **A new prompt is a new condition.** This prompt allows the model's own knowledge and forbids refusing; the stored modes of `lectqa_open_modes.md` use an evidence-only JSON prompt that must abstain. The two full-transcript rows below differ in the prompt, so only rows from this run are compared with each other.
- **Closed book is not no information.** The question text is in the prompt, and the references restate it (the echo row).
- One answering model. A model with more or less background knowledge would move the closed-book row.
- **One run: direction only** (reporting convention).
- **Agents answer in batches, not one call per question.** Each agent read one input file and answered 50 to 69 questions in one pass, so a question's answer could be influenced by its neighbours. Closed-book batches were dealt so that a video's questions fall in different batches where possible. Sampling settings are the agent runtime's defaults: the run is not replayable from this repository, only re-scorable from the stored answers.
- Intervals are 95 % bootstrap intervals over videos (10,000 resamples, seed 20260923).

The prompt (system): *You answer a question about a short computer-science lecture video in one or two sentences. Use the transcript if one is given and your own knowledge otherwise. Always give your best answer: never say that you cannot answer or that information is missing.*

## Results

| level | condition | n | F1 | Sim | refusals |
|---|---|---:|---:|---:|---:|
| simple | closed book (no context) | 100 | 31.01 | 61.52 | 1 |
| simple | full transcript | 100 | 42.99 | 71.67 | 2 |
| simple | *question echoed as the answer* | 100 | 23.70 | — | — |
| simple | *full transcript, evidence-only prompt (stored, repeat 0)* | 100 | 49.45 | 74.75 | — |
| simple | *T1 Table 4 (published; unnamed answerer)* | — | 29.47 | 77.23 | — |
| hard | closed book (no context) | 100 | 27.42 | 61.87 | 1 |
| hard | full transcript | 100 | 39.86 | 68.95 | 3 |
| hard | *question echoed as the answer* | 100 | 22.62 | — | — |
| hard | *full transcript, evidence-only prompt (stored, repeat 0)* | 100 | 38.41 | 65.02 | — |
| hard | *T1 Table 4 (published; unnamed answerer)* | — | 24.38 | 74.56 | — |
| very hard | closed book (no context) | 100 | 25.79 | 58.95 | 2 |
| very hard | full transcript | 100 | 29.81 | 63.23 | 0 |
| very hard | *question echoed as the answer* | 100 | 22.28 | — | — |
| very hard | *full transcript, evidence-only prompt (stored, repeat 0)* | 100 | 29.05 | 60.94 | — |
| very hard | *T1 Table 4 (published; unnamed answerer)* | — | 16.72 | 71.48 | — |
| overall | closed book (no context) | 300 | 28.07 | 60.78 | 4 |
| overall | full transcript | 300 | 37.55 | 67.95 | 5 |
| overall | *question echoed as the answer* | 300 | 22.87 | — | — |
| overall | *full transcript, evidence-only prompt (stored, repeat 0)* | 300 | 38.97 | 66.90 | — |
| overall | *T1 Table 4 (published; unnamed answerer)* | — | 23.52 | 74.42 | — |

## Paired differences (token F1 points)

| difference | level | by question | by video |
|---|---|---:|---:|
| full transcript − closed book (what the video adds) | overall | +9.48 [+7.90, +11.05] | +9.48 [+7.91, +11.06] |
| full transcript − closed book (what the video adds) | simple | +11.97 [+9.24, +14.68] | +11.97 [+9.42, +14.42] |
| full transcript − closed book (what the video adds) | hard | +12.43 [+9.62, +15.32] | +12.43 [+9.53, +15.42] |
| full transcript − closed book (what the video adds) | very hard | +4.03 [+1.85, +6.25] | +4.03 [+1.84, +6.16] |
| closed book − question echo (what the model's own knowledge adds) | overall | +5.21 [+3.93, +6.47] | +5.21 [+3.88, +6.57] |
| closed book − question echo (what the model's own knowledge adds) | simple | +7.31 [+4.74, +9.92] | +7.31 [+4.68, +10.06] |
| closed book − question echo (what the model's own knowledge adds) | hard | +4.81 [+2.92, +6.77] | +4.81 [+2.99, +6.57] |
| closed book − question echo (what the model's own knowledge adds) | very hard | +3.51 [+1.53, +5.51] | +3.51 [+1.47, +5.70] |

Share of the full-transcript F1 that the closed-book answer already reaches: **75%** (28.07 of 37.55). Questions where closed book scores at least as high as the transcript: 86 of 300.

LLM cost (this run): none through the project's API ledger; the answers came from agents.
