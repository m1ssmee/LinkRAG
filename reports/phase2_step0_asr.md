# Step 0 — slide-guided ASR vs plain ASR (`pilot01`)

Whisper `small`, int8, CPU. Plain = no `initial_prompt`. Guided = `initial_prompt`
set to 75 vocabulary terms mined from the 27-slide deck (`build_asr_prompt`,
`ingest.asr_vocab_from_slides`). Both transcripts are kept:

- `data/processed/transcripts/hsieh.plain.json` (3,240 words)
- `data/processed/transcripts/hsieh.guided.json` (3,243 words)
- `data/processed/transcripts/hsieh.guided_v1_hyphenated.json` (the v1 prompt below)

## Cost — measurement retracted

| run | wall-clock | conditions |
|---|---:|---|
| plain | 476s | light background load |
| guided v1 | 990s | **ran concurrently with the alignment build (bge-m3 embedding)** |
| guided v2 | **231s** | light background load |

An earlier draft of this report concluded "the prompt costs 2.1× decode time". **That
was wrong** — v1 was competing for cores with a bge-m3 embedding job. The clean v2 run
transcribed the same audio in 231s, *faster* than the plain run. These three timings
were not taken under controlled conditions and **no claim about the prompt's effect on
decode speed should be made from them.** Re-measure on an idle machine if the number
matters.

## Target term status (word-boundary counts, not substrings)

| term | plain | guided | verdict |
|---|---:|---:|---|
| `ingest` | 7 | **16** | **fixed** — 10 spoken instances recovered |
| `interest` (as a mis-hearing of *ingest*) | 20 | **10** | halved, **not eliminated** |
| `NoScope` | 4 | 4 | unchanged — was already correct |
| `YOLO` | 3 | 3 | unchanged — was already correct |
| `Resnet` / `Resnet18` | 2 + 1 | 0 + 1 | tokenisation change (`Resnet 18` → `Resnet18`), not a loss |
| `top-K` (any spelling) | 8 | **4** | **regression** — 4 became `type -k` |

So of the three terms asked about: **`ingest` improved substantially, `NoScope` and
`YOLO v2` were never broken.** The premise that the vocabulary prompt would help
those two was wrong — plain whisper already got them right.

### Regression the prompt introduced (v1, hyphenated)

The v1 prompt contained `Top-K`. Whisper then emitted **`type -k`** for four spoken
occurrences of "top-K":

```
@701.2s  plain='topk'  ->  guided='type -k'
@712.2s  plain='topk'  ->  guided='type -k'
@716.9s  plain='topk'  ->  guided='type -k'
@759.4s  plain='topk'  ->  guided='type -k'
```

### The hyphen hypothesis was tested and rejected

I hypothesised the hyphen split the single-letter suffix, de-hyphenated the prompt
(`Top K`, `Low Cost`, `Ingest time`) and re-ran the full transcription. **It changed
nothing**: v2 produced `type -k` in the same four places, and identical counts for
every tracked term.

| form | plain | guided v1 (hyphens) | guided v2 (no hyphens) |
|---|---|---|---|
| `top K` / `top k` / `topk` | 8 | 4 | 4 |
| **`type -k`** | 0 | **4** | **4** |

The corruption is caused by the presence of the vocabulary prompt, not by its
punctuation. The workaround was reverted; the cause is **still unknown and the
regression is unfixed**. A prompt without `Top K` at all is the next thing to try.

## WER on a 50-word sample

Window fixed by plain-transcript word indices `[1535:1585]` = **613.6–636.3s**, so
both transcripts are scored over the same span.

```
PLAIN : query a specific video so we can reduce the waste to process all the videos
        at the interest time. So the second technique is that we reduce the redundancy
        to enable fast query. So as we see, using this approximate indexing
        architecture, we introduce an untrivial work at query time,
GUIDED: (identical, except the deletion below)
```

Only one difference across the whole window:

| difference | adjudication |
|---|---|
| plain `at **the** interest time` → guided `at interest time` | **undecidable** — I cannot hear the audio, and both readings are grammatical |
| both: `interest time` | **both wrong**; slides read *Ingest-time* (p10/p11), so the reference is `ingest time` |

| | adjudicable errors | words | WER |
|---|---:|---:|---:|
| plain | 1 | 50 | **2.0%** |
| guided | 1 | 49 | **2.0%** |

**The guided prompt did not help in this window** — it left *interest* uncorrected
here, consistent with fixing only 10 of ~20 instances corpus-wide.

### Honest limits of this WER figure

- **The reference is text-adjudicated, not audio-verified.** I cannot listen to the
  recording. `interest → ingest` is settled by the slide deck; the `the` deletion is
  not settled by anything and is excluded rather than guessed.
- 50 words is thin: the two transcripts differ in only 85 of 3,243 word positions
  (2.6%), so a random 50-word window is expected to contain ~1 difference, and it did.
  **The term-level table above is the substantive result; this WER is nearly
  uninformative and should not be quoted alone.**
- An earlier attempt at this section scored both transcripts against a reference I
  wrote for a *different* time span, producing a meaningless 77%/88%. Discarded.

## Recommendation

Keep `asr_vocab_from_slides: true`, but **it is a genuine trade, not a free win**:
**+9 correct `ingest`, −4 corrupted `top-K`**, reproduced identically across two runs.
The `top-K` cause is unidentified and unfixed.

The cheaper and more reliable remedy for the original problem is a **query-time alias**
(`ingest ↔ interest`, `top-K ↔ topk ↔ type -k`) applied to BM25 tokenisation. It costs
no decode time, introduces no new ASR errors, and fixes the ~10 residual `interest`
instances the prompt does not reach. Recommend doing that regardless of what happens
to the prompt.
