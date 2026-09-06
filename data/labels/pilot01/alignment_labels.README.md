# Provenance of `alignment_labels.csv` — READ BEFORE CITING

These labels were **not** produced by watching the recording. I cannot see or hear
the video. Each row was assigned by finding speech that quotes content appearing on
exactly one slide — verbatim numbers (`89X`, `162X`, `$380`, `5 hours -> 2 mins`,
`11 live traffic`), proper nouns (`YOLO V2`, `NoScope`, `ResNet18`), or a bullet
list read aloud in order. The `note` column records the specific evidence.

**Why not automate it.** Auto-anchoring on "terms unique to one slide" was tried and
discarded: it mapped the opening segment (title + collaborators, obviously slide 1)
to slide 6 on the strength of the words *goal* and *over*. Generic words that happen
to be slide-unique carry no evidence, and the resulting label set would have been
noise dressed as ground truth.

**Known weaknesses.**
- n = 11 of 53 audio segments. Sparse, and biased toward the results section, where
  the speaker reads numbers aloud — exactly the segments any method finds easiest.
- Silent stretches (motivation, the demo walk-through, Q&A) are under-represented
  because they quote nothing unique.
- Rows 807–863s are two consecutive segments both labelled p21; if the speaker had
  already advanced to p22 while still describing the baselines, the second is wrong.
- Labels derive from *text overlap*, which is one of the three signals the aligner
  itself scores on. The monotonic prior is independent of them, so the DP-vs-naive
  comparison is still meaningful — but an absolute accuracy from this file is
  optimistic and should not be quoted as such.

Replace this file with labels made while watching the talk before any number from it
goes in the paper.
