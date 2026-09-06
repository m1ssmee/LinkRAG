# Gold proposal — pilot01 Q1–Q4 (re-issued against the clustered-figure corpus)

**Proposal only. Nothing here is applied.** `tests/regression/pilot01_questions.jsonl`
is unchanged and still stamped `dded007ab45f5b9f` (2-document corpus).

| corpus | manifest | units | figures |
|---|---|---:|---:|
| gold was written against | `dded007ab45f5b9f` | 96 | 18 |
| previous proposal | `6ee13080f9ad2084` | 196 | 34 |
| **current** | **`2f3b35f27e86caf8`** | **209** | **47** |

Slide-deck figure extraction now clusters vector drawing ops as well as raster images,
so deck figures went 18 → 31 and 24 of 27 slides have at least one figure unit (was 11).
**That changed the gold picture for Q3 and Q4** — the sections below mark every
difference from the previous proposal.

---

## Q1 · `slides_only` — **unchanged from the previous proposal**

Gold: deck p.5 (existing) + `osdi18-hsieh:p17:t2` (proposed: the full
`Kang … NoScope … PVLDB, 2017` reference) + `osdi18-hsieh:p17:t1` (partial).

The new figure unit on slide 5 (`osdi18_slides_hsieh:p5:g0`) OCRs to noise
(`Loading... [| PRR a aR ara Ree`) and carries neither `kang` nor `pvldb`. **No change.**

⚠️ Still true: with the paper present the `slides_only` type label is wrong.

---

## Q2 · `audio_only` — **unchanged from the previous proposal**

Gold: `hsieh.mp3` 1338.8–1368.5s (required) + `osdi18-hsieh:p7:t0`, `p7:t1` (optional,
they answer *how* but not *whether it is automatic*).

No new deck figure carries the Q&A content. **No change.** Q2 remains the only question
whose second clause is genuinely audio-only.

---

## Q3 · `cross_modal_split` — 🔴 **changed again, and worse**

**New candidate:** `osdi18_slides_hsieh:p1:g1` — the title slide's logo band, now
extracted as a figure and OCR'd to `… negieMellon © Microsoft Carnes TOS" WISCONSIN`.
`osdi18_slides_hsieh:p27:g1` is the same band on the closing slide, cleaner:
`CarnegieMellon B™ Microsoft`.

| propose | unit | justification | vs previous proposal |
|---|---|---|---|
| **add** | `osdi18_slides_hsieh:p1:g1` | affiliations, OCR'd from the logo band | **NEW** |
| **add** | `osdi18_slides_hsieh:p27:g1` | same band, cleaner OCR | **NEW** |
| add | `osdi18-hsieh:p2:t0` | paper title block: names **and** affiliations | unchanged |
| keep | deck p.1, audio 0.6–30.0s | existing gold | unchanged |

**Q3 has now degraded twice.** The previous proposal noted that the paper's title block
collapsed it into a single-source question. It is now worse: **the deck alone answers
it** — names in `p1:t0` (text) and affiliations in `p1:g1` (figure), same page. The
question was designed to require composing across *modalities and files*; it now
requires composing across two units of the same slide.

Recommendation stands and strengthens: **scope the question, or replace it.** Adding
these units without scoping would make Q3 report success while testing nothing.

---

## Q4 · `cross_modal_deictic` — 🟡 **changed; gold gets stronger, the test gets weaker**

**New candidate:** `osdi18_slides_hsieh:p20:g0` — the tradeoff plot, now extracted, OCR
containing `Optimize for Ingest Cost … Balance`.

| propose | unit | justification | vs previous proposal |
|---|---|---|---|
| **add** | `osdi18_slides_hsieh:p20:g0` | the plot itself; OCR carries two of the three labels | **NEW** |
| add | `osdi18-hsieh:p3:t4` | names `Focus-Opt-Query/-Ingest/-Balance` | unchanged |
| add | `osdi18-hsieh:p9:c0` | paper's counterpart figure (Figure 6) | unchanged |
| add | `osdi18-hsieh:p13:t2` | prose describing the three settings | unchanged |
| keep | deck p.20 text, audio 773.9–806.3s | existing gold | unchanged |

The labels are now reachable from a *figure* as well as the slide's text, which is the
correct modelling of the deck — but it means the "slide-only half" of Q4 is now
satisfiable by two different units on the same page. The deictic half ("which one does
the speaker say they would select") is still audio-only, so Q4 keeps its cross-modal
character better than Q3 does.

---

## Summary

| Q | change since previous proposal | still cross-modal? | action |
|---|---|---|---|
| Q1 | none | n/a by design | rename type or scope out the paper |
| Q2 | none | ✅ yes | add 2 paper units as optional |
| Q3 | **+2 deck figure units; now answerable from one slide** | ❌ **no** | **scope or replace — urgent** |
| Q4 | +1 deck figure unit | ⚠️ partly | add; second half stays audio-only |

Runner capabilities still needed before some options can be applied: **per-question
source scoping** and **required vs optional gold**. Neither is built.

Say which options you want and I will implement and re-stamp with `2f3b35f27e86caf8`.
