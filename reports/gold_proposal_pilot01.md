# Gold proposal — pilot01 Q1–Q4 against the 3-document corpus

**Proposal only. Nothing here is applied.** `tests/regression/pilot01_questions.jsonl`
is unchanged and still stamped `dded007ab45f5b9f` (2-document corpus). Approve or edit,
then I apply and re-stamp with `6ee13080f9ad2084`.

| corpus | manifest | units |
|---|---|---|
| gold was written against | `dded007ab45f5b9f` | 96 — deck + audio |
| current | `6ee13080f9ad2084` | 196 — deck + audio + **OSDI paper** |

Every unit below was verified by reading its indexed content, not inferred from
similarity scores.

---

## Q1 · `slides_only` — which prior system, and its venue/year?

**Current gold:** `osdi18_slides_hsieh.pdf` p.5 — the slide reading
`Kang et al., NoScope, PVLDB'17`.

| propose | unit | justification |
|---|---|---|
| **add** | `osdi18-hsieh:p17:t2` | Reference [51] in full: *"D. Kang, J. Emmons, F. Abuzaid, P. Bailis, and M. Zaharia. NoScope: Optimizing deep CNN-based queries over video streams at scale. **PVLDB, 2017**."* — answers both halves outright. |
| **add (partial)** | `osdi18-hsieh:p17:t1` | Same reference, truncated by chunking before the venue. Sufficient for the system name only. |
| keep | deck p.5 | Still valid and still the only *slide* source. |

⚠️ **The type label is now wrong.** With the paper present this is no longer
`slides_only` — the paper answers it more completely than the slide does. Either
rename the type to `document_text` or exclude the paper from Q1's scope.

---

## Q2 · `audio_only` — how are cheap CNNs produced, and is that automatic?

**Current gold:** `hsieh.mp3` 1338.8–1368.5s — the Q&A answer.

| propose | unit | justification |
|---|---|---|
| **add (partial)** | `osdi18-hsieh:p7:t0` | *"Focus applies various levels of compression, such as removing convolutional layers and reducing input resolution"* — answers **how**. |
| **add (partial)** | `osdi18-hsieh:p7:t1` | *"cheaper ResNet18 models by removing one layer at a time"* — the concrete mechanism the speaker describes. |
| keep | audio 1338.8–1368.5s | **The only source for the second half.** |

✅ **Q2 survives as a genuine `audio_only` question**, but only for its second clause.
The paper says *how* cheap CNNs are made; **no document states that the process is not
automatic** — that admission exists only in the Q&A. Recommend keeping the audio unit
as required gold and marking the two paper units optional, so the question still fails
if the audio is missed.

---

## Q3 · `cross_modal_split` — who are the authors, and from which institutions?

**Current gold:** deck p.1 (author names) + `hsieh.mp3` 0.6–30.0s (affiliations).

| propose | unit | justification |
|---|---|---|
| **add** | `osdi18-hsieh:p2:t0` | Title block carries the **full author list *and* the expanded affiliations** (`†Carnegie Mellon…`, `§Microsoft…`) in one unit. |
| keep | deck p.1, audio 0.6–30.0s | Still valid sources. |

🚨 **This destroys Q3's purpose and is the most important decision here.** Q3 exists to
test cross-modal composition — names in one modality, affiliations in another. The
paper's title block contains **both halves in a single text unit**, so the question is
now answerable without composing anything. This is exactly what the last regression
showed: **4/4 gold terms with 0/2 gold locators retrieved**, 8/8 text.

Three options:
1. **Scope Q3 to exclude the paper** — keeps a real cross-modal test; needs a
   per-question source filter, which the runner does not have yet.
2. **Retire Q3 and write a replacement** whose halves genuinely cannot co-occur in one
   document (audio-only Q&A content + a slide-only figure label).
3. Accept it as a single-source question and lose the cross-modal probe.

I recommend **(1) or (2)**. Do not simply add the paper unit: the question would then
report success while testing nothing.

---

## Q4 · `cross_modal_deictic` — plotted configuration labels + what the speaker selects

**Current gold:** deck p.20 (`Optimize for Ingest Cost` / `Balance` /
`Optimize for Query Latency`) + `hsieh.mp3` 773.9–806.3s (the spoken selection).

| propose | unit | justification |
|---|---|---|
| **add** | `osdi18-hsieh:p3:t4` | Names the same three configurations as **`Focus-Opt-Query`, `Focus-Opt-Ingest`, `Focus-Balance`** on the Figure 1 trade-off plot. |
| **add** | `osdi18-hsieh:p9:c0` | *"Figure 6: Parameter selection based on the ingest cost and query latency trade-off"* — the paper's counterpart to deck p.20. |
| **add** | `osdi18-hsieh:p13:t2` | *"depicts three alternative settings"* — prose describing the same three-way choice. |
| keep | deck p.20, audio 773.9–806.3s | Still valid. |

⚠️ **Weaker than Q3's problem but the same shape.** The paper names the configurations
under *different* labels (`Focus-Opt-Ingest` vs `Optimize for Ingest Cost`), so a
string-matching gold term check will not credit them — but a model reading the paper
can still answer the first half without the slide. The deictic half ("which one does
the speaker say they would select") remains **audio-only**, so Q4 keeps its cross-modal
character better than Q3 does.

---

## Summary

| Q | additions | still cross-modal? | action |
|---|---|---|---|
| Q1 | 2 paper units | n/a (single-modality by design) | rename type or scope out the paper |
| Q2 | 2 paper units (partial) | ✅ yes — "not automatic" is audio-only | add as optional gold |
| Q3 | 1 paper unit | ❌ **no — collapses to one unit** | **scope out the paper, or replace the question** |
| Q4 | 3 paper units | ⚠️ partly — second half stays audio-only | add; consider tightening gold terms |

Two capabilities the runner needs before some of this can be applied:

- **per-question source scoping** (for Q1/Q3 option 1)
- **required vs optional gold**, so Q2 fails when the audio is missed even though the
  paper units were retrieved

Say which options you want and I will implement both and re-stamp the gold file.
