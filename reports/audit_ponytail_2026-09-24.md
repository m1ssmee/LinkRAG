# Ponytail audit — 2026-09-24 (read-only, written before any edit)

Scope: `src/linkrag/**`, `scripts/**` and `configs/default.yaml` at commit `181f171`. The other
session's uncommitted edits (tqdm bars, popup) are not part of this audit and are not touched.

Method:

- ruff (F401 unused import, F541 f-string without placeholders, F841 unused variable, B007 unused loop variable, SIM115 open without a context manager);
- vulture at ≥ 60 % confidence, every hit grepped against `tests/`, docs and configs;
- every `configs/default.yaml` leaf key grepped as a quoted string in `src/`, `scripts/` and `notebooks/`;
- every function name defined more than once, with the bodies compared;
- the section-C items of `reports/audit_phase4.md` rechecked.

Line numbers refer to `181f171`.

**Keep rule.** A recorded ablation branch or an intake/eval instrument stays even when the default path
never calls it. The KEEP list is below the candidates.

Risk to recorded results: **none** = no code path that produces a recorded number changes; **render** =
a report-producing function changes, so the report must re-render byte-identical.

## Candidates, ranked by risk (lowest first)

| # | file:line | category | what | est. lines | risk | group |
|---|---|---|---|---:|---|---|
| 1 | `scripts/adapters/lectqa_vid.py:145` | delete | unused `from PIL import Image` in `extract_frames` | −1 | none | a |
| 2 | `scripts/ask.py:25` | delete | unused import `set_diagnostics` | 0 (import list) | none | a |
| 3 | `scripts/dataset/candidate.py:25`, `scripts/dataset/redundancy.py:23` | delete | unused import `http_completer` | 0 | none | a |
| 4 | `scripts/dataset/propose_questions.py:21` | delete | unused `import re` | −1 | none | a |
| 5 | `scripts/transcribe_openai.py:15,17` | delete | unused `import json`, `from collections import Counter` | −2 | none | a |
| 6 | `src/linkrag/generate/citations.py:23` | delete | unused `typing.Any` | 0 | none | a |
| 7 | `scripts/ask.py:97` | delete | `retrieved` built, never used (pure list comprehension) | −1 | none | a |
| 8 | `scripts/eval_deictic.py:47` | delete | `span` built, never used (pure dict comprehension) | −1 | none | a |
| 9 | `scripts/adapters/lectqa_vid.py:1472` | delete | `first` read from `reports/lectqa_vid_first_run.jsonl`, never used (pure read) | −1 | none | a |
| 10 | `scripts/adapters/lectqa_vid.py:259-260` | delete | `rouge1` alias of `token_f1`, never called (`run` uses `rouge1_raw`); its comment is also wrong | −3 | none | a |
| 11 | `scripts/adapters/lectqa_vid.py:467-473` | delete | `_hms`, superseded by `strict_seconds`, never called | −9 | none | a |
| 12 | `src/linkrag/link/relatedness.py:129-131` | delete | `is_gated_out`, never called anywhere (tests included) | −5 | none | a |
| 13 | `src/linkrag/ingest/audio.py:241,255` | delete | `ingest_audio.last_asr_meta` written twice, never read | −2 | none | a |
| 14 | `src/linkrag/link/figure_text.py:78-80` | yagni | `referenced_numbers`, superseded by the kind-aware `referenced`; only a test calls it (the test moves to `referenced`) | −4 | none | a |
| 15 | `configs/default.yaml:133,338-339,375-376` | dead-config | `ingest.ocr_images` (standalone images are always OCR'd; nothing reads the key), `datasets.mavils` (`repo`; the adapter hardcodes the path), `eval.metrics`, `eval.report_both_modes`: nothing reads them. The config is not hashed into any artefact. | −6 | none | a |
| 16 | `src/linkrag/eval/verify_gold.py:188-190` | dedupe | `_mmss` ≡ `generate.citations.mmss` (same rounding); import it | −3 | render (gold reports) | a |
| 17 | `src/linkrag/eval/redundancy.py:320-323` | dedupe | `dump_json` ≡ `verify_gold.dump_json`; `redundancy` already imports from `verify_gold` | −4 | none | a |
| 18 | `scripts/adapters/lectqa_vid.py` F541 ×2, `scripts/compare_retrieval.py:332`, `scripts/eval_deictic.py:219,241,242`, `scripts/negative_control.py:63` | shrink | f-strings without placeholders | 0 | render (same text) | a |
| 19 | `scripts/adapters/lectqa_vid.py:604-634` | dedupe | `localisation_table` is `gate_table` restricted to two rerank methods; one function with a `methods` argument | −12 | render (`lectqa_v2.md`, `lectqa_frameslides.md`) | b |

Estimated net: about −56 lines, 0 dependencies. Every change is in group (a) or (b). No performance
change is proposed (c): no timing shows a hotspot worth a measured before/after.

## KEEP (unused by default, or looks redundant, but is a recorded branch or instrument)

- `lectqa_vid.py` `run(metrics="ours")` with `_norm`, `token_f1`, `rouge1_raw` and bge-m3 similarity. This path produces `reports/lectqa_vid_first_run.md`, whose jsonl `v2` reads.
- `lectqa_vid.py` `temporal_links`: T1's own signal; the frame-slide comparison rests on it.
- `link/align.py` `path_score`, `total_score` (gate tests); `align_naive` (baseline ablation).
- Every recorded ablation switch: evict/additive expansion, nli entailment backend, mmr, fixed segmentation, build-deck groups, flatness scaling, the modality gate.
- `link/deictic.py` `Cue.strength`: documented as kept for reporting (the word-count heuristic that `tier` replaced).
- `retrieve/rerank.py` `_mmss`: it truncates where `citations.mmss` rounds. Unifying them would change cross-encoder input text.
- `retrieve_pool` in `compare_retrieval.py` and in `lectqa_vid.py`: the parameters differ (config-driven vs fixed LectQA settings).
- `scripts/dataset/*`, `scripts/eval/*`, `scripts/gate_links.py`, `scripts/negative_control.py`, `scripts/eval_deictic.py`, `scripts/transcribe_openai.py`: intake and eval instruments behind recorded reports.
- `configs/pilot01-w1.yaml`: the frozen pilot config; its copies of the dead keys stay.
- `mavils.py` nested `mean_paired` ×2: different signatures, and the MaViLS task that follows edits this file.

## Not applied, with reason

- **B007 unused loop variables** (`lectqa_vid.py:1185,1210`, `mavils.py:618,752,755,853`, `compare_retrieval.py:192`, `eval/redundancy.py:309`): renaming to `_` has zero line delta.
- **SIM115 bare `open()`** in short-lived scripts (`mavils.py:922,999`, `eval_alignment.py:27`, `eval_deictic.py:46,49,55`, `make_alignment_labels.py:41`, `plot_alignment.py:61`): a context manager does not shorten them and changes no behaviour.
- **Typing modernisation** (UP035/UP017/UP037, 30+ sites): style only, zero line delta, touches many files.
- **Stage name `link.figure`** (`figure_text.py:271`), stale since the rename to figure_text: renaming it breaks continuity with the timing tables already recorded under that key.
- **`tests/test_skeleton.py:45` builds a `Link` with the retired type `deictic_visual`**: `Link` does not validate types. Out of scope (tests), and see follow-up 3.

## Earlier audit (`audit_phase4.md` §C), rechecked

Resolved since then: the dead keys `audio_slide_threshold`, `same_topic_threshold`, `link_hops` and
`max_evidence_units`; `same_topic`; `scripts/demo.py`; the unused imports in `plot_alignment.py` and
`graph.py`; the stale docstring in `link/__init__.py`. Still present: the `link.figure` stage name (not
applied, above).

## Three follow-ups that need a real decision

1. **`scripts/adapters/mavils.py` (1,033 lines).** `study`, `final` and `fused` each re-implement the tune/test loop and a paired mean. A shared evaluator would cut roughly 150 lines, but every MaViLS report in `reports/` would then have to re-render byte-identical (long runs). Best decided inside the MaViLS visual task, which edits this file anyway.
2. **`scripts/adapters/lectqa_vid.py` (1,649 lines, 10 steps).** The first-run "ours" metric path is superseded by `linkrag.eval.lectqa_metrics`. Deleting it removes about 70 lines but makes `reports/lectqa_vid_first_run.md` non-reproducible. Is that report a reproducible artefact or history only? The adapter could also split into acquisition, localisation, answering and audit modules.
3. **`Link.link_type` is not validated.** Retired types such as `deictic_visual` are accepted silently. Validation would catch stale types, but it must be checked against every stored `links.jsonl` first.
