"""Automated gold verification -- priority (i) in DESIGN.md.

Replaces the ear-verified gold workflow. No human step except the sampled audit
(`scripts/eval/audit_sample.py`). Two checks, both LLM-judged at temperature 0:

(a) **Unit entailment.** For every proposed gold unit: "does this text contain
    information needed to answer the question? yes/no + supporting span". Run
    `runs` times, majority vote. A `yes` whose quoted span cannot be found in the
    unit text is counted as `no` -- an unquotable yes is the judge guessing.
    Failing units are dropped from gold with the reason logged.

(b) **Cross-modal requirement.** The question is answered from each *single
    source* in full -- the whole transcript; all slide text + figure OCR; the whole
    paper -- and from all three together. Each answer is graded by the judge against
    the reference answer (`runs` times, majority). A question keeps a cross-modal or
    deictic type only if **every** single-source run fails and the all-sources run
    succeeds; otherwise it is relabelled with the source that succeeded. A question
    that fails even with everything is dropped.

**Reference answers must list only the facts the question asks for.** The grader
requires every fact in the reference, so an elaboration ("…, the dashed line in
Figure 6") turns a correct answer into a FAIL. Run 1 on pilot01 lost 5 questions
to exactly that before the references were trimmed.

Two models: the **answerer** (`models.llm`) writes the modality-only answers; the
**judge** (`eval.judge`, a different model) does every yes/no call -- unit entailment
and answer grading. Same-model self-grading is lenient; the sampled human audit bounds
whatever bias remains.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from linkrag.core import EvidenceUnit
from linkrag.eval.metrics import matches_locator

Completer = Callable[[str, str], str]

SOURCES = ("audio", "deck", "paper")
SOURCE_TYPE = {"audio": "audio_only", "deck": "slides_only", "paper": "paper_only"}
CROSS_MODAL_TYPES = {"cross_modal_split", "cross_modal_deictic", "deictic"}

JUDGE_SYSTEM = (
    "You are a strict grader for a question-answering benchmark. You only output "
    "JSON. You never use outside knowledge: judge solely from the text you are given."
)

ENTAIL_PROMPT = """Question: {question}
Reference answer: {expected}

Text:
\"\"\"{text}\"\"\"

Step 1. Split the Reference answer into its distinct facts.
Step 2. For each fact, does the Text state it (verbatim or paraphrased)? If yes, quote
the exact words of the Text that state it.

Verdict is "yes" if the Text states AT LEAST ONE of the facts. A unit of evidence may
carry only part of the answer -- the other part can live in another file -- and that
partial unit is still gold. Verdict is "no" only if the Text states none of the facts.

Output JSON only: {{"facts": [{{"fact": "...", "supported": true|false, "span": "<exact
quote from the Text, or empty>"}}], "verdict": "yes" or "no"}}"""

ANSWER_SYSTEM = (
    "Answer the question using ONLY the context. If the context does not contain the "
    "information, reply exactly: NOT ANSWERABLE. Be concise and specific."
)
ANSWER_PROMPT = "Context ({label}):\n\n{context}\n\nQuestion: {question}\n\nAnswer:"

GRADE_PROMPT = """Question: {question}
Reference answer: {expected}
Candidate answer: {candidate}

Split the Reference answer into its distinct required facts. For each, say whether the
Candidate answer states it (paraphrase is fine; a contradiction is "no"; NOT ANSWERABLE
states nothing). Verdict is PASS only if every required fact is present.

Output JSON only: {{"facts": [{{"fact": "...", "present": true|false}}],
"verdict": "PASS" or "FAIL", "reason": "<one sentence>"}}"""


# ----------------------------------------------------------------- results

@dataclass
class UnitVerdict:
    unit_id: str
    kept: bool
    votes: list[str]              # per run: yes / no / yes-unquoted
    span: str
    reason: str
    location: str
    modality: str
    text: str


@dataclass
class SourceRun:
    source: str                   # audio | deck | paper | all
    answer: str
    passed: bool
    votes: list[str]              # PASS / FAIL per judge run
    reason: str
    missing: list[str]


@dataclass
class QuestionVerdict:
    qid: str
    question: str
    expected: str
    original_type: str
    verified_type: str | None      # None => dropped
    status: str                    # kept | relabelled | dropped:<reason>
    answerable_from: list[str]
    units: list[UnitVerdict]
    runs: list[SourceRun]
    gold_terms: dict[str, list[str]] = field(default_factory=dict)

    @property
    def kept_units(self) -> list[UnitVerdict]:
        return [u for u in self.units if u.kept]


# ----------------------------------------------------------------- helpers

def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def span_in_text(span: str, text: str) -> bool:
    """Normalised substring match, tolerating the judge's '...' elisions.

    A span with an ellipsis is checked piecewise, in order. Anything else must be a
    contiguous normalised substring. Empty spans never match.
    """
    span = span.strip()
    if not span:
        return False
    hay = _norm(text)
    pieces = [p for p in re.split(r"\.\.\.|…", span) if _norm(p)]
    if not pieces:
        return False
    pos = 0
    for piece in pieces:
        idx = hay.find(_norm(piece), pos)
        if idx < 0:
            return False
        pos = idx + len(_norm(piece))
    return True


def parse_json(text: str) -> dict[str, Any]:
    """The model is told 'JSON only'; tolerate a fenced or prefixed reply anyway."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-z]*\s*|\s*```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < 0:
        return {}
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return {}


def majority(votes: Sequence[str], positive: str) -> bool:
    return sum(v == positive for v in votes) * 2 > len(votes)


def locator_str(unit: EvidenceUnit) -> str:
    loc = unit.location
    name = Path(unit.source_file).name
    if loc.page is not None:
        return f"{name} p.{loc.page}"
    if loc.start_s is not None:
        return f"{name} {_mmss(loc.start_s)}-{_mmss(loc.end_s or loc.start_s)}"
    return name


def _mmss(s: float) -> str:
    s = int(round(s))
    return f"{s // 60}:{s % 60:02d}"


def source_of(unit: EvidenceUnit, deck_files: set[str]) -> str:
    if unit.modality == "audio":
        return "audio"
    return "deck" if Path(unit.source_file).name in deck_files else "paper"


def full_contexts(units: Sequence[EvidenceUnit], deck_files: set[str]) -> dict[str, str]:
    """One string per source in reading order, plus 'all' = the three concatenated."""
    groups: dict[str, list[EvidenceUnit]] = {s: [] for s in SOURCES}
    for u in units:
        groups[source_of(u, deck_files)].append(u)

    def order(u: EvidenceUnit):
        loc = u.location
        return (loc.start_s if loc.start_s is not None else -1,
                loc.page if loc.page is not None else -1, u.id)

    out = {}
    for src in SOURCES:
        parts = [f"[{locator_str(u)} · {u.modality}]\n{u.content.strip()}"
                 for u in sorted(groups[src], key=order) if u.content.strip()]
        out[src] = "\n\n".join(parts)
    out["all"] = "\n\n".join(f"===== {src.upper()} =====\n{out[src]}" for src in SOURCES)
    return out


# ----------------------------------------------------------------- checks

def entail_unit(question: str, expected: str, unit: EvidenceUnit, complete: Completer,
                runs: int) -> UnitVerdict:
    """Majority over `runs` judge calls. A run counts as yes only if at least one fact it
    marks supported comes with a span that is actually quotable from the unit."""
    votes, spans = [], []
    prompt = ENTAIL_PROMPT.format(question=question, expected=expected, text=unit.content)
    for _ in range(runs):
        reply = parse_json(complete(JUDGE_SYSTEM, prompt))
        verdict = str(reply.get("verdict", "no")).lower().strip()
        facts = [f for f in reply.get("facts", []) if isinstance(f, dict)]
        quotable = [str(f.get("span", "") or "") for f in facts
                    if f.get("supported") and span_in_text(str(f.get("span", "") or ""), unit.content)]
        if verdict == "yes" and not quotable:
            verdict = "yes-unquoted"      # counts as no
        votes.append(verdict)
        if verdict == "yes":
            spans.append(quotable[0])
    kept = majority(votes, "yes")
    if kept:
        reason = "majority yes with quotable span"
    elif votes.count("yes-unquoted") + votes.count("yes") > len(votes) // 2:
        reason = "judge said yes but could not quote a supporting span"
    else:
        reason = "judge: text states none of the reference facts"
    return UnitVerdict(unit_id=unit.id, kept=kept, votes=votes,
                       span=spans[0] if spans else "", reason=reason,
                       location=locator_str(unit), modality=unit.modality,
                       text=unit.content)


def grade(question: str, expected: str, candidate: str, complete: Completer,
          runs: int) -> tuple[bool, list[str], str, list[str]]:
    votes, reasons, missing = [], [], []
    prompt = GRADE_PROMPT.format(question=question, expected=expected, candidate=candidate)
    for _ in range(runs):
        reply = parse_json(complete(JUDGE_SYSTEM, prompt))
        v = str(reply.get("verdict", "FAIL")).upper().strip()
        votes.append("PASS" if v == "PASS" else "FAIL")
        reasons.append(str(reply.get("reason", "")))
        missing.append([f.get("fact", "") for f in reply.get("facts", [])
                        if isinstance(f, dict) and not f.get("present", False)])
    passed = majority(votes, "PASS")
    # report the reason/missing list of the majority side
    side = [i for i, v in enumerate(votes) if (v == "PASS") == passed]
    return passed, votes, reasons[side[0]], missing[side[0]]


def source_run(source: str, question: str, expected: str, context: str,
               complete: Completer, runs: int, judge: Completer | None = None) -> SourceRun:
    judge = judge or complete
    if not context.strip():
        return SourceRun(source, "NOT ANSWERABLE (empty source)", False, ["FAIL"] * runs,
                         "source has no content", [])
    label = {"audio": "the full lecture transcript",
             "deck": "all slide text and figure OCR from the deck",
             "paper": "the full paper text and figure captions",
             "all": "transcript + deck + paper"}[source]
    answer = complete(ANSWER_SYSTEM, ANSWER_PROMPT.format(label=label, context=context,
                                                           question=question)).strip()
    passed, votes, reason, missing = grade(question, expected, answer, judge, runs)
    return SourceRun(source, answer, passed, votes, reason, missing)


def relabel(original_type: str, runs: dict[str, SourceRun]) -> tuple[str | None, str, list[str]]:
    """(verified_type, status, answerable_from). None type => dropped."""
    single_ok = [s for s in SOURCES if runs[s].passed]
    all_ok = runs["all"].passed
    if not all_ok and not single_ok:
        return None, "dropped: not answerable from the full corpus", []
    if single_ok:
        if len(single_ok) == 1:
            new_type = SOURCE_TYPE[single_ok[0]]
        else:
            new_type = "single_modality"
        status = "kept" if new_type == original_type else f"relabelled from {original_type}"
        return new_type, status, single_ok
    # every single source failed, all sources passed => genuinely cross-modal
    new_type = original_type if original_type in CROSS_MODAL_TYPES else "cross_modal_split"
    status = "kept" if new_type == original_type else f"relabelled from {original_type}"
    return new_type, status, ["all"]


# ----------------------------------------------------------------- driver

def verify_question(row: dict[str, Any], units: Sequence[EvidenceUnit],
                    contexts: dict[str, str], complete: Completer, *, runs: int,
                    workers: int, judge: Completer) -> QuestionVerdict:
    question, expected = row["question"], row["expected_answer"]
    candidates = [u for u in units
                  if any(matches_locator(u, loc) for loc in row["gold_units"])]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        unit_futures = [ex.submit(entail_unit, question, expected, u, judge, runs)
                        for u in candidates]
        run_futures = {s: ex.submit(source_run, s, question, expected, contexts[s],
                                    complete, runs, judge) for s in (*SOURCES, "all")}
        unit_verdicts = [f.result() for f in unit_futures]
        source_runs = {s: f.result() for s, f in run_futures.items()}

    new_type, status, answerable = relabel(row["type"], source_runs)
    if new_type is not None and not any(u.kept for u in unit_verdicts):
        new_type, status = None, "dropped: no proposed gold unit survived entailment"
    return QuestionVerdict(
        qid=row["qid"], question=question, expected=expected,
        original_type=row["type"], verified_type=new_type, status=status,
        answerable_from=answerable, units=unit_verdicts,
        runs=[source_runs[s] for s in (*SOURCES, "all")],
        gold_terms=row.get("gold_terms", {"slide": [], "audio": []}))


def verify_gold(rows: Sequence[dict[str, Any]], units: Sequence[EvidenceUnit],
                complete: Completer, *, judge: Completer, deck_files: set[str],
                runs: int = 3, workers: int = 6,
                progress: Callable[[str], None] | None = None) -> list[QuestionVerdict]:
    """`complete` answers (models.llm); `judge` grades and checks entailment
    (eval.judge). Pass the same callable for both only in a test."""
    contexts = full_contexts(units, deck_files)
    out = []
    for row in rows:
        v = verify_question(row, units, contexts, complete, runs=runs, workers=workers,
                            judge=judge)
        if progress:
            kept = sum(u.kept for u in v.units)
            progress(f"{v.qid:<4} {v.original_type:<20} -> {v.verified_type or 'DROPPED':<20} "
                     f"units {kept}/{len(v.units)}  "
                     + " ".join(f"{r.source}:{'✓' if r.passed else '✗'}" for r in v.runs))
        out.append(v)
    return out


# ----------------------------------------------------------------- outputs

def verified_gold_rows(verdicts: Sequence[QuestionVerdict], units: Sequence[EvidenceUnit]
                       ) -> list[dict[str, Any]]:
    """Regression-format rows. Locators are narrowed to the surviving units so a
    page with three units of which one passed contributes one locator; unit ids are
    also stamped for exact matching against this manifest."""
    by_id = {u.id: u for u in units}
    rows = []
    for v in verdicts:
        if v.verified_type is None:
            continue
        locs, ids, mods = [], [], []
        for uv in v.kept_units:
            u = by_id[uv.unit_id]
            loc: dict[str, Any] = {"source": Path(u.source_file).name,
                                   "id_at_verification": u.id}
            if u.location.page is not None:
                loc["page"] = u.location.page
            else:
                loc["start_s"] = round(u.location.start_s or 0.0, 1)
                loc["end_s"] = round(u.location.end_s or 0.0, 1)
            locs.append(loc)
            ids.append(u.id)
            mods.append(u.modality)
        rows.append({
            "qid": v.qid, "type": v.verified_type, "question": v.question,
            "expected_answer": v.expected,
            "gold_modalities": sorted(set(mods), key=["text", "figure", "table", "audio"].index),
            "gold_units": locs, "gold_unit_ids": ids,
            "gold_terms": v.gold_terms,
            "verification": {"status": v.status, "answerable_from": v.answerable_from,
                             "original_type": v.original_type,
                             "units_proposed": len(v.units), "units_kept": len(ids)},
        })
    return rows


def write_gold(rows: Sequence[dict[str, Any]], path: str | Path, *, manifest_hash: str,
               model: str, runs: int, source: str, answerer: str = "") -> Path:
    from datetime import datetime, timezone
    path = Path(path)
    meta = {"_meta": {
        "manifest_hash": manifest_hash,
        "verified_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "verifier": "linkrag.eval.verify_gold", "judge_model": model,
        "answerer_model": answerer, "judge_runs": runs,
        "proposed_from": source,
        "note": "Gold is machine-verified: unit entailment (majority of judge runs, "
                "quotable span required) and modality-only full-context answering. "
                "Human involvement is limited to the sampled audit sheet.",
    }}
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in [meta, *rows]) + "\n")
    return path


def _short(s: str, n: int = 160) -> str:
    s = " ".join(s.split())
    return s if len(s) <= n else s[:n - 1] + "…"


def write_report(verdicts: Sequence[QuestionVerdict], path: str | Path, *, corpus: str,
                 manifest_hash: str, model: str, runs: int, n_units: int,
                 answerer: str = "") -> Path:
    path = Path(path)
    kept = [v for v in verdicts if v.verified_type is not None]
    relabelled = [v for v in kept if v.status.startswith("relabelled")]
    dropped = [v for v in verdicts if v.verified_type is None]
    units_all = [u for v in verdicts for u in v.units]
    L = [f"# Gold verification — {corpus}", "",
         f"Corpus manifest `{manifest_hash}` ({n_units} units) · judge `{model}` · "
         f"answerer `{answerer or model}` · temperature 0 · {runs} runs per check, "
         f"majority vote · generated by `linkrag.eval.verify_gold`", "",
         "**Checks.** (a) per-unit entailment with a quotable span; (b) the question "
         "answered from each single source in full (transcript / deck text + figure OCR / "
         "paper) and from all three, graded against the reference answer. Cross-modal and "
         "deictic labels survive only if every single-source run fails and the all-source "
         "run passes. The judge is a different model from the answerer; the sampled human "
         "audit bounds whatever bias remains.", "",
         "## Summary", "",
         f"- questions proposed: **{len(verdicts)}** · kept: **{len(kept)}** "
         f"(relabelled: {len(relabelled)}) · dropped: **{len(dropped)}**",
         f"- gold units proposed: **{len(units_all)}** · kept: "
         f"**{sum(u.kept for u in units_all)}** · dropped: {sum(not u.kept for u in units_all)}",
         "- type counts after verification: " + ", ".join(
             f"{t} {n}" for t, n in sorted(Counter(v.verified_type for v in kept).items())),
         "", "| Q | proposed type | verified type | status | units kept | audio | deck | paper | all |",
         "|---|---|---|---|---:|:-:|:-:|:-:|:-:|"]
    for v in verdicts:
        marks = {r.source: ("✓" if r.passed else "✗") for r in v.runs}
        L.append(f"| {v.qid} | {v.original_type} | {v.verified_type or '—'} | {v.status} | "
                 f"{len(v.kept_units)}/{len(v.units)} | {marks['audio']} | {marks['deck']} | "
                 f"{marks['paper']} | {marks['all']} |")
    L += ["", "Per-type counts above must sum to the kept total (measurement rule 2): "
          f"{sum(Counter(v.verified_type for v in kept).values())} = {len(kept)}.", ""]

    L += ["## Per-question detail", ""]
    for v in verdicts:
        L += [f"### {v.qid} · {v.original_type} → {v.verified_type or 'DROPPED'}", "",
              f"**Q.** {v.question}  ", f"**Reference.** {v.expected}  ",
              f"**Status.** {v.status}; answerable from: {', '.join(v.answerable_from) or '—'}", "",
              "| source | verdict | votes | judge reason | answer (truncated) |", "|---|:-:|---|---|---|"]
        for r in v.runs:
            L.append(f"| {r.source} | {'PASS' if r.passed else 'FAIL'} | {'/'.join(r.votes)} | "
                     f"{_short(r.reason, 120)} | {_short(r.answer, 200)} |")
        L += ["", "| unit | location | kept | votes | span / reason |", "|---|---|:-:|---|---|"]
        for u in v.units:
            L.append(f"| `{u.unit_id}` | {u.location} | {'✓' if u.kept else '✗'} | "
                     f"{'/'.join(u.votes)} | {_short(u.span if u.kept else u.reason, 140)} |")
        L.append("")
    path.write_text("\n".join(L) + "\n")
    return path


def dump_json(verdicts: Sequence[QuestionVerdict], path: str | Path) -> Path:
    path = Path(path)
    path.write_text(json.dumps([asdict(v) for v in verdicts], ensure_ascii=False, indent=1))
    return path
