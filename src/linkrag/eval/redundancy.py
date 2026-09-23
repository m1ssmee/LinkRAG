"""Modality redundancy -- how much of one source is already stated by another.

Why this exists: on pilot01 only 5 of 16 proposed cross-modal questions survived
verification, because the paper restates the deck and the transcript narrates it.
Link-following can only help where modalities are *complementary*, so a candidate
lecture must be measured for redundancy before it enters the extended dataset.

Definition. For an ordered pair (source -> target):

    redundancy(source -> target) = fraction of source sentences that the judge says
                                   are entailed by the target's k nearest passages

"Nearest" is by dense similarity between the sentence and the target's units (the
same embedder the index uses). k is small (default 8) so a local judge can read it;
the ceiling is an entailment spread over more than k passages, which is scored as
not entailed. The judge is `eval.judge` (a different model from the answerer), run
`runs` times at temperature 0, majority vote.

Pairs reported by default: transcript->deck, transcript->paper, deck->transcript,
deck->paper. A high transcript->deck number means the speaker reads the slides; a
high deck->paper number means the deck is a summary of the paper. Both make
cross-modal questions collapse into single-source ones.
"""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from linkrag.core import EvidenceUnit
from linkrag.eval.verify_gold import JUDGE_SYSTEM, majority, parse_json, span_in_text
from linkrag.index import Encoder, embeddable_text

Completer = Callable[[str, str], str]

MIN_WORDS = 5          # shorter fragments (slide numbers, "Thank you.") are not claims
DEFAULT_PAIRS = (("transcript", "deck"), ("transcript", "paper"),
                 ("deck", "transcript"), ("deck", "paper"))

ENTAIL_PROMPT = """Sentence (from the {source_label}):
\"\"\"{sentence}\"\"\"

Passages (from the {target_label}):
{passages}

Is the information in the Sentence stated by the Passages -- would a reader of the
Passages already know what the Sentence says? Paraphrase counts; the same numbers or
names in a different form count. Extra detail in the Sentence that the Passages do
not contain means "no".

Output JSON only: {{"entailed": true|false, "span": "<exact quote from one Passage that
states it, or empty>"}}"""


# ----------------------------------------------------------------- sentences

_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")
_BULLET = re.compile(r"\s*[•·▪◦\-–]\s+|\n+")


def split_sentences(text: str, *, bullets: bool = False, min_words: int = MIN_WORDS) -> list[str]:
    """Sentences (or bullet items for slide text) with at least `min_words` words."""
    text = " ".join(text.split()) if not bullets else text
    parts: list[str] = []
    chunks = _BULLET.split(text) if bullets else [text]
    for chunk in chunks:
        chunk = " ".join(chunk.split())
        if not chunk:
            continue
        parts.extend(p.strip() for p in _SENT.split(chunk) if p.strip())
    return [p for p in parts if len(p.split()) >= min_words]


def role_of(unit: EvidenceUnit) -> str:
    if unit.modality == "audio":
        return "transcript"
    return "deck" if unit.metadata.get("slide_deck") else "paper"


def sentences_by_role(units: Sequence[EvidenceUnit]) -> dict[str, list[tuple[str, str]]]:
    """role -> [(sentence, unit_id)], in reading order, deduplicated within a role."""
    out: dict[str, list[tuple[str, str]]] = {"transcript": [], "deck": [], "paper": []}
    seen: dict[str, set[str]] = {r: set() for r in out}

    def order(u: EvidenceUnit):
        loc = u.location
        return (loc.start_s if loc.start_s is not None else -1,
                loc.page if loc.page is not None else -1, u.id)

    for u in sorted(units, key=order):
        role = role_of(u)
        # deck figure OCR is noise-heavy and mostly duplicates the slide text; the
        # deck *text* units are the claims a deck makes. Figures still serve as
        # targets (see passages_for) so OCR'd labels can entail a spoken sentence.
        if role == "deck" and u.modality != "text":
            continue
        for s in split_sentences(u.content, bullets=(role == "deck")):
            key = re.sub(r"\W+", " ", s.lower()).strip()
            if key in seen[role]:
                continue
            seen[role].add(key)
            out[role].append((s, u.id))
    return out


# ----------------------------------------------------------------- retrieval

class TargetIndex:
    """Dense nearest-passage lookup over one role's units."""

    def __init__(self, units: Sequence[EvidenceUnit], encoder: Encoder):
        self.units = list(units)
        texts = [embeddable_text(u) for u in self.units]
        self.matrix = np.asarray(encoder(texts), dtype=np.float32) if texts else np.zeros((0, 1))
        self.encoder = encoder

    def nearest(self, sentence: str, k: int) -> list[EvidenceUnit]:
        if not self.units:
            return []
        q = np.asarray(self.encoder([sentence]), dtype=np.float32)[0]
        scores = self.matrix @ q
        top = np.argsort(-scores, kind="stable")[:k]
        return [self.units[i] for i in top]


def passages_for(units: Sequence[EvidenceUnit]) -> str:
    return "\n\n".join(f"[{i + 1}] {' '.join(u.content.split())}" for i, u in enumerate(units))


# ----------------------------------------------------------------- judging

@dataclass
class SentenceVerdict:
    source: str
    target: str
    sentence: str
    unit_id: str
    entailed: bool
    votes: list[str]
    span: str
    passages: list[str]


LABEL = {"transcript": "lecture transcript", "deck": "slide deck (text and figure OCR)",
         "paper": "paper"}


def judge_sentence(sentence: str, unit_id: str, source: str, target: str,
                   passages: Sequence[EvidenceUnit], judge: Completer, runs: int,
                   backend: str = "nli", device: str = "cpu") -> SentenceVerdict:
    """Is this sentence of A already stated by B's nearest passages?

    `nli` (default): the same local cross-encoder the gold and claim checks use --
    deterministic, free, one pass per passage with the best taken. `llm`: the judge
    model, `runs` calls, majority, quotable span."""
    if backend == "nli":
        from linkrag.eval import nli as _nli
        ok, span, _ = _nli.entails_any("\n".join(u.content for u in passages), [sentence], device=device)
        return SentenceVerdict(source, target, sentence, unit_id, bool(ok),
                               ["yes" if ok else "no"], span if ok else "", [u.id for u in passages])
    if backend != "llm":
        raise ValueError(f"unknown entailment backend {backend!r}: use 'nli' or 'llm'")
    prompt = ENTAIL_PROMPT.format(source_label=LABEL[source], target_label=LABEL[target],
                                  sentence=sentence, passages=passages_for(passages))
    votes, spans = [], []
    hay = "\n".join(u.content for u in passages)
    for _ in range(runs):
        reply = parse_json(judge(JUDGE_SYSTEM, prompt))
        ok = bool(reply.get("entailed", False))
        span = str(reply.get("span", "") or "")
        if ok and not span_in_text(span, hay):
            votes.append("yes-unquoted")       # counts as no, as in verify_gold
            continue
        votes.append("yes" if ok else "no")
        if ok:
            spans.append(span)
    return SentenceVerdict(source, target, sentence, unit_id, majority(votes, "yes"), votes,
                           spans[0] if spans else "", [u.id for u in passages])


def redundancy(units: Sequence[EvidenceUnit], encoder: Encoder, judge: Completer, *,
               pairs: Sequence[tuple[str, str]] = DEFAULT_PAIRS, k: int = 8, runs: int = 3,
               workers: int = 6, limit: int | None = None, backend: str = "nli",
               device: str = "cpu",
               progress: Callable[[str], None] | None = None) -> list[SentenceVerdict]:
    by_role = sentences_by_role(units)
    targets = {role: TargetIndex([u for u in units if role_of(u) == role], encoder)
               for role in {t for _, t in pairs}}
    out: list[SentenceVerdict] = []
    for source, target in pairs:
        sents = by_role[source][:limit] if limit else by_role[source]
        if not sents or not targets[target].units:
            if progress:
                progress(f"{source}->{target}: skipped (no sentences or no target units)")
            continue
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(judge_sentence, s, uid, source, target,
                                 targets[target].nearest(s, k), judge, runs, backend, device)
                       for s, uid in sents]
            verdicts = [f.result() for f in futures]
        out.extend(verdicts)
        if progress:
            n = sum(v.entailed for v in verdicts)
            progress(f"{source}->{target}: {n}/{len(verdicts)} = {n / len(verdicts):.1%} entailed")
    return out


# ----------------------------------------------------------------- reporting

def summarise(verdicts: Sequence[SentenceVerdict]) -> dict[str, Any]:
    pairs: dict[str, dict[str, Any]] = {}
    for v in verdicts:
        key = f"{v.source}->{v.target}"
        d = pairs.setdefault(key, {"source": v.source, "target": v.target, "n": 0, "entailed": 0})
        d["n"] += 1
        d["entailed"] += int(v.entailed)
    for d in pairs.values():
        d["fraction"] = d["entailed"] / d["n"] if d["n"] else float("nan")
    n = sum(d["n"] for d in pairs.values())
    e = sum(d["entailed"] for d in pairs.values())
    return {"pairs": pairs, "overall": {"n": n, "entailed": e, "fraction": e / n if n else float("nan")}}


def write_report(verdicts: Sequence[SentenceVerdict], path: str | Path, *, corpus: str,
                 manifest_hash: str, judge_model: str, k: int, runs: int,
                 files: dict[str, list[str]], sample: int = 8) -> Path:
    summ = summarise(verdicts)
    L = [f"# Modality redundancy — {corpus}", "",
         f"Corpus `{manifest_hash}` · judge `{judge_model}` · temperature 0 · {runs} runs, "
         f"majority · k = {k} nearest target passages per sentence · "
         f"generated by `linkrag.eval.redundancy`", "",
         "**Reading it.** `A->B` is the fraction of A's sentences a reader of B already knows. "
         "High transcript->deck: the speaker reads the slides. High deck->paper: the deck "
         "summarises the paper. Cross-modal questions survive only where these are low.", "",
         "| role | files |", "|---|---|"]
    for role, names in files.items():
        L.append(f"| {role} | {', '.join(names) or '—'} |")
    L += ["", "| pair | sentences | entailed | redundancy |", "|---|---:|---:|---:|"]
    for key, d in summ["pairs"].items():
        L.append(f"| {key} | {d['n']} | {d['entailed']} | **{d['fraction']:.1%}** |")
    o = summ["overall"]
    L.append(f"| **overall** | {o['n']} | {o['entailed']} | **{o['fraction']:.1%}** |")
    L += ["", f"Counts per pair sum to the overall row (measurement rule 2): "
          f"{sum(d['n'] for d in summ['pairs'].values())} = {o['n']}.", ""]
    L += ["## What each source says that the others do not", "",
          "A sample of sentences judged *not* entailed — the complementary content a "
          "cross-modal question can be built on.", ""]
    for key, d in summ["pairs"].items():
        uniq = [v for v in verdicts if f"{v.source}->{v.target}" == key and not v.entailed]
        L += [f"### {key} — {len(uniq)} not entailed", ""]
        for v in uniq[:sample]:
            L.append(f"- `{v.unit_id}` {v.sentence}")
        L.append("")
    path = Path(path)
    path.write_text("\n".join(L) + "\n")
    return path


def dump_json(verdicts: Sequence[SentenceVerdict], path: str | Path) -> Path:
    path = Path(path)
    path.write_text(json.dumps([asdict(v) for v in verdicts], ensure_ascii=False, indent=1))
    return path
