"""Local NLI entailment -- an ABLATION backend for verification, not the verifier.

`cross-encoder/nli-deberta-v3-base` scores (premise, hypothesis) into
contradiction / entailment / neutral. It can stand in for the LLM judge in all three
places that ask an entailment question:

* gold verification -- does this unit state a fact of the reference answer?
* claim verification -- does the cited unit state this claim?
* redundancy        -- does source B already state this sentence of source A?

What it offers: deterministic (one forward pass, so the LLM's 3-run majority collapses
to one run), free, local. What it gives up: no reasoning, no explanation, and a hard
input limit -- a long unit is scored sentence by sentence and the best sentence wins,
which is also what produces the quotable span.

Measured, and rejected as the gold / intake verifier (DESIGN.md finding 11,
`results/nli_vs_llm_pilot01.md`): kappa 0.31/0.36 against LLM judges that agree with
each other at 0.85. The scripts refuse to let it write gold or the stored reports.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Sequence

import numpy as np

MODEL = "cross-encoder/nli-deberta-v3-base"
ENTAIL_MARGIN = 0.0        # entailment logit must beat both others by this much
MAX_PREMISE_CHARS = 1200   # per sentence group handed to the model
MIN_PREMISE_WORDS = 4      # "4.2." or a lone reference number is not a premise
_SPLIT = re.compile(r"(?<=[.!?])\s+")
# a "sentence" ending in one of these is an initial or abbreviation ("Phillip B.",
# "et al.", "Fig."), not a sentence end: splitting there broke author lists into
# fragments nothing could entail
_ABBR = re.compile(r"\b([A-Z]|[A-Z][a-z]{0,3}|e\.g|i\.e|et al|vs|No)\.$")


@lru_cache(maxsize=2)
def _model(name: str = MODEL, device: str = "cpu"):
    from sentence_transformers import CrossEncoder
    return CrossEncoder(name, device=device)


def _entail_index(model) -> int:
    """Label order differs between NLI checkpoints; read it, never assume it."""
    labels = {str(v).lower(): int(k) for k, v in model.model.config.id2label.items()}
    return labels["entailment"]


def sentences(text: str, max_chars: int = MAX_PREMISE_CHARS) -> list[str]:
    """Premise candidates: sentences, with over-long ones hard-wrapped."""
    out: list[str] = []
    for part in _join_abbrev(_SPLIT.split(" ".join(str(text).split()))):
        while len(part) > max_chars:
            out.append(part[:max_chars])
            part = part[max_chars:]
        if part:
            out.append(part)
    return out or [""]


def _join_abbrev(parts: list[str]) -> list[str]:
    out, cur = [], ""
    for part in parts:
        cur = f"{cur} {part.strip()}".strip()
        if not _ABBR.search(cur):
            out.append(cur)
            cur = ""
    return out + ([cur] if cur else [])


def entails(premise_text: str, hypothesis: str, *, device: str = "cpu",
            margin: float = ENTAIL_MARGIN, model: str = MODEL) -> tuple[bool, str, float]:
    """(entailed, the premise sentence that carried it, its entailment logit).

    Scored per premise sentence and maximised, so a 30-second transcript unit is not
    diluted by the clauses that have nothing to do with the hypothesis -- and the
    winning sentence is returned as the span, which is what makes a `yes` checkable.
    """
    hypothesis = " ".join(str(hypothesis).split())
    if not hypothesis:
        return False, "", float("-inf")
    cands = [c for c in sentences(premise_text) if len(c.split()) >= MIN_PREMISE_WORDS] or sentences(premise_text)
    m = _model(model, device)
    e = _entail_index(m)
    scores = np.asarray(m.predict([(c, hypothesis) for c in cands]), dtype="float32")
    if scores.ndim == 1:
        scores = scores.reshape(1, -1)
    ent = scores[:, e]
    best = int(np.argmax(ent))
    others = np.max(np.delete(scores[best], e))
    return bool(ent[best] - others > margin), cands[best], float(ent[best])


def entails_any(premise_text: str, hypotheses: Sequence[str], **kw) -> tuple[bool, str, list[bool]]:
    """True when ANY hypothesis is entailed -- the gold rule: a unit that carries one
    fact of a multi-fact answer is still gold."""
    flags, span = [], ""
    for h in hypotheses:
        ok, s, _ = entails(premise_text, h, **kw)
        flags.append(ok)
        if ok and not span:
            span = s
    return any(flags), span, flags


def entails_all(premise_text: str, hypotheses: Sequence[str], **kw) -> tuple[bool, list[bool]]:
    """True when EVERY hypothesis is entailed -- the grading rule: an answer passes
    only if it states all the required facts."""
    flags = [entails(premise_text, h, **kw)[0] for h in hypotheses]
    return (all(flags) if flags else False), flags


def facts(reference: str, question: str = "") -> list[str]:
    """Split a reference answer into checkable facts. The LLM backend asks the model to
    do this; here it is sentence and clause splitting, which is what the references are
    written as (`expected_answer` lists only the asked facts -- see verify_gold).

    With `question`, each fact is prefixed by it: a reference answer is usually a noun
    phrase ("NoScope, published in PVLDB in 2017"), which is not a proposition an NLI
    model can test; question + answer is. Measured on pilot01: kappa vs the LLM judge
    0.27 -> 0.31/0.36 (results/nli_vs_llm_pilot01.md)."""
    out: list[str] = []
    for sentence in sentences(reference):
        parts = [p.strip(" ;") for p in re.split(r";\s+", sentence) if p.strip(" ;")]
        out.extend(parts or [sentence])
    out = [f for f in out if len(f.split()) >= 2] or [" ".join(str(reference).split())]
    q = " ".join(str(question).split())
    return [f"{q} {f}" for f in out] if q else out
