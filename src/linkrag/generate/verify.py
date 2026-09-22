"""Claim-level verification -- Phase 6, priority (v).

Phase 1 produced the case this exists for: an answer that cited a real unit and was
still false ("Saurabh Bakshi from Purdue" as an author of Focus -- an audience member
introducing himself, sitting in the same audio unit as the talk's closing line). An
id-level citation check passes that answer. A claim-level check does not.

Each claim from `generate.answer.answer_json` is checked against **the text of the
units it cites**, with `linkrag.eval.verify_gold.entail_unit` -- the same prompt,
the same 3-run majority, the same quotable-span requirement used to verify gold. It
is imported, not copied: if that rule changes, both move together.

Verdicts:
  supported   -- a cited unit entails the claim
  weak        -- the claim cites nothing (the model asserted it without evidence)
  unsupported -- every cited unit was checked and none entails it

`generation.strict` drops unsupported claims from what the reader sees; when no claim
survives, the answer abstains with "not found in the provided material".
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from typing import Any, Callable, Sequence

from linkrag.core import EvidenceUnit, stage_timer
from linkrag.eval.verify_gold import entail_unit
from linkrag.generate.answer import Answer, Claim

Completer = Callable[[str, str], str]

ABSTENTION = "not found in the provided material"


def verify_claim(claim: Claim, units: dict[str, EvidenceUnit], judge: Completer, *, runs: int = 3) -> Claim:
    """Entailment of one claim against each unit it cites; first supporting unit wins."""
    if not claim.unit_ids:
        claim.verdict = "weak"
        return claim
    for uid in claim.unit_ids:
        unit = units.get(uid)
        if unit is None:                       # id not in the evidence set
            continue
        v = entail_unit(claim.claim, claim.claim, unit, judge, runs)
        claim.votes = v.votes
        if v.kept:
            claim.verdict, claim.span = "supported", v.span
            return claim
    claim.verdict = "unsupported"
    return claim


def verify_answer(ans: Answer, units: Sequence[EvidenceUnit], judge: Completer, *,
                  runs: int = 3, workers: int = 4, strict: bool = False) -> dict[str, Any]:
    """Verify every claim; returns the claim verdicts and the text to show.

    `strict` removes unsupported claims from the displayed answer and abstains when
    nothing is left. The unverified answer is always kept in the result so a report
    can show what the model said before the check."""
    by_id = {u.id: u for u in units}
    with stage_timer("generate.verify", claims=len(ans.claims), strict=int(strict)) as t:
        if ans.claims:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                ans.claims = list(ex.map(lambda c: verify_claim(c, by_id, judge, runs=runs), ans.claims))
        counts = {v: sum(c.verdict == v for c in ans.claims) for v in ("supported", "weak", "unsupported")}
        t.update({k: v for k, v in counts.items()})

    shown = ans.answer
    if strict:
        kept = [c for c in ans.claims if c.verdict == "supported"]
        if ans.claims and not kept:
            shown = ABSTENTION
    return {"answer": shown, "answer_unverified": ans.answer, "malformed": ans.malformed,
            "claims": [asdict(c) for c in ans.claims], **counts,
            "hallucination_rate": (counts["unsupported"] / len(ans.claims)) if ans.claims else 0.0,
            "abstained": shown == ABSTENTION}


# ----------------------------------------------------------------- metrics

def hallucination_rate(results: Sequence[dict[str, Any]]) -> float:
    """Unsupported claims / total claims, pooled over answers (not a mean of means:
    an answer with one claim should not outweigh one with six)."""
    total = sum(len(r["claims"]) for r in results)
    return sum(r["unsupported"] for r in results) / total if total else 0.0


def citation_correctness(results: Sequence[dict[str, Any]], units: Sequence[EvidenceUnit],
                         gold_locators: Sequence[dict[str, Any]]) -> float:
    """Fraction of cited units that match a gold locator by **file + location**.

    Never by unit id: ids are regenerated whenever ASR or chunking settings change,
    which is why the regression gold is locator-keyed in the first place.
    """
    from linkrag.eval.metrics import matches_locator
    by_id = {u.id: u for u in units}
    cited = [uid for r in results for c in r["claims"] for uid in c["unit_ids"]]
    if not cited:
        return 0.0
    ok = sum(1 for uid in cited
             if uid in by_id and any(matches_locator(by_id[uid], loc) for loc in gold_locators))
    return ok / len(cited)
