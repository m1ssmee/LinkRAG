"""Iterative re-querying -- our approximation of P1 (MI-RAG), used as a baseline.

P1 enlarges the evidence set by reformulating the question and retrieving again. Its
recorded weakness is precision: each extra round is another fuzzy retrieval, so the
model receives a bigger haystack rather than a better one.

This exists so LinkRAG is compared against something that *also* returns more evidence
than plain top-k. Measuring link-following only against single-shot top-k would credit
expansion for the trivial fact that it returns more units. Here both enlarge the set;
the question is whether the extra units are the *right* ones.

Cost is reported, not hidden: each round after the first spends an LLM call to write
the follow-up query, and that call is on the critical path. Link-following spends none.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Callable

from linkrag.core import stage_timer
from linkrag.index import Encoder, Index
from linkrag.retrieve.baseline import RRF_K, retrieve_scored
from linkrag.retrieve.linkrag import RetrievedUnit

Completer = Callable[[str, str], str]

FOLLOWUP_SYSTEM = (
    "You write short search queries. Given a question and the evidence retrieved so "
    "far, write ONE follow-up search query for the information that is still missing. "
    "Reply with the query text only, no preamble, no quotes, at most 20 words."
)


@dataclass
class IterativeResult:
    units: list[RetrievedUnit]
    queries: list[str] = field(default_factory=list)
    llm_calls: int = 0
    latency_s: float = 0.0

    @property
    def ids(self) -> list[str]:
        return [r.id for r in self.units]


def _clean_query(text: str) -> str:
    """Models like to answer with a sentence; keep the first line, drop quoting."""
    first = (text or "").strip().splitlines()[0] if (text or "").strip() else ""
    first = re.sub(r'^["\'`]+|["\'`]+$', "", first.strip())
    first = re.sub(r"^(follow-?up query|query)\s*[:\-]\s*", "", first, flags=re.I)
    return " ".join(first.split()[:20])


def retrieve_iterative(
    question: str,
    index: Index,
    *,
    encoder: Encoder,
    complete: Completer | None = None,
    rounds: int = 2,
    k_per_round: int = 8,
    k_final: int | None = None,
    candidates: int = 50,
    rrf_k: int = RRF_K,
) -> IterativeResult:
    """Retrieve, ask for a follow-up query, retrieve again. `rounds` counts the first.

    Falls back to a single round when no completer is supplied, so the harness can be
    exercised without spending API calls; the round count in the result then says 1.
    """
    started = time.perf_counter()
    result = IterativeResult(units=[], queries=[question])
    chosen: dict[str, RetrievedUnit] = {}

    with stage_timer("retrieve.iterative", rounds=rounds, k=k_per_round) as t:
        for round_no in range(1, max(1, rounds) + 1):
            query = result.queries[-1]
            for unit, score in retrieve_scored(query, index, encoder=encoder,
                                               top_k=k_per_round, candidates=candidates,
                                               rrf_k=rrf_k):
                existing = chosen.get(unit.id)
                if existing is None or score > existing.score:
                    chosen[unit.id] = RetrievedUnit(unit=unit, score=float(score),
                                                    origin=f"round{round_no}")

            if round_no >= rounds or complete is None:
                break

            evidence = "\n".join(
                f"- ({r.unit.modality}) {' '.join(r.unit.content.split())[:160]}"
                for r in sorted(chosen.values(), key=lambda r: -r.score)[:k_per_round]
            )
            follow_up = _clean_query(complete(
                FOLLOWUP_SYSTEM,
                f"Question: {question}\n\nEvidence so far:\n{evidence}\n\nFollow-up query:",
            ))
            result.llm_calls += 1
            if not follow_up or follow_up.lower() == query.lower():
                break  # nothing new to ask; another identical round is pure cost
            result.queries.append(follow_up)

        t["llm_calls"] = result.llm_calls
        t["units"] = len(chosen)

    limit = k_final or k_per_round
    result.units = sorted(chosen.values(), key=lambda r: (-r.score, r.id))[:limit]
    result.latency_s = time.perf_counter() - started
    return result
