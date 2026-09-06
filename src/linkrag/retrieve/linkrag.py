r"""Link-following retrieval -- novel component 2.

The baseline stops at top-k: whatever the query embedding happens to reach is the
whole evidence set. This retriever treats top-k as **seeds** and then follows the
Evidence Linking Layer outward, so a unit enters the set because something already
retrieved *depends on it*, not because it independently looked similar to the query.

That is the difference from P1 (MI-RAG), which also enlarges the evidence set but does
so by re-querying: another fuzzy retrieval, another chance to pull in loosely related
text. Expansion here travels a known typed relation, which is why it can add a slide
that shares no vocabulary with the question at all.

Scoring
-------
A seed keeps its fused retrieval score. An expanded unit gets

.. math::  s_{expanded} = s_{seed} \cdot w_{link} \cdot \gamma

with :math:`\gamma` the decay (`retrieve.linkrag.decay`).

**Note on scale.** RRF scores are ~0.03 while link weights are ~0.6-1.0, so an
expanded unit's score is always well below any seed's. With `k_seed < k_final` that is
harmless and intended: the seeds are kept and expansion fills the remaining slots.
It does mean expansion cannot currently *displace* a weak seed -- Phase 5's
complementarity reranker is what changes that, by scoring the set rather than the unit.

Provenance
----------
Every returned unit records whether it was a seed or expanded, and if expanded, which
seed and which link type brought it in. An evidence set that cannot say why a unit is
present cannot be audited, and the pilot already produced one answer that was correctly
cited and factually wrong.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import networkx as nx

from linkrag.core import EvidenceUnit, Mode, stage_timer
from linkrag.index import Encoder, Index
from linkrag.link.graph import neighbors
from linkrag.retrieve.baseline import RRF_K, retrieve_scored

DEFAULT_LINK_TYPES = ("audio_slide", "figure_text", "deictic")


@dataclass
class RetrievedUnit:
    """A unit in the evidence set, with why it is there."""

    unit: EvidenceUnit
    score: float
    origin: str                      # "seed" | "expanded"
    via_seed: str | None = None      # seed unit id that reached it
    via_link: str | None = None      # link type traversed
    link_score: float | None = None

    @property
    def id(self) -> str:
        return self.unit.id

    def explain(self) -> str:
        if self.origin == "expanded" and self.via_seed:
            return f"expanded from {self.via_seed} via {self.via_link} ({self.link_score:.3f})"
        return self.origin


def retrieve_linkrag(
    question: str,
    index: Index,
    graph: nx.MultiDiGraph,
    *,
    encoder: Encoder,
    mode: Mode = "linkrag",
    k_seed: int = 5,
    k_final: int = 8,
    link_types: Iterable[str] | None = DEFAULT_LINK_TYPES,
    min_link_score: float = 0.0,
    decay: float = 0.5,
    hops: int = 1,
    candidates: int = 50,
    rrf_k: int = RRF_K,
) -> list[RetrievedUnit]:
    """Seed with the hybrid retriever, expand along links, keep the best k_final.

    baseline: no expansion at all -- plain top-k_final, identical to
              `retrieve.baseline`. This is the ablation.
    linkrag:  top-k_seed seeds, then `hops` of link traversal.
    """
    if mode not in ("baseline", "linkrag"):
        raise ValueError(f"unknown mode {mode!r}: use 'baseline' or 'linkrag'")

    by_id = {u.id: u for u in index.units}
    wanted = set(link_types) if link_types else None

    with stage_timer("retrieve.linkrag", mode=mode, k_seed=k_seed, k_final=k_final) as t:
        seed_k = k_final if mode == "baseline" else k_seed
        seeds = retrieve_scored(question, index, encoder=encoder, top_k=seed_k,
                                candidates=candidates, rrf_k=rrf_k)

        chosen: dict[str, RetrievedUnit] = {
            unit.id: RetrievedUnit(unit=unit, score=float(score), origin="seed")
            for unit, score in seeds
        }
        t["seeds"] = len(chosen)

        if mode == "linkrag" and hops > 0:
            frontier = [(rid, chosen[rid].score) for rid in list(chosen)]
            for _hop in range(hops):
                nxt: list[tuple[str, float]] = []
                for source_id, source_score in frontier:
                    for neighbour_id, link_type, link_score in neighbors(
                        graph, source_id, wanted, min_link_score
                    ):
                        if neighbour_id not in by_id:
                            continue  # stale link into a unit the index no longer has
                        candidate = RetrievedUnit(
                            unit=by_id[neighbour_id],
                            score=source_score * float(link_score) * decay,
                            origin="expanded",
                            via_seed=source_id,
                            via_link=link_type,
                            link_score=float(link_score),
                        )
                        existing = chosen.get(neighbour_id)
                        # never demote a seed, and keep the best route in
                        if existing is None or (
                            existing.origin == "expanded" and candidate.score > existing.score
                        ):
                            chosen[neighbour_id] = candidate
                            nxt.append((neighbour_id, candidate.score))
                frontier = nxt
                if not frontier:
                    break

        t["expanded"] = sum(1 for r in chosen.values() if r.origin == "expanded")

    ranked = sorted(chosen.values(), key=lambda r: (-r.score, r.id))[:k_final]
    return ranked



def expansion_report(results: Sequence[RetrievedUnit], graph: nx.MultiDiGraph) -> tuple[int, int]:
    """(expanded units, seeds that had at least one graph edge).

    Zero expansion when the seeds *did* have edges means the link types or the score
    floor filtered everything out; zero expansion when they had none means the graph
    is empty or stale. The two look identical in the output otherwise, and both make
    linkrag results equal to baseline.
    """
    expanded = sum(1 for r in results if r.origin == "expanded")
    seeds_with_edges = sum(
        1 for r in results
        if r.origin == "seed" and r.id in graph and (graph.out_degree(r.id) or graph.in_degree(r.id))
    )
    return expanded, seeds_with_edges
