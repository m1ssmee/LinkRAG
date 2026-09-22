r"""Complementarity-aware reranking -- novel component 3.

Top-k scores each unit in isolation, so an evidence set of eight near-identical audio
segments scores as well as one that also carries the slide they refer to. This module
scores the **set**.

Objective
---------
Given candidates :math:`C` with relevance :math:`rel(u)`, choose :math:`R \subseteq C`,
:math:`|R| = k`, maximising

.. math::
    F(R) = \underbrace{\sum_{u \in R} rel(u)}_{\text{relevance}}
         + \alpha \, \underbrace{\big|\{\, m(u) : u \in R \,\}\big|}_{\text{distinct modalities}}
         + \beta \, \underbrace{\big|\{\, (u,v) \in R^2 : u \!\to\! v \in E \,\}\big|}_{\text{link edges inside } R}
         - \gamma \sum_{\substack{u,v \in R,\; u \neq v \\ m(u) = m(v)}} \cos(e_u, e_v)

:math:`m(u)` is the modality, :math:`E` the Evidence Linking Layer's edges, and
:math:`e_u` the unit's embedding. The three terms say: cover more modalities, prefer
evidence that is *linked* to what is already selected, and stop paying for a unit that
restates something already in the set. Redundancy is charged **only within a modality**
-- an audio segment and the slide it describes are supposed to be similar, and
penalising that would defeat the whole point.

Exact maximisation is NP-hard (this is a facility-location-style objective), so
selection is greedy: start from :math:`\arg\max_u rel(u)` and repeatedly add the unit
with the largest marginal gain

.. math::
    \Delta(u \mid R) = rel(u)
        + \alpha \,[\, m(u) \notin m(R) \,]
        + \beta \, |\{v \in R : u \!\to\! v \in E \text{ or } v \!\to\! u \in E\}|
        - \gamma \!\!\sum_{v \in R,\; m(v) = m(u)}\!\! \cos(e_u, e_v)

Ablations
---------
``none``   -- plain top-k by :math:`rel`. What the baseline does.
``mmr``    -- standard Maximal Marginal Relevance, text diversity only:
              :math:`\lambda\, rel(u) - (1-\lambda) \max_{v \in R} \cos(e_u, e_v)`.
              Modality-blind and link-blind, so it isolates what those two terms add.
``complementarity`` -- the objective above.

All three share one code path, so an ablation cannot diverge by accident.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any, Sequence

from collections import Counter

import numpy as np

from linkrag.core import stage_timer
from linkrag.retrieve.linkrag import RetrievedUnit

log = logging.getLogger("linkrag")

METHODS = ("none", "mmr", "complementarity")


def _mmss(seconds: float) -> str:
    return f"{int(seconds) // 60}:{int(seconds) % 60:02d}"


def modality_tag(result: RetrievedUnit) -> str:
    """"[AUDIO 18:40-19:10]" / "[FIGURE p.12]" -- a cross-encoder scores a bare chunk
    without knowing what kind of evidence it is or where it came from."""
    unit = result.unit
    loc = unit.location
    if unit.modality == "audio" and loc.start_s is not None:
        where = f"{_mmss(loc.start_s)}-{_mmss(loc.end_s or loc.start_s)}"
    elif loc.page is not None:
        where = f"p.{loc.page}"
    else:
        where = "?"
    return f"[{unit.modality.upper()} {where}]"


def cross_encoder_input(result: RetrievedUnit) -> str:
    return f"{modality_tag(result)} {' '.join(result.unit.content.split())}"


@lru_cache(maxsize=2)
def _cross_encoder(name: str, device: str):
    from sentence_transformers import CrossEncoder  # heavy, deferred

    return CrossEncoder(name, device=device)


def cross_encoder_scores(
    question: str, results: Sequence[RetrievedUnit], model: str, device: str = "cpu"
) -> list[float] | None:
    """Relevance from a cross-encoder, or None if it cannot be loaded.

    Returns None rather than raising: a missing reranker model must degrade to the
    retriever's own scores, not abort a query.
    """
    try:
        encoder = _cross_encoder(model, device)
        raw = encoder.predict([(question, cross_encoder_input(r)) for r in results])
    except Exception as exc:
        log.warning("cross-encoder %s unavailable (%s); using retrieval scores",
                    model, type(exc).__name__)
        return None
    raw = np.asarray(raw, dtype="float64")
    lo, hi = float(raw.min()), float(raw.max())
    return list((raw - lo) / (hi - lo)) if hi > lo else [1.0] * len(results)


def _vectors_for(results: Sequence[RetrievedUnit], index) -> np.ndarray | None:
    if index is None:
        return None
    try:
        rows = [index.vectors[index.id_to_pos[r.id]] for r in results]
    except KeyError:
        return None
    matrix = np.asarray(rows, dtype="float32")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-9)


def _edge_matrix(results: Sequence[RetrievedUnit], graph) -> np.ndarray:
    """Symmetric 0/1: is there a link either way between candidates i and j?"""
    n = len(results)
    edges = np.zeros((n, n), dtype="float32")
    if graph is None:
        return edges
    ids = [r.id for r in results]
    pos = {uid: i for i, uid in enumerate(ids)}
    for i, uid in enumerate(ids):
        if uid not in graph:
            continue
        for _s, dst, _d in graph.out_edges(uid, data=True):
            if dst in pos:
                edges[i, pos[dst]] = edges[pos[dst], i] = 1.0
    return edges


def modality_concentration(results: Sequence[RetrievedUnit], k_seeds: int = 8) -> dict[str, Any]:
    """Is the query's evidence concentrated in one modality?

    Looked at the top `k_seeds` candidates (the pool's own ranking):
      * `share`  -- fraction of them in the most common modality
      * `margin` -- best score in that modality minus the best score in any other
                    modality, divided by the best score overall (0 when a second
                    modality ties at the top, 1 when nothing else scores at all)

    The rule (fixed once, 2026-09-23; not tuned on any reported set): the evidence is
    CONCENTRATED when `share >= 0.75` or `margin >= 0.5`. On a concentrated query the
    coverage term alpha buys a modality the answer does not live in, which cost
    recall on pilot01 (-20.8 pp for baseline) and localisation on LectQA-Vid
    (hit@3 68 -> 55 %); beta and gamma are unaffected either way."""
    top = list(results)[:k_seeds]
    if not top:
        return {"share": 1.0, "margin": 1.0, "concentrated": True, "dominant": None}
    counts = Counter(r.unit.modality for r in top)
    dominant, n = counts.most_common(1)[0]
    share = n / len(top)
    best = max(float(r.score) for r in top)
    best_dom = max((float(r.score) for r in top if r.unit.modality == dominant), default=0.0)
    best_other = max((float(r.score) for r in top if r.unit.modality != dominant), default=0.0)
    margin = (best_dom - best_other) / best if best > 0 else 1.0
    return {"share": share, "margin": margin, "dominant": dominant,
            "concentrated": bool(share >= 0.75 or margin >= 0.5)}


def rerank(
    results: Sequence[RetrievedUnit],
    k: int,
    *,
    method: str = "complementarity",
    index=None,
    graph=None,
    alpha: float = 0.3,
    beta: float = 0.2,
    gamma: float = 0.4,
    mmr_lambda: float = 0.5,
    question: str | None = None,
    cross_encoder: str | None = None,
    device: str = "cpu",
    modality_gate: bool = False,
) -> list[RetrievedUnit]:
    """Select k units from `results` under the chosen objective.

    `modality_gate` (config `retrieve.rerank.modality_gate`, default off): when the
    pool's evidence is concentrated in one modality (`modality_concentration`), run
    with alpha = 0 -- keep the link bonus and the redundancy penalty, drop the
    coverage bonus. Only affects `complementarity`."""
    if method not in METHODS:
        raise ValueError(f"unknown rerank method {method!r}: use one of {METHODS}")
    if not results:
        return []
    gate = None
    if modality_gate and method == "complementarity":
        gate = modality_concentration(results)
        if gate["concentrated"]:
            alpha = 0.0
    rerank.last_gate = gate  # type: ignore[attr-defined]

    with stage_timer("retrieve.rerank", method=method, candidates=len(results), k=k,
                     gate=("concentrated" if gate and gate["concentrated"] else
                           ("spread" if gate else "off"))) as t:
        rel = [float(r.score) for r in results]
        if cross_encoder and question:
            scored = cross_encoder_scores(question, results, cross_encoder, device)
            if scored is not None:
                rel = scored
                t["cross_encoder"] = 1

        if method == "none":
            order = sorted(range(len(results)), key=lambda i: (-rel[i], results[i].id))
            t["selected"] = min(k, len(order))
            return [results[i] for i in order[:k]]

        vectors = _vectors_for(results, index)
        sim = vectors @ vectors.T if vectors is not None else np.zeros((len(results),) * 2)
        edges = _edge_matrix(results, graph) if method == "complementarity" else None
        modality = [r.unit.modality for r in results]

        chosen: list[int] = [max(range(len(results)), key=lambda i: (rel[i], -ord(results[i].id[0])))]
        while len(chosen) < min(k, len(results)):
            covered = {modality[i] for i in chosen}
            best_i, best_gain = None, -np.inf
            for i in range(len(results)):
                if i in chosen:
                    continue
                if method == "mmr":
                    gain = (mmr_lambda * rel[i]
                            - (1.0 - mmr_lambda) * max(sim[i, j] for j in chosen))
                else:
                    same = [sim[i, j] for j in chosen if modality[j] == modality[i]]
                    gain = (rel[i]
                            + alpha * (0.0 if modality[i] in covered else 1.0)
                            + beta * float(sum(edges[i, j] for j in chosen))
                            - gamma * float(sum(same)))
                if gain > best_gain or (gain == best_gain and best_i is not None
                                        and results[i].id < results[best_i].id):
                    best_i, best_gain = i, gain
            if best_i is None:
                break
            chosen.append(best_i)

        t["selected"] = len(chosen)
        t["modalities"] = len({modality[i] for i in chosen})
    return [results[i] for i in chosen]


def set_diagnostics(results: Sequence[RetrievedUnit], index=None) -> dict[str, float]:
    """Modality coverage and mean within-modality redundancy of a returned set."""
    if not results:
        return {"units": 0, "modalities": 0, "redundancy": 0.0}
    vectors = _vectors_for(results, index)
    modality = [r.unit.modality for r in results]
    pairs = []
    if vectors is not None:
        sim = vectors @ vectors.T
        pairs = [float(sim[i, j])
                 for i in range(len(results)) for j in range(i + 1, len(results))
                 if modality[i] == modality[j]]
    return {
        "units": len(results),
        "modalities": len(set(modality)),
        "redundancy": (sum(pairs) / len(pairs)) if pairs else 0.0,
    }
