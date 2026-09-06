"""Baseline retrieval: hybrid dense + BM25, fused by reciprocal rank, plain top-k.

This is the comparison point for the paper. No links, no expansion, no set-level
reranking -- one shot, independent units, exactly what P1/P2/P3 do at this stage.
Do not "improve" it: a sandbagged baseline invalidates the result.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from linkrag.core import EvidenceUnit, stage_timer
from linkrag.index import Encoder, Index

RRF_K = 60  # standard constant from Cormack et al. 2009; dampens top-rank dominance


def rrf_fuse(
    rankings: Sequence[Sequence[tuple[int, float]]], rrf_k: int = RRF_K
) -> list[tuple[int, float]]:
    """Reciprocal rank fusion over several (position, score) rankings.

    Only each item's *rank* is used, never its raw score -- which is the point:
    BM25 scores and cosine similarities are not on a comparable scale.
    """
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, (position, _score) in enumerate(ranking, start=1):
            fused[position] = fused.get(position, 0.0) + 1.0 / (rrf_k + rank)
    return sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))


def retrieve_scored(
    query: str,
    index: Index,
    *,
    encoder: Encoder,
    top_k: int = 8,
    candidates: int = 50,
    rrf_k: int = RRF_K,
) -> list[tuple[EvidenceUnit, float]]:
    with stage_timer("retrieve.baseline", k=top_k) as t:
        query_vec = np.asarray(encoder([query]), dtype="float32")[0]
        dense = index.dense_search(query_vec, candidates)
        sparse = index.sparse_search(query, candidates)
        fused = rrf_fuse([dense, sparse], rrf_k)[:top_k]
        t["dense"] = len(dense)
        t["sparse"] = len(sparse)
        t["returned"] = len(fused)
    return [(index.units[position], score) for position, score in fused]


