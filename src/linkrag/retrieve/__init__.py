"""Retrieval -- novel components 2 and 3.

baseline: hybrid dense + BM25 fused by reciprocal rank, plain top-k, no
          expansion (P1 additionally re-queries; that variant lands with eval).
linkrag:  link-following expansion from the top-k seeds, then
          complementarity-aware reranking that scores an evidence *set* for
          coverage rather than scoring each unit in isolation. Not built yet.
"""

from __future__ import annotations

from linkrag.retrieve.baseline import retrieve, retrieve_scored, rrf_fuse

__all__ = ["retrieve", "retrieve_scored", "rrf_fuse"]
