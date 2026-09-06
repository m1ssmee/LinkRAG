"""Retrieval -- novel components 2 and 3.

baseline: hybrid dense + BM25 fused by reciprocal rank, plain top-k, no
          expansion (P1 additionally re-queries; that variant lands with eval).
linkrag:  link-following expansion from the top-k seeds (`linkrag.py`), then
          complementarity-aware reranking that scores an evidence *set* for
          coverage rather than each unit in isolation (Phase 5, not built yet).
          `iterative.py` holds the P1/MI-RAG approximation used as a third,
          fairer baseline than plain top-k.
"""

from __future__ import annotations

from linkrag.retrieve.baseline import retrieve, retrieve_scored, rrf_fuse
from linkrag.retrieve.iterative import IterativeResult, retrieve_iterative
from linkrag.retrieve.linkrag import (
    RetrievedUnit,
    expansion_report,
    retrieve_linkrag,
    units_of,
)

__all__ = ["IterativeResult", "RetrievedUnit", "retrieve", "retrieve_iterative",
           "retrieve_linkrag", "retrieve_scored",
           "rrf_fuse", "units_of"]
