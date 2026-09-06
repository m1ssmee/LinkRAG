"""Retrieval -- novel components 2 and 3.

baseline: hybrid dense + BM25 fused by reciprocal rank, plain top-k, no
          expansion (P1 additionally re-queries; that variant lands with eval).
linkrag:  link-following expansion from the top-k seeds (`linkrag.py`), then
          complementarity-aware reranking that scores an evidence *set* for
          coverage rather than each unit in isolation (Phase 5, not built yet).
          `iterative.py` holds the P1/MI-RAG approximation used as a third,
          fairer baseline than plain top-k.
"""
