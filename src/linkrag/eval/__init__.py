"""Metrics and ablations.

Every metric is computed for both modes and the delta is the result:
answer accuracy, evidence precision/recall, cross-modal coverage, and
citation faithfulness (P2 has no hallucination check -- we do).

baseline: scores the papers' pipeline (independent indexing, plain top-k).
linkrag:  scores ours; a component that can't be ablated this way isn't done.

Standing instruments (reported by every phase, see DESIGN.md):
  1. modality distribution of the retrieved set, per question
  2. the pilot01 Q1-Q4 regression set, appended to reports/regression.md
"""

from __future__ import annotations

from linkrag.eval.metrics import (
    describe_locator,
    format_modality_distribution,
    gold_coverage,
    gold_hits,
    matches_locator,
    modality_distribution,
)

__all__ = [
    "describe_locator",
    "format_modality_distribution",
    "gold_coverage",
    "gold_hits",
    "matches_locator",
    "modality_distribution",
]
