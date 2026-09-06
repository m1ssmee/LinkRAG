"""Standing instruments every phase must report.

Kept here rather than inline in a script so that ask.py, the regression runner
and any future ablation all compute them the same way.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

from linkrag.core import EvidenceUnit

MODALITY_ORDER = ["text", "figure", "table", "audio"]


def modality_distribution(units: Sequence[EvidenceUnit]) -> dict[str, int]:
    counts = Counter(u.modality for u in units)
    return {m: counts[m] for m in MODALITY_ORDER if counts[m]}


def format_modality_distribution(units: Sequence[EvidenceUnit]) -> str:
    """e.g. "8/8 audio" or "audio 6/8, text 2/8".

    Instrument #1. pilot01 showed the baseline collapsing onto whichever modality
    has the most units (audio, 54% of the index), so this number is reported for
    every question in every phase -- a retrieval win that only reshuffles within
    one modality is not the win we are claiming.
    """
    total = len(units)
    if not total:
        return "0 units"
    dist = modality_distribution(units)
    if len(dist) == 1:
        only = next(iter(dist))
        return f"{total}/{total} {only}"
    return ", ".join(f"{m} {n}/{total}" for m, n in dist.items())


def matches_locator(unit: EvidenceUnit, locator: dict[str, Any]) -> bool:
    """Does `unit` cover the gold evidence `locator` describes?

    Matching is by (file, page) or (file, overlapping time span) rather than by
    unit id: audio ids are regenerated on every re-transcription, so an
    id-keyed regression file would go stale the first time ASR settings change.
    """
    if Path(unit.source_file).name != locator["source"]:
        return False
    if "page" in locator:
        return unit.location.page == locator["page"]
    start, end = unit.location.start_s, unit.location.end_s
    if start is None or end is None:
        return False
    # any overlap counts: re-segmentation shifts boundaries by a few seconds
    return start < locator["end_s"] and end > locator["start_s"]


def gold_hits(
    units: Sequence[EvidenceUnit], locators: Iterable[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split gold locators into (found, missed) against a retrieved set."""
    found, missed = [], []
    for locator in locators:
        (found if any(matches_locator(u, locator) for u in units) else missed).append(locator)
    return found, missed


def gold_coverage(units: Sequence[EvidenceUnit], locators: list[dict[str, Any]]) -> str:
    if not locators:
        return "n/a"
    found, _missed = gold_hits(units, locators)
    if len(found) == len(locators):
        return "yes"
    return "partial" if found else "no"


def describe_locator(locator: dict[str, Any]) -> str:
    if "page" in locator:
        return f"{locator['source']} p.{locator['page']}"
    return f"{locator['source']} {locator['start_s']:.1f}-{locator['end_s']:.1f}s"
