"""figure <-> the text of the deck page it sits on.

On a slide deck the "paragraph explaining a figure" is almost always the text of the
same slide -- there is no caption and no "Figure N" cross-reference (13 of 18 pilot01
figures had neither). That relation is *certain* rather than scored: the figure was
rendered on that page. So it is emitted at a fixed score instead of being pushed
through `figure_text`'s similarity machinery, where it would compete on evidence it
structurally cannot produce.

Only applies to documents flagged as decks (`metadata["slide_deck"]`, set at ingest
from page geometry or `ingest.slide_deck_files`). A paper's figures are *not*
co-located with their explanation, which is the whole reason `figure_text` exists.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Sequence

from linkrag.core import EvidenceUnit, Link, Mode, stage_timer


def is_slide_deck(unit: EvidenceUnit) -> bool:
    return bool(unit.metadata.get("slide_deck"))


def link_same_slide(
    figures: Sequence[EvidenceUnit],
    texts: Sequence[EvidenceUnit],
    *,
    mode: Mode = "linkrag",
    enabled: bool = True,
    score: float = 1.0,
) -> list[Link]:
    """One same_slide Link per (deck figure, text unit on the same page).

    baseline: emits nothing -- the ablation for "does deck co-location help?".
    linkrag:  emits the co-location links.

    A page with several text chunks yields one link per chunk: all of them are
    genuinely the text surrounding that figure.
    """
    with stage_timer("link.same_slide", mode=mode, figures=len(figures)) as t:
        if mode == "baseline" or not enabled:
            t["links"] = 0
            return []

        by_page: dict[tuple[str, int], list[EvidenceUnit]] = defaultdict(list)
        for text in texts:
            if text.location.page is not None and is_slide_deck(text):
                by_page[(Path(text.source_file).name, text.location.page)].append(text)

        links: list[Link] = []
        for figure in figures:
            if not is_slide_deck(figure) or figure.location.page is None:
                continue
            key = (Path(figure.source_file).name, figure.location.page)
            for text in by_page.get(key, []):
                links.append(Link(
                    figure.id, text.id, "same_slide", float(score),
                    metadata={"page": figure.location.page,
                              "source": Path(figure.source_file).name},
                ))
        t["links"] = len(links)
        t["deck_pages"] = len(by_page)
    return links
