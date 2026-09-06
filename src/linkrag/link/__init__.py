"""The Evidence Linking Layer -- novel component 1.

Builds typed Links at indexing time: audio_slide, figure_paragraph,
deictic_visual, same_topic.

baseline: no-op, returns an empty link set (what P1/P2/P3 effectively have).
          For alignment specifically, the baseline ablation is `align_naive`:
          per-segment argmax with no sequence structure, roughly P2's behaviour.
linkrag:  full link construction with per-type scoring. Implemented so far:
          - audio_slide      monotonic DP over the whole lecture (Phase 2)
          - figure_text      numbered/descriptive references, bbox layout
                             proximity, page prior and semantics (Phase 3)
          - deictic          resolved by composing with the audio_slide alignment,
                             so "as you can see here" reaches the figure that was on
                             screen rather than any figure in the corpus (Phase 3)
"""

from __future__ import annotations

from linkrag.link.deictic import (
    Cue,
    find_cues,
    resolve_deictic,
    slide_map_from_links,
)
from linkrag.link.figure_text import (
    descriptive_refs,
    figure_kind,
    figure_number,
    figure_text_scores,
    link_figures_to_text,
    referenced,
    referenced_numbers,
    vertical_gap,
)
from linkrag.link.same_slide import is_slide_deck, link_same_slide
from linkrag.link.align import (
    Alignment,
    align,
    align_monotonic,
    align_naive,
    build_links,
    load_links,
    save_links,
    similarity_matrix,
)

__all__ = [
    "Alignment",
    "Cue",
    "is_slide_deck",
    "link_same_slide",
    "find_cues",
    "descriptive_refs",
    "figure_kind",
    "figure_number",
    "figure_text_scores",
    "link_figures_to_text",
    "referenced",
    "referenced_numbers",
    "vertical_gap",
    "resolve_deictic",
    "slide_map_from_links",
    "align",
    "align_monotonic",
    "align_naive",
    "build_links",
    "load_links",
    "save_links",
    "similarity_matrix",
]
