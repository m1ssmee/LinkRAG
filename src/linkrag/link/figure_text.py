r"""figure <-> paragraph links -- second component of the Evidence Linking Layer.

P3 (MARA) treats pages as independent units, so a figure and the paragraph that
explains it three pages later are never connected. That is the gap this closes:
candidates are scored across the **whole document**, with page distance as a soft
prior rather than a hard wall.

Scoring
-------
For figure :math:`f` and text unit :math:`t`,

.. math::
    F_{ft} = w_r R_{ft} + w_p P_{ft} + w_d \cos(e_f, e_t) + w_o O_{ft}

* :math:`R_{ft} \in \{0,1\}` -- **explicit reference**: :math:`t` names the figure
  ("Figure 3", "Fig. 3") and :math:`f`'s caption carries that number. Nearly
  conclusive when present, which is why it gets its own term rather than being
  buried in the text similarity.
* :math:`P_{ft} = \exp(-|page(f)-page(t)| / \tau)` -- page proximity, decaying with
  :math:`\tau` (`page_decay`). A *prior*, never a filter.
* :math:`\cos(e_f, e_t)` -- embedding similarity; the figure side embeds its caption,
  falling back to a provenance descriptor when no caption was found.
* :math:`O_{ft}` -- IDF-weighted rare-term overlap, so "NoScope" counts and "the" does not.

Modes
-----
`baseline` keeps only same-page pairs, which is the strongest thing a
page-independent pipeline can do. `linkrag` scores the whole document. The two share
one code path so the ablation isolates the cross-page reach and nothing else.
"""

from __future__ import annotations

import math
import re
from typing import Sequence

import numpy as np

from linkrag.core import EvidenceUnit, Link, Mode, stage_timer
from linkrag.index import Encoder, embeddable_text, tokenize
from linkrag.link.align import _idf

# "Fig. 3", "Figure 3", "Table 2" -- kind is captured so a figure reference does
# not match a table and vice versa.
NUMBERED_REF = re.compile(r"\b(fig(?:ure)?|tab(?:le)?)\s*\.?\s*(\d{1,2})\b", re.I)

# "the diagram below", "the flowchart above", "the graph on the left" -- a
# reference with no number, resolvable only through layout.
DESCRIPTIVE_REF = re.compile(
    r"\bthe\s+(diagram|flowchart|graph|chart|plot|figure|table|image|picture|screenshot)"
    r"\s+(below|above|on the left|on the right|to the left|to the right|opposite)\b",
    re.I,
)

DIRECTION_BELOW = {"below"}
DIRECTION_ABOVE = {"above"}


def _kind(word: str) -> str:
    return "table" if word.lower().startswith("tab") else "figure"


def figure_number(text: str) -> int | None:
    """The figure's own number, from a caption like 'Figure 1: attention heatmap'."""
    match = NUMBERED_REF.search(text or "")
    return int(match.group(2)) if match else None


def figure_kind(text: str) -> str:
    """"table" if the caption says Table N, else "figure"."""
    match = NUMBERED_REF.search(text or "")
    return _kind(match.group(1)) if match else "figure"


def referenced_numbers(text: str) -> set[int]:
    """Every figure/table number a paragraph refers to (numbers only, kind-blind)."""
    return {int(n) for _kindword, n in NUMBERED_REF.findall(text or "")}


def referenced(text: str) -> set[tuple[str, int]]:
    """(kind, number) pairs, so "Table 2" does not match "Figure 2"."""
    return {(_kind(k), int(n)) for k, n in NUMBERED_REF.findall(text or "")}


def descriptive_refs(text: str) -> list[tuple[str, str]]:
    """(noun, direction) for unnumbered references like "the diagram below"."""
    return [(m.group(1).lower(), m.group(2).lower())
            for m in DESCRIPTIVE_REF.finditer(text or "")]


def vertical_gap(a: tuple[float, float, float, float] | None,
                 b: tuple[float, float, float, float] | None) -> float | None:
    """Points of vertical whitespace between two bboxes; 0 if they overlap.

    PyMuPDF's y grows downward, so `a` is above `b` when a[3] <= b[1].
    """
    if a is None or b is None:
        return None
    if a[3] <= b[1]:
        return b[1] - a[3]
    if b[3] <= a[1]:
        return a[1] - b[3]
    return 0.0


def figure_text_scores(
    figures: Sequence[EvidenceUnit],
    texts: Sequence[EvidenceUnit],
    *,
    encoder: Encoder,
    w_reference: float = 0.30,
    w_layout: float = 0.15,
    w_page: float = 0.15,
    w_dense: float = 0.25,
    w_overlap: float = 0.15,
    page_decay: float = 2.0,
    layout_max_gap_pt: float = 220.0,
    reference_page_window: int = 1,
) -> tuple[np.ndarray, list[list[dict]]]:
    """Returns (scores, reasons) where reasons[i][j] records why the pair scored.

    The reason is kept because a link the retriever cannot explain is a link the
    eval cannot score.
    """
    if not figures or not texts:
        raise ValueError("need at least one figure unit and one text unit")

    fig_text = [embeddable_text(f) for f in figures]
    txt_text = [t.content for t in texts]
    f_vec = np.asarray(encoder(fig_text), dtype="float32")
    t_vec = np.asarray(encoder(txt_text), dtype="float32")
    f_vec /= np.maximum(np.linalg.norm(f_vec, axis=1, keepdims=True), 1e-9)
    t_vec /= np.maximum(np.linalg.norm(t_vec, axis=1, keepdims=True), 1e-9)
    dense = f_vec @ t_vec.T

    txt_tokens = [tokenize(t) for t in txt_text]
    idf = _idf(txt_tokens)
    fig_sets = [set(tokenize(t)) for t in fig_text]
    txt_sets = [set(t) for t in txt_tokens]

    scores = np.zeros((len(figures), len(texts)), dtype="float32")
    reasons: list[list[dict]] = [[{} for _ in texts] for _ in figures]

    txt_numbered = [referenced(t.content) for t in texts]
    txt_descriptive = [descriptive_refs(t.content) for t in texts]

    for i, fig in enumerate(figures):
        number = figure_number(fig.content)
        kind = figure_kind(fig.content)
        fig_mass = max(sum(idf.get(t, 0.0) for t in fig_sets[i]), 1e-9)
        fpage, fbox = fig.location.page, fig.location.bbox

        for j, txt in enumerate(texts):
            why: dict = {}
            tpage, tbox = txt.location.page, txt.location.bbox
            page_gap = (abs(fpage - tpage) if fpage is not None and tpage is not None else None)

            # (a) explicit numbered mention, restricted to this or an adjacent page
            reference = 0.0
            if (number is not None and (kind, number) in txt_numbered[j]
                    and page_gap is not None and page_gap <= reference_page_window):
                reference = 1.0
                why["reference"] = f"{kind} {number}"

            # (a') unnumbered descriptive mention, resolved by direction on the page
            if reference == 0.0 and page_gap == 0 and txt_descriptive[j]:
                gap = vertical_gap(tbox, fbox)
                for noun, direction in txt_descriptive[j]:
                    if gap is None:
                        continue
                    below = fbox is not None and tbox is not None and tbox[3] <= fbox[1]
                    above = fbox is not None and tbox is not None and fbox[3] <= tbox[1]
                    if ((direction in DIRECTION_BELOW and below)
                            or (direction in DIRECTION_ABOVE and above)
                            or direction not in DIRECTION_BELOW | DIRECTION_ABOVE):
                        reference = 0.7  # weaker than a numbered reference
                        why["descriptive"] = f"the {noun} {direction}"
                        break

            # (b) layout proximity: same page, vertical whitespace between bboxes
            layout = 0.0
            if page_gap == 0:
                gap = vertical_gap(fbox, tbox)
                if gap is not None and gap < layout_max_gap_pt:
                    layout = 1.0 - gap / layout_max_gap_pt
                    why["layout_gap_pt"] = round(gap, 1)

            page = (math.exp(-page_gap / max(page_decay, 1e-9))
                    if page_gap is not None else 0.0)

            shared = fig_sets[i] & txt_sets[j]
            overlap = min((sum(idf.get(t, 0.0) for t in shared) / fig_mass) if shared else 0.0, 1.0)

            scores[i, j] = (w_reference * reference + w_layout * layout + w_page * page
                            + w_dense * float(dense[i, j]) + w_overlap * overlap)
            why["dense"] = round(float(dense[i, j]), 4)
            reasons[i][j] = why
    return scores, reasons


def link_figures_to_text(
    figures: Sequence[EvidenceUnit],
    texts: Sequence[EvidenceUnit],
    *,
    encoder: Encoder,
    mode: Mode = "linkrag",
    threshold: float = 0.50,
    max_links_per_unit: int = 8,
    weights: dict[str, float] | None = None,
    page_decay: float = 2.0,
    layout_max_gap_pt: float = 220.0,
    reference_page_window: int = 1,
) -> list[Link]:
    """figure_text Links.

    baseline: same-page candidates only (a page-independent pipeline's best case).
    linkrag:  whole-document candidates, page distance as a soft prior.
    """
    weights = weights or {}
    with stage_timer("link.figure", mode=mode, figures=len(figures), texts=len(texts)) as t:
        scores, reasons = figure_text_scores(
            figures, texts, encoder=encoder,
            w_reference=weights.get("reference", 0.30),
            w_layout=weights.get("layout", 0.15),
            w_page=weights.get("page", 0.15),
            w_dense=weights.get("dense", 0.25),
            w_overlap=weights.get("overlap", 0.15),
            page_decay=page_decay,
            layout_max_gap_pt=layout_max_gap_pt,
            reference_page_window=reference_page_window,
        )
        links: list[Link] = []
        cross_page = 0
        for i, fig in enumerate(figures):
            order = np.argsort(-scores[i])
            kept = 0
            for j in order:
                if kept >= max_links_per_unit:
                    break
                txt = texts[int(j)]
                same_page = (fig.location.page is not None
                             and fig.location.page == txt.location.page)
                if mode == "baseline" and not same_page:
                    continue
                score = float(scores[i, int(j)])
                if score < threshold:
                    break  # order is descending, so nothing later can qualify
                links.append(Link(fig.id, txt.id, "figure_text", score,
                                  metadata=dict(reasons[i][int(j)])))
                kept += 1
                if not same_page:
                    cross_page += 1
        t["links"] = len(links)
        t["cross_page"] = cross_page
    return links
