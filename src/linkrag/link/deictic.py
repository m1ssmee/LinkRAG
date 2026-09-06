r"""deictic speech <-> visual element -- third component of the Evidence Linking Layer.

"As you can see **here**, the attention weights concentrate on the subject token."
The referent of *here* is on a slide, in a different file, and nothing in the words
themselves identifies it. P2 could only resolve this by timestamp inside a single
video; with the deck as a separate PDF it has no mechanism at all.

This module resolves the referent by **composing with Phase 2**: the audio segment is
already aligned to a slide, so the candidate set shrinks from "every figure in the
corpus" to "the figures on the slide that was on screen". Deixis is resolved by
sequence structure, not by the deictic word.

Scoring
-------
For audio segment :math:`a` carrying cue :math:`c`, and figure :math:`v`,

.. math::
    D_{av} = w_s \cdot [\,page(v) = slide(a)\,] + w_c \cdot strength(c)
             + w_d \cdot \cos(e_{ctx(a,c)}, e_v)

where :math:`ctx(a,c)` is the words around the cue (via word timestamps) rather than
the whole segment -- a 30-second unit usually discusses more than the thing pointed at.

Modes
-----
`baseline` drops the slide term and ranks every figure in the corpus by similarity
alone -- what is available without an alignment. `linkrag` restricts candidates to the
aligned slide. The ablation therefore measures exactly what Phase 2 buys Phase 3.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from linkrag.core import EvidenceUnit, Link, Mode, stage_timer
from linkrag.index import Encoder, embeddable_text, tokenize
from linkrag.link.align import _idf

DEFAULT_CUES = ["here", "this", "that", "that one", "this one", "shown",
                "on the left", "on the right", "this slide", "as you can see",
                "as you see", "look at this", "over here", "right here"]

# Unnumbered object deixis -- "the arrow above", "the box on the right". Enumerated
# as noun x direction rather than listed by hand so the coverage is uniform and the
# config stays two short lists instead of eighty phrases.
DEFAULT_OBJECT_NOUNS = ["arrow", "box", "step", "line", "circle", "node", "column",
                        "row", "bar", "point", "block", "region", "curve", "axis"]
DEFAULT_DIRECTIONS = ["above", "below", "on the left", "on the right", "here", "there"]
# "this arrow here" is as common in speech as "the arrow above", and only the
# determiner differs -- so all three are generated rather than just "the".
DEFAULT_DETERMINERS = ["the", "this", "that"]


def expand_cues(
    cues: Sequence[str] = DEFAULT_CUES,
    object_nouns: Sequence[str] = DEFAULT_OBJECT_NOUNS,
    directions: Sequence[str] = DEFAULT_DIRECTIONS,
    determiners: Sequence[str] = DEFAULT_DETERMINERS,
) -> list[str]:
    """Base cues plus every "<determiner> <noun> <direction>" combination."""
    phrases = list(cues)
    phrases += [f"{det} {noun} {direction}"
                for det in determiners for noun in object_nouns for direction in directions]
    return phrases

CONTEXT_WORDS = 12  # words either side of the cue that form its context window


@dataclass(frozen=True)
class Cue:
    text: str
    start_s: float
    end_s: float
    context: str

    @property
    def strength(self) -> float:
        """Note: multi-word cues are more reliably deictic than bare "this",
        so strength is word count capped at 1.0. A learned weight would be better
        and needs labelled referents we do not have yet."""
        return min(1.0, len(self.text.split()) / 3.0)


def find_cues(unit: EvidenceUnit, cues: Sequence[str] = DEFAULT_CUES) -> list[Cue]:
    """Locate deictic cues and their surrounding words, using word timestamps.

    Longer cues win over the shorter ones they contain, so "as you can see" is not
    also reported as "see"/"this".
    """
    words = unit.metadata.get("words") or []
    if not words:
        return []
    tokens = [re.sub(r"[^a-z0-9']", "", str(w[2]).lower()) for w in words]
    joined = " ".join(tokens)

    found: list[Cue] = []
    claimed: list[tuple[int, int]] = []
    for cue in sorted(cues, key=lambda c: -len(c.split())):
        parts = cue.lower().split()
        n = len(parts)
        for i in range(len(tokens) - n + 1):
            if tokens[i:i + n] != parts:
                continue
            if any(s <= i < e or s < i + n <= e for s, e in claimed):
                continue  # already inside a longer cue
            claimed.append((i, i + n))
            lo, hi = max(0, i - CONTEXT_WORDS), min(len(tokens), i + n + CONTEXT_WORDS)
            found.append(Cue(
                text=cue,
                start_s=float(words[i][0]),
                end_s=float(words[i + n - 1][1]),
                context=" ".join(str(w[2]) for w in words[lo:hi]),
            ))
    if not joined:
        return []
    return sorted(found, key=lambda c: c.start_s)


def resolve_deictic(
    audio_units: Sequence[EvidenceUnit],
    figures: Sequence[EvidenceUnit],
    *,
    encoder: Encoder,
    slide_of_audio: dict[str, int] | None = None,
    mode: Mode = "linkrag",
    cues: Sequence[str] | None = None,
    threshold: float = 0.45,
    max_links_per_unit: int = 8,
    weights: dict[str, float] | None = None,
) -> list[Link]:
    """deictic Links from audio segments to the figures they point at.

    baseline: no alignment; every figure is a candidate, ranked by similarity.
    linkrag:  candidates restricted to the aligned slide (requires slide_of_audio).
    """
    if mode == "linkrag" and not slide_of_audio:
        raise ValueError(
            "linkrag mode needs slide_of_audio from the Phase 2 alignment; "
            "run scripts/build_links.py first or pass mode='baseline'"
        )
    weights = weights or {}
    cues = list(cues) if cues else expand_cues()
    w_slide = weights.get("slide", 0.45)
    w_cue = weights.get("cue", 0.15)
    w_dense = weights.get("dense", 0.20)
    w_overlap = weights.get("overlap", 0.20)
    # Normalise by the weight mass each mode can actually earn. baseline has no
    # slide term, so on a raw scale its maximum is w_cue+w_dense=0.5 against a
    # 0.45 threshold -- it would score ~0 links by construction and the ablation
    # would flatter linkrag for a reason that has nothing to do with alignment.
    mass = ((w_slide + w_cue + w_dense + w_overlap) if mode == "linkrag"
            else (w_cue + w_dense + w_overlap))
    mass = max(mass, 1e-9)

    with stage_timer("link.deictic", mode=mode, audio=len(audio_units),
                     figures=len(figures)) as t:
        if not figures:
            t["links"] = 0
            return []

        fig_vec = np.asarray(encoder([embeddable_text(f) for f in figures]), dtype="float32")
        fig_vec /= np.maximum(np.linalg.norm(fig_vec, axis=1, keepdims=True), 1e-9)

        contexts: list[tuple[int, Cue]] = []
        for i, unit in enumerate(audio_units):
            for cue in find_cues(unit, cues):
                contexts.append((i, cue))
        if not contexts:
            t["cues"] = 0
            t["links"] = 0
            return []

        ctx_vec = np.asarray(encoder([c.context for _, c in contexts]), dtype="float32")
        ctx_vec /= np.maximum(np.linalg.norm(ctx_vec, axis=1, keepdims=True), 1e-9)
        dense = ctx_vec @ fig_vec.T

        # Keyword overlap against the figure's OCR/caption text: the spec's
        # tie-break when a slide carries several figures. IDF-weighted so a shared
        # "recall" counts and a shared "the" does not.
        fig_tokens = [set(tokenize(embeddable_text(f))) for f in figures]
        idf = _idf([list(t) for t in fig_tokens])
        fig_mass = [max(sum(idf.get(t, 0.0) for t in s_), 1e-9) for s_ in fig_tokens]

        links: list[Link] = []
        per_unit: dict[str, int] = {}
        for row, (i, cue) in enumerate(contexts):
            unit = audio_units[i]
            # baseline must not see the alignment at all. An earlier version still
            # added w_slide * on_slide here, so the "no alignment" ablation was
            # quietly using the alignment and measured nothing.
            page = (slide_of_audio or {}).get(unit.id) if mode == "linkrag" else None
            scored = []
            for j, fig in enumerate(figures):
                on_slide = 1.0 if (page is not None and fig.location.page == page) else 0.0
                if mode == "linkrag" and on_slide == 0.0:
                    continue  # the alignment says this figure was not on screen
                shared = set(tokenize(cue.context)) & fig_tokens[j]
                overlap = min(sum(idf.get(t, 0.0) for t in shared) / fig_mass[j], 1.0) if shared else 0.0
                score = (w_slide * on_slide + w_cue * cue.strength
                         + w_dense * float(dense[row, j])
                         + w_overlap * overlap) / mass
                scored.append((score, j, overlap))
            scored.sort(reverse=True)
            for score, j, overlap in scored:
                if score < threshold:
                    break
                if per_unit.get(unit.id, 0) >= max_links_per_unit:
                    break
                links.append(Link(
                    unit.id, figures[j].id, "deictic", score,
                    metadata={
                        "phrase": cue.text,
                        "phrase_start_s": round(cue.start_s, 2),
                        "phrase_end_s": round(cue.end_s, 2),
                        "context": cue.context,
                        "slide_page": page,
                        "keyword_overlap": round(float(overlap), 4),
                    },
                ))
                per_unit[unit.id] = per_unit.get(unit.id, 0) + 1
        t["cues"] = len(contexts)
        t["links"] = len(links)
    return links


def slide_map_from_links(links: Sequence[Link], slide_units: Sequence[EvidenceUnit]) -> dict[str, int]:
    """audio id -> slide page, from Phase 2's audio_slide links."""
    page_of = {u.id: u.location.page for u in slide_units}
    return {l.src_id: page_of[l.dst_id] for l in links
            if l.link_type == "audio_slide" and page_of.get(l.dst_id) is not None}
