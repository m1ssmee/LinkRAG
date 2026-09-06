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
from functools import lru_cache
from pathlib import Path
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

# Words that mean the speaker is talking about something on screen. Used only to
# separate tier 2 from tier 3: a bare "this" next to "you can see on the graph" is
# far better evidence of visual deixis than a bare "this" in open prose.
DEFAULT_VISUAL_TERMS = [
    "see", "seen", "show", "shows", "showing", "shown", "look", "looking", "plot",
    "plotted", "screen", "slide", "figure", "graph", "diagram", "arrow", "chart",
    "picture", "image", "axis", "curve", "table", "box", "column", "row", "highlight",
    "highlighted", "display", "displayed", "draw", "drawn", "point", "pointing",
]

# Tier 1 explicit object deixis, tier 2 pronoun + visual context, tier 3 bare pronoun.
DEFAULT_TIER_WEIGHTS = {1: 1.0, 2: 0.6, 3: 0.3}

TERMINAL_PUNCT = re.compile(r"[.!?][\"')\]]?$")


@dataclass(frozen=True)
class Cue:
    text: str
    start_s: float
    end_s: float
    context: str
    sentence: str = ""
    tier: int = 3

    @property
    def strength(self) -> float:
        """Word-count heuristic, kept for reporting. Ranking uses `tier` instead --
        see `tier_of`, which is the evidence-based replacement."""
        return min(1.0, len(self.text.split()) / 3.0)


def tier_of(phrase: str, sentence: str, visual_terms: Sequence[str]) -> int:
    """1 = explicit object deixis, 2 = bare pronoun with visual context, 3 = bare pronoun.

    On pilot01 every emitted deictic link came from a bare "this"/"that", so an
    undifferentiated cue set says almost nothing about whether the speaker was
    actually pointing at anything. The tier is what makes that visible in the output
    instead of hiding it in an average.
    """
    if len(phrase.split()) > 1:
        return 1
    words = {re.sub(r"[^a-z0-9']", "", w.lower()) for w in sentence.split()}
    return 2 if words & {t.lower() for t in visual_terms} else 3


@lru_cache(maxsize=8)
def _cue_pattern(cues: tuple[str, ...]) -> re.Pattern[str]:
    """One alternation, longest phrase first, so `re` picks the longest match at
    each position and `finditer` handles non-overlap. Replaces a phrases x tokens
    sliding window with a manual `claimed` span list."""
    ordered = sorted({c.lower() for c in cues}, key=lambda c: (-len(c.split()), c))
    return re.compile(r"(?<!\S)(?:" + "|".join(re.escape(c) for c in ordered) + r")(?!\S)")


def find_cues(
    unit: EvidenceUnit,
    cues: Sequence[str] = DEFAULT_CUES,
    visual_terms: Sequence[str] = DEFAULT_VISUAL_TERMS,
) -> list[Cue]:
    """Locate deictic cues and their surrounding words, using word timestamps.

    Longer cues win over the shorter ones they contain, so "as you can see" is not
    also reported as "see"/"this".
    """
    words = unit.metadata.get("words") or []
    if not words:
        return []
    tokens = [re.sub(r"[^a-z0-9']", "", str(w[2]).lower()) for w in words]

    # char offset -> token index, so a regex match maps back onto word timestamps
    starts, offset = [], 0
    for tok in tokens:
        starts.append(offset)
        offset += len(tok) + 1
    joined = " ".join(tokens)
    index_of = {c: i for i, c in enumerate(starts)}

    found: list[Cue] = []
    for match in _cue_pattern(tuple(cues)).finditer(joined):
        i = index_of.get(match.start())
        if i is None:
            continue
        n = len(match.group(0).split())
        lo, hi = max(0, i - CONTEXT_WORDS), min(len(tokens), i + n + CONTEXT_WORDS)

        s_lo = i
        while s_lo > 0 and not TERMINAL_PUNCT.search(str(words[s_lo - 1][2])):
            s_lo -= 1
        s_hi = i + n - 1
        while s_hi < len(words) - 1 and not TERMINAL_PUNCT.search(str(words[s_hi][2])):
            s_hi += 1
        sentence = " ".join(str(w[2]) for w in words[s_lo:s_hi + 1])

        found.append(Cue(
            text=match.group(0),
            start_s=float(words[i][0]),
            end_s=float(words[i + n - 1][1]),
            context=" ".join(str(w[2]) for w in words[lo:hi]),
            sentence=sentence,
            tier=tier_of(match.group(0), sentence, visual_terms),
        ))
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
    tier_weights: dict[int, float] | None = None,
    visual_terms: Sequence[str] = DEFAULT_VISUAL_TERMS,
) -> list[Link]:
    """deictic Links from audio segments to the figures they point at.

    **`threshold` does not filter in linkrag mode.** Line ~250 discards every
    off-slide figure, so `on_slide == 1.0` for every candidate that gets scored, and
    `mass` includes `w_slide`. Every candidate therefore starts at
    `w_slide / mass = 0.45`, which is the default threshold -- measured floor across
    98 pilot links was 0.5622, none below 0.50. Treat the emitted set as
    **unfiltered**: it is "every cue on a slide that has a figure", capped by
    `max_links_per_unit`. Not retuned deliberately; changing it changes every
    recorded deictic number.

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
    # int keys, tolerating YAML that hands them back as strings
    tier_weights = {int(k): float(v) for k, v in (tier_weights or DEFAULT_TIER_WEIGHTS).items()}
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
            for cue in find_cues(unit, cues, visual_terms):
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
                # Tier, not word count: an explicit "this arrow here" outranks a
                # bare "this" even when the visual evidence is identical.
                cue_weight = tier_weights.get(cue.tier, 0.0)
                score = (w_slide * on_slide + w_cue * cue_weight
                         + w_dense * float(dense[row, j])
                         + w_overlap * overlap) / mass
                scored.append((score, j, overlap))
            # (-score, j): ties break by ascending figure index. Plain
            # reverse-sort on the tuple broke them by *descending* index.
            scored.sort(key=lambda t: (-t[0], t[1]))
            for score, j, overlap in scored:
                if score < threshold:
                    break
                if per_unit.get(unit.id, 0) >= max_links_per_unit:
                    break
                links.append(Link(
                    unit.id, figures[j].id, "deictic", score,
                    metadata={
                        "phrase": cue.text,
                        "tier": cue.tier,
                        "sentence": cue.sentence,
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


def deictic_pairs(links: Sequence[Link]) -> list[dict]:
    """Collapse deictic links to distinct (segment, figure) pairs, keeping the
    best-scoring cue for each.

    The raw link count double-counts: one segment can fire six cues at the same
    figure, which on pilot01 turned 22 real pairs into 41 links. Pairs are the
    honest unit for "how many things did we resolve"; the cue count is a secondary
    statistic about how noisy the trigger was.
    """
    best: dict[tuple[str, str], dict] = {}
    for link in links:
        if link.link_type != "deictic":
            continue
        key = (link.src_id, link.dst_id)
        current = best.get(key)
        if current is None or link.score > current["score"]:
            # carry the running cue count across the replacement, or a pair whose
            # best cue arrives second reports 1 instead of its true count
            seen = current["cues"] if current else 0
            best[key] = {
                "src_id": link.src_id,
                "dst_id": link.dst_id,
                "score": float(link.score),
                "tier": int(link.metadata.get("tier", 3)),
                "phrase": link.metadata.get("phrase", ""),
                "sentence": link.metadata.get("sentence", ""),
                "slide_page": link.metadata.get("slide_page"),
                "phrase_start_s": link.metadata.get("phrase_start_s"),
                "cues": seen,
            }
        best[key]["cues"] += 1
    return sorted(best.values(), key=lambda r: (-r["score"], r["src_id"]))


def tier_breakdown(pairs: Sequence[dict]) -> dict[int, dict[str, float]]:
    """Per-tier count and mean score over distinct pairs."""
    out: dict[int, dict[str, float]] = {}
    for row in pairs:
        entry = out.setdefault(int(row["tier"]), {"count": 0, "total": 0.0})
        entry["count"] += 1
        entry["total"] += row["score"]
    for entry in out.values():
        entry["mean_score"] = entry["total"] / entry["count"] if entry["count"] else 0.0
        del entry["total"]
    return dict(sorted(out.items()))


def write_pairs_csv(
    pairs: Sequence[dict],
    units_by_id: dict[str, EvidenceUnit],
    path: str | Path,
) -> Path:
    """One row per (segment, figure) pair, for checking against ear labels."""
    import csv

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["segment_id", "segment_start_s", "segment_end_s", "phrase",
                         "phrase_start_s", "tier", "cues", "figure_id", "figure_page",
                         "score", "sentence"])
        for row in pairs:
            seg = units_by_id.get(row["src_id"])
            fig = units_by_id.get(row["dst_id"])
            writer.writerow([
                row["src_id"],
                f"{seg.location.start_s:.2f}" if seg and seg.location.start_s is not None else "",
                f"{seg.location.end_s:.2f}" if seg and seg.location.end_s is not None else "",
                row["phrase"],
                f"{row['phrase_start_s']:.2f}" if row.get("phrase_start_s") is not None else "",
                row["tier"], row["cues"], row["dst_id"],
                fig.location.page if fig and fig.location.page is not None else row.get("slide_page", ""),
                f"{row['score']:.4f}",
                " ".join((row.get("sentence") or "").split()),
            ])
    return path
