"""The Evidence Linking Layer as one call: every link type, in the order build_links has
always run them. scripts/build_links.py (pilot01) and the LectQA-Vid frame-slide path
use this same function, so there is one linking code path.

Order matters: deixis resolves against the audio->slide map, so audio_slide comes first.

Alignment is per (recording, deck) pair: each recording is aligned to each deck on its own, with
its own relatedness gate. Several files are never concatenated into one monotone sequence, whose
order would be arbitrary. With one recording and one deck (pilot01, each LectQA-Vid video) this is
the single alignment it always was.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from linkrag.core import EvidenceUnit, Link
from linkrag.link.align import Alignment, align, align_naive, build_links, monotonic_decoder
from linkrag.link.deictic import expand_cues, resolve_deictic, slide_map_from_links
from linkrag.link.figure_text import link_figures_to_text
from linkrag.link.same_slide import link_same_slide


@dataclass
class LinkRun:
    links: list[Link]
    alignment: Alignment | None          # the one pair's alignment; None when there are several pairs
    audio_slide: list[Link]
    deictic: list[Link]
    gate: dict[str, Any] | None          # the one pair's gate; None when there are several (see `gates`)
    unrelated_pairs: list[dict] = field(default_factory=list)
    alignments: dict[tuple[str, str], Alignment] = field(default_factory=dict)   # (recording, deck) ->
    gates: dict[tuple[str, str], dict[str, Any] | None] = field(default_factory=dict)


def by_file(units: Sequence[EvidenceUnit]) -> dict[str, list[EvidenceUnit]]:
    """source file -> its units, in first-seen order."""
    out: dict[str, list[EvidenceUnit]] = {}
    for u in units:
        out.setdefault(u.source_file, []).append(u)
    return out


def link_corpus(audio: Sequence[EvidenceUnit], slides: Sequence[EvidenceUnit], texts: Sequence[EvidenceUnit],
                figures: Sequence[EvidenceUnit], *, encoder: Callable, cfg: dict[str, Any],
                method: str | None = None, link_mode: str = "linkrag") -> LinkRun:
    acfg = cfg["link"]["align"]
    method = method or acfg["method"]
    links: list[Link] = []
    unrelated_pairs: list[dict] = []
    alignments, gates = {}, {}
    for rec, rec_units in by_file(audio).items():
        for deck, deck_units in by_file(slides).items():
            pair_links, alignments[(rec, deck)], gates[(rec, deck)] = _align_pair(
                rec_units, deck_units, encoder=encoder, cfg=cfg, method=method)
            links += pair_links
            gate = gates[(rec, deck)]
            if gate is not None and not gate["related"]:
                unrelated_pairs.append({"audio": Path(rec).name, "deck": Path(deck).name, "link_type": "audio_slide",
                                        **{k: round(v, 4) for k, v in gate.items() if isinstance(v, float)}})
    one = len(alignments) == 1
    result, gate = (next(iter(alignments.values())), next(iter(gates.values()))) if one else (None, None)
    audio_slide = list(links)

    lcfg = cfg["link"]
    fcfg = lcfg["figure_text"]
    links += link_figures_to_text(
        figures, texts, encoder=encoder, mode=link_mode,
        threshold=lcfg["figure_text_threshold"],
        max_links_per_unit=lcfg["max_links_per_unit"],
        weights=fcfg["weights"], page_decay=fcfg["page_decay"],
        layout_max_gap_pt=fcfg["layout_max_gap_pt"],
        reference_page_window=fcfg["reference_page_window"],
        relatedness_z=acfg.get("relatedness_z"), relatedness_shuffles=int(acfg.get("null_shuffles", 5)),
        gate_cache_dir=fcfg.get("gate_cache_dir"),
    )
    unrelated_pairs += getattr(link_figures_to_text, "unrelated_pairs", [])
    scfg = lcfg["same_slide"]
    links += link_same_slide(figures, texts, mode=link_mode, enabled=scfg["enabled"], score=scfg["score"])

    dcfg = lcfg["deictic"]
    slide_map = slide_map_from_links(audio_slide, slides)
    deictic_links = [] if (link_mode == "linkrag" and not slide_map) else resolve_deictic(
        audio, figures, encoder=encoder,
        slide_of_audio=slide_map,
        mode=link_mode,
        cues=expand_cues(lcfg["deictic_cues"], dcfg["object_nouns"], dcfg["directions"], dcfg["determiners"]),
        threshold=lcfg["deictic_threshold"],
        max_links_per_unit=lcfg["max_links_per_unit"],
        weights=dcfg["weights"],
        tier_weights=dcfg["tier_weights"],
        visual_terms=dcfg["visual_terms"],
    )
    links += deictic_links
    return LinkRun(links=links, alignment=result, audio_slide=audio_slide, deictic=deictic_links,
                   gate=gate, unrelated_pairs=unrelated_pairs, alignments=alignments, gates=gates)


def _align_pair(audio: Sequence[EvidenceUnit], slides: Sequence[EvidenceUnit], *, encoder: Callable,
                cfg: dict[str, Any], method: str) -> tuple[list[Link], Alignment, dict[str, Any] | None]:
    """One recording against one deck: alignment, gated audio_slide links, and the gate."""
    acfg = cfg["link"]["align"]
    result = align(
        audio, slides, encoder=encoder, method=method,
        weights=acfg["weights"], jump_penalty=acfg["jump_penalty"],
        skip_penalty=acfg["skip_penalty"], back_penalty=acfg["back_penalty"],
        max_back=acfg["max_back"], start_prior_mu=acfg.get("start_prior_mu", 0.0),
        flatness_scaling=acfg.get("flatness_scaling", 0.0),
        similarity=acfg.get("similarity", "ours"), fusion_weight=acfg.get("fusion_weight", 0.5),
        device=cfg["device"],
    )
    decode = align_naive if method == "naive" else monotonic_decoder(acfg)
    links = build_links(audio, slides, result, min_score=acfg["min_score"],
                        min_segment_sim=acfg.get("min_segment_sim"),
                        relatedness_z=acfg.get("relatedness_z"), decode=decode,
                        relatedness_shuffles=int(acfg.get("null_shuffles", 5)),
                        penalties={"jump_penalty": acfg["jump_penalty"], "skip_penalty": acfg["skip_penalty"],
                                   "back_penalty": acfg["back_penalty"],
                                   "null_std_floor": acfg.get("null_std_floor", 0.0)})
    return links, result, build_links.gate
