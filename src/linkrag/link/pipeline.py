"""The Evidence Linking Layer as one call: every link type, in the order build_links has
always run them. scripts/build_links.py (pilot01) and the LectQA-Vid frame-slide path
use this same function, so there is one linking code path.

Order matters: deixis resolves against the audio->slide map, so audio_slide comes first.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from linkrag.core import EvidenceUnit, Link
from linkrag.link.align import Alignment, align, align_monotonic, align_naive, build_links
from linkrag.link.deictic import expand_cues, resolve_deictic, slide_map_from_links
from linkrag.link.figure_text import link_figures_to_text
from linkrag.link.same_slide import link_same_slide


@dataclass
class LinkRun:
    links: list[Link]
    alignment: Alignment
    audio_slide: list[Link]
    deictic: list[Link]
    gate: dict[str, Any] | None
    unrelated_pairs: list[dict] = field(default_factory=list)


def link_corpus(audio: Sequence[EvidenceUnit], slides: Sequence[EvidenceUnit], texts: Sequence[EvidenceUnit],
                figures: Sequence[EvidenceUnit], *, encoder: Callable, cfg: dict[str, Any],
                method: str | None = None, link_mode: str = "linkrag") -> LinkRun:
    acfg = cfg["link"]["align"]
    method = method or acfg["method"]
    result = align(
        audio, slides, encoder=encoder, method=method,
        weights=acfg["weights"], jump_penalty=acfg["jump_penalty"],
        skip_penalty=acfg["skip_penalty"], back_penalty=acfg["back_penalty"],
        max_back=acfg["max_back"], start_prior_mu=acfg.get("start_prior_mu", 0.0),
        flatness_scaling=acfg.get("flatness_scaling", 0.0),
        similarity=acfg.get("similarity", "ours"), fusion_weight=acfg.get("fusion_weight", 0.5),
        device=cfg["device"],
    )
    decode = (align_naive if method == "naive" else
              lambda S: align_monotonic(S, jump_penalty=acfg["jump_penalty"], skip_penalty=acfg["skip_penalty"],
                                        back_penalty=acfg["back_penalty"], max_back=acfg["max_back"],
                                        start_prior_mu=acfg.get("start_prior_mu", 0.0),
                                        flatness_scaling=acfg.get("flatness_scaling", 0.0)))
    links = build_links(audio, slides, result, min_score=acfg["min_score"],
                        min_segment_sim=acfg.get("min_segment_sim"),
                        relatedness_z=acfg.get("relatedness_z"), decode=decode,
                        relatedness_shuffles=int(acfg.get("null_shuffles", 5)),
                        penalties={"jump_penalty": acfg["jump_penalty"], "skip_penalty": acfg["skip_penalty"],
                                   "back_penalty": acfg["back_penalty"],
                                   "null_std_floor": acfg.get("null_std_floor", 0.0)})
    gate = build_links.gate
    unrelated_pairs: list[dict] = []
    if gate is not None and not gate["related"]:
        unrelated_pairs.append({"audio": Path(audio[0].source_file).name,
                                "deck": Path(slides[0].source_file).name, "link_type": "audio_slide",
                                **{k: round(v, 4) for k, v in gate.items() if isinstance(v, float)}})
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
        relatedness_z=acfg.get("relatedness_z"),
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
                   gate=gate, unrelated_pairs=unrelated_pairs)
