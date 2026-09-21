#!/usr/bin/env python3
"""Run audio-to-slide alignment over an index and persist the links.

Writes data/processed/links.jsonl plus an .npz holding the similarity matrix and
chosen path, so scripts/plot_alignment.py and scripts/eval_alignment.py can reuse
them without paying the embedding cost again.

    python scripts/build_links.py --method monotonic
    python scripts/build_links.py --method naive --links data/processed/links_naive.jsonl
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from linkrag.core import load_config, setup_logging, stage_timer
from linkrag.manifest import MANIFEST_NAME, load_manifest
from linkrag.index import Index, default_encoder
from linkrag.link.align import align, align_monotonic, align_naive, build_links, save_links
from linkrag.link.deictic import (
    deictic_pairs,
    expand_cues,
    resolve_deictic,
    slide_map_from_links,
    tier_breakdown,
    write_pairs_csv,
)
from linkrag.link.same_slide import link_same_slide
from linkrag.link.figure_text import link_figures_to_text
from linkrag.link.graph import build_graph, export_graphml, summarise


def split_units(index: Index, audio_source: str | None, slide_source: str | None):
    """audio, deck-slide text, all paged text, all figures.

    `slides` is the alignment target and must be *deck* pages only: a lecturer walks
    through a deck, not through a paper. With the OSDI paper in the corpus, taking
    every paged text unit would have aligned 51 audio segments against 111 candidates
    spanning two unrelated documents. `texts` stays corpus-wide, because figure_text
    is supposed to reach across documents.
    """
    audio = [u for u in index.units if u.modality == "audio"]
    paged = [u for u in index.units if u.modality == "text" and u.location.page is not None]
    decks = [u for u in paged if u.metadata.get("slide_deck")]
    slides = decks or paged  # fall back when nothing is flagged as a deck
    figures = [u for u in index.units if u.modality == "figure"]
    if audio_source:
        audio = [u for u in audio if Path(u.source_file).name == audio_source]
    if slide_source:
        slides = [u for u in slides if Path(u.source_file).name == slide_source]
    audio.sort(key=lambda u: (u.location.start_s or 0.0))
    slides.sort(key=lambda u: (u.location.page or 0))
    paged.sort(key=lambda u: (Path(u.source_file).name, u.location.page or 0))
    figures.sort(key=lambda u: (Path(u.source_file).name, u.location.page or 0, u.id))
    return audio, slides, paged, figures


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--index", default=None)
    ap.add_argument("--method", choices=["monotonic", "naive"], default=None)
    ap.add_argument("--audio-source", default=None, help="basename, e.g. hsieh.mp3")
    ap.add_argument("--slide-source", default=None, help="basename, e.g. slides.pdf")
    ap.add_argument("--links", default=None)
    ap.add_argument("--npz", default=None)
    ap.add_argument("--pairs-csv", default=None)
    ap.add_argument("--link-mode", choices=["baseline", "linkrag"], default="linkrag",
                    help="baseline: same-page figures, no alignment for deixis")
    args = ap.parse_args(argv)

    setup_logging()
    cfg = load_config(args.config)
    acfg = cfg["link"]["align"]
    method = args.method or acfg["method"]

    index = Index.load(args.index or cfg["index"]["store_dir"])
    audio, slides, texts, figures = split_units(index, args.audio_source, args.slide_source)
    if not audio or not slides:
        raise SystemExit(f"need both modalities: {len(audio)} audio, {len(slides)} slide units")

    encoder = default_encoder(index.embedding_model, cfg["device"], index.normalize)
    with stage_timer("encoder.warmup", model=index.embedding_model):
        encoder([""])

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
                        penalties=dict(jump_penalty=acfg["jump_penalty"], skip_penalty=acfg["skip_penalty"],
                                       back_penalty=acfg["back_penalty"]))
    gate = build_links.gate
    unrelated_pairs: list[dict] = []
    if gate is not None:
        print(f"audio_slide relatedness gate: score {gate['score']:.3f} vs shuffled "
              f"{gate['null_mean']:.3f} ± {gate['null_std']:.3f} -> z = {gate['z']:.1f} "
              f"({'related' if gate['related'] else 'UNRELATED: no audio_slide links emitted'})")
        if not gate["related"]:
            unrelated_pairs.append({"audio": Path(audio[0].source_file).name,
                                    "deck": Path(slides[0].source_file).name, "link_type": "audio_slide",
                                    **{k: round(v, 4) for k, v in gate.items() if isinstance(v, float)}})
    audio_slide = list(links)

    lcfg = cfg["link"]
    fcfg = lcfg["figure_text"]
    links += link_figures_to_text(
        figures, texts, encoder=encoder, mode=args.link_mode,
        threshold=lcfg["figure_text_threshold"],
        max_links_per_unit=lcfg["max_links_per_unit"],
        weights=fcfg["weights"], page_decay=fcfg["page_decay"],
        layout_max_gap_pt=fcfg["layout_max_gap_pt"],
        reference_page_window=fcfg["reference_page_window"],
        relatedness_z=acfg.get("relatedness_z"),
    )
    unrelated_pairs += getattr(link_figures_to_text, "unrelated_pairs", [])
    scfg = lcfg["same_slide"]
    links += link_same_slide(figures, texts, mode=args.link_mode,
                             enabled=scfg["enabled"], score=scfg["score"])

    dcfg = lcfg["deictic"]
    slide_map = slide_map_from_links(audio_slide, slides)
    deictic_links = [] if (args.link_mode == "linkrag" and not slide_map) else resolve_deictic(
        audio, figures, encoder=encoder,
        slide_of_audio=slide_map,
        mode=args.link_mode,
        cues=expand_cues(lcfg["deictic_cues"], dcfg["object_nouns"],
                         dcfg["directions"], dcfg["determiners"]),
        threshold=lcfg["deictic_threshold"],
        max_links_per_unit=lcfg["max_links_per_unit"],
        weights=dcfg["weights"],
        tier_weights=dcfg["tier_weights"],
        visual_terms=dcfg["visual_terms"],
    )
    links += deictic_links

    index_dir = Path(args.index or cfg["index"]["store_dir"])
    manifest = load_manifest(index_dir.parent / MANIFEST_NAME) or {}
    manifest_hash = manifest.get("hash")

    links_path = Path(args.links or acfg["links_path"])
    if method != acfg["method"] and args.links is None:
        links_path = links_path.with_name(links_path.stem + f"_{method}" + links_path.suffix)
    save_links(links, links_path, manifest_hash=manifest_hash,
               meta={"unrelated_pairs": unrelated_pairs} if unrelated_pairs else None)

    npz = Path(args.npz or links_path.with_suffix(".npz"))
    np.savez(npz, similarity=result.similarity, path=np.asarray(result.path),
             manifest_hash=np.asarray(manifest_hash or ""),
             audio_ids=np.asarray([u.id for u in audio]),
             slide_ids=np.asarray([u.id for u in slides]),
             audio_start=np.asarray([u.location.start_s for u in audio], dtype="float64"),
             audio_end=np.asarray([u.location.end_s for u in audio], dtype="float64"),
             slide_pages=np.asarray([u.location.page for u in slides]))

    graph = build_graph(list(index.units), links)
    graphml = export_graphml(graph, links_path.with_suffix(".graphml"))

    stats = summarise(graph)
    # Measurement rule 2: per-category counts must sum to the stated total.
    assert sum(int(v["count"]) for v in stats.values()) == len(links), (
        f"per-type counts {sum(int(v['count']) for v in stats.values())} != "
        f"total {len(links)}; links are being dropped between list and graph"
    )
    print(f"corpus manifest {manifest_hash or 'NONE'}")
    print(f"align={method} link_mode={args.link_mode}  audio={len(audio)} "
          f"deck_slides={len(slides)} all_text={len(texts)} figures={len(figures)}  "
          f"links={len(links)}")
    print(f"  {'link_type':18} {'count':>6} {'avg score':>10}")
    for name in sorted(stats):
        print(f"  {name:18} {int(stats[name]['count']):>6} {stats[name]['mean_score']:>10.4f}")
    pairs = deictic_pairs(deictic_links)
    if deictic_links:
        tiers = tier_breakdown(pairs)
        print(f"  deictic: {len(pairs)} distinct (segment, figure) pairs "
              f"from {len(deictic_links)} raw cue hits (UNFILTERED -- the "
              f"threshold cannot reject an on-slide candidate)")
        print(f"    {'tier':<6} {'pairs':>6} {'avg score':>10}")
        for tier in sorted(tiers):
            label = {1: "1 explicit", 2: "2 pron+vis", 3: "3 pronoun"}.get(tier, str(tier))
            print(f"    {label:<12} {int(tiers[tier]['count']):>4} "
                  f"{tiers[tier]['mean_score']:>10.4f}")
        csv_path = write_pairs_csv(pairs, {u.id: u for u in index.units},
                                   args.pairs_csv or "reports/deictic_pairs_pilot01.csv")
        print(f"    wrote {csv_path}")

    if graph.graph.get("dropped_links"):
        print(f"  WARNING {graph.graph['dropped_links']} link(s) referenced unknown units")
    print(f"  slides covered   : {result.slides_used()}/{len(slides)}")
    print(f"  back-jumps       : {result.back_jumps()}")
    print(f"  mean align score : {np.mean(result.scores()):.4f}")
    print(f"  wrote {links_path}, {npz}, {graphml}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
