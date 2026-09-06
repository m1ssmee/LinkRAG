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
from linkrag.index import Index, default_encoder
from linkrag.link.align import align, build_links, save_links
from linkrag.link.deictic import expand_cues, resolve_deictic, slide_map_from_links
from linkrag.link.figure_text import link_figures_to_text
from linkrag.link.graph import build_graph, export_graphml, summarise


def split_units(index: Index, audio_source: str | None, slide_source: str | None):
    audio = [u for u in index.units if u.modality == "audio"]
    slides = [u for u in index.units if u.modality == "text" and u.location.page is not None]
    figures = [u for u in index.units if u.modality == "figure"]
    if audio_source:
        audio = [u for u in audio if Path(u.source_file).name == audio_source]
    if slide_source:
        slides = [u for u in slides if Path(u.source_file).name == slide_source]
    if slide_source:
        figures = [u for u in figures if Path(u.source_file).name == slide_source]
    audio.sort(key=lambda u: (u.location.start_s or 0.0))
    slides.sort(key=lambda u: (u.location.page or 0))
    figures.sort(key=lambda u: (u.location.page or 0, u.id))
    return audio, slides, figures


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--index", default=None)
    ap.add_argument("--method", choices=["monotonic", "naive"], default=None)
    ap.add_argument("--audio-source", default=None, help="basename, e.g. hsieh.mp3")
    ap.add_argument("--slide-source", default=None, help="basename, e.g. slides.pdf")
    ap.add_argument("--links", default=None)
    ap.add_argument("--npz", default=None)
    ap.add_argument("--link-mode", choices=["baseline", "linkrag"], default="linkrag",
                    help="baseline: same-page figures, no alignment for deixis")
    args = ap.parse_args(argv)

    setup_logging()
    cfg = load_config(args.config)
    acfg = cfg["link"]["align"]
    method = args.method or acfg["method"]

    index = Index.load(args.index or cfg["index"]["store_dir"])
    audio, slides, figures = split_units(index, args.audio_source, args.slide_source)
    if not audio or not slides:
        raise SystemExit(f"need both modalities: {len(audio)} audio, {len(slides)} slide units")

    encoder = default_encoder(index.embedding_model, cfg["device"], index.normalize)
    with stage_timer("encoder.warmup", model=index.embedding_model):
        encoder([""])

    result = align(
        audio, slides, encoder=encoder, method=method,
        weights=acfg["weights"], jump_penalty=acfg["jump_penalty"],
        skip_penalty=acfg["skip_penalty"], back_penalty=acfg["back_penalty"],
        max_back=acfg["max_back"],
    )
    links = build_links(audio, slides, result, min_score=acfg["min_score"])
    audio_slide = list(links)

    lcfg = cfg["link"]
    fcfg = lcfg["figure_text"]
    links += link_figures_to_text(
        figures, slides, encoder=encoder, mode=args.link_mode,
        threshold=lcfg["figure_text_threshold"],
        max_links_per_unit=lcfg["max_links_per_unit"],
        weights=fcfg["weights"], page_decay=fcfg["page_decay"],
        layout_max_gap_pt=fcfg["layout_max_gap_pt"],
        reference_page_window=fcfg["reference_page_window"],
    )
    dcfg = lcfg["deictic"]
    links += resolve_deictic(
        audio, figures, encoder=encoder,
        slide_of_audio=slide_map_from_links(audio_slide, slides),
        mode=args.link_mode,
        cues=expand_cues(lcfg["deictic_cues"], dcfg["object_nouns"],
                         dcfg["directions"], dcfg["determiners"]),
        threshold=lcfg["deictic_threshold"],
        max_links_per_unit=lcfg["max_links_per_unit"],
        weights=dcfg["weights"],
    )

    links_path = Path(args.links or acfg["links_path"])
    if method != acfg["method"] and args.links is None:
        links_path = links_path.with_name(links_path.stem + f"_{method}" + links_path.suffix)
    save_links(links, links_path)

    npz = Path(args.npz or links_path.with_suffix(".npz"))
    np.savez(npz, similarity=result.similarity, path=np.asarray(result.path),
             audio_ids=np.asarray([u.id for u in audio]),
             slide_ids=np.asarray([u.id for u in slides]),
             audio_start=np.asarray([u.location.start_s for u in audio], dtype="float64"),
             audio_end=np.asarray([u.location.end_s for u in audio], dtype="float64"),
             slide_pages=np.asarray([u.location.page for u in slides]))

    graph = build_graph(list(index.units), links)
    graphml = export_graphml(graph, links_path.with_suffix(".graphml"))

    stats = summarise(graph)
    print(f"align={method} link_mode={args.link_mode}  audio={len(audio)} "
          f"slides={len(slides)} figures={len(figures)}  links={len(links)}")
    print(f"  {'link_type':18} {'count':>6} {'avg score':>10}")
    for name in sorted(stats):
        print(f"  {name:18} {int(stats[name]['count']):>6} {stats[name]['mean_score']:>10.4f}")
    if graph.graph.get("dropped_links"):
        print(f"  WARNING {graph.graph['dropped_links']} link(s) referenced unknown units")
    print(f"  slides covered   : {result.slides_used()}/{len(slides)}")
    print(f"  back-jumps       : {result.back_jumps()}")
    print(f"  mean align score : {np.mean(result.scores()):.4f}")
    print(f"  wrote {links_path}, {npz}, {graphml}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
