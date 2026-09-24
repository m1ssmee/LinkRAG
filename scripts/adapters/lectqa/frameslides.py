"""Frame-derived slide units and their links; localisation with and without them."""

from __future__ import annotations

import collections
import json
from pathlib import Path


from linkrag.index import Index, build_index, default_encoder
from linkrag.link.align import save_links
from linkrag.manifest import MANIFEST_NAME, build_manifest, write_manifest

from lectqa.common import PROCESSED, RAW
from lectqa.localisation import localisation_table, localise


def build_frameslides(vid: str, cfg: dict, encoder) -> dict | None:
    """index_frameslides for one prepared video: its audio units + slide page/figure units
    derived from the frames, and the full linking layer (link_corpus) over them."""
    from linkrag.ingest.video_slides import slide_at, slide_units
    from linkrag.link.align import align_naive
    from linkrag.link.pipeline import link_corpus
    video = RAW / "videos" / f"{vid}.video.mp4"
    base = PROCESSED / vid / "index"
    if not video.exists() or not base.exists():
        return None
    out = PROCESSED / vid / "index_frameslides"
    cached = out / "frameslides_stats.json"          # built before: reuse (delete the dir to rebuild)
    if cached.exists() and (out / "links.jsonl").exists():
        return json.loads(cached.read_text())
    audio = sorted((u for u in Index.load(base).units if u.modality == "audio"), key=lambda u: u.location.start_s or 0.0)
    units, slides = slide_units(video, vid, PROCESSED / vid / "frameslides")
    pages = sorted((u for u in units if u.modality == "text"), key=lambda u: u.location.page)
    figures = [u for u in units if u.modality == "figure"]
    all_units = [*audio, *units]
    index = build_index(all_units, encoder=encoder, embedding_model=cfg["models"]["embedding"],
                        device=cfg["device"], normalize=cfg["index"]["normalize_embeddings"])
    index.save(out)
    manifest = build_manifest([RAW / "videos" / f"{vid}.m4a", video], all_units)
    write_manifest(manifest, out / MANIFEST_NAME)
    if not audio or not pages:
        return {"vid": vid, "slides": len(slides), "figures": len(figures), "links": {}, "aligned": 0, "n_truth": 0}
    run = link_corpus(audio, pages, pages, figures, encoder=encoder, cfg=cfg)
    save_links(run.links, out / "links.jsonl", manifest_hash=manifest.get("hash"))
    # free sanity check: the slide on screen at each audio segment's midpoint is known
    truth = [slide_at(slides, ((u.location.start_s or 0) + (u.location.end_s or 0)) / 2) for u in audio]
    naive = align_naive(run.alignment.similarity)
    ok = [(t, p, nv) for t, p, nv in zip(truth, run.alignment.path, naive) if t is not None]
    stats = {"vid": vid, "slides": len(slides), "figures": len(figures), "audio": len(audio),
             "links": dict(collections.Counter(l.link_type for l in run.links)), "n_links": len(run.links),
             "gate": run.gate, "n_truth": len(ok), "aligned": sum(p == t for t, p, _ in ok),
             "naive": sum(nv == t for t, _, nv in ok)}
    cached.write_text(json.dumps(stats, default=float))
    return stats


def frameslides(ids: list[str], cfg: dict, dcfg: dict, out: Path) -> int:
    """Build index_frameslides for every prepared video, then compare localisation with and
    without frame-derived slides (retrieval only, no LLM: baseline and linkrag modes)."""
    import statistics as st
    encoder = default_encoder(cfg["models"]["embedding"], cfg["device"], cfg["index"]["normalize_embeddings"])
    encoder([""])
    stats = []
    for vid in ids:
        s = build_frameslides(vid, cfg, encoder)
        if s:
            stats.append(s)
            print(f"frameslides {vid}: {s['slides']} slides, {s['figures']} figures, links {s['links']}, "
                  f"aligned {s['aligned']}/{s['n_truth']}")
    done = [s["vid"] for s in stats]
    modes = ("baseline", "linkrag")
    without, info = localise(done, cfg, dcfg, index_name="index", label="without", modes=modes)
    with_, _ = localise(done, cfg, dcfg, index_name="index_frameslides", label="with", modes=modes, link_types=None)
    gold = info["gold"]

    link_tot = collections.Counter()
    for s in stats:
        link_tot.update(s["links"])
    n_links = sum(s.get("n_links", 0) for s in stats)
    assert sum(link_tot.values()) == n_links, "per-type link counts must sum to the total (measurement rule 2)"
    n_truth, aligned, naive = (sum(s[k] for s in stats) for k in ("n_truth", "aligned", "naive"))
    gated_out = sum(1 for s in stats if s.get("gate") and not s["gate"].get("related"))
    L = ["# LectQA-Vid — frame-derived slide units", "",
         f"{len(done)} prepared videos · slides from frames at 1 fps (dHash segmentation, transitions merged, "
         f"revisits deduplicated), OCR tesseract, figures by the deck clustering rules on raster frames "
         f"(`linkrag.ingest.video_slides`) · links from `linkrag.link.pipeline.link_corpus` (the pilot01 path). "
         f"**Retrieval only, no LLM, $0.** Iterative modes need a follow-up LLM call and are not run.", "",
         "*Without* = the prepared index: audio + one OCR unit per changed frame, joined by temporal "
         "co-occurrence links (T1's own signal). *With* = the same audio units + frame-derived slide page and "
         "figure units, joined by the linking layer (alignment, figure_text, same_slide, deictic); no temporal "
         "links, so the alignment has to find the slides from text. A revisited slide counts a hit on any of "
         "its on-screen intervals.", "",
         "## Slides, figures and links", "",
         f"- slides {sum(s['slides'] for s in stats)}, figure units {sum(s['figures'] for s in stats)}, "
         f"audio units {sum(s.get('audio', 0) for s in stats)}",
         f"- relatedness gate rejected the audio↔slide alignment on {gated_out} of {len(stats)} videos", "",
         "| link type | count |", "|---|---:|", *[f"| {k} | {v} |" for k, v in sorted(link_tot.items())],
         f"| **total** | {n_links} |", "",
         "## Alignment against the known on-screen interval (free sanity check)", "",
         f"Audio segments whose midpoint falls inside a slide's interval: n = {n_truth}. "
         f"DP alignment (config `link.align`) picks the on-screen slide for **{aligned}/{n_truth} = "
         f"{aligned / max(n_truth, 1):.1%}**; naive argmax on the same matrix {naive}/{n_truth} = "
         f"{naive / max(n_truth, 1):.1%}. (Their alignment component is T2, MaViLS; this is not their metric.)", "",
         "## Modality mix of the retrieved top-8 (rerank none; mean units per set)", "",
         "| variant | mode | audio | text (slide) | figure |", "|---|---|---:|---:|---:|"]
    for label, rows in (("without", without), ("with", with_)):
        for mode in modes:
            xs = [r["mods"] for r in rows if r["mode"] == mode and r["rerank"] == "none"]
            if xs:
                L.append(f"| {label} | {mode} | {st.mean(x.get('audio', 0) for x in xs):.2f} | "
                         f"{st.mean(x.get('text', 0) for x in xs):.2f} | {st.mean(x.get('figure', 0) for x in xs):.2f} |")
    total_q = sum(gold.values())
    L += ["", "## Localisation (hit@k: a retrieved unit overlaps the gold interval)", "",
          f"Questions with a usable gold interval: **{gold.get('ok', 0)} of {total_q}**. Excluded: "
          + ", ".join(f"{k} {v}" for k, v in sorted(gold.items()) if k != "ok") + ". The published "
          "annotations mix timestamp conventions (HH:MM:SS, SS:cc, SSS:cc, M:SSS:cc) whose meaning differs "
          "between videos, and some stamps end past the fetched video, so only unambiguous, in-range "
          "intervals are scored (`strict_seconds`, `gold_interval`).", ""]
    L += localisation_table(without, "without frame-derived slides") + [""]
    L += localisation_table(with_, "with frame-derived slides") + [""]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n")
    (out.with_suffix(".json")).write_text(json.dumps({"stats": stats, "without": without, "with": with_}, default=str))
    print("\n".join(L))
    return 0
