#!/usr/bin/env python3
"""MaViLS adapter -- priority (iv). Target T2 (Anderer, Reich, Wölfel, Interspeech 2024).

    git clone --depth 1 https://github.com/andererka/MaViLS data/raw/mavils
    python scripts/adapters/mavils.py --repo data/raw/mavils

Their setting: every spoken sentence of 20 lectures is hand-labelled with the slide
on screen (-1 = no slide). Their metric: micro P/R/F1 over sentences with a label,
`labels` = the label set of the ground truth (sklearn, `average="micro"`), computed
per lecture and averaged. This adapter reproduces that metric exactly and runs

    ours       align_monotonic over S = w_d cos + w_b BM25 + w_k IDF-overlap
    naive      per-sentence argmax of the same S (no sequence structure)

from **transcript + slide PDF only** -- no video frames -- so the like-for-like row
in their Table 1 is the *Audio* column (average 0.53). Their 0.82 uses OCR of the
video frames and image features we do not consume; it is printed as the ceiling.

Sentences are taken from their ground-truth files verbatim (their segmentation, their
transcript), so the only thing under test is the alignment.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from linkrag.core import EvidenceUnit, Location, load_config, setup_logging
from linkrag.index import default_encoder
from linkrag.ingest.pdf import ingest_pdf
from linkrag.link.align import align_monotonic, align_naive, similarity_matrix

# ground-truth file stem -> (slides PDF, their Table 1 name, Table 1 audio F1, Table 2 combined F1 @ lambda 0.1)
LECTURES = {
    "deeplearning":                      ("deeplearning_goodfellow.pdf",       "Deep learning",       0.79, 0.99),
    "short_range":                       ("short_range_mit.pdf",               "Short range",         0.57, 0.80),
    "numerics":                          ("numerics.pdf",                      "Numerics",            0.46, 0.81),
    "reinforcement_learning":            ("reinforcement_learning.pdf",        "Reinforcement",       0.31, 0.75),
    "computer_vision_2_2":               ("computer_vision_2_2_geiger.pdf",    "Computer Vision",     0.65, 1.00),
    "climate_and_cities":                ("cities&climate.pdf",                "Climate & Cities",    0.49, 0.86),
    "cities_and_decarbonization":        ("cities&decarbonization.pdf",        "Decarbonization",     0.52, 0.93),
    "cryptocurrency_MIT":                ("cryptocurrency.pdf",                "Cryptocurrency",      0.21, 0.84),
    "solar_resource":                    ("solar_resource.pdf",                "Solar resource",      0.46, 0.53),
    "psychology":                        ("psychology.pdf",                    "Psychology",          0.58, 0.80),
    "creating_breakthrough_products_MIT": ("creating_breakthrough_products.pdf", "Productdesign",     0.72, 0.71),
    "image_processing":                  ("image_processing.pdf",              "Image processing",    0.53, 0.97),
    "sensory_systems":                   ("sensory_systems.pdf",               "Sensory systems",     0.46, 0.53),
    "ML_for_health_MIT":                 ("ML_for_health_care_MIT.pdf",        "ML for health",       0.57, 0.95),
    "climate_science_policy_MIT2":       ("climate_science_policy_MIT.pdf",    "Climate policies",    0.79, 0.93),
    "theory_of_computation":             ("theory_of_computation.pdf",         "Computation theory",  0.56, 0.84),
    "physics":                           ("physics_intro_02.pdf",              "Physics",             0.60, 0.93),
    "phonetics":                         ("phonetics.pdf",                     "Phonetics",           0.05, 0.65),
    "team_dynamics_game_design_MIT":     ("team_dynamics_game_design_mit.pdf", "Team dynamics",       0.89, 0.96),
    "cognitive_robotics_MIT":            ("cognitive_robotics_MIT.pdf",        "Cognitive robotics",  0.42, 0.66),
}
THEIR_AUDIO_AVG, THEIR_COMBINED_AVG, THEIR_SIFT_AVG = 0.53, 0.82, 0.56


def micro_prf(gt: np.ndarray, pred: np.ndarray) -> tuple[float, float, float]:
    """sklearn's micro P/R/F1 with labels=set(gt), reimplemented so the adapter has
    no sklearn dependency. A prediction outside the label set is a miss (FN) but not
    a hit for any label -- which is exactly what `labels=` does."""
    labels = set(int(x) for x in gt)
    tp = int(np.sum(gt == pred))
    pred_in = int(np.sum(np.isin(pred, list(labels))))
    p = tp / pred_in if pred_in else 0.0
    r = tp / len(gt) if len(gt) else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def ocr_missing_pages(pdf_path: Path, slides: list[EvidenceUnit], stem: str) -> tuple[list[EvidenceUnit], int]:
    """Image-only decks (scanned slides: image_processing.pdf has 21 pages and no text
    layer) get one OCR'd text unit per page, so every page is a reachable label.
    MaViLS itself OCRs video frames; this is the deck-side equivalent."""
    import pymupdf
    from linkrag.ingest.image import ocr
    have = {u.location.page for u in slides}
    added = 0
    with pymupdf.open(pdf_path) as doc:
        for i, page in enumerate(doc, start=1):
            if i in have:
                continue
            tmp = Path("data/processed/mavils_figures") / f"{stem}_page{i:03d}.png"
            tmp.parent.mkdir(parents=True, exist_ok=True)
            if not tmp.exists():
                page.get_pixmap(dpi=150).save(tmp)
            text = " ".join(ocr(tmp).split())
            slides.append(EvidenceUnit(id=f"{stem}:p{i}:ocr", modality="text", content=text or f"(page {i})",
                                       source_file=str(pdf_path), location=Location(page=i),
                                       metadata={"slide_deck": True, "content_source": "page_ocr"}))
            added += 1
    return sorted(slides, key=lambda u: u.location.page), added


def load_ground_truth(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    tcol = "Time" if "Time" in df.columns else "Key"
    df = df.rename(columns={tcol: "time", "Slidenumber": "slide", "Value": "text"})
    df["text"] = df["text"].fillna("").astype(str)
    df = df[df["text"].str.strip() != ""].reset_index(drop=True)
    return df[["time", "slide", "text"]]


def sentence_units(df: pd.DataFrame, stem: str) -> list[EvidenceUnit]:
    times = df["time"].astype(float).to_numpy()
    units = []
    for i, row in df.iterrows():
        start = times[i - 1] if i > 0 else max(0.0, times[i] - 3.0)
        units.append(EvidenceUnit(id=f"{stem}:s{i}", modality="audio", content=row["text"],
                                  source_file=f"{stem}.srt",
                                  location=Location(start_s=float(start), end_s=float(times[i]))))
    return units


def window_units(df: pd.DataFrame, stem: str, seconds: float) -> tuple[list[EvidenceUnit], list[int]]:
    """Pack their sentences into ~`seconds` windows (never splitting a sentence) --
    the segment granularity LinkRAG's alignment was designed for and runs on
    (`ingest.audio_segment_seconds`). Returns the units and, per sentence, the index
    of the window it belongs to, so predictions are still scored per sentence."""
    times = df["time"].astype(float).to_numpy()
    texts = df["text"].tolist()
    units, owner = [], []
    buf, start = [], None
    for i, (t, txt) in enumerate(zip(times, texts)):
        if start is None:
            start = times[i - 1] if i > 0 else max(0.0, t - 3.0)
        buf.append(txt)
        owner.append(len(units))
        if t - start >= seconds or i == len(times) - 1:
            units.append(EvidenceUnit(id=f"{stem}:w{len(units)}", modality="audio", content=" ".join(buf),
                                      source_file=f"{stem}.srt", location=Location(start_s=float(start), end_s=float(t))))
            buf, start = [], None
    return units, owner


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="data/raw/mavils", help="clone of github.com/andererka/MaViLS")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--only", default=None, help="comma-separated ground-truth stems")
    ap.add_argument("--out", default="reports/mavils_alignment.md")
    ap.add_argument("--figures-dir", default="data/processed/mavils_figures")
    ap.add_argument("--recompute", action="store_true", help="ignore per-lecture result cache")
    ap.add_argument("--window", type=float, default=None,
                    help="pack sentences into windows of this many seconds before aligning "
                         "(default: ingest.audio_segment_seconds; 0 = their sentence granularity)")
    args = ap.parse_args(argv)

    setup_logging()
    cfg = load_config(args.config)
    acfg = cfg["link"]["align"]
    repo = Path(args.repo)
    gt_dir, pdf_dir = repo / "data" / "ground_truth_files", repo / "data" / "lectures"
    if not gt_dir.exists():
        raise SystemExit(f"{gt_dir} not found -- clone the MaViLS repo to {repo}")

    encoder = default_encoder(cfg["models"]["embedding"], cfg["device"], cfg["index"]["normalize_embeddings"])
    encoder([""])
    stems = [s for s in LECTURES if not args.only or s in set(args.only.split(","))]
    rows = []
    t_total = time.perf_counter()
    window = cfg["ingest"]["audio_segment_seconds"] if args.window is None else args.window
    cache_dir = Path("data/processed/mavils") / (f"w{int(window)}" if window else "sentence")
    cache_dir.mkdir(parents=True, exist_ok=True)
    for stem in stems:
        pdf_name, their_name, their_audio, their_all = LECTURES[stem]
        cached = cache_dir / f"{stem}.json"
        if cached.exists() and not args.recompute:
            rows.append(json.loads(cached.read_text()))
            print(f"{their_name:<20} (cached) ours {rows[-1]['ours_f1']:.2f}")
            continue
        gt_path, pdf_path = gt_dir / f"ground_truth_{stem}.xlsx", pdf_dir / pdf_name
        if not gt_path.exists() or not pdf_path.exists():
            print(f"skip {stem}: missing {gt_path.name if not gt_path.exists() else pdf_name}")
            continue
        df = load_ground_truth(gt_path)
        pages = ingest_pdf(pdf_path, figures_dir=args.figures_dir, slide_deck=True,
                           ocr_figures=False, cluster_deck_figures=False)
        slides = sorted([u for u in pages if u.modality == "text"], key=lambda u: u.location.page)
        # one text unit per page is what the deck ingester produces; keep it that way
        by_page: dict[int, EvidenceUnit] = {}
        for u in slides:
            by_page.setdefault(u.location.page, u)
        slides = [by_page[p] for p in sorted(by_page)]
        slides, ocr_pages = ocr_missing_pages(pdf_path, slides, stem)
        page_numbers = np.array([u.location.page for u in slides])
        if window:
            audio, owner = window_units(df, stem, window)
        else:
            audio, owner = sentence_units(df, stem), list(range(len(df)))
        owner = np.array(owner)

        t0 = time.perf_counter()
        S = similarity_matrix(audio, slides, encoder=encoder, w_dense=acfg["weights"]["dense"],
                              w_bm25=acfg["weights"]["bm25"], w_keyword=acfg["weights"]["keyword"])
        ours = align_monotonic(S, jump_penalty=acfg["jump_penalty"], skip_penalty=acfg["skip_penalty"],
                               back_penalty=acfg["back_penalty"], max_back=acfg["max_back"],
                               start_prior_mu=acfg.get("start_prior_mu", 0.0))
        naive = align_naive(S)
        secs = time.perf_counter() - t0

        gt = df["slide"].astype(int).to_numpy()
        mask = gt != -1
        pred_ours = page_numbers[np.array(ours)][owner]      # window's slide -> each sentence
        pred_naive = page_numbers[np.array(naive)][owner]
        p1, r1, f1 = micro_prf(gt[mask], pred_ours[mask])
        p2, r2, f2 = micro_prf(gt[mask], pred_naive[mask])
        back = sum(1 for a, b in zip(ours, ours[1:]) if b < a)
        rows.append({"stem": stem, "name": their_name, "sentences": int(mask.sum()), "no_slide": int((~mask).sum()),
                     "slides": len(slides), "ocr_pages": ocr_pages, "gt_max_slide": int(gt.max()),
                     "ours_f1": f1, "ours_p": p1, "ours_r": r1, "naive_f1": f2,
                     "their_audio": their_audio, "their_all": their_all, "back_jumps": back, "secs": secs})
        cached.write_text(json.dumps(rows[-1]))
        print(f"{their_name:<20} n={int(mask.sum()):4d} slides={len(slides):3d}  ours {f1:.2f}  naive {f2:.2f}  "
              f"their audio {their_audio:.2f}  their all {their_all:.2f}  ({secs:.1f}s)")

    if not rows:
        return 1
    mean = lambda k: float(np.mean([r[k] for r in rows]))
    gran = f"{int(window)}-second windows of their sentences (our segment granularity), scored per sentence" if window else "their sentence granularity"
    L = ["# MaViLS alignment — target T2" + (f" — {int(window)}s windows" if window else " — sentence level"), "",
         f"{len(rows)} of 20 lectures · transcript + slide PDF only (no video frames) · segments: {gran} · "
         f"embedder `{cfg['models']['embedding']}` · align weights {acfg['weights']} · "
         f"λ={acfg['jump_penalty']} σ={acfg['skip_penalty']} β={acfg['back_penalty']} B={acfg['max_back']} "
         f"μ={acfg.get('start_prior_mu', 0.0)} · metric: their micro-F1 over labelled sentences "
         f"(−1 excluded) · sentences and transcript are theirs verbatim · "
         f"wall time {time.perf_counter() - t_total:.0f}s (embedding included, not a latency claim)", "",
         "Their numbers are copied from Anderer et al. 2024, Table 1 (*Audio* column: transcript only, "
         "λ_jump = 0.1) and Table 2 (all three features, λ_jump = 0.1). **Audio is the like-for-like "
         "column**; *All* uses frame OCR and image features this adapter does not consume.", "",
         "| lecture | sentences | slides | **ours (monotonic)** | naive argmax | their audio | their all | Δ vs their audio |",
         "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        L.append(f"| {r['name']} | {r['sentences']} | {r['slides']} | **{r['ours_f1']:.2f}** | {r['naive_f1']:.2f} | "
                 f"{r['their_audio']:.2f} | {r['their_all']:.2f} | {r['ours_f1'] - r['their_audio']:+.2f} |")
    L.append(f"| **average** | {sum(r['sentences'] for r in rows)} | | **{mean('ours_f1'):.2f}** | {mean('naive_f1'):.2f} | "
             f"{mean('their_audio'):.2f} | {mean('their_all'):.2f} | {mean('ours_f1') - mean('their_audio'):+.2f} |")
    wins = sum(r["ours_f1"] > r["their_audio"] for r in rows)
    L += ["", f"Ours beats their audio-only F1 on **{wins}/{len(rows)}** lectures; beats their all-features "
          f"F1 on {sum(r['ours_f1'] > r['their_all'] for r in rows)}/{len(rows)}. Back-jumps in our paths: "
          f"{sum(r['back_jumps'] or 0 for r in rows)} total (B={acfg['max_back']}).", "",
          "Caveats. Their audio column was produced with distiluse-base-multilingual-cased and their "
          "own DP; ours uses bge-m3 + BM25 + IDF overlap and our DP, so the delta mixes embedder and "
          "algorithm — the naive column isolates the DP's share on *our* similarity. Slide numbering: "
          "we assume their `Slidenumber` is the 1-based PDF page; a lecture where `gt_max_slide` "
          "exceeds the page count would break that assumption and is flagged below.", ""]
    bad = [r for r in rows if r["gt_max_slide"] > r["slides"]]
    if bad:
        L += ["Flagged: " + ", ".join(f"{r['name']} (gt max {r['gt_max_slide']} > {r['slides']} pages)" for r in bad), ""]
    ocrd = [r for r in rows if r.get("ocr_pages")]
    if ocrd:
        L += ["Image-only pages OCR'd (no text layer): " + ", ".join(f"{r['name']} ({r['ocr_pages']})" for r in ocrd), ""]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n")
    out.with_suffix(".json").write_text(json.dumps(rows, indent=1))
    print(f"\naverage ours {mean('ours_f1'):.3f} · naive {mean('naive_f1'):.3f} · their audio {mean('their_audio'):.2f} "
          f"· their all {mean('their_all'):.2f}\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
