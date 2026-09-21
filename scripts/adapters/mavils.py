#!/usr/bin/env python3
"""MaViLS adapter -- priority (iv). Target T2 (Anderer, Reich, Wölfel, Interspeech 2024).

    git clone --depth 1 https://github.com/andererka/MaViLS data/raw/mavils
    python scripts/adapters/mavils.py run     --split all             # like-for-like table
    python scripts/adapters/mavils.py study                           # held-out abstention/flatness study
    python scripts/adapters/mavils.py inspect --lecture cities_and_decarbonization

Their protocol, read from `evaluation/evaluate_recall_precision.py` and
`mavils/matching_algorithm.py` (see `protocol_note()` for the exact text):

* granularity: one prediction per ground-truth row = one transcript *sentence*
  (their `generate_output_dict_by_sentence`; a video frame is sampled at the
  sentence midpoint). Labels are 1-based slide numbers, -1 = no slide on screen.
* score: per lecture, sklearn `f1_score(gt[mask], pred[mask], labels=unique(gt),
  average="micro")` with `mask = gt != -1`; `labels` is taken from the *unfiltered*
  column, so -1 is in the label set. Consequence: a prediction of -1 for a labelled
  sentence is a false positive for label -1 and a miss for the true slide -- under
  their F1, abstaining scores exactly like a wrong slide. The paper's averages are
  the unweighted mean of the 20 per-lecture F1s.
* their "audio-only" column: distiluse-base-multilingual-cased cosine between each
  sentence and the **tesseract OCR of the rendered slide image** (not the PDF text
  layer), decoded with their DP (jump penalty 0.1·|Δslide|, doubled backwards,
  unrestricted back-jumps, no skip penalty).

Our like-for-like column is therefore **sentence granularity + OCR'd slide text +
our DP**, reported as `dp` with `--slide-text ocr`. The 30-second-window variant
and the PDF-text-layer variant are reported separately as protocol deviations.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

from linkrag.core import EvidenceUnit, Location, load_config, setup_logging
from linkrag.index import default_encoder
from linkrag.ingest.pdf import ingest_pdf
from linkrag.link.align import abstain, align_monotonic, align_naive, similarity_matrix

# ground-truth stem -> (slides PDF, their Table 1 name, Table 1 audio F1, Table 2 combined F1 @ lambda 0.1)
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
REPO = Path("data/raw/mavils")
CACHE = Path("data/processed/mavils")
EXTERNAL = Path("results/external")
SPLIT_SEED = 20260922
VARIANTS = ("naive", "dp", "dp+abstain", "dp+abstain+flat")


def protocol_note() -> str:
    return (
        "**Their scoring, verbatim from `evaluation/evaluate_recall_precision.py`:** "
        "`ground_truth_labels = ground_truth_column.unique()`; `mask = (ground_truth_column != -1)`; "
        "`f1_score(filtered_ground_truth, filtered_result, labels=ground_truth_labels, average='micro')`. "
        "One row per transcript sentence (`generate_output_dict_by_sentence`), per lecture, then the "
        "unweighted mean over lectures. `labels` comes from the unfiltered column, so -1 is a label: "
        "predicting -1 on a labelled sentence is a false positive for -1 and a miss for the true slide, "
        "i.e. **abstention cannot raise their F1**; it can only be seen in precision-on-answered and "
        "coverage, which we report alongside. Their audio-only similarity is distiluse cosine between "
        "the sentence and **tesseract OCR of the rendered slide image** (`matching_algorithm.py`), "
        "decoded with their DP (penalty 0.1·|Δslide|, ×2 backwards, no skip penalty)."
    )


# ----------------------------------------------------------------- their metric

def their_prf(gt: np.ndarray, pred: np.ndarray) -> tuple[float, float, float]:
    labels = np.unique(gt)                       # unfiltered, -1 included -- theirs
    mask = gt != -1
    kw = dict(labels=labels, average="micro", zero_division=0)
    return (float(precision_score(gt[mask], pred[mask], **kw)),
            float(recall_score(gt[mask], pred[mask], **kw)),
            float(f1_score(gt[mask], pred[mask], **kw)))


def answered_metrics(gt: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    """(precision on answered sentences, coverage) -- what abstention trades."""
    mask = gt != -1
    answered = mask & (pred != -1)
    cov = float(answered.sum() / max(mask.sum(), 1))
    prec = float((gt[answered] == pred[answered]).mean()) if answered.any() else 0.0
    return prec, cov


# ----------------------------------------------------------------- data

def load_ground_truth(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    tcol = "Time" if "Time" in df.columns else "Key"
    df = df.rename(columns={tcol: "time", "Slidenumber": "slide", "Value": "text"})
    df["text"] = df["text"].fillna("").astype(str)
    df = df[df["text"].str.strip() != ""].reset_index(drop=True)
    return df[["time", "slide", "text"]]


def sentence_units(df: pd.DataFrame, stem: str) -> tuple[list[EvidenceUnit], list[int]]:
    times = df["time"].astype(float).to_numpy()
    units = []
    for i, row in df.iterrows():
        start = times[i - 1] if i > 0 else max(0.0, times[i] - 3.0)
        units.append(EvidenceUnit(id=f"{stem}:s{i}", modality="audio", content=row["text"],
                                  source_file=f"{stem}.srt",
                                  location=Location(start_s=float(start), end_s=float(times[i]))))
    return units, list(range(len(df)))


def window_units(df: pd.DataFrame, stem: str, seconds: float) -> tuple[list[EvidenceUnit], list[int]]:
    """Pack their sentences into ~`seconds` windows (never splitting a sentence): the
    segment granularity LinkRAG runs on. Predictions are broadcast back to sentences."""
    times = df["time"].astype(float).to_numpy()
    texts = df["text"].tolist()
    units, owner, buf, start = [], [], [], None
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


def slide_units(stem: str, mode: str, figures_dir: Path) -> tuple[list[EvidenceUnit], dict]:
    """mode 'layer': PDF text layer, page-OCR only where a page has none.
    mode 'ocr'  : tesseract on every rendered page -- their slide-side input."""
    import pymupdf
    from linkrag.ingest.image import ocr
    pdf_path = REPO / "data" / "lectures" / LECTURES[stem][0]
    by_page: dict[int, EvidenceUnit] = {}
    if mode == "layer":
        pages = ingest_pdf(pdf_path, figures_dir=figures_dir, slide_deck=True, ocr_figures=False,
                           cluster_deck_figures=False)
        for u in sorted([u for u in pages if u.modality == "text"], key=lambda u: u.location.page):
            by_page.setdefault(u.location.page, u)
    ocr_pages = 0
    with pymupdf.open(pdf_path) as doc:
        n_pages = len(doc)
        text_layer_pages = sum(1 for p in doc if p.get_text().strip())
        for i, page in enumerate(doc, start=1):
            if i in by_page:
                continue
            png = figures_dir / f"{stem}_page{i:03d}.png"
            txt = png.with_suffix(".ocr.txt")
            figures_dir.mkdir(parents=True, exist_ok=True)
            if not txt.exists():
                if not png.exists():
                    page.get_pixmap(dpi=150).save(png)
                txt.write_text(" ".join(ocr(png).split()))
            text = txt.read_text()
            by_page[i] = EvidenceUnit(id=f"{stem}:p{i}:ocr", modality="text", content=text or f"(page {i})",
                                      source_file=str(pdf_path), location=Location(page=i),
                                      metadata={"slide_deck": True, "content_source": "page_ocr"})
            ocr_pages += 1
    units = [by_page[p] for p in sorted(by_page)]
    status = {"pages": n_pages, "text_layer_pages": text_layer_pages, "ocr_pages": ocr_pages,
              "text_layer": "full" if text_layer_pages == n_pages else ("none" if text_layer_pages == 0 else "partial")}
    return units, status


def similarity_for(stem: str, *, window: float, slide_text: str, cfg: dict, encoder, figures_dir: Path):
    """S (cached), sentence ownership, gt labels, slide page numbers, text-layer status."""
    tag = f"{stem}.{'w' + str(int(window)) if window else 'sentence'}.{slide_text}"
    path = CACHE / "S" / f"{tag}.npz"
    df = load_ground_truth(REPO / "data" / "ground_truth_files" / f"ground_truth_{stem}.xlsx")
    gt = df["slide"].astype(int).to_numpy()
    if path.exists():
        z = np.load(path)          # our own cache, no object arrays -> no pickle needed
        return z["S"], z["owner"], gt, z["pages"], json.loads(str(z["status"]))
    a = cfg["link"]["align"]
    audio, owner = (window_units(df, stem, window) if window else sentence_units(df, stem))
    slides, status = slide_units(stem, slide_text, figures_dir)
    S = similarity_matrix(audio, slides, encoder=encoder, w_dense=a["weights"]["dense"],
                          w_bm25=a["weights"]["bm25"], w_keyword=a["weights"]["keyword"])
    pages = np.array([u.location.page for u in slides])
    path.parent.mkdir(parents=True, exist_ok=True)
    # status is a JSON string stored as a 0-d array; saved without pickling
    np.savez(path, S=S, owner=np.array(owner), pages=pages, status=np.array(json.dumps(status)))
    return S, np.array(owner), gt, pages, status


# ----------------------------------------------------------------- variants

def decode(S, pages, owner, a: dict, *, variant: str, min_sim: float | None, flat: float) -> np.ndarray:
    if variant == "naive":
        path = align_naive(S)
    else:
        path = align_monotonic(S, jump_penalty=a["jump_penalty"], skip_penalty=a["skip_penalty"],
                               back_penalty=a["back_penalty"], max_back=a["max_back"],
                               start_prior_mu=a.get("start_prior_mu", 0.0),
                               flatness_scaling=flat if variant.endswith("flat") else 0.0)
    if "abstain" in variant:
        path = abstain(S, path, min_sim)
    pred = np.array([pages[j] if j >= 0 else -1 for j in path])
    return pred[owner]


def split_lectures() -> dict:
    path = EXTERNAL / "mavils_split.json"
    if path.exists():
        return json.loads(path.read_text())
    stems = sorted(LECTURES)
    random.Random(SPLIT_SEED).shuffle(stems)
    split = {"seed": SPLIT_SEED, "tune": sorted(stems[:10]), "test": sorted(stems[10:]),
             "note": "fixed once on 2026-09-22; the tune half is the only half any parameter is set on"}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(split, indent=1) + "\n")
    return split


# ----------------------------------------------------------------- commands

def cmd_run(args, cfg, encoder) -> int:
    a = cfg["link"]["align"]
    split = split_lectures()
    stems = sorted(LECTURES) if args.split == "all" else split[args.split]
    rows = []
    for stem in stems:
        S, owner, gt, pages, status = similarity_for(stem, window=args.window, slide_text=args.slide_text,
                                                     cfg=cfg, encoder=encoder, figures_dir=Path(args.figures_dir))
        r = {"stem": stem, "name": LECTURES[stem][1], "their_audio": LECTURES[stem][2], "their_all": LECTURES[stem][3],
             "n": int((gt != -1).sum()), "slides": len(pages), "text_layer": status["text_layer"], "ocr_pages": status["ocr_pages"]}
        for v in ("naive", "dp"):
            pred = decode(S, pages, owner, a, variant=v, min_sim=None, flat=0.0)
            r[f"{v}_f1"] = their_prf(gt, pred)[2]
        rows.append(r)
        print(f"{r['name']:<20} n={r['n']:4d} slides={r['slides']:3d} layer={r['text_layer']:<7} "
              f"dp {r['dp_f1']:.2f} naive {r['naive_f1']:.2f} their audio {r['their_audio']:.2f}")
    mean = lambda k: float(np.mean([r[k] for r in rows]))
    gran = ("their sentence granularity" if not args.window else
            f"{int(args.window)}-second windows of their sentences, broadcast back and scored per sentence")
    side = ("tesseract OCR of every rendered page (their slide-side input)" if args.slide_text == "ocr"
            else "PDF text layer (page-OCR only where a page has none)")
    L = [f"# MaViLS — like-for-like protocol ({args.split}: {len(rows)} lectures)", "",
         f"Segments: {gran} · slide text: {side} · embedder `{cfg['models']['embedding']}` + BM25 + IDF overlap · "
         f"DP λ={a['jump_penalty']} σ={a['skip_penalty']} β={a['back_penalty']} B={a['max_back']} "
         f"μ={a.get('start_prior_mu', 0.0)} · scored with sklearn exactly as their evaluation script.", "",
         protocol_note(), "",
         "**Column correspondence.** Their *audio-only* ⇔ our `dp` column **when run with `--slide-text ocr` at "
         "sentence granularity** (same input protocol; our similarity and DP). `naive` is the same matrix without "
         "sequence structure. Their *all-features* uses video frames we do not consume.", "",
         "| lecture | n | slides | text layer | **dp** | naive | their audio | their all | Δ dp − their audio |",
         "|---|---:|---:|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        layer = r["text_layer"] + (f" ({r['ocr_pages']} OCR)" if r["ocr_pages"] and args.slide_text == "layer" else "")
        L.append(f"| {r['name']} | {r['n']} | {r['slides']} | {layer} | **{r['dp_f1']:.2f}** | {r['naive_f1']:.2f} | "
                 f"{r['their_audio']:.2f} | {r['their_all']:.2f} | {r['dp_f1'] - r['their_audio']:+.2f} |")
    L.append(f"| **mean** | {sum(r['n'] for r in rows)} | | | **{mean('dp_f1'):.2f}** | {mean('naive_f1'):.2f} | "
             f"{mean('their_audio'):.2f} | {mean('their_all'):.2f} | {mean('dp_f1') - mean('their_audio'):+.2f} |")
    wins = sum(r["dp_f1"] > r["their_audio"] for r in rows)
    L += ["", f"`dp` above their audio-only on {wins}/{len(rows)} lectures.", ""]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n")
    out.with_suffix(".json").write_text(json.dumps(rows, indent=1))
    print(f"\nmean dp {mean('dp_f1'):.3f} · naive {mean('naive_f1'):.3f} · their audio {mean('their_audio'):.2f}\nwrote {out}")
    return 0


def cmd_study(args, cfg, encoder) -> int:
    """Held-out study. Tune half only: choose flatness_scaling and min_segment_sim.
    Criteria, stated in advance:
      flatness_scaling -> maximise mean their-F1 of the DP on the tune half
      min_segment_sim  -> maximise precision-on-answered on the tune half subject to
                          coverage >= 0.8 (their F1 cannot reward abstention -- see protocol)
    Then report tune and test halves separately for all four variants."""
    a = cfg["link"]["align"]
    split = split_lectures()
    data = {stem: similarity_for(stem, window=args.window, slide_text=args.slide_text, cfg=cfg,
                                 encoder=encoder, figures_dir=Path(args.figures_dir)) for stem in sorted(LECTURES)}
    sims = [float(v) for stem in split["tune"] for v in data[stem][0].max(axis=1)]
    grid_sim = [None] + [round(float(np.quantile(sims, q)), 4) for q in (0.05, 0.10, 0.20, 0.30)]
    grid_flat = [0.0, 0.5, 1.0]

    def evaluate(stems, variant, min_sim, flat):
        f1s, precs, covs = [], [], []
        for stem in stems:
            S, owner, gt, pages, _ = data[stem]
            pred = decode(S, pages, owner, a, variant=variant, min_sim=min_sim, flat=flat)
            f1s.append(their_prf(gt, pred)[2])
            p, c = answered_metrics(gt, pred)
            precs.append(p)
            covs.append(c)
        return float(np.mean(f1s)), float(np.mean(precs)), float(np.mean(covs))

    flat_rows = [(f, *evaluate(split["tune"], "dp+abstain+flat", None, f)) for f in grid_flat]
    best_flat = max(flat_rows, key=lambda r: r[1])[0]
    sim_rows = [(ms, *evaluate(split["tune"], "dp+abstain+flat", ms, best_flat)) for ms in grid_sim]
    feasible = [r for r in sim_rows if r[3] >= 0.8]
    best_sim = max(feasible, key=lambda r: r[2])[0] if feasible else None
    tuned = {"min_segment_sim": best_sim, "flatness_scaling": best_flat, "window": args.window,
             "slide_text": args.slide_text, "tune_grid_sim": grid_sim, "tune_grid_flat": grid_flat,
             "criteria": "flatness: max tune their-F1; min_sim: max tune answered-precision with coverage >= 0.8"}
    (EXTERNAL / "mavils_tuned.json").write_text(json.dumps(tuned, indent=1) + "\n")

    names = lambda half: ", ".join(LECTURES[s][1] for s in split[half])
    L = ["# MaViLS — held-out abstention / flatness study", "",
         f"Split `results/external/mavils_split.json` (seed {split['seed']}): tune = {names('tune')}; test = {names('test')}. "
         f"Segments: {'sentence' if not args.window else str(int(args.window)) + 's windows'} · slide text: {args.slide_text} · "
         "metric: their F1 (protocol note in `mavils_alignment.md`) plus precision-on-answered and coverage.", "",
         "**Parameters were set on the tune half only** and are recorded in `results/external/mavils_tuned.json`, "
         "not in `configs/default.yaml` (no parameter changes on the full set / pilot01).", "",
         "## Tune half — flatness grid (DP, no abstention)", "",
         "| flatness_scaling | their F1 | prec@answered | coverage |", "|---:|---:|---:|---:|"]
    for f, f1, p, c in flat_rows:
        L.append(f"| {f} | {f1:.3f}{' ←' if f == best_flat else ''} | {p:.3f} | {c:.2f} |")
    L += ["", f"## Tune half — abstention grid (flatness = {best_flat})", "",
          "| min_segment_sim | their F1 | prec@answered | coverage |", "|---:|---:|---:|---:|"]
    for ms, f1, p, c in sim_rows:
        L.append(f"| {ms if ms is not None else 'off'} | {f1:.3f} | {p:.3f}{' ←' if ms == best_sim else ''} | {c:.2f} |")
    L += ["", f"Chosen on the tune half: flatness_scaling = {best_flat}, min_segment_sim = {best_sim}.", ""]

    for half in ("tune", "test"):
        L += [f"## {half.capitalize()} half — their F1 per lecture (prec@answered / coverage for abstaining variants)", "",
              "| lecture | text layer | naive | dp | dp+abstain | dp+abstain+flat | their audio |",
              "|---|---|---:|---:|---:|---:|---:|"]
        agg = {v: [] for v in VARIANTS}
        for stem in split[half]:
            S, owner, gt, pages, status = data[stem]
            cells = []
            for v in VARIANTS:
                pred = decode(S, pages, owner, a, variant=v, min_sim=best_sim, flat=best_flat)
                f1 = their_prf(gt, pred)[2]
                agg[v].append(f1)
                if "abstain" in v:
                    p, c = answered_metrics(gt, pred)
                    cells.append(f"{f1:.2f} ({p:.2f} / {c:.2f})")
                else:
                    cells.append(f"{f1:.2f}")
            L.append(f"| {LECTURES[stem][1]} | {status['text_layer']} | " + " | ".join(cells) + f" | {LECTURES[stem][2]:.2f} |")
        L.append("| **mean** | | " + " | ".join(f"**{np.mean(agg[v]):.3f}**" for v in VARIANTS)
                 + f" | {np.mean([LECTURES[s][2] for s in split[half]]):.2f} |")
        L.append("")
    out = Path(args.out)
    out.write_text("\n".join(L) + "\n")
    print("\n".join(L[-14:]))
    print(f"wrote {out}")
    return 0


def cmd_inspect(args, cfg, encoder) -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    a = cfg["link"]["align"]
    stem = args.lecture
    S, owner, gt, pages, status = similarity_for(stem, window=args.window, slide_text=args.slide_text, cfg=cfg,
                                                 encoder=encoder, figures_dir=Path(args.figures_dir))
    slides, _ = slide_units(stem, args.slide_text, Path(args.figures_dir))
    df = load_ground_truth(REPO / "data" / "ground_truth_files" / f"ground_truth_{stem}.xlsx")
    pred = decode(S, pages, owner, a, variant="dp", min_sim=None, flat=0.0)
    naive = decode(S, pages, owner, a, variant="naive", min_sim=None, flat=0.0)
    name = LECTURES[stem][1]
    page_index = {int(p): j for j, p in enumerate(pages)}

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.imshow(S.T, aspect="auto", origin="lower", cmap="viridis")
    xs = np.arange(len(gt))
    ax.plot(xs[gt != -1], [page_index.get(int(g), -1) for g in gt[gt != -1]], ".", color="white", ms=3, label="their label")
    ax.plot(xs, [page_index.get(int(p), -1) for p in pred], "-", color="orange", lw=1.2, label="our DP path")
    ax.set_xlabel("sentence")
    ax.set_ylabel("slide index")
    ax.set_title(f"{name}: similarity matrix, DP path, labels")
    ax.legend(loc="upper left")
    png = Path("reports") / f"mavils_inspect_{stem}.png"
    fig.tight_layout()
    fig.savefig(png, dpi=110)
    plt.close(fig)

    text_of = {u.location.page: " ".join(u.content.split()) for u in slides}
    contrast = S.max(axis=1) - S.mean(axis=1)
    L = [f"## Inspection — {name} (`{stem}`)", "",
         f"their F1: dp {their_prf(gt, pred)[2]:.2f}, naive {their_prf(gt, naive)[2]:.2f}, their audio {LECTURES[stem][2]:.2f} · "
         f"{len(gt)} sentences ({int((gt == -1).sum())} unlabelled) · {len(pages)} pages, text layer {status['text_layer']} "
         f"({status['text_layer_pages']}/{status['pages']} pages with text) · GT slide range {gt[gt != -1].min()}–{gt.max()} · "
         f"mean row contrast (max−mean) {contrast.mean():.3f} · slide text: {args.slide_text}", "",
         f"![{name}]({png.name})", "",
         "Distinct labels used: " + ", ".join(str(x) for x in sorted(set(int(x) for x in gt[gt != -1]))), "",
         "First 10 mismatches (sentence → our slide vs their label; slide texts truncated):", ""]
    shown = 0
    for i in range(len(gt)):
        if gt[i] == -1 or pred[i] == gt[i]:
            continue
        s_ours = S[owner[i], page_index[int(pred[i])]]
        s_gold = S[owner[i], page_index[int(gt[i])]] if int(gt[i]) in page_index else float("nan")
        L.append(f"- s{i} (t={df['time'][i]:.0f}s) “{df['text'][i][:110]}”  \n"
                 f"  ours p{pred[i]} (S={s_ours:.2f}): {text_of.get(int(pred[i]), '')[:120]}  \n"
                 f"  label p{gt[i]} (S={s_gold:.2f}): {text_of.get(int(gt[i]), '(no such page)')[:120]}")
        shown += 1
        if shown == 10:
            break
    out = Path(args.out if args.out else f"reports/mavils_inspect_{stem}.md")
    out.write_text("\n".join(L) + "\n")
    print("\n".join(L[:4]))
    print(f"wrote {out} and {png}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "study", "inspect"])
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--split", choices=["all", "tune", "test"], default="all")
    ap.add_argument("--window", type=float, default=0.0, help="0 = their sentence granularity (default); e.g. 30")
    ap.add_argument("--slide-text", choices=["layer", "ocr"], default="ocr")
    ap.add_argument("--lecture", default="cities_and_decarbonization")
    ap.add_argument("--figures-dir", default="data/processed/mavils_figures")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    if args.out is None:
        args.out = {"run": "reports/mavils_alignment.md", "study": "reports/mavils_heldout_study.md",
                    "inspect": None}[args.cmd]
    setup_logging()
    cfg = load_config(args.config)
    if not (REPO / "data" / "ground_truth_files").exists():
        raise SystemExit(f"clone https://github.com/andererka/MaViLS to {REPO}")
    encoder = default_encoder(cfg["models"]["embedding"], cfg["device"], cfg["index"]["normalize_embeddings"])
    encoder([""])
    return {"run": cmd_run, "study": cmd_study, "inspect": cmd_inspect}[args.cmd](args, cfg, encoder)


if __name__ == "__main__":
    raise SystemExit(main())
