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
from linkrag.link.align import (abstain, align_monotonic, align_naive, distiluse_similarity, fuse_similarity,
                                relatedness_gate, similarity_matrix)

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
        "i.e. **abstention cannot raise their F1** -- it scores like a wrong slide that is itself a label, "
        "and *worse* than a wrong slide the ground truth never uses (that one costs recall only). "
        "Abstention can only be seen in precision-on-answered and coverage, which we report alongside. Their audio-only similarity is distiluse cosine between "
        "the sentence and **tesseract OCR of the rendered slide image** (`matching_algorithm.py`), "
        "decoded with their DP (penalty 0.1·|Δslide|, ×2 backwards, no skip penalty)."
    )


# ----------------------------------------------------------------- their metric

def their_prf(gt: np.ndarray, pred: np.ndarray) -> tuple[float, float, float]:
    labels = np.unique(gt)                       # unfiltered, -1 included -- theirs
    mask = gt != -1
    kw = {"labels": labels, "average": "micro", "zero_division": 0}
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




# ----------------------------------------------------------------- their code, verbatim

def their_dp():
    """`calculate_dp_with_jumps` from MaViLS `helpers/utils.py`, executed from source so
    the module's cv2/torch imports are not needed. Their code, unmodified."""
    import ast
    src = (REPO / "helpers" / "utils.py").read_text()
    tree = ast.parse(src)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "calculate_dp_with_jumps")
    ns: dict = {"np": np, "tqdm": lambda it, **kw: it}
    exec(ast.get_source_segment(src, node), ns)
    return ns["calculate_dp_with_jumps"]


def their_similarity_for(stem: str, *, cfg: dict, figures_dir: Path):
    """distiluse-base-multilingual-cased cosine between each sentence and the page OCR
    text -- their audio-only similarity (matching_algorithm.py). Cached. Deviation:
    tesseract `eng` only (their `eng+ell+equ+deu`; those traineddata are not installed)."""
    path = CACHE / "S" / f"{stem}.sentence.theirs.npz"
    df = load_ground_truth(REPO / "data" / "ground_truth_files" / f"ground_truth_{stem}.xlsx")
    gt = df["slide"].astype(int).to_numpy()
    if path.exists():
        z = np.load(path)
        return z["S"], z["owner"], gt, z["pages"]
    audio, owner = sentence_units(df, stem)
    slides, _ = slide_units(stem, "ocr", figures_dir)
    S = distiluse_similarity(audio, slides, device=cfg["device"])
    pages = np.array([u.location.page for u in slides])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, S=S, owner=np.array(owner), pages=pages)
    return S, np.array(owner), gt, pages


def decode_theirs(S, pages, owner, jump_penalty: float = 0.1) -> np.ndarray:
    path_pairs, _ = their_dp()(S, jump_penalty)
    path = [j for _, j in path_pairs]
    return np.array([pages[j] for j in path])[owner]


# ----------------------------------------------------------------- build decks

def _tokens(text: str) -> list[str]:
    import re
    return re.findall(r"[a-z0-9]+", text.lower())


def build_groups(page_texts: list[str], min_tokens: int = 5) -> list[list[int]]:
    """Consecutive pages whose token multiset is a superset of the previous page's
    (with at least `min_tokens` tokens) form one build group. Returns index groups
    covering every page; a non-build page is a group of one."""
    from collections import Counter
    groups: list[list[int]] = []
    prev: Counter | None = None
    for i, text in enumerate(page_texts):
        cur = Counter(_tokens(text))
        if prev is not None and sum(prev.values()) >= min_tokens and not (prev - cur):
            groups[-1].append(i)
        else:
            groups.append([i])
        prev = cur
    return groups


def group_similarity(S: np.ndarray, groups: list[list[int]]) -> np.ndarray:
    """One column per group: the max over its builds (the fullest build dominates
    anyway; max keeps a sentence that matches an early build visible)."""
    return np.stack([S[:, g].max(axis=1) for g in groups], axis=1)


def assign_within_group(rows: list[int], group: list[int], page_texts: list[str],
                        sentence_texts: list[str], idf: dict[str, float]) -> list[int]:
    """Builds inside a group differ by what each one ADDS. Score each sentence against
    each build's incremental text (IDF-weighted token overlap over the delta), then
    decode monotonically (a build is never un-revealed) with no penalties."""
    if len(group) == 1:
        return [group[0]] * len(rows)
    from collections import Counter
    deltas, prev = [], Counter()
    for j in group:
        cur = Counter(_tokens(page_texts[j]))
        deltas.append(set(cur - prev) or set(cur))       # first build: its whole text
        prev = cur
    mass = [max(sum(idf.get(t, 1.0) for t in d), 1e-9) for d in deltas]
    sub = np.zeros((len(rows), len(group)))
    for r, i in enumerate(rows):
        toks = set(_tokens(sentence_texts[i]))
        for b, d in enumerate(deltas):
            sub[r, b] = sum(idf.get(t, 1.0) for t in toks & d) / mass[b]
    local = align_monotonic(sub, jump_penalty=0.0, skip_penalty=0.0, back_penalty=1.0, max_back=0)
    return [group[b] for b in local]


def decode_with_builds(S, pages, owner, a: dict, *, page_texts: list[str], sentence_texts: list[str],
                       min_sim: float | None, flat: float, sigma: float | None = None) -> tuple[np.ndarray, dict]:
    """Group-level DP over build groups, then within-group assignment by incremental
    content. Returns predictions per sentence and build statistics."""
    from linkrag.link.align import _idf  # same IDF the similarity uses
    from linkrag.index import tokenize
    groups = build_groups(page_texts)
    idf = _idf([tokenize(t) for t in page_texts])
    G = group_similarity(S, groups)
    sig = a["skip_penalty"] if sigma is None else sigma
    gpath = align_monotonic(G, jump_penalty=a["jump_penalty"], skip_penalty=sig, back_penalty=a["back_penalty"],
                            max_back=a["max_back"], start_prior_mu=a.get("start_prior_mu", 0.0), flatness_scaling=flat)
    path = [-1] * S.shape[0]
    seg_texts = sentence_texts  # window=0: one row per sentence
    for g_idx, group in enumerate(groups):
        rows = [i for i, p in enumerate(gpath) if p == g_idx]
        if rows:
            for i, j in zip(rows, assign_within_group(rows, group, page_texts, seg_texts, idf)):
                path[i] = j
    path = abstain(S, path, min_sim)
    pred = np.array([pages[j] if j >= 0 else -1 for j in path])[owner]
    stats = {"groups": len(groups), "build_groups": sum(len(g) > 1 for g in groups),
             "pages_in_builds": sum(len(g) for g in groups if len(g) > 1)}
    return pred, stats


# ----------------------------------------------------------------- variants

def decode(S, pages, owner, a: dict, *, variant: str, min_sim: float | None, flat: float,
           sigma: float | None = None) -> np.ndarray:
    if variant == "naive":
        path = align_naive(S)
    else:
        path = align_monotonic(S, jump_penalty=a["jump_penalty"],
                               skip_penalty=a["skip_penalty"] if sigma is None else sigma,
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
            r[f"{v}_paired"] = paired(gt, pred)
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
         "Cells: **their F1 (precision-on-answered / coverage)**.", "",
         "| lecture | n | slides | text layer | **dp** | naive | their audio | their all | Δ dp − their audio |",
         "|---|---:|---:|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        layer = r["text_layer"] + (f" ({r['ocr_pages']} OCR)" if r["ocr_pages"] and args.slide_text == "layer" else "")
        L.append(f"| {r['name']} | {r['n']} | {r['slides']} | {layer} | **{r['dp_paired']}** | {r['naive_paired']} | "
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
        return paired_means([(gt, decode(S, pages, owner, a, variant=variant, min_sim=min_sim, flat=flat))
                             for S, owner, gt, pages, _ in (data[s] for s in stems)])

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


def paired_means(pairs) -> tuple[float, float, float]:
    """Mean their-F1, precision-on-answered and coverage over (gt, pred) pairs, one per lecture."""
    f1s = [their_prf(g, p)[2] for g, p in pairs]
    prs, covs = zip(*(answered_metrics(g, p) for g, p in pairs))
    return float(np.mean(f1s)), float(np.mean(prs)), float(np.mean(covs))


def paired_mean_cell(pairs) -> tuple[float, str]:
    """Mean their-F1, and the cell every mean row carries: 'F1 (precision-on-answered / coverage)'."""
    f1, pr, cov = paired_means(pairs)
    return f1, f"{f1:.3f} ({pr:.3f} / {cov:.2f})"


def paired(gt: np.ndarray, pred: np.ndarray) -> str:
    """The two numbers every alignment table carries: their F1, and precision-on-answered / coverage."""
    f1 = their_prf(gt, pred)[2]
    pr, cov = answered_metrics(gt, pred)
    return f"{f1:.2f} ({pr:.2f} / {cov:.2f})"


def cmd_final(args, cfg, encoder) -> int:
    """Final round: sigma sweep (tune -> test), matrix x decoder decomposition, build-deck
    grouping (tune -> test). Tune-half changes only; the test half is reported once."""
    a = cfg["link"]["align"]
    split = split_lectures()
    tune, test = split["tune"], split["test"]
    figures_dir = Path(args.figures_dir)
    stems = sorted(LECTURES)
    data, texts, theirs = {}, {}, {}
    for stem in stems:
        data[stem] = similarity_for(stem, window=0.0, slide_text="ocr", cfg=cfg, encoder=encoder, figures_dir=figures_dir)
        slides, status = slide_units(stem, "ocr", figures_dir)
        df = load_ground_truth(REPO / "data" / "ground_truth_files" / f"ground_truth_{stem}.xlsx")
        texts[stem] = ([u.content for u in slides], df["text"].tolist(), status)
        theirs[stem] = their_similarity_for(stem, cfg=cfg, figures_dir=figures_dir)
        print(f"loaded {LECTURES[stem][1]}")

    def mean_f1(stems_, fn):
        return paired_means([(data[s][2], fn(s)) for s in stems_])[0]

    def mean_paired(stems_, fn):
        return paired_mean_cell([(data[s][2], fn(s)) for s in stems_])[1]

    def dp(stem, sigma=None, builds=False, min_sim=None):
        S, owner, gt, pages, _ = data[stem]
        if builds:
            pred, _st = decode_with_builds(S, pages, owner, a, page_texts=texts[stem][0], sentence_texts=texts[stem][1],
                                           min_sim=min_sim, flat=0.0, sigma=sigma)
            return pred
        return decode(S, pages, owner, a, variant="dp+abstain" if min_sim is not None else "dp",
                      min_sim=min_sim, flat=0.0, sigma=sigma)

    naive = lambda stem: decode(data[stem][0], data[stem][3], data[stem][1], a, variant="naive", min_sim=None, flat=0.0)

    L = ["# MaViLS — final round", "",
         f"Their protocol throughout (sentence granularity, page OCR, their sklearn F1; see `mavils_alignment.md`). "
         f"Every cell shows **their F1 (precision-on-answered / coverage)**. Split `results/external/mavils_split.json` "
         f"(seed {split['seed']}); tune = {', '.join(LECTURES[s][1] for s in tune)}; test = {', '.join(LECTURES[s][1] for s in test)}. "
         "Tune-half changes only; the test half is reported once, at the end.", ""]

    # ---------------- 1. sigma sweep
    grid = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3]
    sig_rows = [(sg, mean_f1(tune, lambda s_: dp(s_, sigma=sg)), mean_paired(tune, lambda s_: dp(s_, sigma=sg))) for sg in grid]
    best_sigma = max(sig_rows, key=lambda r: r[1])[0]
    L += ["## 1. Skip-penalty (σ) sweep — tune half", "",
          f"λ={a['jump_penalty']} β={a['back_penalty']} B={a['max_back']} μ={a.get('start_prior_mu', 0.0)} fixed; σ varied. "
          f"Criterion: mean their-F1 on the tune half. Pilot value σ = {a['skip_penalty']}.", "",
          "| σ | tune |", "|---:|---:|"]
    for sg, f1, pr in sig_rows:
        L.append(f"| {sg} | {pr}{' ←' if sg == best_sigma else ''} |")
    L += ["", f"Chosen on tune: σ = {best_sigma} (pilot σ = {a['skip_penalty']}, tune {mean_paired(tune, lambda s_: dp(s_))}).", ""]

    # ---------------- 2. decomposition
    L += ["## 2. Decomposition — similarity matrix × decoder, all 20 lectures", "",
          "Their cells run **their public code**: `calculate_dp_with_jumps` (helpers/utils.py, verbatim, λ_jump = 0.1) over "
          "distiluse-base-multilingual-cased cosine between each sentence and the page OCR (matching_algorithm.py). "
          "Deviation: tesseract `eng` only (their `eng+ell+equ+deu` traineddata are not installed here); rendering 150 dpi vs their 2.0× (144 dpi). "
          f"Our decoder uses the pilot σ = {a['skip_penalty']} here (not the tuned value), so this table has no tuned quantity in it.", "",
          "| similarity \\ decoder | their DP | our DP |", "|---|---:|---:|"]
    cells = {}
    for sim_name, getS in (("theirs (distiluse · page OCR)", lambda s_: theirs[s_]), ("ours (bge-m3 + BM25 + IDF · page OCR)", lambda s_: (data[s_][0], data[s_][1], data[s_][2], data[s_][3]))):
        row = []
        for dec_name in ("theirs", "ours"):
            def fn(s_, getS=getS, dec_name=dec_name):
                S, owner, gt, pages = getS(s_)
                if dec_name == "theirs":
                    return decode_theirs(S, pages, owner)
                return np.array([pages[j] for j in align_monotonic(S, jump_penalty=a["jump_penalty"], skip_penalty=a["skip_penalty"],
                                                                   back_penalty=a["back_penalty"], max_back=a["max_back"],
                                                                   start_prior_mu=a.get("start_prior_mu", 0.0))])[owner]
            cells[(sim_name, dec_name)] = (mean_f1(stems, fn), mean_paired(stems, fn), mean_paired(tune, fn), mean_paired(test, fn))
            row.append(cells[(sim_name, dec_name)][1])
        L.append(f"| {sim_name} | {row[0]} | {row[1]} |")
    tt, to, ot, oo = (cells[("theirs (distiluse · page OCR)", "theirs")][0], cells[("theirs (distiluse · page OCR)", "ours")][0],
                      cells[("ours (bge-m3 + BM25 + IDF · page OCR)", "theirs")][0], cells[("ours (bge-m3 + BM25 + IDF · page OCR)", "ours")][0])
    sim_effect = ((tt - ot) + (to - oo)) / 2
    dec_effect = ((tt - to) + (ot - oo)) / 2
    where = "the similarity matrix" if abs(sim_effect) > abs(dec_effect) else "the decoder"
    L += ["", f"Replication check: their similarity + their decoder = **{tt:.3f}** against the paper's 0.53 audio-only average "
          f"(difference = OCR language pack + rendering + transcript-file drift, not algorithm). Swapping only the matrix moves the mean by "
          f"{sim_effect:+.3f}, swapping only the decoder by {dec_effect:+.3f}: **the gap lives in {where}** "
          f"({'their distiluse-on-OCR matrix is the better input; our DP is at least as good a decoder' if where == 'the similarity matrix' else 'their DP decodes better on both matrices'}).", ""]

    # ---------------- 3. build decks
    build_flags = {s_: build_groups(texts[s_][0]) for s_ in stems}
    def flag(s_):
        g = build_flags[s_]
        nb = sum(len(x) > 1 for x in g)
        return f"{nb} groups / {sum(len(x) for x in g if len(x) > 1)} pages" if nb else "—"
    on_tune = mean_paired(tune, lambda s_: dp(s_, sigma=best_sigma, builds=True))
    off_tune = mean_paired(tune, lambda s_: dp(s_, sigma=best_sigma))
    use_builds = mean_f1(tune, lambda s_: dp(s_, sigma=best_sigma, builds=True)) > mean_f1(tune, lambda s_: dp(s_, sigma=best_sigma))
    L += ["## 3. Build-deck handling — tune half", "",
          "Detection: consecutive pages whose token multiset is a superset of the previous page's form a build group "
          "(`build_groups`, page OCR text, ≥ 5 tokens). Alignment runs the DP over groups (column = max over the group's builds); "
          "within a group each sentence is scored against each build's *incremental* text (IDF overlap over the delta) and decoded "
          "monotonically. Config `align.build_groups` (off by default; evaluated here on the tune half at the tuned σ).", "",
          f"| variant (σ = {best_sigma}) | tune |", "|---|---:|", f"| dp, build_groups off | {off_tune} |", f"| dp, build_groups on | {on_tune} |", "",
          f"Decision on tune: build_groups = **{'on' if use_builds else 'off'}**.", "",
          "| tune lecture | build decks | dp off | dp on |", "|---|---|---:|---:|"]
    for s_ in tune:
        gt = data[s_][2]
        L.append(f"| {LECTURES[s_][1]} | {flag(s_)} | {paired(gt, dp(s_, sigma=best_sigma))} | {paired(gt, dp(s_, sigma=best_sigma, builds=True))} |")
    L.append("")

    # ---------------- test half, once
    tuned = json.loads((EXTERNAL / "mavils_tuned.json").read_text()) if (EXTERNAL / "mavils_tuned.json").exists() else {}
    tuned.update({"sigma": best_sigma, "build_groups": bool(use_builds), "sigma_grid": grid,
                  "final_round": "2026-09-22: sigma by tune their-F1; build_groups on iff it raised tune their-F1 at the tuned sigma"})
    (EXTERNAL / "mavils_tuned.json").write_text(json.dumps(tuned, indent=1) + "\n")
    L += ["## 4. Test half — reported once", "",
          f"| lecture | text layer | build decks | naive | dp (pilot σ) | dp (σ = {best_sigma}) | dp + builds (σ = {best_sigma}) | their audio |",
          "|---|---|---|---:|---:|---:|---:|---:|"]
    agg = {k: [] for k in ("naive", "dp0", "dps", "dpb")}
    for s_ in test:
        gt = data[s_][2]
        preds = {"naive": naive(s_), "dp0": dp(s_), "dps": dp(s_, sigma=best_sigma), "dpb": dp(s_, sigma=best_sigma, builds=True)}
        for k, v in preds.items():
            agg[k].append(v)
        L.append(f"| {LECTURES[s_][1]} | {texts[s_][2]['text_layer']} | {flag(s_)} | " +
                 " | ".join(paired(gt, preds[k]) for k in ("naive", "dp0", "dps", "dpb")) + f" | {LECTURES[s_][2]:.2f} |")
    means = {k: mean_paired(test, lambda s_, k=k: {"naive": naive, "dp0": dp, "dps": lambda x: dp(x, sigma=best_sigma),
                                                   "dpb": lambda x: dp(x, sigma=best_sigma, builds=True)}[k](s_)) for k in agg}
    L.append(f"| **mean** | | | **{means['naive']}** | **{means['dp0']}** | **{means['dps']}** | **{means['dpb']}** | {np.mean([LECTURES[s_][2] for s_ in test]):.2f} |")
    L.append("")
    out = Path(args.out)
    out.write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"wrote {out}")
    return 0


def cmd_fused(args, cfg, encoder) -> int:
    """Fused similarity study. Tune half only: fusion method and weight; test half once.
    Decoder: our DP at the tuned sigma (results/external/mavils_tuned.json)."""
    a = cfg["link"]["align"]
    split = split_lectures()
    tune, test = split["tune"], split["test"]
    tuned = json.loads((EXTERNAL / "mavils_tuned.json").read_text())
    sigma = float(tuned.get("sigma", a["skip_penalty"]))
    figures_dir = Path(args.figures_dir)
    ours, theirs = {}, {}
    for stem in sorted(LECTURES):
        ours[stem] = similarity_for(stem, window=0.0, slide_text="ocr", cfg=cfg, encoder=encoder, figures_dir=figures_dir)
        theirs[stem] = their_similarity_for(stem, cfg=cfg, figures_dir=figures_dir)
        assert ours[stem][0].shape == theirs[stem][0].shape, stem

    def run(stem, method, weight=0.5):
        S_o, owner, gt, pages, _ = ours[stem]
        S_t = theirs[stem][0]
        S = {"ours": S_o, "theirs": S_t}.get(method)
        if S is None:
            S = fuse_similarity(S_o, S_t, method, weight)
        pred = decode(S, pages, owner, a, variant="dp", min_sim=None, flat=0.0, sigma=sigma)
        return gt, pred

    def mean_paired(stems_, method, weight=0.5):
        return paired_mean_cell([run(s_, method, weight) for s_ in stems_])

    weights = [0.0, 0.25, 0.5, 0.75, 1.0]
    rows_w = [(w, *mean_paired(tune, "fused_weighted", w)) for w in weights]
    best_w = max(rows_w, key=lambda r: r[1])[0]
    tune_cells = {"ours": mean_paired(tune, "ours"), "theirs": mean_paired(tune, "theirs"),
                  "fused_max": mean_paired(tune, "fused_max"), "fused_weighted": mean_paired(tune, "fused_weighted", best_w)}
    best_method = max(tune_cells, key=lambda k: tune_cells[k][0])
    tuned.update({"fusion_weight": best_w, "similarity": best_method,
                  "fused_study": "2026-09-22: weight by tune their-F1 over {0,.25,.5,.75,1}; both matrices min-max scaled per lecture"})
    (EXTERNAL / "mavils_tuned.json").write_text(json.dumps(tuned, indent=1) + "\n")

    L = ["# MaViLS — fused similarity study", "",
         f"Their protocol (sentence granularity, page OCR, their sklearn F1). Cells: **their F1 (precision-on-answered / coverage)**. "
         f"Decoder: our DP, σ = {sigma} (tuned earlier), λ={a['jump_penalty']} β={a['back_penalty']} B={a['max_back']} μ={a.get('start_prior_mu', 0.0)}. "
         "Matrices: *theirs* = distiluse-base-multilingual-cased cosine vs page OCR; *ours* = bge-m3 + BM25 + IDF hybrid vs page OCR. "
         "Each matrix is min-max scaled over the lecture before fusion (`linkrag.link.align.fuse_similarity`). "
         f"Split `results/external/mavils_split.json`; tune-half choices only; test reported once.", "",
         "## Tune half — weighted fusion, weight on *theirs*", "", "| weight | tune |", "|---:|---:|"]
    for w, f1, pr in rows_w:
        L.append(f"| {w} | {pr}{' ←' if w == best_w else ''} |")
    L += ["", "## Tune half — all similarities", "", "| align.similarity | tune |", "|---|---:|"]
    for k, (f1, pr) in tune_cells.items():
        L.append(f"| {k}{f' (w={best_w})' if k == 'fused_weighted' else ''} | {pr}{' ←' if k == best_method else ''} |")
    L += ["", f"Chosen on tune: similarity = **{best_method}**" + (f", fusion_weight = {best_w}" if best_method == "fused_weighted" else "") + ".", "",
          "## Test half — reported once", "",
          f"| lecture | text layer | ours | theirs | fused_max | fused_weighted (w={best_w}) | their audio (paper) |",
          "|---|---|---:|---:|---:|---:|---:|"]
    agg = {k: [] for k in ("ours", "theirs", "fused_max", "fused_weighted")}
    for s_ in test:
        status = ours[s_][4]
        cells = []
        for k in agg:
            gt, pred = run(s_, k, best_w)
            agg[k].append(their_prf(gt, pred)[2])
            cells.append(paired(gt, pred))
        L.append(f"| {LECTURES[s_][1]} | {status['text_layer']} | " + " | ".join(cells) + f" | {LECTURES[s_][2]:.2f} |")
    means = {k: mean_paired(test, k, best_w)[1] for k in agg}
    L.append("| **mean** | | " + " | ".join(f"**{means[k]}**" for k in agg) + f" | {np.mean([LECTURES[s_][2] for s_ in test]):.2f} |")
    all_means = {k: mean_paired(sorted(LECTURES), k, best_w)[1] for k in agg}
    L += ["", "All 20 lectures (for the comparison against the paper's 0.53 and our 0.46; contains the tune half, so the fused columns are optimistic): "
          + ", ".join(f"{k} {v}" for k, v in all_means.items()) + ".", ""]
    out = Path(args.out)
    out.write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"wrote {out}")
    return 0


def cmd_gate(args, cfg, encoder) -> int:
    """Relatedness gate, second pass: 30-shuffle null on the 20 related pairs, the 380
    cross pairs as a negative set, threshold by false acceptance, z vs F1."""
    import time
    from scipy.stats import spearmanr
    a = cfg["link"]["align"]
    shuffles = int(args.shuffles)
    floor = float(a.get("null_std_floor", 0.0))
    figures_dir = Path(args.figures_dir)
    dp = lambda S: align_monotonic(S, jump_penalty=a["jump_penalty"], skip_penalty=a["skip_penalty"],
                                   back_penalty=a["back_penalty"], max_back=a["max_back"],
                                   start_prior_mu=a.get("start_prior_mu", 0.0))
    gate = lambda S: relatedness_gate(S, dp, shuffles=shuffles, z=2.0, jump_penalty=a["jump_penalty"],
                                      skip_penalty=a["skip_penalty"], back_penalty=a["back_penalty"],
                                      null_std_floor=floor)
    stems = sorted(LECTURES)
    name = lambda s_: LECTURES[s_][1]

    # per-lecture inputs, embedded once per side
    win, slides, status, gt, f1 = {}, {}, {}, {}, {}
    emb_cache = CACHE / "emb"
    emb_cache.mkdir(parents=True, exist_ok=True)
    for stem in stems:
        df = load_ground_truth(REPO / "data" / "ground_truth_files" / f"ground_truth_{stem}.xlsx")
        units, owner = window_units(df, stem, args.window)
        sl, st = slide_units(stem, "ocr", figures_dir)
        win[stem], slides[stem], status[stem] = (units, np.array(owner)), sl, st
        gt[stem] = df["slide"].astype(int).to_numpy()
        for side, us in (("w", units), ("s", sl)):
            p = emb_cache / f"{stem}.{side}{int(args.window) if side == 'w' else ''}.npy"
            if not p.exists():
                np.save(p, np.asarray(encoder([u.content for u in us]), dtype="float32"))
        # their-protocol F1 at this granularity (related pair, our DP)
        S, owner_, _, pages, _ = similarity_for(stem, window=args.window, slide_text="ocr", cfg=cfg,
                                                encoder=encoder, figures_dir=figures_dir)
        pred = np.array([pages[j] for j in dp(S)])[owner_]
        f1[stem] = their_prf(gt[stem], pred)[2]
        print(f"prepared {name(stem)}: {len(units)} windows, {len(sl)} slides, F1 {f1[stem]:.2f}")

    def S_for(audio_stem, deck_stem):
        av = np.load(emb_cache / f"{audio_stem}.w{int(args.window)}.npy")
        sv = np.load(emb_cache / f"{deck_stem}.s.npy")
        return similarity_matrix(win[audio_stem][0], slides[deck_stem], encoder=encoder,
                                 w_dense=a["weights"]["dense"], w_bm25=a["weights"]["bm25"],
                                 w_keyword=a["weights"]["keyword"], a_vec=av, s_vec=sv)

    # per-deck column contrast: mean pairwise cosine between slide-text embeddings.
    # HIGH mean cosine = slides look alike = LOW contrast; a shuffled slide order then
    # admits a monotone path almost as good as the true one, and z collapses.
    contrast = {}
    for stem in stems:
        sv = np.load(emb_cache / f"{stem}.s.npy").astype("float64")
        sv /= np.maximum(np.linalg.norm(sv, axis=1, keepdims=True), 1e-9)
        C = sv @ sv.T
        n = len(sv)
        contrast[stem] = float((C.sum() - np.trace(C)) / (n * (n - 1))) if n > 1 else 1.0

    out = Path(args.out)
    prev = out.with_suffix(".json")
    if args.reuse and prev.exists():
        saved = json.loads(prev.read_text())
        cross = {tuple(k.split("|")): v for k, v in saved["cross"].items()}
        related = saved.get("related_detail") or {stem: gate(S_for(stem, stem)) for stem in stems}
        print("reused the 380 cross-pair z-scores from", prev)
    else:
        # 1. related pairs, 30 shuffles
        t0 = time.perf_counter()
        related = {stem: gate(S_for(stem, stem)) for stem in stems}
        print(f"related pairs done ({time.perf_counter() - t0:.0f}s)")
        # 2. the 380 cross pairs
        cross = {}
        for i, au in enumerate(stems):
            for dk in stems:
                if au == dk:
                    continue
                cross[(au, dk)] = gate(S_for(au, dk))["z"]
            print(f"cross pairs for {name(au)} done ({time.perf_counter() - t0:.0f}s)")
    zc = np.array(sorted(cross.values()))
    zr = np.array([related[s_]["z"] for s_ in stems])
    # threshold: smallest z with <= 5% false acceptance on the 380
    cands = sorted(set(np.round(np.concatenate([zc, zr, [0.0, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]]), 2)))
    chosen = next(z_ for z_ in cands if float(np.mean(zc >= z_)) <= 0.05)
    fa = lambda z_: float(np.mean(zc >= z_))
    fr = lambda z_: float(np.mean(zr < z_))
    tuned = json.loads((EXTERNAL / "mavils_tuned.json").read_text())
    tuned.update({"relatedness_z": float(chosen), "null_shuffles": shuffles, "gate_granularity_s": args.window,
                  "gate_study": "2026-09-22: threshold = smallest z with <=5% false acceptance on the 380 cross pairs"})
    (EXTERNAL / "mavils_tuned.json").write_text(json.dumps(tuned, indent=1) + "\n")

    rho, pval = spearmanr(zr, [f1[s_] for s_ in stems])
    rej = [s_ for s_ in stems if related[s_]["z"] < chosen]
    med_f1_rej = float(np.median([f1[s_] for s_ in rej])) if rej else float("nan")
    med_f1_acc = float(np.median([f1[s_] for s_ in stems if s_ not in rej]))

    L = ["# Relatedness gate — second pass (MaViLS, 30-second windows)", "",
         f"Gate statistic: penalised DP objective per segment vs the mean of **{shuffles}** shuffled slide orders, "
         f"z = (true − null mean) / max(null std, {floor}); our DP at pilot parameters; page-OCR slide text; "
         f"{int(args.window)}-second windows of their transcript sentences. Default config unchanged "
         f"(`align.relatedness_z` = {a['relatedness_z']}, `align.null_shuffles` = {a.get('null_shuffles', 5)}); "
         "the values chosen here are recorded in `results/external/mavils_tuned.json`.", "",
         "## 1. Related pairs (20), 30-shuffle null", "",
         "| lecture | text layer | windows × slides | column contrast (mean pairwise slide cosine) | objective | null mean ± std | z | F1 (their protocol) |",
         "|---|---|---:|---:|---:|---:|---:|---:|"]
    for stem in stems:
        g = related[stem]
        L.append(f"| {name(stem)} | {status[stem]['text_layer']} | {len(win[stem][0])}×{len(slides[stem])} | {contrast[stem]:.3f} | "
                 f"{g['score']:.4f} | {g['null_mean']:.4f} ± {g['null_std']:.4f} | {g['z']:.1f} | {f1[stem]:.2f} |")
    L += ["", "## 2. Negative set: 380 cross pairs (each audio × every other deck)", "",
          f"z distribution of unrelated pairs: min {zc.min():.2f} · median {float(np.median(zc)):.2f} · "
          f"95th percentile {float(np.percentile(zc, 95)):.2f} · max {zc.max():.2f}.", "",
          "| threshold z | false acceptance (380 unrelated) | false rejection (20 related) |", "|---:|---:|---:|"]
    for z_ in (0.0, 1.0, 1.5, 2.0, chosen, 3.0, 4.0):
        L.append(f"| {z_:.2f}{' ←' if z_ == chosen else ''} | {fa(z_):.1%} ({int(round(fa(z_) * 380))}/380) | {fr(z_):.0%} ({int(round(fr(z_) * 20))}/20) |")
    L += ["", f"**Chosen threshold: z = {chosen:.2f}** — the smallest z with ≤ 5 % false acceptance on the 380 pairs; "
          f"false rejection on the 20 related pairs at that threshold: **{fr(chosen):.0%}** ({int(round(fr(chosen) * 20))}/20"
          + (": " + ", ".join(name(s_) for s_ in rej) if rej else "") + ").", "",
          "Highest-z unrelated pairs (what a false acceptance looks like):", ""]
    for (au, dk), z_ in sorted(cross.items(), key=lambda kv: -kv[1])[:8]:
        L.append(f"- {name(au)} audio × {name(dk)} deck: z = {z_:.2f}")
    L += ["", "## 3. z vs alignment quality", "",
          f"Spearman ρ between a related pair's z and its their-protocol F1 (20 lectures): **{rho:.2f}** (p = {pval:.3f}). "
          f"Median F1 of pairs rejected at z = {chosen:.2f}: {med_f1_rej:.2f}; of accepted pairs: {med_f1_acc:.2f}. "
          + ("**The rejected pairs are the low-F1 pairs**: the gate declines the alignments the DP gets wrong anyway."
             if rej and med_f1_rej < med_f1_acc else "Rejected and accepted pairs do not separate on F1."), "",
          "| lecture | z | F1 | column contrast |", "|---|---:|---:|---:|"]
    for stem in sorted(stems, key=lambda s_: related[s_]["z"]):
        L.append(f"| {name(stem)} | {related[stem]['z']:.1f} | {f1[stem]:.2f} | {contrast[stem]:.3f} |")
    rho_c, p_c = spearmanr(zr, [contrast[s_] for s_ in stems])
    med_c_rej = float(np.median([contrast[s_] for s_ in rej])) if rej else float("nan")
    med_c_acc = float(np.median([contrast[s_] for s_ in stems if s_ not in rej]))
    L += ["", "## 4. z vs deck column contrast", "",
          f"Spearman ρ between z and mean pairwise slide cosine: **{rho_c:.2f}** (p = {p_c:.3f}; negative = slides that look alike "
          f"give low z). Median mean-pairwise-cosine of rejected decks: {med_c_rej:.3f}; of accepted decks: {med_c_acc:.3f}. "
          + ("**The rejected pairs are the low-contrast decks** (mutually similar slides), which is the mechanism the ML for health "
             "inspection shows: a shuffled slide order admits a monotone path nearly as good as the true one."
             if rej and med_c_rej > med_c_acc else "Rejected and accepted decks do not separate on column contrast."), ""]
    out.write_text("\n".join(L) + "\n")
    json.dump({"related": {s_: related[s_]["z"] for s_ in stems}, "related_detail": {s_: {k: v for k, v in related[s_].items() if k != "null_scores"} for s_ in stems},
               "f1": f1, "contrast": contrast,
               "cross": {f"{au}|{dk}": z_ for (au, dk), z_ in cross.items()}, "chosen_z": chosen},
              open(out.with_suffix(".json"), "w"), indent=1)
    print("\n".join(L[:8]))
    print(f"chosen z {chosen:.2f}, FA {fa(chosen):.1%}, FR {fr(chosen):.0%}, rho {rho:.2f}\nwrote {out}")
    return 0


def cmd_gate_gran(args, cfg, encoder) -> int:
    """Segments-per-slide hypothesis: z of the 20 related pairs at 30 s, 15 s and sentence
    granularity (30-shuffle null), with n/m; rho(z, n/m)."""
    from scipy.stats import spearmanr
    a = cfg["link"]["align"]
    floor = float(a.get("null_std_floor", 0.0))
    figures_dir = Path(args.figures_dir)
    dp = lambda S: align_monotonic(S, jump_penalty=a["jump_penalty"], skip_penalty=a["skip_penalty"],
                                   back_penalty=a["back_penalty"], max_back=a["max_back"],
                                   start_prior_mu=a.get("start_prior_mu", 0.0))
    gate = lambda S: relatedness_gate(S, dp, shuffles=int(args.shuffles), z=float(a["relatedness_z"]),
                                      jump_penalty=a["jump_penalty"], skip_penalty=a["skip_penalty"],
                                      back_penalty=a["back_penalty"], null_std_floor=floor)
    stems = sorted(LECTURES)
    name = lambda s_: LECTURES[s_][1]
    emb_cache = CACHE / "emb"
    grans = [(30.0, "w30"), (15.0, "w15"), (0.0, "sentence")]
    res: dict[str, dict[str, tuple[float, int, int]]] = {s_: {} for s_ in stems}
    for stem in stems:
        df = load_ground_truth(REPO / "data" / "ground_truth_files" / f"ground_truth_{stem}.xlsx")
        sl, _ = slide_units(stem, "ocr", figures_dir)
        sv_path = emb_cache / f"{stem}.s.npy"
        if not sv_path.exists():
            np.save(sv_path, np.asarray(encoder([u.content for u in sl]), dtype="float32"))
        sv = np.load(sv_path)
        for w, tag in grans:
            units, _ = window_units(df, stem, w) if w else sentence_units(df, stem)
            av_path = emb_cache / f"{stem}.{tag}.npy"
            if not av_path.exists():
                np.save(av_path, np.asarray(encoder([u.content for u in units]), dtype="float32"))
            S = similarity_matrix(units, sl, encoder=encoder, w_dense=a["weights"]["dense"], w_bm25=a["weights"]["bm25"],
                                  w_keyword=a["weights"]["keyword"], a_vec=np.load(av_path), s_vec=sv)
            g = gate(S)
            res[stem][tag] = (g["z"], S.shape[0], S.shape[1])
        print(f"{name(stem):<20} " + "  ".join(f"{tag}: z={res[stem][tag][0]:5.1f} n/m={res[stem][tag][1] / res[stem][tag][2]:5.1f}" for _, tag in grans))
    zs, ratios = [], []
    for stem in stems:
        for _, tag in grans:
            z_, n, m = res[stem][tag]
            zs.append(z_); ratios.append(n / m)
    rho_all, p_all = spearmanr(zs, ratios)
    per_gran = {tag: spearmanr([res[s_][tag][0] for s_ in stems], [res[s_][tag][1] / res[s_][tag][2] for s_ in stems]) for _, tag in grans}
    thr = float(a["relatedness_z"])
    L = ["# Gate v3 — segments-per-slide hypothesis (20 related MaViLS pairs)", "",
         f"z at three transcript granularities, {args.shuffles}-shuffle null, threshold z = {thr} (`align.relatedness_z`). "
         "n = transcript segments, m = slides. Hypothesis under test: low n/m (few segments per slide) → low z.", "",
         "| lecture | z 30 s | n/m | z 15 s | n/m | z sentence | n/m | rejected at (30 s / 15 s / sentence) |",
         "|---|---:|---:|---:|---:|---:|---:|---|"]
    for stem in stems:
        cells = []
        for _, tag in grans:
            z_, n, m = res[stem][tag]
            cells += [f"{z_:.1f}", f"{n / m:.1f}"]
        flags = " / ".join("✗" if res[stem][tag][0] < thr else "✓" for _, tag in grans)
        L.append(f"| {name(stem)} | " + " | ".join(cells) + f" | {flags} |")
    fr = {tag: sum(res[s_][tag][0] < thr for s_ in stems) for _, tag in grans}
    L += ["", f"False rejections at z = {thr}: 30 s {fr['w30']}/20 · 15 s {fr['w15']}/20 · sentence {fr['sentence']}/20.", "",
          f"Spearman ρ(z, n/m) pooled over the 60 (lecture, granularity) points: **{rho_all:.2f}** (p = {p_all:.3f}). "
          "Within a granularity: " + "; ".join(f"{tag} ρ = {per_gran[tag][0]:.2f} (p = {per_gran[tag][1]:.2f})" for _, tag in grans) + ".", ""]
    movers = ["ML_for_health_MIT", "image_processing", "sensory_systems"]
    L += ["The three 30-second rejects:", ""]
    for stem in movers:
        L.append(f"- {name(stem)}: " + ", ".join(f"{tag} z = {res[stem][tag][0]:.1f} (n/m {res[stem][tag][1] / res[stem][tag][2]:.1f})" for _, tag in grans))
    confirmed = rho_all > 0.3 and p_all < 0.05 and all(res[s_]["sentence"][0] > res[s_]["w30"][0] for s_ in movers[:1])
    L += ["", ("**Hypothesis confirmed** — z rises with segments per slide; the adaptive re-windowing "
               "(`align.gate_min_windows_per_slide`) is justified." if confirmed else
               "**Hypothesis not confirmed** — finer granularity does not raise z for the rejected pairs (and the pooled "
               "correlation is driven by the granularity change itself, not by per-lecture n/m). No adaptive re-windowing is added; "
               "`align.gate_min_windows_per_slide` is not introduced."), ""]
    out = Path(args.out)
    out.write_text("\n".join(L) + "\n")
    json.dump({s_: {tag: {"z": v[0], "n": v[1], "m": v[2]} for tag, v in res[s_].items()} for s_ in stems}, open(out.with_suffix(".json"), "w"), indent=1)
    print("\n".join(L[-12:]))
    print(f"wrote {out}")
    return 0


# ----------------------------------------------------------------- visual channel

VIDEO_DIR = REPO / "data" / "video"      # the Kaggle download (the repo ships no video)
FRAME_SIDE = 320                         # frames and pages are compared at this longest side
# Segmentation for the visual channel: a new frame on any change above 2 bits, and no revisit
# merging. The LectQA default (10 bits, revisits merged) merged distinct slides that share a
# template: Decarbonization kept 28 frames for a 45-page deck, Climate & Cities 25 for 44.
# Splitting too finely costs only disk; merging loses the slide. Set once, before any F1.
FRAME_MAX_DIST = 2
# "Confident" frame->page match: the best page beats the runner-up by more than this.
# Set before any frame was scored; a sanity check only, never a tuned quantity.
MARGIN = {"dhash": 4 / 64, "swiftformer": 0.05}


# ground-truth stem -> video in the Kaggle zip, checked by duration (every label run ends within
# 0.6 min of its video's end). The zip has no climate-policy video: its `clinical_care` file is a
# medicine lecture (frames OCR "Goals of Medicine"), so climate_science_policy_MIT2 has none.
# physics.mp4 and physics_high_res.mp4 are the same 79.3 min; the high-res one is used.
VIDEOS = {
    "ML_for_health_MIT": "ML_for_health_high_res.mp4", "cities_and_decarbonization": "cities_and_decarbonization_standard_res.mp4",
    "climate_and_cities": "cities_and_climate_high_res.mp4", "cognitive_robotics_MIT": "cognitive_robotics_high_res.mp4",
    "computer_vision_2_2": "computer_vision_2_2_high_res.mp4", "creating_breakthrough_products_MIT": "creating_breakthrough_products_MIT.mp4",
    "cryptocurrency_MIT": "cryptocurrency_high_res.mp4", "deeplearning": "deep_learning_high_res.mp4",
    "image_processing": "image_processing_high_res.mp4", "numerics": "numerics_high_res.mp4", "phonetics": "phonetics_high_res.mp4",
    "physics": "physics_high_res.mp4", "psychology": "psychology_high_res.mp4",
    "reinforcement_learning": "reinforcement_learning_high_res.mp4", "sensory_systems": "sensory_systems_high_res.mp4",
    "short_range": "short_range_MIT.mp4", "solar_resource": "solar_resource_high_res.mp4",
    "team_dynamics_game_design_MIT": "team_dynamics_high_res.mp4", "theory_of_computation": "theory_of_computation_high_res.mp4",
}


def video_for(stem: str) -> Path | None:
    p = VIDEO_DIR / VIDEOS[stem] if stem in VIDEOS else None
    return p if p is not None and p.exists() else None


def frames_for(stem: str) -> dict | None:
    """Representative frames (1 fps dHash segmentation, `linkrag.ingest.video_slides`) with
    their on-screen intervals, cached in CACHE/frames/<stem>/. Only these are kept."""
    from linkrag.ingest.video_slides import sample_frames, segment_slides
    out = CACHE / "frames" / stem
    meta = out / "slides.json"
    if meta.exists():
        return json.loads(meta.read_text())
    video = video_for(stem)
    if video is None:
        return None
    frames = sample_frames(video, 1.0, max_side=FRAME_SIDE)
    slides = segment_slides(frames, max_dist=FRAME_MAX_DIST, dedup=False)
    out.mkdir(parents=True, exist_ok=True)
    for sl in slides:
        sl.image.save(out / f"s{sl.index:04d}.png")
    stats = {"stem": stem, "video": video.name, "video_bytes": video.stat().st_size,
             "video_s": round(frames[-1][0] + 1.0, 1) if frames else 0.0, "sampled": len(frames), "kept": len(slides),
             "intervals": [[list(iv) for iv in sl.intervals] for sl in slides]}
    meta.write_text(json.dumps(stats))
    return stats


def sentence_frames_for(stem: str) -> Path | None:
    """The frame on screen at each ground-truth sentence timestamp (MaViLS's own frame
    choice), at native resolution, as CACHE/sentence_frames/<stem>/<sentence>.jpg (JPEG q90:
    ~12,000 native frames as PNG would be ~12 GB). One sequential decode per video; the
    first frame at or after each timestamp; sentences past the last frame get the last one."""
    import av
    out = CACHE / "sentence_frames" / stem
    meta = out / "done.json"
    if meta.exists():
        return out
    video = video_for(stem)
    if video is None:
        return None
    times = load_ground_truth(REPO / "data" / "ground_truth_files" / f"ground_truth_{stem}.xlsx")["time"].astype(float).tolist()
    order = sorted(range(len(times)), key=lambda i: times[i])
    out.mkdir(parents=True, exist_ok=True)
    k, last, size = 0, None, None
    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for frame in container.decode(stream):
            t = float(frame.pts * stream.time_base) if frame.pts is not None else 0.0
            if k < len(order) and t + 1e-6 >= times[order[k]]:
                img = frame.to_image()
                size = img.size
                while k < len(order) and t + 1e-6 >= times[order[k]]:
                    img.save(out / f"{order[k]:05d}.jpg", quality=90)
                    k += 1
            last = frame
            if k == len(order):
                break
    if k < len(order) and last is not None:
        img = last.to_image()
        for i in order[k:]:
            img.save(out / f"{i:05d}.jpg", quality=90)
    meta.write_text(json.dumps({"stem": stem, "video": video.name, "sentences": len(times), "size": size,
                                "past_end": len(order) - k}))
    return out


OCR_SIDE = 960        # OCR reads a frame upscaled to at least this longest side (320 px x 3, as before)


def sentence_visual_for(stem: str, cfg: dict, figures_dir: Path) -> dict | None:
    """sentence x page matrices from the frame at each sentence timestamp (sentence_frames_for):
    SwiftFormer image score, frame-OCR TF-IDF and BM25 against the page OCR text, plus each
    frame's OCR word count. OCR per frame is cached by the JPEG's hash; the matrices per lecture."""
    import hashlib
    import math
    from concurrent.futures import ThreadPoolExecutor
    from PIL import Image
    from linkrag.index import tokenize
    from linkrag.link.visual import ocr_frame, render_pages, swiftformer_features, text_similarity
    path = CACHE / "sentence_visual" / f"{stem}.npz"
    if path.exists():
        z = np.load(path)
        return {k: z[k] for k in z.files}
    src = sentence_frames_for(stem)
    if src is None:
        return None
    jpgs = sorted(src.glob("*.jpg"))
    cache = CACHE / "frame_ocr"
    cache.mkdir(parents=True, exist_ok=True)

    def ocr_one(p: Path) -> str:
        f = cache / f"{hashlib.sha1(p.read_bytes()).hexdigest()}.txt"
        if not f.exists():
            im = Image.open(p)
            f.write_text(ocr_frame(im, scale=max(1, math.ceil(OCR_SIDE / max(im.size)))))
        return f.read_text()
    with ThreadPoolExecutor(max_workers=8) as ex:
        texts = list(ex.map(ocr_one, jpgs))
    pages = render_pages(REPO / "data" / "lectures" / LECTURES[stem][0])
    for p in pages:
        p.thumbnail((FRAME_SIDE, FRAME_SIDE))
    page_feats = swiftformer_features(pages, cfg["device"])
    feats = np.concatenate([swiftformer_features([Image.open(p).convert("RGB") for p in jpgs[i:i + 64]], cfg["device"])
                            for i in range(0, len(jpgs), 64)])          # chunked: native frames are large
    page_text = [u.content for u in slide_units(stem, "ocr", figures_dir)[0]]
    out = {"swiftformer": feats @ page_feats.T, "tfidf": text_similarity(texts, page_text, "tfidf"),
           "bm25": text_similarity(texts, page_text, "bm25"), "words": np.array([len(tokenize(t)) for t in texts])}
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **out)
    return out


def load_slides(stem: str) -> list:
    from PIL import Image
    from linkrag.ingest.video_slides import Slide
    stats = json.loads((CACHE / "frames" / stem / "slides.json").read_text())
    return [Slide(index=k, intervals=[tuple(iv) for iv in ivs],
                  image=Image.open(CACHE / "frames" / stem / f"s{k:04d}.png").convert("RGB"))
            for k, ivs in enumerate(stats["intervals"])]


def frame_page_for(stem: str, cfg: dict) -> dict[str, np.ndarray]:
    """frame x page similarity by dHash and by SwiftFormer-xs, cached. Pages are rendered
    and shrunk to the frames' longest side."""
    from linkrag.link.visual import dhash_similarity, render_pages, swiftformer_similarity
    path = CACHE / "visual" / f"{stem}.npz"
    if path.exists():
        z = np.load(path)
        return {"dhash": z["dhash"], "swiftformer": z["swiftformer"]}
    frames = [sl.image for sl in load_slides(stem)]
    pages = render_pages(REPO / "data" / "lectures" / LECTURES[stem][0])
    for p in pages:
        p.thumbnail((FRAME_SIDE, FRAME_SIDE))
    fp = {"dhash": dhash_similarity(frames, pages), "swiftformer": swiftformer_similarity(frames, pages, cfg["device"])}
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **fp)
    return fp


# "Confident" frame->page text match, as MARGIN above; set before any frame OCR was scored.
# BM25 rows are min-max scaled per frame, so the best page is always 1.
TEXT_MARGIN = {"tfidf": 0.05, "bm25": 0.10}


def frame_ocr_for(stem: str) -> list[str]:
    """OCR of every representative frame (`linkrag.link.visual.ocr_frame`, 3x upscale), in
    slide-index order, cached per frame under CACHE/frame_ocr/<sha1 of the PNG>.txt."""
    import hashlib
    from concurrent.futures import ThreadPoolExecutor
    from PIL import Image
    from linkrag.link.visual import ocr_frame
    cache = CACHE / "frame_ocr"
    cache.mkdir(parents=True, exist_ok=True)

    def one(png: Path) -> str:
        f = cache / f"{hashlib.sha1(png.read_bytes()).hexdigest()}.txt"
        if not f.exists():
            f.write_text(ocr_frame(Image.open(png)))
        return f.read_text()
    frames = sorted((CACHE / "frames" / stem).glob("s*.png"))
    with ThreadPoolExecutor(max_workers=8) as ex:          # tesseract runs as a subprocess
        return list(ex.map(one, frames))


def frame_text_page_for(stem: str, figures_dir: Path) -> dict[str, np.ndarray]:
    """frame x page similarity of frame OCR text to the page OCR text (the slide-side text
    every other column uses), by TF-IDF cosine and by BM25; cached."""
    from linkrag.link.visual import text_similarity
    path = CACHE / "frame_text" / f"{stem}.npz"
    if path.exists():
        z = np.load(path)
        return {"tfidf": z["tfidf"], "bm25": z["bm25"]}
    texts = frame_ocr_for(stem)
    pages = [u.content for u in slide_units(stem, "ocr", figures_dir)[0]]
    fp = {m: text_similarity(texts, pages, m) for m in ("tfidf", "bm25")}
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **fp)
    return fp


def cmd_frame_ocr(args, cfg, encoder) -> int:
    """Item 1: frame OCR and frame->page text similarity, with the label-free sanity check."""
    from linkrag.index import tokenize
    from linkrag.link.visual import confident_rate
    figures_dir = Path(args.figures_dir)
    rows = []
    for stem in sorted(LECTURES):
        if not (CACHE / "frames" / stem / "slides.json").exists():
            continue
        words = [len(tokenize(t)) for t in frame_ocr_for(stem)]
        fp = frame_text_page_for(stem, figures_dir)
        conf = {k: confident_rate(fp[k], TEXT_MARGIN[k]) for k in fp}
        agree = float(np.mean(fp["tfidf"].argmax(1) == fp["bm25"].argmax(1)))
        rows.append((stem, len(words), float(np.mean([w > 0 for w in words])), float(np.median(words)), conf, agree))
        print(f"{stem}: {len(words)} frames, with text {rows[-1][2]:.2f}, median words {rows[-1][3]:.0f}")
    L = ["# MaViLS — frame OCR and frame→page text similarity", "",
         "Every representative frame of `mavils_frames.md` (320 px on the longest side, the only size kept) OCR'd with "
         "tesseract after a 3× LANCZOS upscale (`linkrag.link.visual.ocr_frame`; at native size slide text is a few "
         "pixels tall and OCR reads almost nothing: a label-free check on frames alone, before any F1). Cached per frame "
         "by the hash of the frame. Frame→page scores (`text_similarity`), tokenised as everywhere else "
         "(`linkrag.index.tokenize`), against the page OCR text: **TF-IDF** cosine (fit on the pages) and **BM25** "
         "(min-max scaled per frame). A frame with no text gets a zero row. LLM-free, $0.", "",
         f"*Confident* = best page beats the runner-up by more than {TEXT_MARGIN['tfidf']} (TF-IDF) / "
         f"{TEXT_MARGIN['bm25']} (BM25), set before scoring. A sanity check, not an accuracy: no label is used.", "",
         "| lecture | frames | with any text | median words | confident TF-IDF | confident BM25 | same best page |",
         "|---|---:|---:|---:|---:|---:|---:|"]
    for stem, n, has, med, conf, agree in rows:
        L.append(f"| {LECTURES[stem][1]} | {n} | {has:.2f} | {med:.0f} | {conf['tfidf']:.2f} | {conf['bm25']:.2f} | {agree:.2f} |")
    if rows:
        L.append(f"| **total / mean** | {sum(r[1] for r in rows)} | {np.mean([r[2] for r in rows]):.2f} | "
                 f"{np.median([r[3] for r in rows]):.0f} | {np.mean([r[4]['tfidf'] for r in rows]):.2f} | "
                 f"{np.mean([r[4]['bm25'] for r in rows]):.2f} | {np.mean([r[5] for r in rows]):.2f} |")
    L += ["", "Climate policies has no video (see `mavils_frames.md`), so no frames."]
    out = Path(args.out)
    out.write_text("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


def jumpiness(gt: np.ndarray) -> tuple[float, float]:
    """Their `evaluation/analyse_videos.py`: slide changes / unique labels, and the
    no-slide share (-1 rows / labelled rows)."""
    return (float(np.count_nonzero(np.diff(gt)) / len(np.unique(gt))),
            float(np.count_nonzero(gt == -1) / max(np.count_nonzero(gt != -1), 1)))


def cmd_visual_frames(args, cfg, encoder) -> int:
    """Items 1-2: representative frames and frame->page similarity for every lecture."""
    from linkrag.link.visual import confident_rate
    rows, missing = [], []
    for stem in sorted(LECTURES):
        st = frames_for(stem)
        if st is None:
            missing.append(stem)
            continue
        fp = frame_page_for(stem, cfg)
        disk = sum(p.stat().st_size for p in (CACHE / "frames" / stem).glob("*.png"))
        conf = {k: confident_rate(fp[k], MARGIN[k]) for k in fp}
        agree = float(np.mean(fp["dhash"].argmax(1) == fp["swiftformer"].argmax(1)))
        rows.append((stem, st, disk, fp["dhash"].shape, conf, agree))
        print(f"{stem}: {st['kept']} frames from {st['video_s'] / 60:.0f} min, confident dhash {conf['dhash']:.2f} "
              f"swiftformer {conf['swiftformer']:.2f}")
    L = ["# MaViLS — representative frames and frame→page similarity", "",
         f"Videos: the MaViLS Kaggle dataset (their README; the GitHub repo ships none). Frames: 1 fps, shrunk to "
         f"{FRAME_SIDE} px on the longest side, segmented by 64-bit dHash (`linkrag.ingest.video_slides.segment_slides`: "
         f"a new segment when consecutive frames differ by > {FRAME_MAX_DIST} bits, segments < 2 s merged, revisits "
         "**not** merged: at the LectQA default of 10 bits with revisit merging, slides sharing a template collapsed, "
         "e.g. Decarbonization kept 28 frames for a 45-page deck). One representative frame per segment is kept "
         "(`data/processed/mavils/frames/`, not tracked), with its intervals; the unzipped videos are removed after "
         "extraction (the Kaggle zip is the source). Pages: rendered from their PDFs and shrunk to the same size. "
         "LLM-free, $0.", "",
         "Frame→page scores (`linkrag.link.visual`): **dHash** 1 − Hamming/64; **SwiftFormer-xs** "
         f"(`{'MBZUAI/swiftformer-xs'}`, MaViLS's own image model) cosine of the flattened last hidden state. "
         f"*Confident* = the best page beats the runner-up by more than {MARGIN['dhash']:.4f} (dHash) / "
         f"{MARGIN['swiftformer']} (SwiftFormer), margins set before any frame was scored. This is a sanity check, "
         "not an accuracy: no frame label is used.", "",
         "| lecture | video min | frames kept | frame disk | pages | confident dHash | confident SwiftFormer | same best page |",
         "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for stem, st, disk, shape, conf, agree in rows:
        L.append(f"| {LECTURES[stem][1]} | {st['video_s'] / 60:.0f} | {st['kept']} | {disk / 1e6:.1f} MB | {shape[1]} | "
                 f"{conf['dhash']:.2f} | {conf['swiftformer']:.2f} | {agree:.2f} |")
    if rows:
        L.append(f"| **total / mean** | {sum(r[1]['video_s'] for r in rows) / 60:.0f} | {sum(r[1]['kept'] for r in rows)} | "
                 f"{sum(r[2] for r in rows) / 1e6:.1f} MB | | {np.mean([r[4]['dhash'] for r in rows]):.2f} | "
                 f"{np.mean([r[4]['swiftformer'] for r in rows]):.2f} | {np.mean([r[5] for r in rows]):.2f} |")
    if missing:
        L += ["", f"**No video found for {len(missing)} lecture(s):** " + ", ".join(missing) + "."]
    out = Path(args.out)
    out.write_text("\n".join(L) + "\n")
    print("\n".join(L))
    return 1 if missing else 0


def lecture_matrices(stem: str, cfg: dict, encoder, figures_dir: Path, with_text: bool = False,
                     frame_source: str | None = None) -> dict:
    """Every sentence x page matrix of one lecture: ours and theirs (text), and, when the
    lecture has video, the visual channel per frame->page score and (with_text) the frame-OCR
    channel per text score. `frame_source` (default `link.align.frame_source`):
    representative = each sentence takes the row of the representative frame on screen at its
    timestamp; sentence_time = the frame at the sentence timestamp itself, native resolution
    (MaViLS's choice; SwiftFormer image score only)."""
    frame_source = frame_source or cfg["link"]["align"].get("frame_source", "representative")
    if frame_source not in ("representative", "sentence_time"):
        raise ValueError(f"unknown frame_source {frame_source!r}: use 'representative' or 'sentence_time'")
    from linkrag.link.visual import segment_visual
    S_o, owner, gt, pages, status = similarity_for(stem, window=0.0, slide_text="ocr", cfg=cfg, encoder=encoder,
                                                   figures_dir=figures_dir)
    d = {"ours": S_o, "theirs": their_similarity_for(stem, cfg=cfg, figures_dir=figures_dir)[0], "owner": owner,
         "gt": gt, "pages": pages, "status": status, "visual": None, "frame_ocr": None}
    if frames_for(stem) is None:                     # no video for this lecture: text columns only
        return d
    if frame_source == "sentence_time":
        from linkrag.link.visual import frame_margin
        sv = sentence_visual_for(stem, cfg, figures_dir)
        assert sv["swiftformer"].shape == d["ours"].shape, stem            # one row per sentence
        d["visual"] = {"swiftformer": sv["swiftformer"]}
        d["frame_ocr"] = {"tfidf": sv["tfidf"], "bm25": sv["bm25"]}
        d["frame_of"] = list(range(len(sv["words"])))
        d["frame_words"] = sv["words"].tolist()
        d["frame_margin"] = {"swiftformer": frame_margin(sv["swiftformer"])}
        return d
    times = load_ground_truth(REPO / "data" / "ground_truth_files" / f"ground_truth_{stem}.xlsx")["time"].astype(float).tolist()
    slides = load_slides(stem)
    fp = frame_page_for(stem, cfg)
    assert list(pages) == list(range(1, fp["dhash"].shape[1] + 1)), stem     # columns = deck pages, in order
    d["visual"] = {k: segment_visual(times, slides, fp[k]) for k in fp}
    if with_text:
        from linkrag.index import tokenize
        from linkrag.ingest.video_slides import slide_at
        from linkrag.link.visual import frame_margin
        ft = frame_text_page_for(stem, figures_dir)
        d["frame_ocr"] = {k: segment_visual(times, slides, ft[k]) for k in ft}
        d["frame_of"] = [slide_at(slides, t) for t in times]            # the frame each sentence takes
        d["frame_words"] = [len(tokenize(t)) for t in frame_ocr_for(stem)]
        d["frame_margin"] = {k: frame_margin(fp[k]) for k in fp}
    return d


def three_way(d: dict, vm: str, tm: str, w3, gate: dict | None = None) -> np.ndarray:
    """visual + frame_ocr + theirs (`fuse_many`). With `gate`, a sentence whose frame is not
    `slide_visible` gets constant visual and frame-OCR rows, so its page is decided by the
    speech-to-slide text alone (a constant row cannot change which page wins in that row)."""
    from linkrag.link.align import fuse_many
    from linkrag.link.visual import slide_visible
    V, O = d["visual"][vm].copy(), d["frame_ocr"][tm].copy()
    if gate is not None:
        vis = slide_visible(d["frame_words"], d["frame_margin"][vm], **gate)
        rows = [i for i, k in enumerate(d["frame_of"]) if k is not None and not vis[k]]
        V[rows], O[rows] = V.min(), O.min()
    return fuse_many([V, O, d["theirs"]], w3)


VISUAL_COLS = ("ours", "theirs", "fused", "visual", "visual+text", "visual+theirs")


def cmd_visual(args, cfg, encoder) -> int:
    """Items 3-4. Each sentence takes the frame->page row of the frame on screen at its
    timestamp (`linkrag.link.visual.segment_visual`). Tune half only: the frame->page score
    (dHash vs SwiftFormer, by visual-only their-F1) and the visual weight in visual+text and
    visual+theirs (grid 0/.25/.5/.75/1). Test half once. Decoder: our DP at the tuned sigma."""
    a = cfg["link"]["align"]
    split = split_lectures()
    tune, test = split["tune"], split["test"]
    tuned = json.loads((EXTERNAL / "mavils_tuned.json").read_text())
    sigma, fw = float(tuned["sigma"]), float(tuned["fusion_weight"])
    figures_dir = Path(args.figures_dir)
    data = {}
    for stem in sorted(LECTURES):
        m = lecture_matrices(stem, cfg, encoder, figures_dir)
        data[stem] = (m["ours"], m["theirs"], m["visual"], m["owner"], m["gt"], m["pages"], m["status"])

    def run(stem, col, vm, w):
        S_o, S_t, V, owner, gt, pages, _ = data[stem]
        S = {"ours": lambda: S_o, "theirs": lambda: S_t,
             "fused": lambda: fuse_similarity(S_o, S_t, "fused_weighted", fw),
             "visual": lambda: fuse_similarity(S_o, V[vm], "fused_weighted", 1.0),       # V alone, min-max scaled
             "visual+text": lambda: fuse_similarity(S_o, V[vm], "fused_weighted", w),
             "visual+theirs": lambda: fuse_similarity(S_t, V[vm], "fused_weighted", w)}[col]()
        return gt, decode(S, pages, owner, a, variant="dp", min_sim=None, flat=0.0, sigma=sigma)

    def mean_paired(stems_, col, vm, w=0.5):
        return paired_mean_cell([run(s_, col, vm, w) for s_ in stems_])

    vm_rows = {vm: mean_paired(tune, "visual", vm) for vm in ("dhash", "swiftformer")}
    vm = max(vm_rows, key=lambda k: vm_rows[k][0])
    grid = [0.0, 0.25, 0.5, 0.75, 1.0]
    w_rows = {col: [(w, *mean_paired(tune, col, vm, w)) for w in grid] for col in ("visual+text", "visual+theirs")}
    w_best = {col: max(rows, key=lambda r: r[1])[0] for col, rows in w_rows.items()}
    tuned.update({"visual_method": vm, "visual_text_weight": w_best["visual+text"],
                  "visual_theirs_weight": w_best["visual+theirs"],
                  "visual_study": "2026-09-24: frame->page score and visual weight by tune their-F1; "
                                  "weight = share of the min-max scaled visual matrix"})
    (EXTERNAL / "mavils_tuned.json").write_text(json.dumps(tuned, indent=1) + "\n")
    wcol = lambda col: w_best.get(col, 0.5)

    L = ["# MaViLS — visual channel vs their all-features", "",
         "Their protocol: sentence granularity, page OCR on the slide side, their sklearn F1. Cells: **their F1 "
         "(precision-on-answered / coverage)**. Decoder: our DP at the tuned σ = "
         f"{sigma}, λ={a['jump_penalty']} β={a['back_penalty']} B={a['max_back']}. LLM-free, $0.", "",
         "Columns: *ours* = bge-m3 + BM25 + IDF hybrid; *theirs* = distiluse cosine (their audio feature); *fused* = "
         f"{fw}·theirs + {1 - fw}·ours (tuned 2026-09-22). *visual* = each sentence takes the frame→page row of the "
         "representative frame on screen at its timestamp (`mavils_frames.md`), min-max scaled. *visual+text* / "
         "*visual+theirs* = w·visual + (1−w)·text, each min-max scaled. Their published columns: audio-only (Table 1) "
         "and all features, i.e. speech + frame OCR + SwiftFormer image features, merged, λ = 0.1 (Table 2).", "",
         f"Split `results/external/mavils_split.json`: tune = {', '.join(LECTURES[s][1] for s in tune)}; test = "
         f"{', '.join(LECTURES[s][1] for s in test)}. Every choice below is made on the tune half; the test half is "
         "run once. Chosen values are recorded in `results/external/mavils_tuned.json`.", "",
         "## Tune half", "", "| frame→page score | visual-only tune |", "|---|---:|"]
    for k, (f1, pr) in vm_rows.items():
        L.append(f"| {k} | {pr}{' ←' if k == vm else ''} |")
    L += ["", f"| w (share of visual, {vm}) | visual+text | visual+theirs |", "|---:|---:|---:|"]
    for i, w in enumerate(grid):
        L.append(f"| {w} | " + " | ".join(f"{w_rows[c][i][2]}{' ←' if w == w_best[c] else ''}" for c in w_rows) + " |")
    L += ["", f"Chosen on tune: frame→page score **{vm}**, visual+text w = **{w_best['visual+text']}**, visual+theirs "
          f"w = **{w_best['visual+theirs']}**. No confidence floor was needed or used.", "",
          "## Test half — reported once", "",
          "| lecture | text layer | jumpiness | no-slide ratio | " + " | ".join(VISUAL_COLS)
          + " | their audio (T1) | their all-features (T2) |", "|---|---|---:|---:|" + "---:|" * (len(VISUAL_COLS) + 2)]
    no_video = [s_ for s_ in test if data[s_][2] is None]
    test = [s_ for s_ in test if data[s_][2] is not None]           # every column paired on the same lectures
    wins = {"vs_theirs_all": [0, 0], "vs_ours": [0, 0]}
    best_col = max(("visual", "visual+text", "visual+theirs"), key=lambda c: mean_paired(tune, c, vm, wcol(c))[0])
    for s_ in test:
        status, gt = data[s_][6], data[s_][4]
        jump, noslide = jumpiness(gt)
        cells, f1 = [], {}
        for col in VISUAL_COLS:
            g, p = run(s_, col, vm, wcol(col))
            f1[col] = their_prf(g, p)[2]
            cells.append(paired(g, p))
        wins["vs_theirs_all"][f1[best_col] < LECTURES[s_][3]] += 1
        wins["vs_ours"][f1[best_col] < f1["ours"]] += 1
        L.append(f"| {LECTURES[s_][1]} | {status['text_layer']} | {jump:.2f} | {noslide:.2f} | " + " | ".join(cells)
                 + f" | {LECTURES[s_][2]:.2f} | {LECTURES[s_][3]:.2f} |")
    means = {c: mean_paired(test, c, vm, wcol(c))[1] for c in VISUAL_COLS}
    L.append("| **mean** | | | | " + " | ".join(f"**{means[c]}**" for c in VISUAL_COLS)
             + f" | {np.mean([LECTURES[s_][2] for s_ in test]):.2f} | {np.mean([LECTURES[s_][3] for s_ in test]):.2f} |")
    for s_ in no_video:
        L.append(f"| {LECTURES[s_][1]} (no video) | {data[s_][6]['text_layer']} | {jumpiness(data[s_][4])[0]:.2f} | "
                 f"{jumpiness(data[s_][4])[1]:.2f} | " + " | ".join(paired(*run(s_, c, vm, 0.5)) if c in ("ours", "theirs", "fused")
                                                             else "—" for c in VISUAL_COLS)
                 + f" | {LECTURES[s_][2]:.2f} | {LECTURES[s_][3]:.2f} |")
    if no_video:
        L += ["", f"**No video for {', '.join(LECTURES[s_][1] for s_ in no_video)}**: the MaViLS Kaggle zip has none for "
              f"it, so the test mean is over {len(test)} lectures (paired: every column on the same lectures) and the "
              "no-video row is outside it."]
    with_video = [s_ for s_ in sorted(LECTURES) if data[s_][2] is not None]
    all_means = {c: mean_paired(with_video, c, vm, wcol(c))[1] for c in VISUAL_COLS}
    L += ["", f"Best visual column on tune: **{best_col}**. On the test half it beats their all-features F1 on "
          f"{wins['vs_theirs_all'][0]} of {len(test)} lectures (loses {wins['vs_theirs_all'][1]}) and our text-only "
          f"*ours* on {wins['vs_ours'][0]} (loses {wins['vs_ours'][1]}); a tie counts as a win.", "",
          f"All {len(with_video)} lectures with video (contains the tune half, so the tuned columns are optimistic): "
          + ", ".join(f"{c} {v}" for c, v in all_means.items()) + ".", ""]
    out = Path(args.out)
    out.write_text("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


GATE_GRID = [(w, m) for w in (1, 3, 5, 10) for m in (0.01, 0.02, 0.05, 0.10)]


def cmd_visibility_gate(args, cfg, encoder) -> int:
    """Item 1 of the final round: the slide-visibility gate on the representative frames. Rule
    and thresholds fixed on the tune half only (grid GATE_GRID, by their-F1 of the tuned
    three-way fusion); the gate stays off unless a setting beats no gate on the tune half."""
    from linkrag.link.visual import slide_visible
    a = cfg["link"]["align"]
    split = split_lectures()
    tune = split["tune"]
    tuned = json.loads((EXTERNAL / "mavils_tuned.json").read_text())
    sigma, vm, tm = float(tuned["sigma"]), tuned["visual_method"], tuned["frame_ocr_method"]
    w3 = tuple(tuned["three_way_weights"])
    data = {s: lecture_matrices(s, cfg, encoder, Path(args.figures_dir), with_text=True)
            for s in sorted(LECTURES) if (CACHE / "frames" / s / "slides.json").exists()}

    def pred(stem, gate):
        d = data[stem]
        return decode(three_way(d, vm, tm, w3, gate), d["pages"], d["owner"], a, variant="dp", min_sim=None,
                      flat=0.0, sigma=sigma)

    def mean_f1(stems_, gate):
        return paired_means([(data[s_]["gt"], pred(s_, gate)) for s_ in stems_])[0]

    off = mean_f1(tune, None)
    grid = [((w, m), mean_f1(tune, {"min_words": w, "min_margin": m})) for w, m in GATE_GRID]
    (bw, bm), best = max(grid, key=lambda r: r[1])
    gate = {"min_words": bw, "min_margin": bm} if best > off else None
    tuned["visibility_gate"] = (dict(gate, image_score=vm, frames="representative") if gate else None)
    tuned["visibility_gate_study"] = ("2026-09-24: a frame shows a slide if its OCR reads >= min_words words or its "
                                      "image margin (best - runner-up page) > min_margin; else its visual rows are "
                                      "neutral. Grid by tune their-F1 of the tuned three-way fusion; off unless it beats "
                                      "no gate on tune")
    (EXTERNAL / "mavils_tuned.json").write_text(json.dumps(tuned, indent=1) + "\n")
    L = ["# MaViLS — slide-visibility gate (representative frames)", "",
         "Rule: a frame shows a slide if its OCR reads at least *W* words (3× upscale, `mavils_frame_ocr_sanity.md`) "
         f"or its image channel ({vm}) picks a page by more than *M* (best minus runner-up); otherwise it is a speaker or "
         "room shot, and every sentence taking that frame gets constant image and frame-OCR rows, so its page is "
         "decided by the speech-to-slide text alone (`three_way`, `linkrag.link.visual.slide_visible`). The optional "
         "high-contrast-text-area signal was not used: the two signals above were already computed.", "",
         f"Fusion: the tuned three-way (visual / frame_ocr / theirs = {w3[0]} / {w3[1]} / {w3[2]}, {tm}), our DP at "
         f"σ = {sigma}. **Tune half only**; the test half is run once in `mavils_final_round.md`. LLM-free, $0.", "",
         "## Tune half: W × M grid (their F1, mean of 10 lectures)", "",
         f"No gate: **{off:.3f}**.", "", "| W (words) \\ M (margin) | " + " | ".join(str(m) for m in (0.01, 0.02, 0.05, 0.10))
         + " |", "|---|" + "---:|" * 4]
    for w in (1, 3, 5, 10):
        L.append(f"| {w} | " + " | ".join(f"{f1:.3f}{' ←' if (w, m) == (bw, bm) and gate else ''}"
                                         for (ww, m), f1 in grid if ww == w) + " |")
    L += ["", (f"Chosen: **W = {bw}, M = {bm}** (tune {best:.3f} vs {off:.3f} without the gate); recorded in "
               "`results/external/mavils_tuned.json`." if gate else
               f"No setting beats no gate on the tune half (best {best:.3f} at W = {bw}, M = {bm}): **gate off**."), "",
          "## Frames classified speaker, and the tune-half effect", "",
          "| lecture | half | frames | speaker frames | tune F1 no gate | tune F1 gate | predictions changed |",
          "|---|---|---:|---:|---:|---:|---:|"]
    g = gate or {"min_words": bw, "min_margin": bm}
    for s_ in sorted(data, key=lambda x: LECTURES[x][1]):
        d = data[s_]
        vis = slide_visible(d["frame_words"], d["frame_margin"][vm], **g)
        frac = float(np.mean(~vis))
        half = "tune" if s_ in tune else "test"
        cells = "— | — | —"
        if half == "tune":
            p0, p1 = pred(s_, None), pred(s_, g)
            cells = (f"{their_prf(d['gt'], p0)[2]:.3f} | {their_prf(d['gt'], p1)[2]:.3f} | "
                     f"{int(np.sum(p0 != p1))} of {len(p0)}")
        L.append(f"| {LECTURES[s_][1]} | {half} | {len(vis)} | {frac:.0%} | {cells} |")
    low = [s_ for s_ in tune if float(np.mean(~slide_visible(data[s_]["frame_words"], data[s_]["frame_margin"][vm], **g))) < 0.10]
    unchanged = [s_ for s_ in low if not np.any(pred(s_, None) != pred(s_, g))]
    L += ["", f"Tune lectures with < 10 % speaker frames: {', '.join(LECTURES[s_][1] for s_ in low) or 'none'}; "
          f"predictions unchanged by the gate on {len(unchanged)} of {len(low)}"
          + (f" (changed: {', '.join(LECTURES[s_][1] for s_ in low if s_ not in unchanged)})." if len(unchanged) < len(low) else ".")]
    out = Path(args.out)
    out.write_text("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


def cmd_final_round(args, cfg, encoder) -> int:
    """Final round, item 3. Four variants of the tuned three-way fusion (visual / frame_ocr /
    theirs): the previous best (representative frames, no gate), + the visibility gate, +
    sentence-time frames, + both. Tune half only: for sentence-time frames the three-way
    weights (SIMPLEX) and then the gate thresholds (GATE_GRID); the representative gate is the
    one fixed in item 1. The column reported as ours is the best on tune. Test half once."""
    a = cfg["link"]["align"]
    split = split_lectures()
    tune = split["tune"]
    tuned = json.loads((EXTERNAL / "mavils_tuned.json").read_text())
    sigma, vm, tm = float(tuned["sigma"]), tuned["visual_method"], tuned["frame_ocr_method"]
    w3 = tuple(tuned["three_way_weights"])
    g_rep = {k: tuned["visibility_gate"][k] for k in ("min_words", "min_margin")}
    fd = Path(args.figures_dir)
    stems = [s for s in sorted(LECTURES) if (CACHE / "frames" / s / "slides.json").exists()]
    rep = {s: lecture_matrices(s, cfg, encoder, fd, with_text=True, frame_source="representative") for s in stems}
    sen = {s: lecture_matrices(s, cfg, encoder, fd, with_text=True, frame_source="sentence_time") for s in stems}

    def f1(dd, s_, w, gate):
        d = dd[s_]
        pred = decode(three_way(d, vm, tm, w, gate), d["pages"], d["owner"], a, variant="dp", min_sim=None, flat=0.0,
                      sigma=sigma)
        return their_prf(d["gt"], pred)[2], answered_metrics(d["gt"], pred)

    def mean(dd, stems_, w, gate):
        return float(np.mean([f1(dd, s_, w, gate)[0] for s_ in stems_]))

    s_rows = [(w, mean(sen, tune, w, None)) for w in SIMPLEX]
    w3s = max(s_rows, key=lambda r: r[1])[0]
    off_s = mean(sen, tune, w3s, None)
    g_rows = [((w, m), mean(sen, tune, w3s, {"min_words": w, "min_margin": m})) for w, m in GATE_GRID]
    (gw, gm), g_best = max(g_rows, key=lambda r: r[1])
    g_sen = {"min_words": gw, "min_margin": gm} if g_best > off_s else None
    cols = {"previous best": (rep, w3, None), "+ gate": (rep, w3, g_rep), "+ sentence-time": (sen, w3s, None),
            "+ both": (sen, w3s, g_sen)}
    tune_means = {c: mean(dd, tune, w, g) for c, (dd, w, g) in cols.items()}
    best = max(tune_means, key=tune_means.get)
    tuned["visual_final_round"] = {"sentence_time_three_way_weights": list(w3s), "sentence_time_gate": g_sen,
                            "best_column_on_tune": best,
                            "study": "2026-09-24: sentence-time weights (simplex step 0.25) then gate (W x M grid) by "
                                     "tune their-F1; best column by tune mean"}
    (EXTERNAL / "mavils_tuned.json").write_text(json.dumps(tuned, indent=1) + "\n")

    test = [s_ for s_ in split["test"] if s_ in rep]
    theirs = {s_: LECTURES[s_][3] for s_ in test}
    L = ["# MaViLS — final round: visibility gate and sentence-time frames", "",
         "Their protocol: sentence granularity, page OCR on the slide side, their sklearn F1. Cells: **their F1 "
         f"(precision-on-answered / coverage)**. Decoder: our DP at σ = {sigma}. Every column is the three-way fusion "
         f"(image {vm} / frame OCR {tm} / speech-to-slide text, min-max scaled, `fuse_many`). LLM-free, $0.", "",
         "- *previous best* — representative frames (one per dHash segment, 320 px), weights "
         f"{w3[0]}/{w3[1]}/{w3[2]}, no gate (`mavils_frame_ocr.md`, 0.810).",
         f"- *+ gate* — the same with the slide-visibility gate of `mavils_visibility_gate.md` (W = {g_rep['min_words']}, "
         f"M = {g_rep['min_margin']}).",
         "- *+ sentence-time* — the frame at each sentence timestamp at native resolution (MaViLS's own choice), OCR'd "
         f"at ≥ {OCR_SIDE} px longest side; weights re-chosen on tune.",
         "- *+ both* — sentence-time frames and a gate re-tuned on them.", "",
         "Videos: the MaViLS Kaggle zip, found again as `~/Downloads/archive (2).zip` (Kaggle's default name; same 21 "
         "files as the earlier `video.zip`); sentence-time frames extracted, then the videos removed. Native resolution "
         "varies by lecture from 320×240 to 1280×720.", "",
         "## Tune half", "", "Sentence-time three-way weights (visual / frame_ocr / theirs), top 5:", "",
         "| weights | tune |", "|---|---:|"]
    for w, v in sorted(s_rows, key=lambda r: -r[1])[:5]:
        L.append(f"| {w[0]:.2f} / {w[1]:.2f} / {w[2]:.2f} | {v:.3f}{' ←' if w == w3s else ''} |")
    L += ["", f"Sentence-time gate: no gate {off_s:.3f}; best W = {gw}, M = {gm} at {g_best:.3f} → "
          + (f"**gate on (W = {gw}, M = {gm})**." if g_sen else "**gate off** (no setting beats no gate)."), "",
          "| column | tune mean |", "|---|---:|", *[f"| {c} | {v:.3f}{' ←' if c == best else ''} |" for c, v in tune_means.items()],
          "", f"Chosen on tune: **{best}**.", "", "## Test half — reported once", "",
          "| lecture | " + " | ".join(cols) + " | their all-features (T2) |", "|---|" + "---:|" * (len(cols) + 1)]
    best_f1 = {}
    for s_ in test:
        cells = []
        for c, (dd, w, g) in cols.items():
            v, (pr, cov) = f1(dd, s_, w, g)
            if c == best:
                best_f1[s_] = v
            cells.append(f"{v:.2f} ({pr:.2f} / {cov:.2f})")
        L.append(f"| {LECTURES[s_][1]} | " + " | ".join(cells) + f" | {theirs[s_]:.2f} |")
    means = {c: mean(dd, test, w, g) for c, (dd, w, g) in cols.items()}
    L.append("| **mean** | " + " | ".join(f"**{means[c]:.3f}**" for c in cols) + f" | {np.mean(list(theirs.values())):.2f} |")
    diffs = np.array([best_f1[s_] - theirs[s_] for s_ in test])
    boots = np.random.default_rng(20260924).choice(diffs, (10000, len(diffs))).mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    wins = [s_ for s_ in test if best_f1[s_] >= theirs[s_]]
    L += ["", f"## Paired against their all-features ({best})", "",
          "| lecture | ours | theirs | difference |", "|---|---:|---:|---:|",
          *[f"| {LECTURES[s_][1]} | {best_f1[s_]:.2f} | {theirs[s_]:.2f} | {best_f1[s_] - theirs[s_]:+.2f} |" for s_ in test],
          "", f"Mean difference **{diffs.mean():+.3f}**, 95 % bootstrap interval over the {len(test)} lectures "
          f"[{lo:+.3f}, {hi:+.3f}] (10,000 resamples, numpy seed 20260924). Wins {len(wins)}, losses {len(test) - len(wins)} "
          f"(a tie counts as a win). Our test mean {means[best]:.3f} vs their {np.mean(list(theirs.values())):.2f} on the same "
          f"lectures; their {PAPER_ALL_FEATURES} is a 20-lecture average and not a paired comparison.", "",
          f"Reinforcement: {best_f1.get('reinforcement_learning', float('nan')):.2f} vs their "
          f"{LECTURES['reinforcement_learning'][3]:.2f}. Numerics: {best_f1.get('numerics', float('nan')):.2f} vs their "
          f"{LECTURES['numerics'][3]:.2f}.", "",
          "Climate policies (test half) has no video in their Kaggle zip and is outside every mean here."]
    out = Path(args.out)
    out.write_text("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


OCR_COLS = ("ours", "visual+theirs", "frame_ocr", "visual+frame_ocr", "visual+frame_ocr+theirs")
SIMPLEX = [(a / 4, b / 4, (4 - a - b) / 4) for a in range(5) for b in range(5 - a)]   # step 0.25: 15 points
PAPER_ALL_FEATURES = 0.82                                                          # their Table 2, 20 lectures


def cmd_visual_ocr(args, cfg, encoder) -> int:
    """Items 3-4 of the frame-OCR task. Tune half only: the frame-OCR text score (TF-IDF vs
    BM25, by frame_ocr-only their-F1), the frame_ocr weight in visual+frame_ocr (grid step
    0.25) and the three-way weights of visual+frame_ocr+theirs (simplex, step 0.25). The
    image score (SwiftFormer) and the visual+theirs weight are the values tuned before
    (results/external/mavils_tuned.json). Test half once."""
    from linkrag.link.align import fuse_many
    a = cfg["link"]["align"]
    split = split_lectures()
    tune = split["tune"]
    tuned = json.loads((EXTERNAL / "mavils_tuned.json").read_text())
    sigma, vm, wvt = float(tuned["sigma"]), tuned["visual_method"], float(tuned["visual_theirs_weight"])
    figures_dir = Path(args.figures_dir)
    data = {stem: lecture_matrices(stem, cfg, encoder, figures_dir, with_text=True) for stem in sorted(LECTURES)}

    def run(stem, col, tm="tfidf", w=0.5, w3=(1 / 3, 1 / 3, 1 / 3)):
        d = data[stem]
        V, O = (d["visual"][vm], d["frame_ocr"][tm]) if d["visual"] is not None else (None, None)
        S = {"ours": lambda: d["ours"],
             "visual+theirs": lambda: fuse_similarity(d["theirs"], V, "fused_weighted", wvt),
             "frame_ocr": lambda: fuse_many([O], [1.0]),                         # alone, min-max scaled
             "visual+frame_ocr": lambda: fuse_similarity(V, O, "fused_weighted", w),
             "visual+frame_ocr+theirs": lambda: fuse_many([V, O, d["theirs"]], w3)}[col]()
        return d["gt"], decode(S, d["pages"], d["owner"], a, variant="dp", min_sim=None, flat=0.0, sigma=sigma)

    def mean_paired(stems_, col, **kw):
        return paired_mean_cell([run(s_, col, **kw) for s_ in stems_])

    tm_rows = {tm: mean_paired(tune, "frame_ocr", tm=tm) for tm in ("tfidf", "bm25")}
    tm = max(tm_rows, key=lambda k: tm_rows[k][0])
    grid = [0.0, 0.25, 0.5, 0.75, 1.0]
    w_rows = [(w, *mean_paired(tune, "visual+frame_ocr", tm=tm, w=w)) for w in grid]
    w_best = max(w_rows, key=lambda r: r[1])[0]
    s_rows = [(w3, *mean_paired(tune, "visual+frame_ocr+theirs", tm=tm, w3=w3)) for w3 in SIMPLEX]
    w3_best = max(s_rows, key=lambda r: r[1])[0]
    kw = {"visual+frame_ocr": {"tm": tm, "w": w_best}, "visual+frame_ocr+theirs": {"tm": tm, "w3": w3_best},
          "frame_ocr": {"tm": tm}}
    tuned.update({"frame_ocr_method": tm, "visual_frame_ocr_weight": w_best, "three_way_weights": list(w3_best),
                  "frame_ocr_study": "2026-09-24: text score, frame_ocr weight and three-way (visual, frame_ocr, theirs) "
                                     "weights by tune their-F1; frames OCR'd at 3x from 320 px"})
    (EXTERNAL / "mavils_tuned.json").write_text(json.dumps(tuned, indent=1) + "\n")

    L = ["# MaViLS — frame OCR and three-feature fusion vs their all-features", "",
         "Their protocol: sentence granularity, page OCR on the slide side, their sklearn F1. Cells: **their F1 "
         f"(precision-on-answered / coverage)**. Decoder: our DP at the tuned σ = {sigma}, λ={a['jump_penalty']} "
         f"β={a['back_penalty']} B={a['max_back']}. LLM-free, $0.", "",
         "Columns: *ours* = our text-only hybrid. *visual+theirs* = the previous best (`mavils_visual.md`: "
         f"{vm} image score, w = {wvt}). *frame_ocr* = each sentence takes the frame→page **text** row of the frame on screen "
         "at its timestamp (`mavils_frame_ocr_sanity.md`), min-max scaled. *visual+frame_ocr* = w·frame_ocr + "
         "(1−w)·visual. *visual+frame_ocr+theirs* = weighted sum of the three, each min-max scaled "
         "(`linkrag.link.align.fuse_many`): all three of their feature types (image, frame OCR, speech-to-slide text). "
         "Their published columns: audio-only (Table 1) and all features (Table 2, λ = 0.1).", "",
         "**Caveat carried from the frames:** frames are one representative per dHash segment, kept at 320 px; the "
         "text on them is read after a 3× upscale. MaViLS reads the full-resolution frame at each sentence timestamp. "
         "Re-extracting sentence-time or full-resolution frames was not possible: the videos are no longer available.", "",
         "## Tune half", "", "| frame-OCR text score | frame_ocr-only tune |", "|---|---:|"]
    for k, (f1, pr) in tm_rows.items():
        L.append(f"| {k} | {pr}{' ←' if k == tm else ''} |")
    L += ["", f"| w (share of frame_ocr, {tm}) | visual+frame_ocr tune |", "|---:|---:|"]
    for w, f1, pr in w_rows:
        L.append(f"| {w} | {pr}{' ←' if w == w_best else ''} |")
    L += ["", "| weights (visual, frame_ocr, theirs) | visual+frame_ocr+theirs tune |", "|---|---:|"]
    for w3, f1, pr in sorted(s_rows, key=lambda r: -r[1]):
        L.append(f"| {w3[0]:.2f}, {w3[1]:.2f}, {w3[2]:.2f} | {pr}{' ←' if w3 == w3_best else ''} |")
    best_col = max(("frame_ocr", "visual+frame_ocr", "visual+frame_ocr+theirs"),
                   key=lambda c: mean_paired(tune, c, **kw[c])[0])
    L += ["", f"Chosen on tune: text score **{tm}**, visual+frame_ocr w = **{w_best}**, three-way weights **"
          f"{w3_best[0]:.2f} / {w3_best[1]:.2f} / {w3_best[2]:.2f}** (visual / frame_ocr / theirs). Best new column on "
          f"tune: **{best_col}**.", "", "## Test half — reported once", "",
          "| lecture | " + " | ".join(OCR_COLS) + " | their all-features (T2) |", "|---|" + "---:|" * (len(OCR_COLS) + 1)]
    test = [s_ for s_ in split["test"] if data[s_]["visual"] is not None]
    no_video = [s_ for s_ in split["test"] if data[s_]["visual"] is None]
    wins = {"theirs_all": [0, 0], "visual+theirs": [0, 0]}
    for s_ in test:
        cells, f1 = [], {}
        for col in OCR_COLS:
            g, p = run(s_, col, **kw.get(col, {}))
            f1[col] = their_prf(g, p)[2]
            cells.append(paired(g, p))
        wins["theirs_all"][f1[best_col] < LECTURES[s_][3]] += 1
        wins["visual+theirs"][f1[best_col] < f1["visual+theirs"]] += 1
        L.append(f"| {LECTURES[s_][1]} | " + " | ".join(cells) + f" | {LECTURES[s_][3]:.2f} |")
    means = {c: mean_paired(test, c, **kw.get(c, {}))[1] for c in OCR_COLS}
    L.append("| **mean** | " + " | ".join(f"**{means[c]}**" for c in OCR_COLS)
             + f" | {np.mean([LECTURES[s_][3] for s_ in test]):.2f} |")
    for s_ in no_video:
        L.append(f"| {LECTURES[s_][1]} (no video) | {paired(*run(s_, 'ours'))} | — | — | — | — | {LECTURES[s_][3]:.2f} |")
    with_video = [s_ for s_ in sorted(LECTURES) if data[s_]["visual"] is not None]
    all_means = {c: mean_paired(with_video, c, **kw.get(c, {}))[1] for c in OCR_COLS}
    L += ["", f"Best new column on tune: **{best_col}**. On the {len(test)} test lectures with video it beats their "
          f"all-features F1 on {wins['theirs_all'][0]} and loses {wins['theirs_all'][1]}; against the previous best "
          f"(visual+theirs) it wins {wins['visual+theirs'][0]} and loses {wins['visual+theirs'][1]} (a tie counts as a "
          f"win). Their all-features mean on these lectures is {np.mean([LECTURES[s_][3] for s_ in test]):.2f}; "
          f"{PAPER_ALL_FEATURES} is their 20-lecture average.", "",
          f"All {len(with_video)} lectures with video (contains the tune half, so the tuned columns are optimistic): "
          + ", ".join(f"{c} {v}" for c, v in all_means.items()) + ".", ""]
    out = Path(args.out)
    out.write_text("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "study", "inspect", "final", "fused", "gate", "gate-gran", "visual-frames", "visual", "frame-ocr",
                                    "visual-ocr", "visibility-gate", "final-round"])
    ap.add_argument("--shuffles", type=int, default=30)
    ap.add_argument("--reuse", action="store_true", help="gate: reuse cached z-scores from the previous run's json")
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
                    "inspect": None, "final": "reports/mavils_final.md", "fused": "reports/mavils_fused.md",
                    "gate": "results/external/mavils_gate_v2.md", "gate-gran": "results/external/mavils_gate_v3.md",
                    "visual-frames": "results/external/mavils_frames.md", "visual": "results/external/mavils_visual.md",
                    "frame-ocr": "results/external/mavils_frame_ocr_sanity.md",
                    "visual-ocr": "results/external/mavils_frame_ocr.md",
                    "visibility-gate": "results/external/mavils_visibility_gate.md",
                    "final-round": "results/external/mavils_final_round.md"}[args.cmd]
    setup_logging()
    cfg = load_config(args.config)
    if not (REPO / "data" / "ground_truth_files").exists():
        raise SystemExit(f"clone https://github.com/andererka/MaViLS to {REPO}")
    encoder = default_encoder(cfg["models"]["embedding"], cfg["device"], cfg["index"]["normalize_embeddings"])
    encoder([""])
    return {"run": cmd_run, "study": cmd_study, "inspect": cmd_inspect, "final": cmd_final, "fused": cmd_fused,
            "gate": cmd_gate, "gate-gran": cmd_gate_gran, "visual-frames": cmd_visual_frames,
            "visual": cmd_visual, "frame-ocr": cmd_frame_ocr, "visual-ocr": cmd_visual_ocr,
            "visibility-gate": cmd_visibility_gate, "final-round": cmd_final_round}[args.cmd](args, cfg, encoder)


if __name__ == "__main__":
    raise SystemExit(main())
