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
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("sentence-transformers/distiluse-base-multilingual-cased", device=cfg["device"])
    audio, owner = sentence_units(df, stem)
    slides, _ = slide_units(stem, "ocr", figures_dir)
    a = model.encode([u.content for u in audio], convert_to_numpy=True, normalize_embeddings=True)
    b = model.encode([u.content for u in slides], convert_to_numpy=True, normalize_embeddings=True)
    S = (a @ b.T).astype(np.float64)
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
        return float(np.mean([their_prf(data[s][2], fn(s))[2] for s in stems_]))

    def mean_paired(stems_, fn):
        f1s, prs, covs = [], [], []
        for s_ in stems_:
            gt = data[s_][2]
            pred = fn(s_)
            f1s.append(their_prf(gt, pred)[2])
            pr, cov = answered_metrics(gt, pred)
            prs.append(pr); covs.append(cov)
        return f"{np.mean(f1s):.3f} ({np.mean(prs):.3f} / {np.mean(covs):.2f})"

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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "study", "inspect", "final"])
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
                    "inspect": None, "final": "reports/mavils_final.md"}[args.cmd]
    setup_logging()
    cfg = load_config(args.config)
    if not (REPO / "data" / "ground_truth_files").exists():
        raise SystemExit(f"clone https://github.com/andererka/MaViLS to {REPO}")
    encoder = default_encoder(cfg["models"]["embedding"], cfg["device"], cfg["index"]["normalize_embeddings"])
    encoder([""])
    return {"run": cmd_run, "study": cmd_study, "inspect": cmd_inspect, "final": cmd_final}[args.cmd](args, cfg, encoder)


if __name__ == "__main__":
    raise SystemExit(main())
