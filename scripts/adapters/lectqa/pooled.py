"""Pooled-corpus retrieval: every prepared video in ONE index, so the video is not given.

T1's setting hands the retriever the right video, and every transcript fits in the answerer's
context (DESIGN.md finding 13), so retrieval cannot show a benefit there. Pooled, the corpus is
~95 transcripts and the retriever has to find the video before it can find the interval.
No answerer, no LLM call, no re-embedding: the stored per-video vectors are stacked. $0.
"""

from __future__ import annotations

import collections
import hashlib
import json
import logging
import re
import subprocess
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from linkrag.core import Link
from linkrag.index import Index, default_encoder, embeddable_text, memoised, tokenize
from linkrag.link.align import load_links
from linkrag.link.graph import build_graph
from linkrag.retrieve.iterative import retrieve_pool
from linkrag.retrieve.linkrag import RetrievedUnit
from linkrag.retrieve.rerank import rerank

from lectqa.common import LEVELS, PROCESSED, gold_interval, iou, load_qa, temporal_links
from lectqa.localisation import lectqa_link
from lectqa.open_ended import paired_cell

KS = (1, 3, 8)
# (label, retrieval mode, rerank method, reranker sees the link graph, overrides of the config's reranker).
# The last two take the objective apart: the recorded modality gate (alpha = 0 on concentrated pools),
# and links alone (coverage and redundancy terms off; beta stays at its config value).
# "raw scores" is baseline mode as the repo runs it: RRF scores (~0.03) against alpha 0.3 and gamma 0.4, so
# relevance barely counts. "reranker, no links" is the control for the links: the same rank-normalised
# seeds as the full system (hops 0: nothing is followed), graph hidden from the reranker.
CELLS = (("plain top-8", "baseline", "none", False, {}),
         ("reranker on raw scores, no links", "baseline", "complementarity", False, {}),
         ("reranker, no links", "linkrag", "complementarity", False, {"hops": 0}),
         ("links + reranker", "linkrag", "complementarity", True, {}),
         ("links + reranker, gated", "linkrag", "complementarity", True, {"modality_gate": True}),
         ("links only", "linkrag", "complementarity", True, {"alpha": 0.0, "gamma": 0.0}))
# (label, per-video index, link types followed): the two corpora of lectqa_frameslides.md
VARIANTS = (("frames", "index", ("audio_slide",)), ("frame-derived slides", "index_frameslides", None))
# wording that only means something once the video is known
DEICTIC = re.compile(r"\b(the|this) (video|lecture|speaker|narrator|presenter|lecturer|instructor|clip)\b", re.I)


def pool(parts: list[tuple[Index, list[Link]]]) -> tuple[Index, object]:
    """One index and one graph over several per-video indexes; vectors are stacked, not re-embedded."""
    units = [u for index, _ in parts for u in index.units]
    index = Index(units=units, vectors=np.vstack([index.vectors for index, _ in parts]),
                  bm25=BM25Okapi([tokenize(embeddable_text(u)) for u in units]),
                  embedding_model=parts[0][0].embedding_model, normalize=parts[0][0].normalize)
    return index, build_graph(units, [link for _, links in parts for link in links])


def hits(picked: list[RetrievedUnit], vid: str, gold: tuple[float, float] | None) -> dict[str, float]:
    """video@k: a top-k unit comes from the question's video. interval@k: it also overlaps the
    gold interval (0 when the question has no usable interval; the caller leaves those out)."""
    same = [r.unit.id.split(":")[0] == vid for r in picked]
    over = [s and gold is not None and any(iou(a, b, *gold) > 0 for a, b in
                                           (r.unit.metadata.get("intervals") or [(r.unit.location.start_s, r.unit.location.end_s)]))
            for s, r in zip(same, picked)]
    return {**{f"video@{k}": float(any(same[:k])) for k in KS}, **{f"interval@{k}": float(any(over[:k])) for k in KS}}


def pooled(ids: list[str], cfg: dict, out: Path, only: str | None = None) -> int:
    """`only`: semicolon-separated method labels to (re)compute; the other rows are kept from the stored jsonl."""
    cells = [c for c in CELLS if not only or c[0] in only.split(";")]
    kept = ([r for r in map(json.loads, out.with_suffix(".jsonl").read_text().splitlines()) if r["cell"] not in {c[0] for c in cells}]
            if only else [])
    qa = load_qa()
    encoder = memoised(default_encoder(cfg["models"]["embedding"], cfg["device"], cfg["index"]["normalize_embeddings"]))
    rcfg = cfg["retrieve"]["rerank"]
    log = logging.getLogger("linkrag")
    level, rows, sizes = log.level, [], {}
    log.setLevel(logging.WARNING)                      # ~70k stage-timer lines otherwise
    for variant, index_name, link_types in VARIANTS:
        single = {}
        for vid in ids:
            d = PROCESSED / vid / index_name
            if d.exists():
                index = Index.load(d)
                links = (load_links(d / "links.jsonl", gated=cfg["link"]["relatedness"]["enabled"])
                         if (d / "links.jsonl").exists() else temporal_links(list(index.units)))
                single[vid] = (index, links)
        pool_index, pool_graph = pool(list(single.values()))
        sizes[variant] = {"videos": len(single), "units": len(pool_index), "links": pool_graph.number_of_edges(),
                          "by_modality": dict(collections.Counter(u.modality for u in pool_index.units))}
        for n, (vid, (index, links)) in enumerate(single.items(), 1):
            graph = build_graph(list(index.units), links)
            for qi, q in enumerate(qa.get(vid, [])):
                gold, _why = gold_interval(vid, q)
                for setting, ix, g in (("single video", index, graph), ("pooled", pool_index, pool_graph)):
                    for cell, mode, method, linked, over in cells:
                        over = dict(over)
                        link = {**lectqa_link(link_types), **({"hops": over.pop("hops")} if "hops" in over else {})}
                        cand, _ = retrieve_pool(mode, q["question"], ix, g, encoder=encoder, cfg=cfg, complete=None,
                                                pool=20, link=link)
                        picked = rerank(cand, 8, method=method, index=ix, graph=g if linked else None,
                                        beta=rcfg["beta"], question=q["question"], device=cfg["device"],
                                        **{"alpha": rcfg["alpha"], "gamma": rcfg["gamma"], **over})
                        rows.append({"variant": variant, "setting": setting, "cell": cell, "vid": vid, "qi": qi,
                                     "kind": q["kind"], "level": q["level"], "gold": gold is not None,
                                     "deictic": bool(DEICTIC.search(q["question"])), **hits(picked, vid, gold)})
            print(f"{variant}: {n}/{len(single)} videos", flush=True)
    log.setLevel(level)
    order = {(v[0], st_, c[0]): i for i, (v, st_, c) in enumerate((v, st_, c) for v in VARIANTS for st_ in ("single video", "pooled") for c in CELLS)}
    rows = sorted(kept + rows, key=lambda r: (order[(r["variant"], r["setting"], r["cell"])], int(r["vid"].split("_")[1]), r["qi"]))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    out.write_text("\n".join(report(rows, sizes, cfg)) + "\n")
    print(out.read_text())
    return 0


def report(rows: list[dict], sizes: dict, cfg: dict) -> list[str]:
    import statistics as st
    pct = lambda xs, k: f"{st.mean(r[k] for r in xs):.1%}" if xs else "—"
    pick = lambda **kw: [r for r in rows if all(r[k] == v for k, v in kw.items())]
    key = lambda r: (r["vid"], r["qi"])

    def diff(a: dict, b: dict, metric: str, gold_only: bool) -> str:
        """Mean paired difference a − b in percentage points, 95 % bootstrap CI over questions."""
        xa = {key(r): r[metric] for r in pick(**a) if r["gold"] or not gold_only}
        xb = {key(r): r[metric] for r in pick(**b) if r["gold"] or not gold_only}
        return paired_cell([100 * (xa[k] - xb[k]) for k in xa if k in xb])

    one = pick(variant="frames", setting="pooled", cell="plain top-8")
    n_all, n_gold = len(one), sum(r["gold"] for r in one)
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True).stdout.strip())
    cfg_hash = hashlib.sha256(json.dumps({k: cfg[k] for k in ("retrieve", "link", "index")}, sort_keys=True, default=str).encode()).hexdigest()[:16]
    s = sizes["frames"]
    L = ["# LectQA-Vid — pooled corpus: retrieval when the video is not given", "",
         "**Exploratory. This setting is not in DESIGN.md's priority order and has not been agreed with the mentor; "
         "do not cite it until it has.** It stays on T1's dataset (dataset policy), but it is not T1's task.", "",
         "## What changed", "",
         f"T1 retrieves inside the one video a question belongs to. Here all {s['videos']} prepared videos are one corpus "
         f"({s['units']} units) and the question is asked without its video. The per-video rows are computed in the same "
         "pass on the same questions, so every difference below is paired. No answerer, no LLM call, no re-embedding "
         "(the stored per-video vectors are stacked): **$0, deterministic, one run.**", "",
         "## Caveats first", "",
         "- **The questions were written for one video.** "
         f"{sum(r['deictic'] for r in one)} of {n_all} name \"the video\", \"the speaker\" or similar, which identifies nothing in a "
         "pooled corpus; such questions bound video@k from above for every method alike. The split is in the last table.",
         f"- **Interval hits are scored on the {n_gold} of {n_all} questions with a usable gold interval** (known issue (h)); "
         "video hits on all of them.",
         "- **Retrieval only.** A hit says a unit was retrieved, not that an answer would be right. Answer quality needs LLM calls.",
         "- **The iterative modes are not run** (one LLM call per question).",
         "- **`linkrag` with no reranker equals plain top-8 by construction** (seed normalisation, DESIGN.md *Design changes*), "
         "so links are tested only through the reranker: *links + reranker* against *reranker, no links*.",
         "- **\"reranker, no links\" hides the graph from the reranker.** `lectqa_frameslides.md` and `compare_retrieval.py` pass "
         "it the graph in baseline mode too, so their `baseline / complementarity` rows are not this cell.",
         "- **The @1 columns barely differ between methods by construction:** the reranker's first pick is the top-scored "
         "unit, and a method can differ at @1 only where two units tie on score and the tie is broken differently. Only @3 "
         "and @8 separate the methods (measurement rule 1: inspected, not a wiring fault).",
         "- **The reranker is not scale-free.** In baseline mode it sees raw RRF scores (about 0.03) next to alpha 0.3 and "
         "gamma 0.4, so relevance barely counts: that is the *raw scores* row, and it is how `baseline / complementarity` "
         "runs everywhere in the repo. The control for the links is *reranker, no links*: the full system's own "
         "rank-normalised scores, nothing followed, graph hidden. That row was added after the first full run showed the "
         "raw-score row confounded links with score scale; it is a correction of the comparison, not a tuned setting.",
         "- **Nothing was tuned.** Config defaults; pool 20, k = 8, one hop, as in `lectqa_frameslides.md`. The method list "
         "was fixed after a 4-video smoke run of the code (120 questions) and before the full run. *links + reranker* "
         "is the system's default and the primary row. *gated* is the recorded modality gate (default off). *links only* "
         "switches the coverage and redundancy terms off (alpha = gamma = 0, not tuned values) to show the links alone.",
         "- Frames are OCR'd with tesseract; many videos are animated explainers, not slide talks (finding 12).", "",
         "## Corpora", "", "| corpus | videos | units | by modality | links in the graph |", "|---|---:|---:|---|---:|"]
    for variant, _i, _l in VARIANTS:
        z = sizes[variant]
        L.append(f"| {variant} | {z['videos']} | {z['units']} | " + ", ".join(f"{k} {v}" for k, v in sorted(z["by_modality"].items()))
                 + f" | {z['links']} |")
    L += ["", "*frames* = transcript segments + one OCR unit per changed frame, joined by temporal co-occurrence. "
          "*frame-derived slides* = the same transcript segments + slide page and figure units recovered from the frames, "
          "joined by the linking layer (`link_corpus`).", "",
          "## Results, all difficulties", "",
          "video@k: a top-k unit comes from the question's video. interval@k: it also overlaps the gold interval.", "",
          "| corpus | setting | method | video@1 | video@3 | video@8 | interval@1 | interval@3 | interval@8 |",
          "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for variant, _i, _l in VARIANTS:
        for setting in ("single video", "pooled"):
            for cell, *_ in CELLS:
                xs = pick(variant=variant, setting=setting, cell=cell)
                g = [r for r in xs if r["gold"]]
                vid_cells = ["given"] * 3 if setting == "single video" else [pct(xs, f"video@{k}") for k in KS]
                L.append(f"| {variant} | {setting} | {cell} | " + " | ".join(vid_cells + [pct(g, f"interval@{k}") for k in KS]) + " |")
    L += ["", f"n = {n_all} questions for video@k, {n_gold} for interval@k, in every row.", "",
          "## Paired differences, pooled setting (percentage points, 95 % bootstrap CI over questions)", "",
          "10,000 resamples, numpy seed 20260923, as in `lectqa_open_modes.md`.", "",
          "| difference | corpus | video@1 | video@8 | interval@1 | interval@3 | interval@8 |", "|---|---|---:|---:|---:|---:|---:|"]
    metrics = (("video@1", False), ("video@8", False), ("interval@1", True), ("interval@3", True), ("interval@8", True))
    for variant, _i, _l in VARIANTS:
        base = {"variant": variant, "setting": "pooled"}
        for name, a, b in (("links + reranker − reranker, no links (what the links add)", "links + reranker", "reranker, no links"),
                           ("reranker, no links − plain top-8 (what set-level reranking adds)", "reranker, no links", "plain top-8"),
                           ("reranker on raw scores, no links − plain top-8", "reranker on raw scores, no links", "plain top-8"),
                           ("links + reranker − plain top-8 (the full system, primary)", "links + reranker", "plain top-8"),
                           ("links + reranker, gated − plain top-8", "links + reranker, gated", "plain top-8"),
                           ("links only − plain top-8 (link-following alone)", "links only", "plain top-8")):
            L.append(f"| {name} | {variant} | " + " | ".join(diff({**base, "cell": a}, {**base, "cell": b}, m, g) for m, g in metrics) + " |")
    plain = {"setting": "pooled", "cell": "plain top-8"}
    L.append("| frame-derived slides − frames, plain top-8 | both | "
             + " | ".join(diff({**plain, "variant": "frame-derived slides"}, {**plain, "variant": "frames"}, m, g) for m, g in metrics) + " |")
    L += ["", "## What not knowing the video costs (frames corpus, plain top-8)", "",
          "| difference | interval@1 | interval@3 | interval@8 |", "|---|---:|---:|---:|",
          "| pooled − single video | " + " | ".join(
              diff({"variant": "frames", "setting": "pooled", "cell": "plain top-8"},
                   {"variant": "frames", "setting": "single video", "cell": "plain top-8"}, f"interval@{k}", True) for k in KS) + " |",
          "", "## Per difficulty, pooled, frames corpus", "",
          "| level | n (video) | n (interval) | method | video@1 | video@8 | interval@1 | interval@8 |", "|---|---:|---:|---|---:|---:|---:|---:|"]
    for lv in LEVELS:
        for cell, *_ in CELLS:
            xs = pick(variant="frames", setting="pooled", cell=cell, level=lv)
            g = [r for r in xs if r["gold"]]
            L.append(f"| {lv} | {len(xs)} | {len(g)} | {cell} | {pct(xs, 'video@1')} | {pct(xs, 'video@8')} | {pct(g, 'interval@1')} | {pct(g, 'interval@8')} |")
    L += ["", "## Question wording (pooled, frames corpus, plain top-8)", "",
          "| question names \"the video / the speaker / …\" | n | video@1 | video@8 |", "|---|---:|---:|---:|"]
    for flag, name in ((True, "yes"), (False, "no")):
        xs = [r for r in one if r["deictic"] == flag]
        L.append(f"| {name} | {len(xs)} | {pct(xs, 'video@1')} | {pct(xs, 'video@8')} |")
    L += ["", "## Reading", "",
          "- **Pooling makes retrieval matter.** Finding the interval is far harder once the video is not given (the "
          "*pooled − single video* row), so this setting can separate retrieval methods where T1's cannot.",
          "- **Plain top-8 is the best row.** No component of the system improves localisation here; each interval of the "
          "*− plain top-8* differences is at or below zero.",
          "- **The links add nothing measurable.** Against the proper control (*reranker, no links*) the full system is "
          "level or slightly lower, and *links only* is level with plain top-8.",
          "- **Most of the reranker's cost in baseline mode is a score-scale artefact.** Compare the two *no links* rows: "
          "the same objective on rank-normalised scores loses about a third as much as on raw RRF scores. Every "
          "`baseline / complementarity` number in the repo runs on raw scores.",
          "- **The metric cannot reward what the system is for.** A gold interval is one stretch of speech. A slide, a figure or "
          "a linked neighbour that completes the answer takes a slot and can only lower interval@k. Whether such evidence "
          "improves answers is an answer-quality question and needs LLM calls.",
          "", "## Provenance", "",
          f"- Dataset: LectQA-Vid (T1), {s['videos']} of 100 videos, MCQ and open-ended questions together (question text only, no options).",
          f"- Embedder `{cfg['models']['embedding']}`; retrieval RRF of dense + BM25, candidates {cfg['retrieve']['candidates']}; "
          f"reranker alpha {cfg['retrieve']['rerank']['alpha']}, beta {cfg['retrieve']['rerank']['beta']}, gamma {cfg['retrieve']['rerank']['gamma']}.",
          f"- Config hash (retrieve, link, index) `{cfg_hash}`; commit `{commit}`{' with uncommitted changes' if dirty else ''}.",
          "- Rows: `lectqa_pooled.jsonl`. Reproduce: `python scripts/adapters/lectqa_vid.py pooled --max-cost 0`."]
    return L
