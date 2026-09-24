"""Temporal localisation (hit@k, IoU against the gold interval) and the v2 segmentation ablation."""

from __future__ import annotations

import collections
import json
from pathlib import Path


from linkrag.core import EvidenceUnit, Location
from linkrag.costs import CacheMiss, cached_completer, price_for, record_run
from linkrag.generate.answer import http_completer
from linkrag.index import Index, build_index, default_encoder
from linkrag.link.align import load_links
from linkrag.link.graph import build_graph
from linkrag.retrieve.baseline import retrieve_scored
from linkrag.retrieve.linkrag import RetrievedUnit, retrieve_linkrag

from lectqa.common import LEVELS, PROCESSED, THEIRS, gold_interval, iou, load_qa, temporal_links


def retrieve_pool(mode, q, index, graph, *, encoder, cfg, complete, pool, link_types=("audio_slide",)):
    link_types = list(link_types) if link_types else None
    lcfg, icfg = cfg["retrieve"]["linkrag"], cfg["retrieve"]["iterative"]
    common = dict(candidates=cfg["retrieve"]["candidates"], rrf_k=cfg["retrieve"]["rrf_k"])
    if mode == "baseline":
        return [RetrievedUnit(unit=u, score=s, origin="seed")
                for u, s in retrieve_scored(q, index, encoder=encoder, top_k=pool, **common)]
    if mode == "linkrag":
        return retrieve_linkrag(q, index, graph, encoder=encoder, mode="linkrag", k_seed=lcfg["k_seed"], k_final=pool,
                                hops=1, link_types=link_types, min_link_score=0.0, decay=lcfg["decay"],
                                normalise_seeds=True, expansion="additive", **common)
    from linkrag.retrieve.iterative import retrieve_iterative, retrieve_linkrag_iter
    if mode == "iterative":
        return retrieve_iterative(q, index, encoder=encoder, complete=complete, rounds=icfg["rounds"],
                                  k_per_round=icfg["k_per_round"], k_final=pool, **common).units
    out, _ = retrieve_linkrag_iter(q, index, graph, encoder=encoder, complete=complete, rounds=icfg["rounds"],
                                   k_seed=lcfg["k_seed"], k_final=pool, hops=1, link_types=link_types,
                                   min_link_score=0.0, decay=lcfg["decay"], normalise_seeds=True,
                                   expansion="additive", **common)
    return out


def localise(ids: list[str], cfg: dict, dcfg: dict, *, index_name: str = "index", label: str = "",
             cache_only: bool = True, max_cost: float = 1.0,
             modes: tuple[str, ...] = ("baseline", "linkrag", "iterative", "linkrag_iter"),
             link_types: tuple[str, ...] | None = ("audio_slide",)) -> tuple[list[dict], dict]:
    """Temporal localisation: does a retrieved unit overlap the gold interval? No answerer.
    The iterative modes need one LLM call per question for the follow-up query; with
    `cache_only` (default) those cells are filled only where that call was already
    paid for, and are marked `not run` otherwise -- no new spend."""
    from linkrag.retrieve.rerank import rerank
    qa = load_qa()
    llm = cfg["models"]["llm"]
    complete = cached_completer(http_completer(llm), PROCESSED / "llm_cache" / str(llm.get("model")),
                                cache_only=cache_only, max_cost_usd=max_cost,
                                price=price_for(str(llm.get("model")), cfg["models"].get("pricing")))
    skipped = 0
    encoder = default_encoder(cfg["models"]["embedding"], cfg["device"], cfg["index"]["normalize_embeddings"])
    encoder([""])
    rcfg = cfg["retrieve"]["rerank"]
    ks, pool = (1, 3, 8), 20
    rows = []
    excluded: collections.Counter = collections.Counter()
    for vid in ids:
        index_dir = PROCESSED / vid / index_name
        if not index_dir.exists():
            continue
        index = Index.load(index_dir)
        units = list(index.units)
        links_file = index_dir / "links.jsonl"            # an index with its own linking layer
        graph = build_graph(units, load_links(links_file) if links_file.exists() else temporal_links(units))
        for q in qa.get(vid, []):
            gi, why = gold_interval(vid, q)
            excluded[why] += 1
            if gi is None:
                continue
            g0, g1 = gi
            for mode in modes:
                try:
                    cand = retrieve_pool(mode, q["question"], index, graph, encoder=encoder, cfg=cfg, complete=complete,
                                         pool=pool, link_types=link_types)
                except CacheMiss:
                    skipped += 1
                    continue
                for method in ("none", "complementarity", "complementarity+gate"):
                    picked = rerank(cand, 8, method=method.split("+")[0], index=index, graph=graph, alpha=rcfg["alpha"],
                                    beta=rcfg["beta"], gamma=rcfg["gamma"], question=q["question"], device=cfg["device"],
                                    modality_gate=method.endswith("+gate"))
                    ious = [max(iou(a, b, g0, g1) for a, b in
                                (r.unit.metadata.get("intervals") or [(r.unit.location.start_s, r.unit.location.end_s)]))
                            for r in picked]
                    g = rerank.last_gate
                    row = {"vid": vid, "kind": q["kind"], "level": q["level"], "mode": mode, "rerank": method,
                           "best_iou": max(ious) if ious else 0.0,
                           "gate": (("concentrated" if g["concentrated"] else "spread") if g else "off")}
                    for k in ks:
                        row[f"hit@{k}"] = float(any(x > 0 for x in ious[:k]))
                    row["mods"] = dict(collections.Counter(r.unit.modality for r in picked))
                    rows.append(row)
        print(f"{label} {vid}: {len(qa.get(vid, []))} QA localised")
    return rows, {"llm_usage": complete.usage, "model": str(llm.get("model")), "skipped_cells": skipped,
                  "gold": dict(excluded)}


WITH_GATE = ("none", "complementarity", "complementarity+gate")


def localisation_table(rows: list[dict], title: str, methods: tuple[str, ...] = ("none", "complementarity")) -> list[str]:
    """hit@k and mean best IoU per difficulty x mode x rerank method."""
    import statistics as st
    L = [f"### {title}", "",
         "| level | n | mode | rerank | hit@1 | hit@3 | hit@8 | mean best IoU |", "|---|---:|---|---|---:|---:|---:|---:|"]
    for level in (*LEVELS, "overall"):
        for mode in ("baseline", "linkrag", "iterative", "linkrag_iter"):
            for method in methods:
                xs = [r for r in rows if r["mode"] == mode and r["rerank"] == method and (level == "overall" or r["level"] == level)]
                if not xs:
                    continue
                m = lambda k: st.mean(r[k] for r in xs)
                L.append(f"| {level} | {len(xs)} | {mode} | {method} | {m('hit@1'):.1%} | {m('hit@3'):.1%} | {m('hit@8'):.1%} | {m('best_iou'):.3f} |")
    return L


def rebuild_fixed_index(vid: str, cfg: dict, dcfg: dict) -> tuple[float, float]:
    """Re-segment the existing transcript (word timestamps live in the audio units'
    metadata) into fixed windows and build `index_fixed`; frames reused. Returns
    (mid-sentence boundary rate sentence-mode, fixed-mode)."""
    from linkrag.ingest.audio import Word, segment_words
    src = Index.load(PROCESSED / vid / "index")
    audio = sorted([u for u in src.units if u.modality == "audio"], key=lambda u: u.location.start_s)
    frames = [u for u in src.units if u.modality != "audio"]
    words = [Word(float(w[0]), float(w[1]), str(w[2])) for u in audio for w in u.metadata.get("words", [])]
    ends_sentence = lambda w: w.text.strip().endswith((".", "?", "!"))
    mid_sentence = lambda buckets: (sum(1 for b in buckets[:-1] if b and not ends_sentence(b[-1])) / max(len(buckets) - 1, 1))
    sent_buckets = [[Word(float(w[0]), float(w[1]), str(w[2])) for w in u.metadata.get("words", [])] for u in audio]
    fixed = segment_words(words, dcfg["audio_segment_seconds"])
    units = [EvidenceUnit(id=f"{vid}:x{i}", modality="audio", content=" ".join(w.text for w in b),
                          source_file=audio[0].source_file, location=Location(start_s=b[0].start, end_s=b[-1].end),
                          metadata={"words": [(w.start, w.end, w.text) for w in b]})
             for i, b in enumerate(fixed) if b]
    out_dir = PROCESSED / vid / "index_fixed"
    if not out_dir.exists():
        encoder = default_encoder(cfg["models"]["embedding"], cfg["device"], cfg["index"]["normalize_embeddings"])
        build_index(units + frames, encoder=encoder, embedding_model=cfg["models"]["embedding"],
                    device=cfg["device"], normalize=cfg["index"]["normalize_embeddings"]).save(out_dir)
    return mid_sentence(sent_buckets), mid_sentence(fixed)


def v2(ids: list[str], cfg: dict, dcfg: dict, out: Path, *, cache_only: bool = True, max_cost: float = 1.0) -> int:
    import statistics as st
    ids = [v for v in ids if (PROCESSED / v / "index").exists()]
    rows_sent, u1 = localise(ids, cfg, dcfg, label="sentence", cache_only=cache_only, max_cost=max_cost)
    rates = [rebuild_fixed_index(v, cfg, dcfg) for v in ids]
    rows_fixed, u2 = localise(ids, cfg, dcfg, index_name="index_fixed", label="fixed", cache_only=cache_only, max_cost=max_cost)
    n_q = len({(r["vid"], r["kind"], r["level"]) for r in rows_sent}) and len(rows_sent) // 8
    L = [f"# LectQA-Vid — second pass ({len(ids)}/100 videos, {n_q} QA pairs)", "",
         f"Subset: the {len(ids)} of the first 35 videos whose YouTube links were still available; 7 were not "
         "(`data/raw/lectqa_vid/fetch_failures.txt`). Every number below is on this 28/100 subset, one run. "
         f"Transcript whisper-{cfg['models']['whisper']}, frames every {dcfg['frame_interval_s']} s + tesseract OCR, "
         f"{dcfg['audio_segment_seconds']}-second segments, retrieval pool 20, k = 8 for localisation.", "",
         "**Caption for every `linkrag` / `linkrag_iter` row:** single-stream setting — one video, no separate deck or paper — so the only link "
         "available is temporal co-occurrence between a transcript segment and the frames on screen; **cross-file linking is inactive by "
         "construction**. These rows test additive expansion + the complementarity reranker over transcript and frame-OCR units, nothing more.", "",
         "## 1. Temporal localisation (primary metric; no answerer)", "",
         "**Their paper reports no localisation metric.** Its evaluation section lists token-level precision/recall/F1, "
         "BLEU, METEOR, ROUGE-1 and semantic similarity for open-ended questions and accuracy/precision/recall/F1 over "
         "option labels for MCQ (§5.2) — nothing about whether a retrieved segment matched the annotated interval, and "
         "it never states its top-K. **This table is therefore ours-on-their-data**, with their fixed-window "
         "configuration replicated as the `fixed` baseline row in §2 rather than a number copied from the paper.", "",
         "hit@k: any of the top-k retrieved units overlaps the question's gold interval `[timestamp_start, timestamp_end]` "
         "(the files mix `HH:MM:SS`, `MM:SS` and plain-seconds stamps; all three are parsed); "
         "mean best IoU: the best temporal IoU among the top 8. `iterative` / `linkrag_iter` need one LLM call per question "
         "for the follow-up query; **this run made no new LLM calls** — those cells are filled only where the call was already "
         f"cached, so their n is smaller (sentence pass: {u1['skipped_cells']} question-mode cells skipped; fixed pass: "
         f"{u2['skipped_cells']}). No answer is generated for this table.", ""]
    L += localisation_table(rows_sent, "Sentence-aware segmentation (`ingest.audio_segmentation: sentence`, the default) — per difficulty, all modes × rerank", WITH_GATE) + [""]
    conc = [r for r in rows_sent if r["rerank"] == "complementarity+gate" and r["gate"] == "concentrated"]
    L += [f"Modality-need gate: {len(conc)}/{len([r for r in rows_sent if r['rerank'] == 'complementarity+gate'])} "
          "query-mode cells were judged *concentrated* (alpha set to 0); the rest ran full complementarity.", ""]
    L += ["## 2. Segmentation ablation on their data", "",
          "Per difficulty, `baseline` / `rerank=none` (the row their fixed-window configuration corresponds to):", "",
          "| level | segmentation | hit@1 | hit@3 | hit@8 | mean best IoU |", "|---|---|---:|---:|---:|---:|"]
    for level in (*LEVELS, "overall"):
        for nm, rws in (("sentence", rows_sent), ("fixed", rows_fixed)):
            xs = [r for r in rws if r["mode"] == "baseline" and r["rerank"] == "none" and (level == "overall" or r["level"] == level)]
            if xs:
                m = lambda k: st.mean(r[k] for r in xs)
                L.append(f"| {level} | {nm} | {m('hit@1'):.1%} | {m('hit@3'):.1%} | {m('hit@8'):.1%} | {m('best_iou'):.3f} |")
    L += ["",
          "Same transcript words, same frames, same retrieval; only the cut points change. `fixed` = equal-time windows "
          f"({dcfg['audio_segment_seconds']} s), what the baseline papers do; `sentence` = split on terminal punctuation, then packed.", "",
          "| segmentation | mid-sentence boundary rate | hit@1 (overall, baseline/none) | hit@8 | mean best IoU |", "|---|---:|---:|---:|---:|"]
    for name_, rows, rate in (("sentence", rows_sent, st.mean(r[0] for r in rates)), ("fixed", rows_fixed, st.mean(r[1] for r in rates))):
        xs = [r for r in rows if r["mode"] == "baseline" and r["rerank"] == "none"]
        L.append(f"| {name_} | {rate:.1%} | {st.mean(r['hit@1'] for r in xs):.1%} | {st.mean(r['hit@8'] for r in xs):.1%} | {st.mean(r['best_iou'] for r in xs):.3f} |")
    L += [""] + localisation_table(rows_fixed, "Fixed windows (their configuration) — per difficulty, all modes × rerank", WITH_GATE) + [""]
    # 3. answer F1 vs their rows
    fr = Path("reports/lectqa_vid_first_run.jsonl")
    L += ["## 3. Answer token-F1 per difficulty vs their Table 4 (not directly comparable)", "",
          "Their paper does not name the answering LLM (\"an instruction-tuned LLM\", §4.4.4) and evaluates against LLaVA-1.6, so no "
          "matched-strength row can be built; ours is `gpt-5.4-2026-03-05`. **Answer-F1 is therefore not directly comparable — lead with "
          "localisation above.** Their semantic similarity used all-MiniLM-L6-v2; the first-run column used bge-m3 and is omitted here.", "",
          "| level | n | mode | token-F1 | ROUGE-1 | their F1 (Table 4) | their ROUGE-1 |", "|---|---:|---|---:|---:|---:|---:|"]
    if fr.exists():
        frs = [json.loads(l) for l in fr.read_text().splitlines()]
        for level in (*LEVELS, "overall"):
            for mode in ("baseline", "linkrag"):
                xs = [r for r in frs if r["kind"] == "open" and r["mode"] == mode and (level == "overall" or r["level"] == level)]
                if xs:
                    tf, tr = THEIRS["open"][level]["f1"], THEIRS["open"][level]["rouge1"]
                    L.append(f"| {level} | {len(xs)} | {mode} | {st.mean(r['f1'] for r in xs):.1%} | {st.mean(r['rouge1'] for r in xs):.1%} | {tf:.2f}% | {tr:.2f}% |")
    footer = record_run("scripts/adapters/lectqa_vid.py", f"lectqa v2 localisation ({len(ids)} videos)",
                        [(u1["model"], u1["llm_usage"]), (u2["model"], u2["llm_usage"])], cfg["models"].get("pricing"))
    L += footer
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n")
    out.with_suffix(".jsonl").write_text("\n".join(json.dumps({**r, "segmentation": "sentence"}) for r in rows_sent)
                                         + "\n" + "\n".join(json.dumps({**r, "segmentation": "fixed"}) for r in rows_fixed) + "\n")
    print("\n".join(L[12:40]))
    print(f"wrote {out}")
    return 0
