"""Open-ended answers: T1 replica vs iterative vs full context, their metrics, and faithfulness."""

from __future__ import annotations

import collections
import json
from pathlib import Path

import numpy as np

from linkrag.core import EvidenceUnit, Location
from linkrag.costs import cached_completer, price_for, record_run, usage_cost
from linkrag.generate.answer import http_completer
from linkrag.index import Index, default_encoder

from lectqa.common import LEVELS, PROCESSED, THEIRS, gold_interval, load_qa


OPEN_SEED = "lectqa-open-subset-20260923"


OPEN_MODES = ("baseline", "iterative", "full_context")


# T1 Table 3 values; M and L (re-rank pool, context size) and the merge gap are not given
# in the paper -- ours: M < K as §4.4.3 requires, L = the top_k every other mode here gets.
T1 = {"theta_sem": 0.75, "theta_t": 2.0, "delta_gap": 5.0, "delta_dup": 0.90, "theta_r": 0.60, "dt": 3.0,
      "K": 10, "M": 8, "L": 4}


T1_EMBEDDER = "BAAI/bge-base-en-v1.5"                 # "BGE base", d = 768, mean-pooled (eqs. 14-15)


T1_RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # "fine-tuned on QA relevance pairs", r in [0, 1]; unnamed


EVIDENCE = PROCESSED / "open_evidence.jsonl"          # item-4 input: the units each subset answer saw


def t1_encoder(device: str):
    from sentence_transformers import SentenceTransformer, models
    tr = models.Transformer(T1_EMBEDDER)
    m = SentenceTransformer(modules=[tr, models.Pooling(tr.get_word_embedding_dimension(), pooling_mode="mean"),
                                     models.Normalize()], device=device)
    return lambda texts: m.encode(list(texts), convert_to_numpy=True)


def t1_chunks(vid: str, index: Index, enc) -> tuple[list[EvidenceUnit], np.ndarray]:
    """T1 §4.2 on our inputs: whisper words re-split into sentences; a frame-OCR unit
    overlapping speech by more than theta_t is suppressed (audio priority, eq. 10);
    near-duplicates (cosine > delta_dup to a kept item) dropped; adjacent items merged
    while cosine > theta_sem and gap < delta_gap (eqs. 11-12). Text normalisation and
    boundary refinement (§4.2.1-2) are not replicated: whisper-small sentences are short."""
    audio = sorted((u for u in index.units if u.modality == "audio"), key=lambda u: u.location.start_s)
    items, cur = [], []
    for w in (w for u in audio for w in u.metadata.get("words", [])):
        cur.append(w)
        if str(w[2]).strip().endswith((".", "?", "!")):
            items.append(cur)
            cur = []
    items = [(" ".join(str(w[2]).strip() for w in s), float(s[0][0]), float(s[-1][1]), "audio") for s in items + [cur] if s]
    speech = list(items)
    for f in index.units:
        if f.modality == "figure" and f.content.strip():
            a, b = f.location.start_s, f.location.end_s
            if max((min(b, e) - max(a, s) for _, s, e, _ in speech), default=0.0) <= T1["theta_t"]:
                items.append((" ".join(f.content.split()), a, b, "figure"))
    items.sort(key=lambda x: x[1])
    vec = np.asarray(enc([x[0] for x in items]), dtype=np.float32)
    keep: list[int] = []
    for i in range(len(items)):
        if not keep or float(np.max(vec[keep] @ vec[i])) <= T1["delta_dup"]:
            keep.append(i)
    groups = [[keep[0]]] if keep else []
    for a, b in zip(keep, keep[1:]):
        if float(vec[a] @ vec[b]) > T1["theta_sem"] and items[b][1] - items[a][2] < T1["delta_gap"]:
            groups[-1].append(b)
        else:
            groups.append([b])
    units = [EvidenceUnit(id=f"{vid}:c{n}", modality="audio" if any(items[i][3] == "audio" for i in g) else "figure",
                          content=" ".join(items[i][0] for i in g), source_file=audio[0].source_file,
                          location=Location(start_s=items[g[0]][1], end_s=max(items[i][2] for i in g)))
             for n, g in enumerate(groups)]
    return units, np.asarray(enc([u.content for u in units]), dtype=np.float32)


def t1_retrieve(q: str, gi: tuple[float, float] | None, chunks: list[EvidenceUnit], vecs: np.ndarray, enc,
                ce) -> list[EvidenceUnit]:
    """T1 §4.4: top-K cosine above theta_r (eqs. 17, 21); the temporal filter of eq. 22 on the
    question's annotated interval +- dt when it is usable; merge of temporally consecutive
    results with gap < delta_gap (eq. 23); cross-encoder over the top-M (eqs. 24-25); the top-L
    in temporal order (eqs. 26-27). Eq. 22 reads the gold timestamps: an oracle."""
    sims = vecs @ np.asarray(enc([q]), dtype=np.float32)[0]
    top = [i for i in np.argsort(-sims)[:T1["K"]] if sims[i] > T1["theta_r"]]
    if gi is not None:
        lo, hi = gi[0] - T1["dt"], gi[1] + T1["dt"]
        top = [i for i in top if chunks[i].location.start_s <= hi and chunks[i].location.end_s >= lo]
    top.sort(key=lambda i: chunks[i].location.start_s)
    groups: list[list[int]] = []
    for i in top:
        if groups and chunks[i].location.start_s - chunks[groups[-1][-1]].location.end_s < T1["delta_gap"]:
            groups[-1].append(i)
        else:
            groups.append([i])
    merged = [(max(sims[i] for i in g),
               EvidenceUnit(id=f"{chunks[g[0]].id}" + (f"-{chunks[g[-1]].id.rsplit(':', 1)[1]}" if len(g) > 1 else ""),
                            modality=chunks[g[0]].modality, content=" ".join(chunks[i].content for i in g),
                            source_file=chunks[g[0]].source_file,
                            location=Location(start_s=chunks[g[0]].location.start_s,
                                              end_s=max(chunks[i].location.end_s for i in g))))
              for g in groups]
    pool = [u for _, u in sorted(merged, key=lambda x: -x[0])[:T1["M"]]]
    if not pool:
        return []
    r = ce.predict([(q, u.content) for u in pool])
    best = [pool[i] for i in np.argsort(-np.asarray(r))[:T1["L"]]]
    return sorted(best, key=lambda u: u.location.start_s)


def open_items(ids: list[str]) -> tuple[list[dict], set[int]]:
    """Every open-ended question of the prepared videos, and a seeded stratified subset
    (100 per difficulty)."""
    import random
    qa, items = load_qa(), []
    for vid in ids:
        if (PROCESSED / vid / "index").exists():
            items += [{"vid": vid, "qi": i, "level": q["level"], "question": q["question"], "ref": q["answer"], "q": q}
                      for i, q in enumerate(x for x in qa.get(vid, []) if x["kind"] == "open")]
    rng = random.Random(OPEN_SEED)
    subset = set()
    for level in LEVELS:
        subset |= set(rng.sample([j for j, it in enumerate(items) if it["level"] == level], 100))
    return items, subset


def open_modes(ids: list[str], cfg: dict, dcfg: dict, out: Path, *, max_cost: float, dry_run: bool,
               repeats: int = 3, rest: int | None = None) -> int:
    """Open-ended answers under T1's protocol and metrics: baseline (T1 replica), iterative
    (ours), full_context (no retrieval). The subset runs `repeats` times; the rest once,
    in seeded order, `rest` questions of it (all by default)."""
    import random
    import threading
    from concurrent.futures import ThreadPoolExecutor
    import tiktoken
    from sentence_transformers import CrossEncoder
    from linkrag.generate.answer import JSON_SYSTEM, answer_json, build_prompt
    from linkrag.retrieve.iterative import retrieve_iterative
    llm = {**cfg["models"]["llm"], "max_tokens": 400}
    price = price_for(str(llm["model"]), cfg["models"].get("pricing"))
    items, subset = open_items(ids)
    others = [j for j in range(len(items)) if j not in subset]
    random.Random(OPEN_SEED + ":rest").shuffle(others)
    others = others[:rest] if rest is not None else others
    run_idx = sorted(subset) + others
    encoder = default_encoder(cfg["models"]["embedding"], cfg["device"], cfg["index"]["normalize_embeddings"])
    encoder([""])
    t1enc, ce = t1_encoder(cfg["device"]), CrossEncoder(T1_RERANKER, device=cfg["device"])
    indexes, chunks = {}, {}
    for vid in sorted({items[j]["vid"] for j in run_idx}):
        indexes[vid] = Index.load(PROCESSED / vid / "index")
        chunks[vid] = t1_chunks(vid, indexes[vid], t1enc)
    fixed: dict[tuple[int, str], list[EvidenceUnit]] = {}
    oracle = {}
    for j in run_idx:
        it = items[j]
        gi, _ = gold_interval(it["vid"], it["q"])
        oracle[j] = gi is not None
        fixed[(j, "baseline")] = t1_retrieve(it["question"], gi, *chunks[it["vid"]], t1enc, ce)
        fixed[(j, "full_context")] = sorted((u for u in indexes[it["vid"]].units if u.modality == "audio"),
                                            key=lambda u: u.location.start_s)
    # repeat-major: a cost cap reached mid-run cuts the last repeat, not whole questions
    tasks = ([(j, m, r) for r in range(repeats) for j in sorted(subset) for m in OPEN_MODES]
             + [(j, m, 0) for j in others for m in OPEN_MODES])

    # cost estimate: exact prompt tokens for the fixed-evidence modes; the iterative prompt is
    # taken as the baseline's plus one follow-up call over 8 units; 110 output tokens per answer
    enc = tiktoken.get_encoding("o200k_base")
    ptok = lambda units, q: len(enc.encode(JSON_SYSTEM + build_prompt(q, units))) + 8
    tok_in = tok_out = 0
    for j, m, _ in tasks:
        q = items[j]["question"]
        if m == "iterative":
            tok_in += ptok(fixed[(j, "baseline")] or fixed[(j, "full_context")][:T1["L"]], q) + 600
            tok_out += 110 + 25
        else:
            tok_in += ptok(fixed[(j, m)], q) if fixed[(j, m)] else 0
            tok_out += 110 if fixed[(j, m)] else 0
    est = tok_in / 1e6 * price["input_per_m"] + tok_out / 1e6 * price["output_per_m"]
    empty = sum(1 for j in run_idx if not fixed[(j, "baseline")])
    print(f"{len(items)} open-ended on {len({it['vid'] for it in items})} videos · subset {len(subset)} x {repeats} "
          f"+ rest {len(others)} x 1 · {len(tasks)} answers · ~{tok_in:,} in / ~{tok_out:,} out · estimated ${est:.2f} "
          f"(cap ${max_cost:.2f}) · baseline: oracle filter on {sum(oracle[j] for j in run_idx)}/{len(run_idx)}, "
          f"empty context {empty}")
    if dry_run:
        return 0
    if est > max_cost:
        raise SystemExit(f"estimated ${est:.2f} exceeds --max-cost {max_cost}; nothing sent (lower --rest)")

    complete = cached_completer(http_completer(llm), PROCESSED / "llm_cache" / f"{llm['model']}-open",
                                max_cost_usd=max_cost, price=price)
    lock = threading.Lock()

    def locked(texts):
        with lock:
            return encoder(texts)

    def one(task):
        try:
            return answer_one(task)
        except RuntimeError as exc:                   # the cost cap: report what ran, never exceed it
            if "cost cap" not in str(exc):
                raise
            return None, None

    def answer_one(task):
        j, mode, rep = task
        it = items[j]
        units = fixed.get((j, mode))
        if mode == "iterative":
            icfg = cfg["retrieve"]["iterative"]
            units = [r.unit for r in retrieve_iterative(
                it["question"], indexes[it["vid"]], encoder=locked, complete=complete, rounds=icfg["rounds"],
                k_per_round=icfg["k_per_round"], k_final=T1["L"], candidates=cfg["retrieve"]["candidates"],
                rrf_k=cfg["retrieve"]["rrf_k"]).units]
        ans = answer_json(it["question"], units, mode="baseline", complete=complete)
        row = {"vid": it["vid"], "qi": j, "level": it["level"], "mode": mode, "rep": rep, "subset": j in subset,
               "oracle": oracle[j] and mode == "baseline", "answer": ans.answer, "ref": it["ref"],
               "malformed": ans.malformed, "claims": [{"claim": c.claim, "unit_ids": c.unit_ids} for c in ans.claims],
               "evidence": [u.id for u in units]}
        ev = None
        if j in subset and rep == 0:
            ev = {"qi": j, "mode": mode, "units": [{"id": u.id, "modality": u.modality, "content": u.content,
                                                    "source_file": u.source_file, "start_s": u.location.start_s,
                                                    "end_s": u.location.end_s} for u in units]}
        return row, ev

    rows, evs = [], []
    with ThreadPoolExecutor(max_workers=8) as ex:
        for n, (row, ev) in enumerate(ex.map(one, tasks), 1):
            if row:
                rows.append(row)
            if ev:
                evs.append(ev)
            if n % 300 == 0:
                print(f"  {n}/{len(tasks)} answered · ${usage_cost(complete.usage, price) or 0:.3f}")
    EVIDENCE.write_text("\n".join(json.dumps(e) for e in evs) + "\n")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return open_report(rows, items, subset, cfg, llm, repeats, out,
                       record_run("scripts/adapters/lectqa_vid.py", "lectqa_vid open-ended baseline/iterative/full_context",
                                  [(str(llm["model"]), complete.usage)], cfg["models"].get("pricing")))


def faithfulness(cfg: dict, src: Path, out: Path, *, max_cost: float, dry_run: bool, runs: int) -> int:
    """Claim-level verification (`linkrag.generate.verify`, judge `eval.judge`) of the subset's
    repeat-0 answers, every mode: each claim against the units it cites."""
    import tiktoken
    from concurrent.futures import ThreadPoolExecutor
    from linkrag.eval.verify_gold import ENTAIL_PROMPT, JUDGE_SYSTEM
    from linkrag.generate.answer import Answer, Claim
    from linkrag.generate.verify import hallucination_rate, verify_answer
    rows = [r for r in map(json.loads, src.read_text().splitlines()) if r["subset"] and r["rep"] == 0]
    ev = {}
    for e in map(json.loads, EVIDENCE.read_text().splitlines()):
        ev[(e["qi"], e["mode"])] = [EvidenceUnit(id=u["id"], modality=u["modality"], content=u["content"],
                                                 source_file=u["source_file"],
                                                 location=Location(start_s=u["start_s"], end_s=u["end_s"]))
                                    for u in e["units"]]
    judge = cfg["eval"]["judge"]
    price = price_for(str(judge["model"]), cfg["models"].get("pricing"))
    enc = tiktoken.get_encoding("o200k_base")
    lo = hi = 0                                        # input tokens: first cited unit only / every cited unit
    n_claims = 0
    for r in rows:
        by_id = {u.id: u for u in ev[(r["qi"], r["mode"])]}
        for c in r["claims"]:
            n_claims += 1
            toks = [len(enc.encode(JUDGE_SYSTEM + ENTAIL_PROMPT.format(question=c["claim"], expected=c["claim"],
                                                                      text=by_id[i].content))) + 8
                    for i in c["unit_ids"] if i in by_id]
            lo, hi = lo + (toks[0] if toks else 0), hi + sum(toks)
    cost = lambda tin, calls: runs * (tin / 1e6 * price["input_per_m"] + calls * 120 / 1e6 * price["output_per_m"])
    est_lo, est_hi = cost(lo, n_claims), cost(hi, n_claims * 2)
    print(f"{len(rows)} answers · {n_claims} claims · judge `{judge['model']}` x {runs} runs · estimated "
          f"${est_lo:.2f}-${est_hi:.2f} (cap ${max_cost:.2f})")
    if dry_run:
        return 0
    if est_hi > max_cost:
        raise SystemExit(f"upper estimate ${est_hi:.2f} exceeds --max-cost {max_cost}; nothing sent (lower --runs)")
    complete = cached_completer(http_completer(judge), PROCESSED / "llm_cache" / f"{judge['model']}-faith",
                                max_cost_usd=max_cost, price=price)

    def check(r):
        ans = Answer(answer=r["answer"], claims=[Claim(**c) for c in r["claims"]], raw="")
        return {**{k: r[k] for k in ("vid", "qi", "level", "mode")},
                **verify_answer(ans, ev[(r["qi"], r["mode"])], complete, runs=runs, workers=4,
                                answer_key=f"{r['qi']}:{r['mode']}")}

    with ThreadPoolExecutor(max_workers=4) as ex:
        res = list(ex.map(check, rows))
    out.with_suffix(".jsonl").write_text("\n".join(json.dumps(x) for x in res) + "\n")
    answerer = str(cfg["models"]["llm"]["model"])
    L = ["# LectQA-Vid — faithfulness per mode", "",
         f"The {len(res)} repeat-0 answers of the open-ended subset (`lectqa_open_modes.md`: 100 questions per "
         f"difficulty, seed `{OPEN_SEED}`, {len({x['vid'] for x in res})} videos) · answerer `{answerer}` · judge "
         f"`{judge['model']}` temperature {judge.get('temperature')}, {runs} run(s) per check"
         + (", majority" if runs > 1 else "") + " · `linkrag.generate.verify.verify_answer`: each claim against "
         "the text of the units it cites; the first entailing unit wins, with a quotable span required.", "",
         "**supported**: a cited unit entails the claim. **weak**: the claim cites no unit of the evidence. "
         "**unsupported**: no cited unit entails it. **Hallucination rate** = unsupported / claims, pooled over "
         "answers (`hallucination_rate`). Supported means the claim is stated in the cited evidence. It does "
         "not mean the answer is correct.", "",
         "| level | mode | answers | claims | supported | weak | unsupported | hallucination rate | answers with no claim |",
         "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for level in (*LEVELS, "overall"):
        for mode in OPEN_MODES:
            sel = [x for x in res if x["mode"] == mode and (level == "overall" or x["level"] == level)]
            if sel:
                claims = sum(len(x["claims"]) for x in sel)
                L.append(f"| {level} | {mode} | {len(sel)} | {claims} | {sum(x['supported'] for x in sel)} | "
                         f"{sum(x['weak'] for x in sel)} | {sum(x['unsupported'] for x in sel)} | "
                         f"{100 * hallucination_rate(sel):.1f} % | {sum(not x['claims'] for x in sel)} |")
    tot = sum(len(x["claims"]) for x in res)
    assert tot == sum(x[k] for x in res for k in ("supported", "weak", "unsupported")), "verdicts must sum to claims"
    L += [""] + record_run("scripts/adapters/lectqa_vid.py", "lectqa_vid faithfulness per mode",
                           [(str(judge["model"]), complete.usage)], cfg["models"].get("pricing"))
    out.write_text("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


METRICS = (("f1", "F1"), ("sim", "Sim"), ("bleu", "BLEU"), ("meteor", "METEOR"), ("rouge1", "R1"))


def open_report(rows: list[dict], items: list[dict], subset: set[int], cfg: dict, llm: dict, repeats: int,
                out: Path, footer: list[str]) -> int:
    import statistics as st
    from linkrag.eval import lectqa_metrics as LM
    score = lambda sel: LM.score_open([r["answer"] for r in sel], [r["ref"] for r in sel], device=cfg["device"])
    sub_vids = len({items[j]["vid"] for j in subset})
    all_vids = len({r["vid"] for r in rows})
    label = (f"answerer `{llm['model']}` temperature {llm.get('temperature')} (JSON answer with claims, the same "
             f"prompt for every mode) · metrics T1 eqs. 29-35 (`linkrag.eval.lectqa_metrics`), in %")
    L = ["# LectQA-Vid — open-ended: T1 replica vs iterative vs full context", "",
         "Three ways to give the same answerer its context, scored with T1's own metric suite:", "",
         f"- **baseline**: T1's pipeline (§4.2-4.4) replicated. Chunks: whisper sentences merged while cosine > "
         f"{T1['theta_sem']} and gap < {T1['delta_gap']} s, near-duplicates (> {T1['delta_dup']}) dropped, frame OCR "
         f"overlapping speech by > {T1['theta_t']} s suppressed; embedder `{T1_EMBEDDER}` mean-pooled; top-{T1['K']} "
         f"above cosine {T1['theta_r']}; temporal filter (eq. 22) on the question's annotated interval ± {T1['dt']} s; "
         f"merge of consecutive results with gap < {T1['delta_gap']} s; cross-encoder `{T1_RERANKER}` over the top-"
         f"{T1['M']}; top-{T1['L']} in temporal order. **Eq. 22 reads the gold timestamps (an oracle).** It is applied "
         "only where the stamp is usable (`gold_interval`); elsewhere the pipeline runs without it. Both are "
         "reported separately below.",
         f"- **iterative**: ours: RRF (bge-m3 dense + BM25) over our sentence-packed transcript and frame-OCR "
         f"units, one LLM follow-up query, top-{T1['L']}. No timestamps are read.",
         "- **full_context**: no retrieval; every transcript unit of the video, in time order. No timestamps "
         "and no frame OCR.", "",
         "T1 leaves open: M, L and the merge gap (ours above); the cross-encoder (ours: MS MARCO MiniLM); one "
         "FAISS index over all videos then a video-id filter (ours: each video's own chunks, i.e. that filter "
         "without other videos competing for the top-K); Gemini captions (ours: tesseract OCR, mostly "
         "suppressed by their θt rule).", ""]
    rest = len(items) - len(subset)
    if not any(not r["subset"] for r in rows):
        L += [f"**Scope.** The single run on the other {rest} open-ended questions was not made: the dry run "
              "estimated $7.43 for subset + rest against the item's $3.50 cap, so only the subset ran.", ""]
    tables = [(f"## Subset: {len(subset)} questions (100 per difficulty, seed `{OPEN_SEED}`), {sub_vids} videos, "
               f"{repeats} repeats, mean ± std over repeats", [r for r in rows if r["subset"]], repeats)]
    if any(not r["subset"] for r in rows):
        tables.append((f"## All run questions, one run (subset repeat 0 + the rest), {all_vids} videos",
                       [r for r in rows if r["rep"] == 0], 1))
    for title, sel_rows, reps in tables:
        L += [title, "", label + ".", "",
              "| level | mode | n | " + " | ".join(h for _, h in METRICS) + " | abstained |",
              "|---|---|---:|" + "---:|" * (len(METRICS) + 1)]
        for level in (*LEVELS, "overall"):
            for mode in OPEN_MODES:
                sel = [r for r in sel_rows if r["mode"] == mode and (level == "overall" or r["level"] == level)]
                if not sel:
                    continue
                per = [score([r for r in sel if r["rep"] == k]) for k in range(reps)]
                cell = lambda key: (f"{100*st.mean(p[key] for p in per):.2f} ± {100*st.pstdev(p[key] for p in per):.2f}"
                                    if reps > 1 else f"{100*per[0][key]:.2f}")
                abst = sum("not found in the provided material" in r["answer"].lower() or
                           r["answer"].startswith("No evidence was retrieved") for r in sel if r["rep"] == 0)
                L.append(f"| {level} | {mode} | {per[0]['n']} | " + " | ".join(cell(k) for k, _ in METRICS)
                         + f" | {abst} |")
            th = THEIRS["open"][level]
            L.append(f"| {level} | **T1 Table 4** | — | " + " | ".join(f"{th[k]:.2f}" for k, _ in METRICS) + " | — |")
        L.append("")
    # paired over questions: per-question F1 averaged over repeats, then a seeded bootstrap of the mean difference
    f1 = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        if r["subset"]:
            f1[(r["qi"], r["level"])][r["mode"]].append(LM.token_prf(r["answer"], r["ref"])[2])
    L += ["## Paired differences (subset, token F1 points, 95 % bootstrap CI over questions)", "",
          "Each question's F1 averaged over its repeats; the mean paired difference and a 10,000-resample "
          "bootstrap over questions (numpy seed 20260923). This is the question-sampling uncertainty; the ± "
          "above is run-to-run noise only.", "", "| difference | " + " | ".join((*LEVELS, "overall")) + " |",
          "|---|" + "---:|" * (len(LEVELS) + 1)]
    for a, b in (("full_context", "iterative"), ("full_context", "baseline"), ("iterative", "baseline")):
        cells = []
        for level in (*LEVELS, "overall"):
            d = np.array([100 * (st.mean(m[a]) - st.mean(m[b])) for (_, lv), m in f1.items()
                          if level == "overall" or lv == level])
            boots = np.random.default_rng(20260923).choice(d, (10000, len(d))).mean(axis=1)
            lo, hi = np.percentile(boots, [2.5, 97.5])
            cells.append(f"{d.mean():+.2f} [{lo:+.2f}, {hi:+.2f}]")
        L.append(f"| {a} − {b} | " + " | ".join(cells) + " |")
    L.append("")
    one = [r for r in rows if r["rep"] == 0]
    L += ["## Baseline with and without its oracle filter (repeat 0, subset)", "", label + ".", "",
          "The two groups are different questions, so the other modes' F1 on the same groups is the control: "
          "a gap they share is the questions, not the filter.", "",
          "| questions | n | baseline F1 | baseline Sim | baseline empty context | iterative F1 | full_context F1 |",
          "|---|---:|---:|---:|---:|---:|---:|"]
    for name, keep in (("eq. 22 applied (usable gold stamp)", True), ("eq. 22 not applicable", False)):
        qs = {r["qi"] for r in one if r["mode"] == "baseline" and r["oracle"] == keep}
        by = {m: [r for r in one if r["mode"] == m and r["qi"] in qs] for m in OPEN_MODES}
        if qs:
            s, si, sf = score(by["baseline"]), score(by["iterative"]), score(by["full_context"])
            L.append(f"| {name} | {len(qs)} | {100*s['f1']:.2f} | {100*s['sim']:.2f} | "
                     f"{sum(not r['evidence'] for r in by['baseline'])} | {100*si['f1']:.2f} | {100*sf['f1']:.2f} |")
    L += ["", f"Malformed JSON replies (answer scored as the raw text): {sum(r['malformed'] for r in rows)} of {len(rows)}."
          f" Answers run: {len(rows)} (subset {sum(r['subset'] for r in rows)}, rest {sum(not r['subset'] for r in rows)})."]
    L += footer
    out.write_text("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


def open_rerender(ids: list[str], cfg: dict, out: Path, repeats: int = 3) -> int:
    """The report again from the stored answers: no LLM call, the original run's cost footer kept."""
    rows = [json.loads(x) for x in out.with_suffix(".jsonl").read_text().splitlines()]
    items, subset = open_items(ids)
    text = out.read_text()
    footer = text[text.index("\nLLM cost (this run):"):].rstrip("\n").split("\n")
    return open_report(rows, items, subset, cfg, cfg["models"]["llm"], repeats, out, footer)
