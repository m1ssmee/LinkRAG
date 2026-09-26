"""Attribution ablation: which part of "ours vs the T1 replica" comes from re-querying, which from
the rest of the pipeline, and what T1's eq. 22 oracle filter does.

Two new modes on the open-ended subset, same answerer, same prompt, same repeats:
- `rrf`: our plain baseline -- RRF (bge-m3 dense + BM25) over our units, top-L, no follow-up query.
- `replica_no_eq22`: the T1 replica with eq. 22 disabled (the gold interval is never read).
The replica (stored mode `baseline`), `iterative` and `full_context` rows are the stored answers of
`lectqa_open_modes.jsonl`, not re-run. Answers go through the same reply cache, so wherever the evidence
is identical to a stored run's (e.g. the replica on questions eq. 22 never applied to) the stored reply
is replayed at $0.
"""

from __future__ import annotations

import collections
import json
import statistics as st
from pathlib import Path

from linkrag.costs import cached_completer, price_for, record_run, usage_cost
from linkrag.generate.answer import http_completer
from linkrag.index import Index, default_encoder

from lectqa.common import LEVELS, PROCESSED, THEIRS, gold_interval
from lectqa.open_ended import (METRICS, OPEN_SEED, T1, T1_RERANKER, open_items, paired_cell, t1_chunks,
                               t1_encoder, t1_retrieve)

NEW_MODES = ("rrf", "replica_no_eq22")
LABEL = {"baseline": "T1 replica (eq. 22 on)", "replica_no_eq22": "T1 replica, eq. 22 off",
         "rrf": "ours: RRF, no re-query", "iterative": "ours: RRF + one re-query (iterative)",
         "full_context": "full transcript"}
ORDER = ("baseline", "replica_no_eq22", "rrf", "iterative", "full_context")
PAIRS = (("baseline", "replica_no_eq22"), ("iterative", "rrf"), ("rrf", "baseline"))


def attribution(ids: list[str], cfg: dict, stored: Path, out: Path, *, max_cost: float, dry_run: bool,
                repeats: int = 3) -> int:
    import tiktoken
    from concurrent.futures import ThreadPoolExecutor
    from sentence_transformers import CrossEncoder
    from linkrag.generate.answer import JSON_SYSTEM, answer_json, build_prompt
    from linkrag.retrieve.baseline import retrieve_scored
    llm = {**cfg["models"]["llm"], "max_tokens": 400}           # as lectqa_open_modes
    price = price_for(str(llm["model"]), cfg["models"].get("pricing"))
    items, subset = open_items(ids)
    old = [r for r in map(json.loads, stored.read_text().splitlines()) if r["subset"]]
    assert {r["mode"] for r in old} == {"baseline", "iterative", "full_context"} and len(old) == 3 * repeats * len(subset)
    encoder = default_encoder(cfg["models"]["embedding"], cfg["device"], cfg["index"]["normalize_embeddings"])
    encoder([""])
    t1enc, ce = t1_encoder(cfg["device"]), CrossEncoder(T1_RERANKER, device=cfg["device"])
    evidence, gold = {}, {}
    for vid in sorted({items[j]["vid"] for j in subset}):
        index = Index.load(PROCESSED / vid / "index")
        chunks = t1_chunks(vid, index, t1enc)
        for j in (j for j in sorted(subset) if items[j]["vid"] == vid):
            q = items[j]["question"]
            gold[j] = gold_interval(vid, items[j]["q"])[0] is not None
            evidence[(j, "replica_no_eq22")] = t1_retrieve(q, None, *chunks, t1enc, ce)
            evidence[(j, "rrf")] = [u for u, _ in retrieve_scored(
                q, index, encoder=encoder, top_k=T1["L"], candidates=cfg["retrieve"]["candidates"],
                rrf_k=cfg["retrieve"]["rrf_k"])]
    tasks = [(j, m, r) for r in range(repeats) for j in sorted(subset) for m in NEW_MODES]
    enc = tiktoken.get_encoding("o200k_base")
    tok_in = sum(len(enc.encode(JSON_SYSTEM + build_prompt(items[j]["question"], evidence[(j, m)]))) + 8
                 for j, m, _ in tasks if evidence[(j, m)])
    tok_out = 110 * sum(1 for j, m, _ in tasks if evidence[(j, m)])
    est = tok_in / 1e6 * price["input_per_m"] + tok_out / 1e6 * price["output_per_m"]
    print(f"subset {len(subset)} x {repeats} x {len(NEW_MODES)} new modes = {len(tasks)} answers · ~{tok_in:,} in / "
          f"~{tok_out:,} out · estimated ${est:.2f} before cache hits (cap ${max_cost:.2f}) · empty evidence: "
          + ", ".join(f"{m} {sum(not evidence[(j, m)] for j in subset)}" for m in NEW_MODES))
    if dry_run:
        return 0
    if est > max_cost:
        raise SystemExit(f"estimated ${est:.2f} exceeds --max-cost {max_cost}; nothing sent")
    complete = cached_completer(http_completer(llm), PROCESSED / "llm_cache" / f"{llm['model']}-open",
                                max_cost_usd=max_cost, price=price)

    def answer_one(task):
        j, mode, rep = task
        units = evidence[(j, mode)]
        ans = answer_json(items[j]["question"], units, mode="baseline", complete=complete)
        return {"vid": items[j]["vid"], "qi": j, "level": items[j]["level"], "mode": mode, "rep": rep,
                "subset": True, "oracle": False, "answer": ans.answer, "ref": items[j]["ref"],
                "malformed": ans.malformed, "claims": [{"claim": c.claim, "unit_ids": c.unit_ids} for c in ans.claims],
                "evidence": [u.id for u in units]}

    with ThreadPoolExecutor(max_workers=8) as ex:
        new = list(ex.map(answer_one, tasks))
    print(f"  {len(new)} answered · ${usage_cost(complete.usage, price) or 0:.3f} uncached")
    out.with_suffix(".jsonl").write_text("\n".join(json.dumps(r) for r in new) + "\n")
    footer = record_run("scripts/adapters/lectqa_vid.py", "lectqa_vid attribution ablation (rrf, replica_no_eq22)",
                        [(str(llm["model"]), complete.usage)], cfg["models"].get("pricing"))
    return attribution_report(old + new, gold, cfg, llm, repeats, out, footer)


def attribution_report(rows: list[dict], gold: dict[int, bool], cfg: dict, llm: dict, repeats: int, out: Path,
                       footer: list[str]) -> int:
    from linkrag.eval import lectqa_metrics as LM
    score = lambda sel: LM.score_open([r["answer"] for r in sel], [r["ref"] for r in sel], device=cfg["device"])
    n_q = len({r["qi"] for r in rows})
    L = ["# LectQA-Vid — attribution ablation: re-query vs pipeline vs eq. 22", "",
         f"Open-ended subset: {n_q} questions (100 per difficulty, seed `{OPEN_SEED}`), "
         f"{len({r['vid'] for r in rows})} videos, {repeats} repeats · answerer `{llm['model']}` temperature "
         f"{llm.get('temperature')}, the same JSON answer prompt for every mode · metrics T1 eqs. 29-35 "
         "(`linkrag.eval.lectqa_metrics`), in %, mean ± std over repeats.", "",
         "Modes (context handed to the same answerer, top-4 units unless stated):", "",
         f"- **{LABEL['baseline']}** (stored mode `baseline`, `lectqa_open_modes.jsonl`): T1's pipeline replicated "
         "(bge-base, semantic chunking, top-10 above cosine 0.6, merge, MS MARCO MiniLM cross-encoder). Eq. 22 keeps "
         "only chunks overlapping the question's **annotated (gold) interval ± 3 s, read at test time**, where the "
         "stamp is usable.",
         f"- **{LABEL['replica_no_eq22']}** (new): the same pipeline; the gold interval is never read.",
         f"- **{LABEL['rrf']}** (new): our plain baseline, RRF of bge-m3 dense + BM25 over our sentence-packed "
         "transcript and frame-OCR units; no LLM call before answering.",
         f"- **{LABEL['iterative']}** (stored): the same RRF, one LLM follow-up query, top-4 of the merged rounds.",
         f"- **{LABEL['full_context']}** (stored): every transcript unit, no retrieval.", "",
         "The replica, iterative and full-context rows are the stored answers of `lectqa_open_modes.jsonl` (not "
         "re-run). New answers used the same reply cache: where a new mode's evidence equals a stored run's, the "
         "stored reply was replayed.", "",
         "| level | mode | n | " + " | ".join(h for _, h in METRICS) + " | abstained |",
         "|---|---|---:|" + "---:|" * (len(METRICS) + 1)]
    for level in (*LEVELS, "overall"):
        for mode in ORDER:
            sel = [r for r in rows if r["mode"] == mode and (level == "overall" or r["level"] == level)]
            per = [score([r for r in sel if r["rep"] == k]) for k in range(repeats)]
            cell = lambda key: f"{100*st.mean(p[key] for p in per):.2f} ± {100*st.pstdev(p[key] for p in per):.2f}"
            abst = sum("not found in the provided material" in r["answer"].lower() or
                       r["answer"].startswith("No evidence was retrieved") for r in sel if r["rep"] == 0)
            L.append(f"| {level} | {LABEL[mode]} | {per[0]['n']} | " + " | ".join(cell(k) for k, _ in METRICS)
                     + f" | {abst} |")
        th = THEIRS["open"][level]
        L.append(f"| {level} | *T1 Table 4 (published; unnamed answerer)* | — | "
                 + " | ".join(f"{th[k]:.2f}" for k, _ in METRICS) + " | — |")
    f1 = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        f1[(r["qi"], r["level"])][r["mode"]].append(LM.token_prf(r["answer"], r["ref"])[2])
    L += ["", "## Paired differences (token F1 points, 95 % bootstrap CI over questions)", "",
          "Each question's F1 averaged over its repeats; mean paired difference, 10,000-resample bootstrap over "
          "questions (numpy seed 20260923), as in `lectqa_open_modes.md`.", "",
          "| difference | what it isolates | " + " | ".join((*LEVELS, "overall")) + " |",
          "|---|---|" + "---:|" * (len(LEVELS) + 1)]
    what = {("baseline", "replica_no_eq22"): "the eq. 22 oracle filter",
            ("iterative", "rrf"): "the re-query (same index, same top-4)",
            ("rrf", "baseline"): "the rest of the pipeline: embedder, index, chunking, reranker, eq. 22"}
    for a, b in PAIRS:
        cells = [paired_cell([100 * (st.mean(m[a]) - st.mean(m[b])) for (_, lv), m in f1.items()
                              if level == "overall" or lv == level]) for level in (*LEVELS, "overall")]
        L.append(f"| {LABEL[a]} − {LABEL[b]} | {what[(a, b)]} | " + " | ".join(cells) + " |")
    one = [r for r in rows if r["rep"] == 0]
    on = {j for j, g in gold.items() if g}
    empty = {r["qi"] for r in one if r["mode"] == "baseline" and not r["evidence"]} & on
    L += ["", "## Eq. 22 at test time", "",
          f"Eq. 22 read the gold interval on {len(on)} of the {n_q} questions (the others have no usable stamp, so "
          "the replica ran without it and its evidence there is identical with eq. 22 on or off). "
          f"On {len(empty)} questions it left the replica **no context at all**; the answer is then the fixed "
          "\"No evidence was retrieved\" reply, with no model call.", "",
          "| questions | n | mode | F1 (repeat 0) | empty context |", "|---|---:|---|---:|---:|"]
    for name, qs in (("eq. 22 applied", on), ("of which eq. 22 left no context", empty)):
        for mode in ("baseline", "replica_no_eq22", "rrf"):
            sel = [r for r in one if r["mode"] == mode and r["qi"] in qs]
            L.append(f"| {name} | {len(qs)} | {LABEL[mode]} | {100*score(sel)['f1']:.2f} | "
                     f"{sum(not r['evidence'] for r in sel)} |")
    new = [r for r in rows if r["mode"] in NEW_MODES]
    L += ["", f"Malformed JSON replies among the new answers: {sum(r['malformed'] for r in new)} of {len(new)}."]
    L += footer
    out.write_text("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


def attribution_rerender(ids: list[str], cfg: dict, stored: Path, out: Path, repeats: int = 3) -> int:
    """The report again from the stored answers: no LLM call, the original run's cost footer kept."""
    items, subset = open_items(ids)
    rows = [r for r in map(json.loads, stored.read_text().splitlines()) if r["subset"]]
    rows += [json.loads(x) for x in out.with_suffix(".jsonl").read_text().splitlines()]
    gold = {j: gold_interval(items[j]["vid"], items[j]["q"])[0] is not None for j in subset}
    text = out.read_text()
    footer = text[text.index("\nLLM cost (this run):"):].rstrip("\n").split("\n")
    return attribution_report(rows, gold, cfg, cfg["models"]["llm"], repeats, out, footer)
