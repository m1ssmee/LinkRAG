"""Closed-book control on LectQA-Vid: the open-ended subset answered with no context and with
the full transcript, under one prompt that allows the model's own knowledge. It measures how
much of the benchmark an answerer gets without the video. Paid: the only step here that is."""

from __future__ import annotations

import json
import statistics as st
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from linkrag.costs import BillingRefused, cached_completer, price_for, record_run, usage_cost
from linkrag.generate.answer import http_completer
from linkrag.index import Index

from lectqa.common import LEVELS, PROCESSED, THEIRS
from lectqa.open_ended import OPEN_SEED, open_items
from lectqa.recheck import cells

MODES = ("closed_book", "transcript")
LABEL = {"closed_book": "closed book (no context)", "transcript": "full transcript"}
SYSTEM = ("You answer a question about a short computer-science lecture video in one or two sentences. "
          "Use the transcript if one is given and your own knowledge otherwise. Always give your best answer: "
          "never say that you cannot answer or that information is missing.")
REFUSAL = ("cannot", "can't", "not provided", "no transcript", "unable to", "don't have", "do not have")


def prompt(question: str, transcript: str | None) -> str:
    return f"Transcript:\n{transcript if transcript else '(not provided)'}\n\nQuestion: {question}\n\nAnswer:"


def closed_book(ids: list[str], cfg: dict, out: Path, *, max_cost: float, dry_run: bool, repeats: int = 3) -> int:
    import tiktoken
    llm = {**cfg["models"]["llm"], "max_tokens": 400}      # room for a reasoning model's hidden tokens
    price = price_for(str(llm["model"]), cfg["models"].get("pricing"))
    items, subset = open_items(ids)
    qis = sorted(subset)
    text = {}
    for v in {items[j]["vid"] for j in qis}:
        audio = sorted((u for u in Index.load(PROCESSED / v / "index").units if u.modality == "audio"),
                       key=lambda u: u.location.start_s)
        text[v] = " ".join(u.content for u in audio)
    user = {(j, m): prompt(items[j]["question"], text[items[j]["vid"]] if m == "transcript" else None)
            for j in qis for m in MODES}
    tasks = [(j, m, r) for r in range(repeats) for j in qis for m in MODES]     # repeat-major, as open_modes
    enc = tiktoken.get_encoding("o200k_base")
    tok_in = sum(len(enc.encode(SYSTEM + user[j, m])) + 8 for j, m, _ in tasks)
    tok_out = 60 * len(tasks)
    est = tok_in / 1e6 * price["input_per_m"] + tok_out / 1e6 * price["output_per_m"] if price else 0.0   # free backend
    print(f"{len(qis)} questions x {len(MODES)} conditions x {repeats} repeats = {len(tasks)} answers · "
          f"~{tok_in:,} in / ~{tok_out:,} out · estimated ${est:.2f} if nothing is cached (cap ${max_cost:.2f})")
    if dry_run:
        return 0
    if est > max_cost:
        raise SystemExit(f"estimated ${est:.2f} exceeds --max-cost {max_cost}; nothing sent")
    complete = cached_completer(http_completer(llm), PROCESSED / "llm_cache" / f"{llm['model']}-closedbook")

    def one(task):
        j, m, r = task
        try:        # keyed by question, condition and repeat: a replay is deterministic and free
            return {"vid": items[j]["vid"], "qi": j, "level": items[j]["level"], "mode": m, "rep": r,
                    "answer": complete(SYSTEM, user[j, m], key=(f"{j}:{m}", r)).strip(), "ref": items[j]["ref"]}
        except BillingRefused:
            return None

    with ThreadPoolExecutor(max_workers=8) as ex:
        rows = [r for r in ex.map(one, tasks) if r]
    footer = record_run("scripts/adapters/lectqa_vid.py", "lectqa_vid closed-book control",
                        [(str(llm["model"]), complete.usage)], cfg["models"].get("pricing"))
    print(f"{len(rows)}/{len(tasks)} answered · ${usage_cost(complete.usage, price) or 0:.3f}")
    answerer = f"`{llm['model']}` at temperature {llm.get('temperature')}, one call per question"
    return report(rows, items, qis, cfg, out, answerer=answerer, repeats=repeats, planned=len(tasks), footer=footer)


def from_agents(ids: list[str], cfg: dict, out: Path, answers: Path, answerer: str) -> int:
    """The same report from answers written by agents: `answers` holds out_closed_*.json and
    out_transcript_*.json, each {question id: answer}. No model is called here."""
    items, subset = open_items(ids)
    qis = sorted(subset)
    rows = []
    for mode, stem in (("closed_book", "closed"), ("transcript", "transcript")):
        got = {int(k): str(v).strip() for f in sorted(answers.glob(f"out_{stem}_*.json")) for k, v in json.loads(f.read_text()).items()}
        missing = [j for j in qis if not got.get(j)]
        if missing or set(got) - set(qis):
            raise SystemExit(f"{mode}: {len(missing)} of {len(qis)} answers missing, {len(set(got) - set(qis))} unknown ids")
        rows += [{"vid": items[j]["vid"], "qi": j, "level": items[j]["level"], "mode": mode, "rep": 0, "answer": got[j],
                  "ref": items[j]["ref"]} for j in qis]
    return report(rows, items, qis, cfg, out, answerer=answerer, repeats=1, planned=len(rows),
                  footer=["", "LLM cost (this run): none through the project's API ledger; the answers came from agents."],
                  caveats=["- **Agents answer in batches, not one call per question.** Each agent read one input file "
                           "and answered 50 to 69 questions in one pass, so a question's answer could be influenced by "
                           "its neighbours. Closed-book batches were dealt so that a video's questions fall in different "
                           "batches where possible. Sampling settings are the agent runtime's defaults: the run is not "
                           "replayable from this repository, only re-scorable from the stored answers."])


def report(rows: list[dict], items: list[dict], qis: list[int], cfg: dict, out: Path, *, answerer: str, repeats: int,
           planned: int, footer: list[str], caveats: tuple | list = ()) -> int:
    from linkrag.eval import lectqa_metrics as LM
    stopped = len(rows) < planned
    if stopped:                         # a stopped run never replaces a complete report
        out = out.with_name(f"{out.stem}_partial{out.suffix}")
    out.with_suffix(".jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    def score(sel):
        return LM.score_open([r["answer"] for r in sel], [r["ref"] for r in sel], device=cfg["device"])

    def pick(m, lv, r=None):
        return [x for x in rows if x["mode"] == m and (lv == "overall" or x["level"] == lv) and (r is None or x["rep"] == r)]

    stored = [r for r in map(json.loads, (out.parent / "lectqa_open_modes.jsonl").read_text().splitlines())
              if r["subset"] and r["mode"] == "full_context"]
    echo = {j: 100 * LM.token_prf(items[j]["question"], items[j]["ref"])[2] for j in qis}
    L = ["# LectQA-Vid — closed-book control: how much is answered without the video", "",
         *(["**Not reportable: stopped at the budget.** "
            f"{len(rows)} of {planned} answers ran; repeats below are unequal.", ""] if stopped else []),
         f"The open-ended subset ({len(qis)} questions, seed `{OPEN_SEED}`), answered by {answerer}, under one prompt, "
         f"{repeats} run(s): once with no context and once with the video's "
         "full transcript. Metrics as in `lectqa_open_modes.md`, in % (mean ± std over repeats when there is more than one).", "",
         "## Caveats first", "",
         "- **A new prompt is a new condition.** This prompt allows the model's own knowledge and forbids refusing; the "
         "stored modes of `lectqa_open_modes.md` use an evidence-only JSON prompt that must abstain. The two "
         "full-transcript rows below differ in the prompt, so only rows from this run are compared with each other.",
         "- **Closed book is not no information.** The question text is in the prompt, and the references restate "
         "it (the echo row).",
         "- One answering model. A model with more or less background knowledge would move the closed-book row.",
         *(["- **One run: direction only** (reporting convention)."] if repeats == 1 else []), *caveats,
         "- Intervals are 95 % bootstrap intervals over videos (10,000 resamples, seed 20260923).", "",
         "The prompt (system): *" + SYSTEM + "*", "",
         "## Results", "",
         "| level | condition | n | F1 | Sim | refusals |", "|---|---|---:|---:|---:|---:|"]
    for lv in (*LEVELS, "overall"):
        for m in MODES:
            per = [score(x) for r in range(repeats) if (x := pick(m, lv, r))]
            ref = sum(any(w in x["answer"].lower() for w in REFUSAL) for x in pick(m, lv, 0))
            cell = lambda k: (f"{100 * st.mean(p[k] for p in per):.2f}"                                      # noqa: E731
                              + (f" ± {100 * st.pstdev(p[k] for p in per):.2f}" if len(per) > 1 else ""))
            L.append(f"| {lv} | {LABEL[m]} | {per[0]['n']} | {cell('f1')} | {cell('sim')} | {ref} |")
        js = [j for j in qis if lv == "overall" or items[j]["level"] == lv]
        s = score([r for r in stored if r["rep"] == 0 and (lv == "overall" or r["level"] == lv)])
        L += [f"| {lv} | *question echoed as the answer* | {len(js)} | {np.mean([echo[j] for j in js]):.2f} | — | — |",
              f"| {lv} | *full transcript, evidence-only prompt (stored, repeat 0)* | {s['n']} | {100 * s['f1']:.2f} | {100 * s['sim']:.2f} | — |",
              f"| {lv} | *T1 Table 4 (published; unnamed answerer)* | — | {THEIRS['open'][lv]['f1']:.2f} | {THEIRS['open'][lv]['sim']:.2f} | — |"]
    f1 = {m: {j: st.mean(100 * LM.token_prf(r["answer"], r["ref"])[2] for r in rows if r["mode"] == m and r["qi"] == j)
              for j in qis if any(r["mode"] == m and r["qi"] == j for r in rows)} for m in MODES}
    both = [j for j in qis if j in f1["closed_book"] and j in f1["transcript"]]
    vids = np.array([items[j]["vid"] for j in both])
    L += ["", "## Paired differences (token F1 points)", "",
          "| difference | level | by question | by video |", "|---|---|---:|---:|"]
    for name, a, b in (("full transcript − closed book (what the video adds)", f1["transcript"], f1["closed_book"]),
                       ("closed book − question echo (what the model's own knowledge adds)", f1["closed_book"], echo)):
        for lv in ("overall", *LEVELS):
            k = np.array([lv == "overall" or items[j]["level"] == lv for j in both])
            L.append(f"| {name} | {lv} | {cells(np.array([a[j] - b[j] for j in both])[k], vids[k])} |")
    cb, tr = np.array([f1["closed_book"][j] for j in both]), np.array([f1["transcript"][j] for j in both])
    L += ["", f"Share of the full-transcript F1 that the closed-book answer already reaches: "
          f"**{cb.mean() / tr.mean():.0%}** ({cb.mean():.2f} of {tr.mean():.2f}). "
          f"Questions where closed book scores at least as high as the transcript: {int((cb >= tr).sum())} of {len(both)}.",
          *footer]
    out.write_text("\n".join(L) + "\n")
    print("\n".join(L))
    return 0
