#!/usr/bin/env python3
"""Measure the nli-for-llm switch: agreement of two entailment runs on the same corpus.

Gold verification (`verify_gold.py` JSON): unit-level Cohen's kappa over every
(question, unit) pair both runs scored, plus question-level agreement on the
verified type and on kept/dropped. Redundancy (`redundancy.py` JSON): sentence-level
kappa per pair and the redundancy fractions side by side.

A reference pair (two LLM judges) gives the ceiling kappa to read the nli number
against: if two LLMs only agree at 0.6, nli at 0.55 is not a regression.

    python scripts/eval/compare_entailment.py \
        --gold reports/gold_verified_pilot01.json=llm:gpt-4.1-mini \
        --gold reports/gold_verified_pilot01.judge-gpt54.json=llm:gpt-5.4 \
        --gold results/nli_pilot01/gold_verified_pilot01.json=nli \
        --redundancy reports/redundancy_pilot01.json=llm:gpt-4.1-mini \
        --redundancy results/nli_pilot01/redundancy_pilot01.json=nli \
        --out results/nli_vs_llm_pilot01.md
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from itertools import combinations
from pathlib import Path

from linkrag.eval.redundancy import DEFAULT_PAIRS


def kappa(a: list[bool], b: list[bool]) -> float:
    n = len(a)
    if not n:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def confusion(a: list[bool], b: list[bool]) -> str:
    c = Counter(zip(a, b))
    return f"yes/yes {c[True, True]} · yes/no {c[True, False]} · no/yes {c[False, True]} · no/no {c[False, False]}"


def load_named(spec: str) -> tuple[str, object]:
    path, _, name = spec.partition("=")
    return name or Path(path).stem, json.loads(Path(path).read_text())


def reason(a: dict, b: dict, na: str, nb: str) -> str:
    """One line on why two verification runs typed a question differently."""
    kept = lambda q: sum(u["kept"] for u in q["units"])
    passed = lambda q: sorted(r["source"] for r in q["runs"] if r["passed"])
    ka, kb, n = kept(a), kept(b), len(b["units"])
    pa, pb = passed(a), passed(b)
    if b["verified_type"] is None and "full corpus" in b["status"]:
        fails = next((r["reason"] for r in b["runs"] if r["source"] == "all"), "")
        return (f"{nb} grader failed the full-corpus answer ({fails.removeprefix('nli: ')}); "
                f"{na} passed it from {', '.join(pa) or 'no source'}")
    if b["verified_type"] is None:
        return f"no proposed unit survived {nb} entailment (kept {kb}/{n}; {na} kept {ka})"
    if a["verified_type"] is None:
        return f"{na} dropped it ({a['status']}); {nb} kept {kb}/{n} units"
    return (f"answerable from {', '.join(pa) or '—'} under {na} vs {', '.join(pb) or '—'} under {nb}; "
            f"units kept {ka} vs {kb} of {n}")


def gold_section(runs: list[tuple[str, list]]) -> list[str]:
    L = ["## Gold verification", "",
         "Unit level: every (question, candidate unit) both runs scored; `kept` is the entailment "
         "verdict. Question level: verified type and kept/dropped status.", "",
         "| A | B | unit pairs | unit agreement | unit κ | confusion (A/B) | type agreement | kept agreement |",
         "|---|---|---:|---:|---:|---|---:|---:|"]
    diffs: list[str] = []
    for (na, ra), (nb, rb) in combinations(runs, 2):
        ua = {(q["qid"], u["unit_id"]): u["kept"] for q in ra for u in q["units"]}
        ub = {(q["qid"], u["unit_id"]): u["kept"] for q in rb for u in q["units"]}
        keys = sorted(set(ua) & set(ub))
        a, b = [ua[k] for k in keys], [ub[k] for k in keys]
        qa, qb = {q["qid"]: q for q in ra}, {q["qid"]: q for q in rb}
        qids = sorted(set(qa) & set(qb), key=lambda s: (len(s), s))
        types = sum(qa[q]["verified_type"] == qb[q]["verified_type"] for q in qids)
        kept = sum(("drop" in qa[q]["status"]) == ("drop" in qb[q]["status"]) for q in qids)
        agree = sum(x == y for x, y in zip(a, b))
        L.append(f"| {na} | {nb} | {len(keys)} | {agree}/{len(keys)} = {agree / max(len(keys), 1):.1%} | "
                 f"**{kappa(a, b):.2f}** | {confusion(a, b)} | {types}/{len(qids)} | {kept}/{len(qids)} |")
        if "nli" in (na, nb) and not diffs:
            rows = [q for q in qids if qa[q]["verified_type"] != qb[q]["verified_type"]]
            diffs = ["", f"### Questions whose type differs ({na} vs {nb}): {len(rows)} of {len(qids)}", "",
                     "Reason is read off the two runs: which full-context answers each grader passed, and how many "
                     "proposed gold units each entailment check kept.", "",
                     f"| qid | {na} | {nb} | reason |", "|---|---|---|---|",
                     *[f"| {q} | {qa[q]['verified_type']} | {qb[q]['verified_type']} | {reason(qa[q], qb[q], na, nb)} |" for q in rows]]
    for name, r in runs:
        c = Counter(str(q["verified_type"]) for q in r)
        L.append("")
        L.append(f"- **{name}** types: " + ", ".join(f"{t} {n}" for t, n in sorted(c.items()))
                 + f" (sum {sum(c.values())} = {len(r)} questions)")
    return L + diffs


def redundancy_section(runs: list[tuple[str, list]]) -> list[str]:
    L = ["", "## Redundancy", "",
         "Sentence level: the same source sentence against the same target role; kappa on the "
         "entailed verdict. Fractions are the redundancy each run reports.", "",
         "| pair | sentences | " + " | ".join(f"{n} redundancy" for n, _ in runs) + " | agreement | κ |",
         "|---|---:|" + "---:|" * len(runs) + "---:|---:|"]
    idx = [{(v["source"], v["target"], v["sentence"]): v["entailed"] for v in r} for _, r in runs]
    present = {(s, t) for s, t, _ in idx[0]}
    pairs = [p for p in DEFAULT_PAIRS if p in present] + sorted(present - set(DEFAULT_PAIRS))
    tot_a: list[bool] = []
    tot_b: list[bool] = []
    for s, t in pairs:
        keys = [k for k in idx[0] if k[:2] == (s, t) and all(k in i for i in idx)]
        fr = [sum(i[k] for k in keys) / max(len(keys), 1) for i in idx]
        a, b = [idx[0][k] for k in keys], [idx[-1][k] for k in keys]
        tot_a += a
        tot_b += b
        agree = sum(x == y for x, y in zip(a, b))
        L.append(f"| {s}->{t} | {len(keys)} | " + " | ".join(f"{f:.1%}" for f in fr)
                 + f" | {agree / max(len(keys), 1):.1%} | **{kappa(a, b):.2f}** |")
    agree = sum(x == y for x, y in zip(tot_a, tot_b))
    L.append(f"| **overall** | {len(tot_a)} | " + " | ".join(f"{sum(x) / max(len(x), 1):.1%}" for x in (tot_a, tot_b))
             + f" | {agree / max(len(tot_a), 1):.1%} | **{kappa(tot_a, tot_b):.2f}** |")
    L.append("")
    L.append(f"Confusion ({runs[0][0]}/{runs[-1][0]}), all pairs: {confusion(tot_a, tot_b)}.")
    return L


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", action="append", default=[], help="path=name, repeatable; put the nli run last")
    ap.add_argument("--redundancy", action="append", default=[], help="path=name; first vs last are compared")
    ap.add_argument("--title", default="Entailment backend agreement")
    ap.add_argument("--preamble", default=None, help="markdown file inserted after the title (findings, provenance)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    L = [f"# {args.title}", ""]
    if args.preamble:
        L += [Path(args.preamble).read_text().rstrip(), ""]
    L += ["Tables below generated by `scripts/eval/compare_entailment.py`.", ""]
    if args.gold:
        L += gold_section([load_named(s) for s in args.gold])
    if args.redundancy:
        L += redundancy_section([load_named(s) for s in args.redundancy])
    Path(args.out).write_text("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
