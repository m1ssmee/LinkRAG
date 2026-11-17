#!/usr/bin/env python3
"""Measure a candidate judge against the stored pilot01 verdicts, without the pilot01 corpus.

    LINKRAG_JUDGE_BACKEND=claude_agents_judge python scripts/eval/judge_agreement_offline.py --runs 1
    # -> queues prompts; answer them with scripts/agent_queue.py, then run again for the report

``judge_agreement.py`` re-runs verify_gold over the pilot01 index and its stored answer
cache. Neither is in a fresh clone (both are git-ignored, and so is the pilot01 media). The
stored verdict files, though, hold everything the judge saw: each unit's full text, the
question, the reference answer, and every stored source-run answer. This script rebuilds the
same judge prompts from them through verify_gold's own ``entail_unit`` and ``grade``, and
compares the candidate with the stored judges:

* **unit verdicts**: 147 units, kept or not. Cohen's kappa vs gpt-4.1-mini and vs gpt-5.4.
* **grades**: 100 source runs (audio / deck / paper / all), PASS or FAIL. Agreement and kappa.
* **type labels**: 25 questions, relabelled from the candidate's grades with verify_gold's
  ``relabel``.

**Bar (set 2026-11-17, before the first run):** the candidate may judge XFile if unit-level
kappa vs gpt-4.1-mini is at least 0.70. The LLM-LLM reference is 0.85, and the rejected NLI
judges reached 0.31 and 0.36 (DESIGN.md finding 11).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from sklearn.metrics import cohen_kappa_score

from linkrag.core import EvidenceUnit, Location, load_config
from linkrag.costs import cached_completer
from linkrag.eval.verify_gold import SourceRun, entail_unit, grade, relabel
from linkrag.generate.answer import judge_completer

STORED = {"gpt-4.1-mini": "reports/gold_verified_pilot01.json",
          "gpt-5.4": "reports/gold_verified_pilot01.judge-gpt54.json"}
BAR = 0.70


def unit_of(u: dict) -> EvidenceUnit:
    return EvidenceUnit(id=u["unit_id"], modality=u["modality"], content=u["text"],
                        source_file=u["location"].split(" ")[0], location=Location())


def as_bool(v) -> bool:
    return v is True or str(v).lower() == "true"


def kappa(a: list[bool], b: list[bool]) -> float:
    return float(cohen_kappa_score(a, b)) if len(set(a) | set(b)) > 1 else 1.0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--runs", type=int, default=1, help="judge calls per prompt (stored verdicts: majority of 3)")
    ap.add_argument("--cache", default="data/processed/judge_cache_offline")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    jcfg = cfg["eval"]["judge"]
    name = f"{jcfg.get('backend', 'default')}:{jcfg.get('model')}"
    inner = judge_completer(cfg, "llm")
    judge = cached_completer(inner, Path(args.cache) / str(jcfg.get("model")))
    stored = {k: json.loads(Path(p).read_text()) for k, p in STORED.items()}
    ref = stored["gpt-4.1-mini"]

    unit_rows, grade_rows, type_rows = [], [], []
    for qi, q in enumerate(ref):
        others = {k: {u["unit_id"]: u for u in v[qi]["units"]} for k, v in stored.items()}
        for u in q["units"]:
            v = entail_unit(q["question"], q["expected"], unit_of(u), judge, runs=args.runs)
            unit_rows.append((q["qid"], u["unit_id"], v.kept,
                              {k: as_bool(o[u["unit_id"]]["kept"]) for k, o in others.items()}))
        runs = {}
        for r in q["runs"]:
            passed, votes, reason, missing = grade(q["question"], q["expected"], r["answer"], judge, args.runs)
            runs[r["source"]] = SourceRun(r["source"], r["answer"], passed, votes, reason, missing)
            grade_rows.append((q["qid"], r["source"], passed, as_bool(r["passed"])))
        vtype, _status, _from = relabel(q["original_type"], runs)
        type_rows.append((q["qid"], vtype, q["verified_type"]))

    pending = getattr(inner, "pending", 0)
    if pending:
        print(f"{pending} judge calls deferred to the agent queue; answer them "
              f"(scripts/agent_queue.py), then run this again. No report written.")
        return 3

    lines = [f"# Judge agreement (offline): {name}, pilot01", "",
             f"Candidate judge `{name}`, {args.runs} run(s) per prompt, against the stored verdicts "
             "(majority of 3) in `reports/gold_verified_pilot01.json` (gpt-4.1-mini) and "
             "`.judge-gpt54.json` (gpt-5.4). Prompts rebuilt from the stored unit texts and answers with "
             "verify_gold's `entail_unit` / `grade` (`scripts/eval/judge_agreement_offline.py`). "
             f"Bar, set before the run: unit kappa vs gpt-4.1-mini >= {BAR}.", "",
             "| comparison | n | agreement | Cohen's kappa |", "|---|---|---|---|"]
    cand = [r[2] for r in unit_rows]
    for k in STORED:
        other = [r[3][k] for r in unit_rows]
        lines.append(f"| unit kept vs {k} | {len(cand)} | {sum(a == b for a, b in zip(cand, other))}/{len(cand)} "
                     f"| {kappa(cand, other):.2f} |")
    a41, a54 = [r[3]["gpt-4.1-mini"] for r in unit_rows], [r[3]["gpt-5.4"] for r in unit_rows]
    lines.append(f"| *reference: gpt-4.1-mini vs gpt-5.4* | {len(a41)} | {sum(x == y for x, y in zip(a41, a54))}/{len(a41)} "
                 f"| {kappa(a41, a54):.2f} |")
    g_c, g_s = [r[2] for r in grade_rows], [r[3] for r in grade_rows]
    lines.append(f"| grade PASS vs gpt-4.1-mini | {len(g_c)} | {sum(a == b for a, b in zip(g_c, g_s))}/{len(g_c)} "
                 f"| {kappa(g_c, g_s):.2f} |")
    t_ok = sum(a == b for _, a, b in type_rows)
    lines.append(f"| type label vs gpt-4.1-mini | {len(type_rows)} | {t_ok}/{len(type_rows)} | - |")
    k41 = kappa(cand, a41)
    lines += ["", f"**Verdict: {'PASSES' if k41 >= BAR else 'FAILS'} the bar** "
              f"(unit kappa vs gpt-4.1-mini {k41:.2f}, bar {BAR}).", "",
              "## Unit disagreements with gpt-4.1-mini", "",
              "| qid | unit | candidate | gpt-4.1-mini | gpt-5.4 |", "|---|---|---|---|---|"]
    lines += [f"| {q} | `{u}` | {c} | {s['gpt-4.1-mini']} | {s['gpt-5.4']} |"
              for q, u, c, s in unit_rows if c != s["gpt-4.1-mini"]]
    lines += ["", "## Type-label disagreements", "", "| qid | candidate | stored |", "|---|---|---|"]
    lines += [f"| {q} | {a} | {b} |" for q, a, b in type_rows if a != b]
    out = Path(args.out or f"results/judge_agreement_offline_{jcfg.get('backend', 'default')}.md")
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out}: unit kappa vs gpt-4.1-mini {k41:.2f} ({'pass' if k41 >= BAR else 'FAIL'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
