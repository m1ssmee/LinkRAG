#!/usr/bin/env python3
"""Sampled human audit of the automated gold verification.

    python scripts/eval/audit_sample.py                      # write the sheet
    python scripts/eval/audit_sample.py --score              # after 'agree' is filled

Stratified by question type: 10% of the unit verdicts per type, minimum 5 per type
(all of them if a type has fewer). Kept and dropped verdicts are both sampled, so the
audit bounds false accepts *and* false rejects. The sheet has one row per unit with
a blank `agree` column; fill it with y / n and re-run with --score.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

FIELDS = ["sample_id", "auditor", "qid", "type", "verdict", "question", "expected_answer", "unit_id",
          "location", "unit_text", "judge_span_or_reason", "agree", "auditor_note"]


def stratified_sample(rows: list[dict], *, fraction: float, minimum: int, seed: int) -> list[dict]:
    by_type: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_type[r["type"]].append(r)
    rng = random.Random(seed)
    out = []
    for t in sorted(by_type):
        pool = by_type[t]
        n = min(len(pool), max(minimum, round(fraction * len(pool))))
        out += rng.sample(pool, n)
    return out


def top_up(sample: list[dict], rows: list[dict], *, prefixes: list[str], minimum: int, seed: int) -> list[dict]:
    """Add rows until each qid prefix (topic) has at least `minimum` in the sample -- for topics
    whose human review took Claude's recommendation unread, so the audit reaches them."""
    rng = random.Random(seed)
    taken = {id(r) for r in sample}
    for pre in prefixes:
        have = sum(r["qid"].startswith(pre) for r in sample)
        pool = [r for r in rows if r["qid"].startswith(pre) and id(r) not in taken]
        extra = rng.sample(pool, min(len(pool), max(0, minimum - have)))
        taken |= {id(r) for r in extra}
        sample = sample + extra
    return sample


def unit_rows(verdicts: list[dict]) -> list[dict]:
    rows = []
    for v in verdicts:
        qtype = v["verified_type"] or f"dropped({v['original_type']})"
        for u in v["units"]:
            rows.append({
                "qid": v["qid"], "type": qtype,
                "verdict": "kept" if u["kept"] else "dropped",
                "question": v["question"], "expected_answer": v["expected"],
                "unit_id": u["unit_id"], "location": u["location"],
                "unit_text": " ".join(u["text"].split()),
                "judge_span_or_reason": u["span"] if u["kept"] else u["reason"],
                "auditor": "", "agree": "", "auditor_note": "",
            })
    return rows


def score(path: Path) -> int:
    rows = list(csv.DictReader(path.open()))
    filled = [r for r in rows if r["agree"].strip().lower() in ("y", "n", "yes", "no")]
    if not filled:
        print(f"{path}: no rows filled yet (expects y/n in the 'agree' column)")
        return 1
    agree = lambda r: r["agree"].strip().lower() in ("y", "yes")
    print(f"{len(filled)}/{len(rows)} rows filled · overall agreement "
          f"{sum(map(agree, filled))}/{len(filled)} = {sum(map(agree, filled)) / len(filled):.0%}")
    by = defaultdict(list)
    for r in filled:
        by[(r["type"], r["verdict"])].append(agree(r))
    print("\n| type | verdict | n | agreement |\n|---|---|---:|---:|")
    for (t, vd), xs in sorted(by.items()):
        print(f"| {t} | {vd} | {len(xs)} | {sum(xs) / len(xs):.0%} |")
    dis = [r for r in filled if not agree(r)]
    if dis:
        print("\nDisagreements:")
        for r in dis:
            print(f"- {r['sample_id']} {r['qid']} `{r['unit_id']}` judged {r['verdict']}: "
                  f"{r['auditor_note'] or '(no note)'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verdicts", default="reports/gold_verified_pilot01.json")
    ap.add_argument("--out", default="reports/audit_sheet_pilot01.csv")
    ap.add_argument("--fraction", type=float, default=0.10)
    ap.add_argument("--min-per-type", type=int, default=5)
    ap.add_argument("--seed", type=int, default=20260920)
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--min-per-prefix", default="",
                    help="'GP,NF,MD:5' -- at least 5 sampled rows from each of these qid prefixes")
    ap.add_argument("--assign", default="",
                    help="'alice:RL,GP;bob:PS' -- fill the auditor column by qid prefix")
    args = ap.parse_args(argv)

    if args.score:
        return score(Path(args.out))

    verdicts = json.loads(Path(args.verdicts).read_text())
    rows = unit_rows(verdicts)
    sample = stratified_sample(rows, fraction=args.fraction, minimum=args.min_per_type, seed=args.seed)
    if args.min_per_prefix:
        pres, n = args.min_per_prefix.rsplit(":", 1)
        sample = top_up(sample, rows, prefixes=pres.split(","), minimum=int(n), seed=args.seed)
    who = {pre: name for part in filter(None, args.assign.split(";"))
           for name, pres in [part.split(":")] for pre in pres.split(",")}
    for r in sample:
        r["auditor"] = next((n for pre, n in who.items() if r["qid"].startswith(pre)), "")
    for i, r in enumerate(sample, 1):
        r["sample_id"] = f"S{i:03d}"
    out = Path(args.out)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(sample)
    print(f"{len(sample)} of {len(rows)} unit verdicts sampled "
          f"({args.fraction:.0%}, min {args.min_per_type} per type) -> {out}")
    from collections import Counter
    for (t, v), n in sorted(Counter((r["type"], r["verdict"]) for r in sample).items()):
        print(f"  {t:<28} {v:<8} {n}")
    print("Fill the 'agree' column with y/n, then: python scripts/eval/audit_sample.py --score")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
