#!/usr/bin/env python3
"""Accuracy of monotonic DP vs naive argmax against a hand-labelled CSV.

CSV columns: audio_start,audio_end,true_slide   (true_slide is a 1-based PDF page)

    python scripts/eval_alignment.py --labels data/labels/pilot01/alignment_labels.csv

A label matches an audio unit when their time spans overlap; a unit covered by
several labels is scored against each. Accuracy is over labels, not units, so
denser labelling in one part of the talk does not silently reweight the result.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from linkrag.core import load_config
from linkrag.link.align import align_monotonic, align_naive


def load_labels(path: str | Path) -> list[tuple[float, float, int]]:
    rows = []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("audio_start", "").strip().startswith("#"):
                continue
            rows.append((float(row["audio_start"]), float(row["audio_end"]),
                         int(row["true_slide"])))
    return rows


def score(path: list[int], labels, starts, ends, pages) -> tuple[int, int, list]:
    hits, total, misses = 0, 0, []
    for ls, le, true_page in labels:
        overlapping = [i for i in range(len(starts)) if starts[i] < le and ends[i] > ls]
        if not overlapping:
            continue
        # the unit with the greatest overlap represents this label
        best = max(overlapping, key=lambda i: min(ends[i], le) - max(starts[i], ls))
        predicted = int(pages[path[best]])
        total += 1
        if predicted == true_page:
            hits += 1
        else:
            misses.append((ls, le, true_page, predicted))
    return hits, total, misses


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--npz", default="data/processed/links.npz")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--tolerance", type=int, default=0,
                    help="count |predicted-true| <= tolerance as correct")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)["link"]["align"]
    data = np.load(args.npz, allow_pickle=True)
    S = data["similarity"]
    starts, ends, pages = data["audio_start"], data["audio_end"], data["slide_pages"]
    labels = load_labels(args.labels)
    if not labels:
        raise SystemExit(f"no labels in {args.labels}")

    paths = {
        "monotonic DP (ours)": align_monotonic(
            S, jump_penalty=cfg["jump_penalty"], skip_penalty=cfg["skip_penalty"],
            back_penalty=cfg["back_penalty"], max_back=cfg["max_back"]),
        "naive argmax (P2-style)": align_naive(S),
    }

    print(f"{len(labels)} labels · {S.shape[0]} audio units · {S.shape[1]} slides"
          + (f" · tolerance ±{args.tolerance}" if args.tolerance else ""))
    print()
    results = {}
    for name, path in paths.items():
        hits, total, misses = score(path, labels, starts, ends, pages)
        if args.tolerance:
            hits = sum(1 for ls, le, tp, pp in misses if abs(pp - tp) <= args.tolerance) + hits
            misses = [m for m in misses if abs(m[3] - m[2]) > args.tolerance]
        acc = hits / total if total else 0.0
        results[name] = acc
        print(f"{name:26} {hits:3}/{total:3} = {acc:6.1%}")
        for ls, le, tp, pp in misses[:8]:
            print(f"    miss {ls:7.1f}-{le:7.1f}s  true p{tp:<3} predicted p{pp}")
    print()
    dp, naive = results["monotonic DP (ours)"], results["naive argmax (P2-style)"]
    print(f"delta (DP - naive): {dp - naive:+.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
