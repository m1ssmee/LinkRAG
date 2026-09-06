#!/usr/bin/env python3
"""DP vs naive alignment against the ear-labelled slide timeline.

Labels: segment_id,start,end,true_slide,section,ambiguous
(`data/labels/pilot01/pilot01_alignment_labels.csv`, built by make_alignment_labels.py).

Reported per condition: exact accuracy, +/-1 accuracy, backward steps, and slides
covered out of the showable count. The Q&A section is scored separately -- it has no
true slide, so accuracy is undefined there; what matters is whether a method stays put
or wanders.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from linkrag.core import load_config
from linkrag.link.align import align_monotonic, align_naive


def load_labels(path: str | Path) -> list[dict]:
    rows = []
    for row in csv.DictReader(open(path, newline="")):
        rows.append({
            "segment_id": row["segment_id"],
            "start": float(row["start"]),
            "end": float(row["end"]),
            "true_slide": None if row["true_slide"] in ("none", "") else int(row["true_slide"]),
            "section": row.get("section", "talk"),
            "ambiguous": str(row.get("ambiguous", "")).lower() == "true",
        })
    return rows


def segment_index(labels, starts, ends) -> dict[str, int]:
    """Map each label to the audio unit it overlaps most."""
    out = {}
    for row in labels:
        overlaps = [(min(ends[i], row["end"]) - max(starts[i], row["start"]), i)
                    for i in range(len(starts))]
        best, idx = max(overlaps)
        if best > 0:
            out[row["segment_id"]] = idx
    return out


def score(path, labels, index_of, pages, tolerance=0):
    hits = total = 0
    misses = []
    for row in labels:
        i = index_of.get(row["segment_id"])
        if i is None or row["true_slide"] is None:
            continue
        predicted = int(pages[path[i]])
        total += 1
        if abs(predicted - row["true_slide"]) <= tolerance:
            hits += 1
        else:
            misses.append((row["segment_id"], row["true_slide"], predicted))
    return hits, total, misses


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/labels/pilot01/pilot01_alignment_labels.csv")
    ap.add_argument("--npz", default="data/processed/links.npz")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--showable", type=int, default=23)
    ap.add_argument("--mu", type=float, default=None, help="override start_prior_mu")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)["link"]["align"]
    mu = args.mu if args.mu is not None else cfg.get("start_prior_mu", 0.0)
    data = np.load(args.npz, allow_pickle=True)
    S, starts, ends, pages = data["similarity"], data["audio_start"], data["audio_end"], data["slide_pages"]

    labels = load_labels(args.labels)
    index_of = segment_index(labels, starts, ends)
    talk = [r for r in labels if r["section"] == "talk" and r["true_slide"] is not None]
    firm = [r for r in talk if not r["ambiguous"]]
    qa = [r for r in labels if r["section"] == "qa"]

    paths = {
        "monotonic DP (ours)": align_monotonic(
            S, jump_penalty=cfg["jump_penalty"], skip_penalty=cfg["skip_penalty"],
            back_penalty=cfg["back_penalty"], max_back=cfg["max_back"],
            start_prior_mu=mu),
        "naive argmax (P2-style)": align_naive(S),
    }

    print(f"labels: {len(labels)} segments — talk {len(talk)} (non-ambiguous {len(firm)}), "
          f"qa {len(qa)} · start_prior_mu={mu}")
    for subset_name, subset in (("TALK (all)", talk), ("TALK (non-ambiguous)", firm)):
        print(f"\n=== {subset_name}, n={len(subset)} ===")
        print(f"  {'method':<26}{'exact':>10}{'±1':>10}{'back':>7}{'slides':>9}")
        for name, path in paths.items():
            e_hits, e_tot, misses = score(path, subset, index_of, pages, 0)
            t_hits, t_tot, _ = score(path, subset, index_of, pages, 1)
            idxs = [index_of[r["segment_id"]] for r in subset if r["segment_id"] in index_of]
            seq = [path[i] for i in sorted(idxs)]
            back = sum(1 for a, b in zip(seq, seq[1:]) if b < a)
            covered = len({int(pages[j]) for j in seq})
            print(f"  {name:<26}{e_hits}/{e_tot} = {100*e_hits/max(e_tot,1):5.1f}%"
                  f"{100*t_hits/max(t_tot,1):9.1f}%{back:7}{covered:>5}/{args.showable}")
        if subset_name.startswith("TALK (all)"):
            _, _, dp_misses = score(paths["monotonic DP (ours)"], subset, index_of, pages, 0)
            print(f"  DP misses ({len(dp_misses)}): " +
                  ", ".join(f"{m[0].split(':')[-1]} true p{m[1]}→p{m[2]}" for m in dp_misses[:12]))

    print(f"\n=== Q&A, n={len(qa)} (no true slide; expect DP to hold, naive to wander) ===")
    for name, path in paths.items():
        assigned = [int(pages[path[index_of[r["segment_id"]]]]) for r in qa
                    if r["segment_id"] in index_of]
        distinct = sorted(set(assigned))
        print(f"  {name:<26} assigned {assigned}")
        print(f"  {'':<26} {len(distinct)} distinct slide(s): {distinct}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
