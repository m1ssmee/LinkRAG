#!/usr/bin/env python3
"""Similarity heatmap with the chosen alignment path overlaid.

    python scripts/plot_alignment.py                       # reads the .npz build_links wrote
    python scripts/plot_alignment.py --overlay-naive       # both paths on one figure

Saves to reports/alignment_<audio stem>.png.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from linkrag.link.align import align_monotonic, align_naive


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="data/processed/links.npz")
    ap.add_argument("--out", default=None)
    ap.add_argument("--overlay-naive", action="store_true",
                    help="also draw the per-segment argmax path for comparison")
    ap.add_argument("--title", default=None)
    args = ap.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")  # headless: this runs in a terminal, not a notebook
    import matplotlib.pyplot as plt

    data = np.load(args.npz, allow_pickle=True)
    S, path = data["similarity"], list(data["path"])
    audio_ids, slide_pages = data["audio_ids"], data["slide_pages"]
    stem = str(audio_ids[0]).split(":")[0] if len(audio_ids) else "alignment"
    out = Path(args.out or f"reports/alignment_{stem}.png")
    out.parent.mkdir(parents=True, exist_ok=True)

    n, m = S.shape
    fig, ax = plt.subplots(figsize=(max(7.0, m * 0.42), max(5.0, n * 0.20)))
    im = ax.imshow(S, aspect="auto", origin="upper", cmap="magma",
                   interpolation="nearest")
    fig.colorbar(im, ax=ax, label="similarity  $S_{ij}$", shrink=0.8)

    ys = np.arange(n)
    if args.overlay_naive:
        naive = align_naive(S)
        ax.plot(naive, ys, color="#4FC3F7", lw=1.0, ls="--", marker="o", ms=2.6,
                alpha=0.85, label="naive argmax (P2-style)")
    ax.plot(path, ys, color="#00E676", lw=1.6, marker="o", ms=3.4,
            label="monotonic DP (ours)")

    # mark the back-jumps: the relaxation is the interesting part of the path
    backs = [(j, i) for i, (a, j) in enumerate(zip(path, path[1:]), start=1) if j < a]
    if backs:
        ax.scatter([j for j, _ in backs], [i for _, i in backs], s=90,
                   facecolors="none", edgecolors="#FFD54F", lw=1.6,
                   label=f"back-jump ({len(backs)})", zorder=5)

    ax.set_xlabel("slide (page)")
    ax.set_ylabel("audio segment (time order) →")
    step = max(1, m // 20)
    ax.set_xticks(range(0, m, step))
    ax.set_xticklabels([str(slide_pages[j]) for j in range(0, m, step)], fontsize=8)
    ax.set_yticks(range(0, n, max(1, n // 25)))
    ax.set_title(args.title or f"Audio-to-slide alignment — {stem}  (n={n}, m={m})")
    ax.legend(loc="lower right", fontsize=8, framealpha=0.85)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
