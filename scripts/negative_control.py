#!/usr/bin/env python3
"""Negative control for the relatedness gate, and its false-rejection rate.

    python scripts/negative_control.py              # pilot01 audio + unrelated deck -> link counts
    python scripts/negative_control.py --mavils     # gate on 20 related MaViLS pairs -> false rejections

Control corpus: data/raw/control/*.pdf (a deck that has nothing to do with the talk;
the shipped one is MaViLS's MIT psychology lecture) + pilot01's frozen transcript.
Expected: zero cross-file links (audio_slide, cross-document figure_text, deictic);
within-document links are allowed and reported.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from linkrag.core import load_config, setup_logging
from linkrag.link.align import align_monotonic, load_links, relatedness_gate

CONTROL = Path("data/raw/control")
OUT = Path("reports/relatedness_gate.md")


def control(cfg: dict, args) -> list[str]:
    pdfs = sorted(CONTROL.glob("*.pdf"))
    audio = Path("data/raw/pilot01/hsieh.mp3")
    if not pdfs or not audio.exists():
        raise SystemExit(f"need {CONTROL}/*.pdf and {audio}")
    index_dir = Path("data/processed/control/index")
    links_path = Path("data/processed/control/links.jsonl")
    if args.reingest or not index_dir.exists():
        rc = subprocess.call([sys.executable, "scripts/ingest.py", *map(str, pdfs), str(audio),
                              "--config", args.config, "--out", str(index_dir)])
        if rc not in (0, 2):
            raise SystemExit(rc)
    rc = subprocess.call([sys.executable, "scripts/build_links.py", "--config", args.config,
                          "--index", str(index_dir), "--links", str(links_path)])
    if rc != 0:
        raise SystemExit(rc)
    links = load_links(links_path, gated=False)
    meta = load_links.last_meta
    deck = {p.name for p in pdfs}
    by_type = Counter(l.link_type for l in links)
    cross = Counter()
    for l in links:
        a, b = l.src_id.split(":")[0], l.dst_id.split(":")[0]
        if a != b:
            cross[l.link_type] += 1
    L = ["## Negative control — pilot01 audio × unrelated deck", "",
         f"Audio: `{audio.name}` (frozen transcript, 51 segments). Deck: {', '.join(f'`{n}`' for n in deck)}. "
         f"Gate: `align.relatedness_z` = {cfg['link']['align'].get('relatedness_z')}.", "",
         "| link type | total | cross-file |", "|---|---:|---:|"]
    for t in sorted(set(by_type) | {"audio_slide", "figure_text", "deictic", "same_slide"}):
        L.append(f"| {t} | {by_type.get(t, 0)} | {cross.get(t, 0)} |")
    L.append(f"| **all** | {sum(by_type.values())} | **{sum(cross.values())}** |")
    L += ["", f"Pairs recorded as unrelated in `links.jsonl` `_meta`: "
          + (json.dumps(meta.get("unrelated_pairs"), indent=None) if meta.get("unrelated_pairs") else "none") + "",
          "", f"Expected zero cross-file links: **{'PASS' if sum(cross.values()) == 0 else 'FAIL'}**.", ""]
    return L


def mavils_false_rejections(cfg: dict, args) -> list[str]:
    import importlib.util
    spec = importlib.util.spec_from_file_location("mavils_adapter", "scripts/adapters/mavils.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    a = cfg["link"]["align"]
    z = float(a.get("relatedness_z") or 2.0)
    dp = lambda S: align_monotonic(S, jump_penalty=a["jump_penalty"], skip_penalty=a["skip_penalty"],
                                   back_penalty=a["back_penalty"], max_back=a["max_back"],
                                   start_prior_mu=a.get("start_prior_mu", 0.0))
    rows = []
    tag = "sentence" if not args.window else f"w{int(args.window)}"
    for stem in sorted(m.LECTURES):
        path = m.CACHE / "S" / f"{stem}.{tag}.ocr.npz"
        if not path.exists():
            continue
        S = np.load(path)["S"]
        g = relatedness_gate(S, dp, z=z, jump_penalty=a["jump_penalty"], skip_penalty=a["skip_penalty"],
                             back_penalty=a["back_penalty"])
        rows.append((m.LECTURES[stem][1], S.shape, g))
        print(f"{m.LECTURES[stem][1]:<20} z={g['z']:6.1f}  score {g['score']:.3f} vs null {g['null_mean']:.3f}±{g['null_std']:.3f}  "
              f"{'ok' if g['related'] else 'REJECTED'}")
    if not rows:
        return []
    rejected = [r for r in rows if not r[2]["related"]]
    zs = [r[2]["z"] for r in rows]
    gran = "sentence granularity (their protocol)" if not args.window else f"{int(args.window)}-second windows (LinkRAG's segment granularity)"
    L = [f"## False-rejection check — {len(rows)} MaViLS lectures, {gran} (all related by construction)", "",
         f"Gate on the cached matrices ({gran}, page OCR, our hybrid similarity), our DP at pilot "
         f"parameters, 5 shuffled slide orders, threshold z = {z}.", "",
         "| lecture | segments × slides | path score | shuffled mean ± std | z | verdict |", "|---|---:|---:|---:|---:|---|"]
    for name, shape, g in rows:
        L.append(f"| {name} | {shape[0]}×{shape[1]} | {g['score']:.3f} | {g['null_mean']:.3f} ± {g['null_std']:.3f} | "
                 f"{g['z']:.1f} | {'related' if g['related'] else '**REJECTED**'} |")
    L += ["", f"**False-rejection rate at z = {z}: {len(rejected)}/{len(rows)} = {len(rejected) / max(len(rows), 1):.0%}.** "
          f"z-scores: min {min(zs):.1f}, median {float(np.median(zs)):.1f}, max {max(zs):.1f}."
          + (" Rejected: " + ", ".join(r[0] for r in rejected) + "." if rejected else ""), ""]
    return L


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--skip-control", action="store_true", help="only the MaViLS false-rejection sections")
    ap.add_argument("--reingest", action="store_true")
    args = ap.parse_args(argv)
    setup_logging()
    cfg = load_config(args.config)
    head = ["# Relatedness gate — negative control and false-rejection check", "",
            "The gate (`linkrag.link.align.relatedness_gate`): the penalised DP objective per segment must exceed the mean "
            "of 5 shuffled-slide-order alignments by `align.relatedness_z` shuffled standard deviations; otherwise no audio_slide "
            "links are emitted and the pair is recorded as unrelated in `links.jsonl` `_meta`. Cross-document semantic figure_text "
            "links use the same z over a word-shuffle null (`figure_text.document_pair_gate`). Per-segment abstention "
            "(`align.min_segment_sim`) runs after the gate on surviving pairs.", ""]
    sections: list[str] = []
    if not args.skip_control:
        sections += control(cfg, args)
    for window in (0.0, 30.0):
        args.window = window
        sec = mavils_false_rejections(cfg, args)
        if sec:
            sections += sec
    OUT.write_text("\n".join(head + sections) + "\n")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
