#!/usr/bin/env python3
"""Ingest files into a searchable index.

    python scripts/ingest.py data/raw/*.pdf data/raw/lecture.wav
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from linkrag.core import load_config, setup_logging, stage_timer
from linkrag.index import build_index, default_encoder
from linkrag.ingest import ingest_files
from linkrag.manifest import MANIFEST_NAME, build_manifest, write_manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ingest documents, images and audio into an index.")
    parser.add_argument("files", nargs="+")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--mode", choices=["baseline", "linkrag"], default="baseline")
    parser.add_argument("--out", default=None, help="index dir (default: index.store_dir from config)")
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default=None,
                        help="override config `device` (whisper + embedder); cuda switches whisper to float16")
    parser.add_argument("--asr", choices=["local", "openai"], default=None,
                        help="override ingest.asr_backend; openai (whisper-1) bills and needs --max-cost")
    args = parser.parse_args(argv)

    setup_logging()
    cfg = load_config(args.config)
    if args.device:
        cfg["device"] = args.device
        if args.device == "cuda":
            cfg["models"]["whisper_compute_type"] = "float16"
    if args.asr:
        cfg["ingest"]["asr_backend"] = args.asr
    out = args.out or cfg["index"]["store_dir"]

    with stage_timer("ingest.total", files=len(args.files)) as t:
        # ingest_files orders documents before audio so slide vocabulary can
        # prime the ASR, and logs+skips any single unreadable file.
        units = ingest_files(args.files, cfg, mode=args.mode)
        t["units"] = len(units)

    if not units:
        print("no units extracted; nothing to index", file=sys.stderr)
        return 1

    encoder = default_encoder(
        cfg["models"]["embedding"], cfg["device"], cfg["index"]["normalize_embeddings"]
    )
    # Warm outside the build timer, or index.build reports the model load as
    # embedding time -- every other script already does this.
    with stage_timer("encoder.warmup", model=cfg["models"]["embedding"]):
        encoder([""])

    index = build_index(
        units,
        encoder=encoder,
        embedding_model=cfg["models"]["embedding"],
        device=cfg["device"],
        normalize=cfg["index"]["normalize_embeddings"],
    )
    with stage_timer("index.save", dir=out):
        index.save(out)

    from linkrag.ingest import transcripts_used
    manifest = build_manifest(args.files, units, derived=transcripts_used())
    manifest_path = write_manifest(manifest, Path(out).parent / MANIFEST_NAME)

    by_modality: dict[str, int] = {}
    for unit in units:
        by_modality[unit.modality] = by_modality.get(unit.modality, 0) + 1
    print(f"indexed {len(units)} units -> {out}  {by_modality}")
    print(f"corpus manifest {manifest['hash']} -> {manifest_path}")
    failed = getattr(ingest_files, "last_failures", [])
    if failed:
        print(f"WARNING: {len(failed)} file(s) failed to ingest:", file=sys.stderr)
        for path, exc in failed:
            print(f"  {path}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
