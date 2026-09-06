"""Parse PDF/DOCX, images and audio into EvidenceUnits.

baseline: each file is parsed independently; text chunked by size, figures
          captioned naively, audio cut into fixed windows. No cross-refs.
linkrag:  same parse, but keeps the provenance the linker needs -- figure
          bboxes, caption text, page numbers, word-level audio timestamps.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from linkrag.core import EvidenceUnit, Mode

log = logging.getLogger("linkrag")

SUFFIXES = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".wav": "audio",
    ".mp3": "audio",
    ".m4a": "audio",
    ".png": "image",
    ".jpg": "image",
    ".jpeg": "image",
    ".tif": "image",
    ".tiff": "image",
}


def ingest_file(
    path: str | Path,
    cfg: dict[str, Any] | None = None,
    *,
    mode: Mode = "baseline",
    asr_prompt: str | None = None,
) -> list[EvidenceUnit]:
    """Dispatch one file to its parser. `mode` is accepted for the two-mode rule;
    extraction is identical in both -- only what happens downstream differs."""
    path = Path(path)
    kind = SUFFIXES.get(path.suffix.lower())
    if kind is None:
        raise ValueError(f"unsupported file type: {path.suffix} ({path})")

    cfg = cfg or {}
    ing = cfg.get("ingest", {})
    models = cfg.get("models", {})
    chunk = {
        "chunk_tokens": ing.get("chunk_tokens", 300),
        "overlap_tokens": ing.get("chunk_overlap_tokens", 60),
    }

    if kind == "audio":
        from linkrag.ingest.audio import frozen_transcript_for, ingest_audio

        frozen = frozen_transcript_for(path, ing.get("frozen_transcript_dir"))
        if frozen:
            log.info("using FROZEN transcript %s (not re-transcribing)", frozen)
        return ingest_audio(
            path,
            model_size=models.get("whisper", "small"),
            device=cfg.get("device", "cpu"),
            compute_type=models.get("whisper_compute_type", "int8"),
            window_seconds=ing.get("audio_segment_seconds", 30),
            segmentation=ing.get("audio_segmentation", "sentence"),
            initial_prompt=asr_prompt,
            transcript=frozen,
        )

    if kind == "pdf":
        from linkrag.ingest.pdf import ingest_pdf

        return ingest_pdf(
            path,
            figures_dir=ing.get("figures_dir", "data/processed/figures"),
            min_figure_area_px=ing.get("min_figure_area_px", 10_000),
            ocr_figures=ing.get("ocr_figures", True),
            vlm_cfg=ing.get("vlm_captions"),
            slide_deck=(ing.get("slide_deck_files") or {}).get(path.name),
            landscape_ratio=ing.get("slide_deck_landscape_ratio", 0.6),
            figures_from_captions=ing.get("figures_from_captions"),
            **chunk,
        )
    if kind == "docx":
        from linkrag.ingest.docx import ingest_docx

        return ingest_docx(path, **chunk)
    if kind == "image":
        from linkrag.ingest.image import ingest_image

        return ingest_image(path)


def dedupe_ids(units: list[EvidenceUnit]) -> list[EvidenceUnit]:
    """Unit ids are built from the filename stem, so `a/notes.pdf` and
    `b/notes.pdf` mint the same ids. Ids are what the LLM cites and what
    id_to_pos keys on, so a collision silently mis-attributes evidence.
    Suffix the later one rather than failing: sibling directories per lecture
    are a normal way to organise a corpus."""
    seen: dict[str, int] = {}
    for unit in units:
        if unit.id in seen:
            seen[unit.id] += 1
            unit.id = f"{unit.id}#{seen[unit.id]}"
        else:
            seen[unit.id] = 1
    return units


def ingest_files(
    paths: list[str | Path],
    cfg: dict[str, Any] | None = None,
    *,
    mode: Mode = "baseline",
    skip_failures: bool = True,
) -> list[EvidenceUnit]:
    """Documents first, then audio -- so the slide deck's vocabulary can prime
    whisper's decoder before the lecture is transcribed (`ingest.asr_vocab_from_
    slides`). Untuned, whisper wrote *interest* for **ingest** throughout pilot01.
    """
    from linkrag.ingest.audio import build_asr_prompt

    cfg = cfg or {}
    paths = [Path(p) for p in paths]
    audio = [p for p in paths if SUFFIXES.get(p.suffix.lower()) == "audio"]
    docs = [p for p in paths if p not in audio]

    failures: list[tuple[Path, Exception]] = []

    def run(path: Path, **kw: Any) -> list[EvidenceUnit]:
        try:
            return ingest_file(path, cfg, mode=mode, **kw)
        except Exception as exc:
            if not skip_failures:
                raise
            log.warning("skipped %s: %s: %s", path, type(exc).__name__, exc)
            failures.append((path, exc))
            return []

    units = [u for p in docs for u in run(p)]

    prompt = None
    if audio and cfg.get("ingest", {}).get("asr_vocab_from_slides", True):
        slide_text = [u.content for u in units if u.modality in ("text", "figure")]
        prompt = build_asr_prompt(slide_text) or None
        if prompt:
            log.info("asr vocab from %d document units: %d chars", len(slide_text), len(prompt))

    units += [u for p in audio for u in run(p, asr_prompt=prompt)]
    if failures:
        # Loud, because a skipped audio file yields an index that looks fine and
        # is missing an entire modality.
        log.error("%d of %d files failed to ingest: %s", len(failures), len(paths),
                  ", ".join(str(p) for p, _ in failures))
    ingest_files.last_failures = failures  # type: ignore[attr-defined]
    return dedupe_ids(units)
