"""Standalone images (diagrams, scanned notes) -> figure EvidenceUnits via OCR.

baseline: OCR text is the unit's whole content; nothing connects it to the
          prose that discusses the same diagram.
linkrag:  the linker later attaches these to paragraphs and to deictic speech.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytesseract
from PIL import Image

from linkrag.core import EvidenceUnit, Location, stage_timer

TESSERACT_HINT = (
    "pytesseract needs the tesseract binary, which is not on PATH. "
    "Install it with `brew install tesseract` (macOS) or "
    "`apt-get install tesseract-ocr` (Debian/Ubuntu)."
)


def tesseract_available() -> bool:
    return shutil.which(pytesseract.pytesseract.tesseract_cmd) is not None


def ocr(path: str | Path) -> str:
    if not tesseract_available():
        raise RuntimeError(TESSERACT_HINT)
    with Image.open(path) as img:
        return " ".join(pytesseract.image_to_string(img).split())


def ingest_image(path: str | Path) -> list[EvidenceUnit]:
    """One image -> one figure unit. Empty OCR still yields a unit: the image
    exists and the linker may reach it visually even when it holds no text."""
    path = Path(path)
    with stage_timer("ingest.image", file=path.name) as t:
        text = ocr(path)
        with Image.open(path) as img:
            width, height = img.size
        t["chars"] = len(text)
    return [
        EvidenceUnit(
            id=f"{path.stem}:img",
            modality="figure",
            content=text,
            source_file=str(path),
            location=Location(bbox=(0.0, 0.0, float(width), float(height))),
            metadata={"image_path": str(path), "ocr": True},
        )
    ]
