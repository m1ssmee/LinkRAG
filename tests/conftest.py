"""Fixtures are generated, not committed: pymupdf draws the PDF, PIL the PNG,
stdlib `wave` the audio. Keeps binaries out of git and the corpus reproducible."""

from __future__ import annotations

import math
import shutil
import struct
import wave
from pathlib import Path
from typing import Sequence

import numpy as np
import pytest

FIXTURES = Path(__file__).parent / "fixtures"

BODY = (
    "Self-attention assigns each token a weight over every other token in the sequence. "
    "The weights are produced by a scaled dot product between queries and keys. "
    "Multiple heads let the model attend to several relations at once. "
    "Positional encodings restore the order information the attention operation discards. "
)
CAPTION = "Figure 1: attention heatmap for layer six head three."
# A cross-reference on the PREVIOUS page, so figure_text has to reach across a page
# boundary to connect it -- exactly what P3 cannot do.
CROSS_REF = "As Figure 1 shows, attention concentrates on the subject token."
TABLE_REF = "Table 2 reports recall at each value of K."


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    FIXTURES.mkdir(parents=True, exist_ok=True)
    return FIXTURES


@pytest.fixture(scope="session")
def png_path(fixtures_dir: Path) -> Path:
    """A PNG with legible text on it, for the OCR path."""
    from PIL import Image, ImageDraw, ImageFont

    path = fixtures_dir / "diagram.png"
    image = Image.new("RGB", (900, 260), "white")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.load_default(size=64)
    except TypeError:  # Pillow < 10 has no size argument
        font = ImageFont.load_default()
    draw.text((40, 40), "ATTENTION HEATMAP", fill="black", font=font)
    draw.text((40, 140), "LAYER SIX HEAD THREE", fill="black", font=font)
    image.save(path)
    return path


@pytest.fixture(scope="session")
def pdf_path(fixtures_dir: Path, png_path: Path) -> Path:
    """Two pages: prose on both, plus an embedded figure with a caption below it."""
    import pymupdf

    path = fixtures_dir / "notes.pdf"
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)          # US letter; y grows downwards from the top
    lines = [line.strip() + "." for line in (BODY * 3).split(". ") if line.strip()] + [CROSS_REF, TABLE_REF]
    page.insert_text((72, 72), "\n".join(lines), fontname="helv", fontsize=12, lineheight=1.2)

    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 72), "Attention weights concentrate on the subject token.", fontname="helv", fontsize=12)
    page.insert_image(pymupdf.Rect(72, 272, 472, 392), filename=str(png_path))
    page.insert_text((72, 412), CAPTION, fontname="helv", fontsize=12)   # the caption sits below the image
    doc.save(path)
    doc.close()
    return path


@pytest.fixture(scope="session")
def docx_path(fixtures_dir: Path) -> Path:
    import docx

    path = fixtures_dir / "notes.docx"
    document = docx.Document()
    for _ in range(4):
        document.add_paragraph(BODY)
    document.save(str(path))
    return path


@pytest.fixture(scope="session")
def wav_path(fixtures_dir: Path) -> Path:
    """5 seconds of 440 Hz, 16 kHz mono 16-bit. Valid audio; says nothing."""
    path = fixtures_dir / "tone.wav"
    rate, seconds = 16_000, 5
    frames = b"".join(
        struct.pack("<h", int(0.3 * 32767 * math.sin(2 * math.pi * 440 * n / rate)))
        for n in range(rate * seconds)
    )
    with wave.open(str(path), "w") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(frames)
    return path


@pytest.fixture
def deictic_audio_unit():
    """An audio unit whose speech points at a visual: "this arrow here"."""
    from linkrag.core import EvidenceUnit, Location

    sentence = ("so this arrow here shows the attention weights concentrating "
                "on the subject token")
    words = [[i * 0.5, (i + 1) * 0.5, w] for i, w in enumerate(sentence.split())]
    return EvidenceUnit(
        id="lecture:a7", modality="audio", content=sentence,
        source_file="data/raw/lecture.wav",
        location=Location(start_s=words[0][0], end_s=words[-1][1]),
        metadata={"words": words},
    )


@pytest.fixture
def stub_encoder():
    """Deterministic bag-of-words encoder.

    Real embeddings mean a 2 GB bge-m3 download in CI. This shares the property
    that matters for testing retrieval -- shared vocabulary means high cosine --
    and nothing else.
    """

    def make(corpus: Sequence[str]):
        vocab = sorted({t for text in corpus for t in text.lower().split()})
        position = {t: i for i, t in enumerate(vocab)}

        def encode(texts: Sequence[str]) -> np.ndarray:
            matrix = np.zeros((len(texts), max(len(vocab), 1)), dtype="float32")
            for row, text in enumerate(texts):
                for token in text.lower().split():
                    if token in position:
                        matrix[row, position[token]] += 1.0
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            return matrix / np.maximum(norms, 1e-9)

        return encode

    return make


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "slow: downloads a model or otherwise takes minutes")


tesseract_missing = shutil.which("tesseract") is None
requires_tesseract = pytest.mark.skipif(tesseract_missing, reason="tesseract binary not installed")
