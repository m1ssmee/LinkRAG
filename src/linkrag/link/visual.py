"""Visual channel for audio->slide alignment: what is on screen, matched to deck pages.

A lecture video shows the slide being talked about. `linkrag.ingest.video_slides`
recovers one representative frame per on-screen interval (1 fps dHash segmentation).
Here each frame is scored against every rendered deck page, and each transcript
segment inherits the row of the frame on screen at its midpoint. The result is a
segment x page matrix with the same shape as the text similarity, so the existing DP
decodes it and `link.align.fuse_similarity` combines it with text.

Frame->page scores from the image:
  dhash     1 - Hamming/64 between 64-bit difference hashes. Cheap, no model; blind to
            a slide shown small inside a camera shot.
  swiftformer  cosine of MBZUAI/swiftformer-xs features, flattened last hidden state.
            This is MaViLS's own image feature (`mavils/matching_algorithm.py`), so the
            visual column is like-for-like. Preprocessing is the model's published
            config done by hand (224x224 bilinear, /255, ImageNet mean/std), because
            `AutoImageProcessor` would pull in torchvision. MaViLS resizes a frame to
            the page size before that same 224x224 resize; we resize once.

and from the text on the frame (`ocr_frame`, then `text_similarity`: TF-IDF or BM25
against the page OCR text) -- MaViLS's third feature.

A segment whose midpoint falls outside every on-screen interval gets a zero row: no
visual evidence, and the decoder's transition terms decide.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from linkrag.ingest.video_slides import Slide, dhash, hamming, slide_at

SWIFTFORMER = "MBZUAI/swiftformer-xs"


def render_pages(pdf: str | Path, dpi: int = 72) -> list[Any]:
    """Every page of a deck as a PIL image."""
    import pymupdf
    from PIL import Image
    with pymupdf.open(pdf) as doc:
        return [Image.frombytes("RGB", (p.width, p.height), p.samples)
                for p in (page.get_pixmap(dpi=dpi) for page in doc)]


def dhash_similarity(frames: Sequence[Any], pages: Sequence[Any]) -> np.ndarray:
    """frame x page, 1 - Hamming/64 of the 64-bit dHash."""
    f, p = [dhash(im) for im in frames], [dhash(im) for im in pages]
    return np.array([[1.0 - hamming(a, b) / 64.0 for b in p] for a in f])


@lru_cache(maxsize=1)
def _swiftformer(device: str):
    from transformers import SwiftFormerModel
    return SwiftFormerModel.from_pretrained(SWIFTFORMER).to(device).eval()


def swiftformer_features(images: Sequence[Any], device: str = "cpu", batch: int = 16) -> np.ndarray:
    """L2-normalised flattened last hidden state of each image."""
    import torch
    from PIL import Image
    mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
    model, out = _swiftformer(device), []
    for i in range(0, len(images), batch):
        a = np.stack([np.asarray(im.convert("RGB").resize((224, 224), Image.BILINEAR), dtype=np.float64) / 255.0
                      for im in images[i:i + batch]])
        x = torch.from_numpy(((a - mean) / std).transpose(0, 3, 1, 2).astype(np.float32)).to(device)
        with torch.no_grad():
            h = model(pixel_values=x).last_hidden_state
        out.append(h.reshape(h.shape[0], -1).cpu().numpy())
    v = np.concatenate(out) if out else np.zeros((0, 1))
    return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)


def swiftformer_similarity(frames: Sequence[Any], pages: Sequence[Any], device: str = "cpu") -> np.ndarray:
    """frame x page cosine of SwiftFormer-xs features."""
    return swiftformer_features(frames, device) @ swiftformer_features(pages, device).T


def ocr_frame(image: Any, scale: int = 3) -> str:
    """Tesseract on a frame upscaled `scale` times (LANCZOS). The stored frames are 320 px on
    the longest side; at native size slide text is a few pixels tall and OCR returns
    almost nothing (3x roughly triples the words read, on frames alone, no labels)."""
    import pytesseract
    from PIL import Image
    im = image.convert("L")
    if scale > 1:
        im = im.resize((im.width * scale, im.height * scale), Image.LANCZOS)
    return " ".join(pytesseract.image_to_string(im).split())


def text_similarity(frame_texts: Sequence[str], page_texts: Sequence[str], method: str = "tfidf") -> np.ndarray:
    """frame x page similarity of frame OCR text to page text, tokenised as everywhere else
    (`linkrag.index.tokenize`). tfidf: cosine of TF-IDF vectors fit on the pages. bm25: BM25
    scores of each frame's words against the pages, min-max scaled per frame. A frame with no
    text gets a zero row."""
    from linkrag.index import tokenize
    if method == "tfidf":
        from sklearn.feature_extraction.text import TfidfVectorizer
        vec = TfidfVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None).fit(page_texts)
        return (vec.transform(frame_texts) @ vec.transform(page_texts).T).toarray()
    if method == "bm25":
        from rank_bm25 import BM25Okapi
        bm25 = BM25Okapi([tokenize(t) or ["_"] for t in page_texts])
        raw = np.vstack([bm25.get_scores(tokenize(t)) if tokenize(t) else np.zeros(len(page_texts))
                         for t in frame_texts])
        lo, hi = raw.min(axis=1, keepdims=True), raw.max(axis=1, keepdims=True)
        return (raw - lo) / np.maximum(hi - lo, 1e-9)
    raise ValueError(f"unknown text similarity {method!r}: use 'tfidf' or 'bm25'")


def segment_visual(midpoints: Sequence[float], slides: Sequence[Slide], frame_page: np.ndarray) -> np.ndarray:
    """segment x page: each segment takes the frame->page row of the slide on screen at
    its midpoint (`slide_at`); zeros where no interval covers it."""
    out = np.zeros((len(midpoints), frame_page.shape[1]))
    for i, t in enumerate(midpoints):
        k = slide_at(list(slides), float(t))
        if k is not None:
            out[i] = frame_page[k]
    return out


def confident_rate(frame_page: np.ndarray, margin: float) -> float:
    """Share of frames whose best page beats the runner-up by more than `margin`."""
    if frame_page.shape[1] < 2 or not len(frame_page):
        return float("nan")
    top2 = np.sort(frame_page, axis=1)[:, -2:]
    return float(np.mean(top2[:, 1] - top2[:, 0] > margin))
