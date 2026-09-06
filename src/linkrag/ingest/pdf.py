"""PDF -> EvidenceUnits via PyMuPDF.

baseline: text chunks and figures are extracted independently and never
          related to each other -- exactly what P3 (MARA) does with pages.
linkrag:  same extraction; the linker later consumes the bboxes and captions
          this module already records.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pymupdf

from linkrag.core import EvidenceUnit, Location, stage_timer

# Note: word count stands in for a real tokenizer. ~0.75 words per token on
# English prose; swap in the bge-m3 tokenizer if chunk sizes ever need to be exact.
WORDS_PER_TOKEN = 0.75

CAPTION_MAX_GAP_PT = 120.0  # a text block further below an image isn't its caption


@dataclass
class _Block:
    """One PyMuPDF text block: its text and where it sits on the page."""

    text: str
    bbox: tuple[float, float, float, float]


def _union(boxes: list[tuple[float, float, float, float]]) -> tuple[float, float, float, float]:
    return (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )


def _page_blocks(page: pymupdf.Page) -> list[_Block]:
    """Text blocks in reading order (PyMuPDF sorts top-to-bottom, left-to-right)."""
    blocks = []
    for x0, y0, x1, y1, text, _no, block_type in page.get_text("blocks"):
        if block_type != 0:  # 1 == image block; figures are handled separately
            continue
        text = " ".join(text.split())
        if text:
            blocks.append(_Block(text, (x0, y0, x1, y1)))
    return blocks


def chunk_page(
    blocks: list[_Block],
    *,
    chunk_tokens: int = 300,
    overlap_tokens: int = 60,
) -> list[tuple[str, tuple[float, float, float, float]]]:
    """Sliding word window over a page's blocks -> (text, bbox) pairs.

    bbox is the union of every block the window touches, so a chunk spanning the
    first through last block carries a rectangle covering both.

    Note: chunks never cross a page boundary. Costs a little context at page
    seams, buys exact page+bbox citations and stops slide N bleeding into N+1.
    """
    words: list[tuple[str, int]] = [
        (word, i) for i, block in enumerate(blocks) for word in block.text.split()
    ]
    if not words:
        return []

    size = max(1, int(chunk_tokens * WORDS_PER_TOKEN))
    overlap = min(int(overlap_tokens * WORDS_PER_TOKEN), size - 1)
    step = size - overlap

    chunks = []
    for start in range(0, len(words), step):
        window = words[start : start + size]
        if not window:
            break
        # A tail shorter than the overlap is already fully inside the previous chunk.
        if start > 0 and len(window) <= overlap:
            break
        touched = sorted({i for _, i in window})
        chunks.append(
            (" ".join(w for w, _ in window), _union([blocks[i].bbox for i in touched]))
        )
    return chunks


def find_caption(image_bbox: tuple[float, float, float, float], blocks: list[_Block]) -> str:
    """Nearest text block below the image, if there is one.

    Note: deliberately naive per spec -- no "Figure N" pattern, no block above.
    Caption quality is the linker's problem, not the baseline's.
    """
    _, _, ix1, iy1 = image_bbox
    ix0 = image_bbox[0]
    best: tuple[float, str] | None = None
    for block in blocks:
        bx0, by0, bx1, _ = block.bbox
        gap = by0 - iy1
        if gap < 0 or gap > CAPTION_MAX_GAP_PT:
            continue
        if min(bx1, ix1) - max(bx0, ix0) <= 0:  # no horizontal overlap: different column
            continue
        if best is None or gap < best[0]:
            best = (gap, block.text)
    return best[1] if best else ""


def _save_image(doc: pymupdf.Document, xref: int, out: Path) -> bool:
    pix = pymupdf.Pixmap(doc, xref)
    try:
        if pix.n - pix.alpha >= 4:  # CMYK / separation -> RGB so PNG can hold it
            pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
        out.parent.mkdir(parents=True, exist_ok=True)
        pix.save(out)
        return True
    finally:
        pix = None


def ingest_pdf(
    path: str | Path,
    *,
    figures_dir: str | Path = "data/processed/figures",
    chunk_tokens: int = 300,
    overlap_tokens: int = 60,
    min_figure_area_px: int = 10_000,
    ocr_figures: bool = False,
    vlm_cfg: dict | None = None,
    slide_deck: bool | None = None,
    landscape_ratio: float = 0.6,
) -> list[EvidenceUnit]:
    """ocr_figures: when a figure has no caption, read the text inside the image.

    Slide decks caption almost nothing -- 13 of 18 figures in pilot01 had empty
    content, because `find_caption` looks for a text block below the image, which
    is a *paper* convention. The axis labels, legends and titles rendered inside
    the image are the figure's real text, and OCR is the only way to reach them.
    """
    path = Path(path)
    figures_dir = Path(figures_dir)
    units: list[EvidenceUnit] = []

    ocr_used = 0
    vlm_used = 0
    with stage_timer("ingest.pdf", file=path.name) as t:
        doc = pymupdf.open(path)
        page_count = doc.page_count
        # A deck is landscape; a paper is portrait. Explicit config wins over the
        # heuristic so an unusual document can always be declared by hand.
        landscape = sum(1 for pg in doc if pg.rect.width > pg.rect.height)
        is_deck = (slide_deck if slide_deck is not None
                   else page_count > 0 and landscape / page_count >= landscape_ratio)
        try:
            for page_no, page in enumerate(doc, start=1):
                blocks = _page_blocks(page)

                for i, (text, bbox) in enumerate(
                    chunk_page(blocks, chunk_tokens=chunk_tokens, overlap_tokens=overlap_tokens)
                ):
                    units.append(
                        EvidenceUnit(
                            id=f"{path.stem}:p{page_no}:t{i}",
                            modality="text",
                            content=text,
                            source_file=str(path),
                            location=Location(page=page_no, bbox=bbox),
                        )
                    )

                for i, (xref, *_rest) in enumerate(page.get_images(full=True)):
                    rects = page.get_image_rects(xref)
                    if not rects:
                        continue  # referenced but not placed on this page
                    rect = rects[0]
                    if rect.width * rect.height < min_figure_area_px:
                        continue  # rules, bullets, logos
                    bbox = (rect.x0, rect.y0, rect.x1, rect.y1)
                    out = figures_dir / f"{path.stem}_p{page_no}_{i}.png"
                    if not _save_image(doc, xref, out):
                        continue
                    caption = find_caption(bbox, blocks)
                    source = "caption"
                    if not caption and ocr_figures:
                        from linkrag.ingest.image import ocr, tesseract_available

                        if tesseract_available():
                            caption = ocr(out)
                            source = "ocr" if caption else "none"
                            ocr_used += 1
                        else:
                            source = "none"
                    if vlm_cfg:
                        from linkrag.ingest.vlm_caption import describe_if_enabled

                        described = describe_if_enabled(out, {"ingest": {"vlm_captions": vlm_cfg}})
                        if described:
                            caption = f"{caption} {described}".strip()
                            source = f"{source}+vlm" if source != "none" else "vlm"
                            vlm_used += 1
                    units.append(
                        EvidenceUnit(
                            id=f"{path.stem}:p{page_no}:f{i}",
                            modality="figure",
                            content=caption,
                            source_file=str(path),
                            location=Location(page=page_no, bbox=bbox),
                            metadata={"image_path": str(out), "content_source": source},
                        )
                    )
        finally:
            doc.close()

        t["pages"] = page_count
        t["text"] = sum(u.modality == "text" for u in units)
        for unit in units:
            unit.metadata["slide_deck"] = is_deck
        t["deck"] = is_deck
        t["figures"] = sum(u.modality == "figure" for u in units)
        t["ocr"] = ocr_used
        t["vlm"] = vlm_used
    return units
