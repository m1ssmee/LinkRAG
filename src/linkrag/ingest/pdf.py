"""PDF -> EvidenceUnits via PyMuPDF.

baseline: text chunks and figures are extracted independently and never
          related to each other -- exactly what P3 (MARA) does with pages.
linkrag:  same extraction; the linker later consumes the bboxes and captions
          this module already records.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pymupdf

import logging

from linkrag.core import EvidenceUnit, Location, stage_timer

log = logging.getLogger("linkrag")

# Note: word count stands in for a real tokenizer. ~0.75 words per token on
# English prose; swap in the bge-m3 tokenizer if chunk sizes ever need to be exact.
WORDS_PER_TOKEN = 0.75

CAPTION_MAX_GAP_PT = 120.0  # a text block further below an image isn't its caption

# "Figure 3:" / "Table 2." at the start of a block -- a real caption, not a mention.
CAPTION_START = re.compile(r"^\s*(figure|table)\s+(\d{1,2})\s*[:.]", re.I)
MIN_CAPTION_FIGURE_PT = 40.0    # a region shorter than this is not a figure
MAX_CAPTION_FIGURE_PT = 420.0   # nor is half a page of prose above a caption
COLUMN_OVERLAP = 0.3            # fraction of caption width a block must share to count
BODY_TEXT_WORDS = 12            # a block this long is prose, not a figure's own label

# --- slide-deck figure clustering -------------------------------------------
# A deck draws its plots as vector operators, so page.get_images() sees nothing:
# slides 13, 22 and 23 of pilot01 have zero raster images and produced zero figure
# units. Cluster the drawing and image boxes instead. These are new parameters, not
# retuned ones -- no existing threshold is changed.
CLUSTER_GAP_PT = 12.0           # boxes closer than this belong to one figure
CLUSTER_MIN_AREA_PT2 = 8000.0   # ~90x90pt; below this it is a rule or a bullet
BACKGROUND_PAGE_FRACTION = 0.8  # a box covering this much of the page is the backdrop
TEMPLATE_MIN_PAGES = 3          # an identical box on this many pages is deck chrome
LOGO_MAX_AREA_PT2 = 15000.0     # a small box in a page corner is branding
LOGO_CORNER_PT = 100.0


def _merge_boxes(boxes, gap: float):
    """Union boxes that overlap or sit within `gap` of each other, to fixpoint."""
    boxes = [tuple(b) for b in boxes]
    changed = True
    while changed:
        changed = False
        merged = []
        while boxes:
            b = boxes.pop()
            touching = [o for o in boxes
                        if not (b[2] + gap < o[0] or o[2] + gap < b[0]
                                or b[3] + gap < o[1] or o[3] + gap < b[1])]
            for o in touching:
                boxes.remove(o)
                b = (min(b[0], o[0]), min(b[1], o[1]), max(b[2], o[2]), max(b[3], o[3]))
                changed = True
            merged.append(b)
        boxes = merged
    return boxes


def _is_logo(box, page_rect, max_area: float, corner: float) -> bool:
    area = (box[2] - box[0]) * (box[3] - box[1])
    if area > max_area:
        return False
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return min(cx, page_rect.width - cx) < corner and min(cy, page_rect.height - cy) < corner


def cluster_figure_regions(
    page: pymupdf.Page,
    *,
    gap_pt: float = CLUSTER_GAP_PT,
    min_area_pt2: float = CLUSTER_MIN_AREA_PT2,
    background_fraction: float = BACKGROUND_PAGE_FRACTION,
    logo_max_area_pt2: float = LOGO_MAX_AREA_PT2,
    logo_corner_pt: float = LOGO_CORNER_PT,
) -> tuple[list[tuple[float, float, float, float]], dict[str, int]]:
    """Spatial clusters of vector drawings + raster images on one deck page.

    Returns (kept boxes, counters). Drops, in order: the page backdrop, degenerate
    boxes, clusters below `min_area_pt2`, and corner logos.
    """
    rect = page.rect
    page_area = rect.width * rect.height
    boxes = [tuple(d["rect"]) for d in page.get_drawings()]
    n_draw = len(boxes)
    n_img = 0
    for xref, *_rest in page.get_images(full=True):
        for r in page.get_image_rects(xref):
            boxes.append(tuple(r))
            n_img += 1

    boxes = [b for b in boxes
             if b[2] > b[0] and b[3] > b[1]
             and (b[2] - b[0]) * (b[3] - b[1]) < background_fraction * page_area]

    clusters = _merge_boxes(boxes, gap_pt)
    kept, small, logos = [], 0, 0
    for c in clusters:
        if (c[2] - c[0]) * (c[3] - c[1]) < min_area_pt2:
            small += 1
        elif _is_logo(c, rect, logo_max_area_pt2, logo_corner_pt):
            logos += 1
        else:
            kept.append(c)
    return sorted(kept), {"draw_ops": n_draw, "images": n_img, "clusters": len(clusters),
                          "kept": len(kept), "small": small, "logos": logos}


def slide_title(blocks: list[_Block]) -> str:
    """Topmost text block on the page -- a deck's de-facto caption for everything on it."""
    return min(blocks, key=lambda b: b.bbox[1]).text if blocks else ""


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


def caption_regions(
    page: pymupdf.Page, blocks: list[_Block]
) -> list[tuple[tuple[float, float, float, float], str]]:
    """Figure regions inferred from their captions, for documents whose figures are
    vector drawings.

    `page.get_images()` only sees *raster* images. A LaTeX paper draws its plots with
    vector operators, so the extractor finds nothing: the Focus paper has 13 captioned
    figures and yielded 2 units, both logo fragments. The caption, however, is real
    text at a known position -- so take the region directly above it, bounded by the
    next block up in the same column, and render that.

    Returns (bbox, caption_text) pairs. Column membership is by horizontal overlap
    with the caption, which is what keeps a two-column paper's left figure from
    swallowing the right column.
    """
    regions = []
    for block in blocks:
        if not CAPTION_START.match(block.text):
            continue
        x0, y0, x1, y1 = block.bbox
        width = max(x1 - x0, 1.0)
        # Stop at the nearest *prose* block, not the nearest block: a vector plot's
        # axis labels and legend are themselves text blocks sitting a few points
        # above the caption, and bounding on those collapsed 9 of 14 regions in the
        # Focus paper to 6-16pt of nothing.
        above = [
            other.bbox[3] for other in blocks
            if other is not block and other.bbox[3] <= y0
            and (min(other.bbox[2], x1) - max(other.bbox[0], x0)) > COLUMN_OVERLAP * width
            and len(other.text.split()) >= BODY_TEXT_WORDS
        ]
        top = max(above) if above else max(y0 - MAX_CAPTION_FIGURE_PT, 0.0)
        top = max(top, y0 - MAX_CAPTION_FIGURE_PT)
        height = y0 - top
        if height < MIN_CAPTION_FIGURE_PT:
            continue
        regions.append(((x0, top, x1, y0), block.text))
    return regions


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
    figures_from_captions: bool | None = None,
    cluster_deck_figures: bool = True,
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
    caption_figures = 0
    cluster_figs = 0
    template_dropped = 0
    deck_candidates: list[tuple[int, tuple[float, float, float, float], str]] = []
    cluster_log: list[tuple[int, dict[str, int]]] = []
    with stage_timer("ingest.pdf", file=path.name) as t:
        doc = pymupdf.open(path)
        page_count = doc.page_count
        # A deck is landscape; a paper is portrait. Explicit config wins over the
        # heuristic so an unusual document can always be declared by hand.
        landscape = sum(1 for pg in doc if pg.rect.width > pg.rect.height)
        is_deck = (slide_deck if slide_deck is not None
                   else page_count > 0 and landscape / page_count >= landscape_ratio)
        # Decks caption nothing and draw nothing vectorially; papers do both.
        if figures_from_captions is None:
            figures_from_captions = not is_deck
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

                if is_deck and cluster_deck_figures:
                    boxes, stats = cluster_figure_regions(page)
                    cluster_log.append((page_no, stats))
                    for box in boxes:
                        deck_candidates.append((page_no, box, slide_title(blocks)))
                    continue  # deck figures are emitted after template filtering

                # Caption-anchored figures: the only way to reach vector plots.
                if figures_from_captions:
                    for c, (bbox, caption) in enumerate(caption_regions(page, blocks)):
                        out = figures_dir / f"{path.stem}_p{page_no}_c{c}.png"
                        out.parent.mkdir(parents=True, exist_ok=True)
                        try:
                            pix = page.get_pixmap(clip=pymupdf.Rect(*bbox), dpi=120)
                            pix.save(out)
                        except Exception:  # a region we cannot render is not fatal
                            continue
                        caption_figures += 1
                        units.append(
                            EvidenceUnit(
                                id=f"{path.stem}:p{page_no}:c{c}",
                                modality="figure",
                                content=" ".join(caption.split()),
                                source_file=str(path),
                                location=Location(page=page_no, bbox=bbox),
                                metadata={"image_path": str(out),
                                          "content_source": "caption_region"},
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
            # Template chrome (title banners, footers) clusters identically on many
            # pages. Emitting it would add one junk figure per slide, so drop any box
            # whose rounded geometry repeats across TEMPLATE_MIN_PAGES pages.
            repeats: dict[tuple[int, ...], int] = {}
            for _pno, box, _title in deck_candidates:
                repeats[tuple(round(v) for v in box)] = \
                    repeats.get(tuple(round(v) for v in box), 0) + 1
            per_page: dict[int, int] = {}
            for page_no, box, title in deck_candidates:
                if repeats[tuple(round(v) for v in box)] >= TEMPLATE_MIN_PAGES:
                    template_dropped += 1
                    continue
                i = per_page.get(page_no, 0)
                per_page[page_no] = i + 1
                out = figures_dir / f"{path.stem}_p{page_no}_g{i}.png"
                out.parent.mkdir(parents=True, exist_ok=True)
                try:
                    doc[page_no - 1].get_pixmap(clip=pymupdf.Rect(*box), dpi=120).save(out)
                except Exception as exc:
                    log.warning("cluster crop failed on p%s: %s", page_no, exc)
                    continue
                text = ""
                source = "slide_title"
                if ocr_figures:
                    from linkrag.ingest.image import ocr, tesseract_available

                    if tesseract_available():
                        text = ocr(out)
                        if text:
                            source = "slide_title+ocr"
                            ocr_used += 1
                content = " ".join(f"{title} {text}".split())
                units.append(
                    EvidenceUnit(
                        id=f"{path.stem}:p{page_no}:g{i}",
                        modality="figure",
                        content=content,
                        source_file=str(path),
                        location=Location(page=page_no, bbox=box),
                        metadata={"image_path": str(out), "content_source": source,
                                  "cluster": True},
                    )
                )
                cluster_figs += 1
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
        t["cap_figs"] = caption_figures
        t["cluster_figs"] = cluster_figs
        t["tmpl_dropped"] = template_dropped
        for page_no, st in cluster_log:
            log.info("  p%-3d draw_ops=%-4d images=%-3d clusters=%-3d kept=%-2d "
                     "small=%-2d logos=%d", page_no, st["draw_ops"], st["images"],
                     st["clusters"], st["kept"], st["small"], st["logos"])
    return units
