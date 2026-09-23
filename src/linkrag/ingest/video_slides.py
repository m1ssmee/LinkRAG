"""Slide units derived from a lecture video's frames.

LectQA-Vid is one video per lecture, so there is no separate deck for the linking layer to
link to. Its videos are slide lectures, though, so the deck can be recovered from the
frames:

1. **Sample** frames at `fps` (default 1 per second).
2. **Segment** by a difference hash (dHash, 64 bits). A new slide starts when consecutive
   frames differ by more than `max_dist` bits. Comparing consecutive frames rather than
   the segment's first frame keeps a progressive build (bullets appearing one by one) on
   one slide. Segments shorter than `min_dur` are transitions and merge into the previous
   segment.
3. **Deduplicate.** A segment whose representative frame is within `max_dist` of an
   earlier slide is that slide revisited: one slide, several on-screen intervals.
4. **Per slide:** OCR the representative frame (the last frame of its longest interval,
   i.e. the most complete build). Find figures the way the deck path does
   (`ingest.pdf`): boxes merged to a fixpoint with `_merge_boxes`, then the backdrop,
   specks and corner logos dropped. On a raster frame the boxes are the connected
   non-background regions left after OCR word boxes are masked out.
5. **Emit** one page unit per slide (`modality="text"`) and one unit per figure
   (`modality="figure"`), all with `metadata["source"] = "frame"` and
   `slide_deck = True`. The location is the slide's first on-screen interval, with page =
   slide number; all intervals are in `metadata["intervals"]`.

Note: dHash thresholds are set by eye on LectQA-Vid lectures (static slides, 360p), not
tuned against an evaluation. The known interval of every slide is what lets the audio->slide
alignment be checked for free (see scripts/adapters/lectqa_vid.py frameslides).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from linkrag.core import EvidenceUnit, Location
from linkrag.ingest.pdf import _merge_boxes


@dataclass
class Slide:
    index: int
    intervals: list[tuple[float, float]] = field(default_factory=list)
    hash: int = 0
    frame_time: float = 0.0
    image: Any = None                                   # PIL image of the representative frame


def sample_frames(video: str | Path, fps: float = 1.0) -> list[tuple[float, Any]]:
    """(time_s, PIL image) at `fps`, decoded with PyAV."""
    import av
    out: list[tuple[float, Any]] = []
    next_t, step = 0.0, 1.0 / fps
    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            t = float(frame.pts * stream.time_base) if frame.pts is not None else 0.0
            if t + 1e-6 < next_t:
                continue
            next_t = t + step
            out.append((t, frame.to_image()))
    return out


def dhash(img: Any, size: int = 8) -> int:
    """Difference hash: sign of horizontal gradients on a (size+1) x size grey thumbnail."""
    g = np.asarray(img.convert("L").resize((size + 1, size)), dtype=np.int16)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def segment_slides(frames: list[tuple[float, Any]], *, max_dist: int = 10, min_dur: float = 2.0,
                   step: float = 1.0) -> list[Slide]:
    """Frames -> slides (segmented, transitions merged, revisits deduplicated)."""
    if not frames:
        return []
    hashes = [dhash(img) for _, img in frames]
    segs: list[list[int]] = [[0]]                       # frame indices per segment
    for i in range(1, len(frames)):
        if hamming(hashes[i], hashes[i - 1]) > max_dist:
            segs.append([i])
        else:
            segs[-1].append(i)
    # transitions: a segment shorter than min_dur joins the previous one's interval, but the
    # previous segment keeps its own representative frame (a transition frame is not the slide)
    merged: list[tuple[list[int], int]] = []            # (frame indices, representative index)
    for s in segs:
        dur = frames[s[-1]][0] - frames[s[0]][0] + step
        if merged and dur < min_dur:
            merged[-1][0].extend(s)
        else:
            merged.append((list(s), s[-1]))             # last frame: the most complete build
    slides: list[Slide] = []
    for s, rep in merged:
        start, end = frames[s[0]][0], frames[s[-1]][0] + step
        match = next((sl for sl in slides if hamming(sl.hash, hashes[rep]) <= max_dist), None)
        if match is None:
            slides.append(Slide(index=len(slides), intervals=[(start, end)], hash=hashes[rep],
                                frame_time=frames[rep][0], image=frames[rep][1]))
        else:                                            # revisited: one slide, another interval
            longest = max(match.intervals, key=lambda iv: iv[1] - iv[0])
            if abs(match.intervals[-1][1] - start) < 1e-6:   # back to back: one interval
                match.intervals[-1] = (match.intervals[-1][0], end)
            else:
                match.intervals.append((start, end))
            if end - start > longest[1] - longest[0]:
                match.hash, match.frame_time, match.image = hashes[rep], frames[rep][0], frames[rep][1]
    return slides


def ocr_words(img: Any, min_conf: float = 50.0) -> tuple[str, list[tuple[int, int, int, int, str]]]:
    """Full text plus word boxes (x0, y0, x1, y1, word) from tesseract."""
    import pytesseract
    d = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    words = []
    for i, w in enumerate(d["text"]):
        if w.strip() and float(d["conf"][i]) >= min_conf:
            x, y, ww, hh = d["left"][i], d["top"][i], d["width"][i], d["height"][i]
            words.append((x, y, x + ww, y + hh, w.strip()))
    return " ".join(w[4] for w in words), words


def figure_boxes(img: Any, words: list[tuple[int, int, int, int, str]], *, contrast: int = 40,
                 min_frac: float = 0.02, max_frac: float = 0.9, gap_frac: float = 0.02,
                 logo_max_frac: float = 0.015, corner_frac: float = 0.12) -> list[tuple[int, int, int, int]]:
    """Non-text regions of a slide frame, clustered like the deck path's figure regions."""
    from scipy import ndimage
    g = np.asarray(img.convert("L"), dtype=np.int16)
    h, w = g.shape
    border = np.concatenate([g[0], g[-1], g[:, 0], g[:, -1]])
    background = int(np.median(border))
    mask = np.abs(g - background) > contrast
    pad = max(2, h // 120)
    for x0, y0, x1, y1, _ in words:                     # text is not a figure
        mask[max(0, y0 - pad):y1 + pad, max(0, x0 - pad):x1 + pad] = False
    labels, _ = ndimage.label(mask)
    boxes = []
    for sl in ndimage.find_objects(labels):
        if sl is None:
            continue
        y0, y1, x0, x1 = sl[0].start, sl[0].stop, sl[1].start, sl[1].stop
        if (x1 - x0) * (y1 - y0) >= 0.0005 * w * h:     # drop specks before merging
            boxes.append((x0, y0, x1, y1))
    area = float(w * h)
    kept = []
    for b in _merge_boxes(boxes, gap_frac * w):
        frac = (b[2] - b[0]) * (b[3] - b[1]) / area
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        in_corner = min(cx, w - cx) < corner_frac * w and min(cy, h - cy) < corner_frac * h
        if min_frac <= frac <= max_frac and not (frac < logo_max_frac and in_corner):
            kept.append(tuple(int(v) for v in b))
    return sorted(kept)


def slide_units(video: str | Path, vid: str, out_dir: str | Path, *, fps: float = 1.0, max_dist: int = 10,
                min_dur: float = 2.0) -> tuple[list[EvidenceUnit], list[Slide]]:
    """Page and figure units for every slide of one video (frames saved under out_dir)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    slides = segment_slides(sample_frames(video, fps), max_dist=max_dist, min_dur=min_dur, step=1.0 / fps)
    units: list[EvidenceUnit] = []
    for sl in slides:
        page = sl.index + 1
        frame_path = out_dir / f"{vid}_slide{page:03d}.png"
        sl.image.save(frame_path)
        text, words = ocr_words(sl.image)
        (t0, t1) = sl.intervals[0]
        meta = {"slide_deck": True, "source": "frame", "intervals": [list(iv) for iv in sl.intervals],
                "frame_path": str(frame_path), "frame_time": sl.frame_time, "content_source": "ocr"}
        units.append(EvidenceUnit(id=f"{vid}:s{page}", modality="text", content=text, source_file=str(video),
                                  location=Location(page=page, start_s=t0, end_s=t1), metadata=dict(meta)))
        for j, box in enumerate(figure_boxes(sl.image, words)):
            inside = " ".join(wd[4] for wd in words if wd[0] >= box[0] and wd[2] <= box[2]
                              and wd[1] >= box[1] and wd[3] <= box[3])
            crop = out_dir / f"{vid}_slide{page:03d}_fig{j}.png"
            sl.image.crop(box).save(crop)
            units.append(EvidenceUnit(
                id=f"{vid}:s{page}:g{j}", modality="figure",
                content=f"Figure on slide {page}. {inside}".strip(), source_file=str(video),
                location=Location(page=page, bbox=tuple(float(v) for v in box), start_s=t0, end_s=t1),
                metadata={**meta, "crop_path": str(crop), "content_source": "ocr-in-box"}))
    return units, slides


def slide_at(slides: list[Slide], t: float) -> int | None:
    """Index of the slide on screen at time t (None between intervals)."""
    for sl in slides:
        if any(a <= t < b for a, b in sl.intervals):
            return sl.index
    return None
