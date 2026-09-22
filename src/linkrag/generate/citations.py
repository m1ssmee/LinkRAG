"""Citations a reader can check: a page crop, a figure, or an audio clip.

A unit id is a citation only to someone who can open the index. Phase 6 turns each
cited unit into something inspectable:

* text  -> the page number plus a PNG crop of the unit's bbox region (PyMuPDF render)
* figure-> the figure image already extracted at ingest (`metadata["image_path"]`),
           falling back to a bbox crop of its page
* audio -> mm:ss-mm:ss plus a clip cut from the source file

Crops and clips are written under `data/processed/citations/` and named by unit id, so
the same citation resolves to the same file and nothing is re-rendered twice.

Audio is cut with PyAV (already a dependency for the LectQA adapter), not ffmpeg: no
system binary to install, and the clip is re-encoded only when the source codec cannot
be copied into the container.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from linkrag.core import EvidenceUnit

CITATION_DIR = Path("data/processed/citations")
CROP_DPI = 110
CROP_PAD_PT = 6.0


@dataclass
class Citation:
    unit_id: str
    modality: str
    where: str                      # human-readable locator: "slides.pdf p.12" / "talk.mp3 12:44-13:31"
    path: Path | None = None        # crop, figure image or audio clip; None if it could not be made
    note: str = ""                  # why there is no path

    def render(self) -> str:
        tail = f" -> {self.path}" if self.path else (f" ({self.note})" if self.note else "")
        return f"[{self.unit_id}] {self.where}{tail}"


def mmss(seconds: float) -> str:
    s = int(round(seconds))
    return f"{s // 60}:{s % 60:02d}"


def locator(unit: EvidenceUnit) -> str:
    name = Path(unit.source_file).name
    loc = unit.location
    if loc.page is not None:
        return f"{name} p.{loc.page}"
    if loc.start_s is not None:
        return f"{name} {mmss(loc.start_s)}-{mmss(loc.end_s or loc.start_s)}"
    return name


def crop_page(unit: EvidenceUnit, out_dir: Path = CITATION_DIR, dpi: int = CROP_DPI) -> Path | None:
    """PNG of the unit's bbox region on its page (whole page when no bbox)."""
    import pymupdf
    src = Path(unit.source_file)
    if unit.location.page is None or src.suffix.lower() != ".pdf" or not src.exists():
        return None
    out = out_dir / f"{unit.id.replace(':', '_')}.png"
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    with pymupdf.open(src) as doc:
        page = doc[unit.location.page - 1]
        clip = None
        if unit.location.bbox:
            x0, y0, x1, y1 = unit.location.bbox
            clip = pymupdf.Rect(x0 - CROP_PAD_PT, y0 - CROP_PAD_PT, x1 + CROP_PAD_PT, y1 + CROP_PAD_PT) & page.rect
        pix = page.get_pixmap(clip=clip, dpi=dpi)
        if pix.colorspace is None or pix.colorspace.n not in (1, 3):
            pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
        pix.save(out)
    return out


def clip_audio(unit: EvidenceUnit, out_dir: Path = CITATION_DIR, pad_s: float = 0.25) -> Path | None:
    """Cut [start-pad, end+pad] from the source media into an .m4a next to the crops."""
    import av
    src = Path(unit.source_file)
    if unit.location.start_s is None or not src.exists():
        return None
    out = out_dir / f"{unit.id.replace(':', '_')}.m4a"
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    start = max(0.0, float(unit.location.start_s) - pad_s)
    end = float(unit.location.end_s or start) + pad_s
    with av.open(str(src)) as inp:
        stream = inp.streams.audio[0]
        with av.open(str(out), "w") as outp:
            # A decoder can report a generic layout ("1 channels"); the AAC encoder
            # rejects it. Name it by channel count instead of copying the source.
            channels = getattr(stream.layout, "nb_channels", None) or stream.codec_context.channels or 1
            ostream = outp.add_stream("aac", rate=stream.codec_context.sample_rate or 44100)
            ostream.layout = "mono" if channels == 1 else "stereo"
            resampler = av.audio.resampler.AudioResampler(format=ostream.format, layout=ostream.layout,
                                                          rate=ostream.rate)
            inp.seek(int(start / stream.time_base), stream=stream)
            for frame in inp.decode(stream):
                t = float(frame.pts * stream.time_base) if frame.pts is not None else 0.0
                if t < start:
                    continue
                if t > end:
                    break
                frame.pts = None
                for resampled in resampler.resample(frame):
                    for packet in ostream.encode(resampled):
                        outp.mux(packet)
            for packet in ostream.encode(None):
                outp.mux(packet)
    return out


def cite(unit: EvidenceUnit, out_dir: Path = CITATION_DIR, *, media: bool = True) -> Citation:
    """One inspectable citation for one unit."""
    c = Citation(unit_id=unit.id, modality=unit.modality, where=locator(unit))
    if not media:
        return c
    try:
        if unit.modality == "audio":
            c.path = clip_audio(unit, out_dir)
        elif unit.modality in ("figure", "table"):
            img = (unit.metadata or {}).get("image_path")
            c.path = Path(img) if img and Path(img).exists() else crop_page(unit, out_dir)
        else:
            c.path = crop_page(unit, out_dir)
    except Exception as exc:                      # a citation must never break an answer
        c.note = f"{type(exc).__name__}: {exc}"
        return c
    if c.path is None:
        c.note = "source file not available"
    return c


def citations_for(units: list[EvidenceUnit], cited_ids: list[str] | None = None,
                  out_dir: Path = CITATION_DIR, *, media: bool = True) -> dict[str, Citation]:
    """id -> Citation for the cited units (all of them when `cited_ids` is None)."""
    by_id = {u.id: u for u in units}
    ids = list(by_id) if cited_ids is None else [i for i in cited_ids if i in by_id]
    return {i: cite(by_id[i], out_dir, media=media) for i in ids}
