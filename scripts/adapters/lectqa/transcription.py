"""Whisper transcript + changed frames with OCR -> the per-video index."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from linkrag.core import EvidenceUnit, Location
from linkrag.index import build_index, default_encoder
from linkrag.manifest import MANIFEST_NAME, build_manifest, write_manifest

from lectqa.common import PROCESSED, RAW


def extract_frames(video: Path, out_dir: Path, every_s: float, min_change: float = 0.08) -> list[tuple[float, Path]]:
    """One frame every `every_s` seconds, kept only if it differs from the last kept
    frame (mean absolute pixel difference on a 64x36 grey thumbnail > min_change)."""
    import av
    out_dir.mkdir(parents=True, exist_ok=True)
    kept: list[tuple[float, Path]] = []
    last_thumb = None
    next_t = 0.0
    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            t = float(frame.pts * stream.time_base) if frame.pts is not None else 0.0
            if t + 1e-6 < next_t:
                continue
            next_t = t + every_s
            img = frame.to_image()
            thumb = np.asarray(img.convert("L").resize((64, 36)), dtype=np.float32) / 255.0
            if last_thumb is not None and float(np.abs(thumb - last_thumb).mean()) < min_change:
                continue
            last_thumb = thumb
            path = out_dir / f"{video.stem}_{int(round(t)):05d}.png"
            if not path.exists():
                img.save(path)
            kept.append((t, path))
    return kept


def prepare(ids: list[str], cfg: dict, dcfg: dict) -> None:
    from linkrag.ingest.audio import ingest_audio
    from linkrag.ingest.image import ocr
    encoder = None
    for vid in ids:
        audio_path, video = RAW / "videos" / f"{vid}.m4a", RAW / "videos" / f"{vid}.video.mp4"
        index_dir = PROCESSED / vid / "index"
        if not audio_path.exists() or not video.exists():
            print(f"skip {vid}: not fetched")
            continue
        if index_dir.exists():
            continue
        t0 = time.perf_counter()
        frozen = PROCESSED / vid / "transcript.frozen.json"     # whisper runs once per video
        audio = ingest_audio(audio_path, model_size=cfg["models"]["whisper"], device=cfg["device"],
                             compute_type=cfg["models"]["whisper_compute_type"],
                             window_seconds=dcfg["audio_segment_seconds"], segmentation="sentence",
                             transcript=frozen if frozen.exists() else None,
                             freeze_to=None if frozen.exists() else frozen)
        frames = extract_frames(video, PROCESSED / vid / "frames", dcfg["frame_interval_s"])
        units = list(audio)
        for i, (t, path) in enumerate(frames):
            end = frames[i + 1][0] if i + 1 < len(frames) else t + dcfg["frame_interval_s"]
            text = " ".join(ocr(path).split())
            units.append(EvidenceUnit(id=f"{vid}:f{i}", modality="figure", content=text,
                                      source_file=str(audio_path), location=Location(start_s=t, end_s=end),
                                      metadata={"frame_path": str(path), "content_source": "ocr",
                                                "slide_deck": False}))
        if encoder is None:
            encoder = default_encoder(cfg["models"]["embedding"], cfg["device"], cfg["index"]["normalize_embeddings"])
            encoder([""])
        index = build_index(units, encoder=encoder, embedding_model=cfg["models"]["embedding"],
                            device=cfg["device"], normalize=cfg["index"]["normalize_embeddings"])
        index.save(index_dir)
        write_manifest(build_manifest([audio_path, video], units), PROCESSED / vid / MANIFEST_NAME)
        print(f"prepared {vid}: {len(audio)} audio + {len(frames)} frames ({time.perf_counter() - t0:.0f}s)")
