"""OpenAI whisper-1 ASR backend: word timestamps, silence-aware chunking.

`ingest.asr_backend: openai` sends the audio to `/v1/audio/transcriptions` with
`response_format=verbose_json` and `timestamp_granularities[]=word`, so the result is
the same `list[Word]` the local faster-whisper path produces and **the sentence-split
path downstream is unchanged**.

Files above the API's 25 MB limit are split into <= `max_upload_mb` pieces. The cut
points are not arbitrary: each target boundary is moved to the quietest 0.5 s window
within `chunk_search_s` of it, so a cut lands between utterances rather than through
a word. Each chunk is transcribed independently and its word timestamps are offset by
the chunk's start, so the merged transcript is on the original timeline.

Splitting and cutting use **PyAV** (the ffmpeg libraries, via Python bindings) rather
than the `ffmpeg` CLI -- PyAV is already a dependency and needs no system binary.

Cost is per audio minute, not per token; `linkrag.costs` prices it from
`models.pricing.whisper-1.per_minute` and the caller's `--max-cost` cap applies.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

import re

from linkrag.core import stage_timer
from linkrag.ingest.audio import Word

TRANSCRIBE_PATH = "/audio/transcriptions"
SILENCE_WINDOW_S = 0.5          # width of the quiet window we look for at a cut point
_CORE = re.compile(r"[^\w']+")  # strip punctuation to compare a word with a segment token


@dataclass
class Chunk:
    path: Path
    offset_s: float
    duration_s: float


# ----------------------------------------------------------------- splitting

def audio_duration(path: str | Path) -> float:
    import av
    with av.open(str(path)) as container:
        if container.duration:
            return float(container.duration) / 1_000_000.0
        stream = container.streams.audio[0]
        return float((stream.duration or 0) * stream.time_base)


def quiet_offsets(path: str | Path, targets: list[float], search_s: float) -> list[float]:
    """Move each target time to the quietest `SILENCE_WINDOW_S` window within
    +/- `search_s` of it. One decode pass; RMS is accumulated per 0.5 s bucket."""
    import av
    import numpy as np

    if not targets:
        return []
    lo, hi = min(targets) - search_s, max(targets) + search_s
    energies: dict[int, list[float]] = {}
    with av.open(str(path)) as container:
        stream = container.streams.audio[0]
        for frame in container.decode(stream):
            t = float(frame.pts * stream.time_base) if frame.pts is not None else 0.0
            if t < lo:
                continue
            if t > hi:
                break
            samples = frame.to_ndarray().astype("float32")
            if samples.size:
                energies.setdefault(int(t / SILENCE_WINDOW_S), []).append(float(np.sqrt((samples ** 2).mean())))
    out = []
    for target in targets:
        lo_b, hi_b = int((target - search_s) / SILENCE_WINDOW_S), int((target + search_s) / SILENCE_WINDOW_S)
        buckets = [(sum(v) / len(v), b) for b, v in energies.items() if lo_b <= b <= hi_b and v]
        out.append(min(buckets)[1] * SILENCE_WINDOW_S if buckets else target)
    return out


def cut(path: str | Path, start: float, end: float, out: Path) -> Path:
    """[start, end) of the source into an .m4a, re-encoded to AAC."""
    import av
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        return out
    with av.open(str(path)) as inp:
        stream = inp.streams.audio[0]
        with av.open(str(out), "w") as outp:
            # A decoder can report a generic layout ("1 channels"); the AAC encoder
            # rejects it. Name it by channel count instead of copying the source.
            channels = getattr(stream.layout, "nb_channels", None) or stream.codec_context.channels or 1
            ostream = outp.add_stream("aac", rate=stream.codec_context.sample_rate or 44100)
            ostream.layout = "mono" if channels == 1 else "stereo"
            resampler = av.audio.resampler.AudioResampler(format=ostream.format, layout=ostream.layout,
                                                          rate=ostream.rate)
            if start > 0:
                inp.seek(int(start / stream.time_base), stream=stream)
            for frame in inp.decode(stream):
                t = float(frame.pts * stream.time_base) if frame.pts is not None else 0.0
                if t < start:
                    continue
                if t >= end:
                    break
                frame.pts = None
                for resampled in resampler.resample(frame):
                    for packet in ostream.encode(resampled):
                        outp.mux(packet)
            for packet in ostream.encode(None):
                outp.mux(packet)
    return out


def split_audio(path: str | Path, out_dir: Path, *, max_upload_mb: int = 20,
                search_s: float = 30.0) -> list[Chunk]:
    """The file as one chunk when it is small enough, else silence-aligned pieces."""
    path = Path(path)
    size_mb = path.stat().st_size / 1e6
    duration = audio_duration(path)
    if size_mb <= max_upload_mb:
        return [Chunk(path=path, offset_s=0.0, duration_s=duration)]

    n = math.ceil(size_mb / max_upload_mb)
    step = duration / n
    targets = [step * i for i in range(1, n)]
    cuts = [0.0] + sorted(set(quiet_offsets(path, targets, search_s))) + [duration]
    chunks = []
    for i, (a, b) in enumerate(zip(cuts, cuts[1:])):
        if b - a < 1.0:
            continue
        chunks.append(Chunk(path=cut(path, a, b, out_dir / f"{path.stem}.chunk{i:02d}.m4a"),
                            offset_s=a, duration_s=b - a))
    return chunks


# ----------------------------------------------------------------- transcription

def transcribe_chunk(chunk: Chunk, cfg: dict[str, Any], *, prompt: str | None = None) -> tuple[list[Word], dict]:
    acfg = cfg.get("ingest", {}).get("asr_openai", {}) or {}
    llm = cfg.get("models", {}).get("llm", {})
    base = (llm.get("base_url") or "https://api.openai.com/v1").rstrip("/")
    key_env = llm.get("api_key_env", "OPENAI_API_KEY")
    key = os.environ.get(key_env)
    if not key:
        raise RuntimeError(f"{key_env} is not set (models.llm.api_key_env)")
    model = acfg.get("model", "whisper-1")

    # BOTH granularities: whisper-1's word timestamps carry NO punctuation, and the
    # sentence splitter downstream cuts on terminal punctuation -- with words alone the
    # whole talk becomes one segment (found on pilot01-w1, 1 audio unit instead of 51).
    # The segment texts are punctuated, so the punctuation is re-attached below.
    data = [("model", model), ("response_format", "verbose_json"),
            ("timestamp_granularities[]", "word"), ("timestamp_granularities[]", "segment")]
    if prompt:
        data.append(("prompt", prompt[:1000]))   # whisper conditions on ~224 tokens
    with open(chunk.path, "rb") as fh:
        response = requests.post(f"{base}{TRANSCRIBE_PATH}",
                                 headers={"Authorization": f"Bearer {key}"},
                                 files={"file": (chunk.path.name, fh, "audio/m4a")},
                                 data=data, timeout=acfg.get("timeout_s", 600))
    if response.status_code != 200:
        raise RuntimeError(f"whisper-1 returned {response.status_code}: {response.text[:300]}")
    body = response.json()
    raw = [(float(w["start"]), float(w["end"]), str(w["word"])) for w in body.get("words") or []]
    punctuated = restore_punctuation(raw, body.get("segments") or [])
    words = [Word(a + chunk.offset_s, b + chunk.offset_s, t) for a, b, t in punctuated]
    return words, {"model": model, "duration_s": float(body.get("duration") or chunk.duration_s),
                   "segments": len(body.get("segments") or [])}


def restore_punctuation(words: list[tuple[float, float, str]], segments: list[dict]
                        ) -> list[tuple[float, float, str]]:
    """Re-attach the punctuation the word-level response drops.

    The segment texts are the same token sequence *with* punctuation, so walk the two
    in step: a word adopts the trailing punctuation of the segment token whose
    alphanumeric core matches it. A mismatch (rare: a hyphenation or a number written
    differently) skips forward rather than shifting everything after it."""
    tokens: list[str] = []
    for seg in segments:
        tokens.extend(str(seg.get("text", "")).split())
    if not tokens:
        return words
    out, j = [], 0
    for start, end, word in words:
        core = _CORE.sub("", word).lower()
        match = None
        for k in range(j, min(j + 4, len(tokens))):       # small look-ahead, never a global resync
            if _CORE.sub("", tokens[k]).lower() == core:
                match, j = tokens[k], k + 1
                break
        out.append((start, end, match if match else word))
    return out


def transcribe(path: str | Path, cfg: dict[str, Any], *, prompt: str | None = None,
               work_dir: Path | None = None) -> tuple[list[Word], dict]:
    """Whole file -> words on the original timeline, plus provenance for the transcript.

    Returns (words, meta) where meta carries the model, the request date, the chunk
    layout and the billed audio seconds -- so a frozen transcript records exactly how
    it was produced."""
    path = Path(path)
    acfg = cfg.get("ingest", {}).get("asr_openai", {}) or {}
    work_dir = work_dir or Path("data/processed/asr_chunks")
    with stage_timer("ingest.asr_openai", file=path.name) as t:
        chunks = split_audio(path, work_dir, max_upload_mb=int(acfg.get("max_upload_mb", 20)),
                             search_s=float(acfg.get("chunk_search_s", 30.0)))
        t["chunks"] = len(chunks)
        words: list[Word] = []
        billed = 0.0
        for chunk in chunks:
            got, info = transcribe_chunk(chunk, cfg, prompt=prompt)
            words.extend(got)
            billed += info["duration_s"]
        words.sort(key=lambda w: w.start)
        t["words"] = len(words)
        t["audio_seconds"] = round(billed, 1)
    meta = {"asr_backend": "openai", "model": acfg.get("model", "whisper-1"),
            "requested_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "chunks": [{"offset_s": round(c.offset_s, 2), "duration_s": round(c.duration_s, 2),
                        "file": c.path.name} for c in chunks],
            "audio_seconds": round(billed, 1), "initial_prompt": prompt or ""}
    return words, meta


def write_transcript(words: list[Word], path: str | Path, audio: str | Path, meta: dict) -> Path:
    """Same schema the local backend freezes, plus the backend's provenance."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"audio": str(audio), "words": [[w.start, w.end, w.text] for w in words], **meta}
    path.write_text(json.dumps(payload))
    return path
