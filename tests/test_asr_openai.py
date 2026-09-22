"""OpenAI whisper-1 ASR backend: chunking at silence, timestamp offsetting, freezing."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from linkrag.ingest import asr_openai as A
from linkrag.ingest.audio import Word, ingest_audio


def _tone(path: Path, seconds: float, quiet_at: list[float] | None = None, rate: int = 8000):
    """A sine wave with 1-second silences at `quiet_at`, written as wav."""
    import av
    t = np.arange(int(seconds * rate)) / rate
    sig = (0.5 * np.sin(2 * np.pi * 440 * t)).astype("float32")
    for q in quiet_at or []:
        sig[int(q * rate):int((q + 1.0) * rate)] = 0.0
    with av.open(str(path), "w") as out:
        stream = out.add_stream("pcm_s16le", rate=rate)
        stream.layout = "mono"
        frame = av.AudioFrame.from_ndarray((sig * 32767).astype("int16").reshape(1, -1), format="s16", layout="mono")
        frame.rate = rate
        for packet in stream.encode(frame):
            out.mux(packet)
        for packet in stream.encode(None):
            out.mux(packet)
    return path


def test_small_file_is_one_chunk_and_is_not_copied(tmp_path):
    src = _tone(tmp_path / "a.wav", 3.0)
    chunks = A.split_audio(src, tmp_path / "out", max_upload_mb=20)
    assert len(chunks) == 1 and chunks[0].path == src and chunks[0].offset_s == 0.0


def test_cut_points_land_on_the_quiet_moments(tmp_path):
    src = _tone(tmp_path / "b.wav", 30.0, quiet_at=[9.5, 19.5])
    got = A.quiet_offsets(src, [10.0, 20.0], search_s=3.0)
    assert all(abs(g - q) <= 1.0 for g, q in zip(got, [9.5, 19.5])), got


def test_large_file_splits_and_chunks_tile_the_timeline(tmp_path):
    src = _tone(tmp_path / "c.wav", 40.0, quiet_at=[19.5])      # ~640 KB at 8 kHz s16
    chunks = A.split_audio(src, tmp_path / "out", max_upload_mb=0.3, search_s=3.0)
    assert len(chunks) >= 2
    assert chunks[0].offset_s == 0.0
    for a, b in zip(chunks, chunks[1:]):                        # no gap, no overlap
        assert math.isclose(a.offset_s + a.duration_s, b.offset_s, abs_tol=0.05)
    assert all(c.path.exists() for c in chunks)


def test_word_timestamps_are_offset_by_the_chunk_start(monkeypatch, tmp_path):
    """The API returns times relative to the chunk; the merged transcript must be on
    the original timeline."""
    replies = iter([{"words": [{"start": 0.0, "end": 1.0, "word": "first"}], "duration": 10.0},
                    {"words": [{"start": 0.5, "end": 1.5, "word": "second"}], "duration": 10.0}])

    class R:
        status_code = 200

        def json(self):
            return next(replies)
    monkeypatch.setattr(A.requests, "post", lambda *a, **k: R())
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    monkeypatch.setattr(A, "split_audio", lambda *a, **k: [
        A.Chunk(path=tmp_path / "x.m4a", offset_s=0.0, duration_s=10.0),
        A.Chunk(path=tmp_path / "y.m4a", offset_s=10.0, duration_s=10.0)])
    (tmp_path / "x.m4a").write_bytes(b"0")
    (tmp_path / "y.m4a").write_bytes(b"0")
    words, meta = A.transcribe(tmp_path / "src.mp3", {"models": {"llm": {}}}, work_dir=tmp_path)
    assert [(w.start, w.text) for w in words] == [(0.0, "first"), (10.5, "second")]
    assert meta["audio_seconds"] == 20.0 and meta["model"] == "whisper-1" and meta["requested_utc"]


def test_transcript_records_backend_and_date_and_is_reusable(tmp_path):
    words = [Word(0.0, 1.0, "hello"), Word(1.0, 2.0, "world")]
    p = A.write_transcript(words, tmp_path / "t.json", "a.mp3",
                           {"asr_backend": "openai", "model": "whisper-1", "requested_utc": "2026-09-23T00:00:00Z"})
    d = json.loads(p.read_text())
    assert d["asr_backend"] == "openai" and d["model"] == "whisper-1" and d["requested_utc"]
    from linkrag.ingest.audio import load_frozen_transcript
    assert [w.text for w in load_frozen_transcript(p)] == ["hello", "world"]


def test_unknown_backend_is_rejected_before_any_work(tmp_path):
    with pytest.raises(ValueError, match="unknown asr_backend"):
        ingest_audio(tmp_path / "missing.mp3", backend="deepgram")


@pytest.mark.skipif(not Path("data/processed/transcripts/hsieh.openai.json").exists(),
                    reason="pilot01 openai transcript not produced")
def test_pilot01_openai_run_is_on_the_original_timeline():
    d = json.loads(Path("data/processed/transcripts/hsieh.openai.json").read_text())
    assert len(d["chunks"]) == 2, "21.9 MB must split at the 20 MB limit"
    assert d["words"][-1][1] > 1300, "the merged transcript must span the whole 22.9-minute talk"
    assert Path("data/processed/transcripts/hsieh.frozen.json").exists(), "the frozen transcript stays"
