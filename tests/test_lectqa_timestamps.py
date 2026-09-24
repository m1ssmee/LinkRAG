"""LectQA-Vid gold timestamps: only unambiguous stamps become gold intervals."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "adapters"))
from lectqa import common as L  # noqa: E402


def test_strict_seconds_accepts_only_unambiguous_forms():
    assert L.strict_seconds("00:01:20") == 80.0          # HH:MM:SS
    assert L.strict_seconds("01:20") == 80.0             # MM:SS
    assert L.strict_seconds("210.72") == 210.72          # plain seconds
    # forms seen in the published files whose meaning differs between videos
    assert L.strict_seconds("00:12:70") is None          # SS:cc
    assert L.strict_seconds("258:36") == 258 * 60 + 36   # parses as MM:SS; gold_interval's range check rejects it
    assert L.strict_seconds("01:91:60") is None          # M:SSS:cc
    assert L.strict_seconds("00:258:36") is None
    assert L.strict_seconds("abc") is None


def test_gold_interval_rejects_out_of_range_and_reversed(monkeypatch):
    monkeypatch.setitem(L._DURATION, "video_x", 272.0)
    assert L.gold_interval("video_x", {"timestamp_start": "00:00:09", "timestamp_end": "00:00:13"}) == ((9.0, 13.0), "ok")
    assert L.gold_interval("video_x", {"timestamp_start": "00:00:09", "timestamp_end": "00:00:09"}) == ((9.0, 10.0), "ok")
    assert L.gold_interval("video_x", {"timestamp_start": "4:18:36", "timestamp_end": "4:20:00"})[1] == \
        "beyond the fetched video's end"
    assert L.gold_interval("video_x", {"timestamp_start": "258:36", "timestamp_end": "260:00"})[1] == \
        "beyond the fetched video's end"
    assert L.gold_interval("video_x", {"timestamp_start": "00:12:70", "timestamp_end": "00:22:00"})[1] == "ambiguous format"
    assert L.gold_interval("video_x", {"timestamp_start": "00:00:20", "timestamp_end": "00:00:10"})[1] == "end before start"
