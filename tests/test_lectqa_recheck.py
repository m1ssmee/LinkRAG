"""By-video resampling: questions on one video are not independent draws."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "adapters"))
from lectqa import recheck  # noqa: E402


def test_by_video_interval_is_wider_when_a_video_moves_together():
    d = np.repeat([4.0, -2.0, 3.0, -3.0, 5.0, -1.0], 20)         # six videos, twenty identical questions each
    vids = np.repeat(list("abcdef"), 20)
    lo, hi = recheck.by_video(d, vids)
    qlo, qhi = np.percentile(np.random.default_rng(0).choice(d, (2000, len(d))).mean(axis=1), [2.5, 97.5])
    assert lo < d.mean() < hi
    assert hi - lo > 2 * (qhi - qlo)
    one_each = recheck.by_video(d, np.arange(len(d)))             # one question per video: back to by-question
    assert abs((one_each[1] - one_each[0]) - (qhi - qlo)) < 0.2
