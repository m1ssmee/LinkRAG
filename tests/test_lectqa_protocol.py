"""LectQA-Vid comparison protocol: shuffled MCQ options and the T1 retrieval replica."""

from __future__ import annotations

import collections
import sys
from pathlib import Path

import numpy as np

from linkrag.core import EvidenceUnit, Location

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "adapters"))
from lectqa import mcq, open_ended  # noqa: E402


def test_shuffled_options_are_seeded_and_move_the_gold():
    L = mcq
    q = {"options": ["right", "w1", "w2", "w3"], "answer": "right"}
    assert L.shuffled(q, "video_1", 0) == L.shuffled(q, "video_1", 0)           # fixed seed
    pos = collections.Counter(L.shuffled(q, f"video_{v}", i)[1] for v in range(1, 21) for i in range(10))
    assert set(pos) == {0, 1, 2, 3} and max(pos.values()) < 100                 # no longer always A
    opts, gold = L.shuffled(q, "video_3", 2)
    assert opts[gold] == "right" and sorted(opts) == sorted(q["options"])


class FakeCE:
    def predict(self, pairs):                          # prefers longer text
        return [len(c) for _, c in pairs]


def test_t1_retrieve_threshold_oracle_filter_merge_and_top_l():
    L = open_ended
    unit = lambda n, a, b, text: EvidenceUnit(id=f"v:c{n}", modality="audio", content=text, source_file="v.m4a",
                                              location=Location(start_s=a, end_s=b))
    chunks = [unit(0, 0, 10, "a"), unit(1, 12, 20, "bb"), unit(2, 40, 50, "ccc"), unit(3, 90, 99, "dddd")]
    vecs = np.eye(4, dtype=np.float32)
    enc = lambda texts: [np.array([0.9, 0.8, 0.7, 0.5], dtype=np.float32)]   # sims to the query
    # no gold interval: theta_r 0.60 drops c3; c0 and c1 are 2 s apart (< delta_gap 5 s) and merge
    got = L.t1_retrieve("q", None, chunks, vecs, enc, FakeCE())
    assert [u.id for u in got] == ["v:c0-c1", "v:c2"]
    assert got[0].content == "a bb" and (got[0].location.start_s, got[0].location.end_s) == (0, 20)
    # eq. 22: only chunks overlapping the gold interval +- 3 s survive
    got = L.t1_retrieve("q", (44.0, 45.0), chunks, vecs, enc, FakeCE())
    assert [u.id for u in got] == ["v:c2"]
    # a gold interval nothing retrieved overlaps leaves no context
    assert L.t1_retrieve("q", (95.0, 96.0), chunks, vecs, enc, FakeCE()) == []
