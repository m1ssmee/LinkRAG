"""LectQA-Vid comparison protocol: shuffled MCQ options."""

from __future__ import annotations

import collections

from test_lectqa_timestamps import load


def test_shuffled_options_are_seeded_and_move_the_gold():
    L = load()
    q = {"options": ["right", "w1", "w2", "w3"], "answer": "right"}
    assert L.shuffled(q, "video_1", 0) == L.shuffled(q, "video_1", 0)           # fixed seed
    pos = collections.Counter(L.shuffled(q, f"video_{v}", i)[1] for v in range(1, 21) for i in range(10))
    assert set(pos) == {0, 1, 2, 3} and max(pos.values()) < 100                 # no longer always A
    opts, gold = L.shuffled(q, "video_3", 2)
    assert opts[gold] == "right" and sorted(opts) == sorted(q["options"])
