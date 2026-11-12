"""Pooled-corpus LectQA-Vid retrieval: stacking per-video indexes and scoring video / interval hits."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from linkrag.core import EvidenceUnit, Link, Location
from linkrag.index import Index, tokenize
from linkrag.retrieve.linkrag import RetrievedUnit

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "adapters"))
from lectqa import pooled  # noqa: E402


def _index(vid: str, texts: list[str], axis: int) -> Index:
    units = [EvidenceUnit(f"{vid}:a{i}", "audio", t, f"{vid}.m4a", Location(start_s=10.0 * i, end_s=10.0 * i + 10))
             for i, t in enumerate(texts)]
    vectors = np.zeros((len(units), 4), dtype="float32")
    vectors[:, axis] = 1.0
    return Index(units=units, vectors=vectors, bm25=BM25Okapi([tokenize(t) for t in texts]))


def test_pool_stacks_vectors_and_keeps_every_link():
    a, b = _index("video_1", ["alpha", "beta"], 0), _index("video_2", ["gamma", "delta"], 1)
    index, graph = pooled.pool([(a, [Link("video_1:a0", "video_1:a1", "audio_slide", 1.0)]),
                                (b, [Link("video_2:a0", "video_2:a1", "audio_slide", 1.0)])])
    assert [u.id for u in index.units] == ["video_1:a0", "video_1:a1", "video_2:a0", "video_2:a1"]
    assert index.vectors.shape == (4, 4) and graph.number_of_edges() == 2
    top = index.dense_search(np.array([0, 1, 0, 0], dtype="float32"), 2)       # video_2's direction
    assert {index.units[p].id.split(":")[0] for p, _ in top} == {"video_2"}
    assert index.units[index.sparse_search("delta", 1)[0][0]].id == "video_2:a1"


def test_hits_need_the_right_video_before_the_interval_counts():
    ix = _index("video_1", ["x", "y"], 0)
    other = _index("video_2", ["x", "y"], 0)
    r = lambda u: RetrievedUnit(unit=u, score=1.0, origin="seed")
    # rank 1: the wrong video at the right time; rank 2: the right video, wrong time; rank 3: right both
    picked = [r(other.units[1]), r(ix.units[0]), r(ix.units[1])]
    h = pooled.hits(picked, "video_1", (12.0, 15.0))
    assert (h["video@1"], h["video@3"]) == (0.0, 1.0)
    assert (h["interval@1"], h["interval@3"]) == (0.0, 1.0)
    assert pooled.hits(picked[:2], "video_1", (12.0, 15.0))["interval@8"] == 0.0     # overlap in the wrong video is no hit
    assert pooled.hits(picked, "video_1", None)["interval@8"] == 0.0 and pooled.hits(picked, "video_1", None)["video@8"] == 1.0
