from __future__ import annotations

import pytest

from linkrag.core import EvidenceUnit, Location
from linkrag.index import build_index
from linkrag.retrieve.baseline import retrieve_scored, rrf_fuse

CORPUS = [
    ("u0", "text", "self attention assigns each token a weight over every other token"),
    ("u1", "text", "positional encodings restore the order information attention discards"),
    ("u2", "text", "the optimizer uses a cosine learning rate schedule with warmup"),
    ("u3", "figure", "attention heatmap for layer six head three"),
    ("u4", "audio", "and as you can see here the attention weights concentrate on the subject"),
]


@pytest.fixture
def index(stub_encoder):
    units = [
        EvidenceUnit(id=uid, modality=m, content=c, source_file="corpus.pdf",
                     location=Location(page=i + 1))
        for i, (uid, m, c) in enumerate(CORPUS)
    ]
    return build_index(units, encoder=stub_encoder([u.content for u in units]),
                       embedding_model="stub")


@pytest.fixture
def encode(stub_encoder):
    return stub_encoder([c for _, _, c in CORPUS])


# ----------------------------------------------------------------------- RRF

def test_rrf_uses_rank_not_score() -> None:
    # B is rank 2 in both lists; A is rank 1 in one and absent from the other.
    # RRF must prefer the consistently-high item despite A's huge raw score.
    fused = dict(rrf_fuse([[(0, 999.0), (1, 0.1)], [(2, 5.0), (1, 4.9)]]))
    assert fused[1] > fused[0]


def test_rrf_sums_across_rankers() -> None:
    fused = dict(rrf_fuse([[(7, 1.0)], [(7, 1.0)]], rrf_k=60))
    assert fused[7] == pytest.approx(2 / 61)


def test_rrf_is_deterministic_on_ties() -> None:
    fused = rrf_fuse([[(5, 1.0), (3, 1.0)], [(3, 1.0), (5, 1.0)]])
    assert [position for position, _ in fused] == [3, 5], "ties break on position, ascending"


def test_rrf_on_empty_rankings() -> None:
    assert rrf_fuse([]) == []
    assert rrf_fuse([[], []]) == []


# ----------------------------------------------------------------- retrieval

def test_retrieve_returns_top_k_units(index, encode) -> None:
    units = [u for u, _ in retrieve_scored("attention heatmap layer six", index, encoder=encode, top_k=3)]
    assert len(units) == 3
    assert all(isinstance(u, EvidenceUnit) for u in units)
    assert len(set(u.id for u in units)) == 3, "no duplicates across the two rankers"


def test_retrieve_ranks_the_relevant_unit_first(index, encode) -> None:
    units = [u for u, _ in retrieve_scored("attention heatmap for layer six head three", index, encoder=encode, top_k=2)]
    assert units[0].id == "u3"


def test_retrieve_is_modality_blind(index, encode) -> None:
    """The baseline has one shared store; an audio unit can outrank a text one.
    This is the property the linkrag mode must beat, not a bug."""
    units = [u for u, _ in retrieve_scored("as you can see here the weights concentrate", index, encoder=encode, top_k=1)]
    assert units[0].modality == "audio"


def test_retrieve_scored_is_descending(index, encode) -> None:
    scored = retrieve_scored("attention", index, encoder=encode, top_k=5)
    scores = [s for _, s in scored]
    assert scores == sorted(scores, reverse=True)


def test_retrieve_k_larger_than_corpus(index, encode) -> None:
    assert len([u for u, _ in retrieve_scored("attention", index, encoder=encode, top_k=99)]) == len(index)


def test_retrieve_makes_no_links(index, encode) -> None:
    """Guard the baseline's defining property: it never expands beyond top-k."""
    units = [u for u, _ in retrieve_scored("attention", index, encoder=encode, top_k=2)]
    assert len(units) == 2
