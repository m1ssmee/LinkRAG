from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from linkrag.core import EvidenceUnit, Location
from linkrag.index import Index, build_index, embeddable_text, tokenize

CORPUS = [
    ("u0", "text", "self attention assigns each token a weight over every other token"),
    ("u1", "text", "positional encodings restore the order information attention discards"),
    ("u2", "text", "the optimizer uses a cosine learning rate schedule with warmup"),
    ("u3", "figure", "attention heatmap for layer six head three"),
]


@pytest.fixture
def units() -> list[EvidenceUnit]:
    return [
        EvidenceUnit(
            id=uid,
            modality=modality,
            content=content,
            source_file="notes.pdf",
            location=Location(page=i + 1),
        )
        for i, (uid, modality, content) in enumerate(CORPUS)
    ]


@pytest.fixture
def index(units: list[EvidenceUnit], stub_encoder) -> Index:
    encode = stub_encoder([u.content for u in units])
    return build_index(units, encoder=encode, embedding_model="stub")


def test_tokenize_strips_punctuation() -> None:
    assert tokenize("Hello, World! attention-weights") == ["hello", "world", "attention", "weights"]


def test_embeddable_text_falls_back_for_a_captionless_figure() -> None:
    bare = EvidenceUnit(
        id="f0", modality="figure", content="  ", source_file="a/slides.pdf",
        location=Location(page=14),
    )
    text = embeddable_text(bare)
    assert "figure" in text and "slides.pdf" in text and "p.14" in text


def test_build_index_rejects_an_empty_corpus() -> None:
    with pytest.raises(ValueError, match="zero units"):
        build_index([], encoder=lambda texts: np.zeros((0, 3), dtype="float32"))


def test_build_index_rejects_a_mismatched_encoder(units: list[EvidenceUnit]) -> None:
    with pytest.raises(ValueError, match="expected"):
        build_index(units, encoder=lambda texts: np.zeros((2, 3), dtype="float32"))


def test_dense_search_ranks_the_matching_unit_first(index: Index, stub_encoder) -> None:
    encode = stub_encoder([u.content for u in index.units])
    query = encode(["attention heatmap layer six head three"])[0]
    (position, score), *_ = index.dense_search(query, k=4)
    assert index.units[position].id == "u3"
    assert score > 0.9, "near-identical text should be a near-1.0 cosine"


def test_sparse_search_finds_the_lexical_match(index: Index) -> None:
    (position, _score), *_ = index.sparse_search("cosine learning rate warmup", k=4)
    assert index.units[position].id == "u2"


def test_search_k_is_clamped_to_the_corpus_size(index: Index, stub_encoder) -> None:
    encode = stub_encoder([u.content for u in index.units])
    assert len(index.dense_search(encode(["attention"])[0], k=99)) == len(index)
    assert len(index.sparse_search("attention", k=99)) == len(index)


def test_index_round_trips_through_disk(index: Index, tmp_path: Path, stub_encoder) -> None:
    index.save(tmp_path / "idx")
    loaded = Index.load(tmp_path / "idx")

    assert [u.id for u in loaded.units] == [u.id for u in index.units]
    assert loaded.units[0].location.page == 1
    assert loaded.embedding_model == "stub"
    assert loaded.id_to_pos["u3"] == 3

    encode = stub_encoder([u.content for u in index.units])
    query = encode(["attention heatmap layer six head three"])[0]
    assert loaded.dense_search(query, 1) == index.dense_search(query, 1)
    assert loaded.sparse_search("cosine warmup", 1)[0][0] == index.sparse_search("cosine warmup", 1)[0][0]


def test_saved_units_carry_no_embedding_copy(index: Index, tmp_path: Path) -> None:
    import json

    index.save(tmp_path / "idx")
    records = json.loads((tmp_path / "idx" / "units.json").read_text())
    assert all("embedding" not in r for r in records), "vectors belong to FAISS only"


def test_loaded_bbox_is_a_tuple(tmp_path: Path, stub_encoder) -> None:
    unit = EvidenceUnit(
        id="t0", modality="text", content="alpha beta", source_file="a.pdf",
        location=Location(page=1, bbox=(1.0, 2.0, 3.0, 4.0)),
    )
    build_index([unit], encoder=stub_encoder(["alpha beta"])).save(tmp_path / "idx")
    assert Index.load(tmp_path / "idx").units[0].location.bbox == (1.0, 2.0, 3.0, 4.0)


def test_build_index_rejects_duplicate_ids(stub_encoder) -> None:
    """A duplicate id silently overwrites id_to_pos and breaks citation."""
    units = [
        EvidenceUnit(id="dup", modality="text", content=c, source_file="a.pdf",
                     location=Location(page=i))
        for i, c in enumerate(["alpha", "beta"])
    ]
    with pytest.raises(ValueError, match="duplicate unit ids"):
        build_index(units, encoder=stub_encoder(["alpha", "beta"]))


def test_dense_search_is_exact(stub_encoder) -> None:
    """Dense search must be exact brute force, matching a full argsort.

    This is what made faiss droppable: IndexFlatIP was doing the same matmul,
    while its bundled OpenMP runtime clashed with torch's and aborted the
    process. If dense search is ever swapped for an approximate index, this
    test should be replaced with a recall@k check, not deleted.
    """
    corpus = [f"token{i} shared" for i in range(50)]
    units = [
        EvidenceUnit(id=f"u{i}", modality="text", content=c, source_file="a.pdf",
                     location=Location(page=1))
        for i, c in enumerate(corpus)
    ]
    encode = stub_encoder(corpus)
    index = build_index(units, encoder=encode)

    query = encode(["token7 shared"])[0]
    got = index.dense_search(query, k=5)
    # Compare scores, not positions: most of this corpus ties, and tie-break
    # order is not a promise the retriever makes.
    expected = sorted(index.vectors @ query, reverse=True)[:5]
    assert [s for _, s in got] == pytest.approx(expected)
    assert index.units[got[0][0]].id == "u7"


def test_vectors_persist_as_npy(index: Index, tmp_path: Path) -> None:
    index.save(tmp_path / "idx")
    assert (tmp_path / "idx" / "vectors.npy").exists()
    loaded = Index.load(tmp_path / "idx")
    assert np.allclose(loaded.vectors, index.vectors)
