"""Embed EvidenceUnits and store them for search.

baseline: one dense vector store plus one BM25 store over all units; each unit
          is an island, retrieved on its own similarity.
linkrag:  the same two stores, plus a link table so retrieval can traverse
          edges instead of stopping at the top-k.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from rank_bm25 import BM25Okapi

from linkrag.core import EvidenceUnit, Location, stage_timer

Encoder = Callable[[Sequence[str]], np.ndarray]
"""Anything that turns texts into a (n, dim) float matrix. Injectable so tests
run offline without pulling 2GB of bge-m3."""

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"


@lru_cache(maxsize=2)
def _sentence_transformer(name: str, device: str):
    from sentence_transformers import SentenceTransformer  # heavy, deferred

    return SentenceTransformer(name, device=device)


def default_encoder(
    model: str = DEFAULT_EMBEDDING_MODEL, device: str = "cpu", normalize: bool = True
) -> Encoder:
    def encode(texts: Sequence[str]) -> np.ndarray:
        st = _sentence_transformer(model, device)
        return np.asarray(
            st.encode(list(texts), normalize_embeddings=normalize, show_progress_bar=False),
            dtype="float32",
        )

    return encode


def embeddable_text(unit: EvidenceUnit) -> str:
    """What actually gets embedded.

    A figure whose caption was not found has empty content; embedding "" puts it
    at an arbitrary point in the space. Fall back to a descriptor so the unit is
    at least reachable by modality and provenance.
    """
    if unit.content.strip():
        return unit.content
    where = unit.location.cite()
    return f"{unit.modality} from {Path(unit.source_file).name} at {where}"


def tokenize(text: str) -> list[str]:
    # Note: lowercase split is the standard BM25 baseline. Add a stemmer only
    # if the eval shows sparse recall is the bottleneck.
    return [t for t in "".join(c if c.isalnum() else " " for c in text.lower()).split() if t]


@dataclass
class Index:
    """Exact dense search plus BM25.

    Note: dense search is a numpy matmul, not faiss. faiss's IndexFlatIP is
    also exact brute force, so it bought nothing here but a second OpenMP runtime
    -- faiss-cpu and torch each bundle libomp, and loading both aborts the process
    on macOS arm64 ("OMP: Error #15"), whose only documented workaround is
    flagged as possibly producing silently incorrect results. Measured 8.8 ms per
    query over 100k x 1024d, so the ceiling is ~200k units; past that, add an
    approximate index (HNSW via hnswlib, or faiss from conda-forge which links
    one shared libomp).
    """

    units: list[EvidenceUnit]
    vectors: np.ndarray              # (n, dim), row-normalized when normalize=True
    bm25: BM25Okapi
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    normalize: bool = True
    id_to_pos: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.id_to_pos = {u.id: i for i, u in enumerate(self.units)}

    def __len__(self) -> int:
        return len(self.units)

    def dense_search(self, query_vec: np.ndarray, k: int) -> list[tuple[int, float]]:
        """Exact inner product. On normalized vectors that is cosine similarity."""
        k = min(k, len(self.units))
        if k == 0:
            return []
        scores = self.vectors @ np.asarray(query_vec, dtype="float32").ravel()
        # argpartition is O(n) to find the top k, then sort just those k.
        top = np.argpartition(-scores, k - 1)[:k] if k < len(scores) else np.arange(len(scores))
        top = top[np.argsort(-scores[top])]
        return [(int(p), float(scores[p])) for p in top]

    def sparse_search(self, query: str, k: int) -> list[tuple[int, float]]:
        k = min(k, len(self.units))
        if k == 0:
            return []
        scores = self.bm25.get_scores(tokenize(query))
        top = np.argsort(scores)[::-1][:k]
        return [(int(p), float(scores[p])) for p in top]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "vectors.npy", self.vectors)
        # Embeddings live in vectors.npy only -- a second copy inside units.json
        # would double the store for nothing.
        payload = []
        for unit in self.units:
            record = asdict(unit)
            record.pop("embedding")
            payload.append(record)
        (path / "units.json").write_text(json.dumps(payload, ensure_ascii=False))
        (path / "meta.json").write_text(
            json.dumps({"embedding_model": self.embedding_model, "normalize": self.normalize})
        )

    @classmethod
    def load(cls, path: str | Path) -> Index:
        path = Path(path)
        meta = json.loads((path / "meta.json").read_text())
        units = []
        for record in json.loads((path / "units.json").read_text()):
            location = record.pop("location")
            if location.get("bbox") is not None:
                location["bbox"] = tuple(location["bbox"])
            units.append(EvidenceUnit(location=Location(**location), **record))
        return cls(
            units=units,
            vectors=np.load(path / "vectors.npy"),
            # Rebuilt rather than pickled: pickle of a third-party object is a
            # version trap, and re-fitting BM25 is milliseconds.
            bm25=BM25Okapi([tokenize(embeddable_text(u)) for u in units]),
            embedding_model=meta["embedding_model"],
            normalize=meta["normalize"],
        )


def build_index(
    units: list[EvidenceUnit],
    *,
    encoder: Encoder | None = None,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    device: str = "cpu",
    normalize: bool = True,
) -> Index:
    if not units:
        raise ValueError("cannot build an index over zero units")
    if len({u.id for u in units}) != len(units):
        from collections import Counter

        dupes = [i for i, n in Counter(u.id for u in units).items() if n > 1]
        raise ValueError(f"duplicate unit ids would break citation: {dupes[:5]}")
    encoder = encoder or default_encoder(embedding_model, device, normalize)

    with stage_timer("index.build", units=len(units)) as t:
        texts = [embeddable_text(u) for u in units]
        matrix = np.asarray(encoder(texts), dtype="float32")
        if matrix.ndim != 2 or len(matrix) != len(units):
            raise ValueError(f"encoder returned {matrix.shape}, expected ({len(units)}, dim)")
        t["dim"] = matrix.shape[1]

    return Index(
        units=units,
        vectors=matrix,
        bm25=BM25Okapi([tokenize(t) for t in texts]),
        embedding_model=embedding_model,
        normalize=normalize,
    )
