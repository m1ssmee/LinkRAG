r"""Audio-to-slide alignment -- first component of the Evidence Linking Layer.

Given audio EvidenceUnits :math:`a_1 \dots a_n` in time order and slide/page text
units :math:`s_1 \dots s_m` in page order (from a *separate* PDF, not a video
track), decide which slide each audio segment is discussing.

P2 fuses audio and visual streams by timestamp *inside one video*, so it cannot
address a slide deck that lives in another file. P3 has no audio at all. Neither
exploits the structural fact this module is built on: **a lecturer walks forward
through a deck**, so the correct assignment is (near-)monotonic in time, and a
segment's slide is constrained by its neighbours' slides -- not just by its own
similarity score.

Scoring
-------
For audio segment :math:`i` and slide :math:`j`,

.. math::
    S_{ij} = w_d \cdot \cos(e_i, e_j) + w_b \cdot \widehat{BM25}(a_i, s_j)
             + w_k \cdot K_{ij}

where :math:`e` are bge-m3 embeddings, :math:`\widehat{BM25}` is the BM25 score of
:math:`a_i`'s tokens against slide :math:`s_j` rescaled to :math:`[0,1]` across
that row, and :math:`K_{ij}` is an IDF-weighted rare-term overlap bonus

.. math::
    K_{ij} = \frac{\sum_{t \in a_i \cap s_j} \mathrm{idf}(t)}
                  {\sum_{t \in s_j} \mathrm{idf}(t)}

so that sharing "NoScope" counts for far more than sharing "the". Weights
:math:`w_d, w_b, w_k` are config (`link.align.weights`).

Monotonic alignment (the contribution)
--------------------------------------
Let :math:`D_{ij}` be the best total score of aligning segments :math:`1..i` with
segment :math:`i` on slide :math:`j`. With forward-jump penalty :math:`\lambda`,
per-slide skip penalty :math:`\sigma`, back-jump penalty :math:`\beta` and
back-jump limit :math:`B`:

.. math::
    D_{1j} &= S_{1j} - \sigma j \\
    D_{ij} &= S_{ij} + \max \begin{cases}
        D_{i-1,j} & \text{stay on the slide} \\
        D_{i-1,j-1} & \text{advance one slide} \\
        \max\limits_{j' \le j-2} \big[ D_{i-1,j'} - \lambda (j - j') - \sigma (j - j' - 1) \big]
            & \text{jump forward, skipping slides} \\
        \max\limits_{1 \le b \le B} \big[ D_{i-1,j+b} - \beta b \big]
            & \text{jump back up to } B \text{ slides}
    \end{cases} \\
    \text{answer} &= \max_j \big[ D_{nj} - \sigma (m - 1 - j) \big]

The two :math:`\sigma` terms outside the gap case charge for slides skipped before
the first assignment and after the last, so the penalty counts *every* unassigned
slide, not only those inside gaps.

**Complexity is O(nm), not O(nm²).** The forward-jump case is a maximum over all
:math:`j' \le j-2`, which looks quadratic, but the penalty is affine in
:math:`j-j'`, so

.. math::
    D_{i-1,j'} - (\lambda + \sigma)(j - j') + \sigma
      = \underbrace{\big[D_{i-1,j'} + (\lambda + \sigma) j'\big]}_{\text{independent of } j}
        - (\lambda + \sigma) j + \sigma

and the bracketed quantity admits a running prefix maximum computed in one
left-to-right sweep. The back-jump case is a maximum over :math:`B` terms with
:math:`B` a small constant (2). Both are O(1) amortised per cell.

Strict monotonicity is *relaxed*, not abandoned: :math:`j(i) \ge j(i-1) - B`.
Lecturers do flip back to a previous slide to answer a question, and forbidding it
outright forces the path to absorb that error somewhere else. :math:`\beta > \lambda`
keeps it rare. With :math:`B = 0` the recurrence is exactly monotonic.

Baseline
--------
`align_naive` takes :math:`\arg\max_j S_{ij}` independently per segment -- local
similarity only, no sequence structure. This is roughly P2's behaviour and is the
ablation the monotonic path is measured against, selected by `link.align.method`.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from rank_bm25 import BM25Okapi

from linkrag.core import EvidenceUnit, Link, stage_timer
from linkrag.index import Encoder, tokenize

NEG = -1e18


@dataclass
class Alignment:
    """Result of aligning one audio track against one deck."""

    path: list[int]              # slide index (0-based) per audio segment
    similarity: np.ndarray       # (n, m) score matrix
    method: str
    total_score: float

    @property
    def n(self) -> int:
        return self.similarity.shape[0]

    @property
    def m(self) -> int:
        return self.similarity.shape[1]

    def scores(self) -> list[float]:
        return [float(self.similarity[i, j]) for i, j in enumerate(self.path)]

    def slides_used(self) -> int:
        return len(set(self.path))

    def back_jumps(self) -> int:
        return sum(1 for a, b in zip(self.path, self.path[1:]) if b < a)


def _idf(docs_tokens: Sequence[Sequence[str]]) -> dict[str, float]:
    """Smoothed inverse document frequency over the slide deck."""
    n = len(docs_tokens)
    seen: dict[str, int] = {}
    for tokens in docs_tokens:
        for token in set(tokens):
            seen[token] = seen.get(token, 0) + 1
    return {t: math.log(1.0 + n / (1.0 + df)) for t, df in seen.items()}


def _rescale_rows(matrix: np.ndarray) -> np.ndarray:
    """Min-max each row into [0,1]. BM25 is unbounded and its scale varies wildly
    between queries, so a raw score cannot be summed with a cosine."""
    lo = matrix.min(axis=1, keepdims=True)
    hi = matrix.max(axis=1, keepdims=True)
    return (matrix - lo) / np.maximum(hi - lo, 1e-9)


def similarity_matrix(
    audio_units: Sequence[EvidenceUnit],
    slide_units: Sequence[EvidenceUnit],
    *,
    encoder: Encoder,
    w_dense: float = 0.6,
    w_bm25: float = 0.25,
    w_keyword: float = 0.15,
) -> np.ndarray:
    """S[i][j] for every (audio segment, slide) pair. See module docstring."""
    if not audio_units or not slide_units:
        raise ValueError("need at least one audio unit and one slide unit")

    audio_text = [u.content for u in audio_units]
    slide_text = [u.content for u in slide_units]

    a_vec = np.asarray(encoder(audio_text), dtype="float32")
    s_vec = np.asarray(encoder(slide_text), dtype="float32")
    # encoders here are configured to normalise; guard anyway so a custom one
    # cannot silently turn cosine into an unbounded dot product.
    a_vec /= np.maximum(np.linalg.norm(a_vec, axis=1, keepdims=True), 1e-9)
    s_vec /= np.maximum(np.linalg.norm(s_vec, axis=1, keepdims=True), 1e-9)
    dense = a_vec @ s_vec.T

    slide_tokens = [tokenize(t) for t in slide_text]
    audio_tokens = [tokenize(t) for t in audio_text]
    bm25_model = BM25Okapi(slide_tokens)
    bm25 = _rescale_rows(np.vstack([bm25_model.get_scores(q) for q in audio_tokens]))

    idf = _idf(slide_tokens)
    slide_sets = [set(t) for t in slide_tokens]
    slide_mass = [max(sum(idf.get(t, 0.0) for t in s), 1e-9) for s in slide_sets]
    keyword = np.zeros_like(dense)
    for i, q in enumerate(audio_tokens):
        qset = set(q)
        for j, sset in enumerate(slide_sets):
            shared = qset & sset
            if shared:
                keyword[i, j] = sum(idf.get(t, 0.0) for t in shared) / slide_mass[j]
    keyword = np.clip(keyword, 0.0, 1.0)

    return w_dense * dense + w_bm25 * bm25 + w_keyword * keyword


def align_naive(similarity: np.ndarray) -> list[int]:
    """argmax per segment, independently. No sequence structure -- the P2-style
    ablation. Nothing stops it assigning slide 20 then slide 3 then slide 20."""
    return [int(j) for j in similarity.argmax(axis=1)]


def align_monotonic(
    similarity: np.ndarray,
    *,
    jump_penalty: float = 0.05,
    skip_penalty: float = 0.02,
    back_penalty: float = 0.15,
    max_back: int = 2,
    start_prior_mu: float = 0.0,
) -> list[int]:
    """Viterbi-style DP over the recurrence in the module docstring. O(n*m).

    `start_prior_mu` adds -mu*j to the initialisation, pulling the first segment
    toward the front of the deck. sigma already charges for skipped head slides, but
    at sigma=0.02 that pull is weak: on pilot01 the opening segment (title slide,
    certain) was assigned p3. mu defaults to 0.0, which is exactly the behaviour every
    number recorded before this parameter existed.
    """
    n, m = similarity.shape
    if n == 0 or m == 0:
        return []
    lam, sig, beta = float(jump_penalty), float(skip_penalty), float(back_penalty)
    max_back = max(0, int(max_back))

    # D[j] = best score for the current segment ending on slide j.
    prev = similarity[0] - (sig + float(start_prior_mu)) * np.arange(m)
    backptr = np.full((n, m), -1, dtype=np.int32)

    for i in range(1, n):
        cur = np.full(m, NEG)
        choice = np.full(m, -1, dtype=np.int32)

        # Forward jumps (j' <= j-2): affine penalty lets one prefix max serve
        # every j, which is what keeps this O(m) per segment instead of O(m^2).
        shifted = prev + (lam + sig) * np.arange(m)
        best_prefix, best_prefix_j = NEG, -1

        for j in range(m):
            best, arg = NEG, -1

            if prev[j] > best:                      # stay
                best, arg = prev[j], j
            if j >= 1 and prev[j - 1] > best:       # advance one
                best, arg = prev[j - 1], j - 1
            if j >= 2:                              # forward jump over a gap
                if shifted[j - 2] > best_prefix:
                    best_prefix, best_prefix_j = shifted[j - 2], j - 2
                if best_prefix > NEG:
                    cand = best_prefix - (lam + sig) * j + sig
                    if cand > best:
                        best, arg = cand, best_prefix_j
            for b in range(1, max_back + 1):        # limited back-jump
                if j + b < m:
                    cand = prev[j + b] - beta * b
                    if cand > best:
                        best, arg = cand, j + b

            cur[j] = similarity[i, j] + best
            choice[j] = arg

        backptr[i] = choice
        prev = cur

    # Charge for slides never reached after the last assignment.
    final = prev - sig * (m - 1 - np.arange(m))
    j = int(final.argmax())
    path = [j]
    for i in range(n - 1, 0, -1):
        j = int(backptr[i, j])
        path.append(j)
    path.reverse()
    return path


def align(
    audio_units: Sequence[EvidenceUnit],
    slide_units: Sequence[EvidenceUnit],
    *,
    encoder: Encoder,
    method: str = "monotonic",
    weights: dict[str, float] | None = None,
    jump_penalty: float = 0.05,
    skip_penalty: float = 0.02,
    back_penalty: float = 0.15,
    max_back: int = 2,
    start_prior_mu: float = 0.0,
) -> Alignment:
    if method not in ("monotonic", "naive"):
        raise ValueError(f"unknown alignment method {method!r}: use 'monotonic' or 'naive'")
    weights = weights or {}
    with stage_timer("link.align", method=method,
                     n=len(audio_units), m=len(slide_units)) as t:
        similarity = similarity_matrix(
            audio_units, slide_units, encoder=encoder,
            w_dense=weights.get("dense", 0.6),
            w_bm25=weights.get("bm25", 0.25),
            w_keyword=weights.get("keyword", 0.15),
        )
        path = (align_naive(similarity) if method == "naive" else
                align_monotonic(similarity, jump_penalty=jump_penalty,
                                skip_penalty=skip_penalty, back_penalty=back_penalty,
                                max_back=max_back, start_prior_mu=start_prior_mu))
        total = float(sum(similarity[i, j] for i, j in enumerate(path)))
        t["slides_used"] = len(set(path))
        t["back_jumps"] = sum(1 for a, b in zip(path, path[1:]) if b < a)
    return Alignment(path=path, similarity=similarity, method=method, total_score=total)


def build_links(
    audio_units: Sequence[EvidenceUnit],
    slide_units: Sequence[EvidenceUnit],
    alignment: Alignment,
    *,
    min_score: float = 0.0,
) -> list[Link]:
    """One audio_slide Link per aligned segment, above `min_score`."""
    links = []
    for i, j in enumerate(alignment.path):
        score = float(alignment.similarity[i, j])
        if score < min_score:
            continue
        links.append(Link(src_id=audio_units[i].id, dst_id=slide_units[j].id,
                          link_type="audio_slide", score=score))
    return links


def save_links(
    links: Sequence[Link], path: str | Path, manifest_hash: str | None = None
) -> Path:
    """JSONL, one Link per line -- append-friendly and diffable, unlike a pickle.

    The first line is a `_meta` record carrying the corpus manifest hash. Link ids
    only mean anything against the index that produced them; a links file silently
    paired with a different corpus degrades link-following to plain top-k with no
    error, which is exactly the ablation-looks-identical failure the measurement
    rules exist to catch.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        fh.write(json.dumps({"_meta": {"manifest_hash": manifest_hash}}) + "\n")
        for link in links:
            record = {
                "src_id": link.src_id, "dst_id": link.dst_id,
                "link_type": link.link_type, "score": round(link.score, 6),
            }
            if link.metadata:
                record["metadata"] = link.metadata
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


class LinkManifestMismatch(RuntimeError):
    """Links were built against a different corpus than the one being queried."""


def load_links(path: str | Path, expect_manifest: str | None = None, *,
               gated: bool = True) -> list[Link]:
    """Links from JSONL. Raises if the file's corpus stamp disagrees with
    `expect_manifest` -- a hard error, not a warning: every link id would be
    meaningless and expansion would silently return nothing.

    `gated=True` drops links the relatedness gate failed (priority ii; the verdict
    lives in `metadata["relatedness"]`). Files that were never gated are returned
    whole either way, and `load_links.last_dropped` says how many were removed so a
    caller can print it."""
    records = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    meta = next((r["_meta"] for r in records if "_meta" in r), {})
    stamp = meta.get("manifest_hash")
    if expect_manifest is not None and stamp != expect_manifest:
        raise LinkManifestMismatch(
            f"{path} was built against corpus {stamp or 'UNSTAMPED'}, but the index "
            f"is corpus {expect_manifest}. Re-run scripts/build_links.py."
        )
    links = [Link(**r) for r in records if "_meta" not in r]
    dropped = 0
    if gated:
        kept = []
        for l in links:
            r = (l.metadata or {}).get("relatedness")
            if r and not r.get("passed", True):
                dropped += 1
            else:
                kept.append(l)
        links = kept
    load_links.last_dropped = dropped  # type: ignore[attr-defined]
    return links
