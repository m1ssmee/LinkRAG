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
from typing import Any, Callable, Sequence

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
    a_vec: np.ndarray | None = None,
    s_vec: np.ndarray | None = None,
) -> np.ndarray:
    """S[i][j] for every (audio segment, slide) pair. See module docstring.

    `a_vec` / `s_vec`: precomputed embeddings (rows aligned with the units), so a
    many-pairs study can embed each side once instead of once per pair."""
    if not audio_units or not slide_units:
        raise ValueError("need at least one audio unit and one slide unit")

    audio_text = [u.content for u in audio_units]
    slide_text = [u.content for u in slide_units]

    a_vec = np.asarray(encoder(audio_text) if a_vec is None else a_vec, dtype="float32")
    s_vec = np.asarray(encoder(slide_text) if s_vec is None else s_vec, dtype="float32")
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


def distiluse_similarity(
    audio_units: Sequence[EvidenceUnit], slide_units: Sequence[EvidenceUnit], *, device: str = "cpu",
    model_name: str = "sentence-transformers/distiluse-base-multilingual-cased",
) -> np.ndarray:
    """MaViLS's audio-only similarity: distiluse cosine between each segment and the
    slide text (their slide text is page OCR; the caller decides what the slide units
    hold). Kept as a separate matrix so it can be fused with ours -- the MaViLS
    decomposition (reports/mavils_final.md) showed the gap to their number is in the
    similarity features, not the decoder."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name, device=device)
    a = model.encode([u.content for u in audio_units], convert_to_numpy=True, normalize_embeddings=True)
    b = model.encode([u.content for u in slide_units], convert_to_numpy=True, normalize_embeddings=True)
    return (a @ b.T).astype(np.float64)


def _minmax(S: np.ndarray) -> np.ndarray:
    lo, hi = float(S.min()), float(S.max())
    return (S - lo) / (hi - lo) if hi > lo else np.zeros_like(S)


def fuse_similarity(ours: np.ndarray, theirs: np.ndarray, method: str = "fused_weighted",
                    weight: float = 0.5) -> np.ndarray:
    """Combine two similarity matrices of the same shape. Each is min-max scaled over
    the whole matrix first (a hybrid in ~[0.1, 0.9] and a cosine in ~[0, 0.6] are not
    on one scale). `fused_max` = cellwise max; `fused_weighted` = weight*theirs +
    (1-weight)*ours. `weight` is a tuned quantity: set it on a tune half only."""
    if ours.shape != theirs.shape:
        raise ValueError(f"shape mismatch {ours.shape} vs {theirs.shape}")
    a, b = _minmax(ours), _minmax(theirs)
    if method == "fused_max":
        return np.maximum(a, b)
    if method == "fused_weighted":
        return float(weight) * b + (1.0 - float(weight)) * a
    raise ValueError(f"unknown fusion {method!r}: use 'fused_max' or 'fused_weighted'")


def fuse_many(matrices: Sequence[np.ndarray], weights: Sequence[float]) -> np.ndarray:
    """Weighted sum of several same-shape similarity matrices, each min-max scaled over the
    whole matrix first (as `fuse_similarity`). The weights are tuned quantities."""
    if len(matrices) != len(weights) or len({m.shape for m in matrices}) != 1:
        raise ValueError("fuse_many needs one weight per matrix and matrices of one shape")
    return sum(float(w) * _minmax(m) for m, w in zip(matrices, weights))


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
    flatness_scaling: float = 0.0,
) -> list[int]:
    """Viterbi-style DP over the recurrence in the module docstring. O(n*m).

    `start_prior_mu` adds -mu*j to the initialisation, pulling the first segment
    toward the front of the deck. sigma already charges for skipped head slides, but
    at sigma=0.02 that pull is weak: on pilot01 the opening segment (title slide,
    certain) was assigned p3. mu defaults to 0.0, which is exactly the behaviour every
    number recorded before this parameter existed.

    `flatness_scaling` (MaViLS study, 2026-09-22) makes the skip penalty row-aware:
    sigma_i = sigma * (1 - f + f * c_i / mean(c)), with c_i = max_i - mean_i the row's
    contrast. f = 0 is the constant sigma every earlier number used; f = 1 scales it
    fully, so a flat row (no evidence for any slide) charges almost nothing for
    moving on and a peaked row keeps the full penalty. The head/tail charges use the
    first/last row's sigma.
    """
    n, m = similarity.shape
    if n == 0 or m == 0:
        return []
    lam, sig0, beta = float(jump_penalty), float(skip_penalty), float(back_penalty)
    max_back = max(0, int(max_back))
    f = float(flatness_scaling)
    if f > 0:
        contrast = similarity.max(axis=1) - similarity.mean(axis=1)
        scale = contrast / max(float(contrast.mean()), 1e-9)
        sig_rows = sig0 * (1.0 - f + f * scale)
    else:
        sig_rows = np.full(n, sig0)

    # D[j] = best score for the current segment ending on slide j.
    sig = float(sig_rows[0])
    prev = similarity[0] - (sig + float(start_prior_mu)) * np.arange(m)
    backptr = np.full((n, m), -1, dtype=np.int32)

    for i in range(1, n):
        sig = float(sig_rows[i])
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
    final = prev - float(sig_rows[-1]) * (m - 1 - np.arange(m))
    j = int(final.argmax())
    path = [j]
    for i in range(n - 1, 0, -1):
        j = int(backptr[i, j])
        path.append(j)
    path.reverse()
    return path


# ----------------------------------------------------------------- relatedness gate

def path_score(similarity: np.ndarray, path: Sequence[int]) -> float:
    """DP path score normalised by segment count: mean S[i, path[i]]."""
    if not len(path):
        return 0.0
    return float(np.mean([similarity[i, j] for i, j in enumerate(path)]))


def path_objective(similarity: np.ndarray, path: Sequence[int], *, jump_penalty: float,
                   skip_penalty: float, back_penalty: float) -> float:
    """What the DP maximises, per segment: path similarity minus the transition
    penalties it paid. Unlike the bare path score this drops when a path has to
    jump around to follow the evidence -- which is exactly what happens on a
    shuffled slide order -- so it is the statistic the relatedness gate uses."""
    if not len(path):
        return 0.0
    sim = sum(float(similarity[i, j]) for i, j in enumerate(path))
    pen = 0.0
    for x, y in zip(path, path[1:]):
        if y > x + 1:
            pen += jump_penalty * (y - x) + skip_penalty * (y - x - 1)
        elif y < x:
            pen += back_penalty * (x - y)
    return (sim - pen) / len(path)


def relatedness_gate(similarity: np.ndarray, decode: Callable[[np.ndarray], list[int]], *,
                     jump_penalty: float = 0.05, skip_penalty: float = 0.02, back_penalty: float = 0.15,
                     shuffles: int = 5, z: float = 2.0, seed: int = 20260923,
                     null_std_floor: float = 0.0) -> dict[str, Any]:
    """Is this audio track about this deck at all?

    Null model: shuffle the slide order (columns) `shuffles` times and decode each
    shuffled matrix with the same DP. Statistic: the penalised DP objective per
    segment (`path_objective`). A related pair has a monotone path that follows the
    evidence cheaply; on a shuffled order the path must either pay jump penalties or
    give up similarity. The pair passes when the true objective exceeds the shuffled
    mean by `z` shuffled standard deviations. With 5 shuffles the std is a rough
    estimate -- z = 2.0 is deliberately conservative; its false-rejection rate on
    related pairs is measured on MaViLS (reports/relatedness_gate.md). The bare path
    score was tried first and rejected 11/20 related MaViLS pairs: with weak
    penalties the DP finds near-argmax paths on any column order."""
    rng = np.random.default_rng(seed)
    obj = lambda S: path_objective(S, decode(S), jump_penalty=jump_penalty, skip_penalty=skip_penalty,
                                   back_penalty=back_penalty)
    true = obj(similarity)
    nulls = [obj(similarity[:, rng.permutation(similarity.shape[1])]) for _ in range(shuffles)]
    mean, std = float(np.mean(nulls)), float(np.std(nulls))
    # Five shuffles give a rough std; a near-zero estimate would turn a tiny margin
    # into a huge z. `null_std_floor` (config align.null_std_floor) bounds it below.
    std_used = max(std, float(null_std_floor))
    zscore = (true - mean) / std_used if std_used > 1e-12 else (float("inf") if true > mean else 0.0)
    return {"score": true, "null_mean": mean, "null_std": std, "null_std_used": std_used,
            "null_scores": nulls, "z": zscore, "threshold_z": z, "related": bool(zscore >= z)}


def abstain(similarity: np.ndarray, path: Sequence[int], min_segment_sim: float | None) -> list[int]:
    """Per-segment abstention: a segment whose best similarity to ANY slide is below
    `min_segment_sim` gets -1 (no slide) instead of the path's slide. None = off.
    The DP path itself is unchanged -- abstention is applied after decoding, so the
    sequence structure still benefits neighbouring segments."""
    if min_segment_sim is None:
        return list(path)
    row_max = similarity.max(axis=1)
    return [-1 if row_max[i] < min_segment_sim else int(j) for i, j in enumerate(path)]


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
    flatness_scaling: float = 0.0,
    similarity: str = "ours",
    fusion_weight: float = 0.5,
    device: str = "cpu",
    visual: np.ndarray | None = None,
    frame_ocr: np.ndarray | None = None,
    fusion_weights: Sequence[float] | None = None,
) -> Alignment:
    """`similarity`: ours (bge-m3 + BM25 + IDF hybrid, the default), theirs (MaViLS's
    distiluse cosine), fused_max, fused_weighted -- see `fuse_similarity`; visual (the
    segment x page matrix passed as `visual`, `linkrag.link.visual`), visual+text and
    visual+theirs (fusion_weight*visual + (1-fusion_weight)*text, each min-max scaled);
    frame_ocr (segment x page matrix of the text on the frame, passed as `frame_ocr`),
    visual+frame_ocr (fusion_weight*frame_ocr + (1-fusion_weight)*visual) and
    visual+frame_ocr+theirs (`fuse_many` with `fusion_weights` for visual, frame_ocr, theirs)."""
    if method not in ("monotonic", "naive"):
        raise ValueError(f"unknown alignment method {method!r}: use 'monotonic' or 'naive'")
    if similarity not in ("ours", "theirs", "fused_max", "fused_weighted", "visual", "visual+text", "visual+theirs",
                          "frame_ocr", "visual+frame_ocr", "visual+frame_ocr+theirs"):
        raise ValueError(f"unknown similarity {similarity!r}")
    if similarity.startswith("visual") and visual is None:
        raise ValueError(f"similarity {similarity!r} needs the segment x page `visual` matrix")
    if "frame_ocr" in similarity and frame_ocr is None:
        raise ValueError(f"similarity {similarity!r} needs the segment x page `frame_ocr` matrix")
    if similarity == "visual+frame_ocr+theirs" and (fusion_weights is None or len(fusion_weights) != 3):
        raise ValueError("visual+frame_ocr+theirs needs three fusion_weights (visual, frame_ocr, theirs)")
    weights = weights or {}
    with stage_timer("link.align", method=method, similarity=similarity,
                     n=len(audio_units), m=len(slide_units)) as t:
        ours = None if similarity in ("theirs", "visual", "visual+theirs", "frame_ocr", "visual+frame_ocr",
                                      "visual+frame_ocr+theirs") else similarity_matrix(
            audio_units, slide_units, encoder=encoder,
            w_dense=weights.get("dense", 0.6),
            w_bm25=weights.get("bm25", 0.25),
            w_keyword=weights.get("keyword", 0.15),
        )
        if similarity == "ours":
            sim = ours
        elif similarity == "visual":
            sim = visual
        elif similarity == "visual+text":
            sim = fuse_similarity(ours, visual, "fused_weighted", fusion_weight)
        elif similarity == "frame_ocr":
            sim = frame_ocr
        elif similarity == "visual+frame_ocr":
            sim = fuse_similarity(visual, frame_ocr, "fused_weighted", fusion_weight)
        else:
            theirs = distiluse_similarity(audio_units, slide_units, device=device)
            sim = (theirs if similarity == "theirs" else
                   fuse_similarity(theirs, visual, "fused_weighted", fusion_weight) if similarity == "visual+theirs" else
                   fuse_many([visual, frame_ocr, theirs], fusion_weights) if similarity == "visual+frame_ocr+theirs" else
                   fuse_similarity(ours, theirs, similarity, fusion_weight))
        similarity = sim  # the matrix from here on
        path = (align_naive(similarity) if method == "naive" else
                align_monotonic(similarity, jump_penalty=jump_penalty,
                                skip_penalty=skip_penalty, back_penalty=back_penalty,
                                max_back=max_back, start_prior_mu=start_prior_mu,
                                flatness_scaling=flatness_scaling))
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
    min_segment_sim: float | None = None,
    relatedness_z: float | None = None,
    relatedness_shuffles: int = 5,
    decode: Callable[[np.ndarray], list[int]] | None = None,
    penalties: dict[str, float] | None = None,
) -> list[Link]:
    """One audio_slide Link per aligned segment, above `min_score`.

    Two gates compose, coarse to fine. `relatedness_z` (with `decode`, the same DP
    the alignment used): the file-pair gate -- if the track is not about the deck at
    all, emit NOTHING and record why in `build_links.gate`. `min_segment_sim`: the
    per-segment abstention -- a segment with no recognisable slide gets no link. The
    gate runs first, so abstention never has to rescue an unrelated pair, and a
    related pair still loses only its flat segments."""
    build_links.gate = None  # type: ignore[attr-defined]
    if relatedness_z is not None:
        if decode is None:
            raise ValueError("relatedness_z needs `decode` (the DP used for the alignment)")
        gate = relatedness_gate(alignment.similarity, decode, shuffles=relatedness_shuffles, z=relatedness_z,
                                **(penalties or {}))
        build_links.gate = gate  # type: ignore[attr-defined]
        if not gate["related"]:
            return []
    path = abstain(alignment.similarity, alignment.path, min_segment_sim)
    links = []
    for i, j in enumerate(path):
        if j < 0:
            continue  # abstained
        score = float(alignment.similarity[i, j])
        if score < min_score:
            continue
        links.append(Link(src_id=audio_units[i].id, dst_id=slide_units[j].id,
                          link_type="audio_slide", score=score))
    return links


def save_links(
    links: Sequence[Link], path: str | Path, manifest_hash: str | None = None,
    meta: dict[str, Any] | None = None,
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
        fh.write(json.dumps({"_meta": {"manifest_hash": manifest_hash, **(meta or {})}}) + "\n")
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
    try:
        links = [Link(**r) for r in records if "_meta" not in r]
    except ValueError as exc:                            # a retired link type: say which file
        raise ValueError(f"{path}: {exc}") from exc
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
    load_links.last_meta = meta        # type: ignore[attr-defined]
    return links
