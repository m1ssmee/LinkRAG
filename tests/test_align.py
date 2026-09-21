"""Audio-to-slide alignment: the monotonic DP and its naive ablation."""

from __future__ import annotations

import numpy as np
import pytest

from linkrag.core import EvidenceUnit, Location
from linkrag.link.align import (
    Alignment,
    align,
    align_monotonic,
    align_naive,
    build_links,
    load_links,
    save_links,
    similarity_matrix,
)

NEG = -1e18


def brute_force(S, lam, sig, beta, B):
    """Reference O(n*m^2) DP, written straight from the recurrence in the module
    docstring. The shipped implementation replaces the inner maximum with a
    prefix max; this exists to prove that shortcut is exact, because an
    off-by-one there would silently return slightly wrong paths forever."""
    n, m = S.shape
    D = [[NEG] * m for _ in range(n)]
    bp = [[-1] * m for _ in range(n)]
    for j in range(m):
        D[0][j] = S[0][j] - sig * j
    for i in range(1, n):
        for j in range(m):
            best, arg = NEG, -1
            for jp in range(m):
                d = j - jp
                if d in (0, 1):
                    pen = 0.0
                elif d >= 2:
                    pen = lam * d + sig * (d - 1)
                elif -B <= d <= -1:
                    pen = beta * (-d)
                else:
                    continue
                if D[i - 1][jp] - pen > best:
                    best, arg = D[i - 1][jp] - pen, jp
            D[i][j], bp[i][j] = S[i][j] + best, arg
    final = [D[n - 1][j] - sig * (m - 1 - j) for j in range(m)]
    j = max(range(m), key=lambda k: final[k])
    path = [j]
    for i in range(n - 1, 0, -1):
        j = bp[i][j]
        path.append(j)
    return list(reversed(path)), max(final)


# --------------------------------------------------------------- DP correctness

@pytest.mark.parametrize("seed", range(8))
def test_dp_matches_brute_force_reference(seed: int) -> None:
    rng = np.random.default_rng(seed)
    S = rng.random((9, 7))
    kw = dict(jump_penalty=0.05, skip_penalty=0.02, back_penalty=0.15, max_back=2)
    got = align_monotonic(S, **kw)
    want, _ = brute_force(S, kw["jump_penalty"], kw["skip_penalty"],
                          kw["back_penalty"], kw["max_back"])
    assert [S[i, j] for i, j in enumerate(got)] == pytest.approx(
        [S[i, j] for i, j in enumerate(want)]
    ), "prefix-max shortcut must be exact"


def test_dp_respects_the_back_jump_limit() -> None:
    """j(i) >= j(i-1) - B is the relaxation; anything further back is forbidden."""
    rng = np.random.default_rng(3)
    S = rng.random((25, 12))
    for max_back in (0, 1, 2):
        path = align_monotonic(S, max_back=max_back, back_penalty=0.0)
        drops = [a - b for a, b in zip(path, path[1:]) if b < a]
        assert all(d <= max_back for d in drops), f"max_back={max_back} violated: {drops}"


def test_max_back_zero_is_strictly_monotonic() -> None:
    rng = np.random.default_rng(5)
    path = align_monotonic(rng.random((30, 10)), max_back=0)
    assert path == sorted(path)


def test_jump_penalty_suppresses_large_jumps() -> None:
    """lambda charges for advancing more than one slide at a time."""
    rng = np.random.default_rng(7)
    S = rng.random((12, 10))
    big = lambda p: sum(1 for a, b in zip(p, p[1:]) if b - a > 1)
    loose = align_monotonic(S, jump_penalty=0.0, skip_penalty=0.0)
    tight = align_monotonic(S, jump_penalty=5.0, skip_penalty=0.0)
    assert big(tight) < big(loose)


def test_skip_penalty_pushes_toward_covering_the_deck() -> None:
    """sigma charges per *unassigned* slide, so raising it must not reduce
    coverage. (An earlier version of this test asserted the opposite and was
    simply wrong: a high skip penalty forces the path to span the whole deck,
    including the head and tail terms.)"""
    rng = np.random.default_rng(7)
    S = rng.random((12, 10))
    loose = align_monotonic(S, jump_penalty=0.0, skip_penalty=0.0)
    tight = align_monotonic(S, jump_penalty=0.0, skip_penalty=5.0)
    assert len(set(tight)) >= len(set(loose))


def test_empty_matrix() -> None:
    assert align_monotonic(np.zeros((0, 0))) == []


def test_single_slide_forces_every_segment_onto_it() -> None:
    assert align_monotonic(np.random.default_rng(0).random((5, 1))) == [0] * 5


# ------------------------------------------ the case that motivates the method

def _ordered_deck_with_one_back_jump():
    """8 slides; 10 segments walking forward with one deliberate back-jump at
    index 6 (slide 4 -> 3, the lecturer returning to an earlier slide).

    Two segments carry a *distractor*: a slightly higher score on a far-away
    slide, as happens when a segment reuses vocabulary from the conclusion.
    Local argmax takes the bait; the sequence prior should not.
    """
    n, m = 10, 8
    true_path = [0, 1, 1, 2, 3, 4, 3, 5, 6, 7]
    S = np.full((n, m), 0.10)
    for i, j in enumerate(true_path):
        S[i, j] = 0.70
    S[2, 7] = 0.76   # distractor: segment 2 looks like the final slide
    S[6, 7] = 0.74   # distractor on the back-jump segment
    return S, true_path


def test_monotonic_dp_recovers_the_order_and_naive_does_not() -> None:
    S, true_path = _ordered_deck_with_one_back_jump()

    dp = align_monotonic(S, jump_penalty=0.05, skip_penalty=0.02,
                         back_penalty=0.15, max_back=2)
    naive = align_naive(S)

    assert dp == true_path, f"DP should recover the walk, got {dp}"
    assert naive != true_path, "naive must fail here or the test proves nothing"
    assert naive[2] == 7 and naive[6] == 7, "naive takes the distractor bait"
    # and the failure is a sequence failure, not just two bad cells
    assert naive[2] > naive[3], "naive produces a non-monotonic jumble"


def test_monotonic_dp_recovers_the_single_back_jump() -> None:
    S, true_path = _ordered_deck_with_one_back_jump()
    dp = align_monotonic(S, max_back=2, back_penalty=0.15)
    assert dp[5] == 4 and dp[6] == 3, "the real back-jump must survive"

    # Forbidding back-jumps must lose it -- the constraint is load-bearing.
    strict = align_monotonic(S, max_back=0)
    assert strict[6] >= strict[5], "max_back=0 cannot go backwards"
    assert strict != true_path


def test_back_penalty_trades_off_against_evidence() -> None:
    S, _ = _ordered_deck_with_one_back_jump()
    assert align_monotonic(S, max_back=2, back_penalty=99.0).count(3) < 2, (
        "a prohibitive back penalty should abandon the back-jump"
    )


# ------------------------------------------------------- similarity + plumbing

def _units(texts, modality, source, page_offset=1):
    out = []
    for i, t in enumerate(texts):
        loc = (Location(page=i + page_offset) if modality == "text"
               else Location(start_s=float(i * 30), end_s=float((i + 1) * 30)))
        out.append(EvidenceUnit(id=f"{source}:{modality}{i}", modality=modality,
                                content=t, source_file=source, location=loc))
    return out


def test_similarity_matrix_shape_and_range(stub_encoder) -> None:
    slides = ["attention weights heatmap", "optimizer cosine schedule", "dataset statistics"]
    audio = ["so the attention weights here", "we use a cosine schedule", "about our dataset"]
    encode = stub_encoder(slides + audio)
    S = similarity_matrix(_units(audio, "audio", "a.mp3"), _units(slides, "text", "s.pdf"),
                          encoder=encode)
    assert S.shape == (3, 3)
    assert S.min() >= -1e-6 and S.max() <= 1.0 + 1e-6
    assert [int(j) for j in S.argmax(axis=1)] == [0, 1, 2], "diagonal should win"


def test_similarity_weights_are_honoured(stub_encoder) -> None:
    slides, audio = ["alpha beta"], ["alpha beta"]
    encode = stub_encoder(slides + audio)
    args = (_units(audio, "audio", "a.mp3"), _units(slides, "text", "s.pdf"))
    only_dense = similarity_matrix(*args, encoder=encode, w_dense=1.0, w_bm25=0.0, w_keyword=0.0)
    nothing = similarity_matrix(*args, encoder=encode, w_dense=0.0, w_bm25=0.0, w_keyword=0.0)
    assert only_dense[0, 0] > 0.9
    assert nothing[0, 0] == pytest.approx(0.0)


def test_similarity_rejects_empty_input(stub_encoder) -> None:
    with pytest.raises(ValueError, match="at least one"):
        similarity_matrix([], _units(["x"], "text", "s.pdf"), encoder=stub_encoder(["x"]))


def test_align_rejects_an_unknown_method(stub_encoder) -> None:
    with pytest.raises(ValueError, match="unknown alignment method"):
        align(_units(["a"], "audio", "a.mp3"), _units(["a"], "text", "s.pdf"),
              encoder=stub_encoder(["a"]), method="magic")


def test_build_links_and_round_trip(tmp_path, stub_encoder) -> None:
    slides = ["attention heatmap", "cosine schedule"]
    audio = ["the attention heatmap", "a cosine schedule"]
    encode = stub_encoder(slides + audio)
    a_units, s_units = _units(audio, "audio", "a.mp3"), _units(slides, "text", "s.pdf")
    result = align(a_units, s_units, encoder=encode, method="monotonic")

    links = build_links(a_units, s_units, result)
    assert len(links) == 2
    assert all(l.link_type == "audio_slide" for l in links)
    assert links[0].src_id == "a.mp3:audio0" and links[0].dst_id == "s.pdf:text0"

    path = save_links(links, tmp_path / "links.jsonl")
    reloaded = load_links(path)
    assert [(l.src_id, l.dst_id, l.link_type) for l in reloaded] == \
           [(l.src_id, l.dst_id, l.link_type) for l in links]
    # scores are rounded to 6dp on write to keep the file diffable
    assert [l.score for l in reloaded] == pytest.approx([l.score for l in links], abs=1e-6)


def test_build_links_drops_low_scores(stub_encoder) -> None:
    a_units, s_units = _units(["zzz"], "audio", "a.mp3"), _units(["qqq"], "text", "s.pdf")
    result = align(a_units, s_units, encoder=stub_encoder(["zzz", "qqq"]), method="monotonic")
    assert build_links(a_units, s_units, result, min_score=0.99) == []


def test_alignment_reports_its_own_diagnostics() -> None:
    S, true_path = _ordered_deck_with_one_back_jump()
    a = Alignment(path=true_path, similarity=S, method="monotonic", total_score=0.0)
    assert a.n == 10 and a.m == 8
    assert a.slides_used() == 8
    assert a.back_jumps() == 1


# ------------------------------------------------- ear-label conversion rules

def test_start_prior_pulls_the_first_segment_toward_the_front() -> None:
    """mu fixes the head-of-sequence error: on pilot01 the opening segment (the
    title slide, certain) was assigned p3 at mu=0."""
    rng = np.random.default_rng(4)
    S = rng.random((12, 10))
    S[0, 6] = 0.95   # a late slide looks best locally for segment 1
    S[0, 0] = 0.90
    assert align_monotonic(S, start_prior_mu=0.0)[0] == 6
    assert align_monotonic(S, start_prior_mu=0.5)[0] == 0


def test_start_prior_defaults_to_zero_reproducing_earlier_numbers() -> None:
    rng = np.random.default_rng(9)
    S = rng.random((15, 8))
    assert align_monotonic(S) == align_monotonic(S, start_prior_mu=0.0)


# ------------------------------------------------ flatness scaling / abstention (2026-09-22)

def test_flatness_zero_reproduces_constant_sigma():
    rng = np.random.default_rng(3)
    S = rng.random((12, 6))
    from linkrag.link.align import align_monotonic
    assert align_monotonic(S, flatness_scaling=0.0) == align_monotonic(S)


def test_flat_rows_get_a_smaller_skip_penalty():
    """Peaked row 0 (slide 0), two flat rows, peaked row 3 (slide 4). Every monotone
    path skips slides 1-3 somewhere; with f=1 a flat row's sigma is ~0, so the skip is
    taken there, whereas constant sigma spreads the walk one slide per row."""
    from linkrag.link.align import align_monotonic
    S = np.full((4, 5), 0.1)
    S[0, 0] = 0.9
    S[3, 4] = 0.9
    assert align_monotonic(S, jump_penalty=0.0, skip_penalty=0.1, max_back=0) == [0, 1, 3, 4]
    assert align_monotonic(S, jump_penalty=0.0, skip_penalty=0.1, max_back=0,
                           flatness_scaling=1.0) == [0, 4, 4, 4]


def test_abstain_marks_low_max_rows_and_leaves_the_rest():
    from linkrag.link.align import abstain
    S = np.array([[0.9, 0.1], [0.2, 0.25], [0.1, 0.8]])
    assert abstain(S, [0, 1, 1], 0.5) == [0, -1, 1]
    assert abstain(S, [0, 1, 1], None) == [0, 1, 1]
