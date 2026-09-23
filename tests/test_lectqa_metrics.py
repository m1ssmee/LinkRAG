"""T1's answer metrics (CMC 2026, eqs. 29-36) against hand-computed cases."""

from __future__ import annotations

import math

import pytest

from linkrag.eval import lectqa_metrics as M


def test_tokens_and_set_based_token_prf_eqs_29_30():
    assert M.tokens("The Cat, the hat!") == ["the", "cat", "the", "hat"]
    # sets: pred {the,cat,sat}, ref {the,cat,sat,on,mat}: |∩| = 3
    p, r, f = M.token_prf("the cat sat", "the cat sat on the mat")
    assert (p, r) == (1.0, 0.6) and f == pytest.approx(0.75)
    assert M.token_prf("", "x") == (0.0, 0.0, 0.0) and M.token_prf("dog", "cat") == (0.0, 0.0, 0.0)


def test_rouge1_is_unigram_recall_eq_34():
    assert M.rouge1("the cat sat", "the cat sat on the mat") == pytest.approx(3 / 5)


def test_bleu4_eq_31_hand_computed():
    # pred "the cat sat on mat" vs ref "the cat sat on the mat":
    # p1 5/5, p2 3/4, p3 2/3, p4 1/2; geometric mean 0.25^(1/4); BP exp(1 - 6/5)
    expected = math.exp(1 - 6 / 5) * (1 * 0.75 * (2 / 3) * 0.5) ** 0.25
    assert M.bleu4("the cat sat on mat", "the cat sat on the mat") == pytest.approx(expected, abs=1e-6)
    assert expected == pytest.approx(0.5789, abs=1e-4)
    assert M.bleu4("cat mat", "the cat sat on the mat") == 0.0          # no 4-gram, no smoothing -> 0
    assert M.bleu4("the cat sat on the mat", "the cat sat on the mat") == pytest.approx(1.0)


def test_meteor_eqs_32_33_hand_computed():
    # identical, 6 matches in 1 chunk: F = 1, penalty 0.5 * (1/6)^3
    assert M.meteor("the cat sat on the mat", "the cat sat on the mat") == pytest.approx(1 - 0.5 / 216, abs=1e-6)
    # pred "cat sat quietly" vs "the cat sat on a mat": m = 2 in 1 chunk, P = 2/3, R = 1/3
    p, r = 2 / 3, 1 / 3
    f = 10 * p * r / (r + 9 * p)
    assert M.meteor("cat sat quietly", "the cat sat on a mat") == pytest.approx(f * (1 - 0.5 * (1 / 2) ** 3), abs=1e-6)
    # pred "mat cat": m = 2 in 2 chunks (order reversed), P = 1, R = 1/3, penalty 0.5
    f = 10 * 1 * (1 / 3) / (1 / 3 + 9)
    assert M.meteor("mat cat", "the cat sat on a mat") == pytest.approx(f * (1 - 0.5), abs=1e-6)


def test_meteor_nltk_greedy_alignment_on_repeated_words_is_a_documented_deviation():
    # "the" twice in the reference: the minimal alignment has 1 chunk (0.5166),
    # nltk's greedy alignment takes 2 (0.4483). Pinned so a library change shows up.
    f = 5 / 9.5
    assert M.meteor("the cat sat", "the cat sat on the mat") == pytest.approx(f * (1 - 0.5 * (2 / 3) ** 3), abs=1e-6)


def test_mcq_accuracy_and_macro_prf_hand_computed():
    s = M.score_mcq(["A", "B", "B", None], ["A", "B", "C", "D"])
    # per label (P, R): A (1, 1), B (0.5, 1), C (0, 0), D (0, 0); F: 1, 2/3, 0, 0
    assert s["accuracy"] == 0.5
    assert s["precision"] == pytest.approx(0.375) and s["recall"] == pytest.approx(0.5)
    assert s["f1"] == pytest.approx((1 + 2 / 3) / 4)


def test_similarity_eq_35_uses_minilm():
    try:
        sims = M.similarity(["the cat sat on the mat", "quantum chromodynamics"],
                            ["the cat sat on the mat", "a cat on a mat"])
    except Exception as exc:                                  # model not downloadable offline
        pytest.skip(f"all-MiniLM-L6-v2 unavailable: {exc}")
    assert sims[0] == pytest.approx(1.0, abs=1e-5) and sims[1] < 0.5
