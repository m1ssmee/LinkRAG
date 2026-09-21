"""MaViLS adapter pieces that have logic: their scorer, build-group detection,
within-group assignment, and their DP loaded from their source."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("mavils_adapter", "scripts/adapters/mavils.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_their_f1_counts_abstention_as_wrong():
    gt = np.array([-1, 1, 1, 2, 2])
    assert m.their_prf(gt, np.array([5, 1, 1, 2, 2]))[2] == 1.0
    f_wrong_in_set = m.their_prf(gt, np.array([5, 1, 2, 2, 2]))[2]     # wrong slide that IS a GT label
    f_wrong_out = m.their_prf(gt, np.array([5, 1, 3, 2, 2]))[2]        # wrong slide never labelled
    f_abst = m.their_prf(gt, np.array([5, 1, -1, 2, 2]))[2]
    # -1 is in labels (unique of the unfiltered column): an abstention is a false
    # positive like any in-set wrong slide; an out-of-set wrong slide only costs recall.
    assert f_abst == f_wrong_in_set == 0.75 and f_wrong_out > f_abst
    pr, cov = m.answered_metrics(gt, np.array([5, 1, -1, 2, 2]))
    assert (pr, cov) == (1.0, 0.75)


def test_build_groups_detects_supersets():
    pages = ["Preparing for class intro reading short", "Preparing for class intro reading short skimmable",
             "Preparing for class intro reading short skimmable catch up", "Materials for today Williams et al carbon",
             "Materials for today Williams et al carbon neutral pathways", "Totally different slide about cities here"]
    assert m.build_groups(pages) == [[0, 1, 2], [3, 4], [5]]
    assert m.build_groups(["a b", "a b c"]) == [[0], [1]]          # too few tokens to call it a build


def test_within_group_assignment_follows_incremental_content():
    pages = ["reading assignment short and skimmable", "reading assignment short and skimmable mackay chapter two",
             "reading assignment short and skimmable mackay chapter two problem set monday"]
    sents = ["the reading is short and skimmable", "look at mackay chapter two", "problem set is due monday"]
    out = m.assign_within_group([0, 1, 2], [0, 1, 2], pages, sents, idf={})
    assert out == [0, 1, 2]


@pytest.mark.skipif(not Path("data/raw/mavils/helpers/utils.py").exists(), reason="MaViLS repo not cloned")
def test_their_dp_loads_and_decodes_monotone_case():
    S = np.eye(4)
    pairs, _ = m.their_dp()(S, 0.1)
    assert [j for _, j in pairs] == [0, 1, 2, 3]
