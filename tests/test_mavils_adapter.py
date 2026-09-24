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


def test_sentence_frames_take_the_first_frame_at_each_timestamp(tmp_path, monkeypatch):
    import json

    import av
    import pandas as pd
    from PIL import Image
    video = tmp_path / "v.mp4"
    with av.open(str(video), "w") as out:                  # 10 frames at 1 fps, grey level 20*k
        st = out.add_stream("mpeg4", rate=1)
        st.width, st.height, st.pix_fmt = 32, 32, "yuv420p"
        for k in range(10):
            frame = av.VideoFrame.from_image(Image.new("RGB", (32, 32), (20 * k,) * 3))
            for pkt in st.encode(frame):
                out.mux(pkt)
        for pkt in st.encode():
            out.mux(pkt)
    monkeypatch.setattr(m, "CACHE", tmp_path / "cache")
    monkeypatch.setattr(m, "video_for", lambda stem: video)
    monkeypatch.setattr(m, "load_ground_truth", lambda p: pd.DataFrame({"time": [2.0, 0.5, 2.0, 30.0]}))
    out = m.sentence_frames_for("x")
    grey = [np.asarray(Image.open(out / f"{i:05d}.jpg").convert("L")).mean() for i in range(4)]
    assert [round(g / 20) for g in grey] == [2, 1, 2, 9]   # first frame at/after t; past the end -> last frame
    assert json.loads((out / "done.json").read_text())["past_end"] == 1
