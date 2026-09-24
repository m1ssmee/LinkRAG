"""Visual channel: frame->page scores, the segment x page lookup, and align's visual options."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image, ImageDraw

from linkrag.core import EvidenceUnit, Location
from linkrag.ingest.video_slides import Slide
from linkrag.link.align import align
from linkrag.link.visual import confident_rate, dhash_similarity, segment_visual


def _img(kind: int) -> Image.Image:
    im = Image.new("RGB", (160, 120), "white")
    d = ImageDraw.Draw(im)
    if kind == 0:
        d.rectangle([10, 10, 80, 110], fill="black")
    else:
        d.ellipse([60, 20, 150, 100], fill="black")
    return im


def test_dhash_similarity_identity_and_difference():
    S = dhash_similarity([_img(0), _img(1)], [_img(0), _img(1)])
    assert S[0, 0] == 1.0 and S[1, 1] == 1.0
    assert S[0, 1] < 1.0 and S[1, 0] < 1.0


def test_segment_visual_takes_the_row_of_the_frame_on_screen():
    slides = [Slide(index=0, intervals=[(0.0, 10.0), (30.0, 40.0)]), Slide(index=1, intervals=[(10.0, 30.0)])]
    frame_page = np.array([[0.9, 0.1, 0.0], [0.2, 0.8, 0.3]])
    M = segment_visual([5.0, 15.0, 35.0, 50.0], slides, frame_page)
    assert M.tolist() == [[0.9, 0.1, 0.0], [0.2, 0.8, 0.3], [0.9, 0.1, 0.0], [0.0, 0.0, 0.0]]   # revisit, then gap


def test_confident_rate_counts_margin_over_runner_up():
    assert confident_rate(np.array([[0.9, 0.5], [0.6, 0.55]]), margin=0.1) == 0.5


def test_align_visual_options():
    audio = [EvidenceUnit(id=f"a{i}", modality="audio", content="x", source_file="a", location=Location(start_s=i, end_s=i + 1))
             for i in range(3)]
    slides = [EvidenceUnit(id=f"p{j}", modality="text", content="y", source_file="d", location=Location(page=j + 1))
              for j in range(3)]
    V = np.eye(3)
    got = align(audio, slides, encoder=None, similarity="visual", visual=V, jump_penalty=0.0, skip_penalty=0.0)
    assert got.path == [0, 1, 2] and np.array_equal(got.similarity, V)
    with pytest.raises(ValueError, match="visual"):
        align(audio, slides, encoder=None, similarity="visual+text")
