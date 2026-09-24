"""Visual channel: frame->page scores."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from linkrag.link.visual import confident_rate, dhash_similarity


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


def test_confident_rate_counts_margin_over_runner_up():
    assert confident_rate(np.array([[0.9, 0.5], [0.6, 0.55]]), margin=0.1) == 0.5
