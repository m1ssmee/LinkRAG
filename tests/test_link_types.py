"""Link.link_type: the closed set and the checker, at construction and at load."""

from __future__ import annotations

import json

import pytest

from linkrag.core import LINK_TYPES, check_link_type


def test_closed_set_and_retired_names():
    assert LINK_TYPES == {"audio_slide", "figure_text", "deictic", "same_slide"}
    assert check_link_type("deictic") == "deictic"
    for retired in ("deictic_visual", "figure_paragraph", "same_topic"):
        with pytest.raises(ValueError, match="unknown link_type"):
            check_link_type(retired)


def test_link_and_load_links_reject_a_retired_type(tmp_path):
    from linkrag.core import Link
    from linkrag.link.align import load_links
    with pytest.raises(ValueError, match="unknown link_type 'deictic_visual'"):
        Link("a", "b", "deictic_visual", 0.5)
    f = tmp_path / "links.jsonl"
    f.write_text(json.dumps({"src_id": "a", "dst_id": "b", "link_type": "figure_paragraph", "score": 1.0}) + "\n")
    with pytest.raises(ValueError, match=r"links\.jsonl: unknown link_type 'figure_paragraph'"):
        load_links(f)
