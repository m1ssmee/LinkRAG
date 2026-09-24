"""Link.link_type: the closed set, the checker, and the stored-file survey."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from linkrag.core import LINK_TYPES, check_link_type

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import check_link_types  # noqa: E402


def test_closed_set_and_retired_names():
    assert LINK_TYPES == {"audio_slide", "figure_text", "deictic", "same_slide"}
    assert check_link_type("deictic") == "deictic"
    for retired in ("deictic_visual", "figure_paragraph", "same_topic"):
        with pytest.raises(ValueError, match="unknown link_type"):
            check_link_type(retired)


def test_survey_flags_a_file_with_a_retired_type(tmp_path, capsys):
    rows = [{"_meta": {"manifest_hash": "x"}}, {"src_id": "a", "dst_id": "b", "link_type": "audio_slide", "score": 1.0}]
    (tmp_path / "links.jsonl").write_text("\n".join(map(json.dumps, rows)) + "\n")
    assert check_link_types.main([str(tmp_path)]) == 0
    rows.append({"src_id": "a", "dst_id": "c", "link_type": "deictic_visual", "score": 0.5})
    (tmp_path / "old_links.jsonl").write_text("\n".join(map(json.dumps, rows)) + "\n")
    assert check_link_types.main([str(tmp_path)]) == 1
    assert "OUTSIDE THE SET: {'deictic_visual': 1}" in capsys.readouterr().out
