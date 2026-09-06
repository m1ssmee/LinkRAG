"""Converting the ear-labelled slide timeline into per-segment ground truth."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from make_alignment_labels import label_segment, load_timeline, parse_mmss  # noqa: E402

TIMELINE = Path("data/labels/pilot01/pilot01_slide_timeline_v2.csv")
LABELS = Path("data/labels/pilot01/pilot01_alignment_labels_v2.csv")


@pytest.fixture(scope="module")
def timeline():
    return load_timeline(TIMELINE)


def test_parse_mmss() -> None:
    assert parse_mmss("0:30") == 30
    assert parse_mmss("19:05") == 19 * 60 + 5
    assert parse_mmss("") is None and parse_mmss("  ") is None


def test_never_shown_slide_is_excluded(timeline) -> None:
    slides, _outro = timeline
    assert 27 not in {s["slide"] for s in slides}, "slide 27 was never shown"


def test_v2_has_no_zero_length_slides(timeline) -> None:
    """v1 inferred three zero-length slides (9, 11, 18). v2 resolves all of them:
    9 is a real 5 s, and 10/11 and 17/18 are build slides sharing a span."""
    slides, _ = timeline
    assert {s["slide"] for s in slides if s["zero_length"]} == set()


def test_build_slides_share_a_span_and_both_pages_are_accepted(timeline) -> None:
    from make_alignment_labels import build_classes

    slides, _ = timeline
    classes = build_classes(slides)
    assert classes[10] == {10, 11} and classes[17] == {17, 18}
    assert 4 not in classes, "a normal slide has no build partner"


def test_majority_overlap_wins(timeline) -> None:
    slides, outro = timeline
    # 0:45-1:37 is slide 3; a segment mostly inside it must be slide 3
    slide, section, _amb = label_segment(50.0, 73.0, slides, outro)
    assert (slide, section) == (3, "talk")


def test_build_pair_resolves_to_the_lower_page(timeline) -> None:
    """10 and 11 share 5:40-6:28; max() on tuples would pick 11 by accident."""
    slides, outro = timeline
    slide, section, _amb = label_segment(345.0, 380.0, slides, outro)
    assert slide == 10 and section == "talk"


def test_outro_majority_becomes_qa(timeline) -> None:
    slides, outro = timeline
    slide, section, _ = label_segment(1200.0, 1230.0, slides, outro)  # deep in the Q&A
    assert slide is None and section == "qa"


def test_segment_only_slightly_into_the_outro_stays_talk(timeline) -> None:
    slides, outro = timeline
    start = outro[0] - 25.0
    slide, section, _ = label_segment(start, start + 30.0, slides, outro)
    assert section == "talk" and slide is not None


def test_boundary_proximity_marks_ambiguous(timeline) -> None:
    slides, outro = timeline
    _s, _sec, near = label_segment(30.5, 44.0, slides, outro)   # starts 0.5s after 0:30
    _s2, _sec2, far = label_segment(52.0, 72.0, slides, outro)  # well inside slide 3
    assert near is True and far is False


# ------------------------------------------------- the shipped label file

def test_shipped_labels_cover_every_frozen_segment() -> None:
    rows = list(csv.DictReader(LABELS.open()))
    assert len(rows) == 51
    assert {r["section"] for r in rows} == {"talk", "qa"}
    assert sum(r["section"] == "qa" for r in rows) == 9


def test_shipped_labels_are_monotonic() -> None:
    """Independent confirmation of the assumption align_monotonic is built on."""
    seq = [int(r["true_slide"]) for r in csv.DictReader(LABELS.open())
           if r["true_slide"] != "none"]
    assert seq == sorted(seq)


def test_shipped_labels_never_use_a_slide_that_was_not_shown() -> None:
    used = {int(r["true_slide"]) for r in csv.DictReader(LABELS.open())
            if r["true_slide"] != "none"}
    assert not (used & {27}), "slide 27 was never shown"


def test_shipped_labels_carry_build_pair_alternatives() -> None:
    rows = list(csv.DictReader(LABELS.open()))
    alts = {r["also_correct"] for r in rows if r["also_correct"]}
    assert alts == {"10 11", "17 18"}


def test_pointing_windows_are_well_formed() -> None:
    rows = list(csv.DictReader(open("data/labels/pilot01/pilot01_pointing_windows.csv")))
    assert len(rows) == 10
    for r in rows:
        assert parse_mmss(r["start"]) < parse_mmss(r["end"])
        assert r["coverage"] in ("from_timestamp", "entire_slide")
