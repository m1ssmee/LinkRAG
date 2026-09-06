"""Converting the ear-labelled slide timeline into per-segment ground truth."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from make_alignment_labels import label_segment, load_timeline, parse_mmss  # noqa: E402

TIMELINE = Path("data/labels/pilot01/pilot01_slide_timeline.csv")
LABELS = Path("data/labels/pilot01/pilot01_alignment_labels.csv")


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


def test_zero_length_slides_are_flagged(timeline) -> None:
    slides, _ = timeline
    assert {s["slide"] for s in slides if s["zero_length"]} == {9, 11, 18}


def test_majority_overlap_wins(timeline) -> None:
    slides, outro = timeline
    # 0:45-1:37 is slide 3; a segment mostly inside it must be slide 3
    slide, section, _amb = label_segment(50.0, 73.0, slides, outro)
    assert (slide, section) == (3, "talk")


def test_zero_length_slide_can_never_be_the_majority(timeline) -> None:
    """Slides 9, 11 and 18 have start == end. A segment straddling one takes the
    neighbour covering most of it, and is marked ambiguous."""
    slides, outro = timeline
    slide, section, ambiguous = label_segment(330.0, 360.0, slides, outro)  # straddles 9 @5:40
    assert slide not in (9, 11, 18)
    assert section == "talk" and ambiguous is True


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


def test_shipped_labels_never_use_a_zero_length_slide() -> None:
    used = {int(r["true_slide"]) for r in csv.DictReader(LABELS.open())
            if r["true_slide"] != "none"}
    assert not (used & {9, 11, 18})
    assert not (used & {27}), "slide 27 was never shown"
