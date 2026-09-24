"""Skeleton contract: packages import, the data model round-trips, config has
the keys every stage reads. Fails loudly if someone renames a config key."""

from __future__ import annotations

import importlib
from dataclasses import asdict

import pytest

from linkrag.core import EvidenceUnit, Link, Location, load_config

SUBPACKAGES = ["ingest", "index", "link", "retrieve", "generate", "eval", "ui"]


@pytest.mark.parametrize("name", SUBPACKAGES)
def test_subpackage_imports(name: str) -> None:
    mod = importlib.import_module(f"linkrag.{name}")
    doc = mod.__doc__ or ""
    # The two-mode rule (DESIGN.md): every module documents both modes.
    assert "baseline" in doc and "linkrag" in doc, f"{name} must document both modes"


def test_evidence_unit_round_trips() -> None:
    unit = EvidenceUnit(
        id="u1",
        modality="audio",
        content="as you can see here",
        source_file="lecture.wav",
        location=Location(start_s=1.0, end_s=2.5),
        metadata={"speaker": "lecturer"},
    )
    restored = EvidenceUnit(**{**asdict(unit), "location": Location(**asdict(unit)["location"])})
    assert restored == unit
    assert unit.embedding is None


def test_location_cite_picks_the_right_axis() -> None:
    assert Location(start_s=612.0, end_s=628.5).cite() == "612.0s-628.5s"
    assert Location(page=14).cite() == "p.14"
    assert Location().cite() == "?"


def test_link_is_directed_and_scored() -> None:
    link = Link("u1", "u2", "deictic", 0.81)
    assert (link.src_id, link.dst_id) != (link.dst_id, link.src_id)
    assert 0.0 <= link.score <= 1.0


def test_default_config_has_keys_every_stage_reads() -> None:
    cfg = load_config()
    assert cfg["mode"] in ("baseline", "linkrag")
    assert cfg["models"]["embedding"] == "BAAI/bge-m3"
    assert cfg["retrieve"]["top_k"] > 0
    assert cfg["retrieve"]["linkrag"]["k_seed"] > 0
    assert cfg["retrieve"]["linkrag"]["k_final"] >= cfg["retrieve"]["linkrag"]["k_seed"]
    # Only thresholds the code actually reads. Seven keys that were read nowhere
    # were removed in the audit; audio_slide filtering is link.align.min_score.
    for link_type in ("figure_text", "deictic"):
        assert 0.0 < cfg["link"][f"{link_type}_threshold"] < 1.0
    assert cfg["link"]["align"]["min_score"] >= 0.0
