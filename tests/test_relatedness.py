"""Relatedness gate: majority verdicts land in link metadata; gated loading drops
failures; pass rates sum per type."""

from __future__ import annotations

import json

from linkrag.core import EvidenceUnit, Link, Location
from linkrag.link.align import load_links, save_links
from linkrag.link.relatedness import apply_verdicts, gate_links, pass_rates


def _u(uid, modality, content):
    return EvidenceUnit(id=uid, modality=modality, content=content, source_file="x.pdf",
                        location=Location(page=1))


def test_gate_flags_and_load_drops(tmp_path):
    units = [_u("a1", "audio", "Focus is 57 times cheaper than ingest heavy"),
             _u("p1", "text", "Focus: 57X cheaper ($380 -> $7 per month per stream)"),
             _u("p2", "text", "Thank you for coming, questions at poster 47")]
    links = [Link("a1", "p1", "audio_slide", 0.7), Link("a1", "p2", "deictic", 0.6)]
    judge = lambda s, p: json.dumps({"related": "57" in p.split("B (")[1] and "57 times" in p,
                                     "subject": "57x cheaper"})
    verdicts = gate_links(links, units, judge, runs=3, workers=2)
    assert [v.passed for v in verdicts] == [True, False]
    assert pass_rates(verdicts) == {"audio_slide": (1, 1), "deictic": (1, 0)}
    save_links(apply_verdicts(links, verdicts), tmp_path / "l.jsonl", manifest_hash="h")
    assert [l.link_type for l in load_links(tmp_path / "l.jsonl", "h")] == ["audio_slide"]
    assert load_links.last_dropped == 1
    assert len(load_links(tmp_path / "l.jsonl", "h", gated=False)) == 2
    flagged = load_links(tmp_path / "l.jsonl", "h", gated=False)[1].metadata["relatedness"]
    assert flagged == {"passed": False, "votes": ["no", "no", "no"], "subject": ""}


def test_ungated_file_loads_whole(tmp_path):
    save_links([Link("a", "b", "audio_slide", 0.5)], tmp_path / "l.jsonl", manifest_hash="h")
    assert len(load_links(tmp_path / "l.jsonl", "h")) == 1 and load_links.last_dropped == 0
