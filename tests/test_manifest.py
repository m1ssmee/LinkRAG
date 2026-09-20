"""Corpus manifest and the gold-vs-corpus staleness check."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from linkrag.core import EvidenceUnit, Location
from linkrag.manifest import (
    build_manifest,
    check_gold_manifest,
    file_digest,
    load_manifest,
    manifest_hash,
    write_manifest,
)


@pytest.fixture
def corpus(tmp_path: Path):
    a, b = tmp_path / "deck.pdf", tmp_path / "talk.mp3"
    a.write_bytes(b"slides bytes")
    b.write_bytes(b"audio bytes")
    units = [
        EvidenceUnit(id="d0", modality="text", content="x", source_file=str(a),
                     location=Location(page=1)),
        EvidenceUnit(id="d1", modality="figure", content="y", source_file=str(a),
                     location=Location(page=1)),
        EvidenceUnit(id="a0", modality="audio", content="z", source_file=str(b),
                     location=Location(start_s=0.0, end_s=3.0)),
    ]
    return [a, b], units


def test_manifest_records_files_hashes_and_unit_counts(corpus) -> None:
    paths, units = corpus
    m = build_manifest(paths, units)
    assert m["total_units"] == 3
    assert m["units_by_modality"] == {"audio": 1, "figure": 1, "text": 1}
    by_name = {f["name"]: f for f in m["files"]}
    assert by_name["deck.pdf"]["units"] == {"figure": 1, "text": 1}
    assert by_name["talk.mp3"]["units"] == {"audio": 1}
    assert by_name["deck.pdf"]["sha256"] == file_digest(paths[0])


def test_hash_ignores_timestamp_so_a_rebuild_does_not_invalidate_gold(corpus) -> None:
    """Re-ingesting the same corpus must produce the same hash, or every gold
    stamp would go stale on an unrelated rebuild."""
    paths, units = corpus
    first, second = build_manifest(paths, units), build_manifest(paths, units)
    assert first["hash"] == second["hash"]
    stale = dict(first, created_utc="1999-01-01T00:00:00Z")
    assert manifest_hash(stale) == manifest_hash(first)


def test_hash_changes_when_file_content_changes(corpus, tmp_path: Path) -> None:
    paths, units = corpus
    before = build_manifest(paths, units)["hash"]
    paths[0].write_bytes(b"different slides")
    assert build_manifest(paths, units)["hash"] != before


def test_hash_changes_when_unit_counts_change(corpus) -> None:
    """Same bytes, different extraction -- still a different corpus."""
    paths, units = corpus
    before = build_manifest(paths, units)["hash"]
    extra = units + [EvidenceUnit(id="d2", modality="figure", content="q",
                                  source_file=str(paths[0]), location=Location(page=2))]
    assert build_manifest(paths, extra)["hash"] != before


def test_missing_files_are_skipped_not_fatal(corpus, tmp_path: Path) -> None:
    paths, units = corpus
    m = build_manifest(paths + [tmp_path / "gone.pdf"], units)
    assert {f["name"] for f in m["files"]} == {"deck.pdf", "talk.mp3"}


def test_round_trip(corpus, tmp_path: Path) -> None:
    paths, units = corpus
    m = build_manifest(paths, units)
    out = write_manifest(m, tmp_path / "manifest.json")
    assert load_manifest(out) == m
    assert load_manifest(tmp_path / "absent.json") is None


# --------------------------------------------------------------- gold check

def test_matching_hash_produces_no_warning(corpus) -> None:
    m = build_manifest(*corpus)
    assert check_gold_manifest(m["hash"], m) is None


def test_mismatched_hash_warns_and_explains_the_risk(corpus) -> None:
    """Adding the OSDI paper made Q3 answerable from a source its gold locators do
    not mention: 4/4 gold terms with 0/2 locators retrieved. Nothing said so."""
    m = build_manifest(*corpus)
    warning = check_gold_manifest("deadbeefdeadbeef", m)
    assert warning and "deadbeefdeadbeef" in warning and m["hash"] in warning
    assert "understated" in warning


def test_unstamped_gold_warns(corpus) -> None:
    assert "no manifest stamp" in check_gold_manifest(None, build_manifest(*corpus))


def test_missing_manifest_warns_rather_than_crashing(corpus) -> None:
    assert "no corpus manifest" in check_gold_manifest("abc", None)


# ------------------------------------------------------- the real gold file

def test_shipped_gold_file_is_stamped() -> None:
    rows = [json.loads(l) for l in
            Path("tests/regression/pilot01_questions.jsonl").read_text().splitlines() if l.strip()]
    meta = next((r["_meta"] for r in rows if "_meta" in r), None)
    assert meta and meta.get("manifest_hash"), "gold must carry a corpus stamp"
    questions = [r for r in rows if "_meta" not in r]
    # machine-verified since 2026-09-20: every row carries verified unit ids and the
    # verification record, and Q1-Q4 are still present (possibly relabelled).
    assert meta.get("verifier") == "linkrag.eval.verify_gold"
    assert {"Q1", "Q2", "Q3", "Q4"} <= {q["qid"] for q in questions}
    for q in questions:
        assert q["gold_unit_ids"] and q["gold_units"] and q["verification"]["units_kept"] == len(q["gold_unit_ids"])


# ------------------------------------------- links <-> corpus manifest binding

def test_links_carry_the_manifest_stamp_and_mismatch_is_a_hard_error(tmp_path: Path) -> None:
    """Link ids only mean anything against the index that produced them. A links
    file paired with a different corpus makes link-following silently return the
    seeds -- indistinguishable from baseline."""
    from linkrag.core import Link
    from linkrag.link.align import LinkManifestMismatch, load_links, save_links

    path = save_links([Link("a", "b", "audio_slide", 0.9)], tmp_path / "links.jsonl",
                      manifest_hash="corpusAAA")
    assert load_links(path, expect_manifest="corpusAAA")[0].src_id == "a"
    assert len(load_links(path)) == 1, "no expectation given: load without checking"

    with pytest.raises(LinkManifestMismatch, match="corpusAAA"):
        load_links(path, expect_manifest="corpusBBB")


def test_unstamped_links_fail_against_any_expectation(tmp_path: Path) -> None:
    from linkrag.core import Link
    from linkrag.link.align import LinkManifestMismatch, load_links, save_links

    path = save_links([Link("a", "b", "audio_slide", 0.9)], tmp_path / "l.jsonl")
    with pytest.raises(LinkManifestMismatch, match="UNSTAMPED"):
        load_links(path, expect_manifest="corpusAAA")


def test_meta_line_is_not_returned_as_a_link(tmp_path: Path) -> None:
    from linkrag.core import Link
    from linkrag.link.align import load_links, save_links

    path = save_links([Link("a", "b", "deictic", 0.5)], tmp_path / "l.jsonl",
                      manifest_hash="h")
    links = load_links(path, expect_manifest="h")
    assert len(links) == 1 and links[0].link_type == "deictic"
