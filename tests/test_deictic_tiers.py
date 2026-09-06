"""Cue tiering, pair collapsing, and the same_slide link type."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from linkrag.core import EvidenceUnit, Link, Location
from linkrag.link.deictic import (
    DEFAULT_TIER_WEIGHTS,
    deictic_pairs,
    expand_cues,
    find_cues,
    resolve_deictic,
    tier_breakdown,
    tier_of,
    write_pairs_csv,
)
from linkrag.link.same_slide import link_same_slide


def _audio(aid, sentence, start=0.0, deck=False):
    ws = [[start + i * 0.5, start + (i + 1) * 0.5, w] for i, w in enumerate(sentence.split())]
    return EvidenceUnit(id=aid, modality="audio", content=sentence, source_file="talk.mp3",
                        location=Location(start_s=ws[0][0], end_s=ws[-1][1]),
                        metadata={"words": ws, "slide_deck": deck})


def _fig(fid, page, caption="", deck=True):
    return EvidenceUnit(id=fid, modality="figure", content=caption, source_file="deck.pdf",
                        location=Location(page=page), metadata={"slide_deck": deck})


def _txt(tid, page, content="slide text", deck=True):
    return EvidenceUnit(id=tid, modality="text", content=content, source_file="deck.pdf",
                        location=Location(page=page), metadata={"slide_deck": deck})


# ------------------------------------------------------------------ tiering

def test_tier_1_is_explicit_object_deixis() -> None:
    assert tier_of("this arrow here", "so this arrow here shows it.", ["see"]) == 1
    assert tier_of("the box on the right", "look at the box on the right.", ["see"]) == 1


def test_tier_2_is_a_bare_pronoun_with_visual_context() -> None:
    assert tier_of("this", "you can see this on the graph.", ["see", "graph"]) == 2


def test_tier_3_is_a_bare_pronoun_alone() -> None:
    assert tier_of("this", "and this is what we do next.", ["see", "graph"]) == 3


def test_tier_uses_the_sentence_not_the_whole_segment() -> None:
    """A visual verb two sentences away is not context for this pronoun."""
    sentence = "and this is the next step."
    assert tier_of("this", sentence, ["see", "graph"]) == 3


def test_find_cues_assigns_tiers_and_sentences() -> None:
    unit = _audio("a1", "we plot the recall. and this is the result. so this arrow here helps.")
    by_phrase = {c.text: c for c in find_cues(unit, expand_cues())}
    assert by_phrase["this arrow here"].tier == 1
    # "this" inside "and this is the result." has no visual word in ITS sentence
    assert by_phrase["this"].tier == 3
    assert by_phrase["this"].sentence.strip().startswith("and this is")


def test_tier_weight_changes_ranking_without_moving_thresholds(stub_encoder) -> None:
    """Same figure, same similarity: an explicit cue must outrank a bare pronoun."""
    explicit = _audio("a_hi", "so this arrow here shows the attention heatmap.")
    bare = _audio("a_lo", "and this is the attention heatmap.")
    fig = _fig("f", 5, "attention heatmap")
    enc = stub_encoder(["attention heatmap", explicit.content, bare.content])

    hi, = resolve_deictic([explicit], [fig], encoder=enc, slide_of_audio={"a_hi": 5},
                          mode="linkrag", cues=expand_cues(), threshold=0.0)
    lo, = resolve_deictic([bare], [fig], encoder=enc, slide_of_audio={"a_lo": 5},
                          mode="linkrag", cues=expand_cues(), threshold=0.0)
    assert hi.metadata["tier"] == 1 and lo.metadata["tier"] == 3
    assert hi.score > lo.score


def test_tier_weights_are_configurable(stub_encoder) -> None:
    bare = _audio("a1", "and this is the heatmap.")
    fig = _fig("f", 5, "attention heatmap")
    enc = stub_encoder(["attention heatmap", bare.content])
    kw = dict(encoder=enc, slide_of_audio={"a1": 5}, mode="linkrag",
              cues=expand_cues(), threshold=0.0)
    low, = resolve_deictic([bare], [fig], tier_weights={1: 1.0, 2: 0.6, 3: 0.0}, **kw)
    high, = resolve_deictic([bare], [fig], tier_weights={1: 1.0, 2: 0.6, 3: 1.0}, **kw)
    assert high.score > low.score
    assert DEFAULT_TIER_WEIGHTS[1] > DEFAULT_TIER_WEIGHTS[2] > DEFAULT_TIER_WEIGHTS[3]


# ----------------------------------------------------------- pair collapsing

def test_pairs_collapse_repeated_cues_keeping_the_best() -> None:
    """41 raw links over 22 real pairs on pilot01: the raw count double-counts."""
    links = [
        Link("a1", "f1", "deictic", 0.61, {"tier": 3, "phrase": "this"}),
        Link("a1", "f1", "deictic", 0.62, {"tier": 2, "phrase": "that"}),
        Link("a1", "f2", "deictic", 0.55, {"tier": 3, "phrase": "here"}),
        Link("a1", "s1", "audio_slide", 0.9),
    ]
    pairs = deictic_pairs(links)
    assert len(pairs) == 2, "audio_slide must be ignored, duplicates collapsed"
    top = pairs[0]
    assert (top["src_id"], top["dst_id"]) == ("a1", "f1")
    assert top["score"] == pytest.approx(0.62) and top["tier"] == 2
    assert top["cues"] == 2, "cue count is kept as a secondary statistic"


def test_tier_breakdown_counts_pairs_not_cues() -> None:
    pairs = [{"tier": 3, "score": 0.5}, {"tier": 3, "score": 0.7}, {"tier": 1, "score": 0.9}]
    stats = tier_breakdown(pairs)
    assert stats[3]["count"] == 2 and stats[3]["mean_score"] == pytest.approx(0.6)
    assert stats[1]["count"] == 1


def test_pairs_csv_has_a_row_per_pair_with_verifiable_columns(tmp_path: Path) -> None:
    audio = _audio("a1", "and this is the heatmap.", start=120.0)
    fig = _fig("f1", 7)
    pairs = deictic_pairs([Link("a1", "f1", "deictic", 0.61,
                                {"tier": 3, "phrase": "this", "sentence": "and this is the heatmap.",
                                 "phrase_start_s": 120.5, "slide_page": 7})])
    out = write_pairs_csv(pairs, {"a1": audio, "f1": fig}, tmp_path / "pairs.csv")
    row, = list(csv.DictReader(out.open()))
    assert row["segment_id"] == "a1" and row["figure_page"] == "7"
    assert row["phrase"] == "this" and row["tier"] == "3"
    assert float(row["segment_start_s"]) == pytest.approx(120.0)
    assert float(row["phrase_start_s"]) == pytest.approx(120.5)


# -------------------------------------------------------------- same_slide

def test_same_slide_links_each_deck_figure_to_its_page_text() -> None:
    figs = [_fig("f1", 3), _fig("f2", 3), _fig("f3", 9)]
    texts = [_txt("t3", 3), _txt("t9", 9)]
    links = link_same_slide(figs, texts)
    assert {(l.src_id, l.dst_id) for l in links} == {("f1", "t3"), ("f2", "t3"), ("f3", "t9")}
    assert all(l.link_type == "same_slide" and l.score == 1.0 for l in links)
    assert links[0].metadata["page"] == 3


def test_same_slide_ignores_documents_not_flagged_as_decks() -> None:
    """A paper's figures are not co-located with their explanation -- that is
    exactly why figure_text exists."""
    paper_fig = _fig("pf", 3, deck=False)
    paper_txt = _txt("pt", 3, deck=False)
    assert link_same_slide([paper_fig], [paper_txt]) == []


def test_same_slide_is_ablatable() -> None:
    figs, texts = [_fig("f1", 3)], [_txt("t3", 3)]
    assert link_same_slide(figs, texts, mode="baseline") == []
    assert link_same_slide(figs, texts, enabled=False) == []
    assert len(link_same_slide(figs, texts, mode="linkrag", enabled=True)) == 1


def test_same_slide_covers_every_text_chunk_on_the_page() -> None:
    links = link_same_slide([_fig("f1", 3)], [_txt("t3a", 3), _txt("t3b", 3)])
    assert {l.dst_id for l in links} == {"t3a", "t3b"}


def test_same_slide_does_not_cross_files() -> None:
    other = EvidenceUnit(id="other", modality="text", content="x",
                         source_file="other_deck.pdf", location=Location(page=3),
                         metadata={"slide_deck": True})
    assert link_same_slide([_fig("f1", 3)], [other]) == []
