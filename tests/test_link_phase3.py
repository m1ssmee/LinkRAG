"""figure<->paragraph links and deictic resolution (Phase 3)."""

from __future__ import annotations

import pytest

from linkrag.core import EvidenceUnit, Link, Location
from linkrag.link.deictic import (
    DEFAULT_CUES,
    find_cues,
    resolve_deictic,
    slide_map_from_links,
)
from linkrag.link.figure_text import (
    figure_number,
    figure_text_scores,
    link_figures_to_text,
    referenced,
)


def _fig(fid, page, caption=""):
    return EvidenceUnit(id=fid, modality="figure", content=caption,
                        source_file="deck.pdf", location=Location(page=page))


def _txt(tid, page, content):
    return EvidenceUnit(id=tid, modality="text", content=content,
                        source_file="deck.pdf", location=Location(page=page))


def _audio(aid, words, start=0.0):
    ws = [[start + i * 0.5, start + (i + 1) * 0.5, w] for i, w in enumerate(words.split())]
    return EvidenceUnit(id=aid, modality="audio", content=words, source_file="talk.mp3",
                        location=Location(start_s=ws[0][0], end_s=ws[-1][1]),
                        metadata={"words": ws})


# ------------------------------------------------------------- figure numbers

def test_figure_number_and_references() -> None:
    assert figure_number("Figure 1: attention heatmap") == 1
    assert figure_number("Fig. 12 shows") == 12
    assert figure_number("no number") is None
    assert referenced("As Fig. 2 shows, and Figure 10 confirms") == {("figure", 2), ("figure", 10)}
    assert referenced("") == set()


# ------------------------------------------------------- figure <-> paragraph

def test_explicit_reference_beats_page_proximity(stub_encoder) -> None:
    """P3's gap: the explaining paragraph can be pages away. A paragraph that
    names the figure must outrank an unrelated one sitting on the same page."""
    fig = _fig("f1", 2, "Figure 3: attention heatmap layer six")
    same_page = _txt("t_near", 2, "unrelated boilerplate about scheduling")
    far_ref = _txt("t_far", 9, "Figure 3 shows the attention heatmap in detail")
    corpus = [fig.content, same_page.content, far_ref.content]

    scores, _reasons = figure_text_scores([fig], [same_page, far_ref], encoder=stub_encoder(corpus))
    assert scores[0, 1] > scores[0, 0], "explicit 'Figure 3' reference must win"


def test_page_proximity_decays_but_does_not_filter(stub_encoder) -> None:
    fig = _fig("f1", 1, "attention heatmap")
    near, far = _txt("t1", 2, "attention heatmap"), _txt("t2", 20, "attention heatmap")
    s, _r = figure_text_scores([fig], [near, far], encoder=stub_encoder(["attention heatmap"]))
    assert s[0, 0] > s[0, 1], "nearer page should score higher"
    assert s[0, 1] > 0.0, "a distant page must remain a candidate, not be filtered"


def test_baseline_mode_keeps_only_same_page_links(stub_encoder) -> None:
    """The ablation: a page-independent pipeline cannot reach across pages."""
    fig = _fig("f1", 2, "Figure 3: attention heatmap")
    same = _txt("t_same", 2, "Figure 3 attention heatmap discussion")
    far = _txt("t_far", 9, "Figure 3 shows the attention heatmap in detail")
    enc = stub_encoder([fig.content, same.content, far.content])

    base = link_figures_to_text([fig], [same, far], encoder=enc, mode="baseline", threshold=0.0)
    link = link_figures_to_text([fig], [same, far], encoder=enc, mode="linkrag", threshold=0.0)
    assert {l.dst_id for l in base} == {"t_same"}
    assert {l.dst_id for l in link} == {"t_same", "t_far"}
    assert all(l.link_type == "figure_text" for l in base + link)


def test_threshold_and_cap_are_honoured(stub_encoder) -> None:
    fig = _fig("f1", 1, "attention heatmap")
    texts = [_txt(f"t{i}", 1, "attention heatmap") for i in range(5)]
    enc = stub_encoder(["attention heatmap"])
    assert len(link_figures_to_text([fig], texts, encoder=enc, threshold=0.0,
                                    max_links_per_unit=2)) == 2
    assert link_figures_to_text([fig], texts, encoder=enc, threshold=0.99) == []


def test_figure_scores_reject_empty_input(stub_encoder) -> None:
    with pytest.raises(ValueError, match="at least one"):
        figure_text_scores([], [_txt("t", 1, "x")], encoder=stub_encoder(["x"]))


# --------------------------------------------------------------------- cues

def test_find_cues_prefers_the_longest_match() -> None:
    unit = _audio("a1", "and as you can see here the weights concentrate on this slide")
    cues = find_cues(unit)
    texts = [c.text for c in cues]
    assert "as you can see" in texts and "this slide" in texts
    assert texts.count("this") == 0, "'this' must not double-report inside 'this slide'"


def test_cue_strength_favours_multiword_deictics() -> None:
    unit = _audio("a1", "as you can see here")
    by_text = {c.text: c.strength for c in find_cues(unit)}
    assert by_text["as you can see"] > by_text["here"]


def test_find_cues_needs_word_timestamps() -> None:
    bare = EvidenceUnit(id="a", modality="audio", content="look here", source_file="t.mp3")
    assert find_cues(bare) == []


def test_cue_carries_its_own_timestamp_and_context() -> None:
    unit = _audio("a1", "alpha beta as you can see here gamma delta", start=10.0)
    cue = next(c for c in find_cues(unit) if c.text == "as you can see")
    assert cue.start_s == pytest.approx(11.0)
    assert "gamma" in cue.context and "alpha" in cue.context


# ------------------------------------------------------- deictic resolution

def test_linkrag_restricts_candidates_to_the_aligned_slide(stub_encoder) -> None:
    audio = _audio("a1", "as you can see here the attention heatmap")
    on_slide = _fig("f_on", 5, "attention heatmap")
    off_slide = _fig("f_off", 20, "attention heatmap")
    enc = stub_encoder(["attention heatmap", audio.content])

    links = resolve_deictic([audio], [on_slide, off_slide], encoder=enc,
                            slide_of_audio={"a1": 5}, mode="linkrag", threshold=0.0)
    assert {l.dst_id for l in links} == {"f_on"}, "off-slide figure must be excluded"
    assert all(l.link_type == "deictic" for l in links)


def test_baseline_does_not_use_the_alignment(stub_encoder) -> None:
    """Regression: baseline once still added w_slide * on_slide, so the 'no
    alignment' ablation was silently using the alignment and measured nothing."""
    audio = _audio("a1", "as you can see here the attention heatmap")
    on_slide = _fig("f_on", 5, "unrelated scheduling text")
    off_slide = _fig("f_off", 20, "attention heatmap")
    enc = stub_encoder(["attention heatmap", "unrelated scheduling text", audio.content])
    slide_map = {"a1": 5}

    with_map = resolve_deictic([audio], [on_slide, off_slide], encoder=enc,
                               slide_of_audio=slide_map, mode="baseline", threshold=0.0)
    without_map = resolve_deictic([audio], [on_slide, off_slide], encoder=enc,
                                  slide_of_audio=None, mode="baseline", threshold=0.0)
    assert [(l.dst_id, round(l.score, 6)) for l in with_map] == \
           [(l.dst_id, round(l.score, 6)) for l in without_map], \
        "baseline scores must be identical with and without the alignment"
    assert with_map[0].dst_id == "f_off", "baseline should follow similarity alone"


def test_linkrag_mode_requires_the_alignment(stub_encoder) -> None:
    audio = _audio("a1", "see here")
    with pytest.raises(ValueError, match="needs slide_of_audio"):
        resolve_deictic([audio], [_fig("f", 1)], encoder=stub_encoder(["x"]),
                        slide_of_audio=None, mode="linkrag")


def test_no_cues_means_no_links(stub_encoder) -> None:
    audio = _audio("a1", "the optimizer uses a cosine schedule with warmup")
    assert resolve_deictic([audio], [_fig("f", 1, "x")], encoder=stub_encoder(["x"]),
                           slide_of_audio={"a1": 1}, mode="linkrag") == []


def test_no_figures_means_no_links(stub_encoder) -> None:
    audio = _audio("a1", "as you can see here")
    assert resolve_deictic([audio], [], encoder=stub_encoder(["x"]),
                           slide_of_audio={"a1": 1}, mode="linkrag") == []


def test_slide_map_from_links_ignores_other_link_types() -> None:
    slides = [_txt("s1", 7, "x"), _txt("s2", 8, "y")]
    links = [Link("a1", "s1", "audio_slide", 0.9),
             Link("f1", "s2", "figure_text", 0.8)]
    assert slide_map_from_links(links, slides) == {"a1": ("deck.pdf", 7)}


def test_scores_are_normalised_so_the_threshold_means_the_same_in_both_modes(
    stub_encoder,
) -> None:
    """baseline cannot earn the slide term, so on a raw scale its ceiling is
    w_cue+w_dense = 0.5 against a 0.45 threshold -- it would emit ~nothing by
    construction and the ablation would flatter linkrag for the wrong reason."""
    audio = _audio("a1", "as you can see here the attention heatmap")
    fig = _fig("f", 5, "attention heatmap")
    enc = stub_encoder(["attention heatmap", audio.content])

    link = resolve_deictic([audio], [fig], encoder=enc, slide_of_audio={"a1": 5},
                           mode="linkrag", threshold=0.0)
    base = resolve_deictic([audio], [fig], encoder=enc, mode="baseline", threshold=0.0)
    assert 0.0 <= base[0].score <= 1.0 and 0.0 <= link[0].score <= 1.0
    # identical cue + identical text similarity, so the only difference is the
    # slide term; normalised, baseline must not be crushed to near zero
    assert base[0].score > 0.3


# ------------------------------------------------- spec 1a: tables + descriptive

def test_table_reference_does_not_match_a_figure(stub_encoder) -> None:
    """"Table 2" must not link to "Figure 2" -- kind is part of the identity."""
    table = _fig("tbl", 3, "Table 2: recall at each K")
    fig = _fig("f2", 3, "Figure 2: attention heatmap")
    txt = _txt("t", 3, "Table 2 reports recall at each value of K")
    enc = stub_encoder([table.content, fig.content, txt.content])
    s, _r = figure_text_scores([table, fig], [txt], encoder=enc)
    assert s[0, 0] > s[1, 0], "the Table must outscore the Figure for a 'Table 2' mention"


def test_numbered_reference_is_limited_to_adjacent_pages(stub_encoder) -> None:
    fig = _fig("f", 2, "Figure 3: heatmap")
    near, far = _txt("near", 3, "see Figure 3"), _txt("far", 15, "see Figure 3")
    enc = stub_encoder(["figure 3 heatmap", "see figure 3"])
    s, r = figure_text_scores([fig], [near, far], encoder=enc, reference_page_window=1)
    assert r[0][0].get("reference") == "figure 3"
    assert "reference" not in r[0][1], "a reference 13 pages away is not a reference"
    assert s[0, 0] > s[0, 1]


def test_descriptive_reference_respects_direction(stub_encoder) -> None:
    """"the diagram below" should match a figure below the text, not above it."""
    text = EvidenceUnit(id="t", modality="text", content="as the diagram below shows",
                        source_file="d.pdf", location=Location(page=1, bbox=(72, 100, 400, 130)))
    below = EvidenceUnit(id="f_below", modality="figure", content="",
                         source_file="d.pdf", location=Location(page=1, bbox=(72, 200, 400, 400)))
    above = EvidenceUnit(id="f_above", modality="figure", content="",
                         source_file="d.pdf", location=Location(page=1, bbox=(72, 10, 400, 60)))
    enc = stub_encoder(["as the diagram below shows"])
    _s, r = figure_text_scores([below, above], [text], encoder=enc)
    assert "descriptive" in r[0][0], "figure below the text should match 'the diagram below'"
    assert "descriptive" not in r[1][0], "figure above it should not"


# ------------------------------------------------- spec 1b: bbox layout proximity

def test_layout_proximity_uses_bbox_distance_not_page_number(stub_encoder) -> None:
    text = EvidenceUnit(id="t", modality="text", content="attention heatmap discussion",
                        source_file="d.pdf", location=Location(page=1, bbox=(72, 100, 400, 130)))
    near = EvidenceUnit(id="f_near", modality="figure", content="attention heatmap",
                        source_file="d.pdf", location=Location(page=1, bbox=(72, 140, 400, 300)))
    far = EvidenceUnit(id="f_far", modality="figure", content="attention heatmap",
                       source_file="d.pdf", location=Location(page=1, bbox=(72, 700, 400, 780)))
    enc = stub_encoder(["attention heatmap", "attention heatmap discussion"])
    _s, r = figure_text_scores([near, far], [text], encoder=enc, layout_max_gap_pt=220.0)
    assert r[0][0]["layout_gap_pt"] == pytest.approx(10.0)
    assert "layout_gap_pt" not in r[1][0], "570pt away is beyond the layout window"


def test_vertical_gap_is_zero_when_boxes_overlap() -> None:
    from linkrag.link.figure_text import vertical_gap

    assert vertical_gap((0, 0, 10, 100), (0, 50, 10, 150)) == 0.0
    assert vertical_gap((0, 0, 10, 100), (0, 140, 10, 200)) == pytest.approx(40.0)
    assert vertical_gap(None, (0, 0, 1, 1)) is None


# --------------------------------------------- spec 7: end-to-end on the fixture

def test_figure_text_links_the_cross_page_reference(pdf_path, tmp_path, stub_encoder) -> None:
    """The fixture deck says "As Figure 1 shows" on page 1; the figure is on page 2.
    linkrag must connect them, baseline (same-page only) must not."""
    from linkrag.ingest.pdf import ingest_pdf

    units = ingest_pdf(pdf_path, figures_dir=tmp_path / "f",
                       chunk_tokens=400, overlap_tokens=20)
    figures = [u for u in units if u.modality == "figure"]
    texts = [u for u in units if u.modality == "text"]
    ref_text = next(t for t in texts if "Figure 1" in t.content and t.location.page == 1)
    enc = stub_encoder([u.content for u in units])

    link = link_figures_to_text(figures, texts, encoder=enc, mode="linkrag", threshold=0.0)
    pairs = {(l.src_id, l.dst_id) for l in link}
    assert any(dst == ref_text.id for _src, dst in pairs), "cross-page reference missed"
    assert all(l.link_type == "figure_text" for l in link)

    base = link_figures_to_text(figures, texts, encoder=enc, mode="baseline", threshold=0.0)
    assert not any(l.dst_id == ref_text.id for l in base), \
        "baseline is same-page only and must not reach page 1"


def test_links_record_why_they_were_made(stub_encoder) -> None:
    fig = _fig("f", 2, "Figure 3: heatmap")
    txt = _txt("t", 2, "see Figure 3 for the heatmap")
    enc = stub_encoder([fig.content, txt.content])
    link, = link_figures_to_text([fig], [txt], encoder=enc, threshold=0.0)
    assert link.metadata.get("reference") == "figure 3"
    assert "dense" in link.metadata


# --------------------------------------- spec 2a/2c: "this arrow here" + metadata

def test_this_arrow_here_is_detected_as_one_phrase(deictic_audio_unit) -> None:
    from linkrag.link.deictic import expand_cues

    cues = [c.text for c in find_cues(deictic_audio_unit, expand_cues())]
    assert "this arrow here" in cues
    assert "this" not in cues and "here" not in cues, "must not also fire as bare cues"


def test_deictic_link_stores_the_phrase_in_metadata(deictic_audio_unit, stub_encoder) -> None:
    from linkrag.link.deictic import expand_cues

    fig = _fig("f_on", 5, "attention heatmap with an arrow on the subject token")
    enc = stub_encoder([fig.content, deictic_audio_unit.content])
    link, = resolve_deictic([deictic_audio_unit], [fig], encoder=enc,
                            slide_of_audio={deictic_audio_unit.id: 5}, mode="linkrag",
                            cues=expand_cues(), threshold=0.0)
    assert link.link_type == "deictic"
    assert link.metadata["phrase"] == "this arrow here"
    assert link.metadata["slide_page"] == 5
    assert link.metadata["phrase_start_s"] < link.metadata["phrase_end_s"]
    assert "keyword_overlap" in link.metadata


def test_keyword_overlap_breaks_ties_between_figures_on_one_slide(stub_encoder) -> None:
    """spec 2b: when a slide has several figures, pick by keyword overlap with the
    figure's OCR/caption text."""
    audio = _audio("a1", "as you can see here the recall curve for resnet eighteen")
    match = _fig("f_match", 5, "recall curve resnet eighteen fewer layers")
    other = _fig("f_other", 5, "ingest cost versus query latency tradeoff")
    enc = stub_encoder([match.content, other.content, audio.content])
    links = resolve_deictic([audio], [match, other], encoder=enc,
                            slide_of_audio={"a1": 5}, mode="linkrag", threshold=0.0)
    assert links[0].dst_id == "f_match"
    assert links[0].metadata["keyword_overlap"] > 0.0


def test_figure_text_ties_break_by_text_position(stub_encoder) -> None:
    """np.argsort default is quicksort: identical scores ordered arbitrarily,
    deciding which figure_text links get emitted under max_links_per_unit."""
    fig = _fig("f", 1, "attention heatmap")
    texts = [_txt(f"t{i}", 1, "attention heatmap") for i in range(5)]
    links = link_figures_to_text([fig], texts, encoder=stub_encoder(["attention heatmap"]),
                                 threshold=0.0, max_links_per_unit=3)
    assert [l.dst_id for l in links] == ["t0", "t1", "t2"]


def test_deictic_ties_break_by_ascending_figure_index(stub_encoder) -> None:
    """`scored.sort(reverse=True)` on (score, j, overlap) broke ties by *descending*
    figure index, so the last figure on a slide silently won every tie."""
    audio = _audio("a1", "as you can see here the attention heatmap")
    figs = [_fig(f"f{i}", 5, "attention heatmap") for i in range(4)]
    enc = stub_encoder(["attention heatmap", audio.content])
    links = resolve_deictic([audio], figs, encoder=enc, slide_of_audio={"a1": 5},
                            mode="linkrag", threshold=0.0, max_links_per_unit=2)
    assert [l.dst_id for l in links] == ["f0", "f1"]


def test_deictic_candidates_require_the_deck_file_not_just_the_page_number() -> None:
    """Found by the relatedness gate (2026-09-21): with the paper in the corpus,
    54 of 169 deictic links pointed at paper figures whose page number happened to
    equal the aligned slide number."""
    from linkrag.core import EvidenceUnit, Location
    enc = lambda texts: [[1.0, 0.0] for _ in texts]
    audio = EvidenceUnit(id="a1", modality="audio", content="as you can see here the recall goes up",
                         source_file="talk.mp3", location=Location(start_s=0.0, end_s=5.0),
                         metadata={"words": [(i * 0.5, i * 0.5 + 0.4, w)
                                             for i, w in enumerate("as you can see here the recall goes up".split())]})
    deck_fig = EvidenceUnit(id="deck:p5:g0", modality="figure", content="recall vs K",
                            source_file="deck.pdf", location=Location(page=5))
    paper_fig = EvidenceUnit(id="paper:p5:c0", modality="figure", content="recall vs K",
                             source_file="paper.pdf", location=Location(page=5))
    links = resolve_deictic([audio], [deck_fig, paper_fig], encoder=enc,
                            slide_of_audio={"a1": ("deck.pdf", 5)}, mode="linkrag", threshold=0.0)
    assert {l.dst_id for l in links} == {"deck:p5:g0"}
