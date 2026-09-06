from __future__ import annotations

from pathlib import Path

import pytest

from linkrag.ingest import ingest_file
from linkrag.ingest.audio import Segment, Word, pack, segment_words, split_sentences
from linkrag.ingest.docx import ingest_docx
from linkrag.ingest.pdf import _Block, chunk_page, find_caption, ingest_pdf

from conftest import requires_tesseract


# ------------------------------------------------------------------ chunking

def test_chunk_page_respects_size_and_overlap() -> None:
    blocks = [_Block(" ".join(f"w{i}" for i in range(100)), (0.0, 0.0, 10.0, 10.0))]
    chunks = chunk_page(blocks, chunk_tokens=40, overlap_tokens=20)  # 30 words, 15 overlap
    assert len(chunks) > 1
    assert all(len(text.split()) <= 30 for text, _ in chunks)
    first, second = chunks[0][0].split(), chunks[1][0].split()
    assert first[-15:] == second[:15], "consecutive chunks must share the overlap"


def test_chunk_page_bbox_spans_first_to_last_block() -> None:
    blocks = [
        _Block("alpha beta", (10.0, 100.0, 200.0, 120.0)),
        _Block("gamma delta", (12.0, 130.0, 260.0, 150.0)),
    ]
    (text, bbox), = chunk_page(blocks, chunk_tokens=100, overlap_tokens=0)
    assert text == "alpha beta gamma delta"
    assert bbox == (10.0, 100.0, 260.0, 150.0)


def test_chunk_page_handles_empty_input() -> None:
    assert chunk_page([]) == []
    assert chunk_page([_Block("   ", (0.0, 0.0, 1.0, 1.0))]) == []


def test_chunk_page_does_not_emit_a_redundant_tail() -> None:
    # 32 words at size 30/overlap 15: the 2nd window covers the rest, so no 3rd.
    blocks = [_Block(" ".join(f"w{i}" for i in range(32)), (0.0, 0.0, 1.0, 1.0))]
    assert len(chunk_page(blocks, chunk_tokens=40, overlap_tokens=20)) == 2


# ------------------------------------------------------------------ captions

def test_find_caption_picks_nearest_block_below() -> None:
    image = (72.0, 100.0, 472.0, 220.0)
    blocks = [
        _Block("above the image", (72.0, 40.0, 472.0, 60.0)),
        _Block("the caption", (72.0, 226.0, 472.0, 244.0)),
        _Block("further below", (72.0, 300.0, 472.0, 320.0)),
    ]
    assert find_caption(image, blocks) == "the caption"


def test_find_caption_ignores_distant_and_non_overlapping_blocks() -> None:
    image = (72.0, 100.0, 300.0, 220.0)
    assert find_caption(image, [_Block("far below", (72.0, 900.0, 300.0, 920.0))]) == ""
    # different column: no horizontal overlap
    assert find_caption(image, [_Block("other column", (400.0, 226.0, 560.0, 244.0))]) == ""


# ----------------------------------------------------------------- pdf / docx

def test_ingest_pdf_produces_text_and_figure_units(pdf_path: Path, tmp_path: Path) -> None:
    units = ingest_pdf(pdf_path, figures_dir=tmp_path / "figures", chunk_tokens=60, overlap_tokens=10)

    text_units = [u for u in units if u.modality == "text"]
    figures = [u for u in units if u.modality == "figure"]
    assert len(text_units) > 1
    assert len(figures) == 1

    for unit in text_units:
        assert unit.location.page in (1, 2)
        assert unit.location.bbox is not None
        assert unit.id.startswith("notes:p")

    figure = figures[0]
    assert figure.location.page == 2
    assert "attention heatmap" in figure.content.lower(), "naive caption lookup should find it"
    assert Path(figure.metadata["image_path"]).exists()


def test_ingest_pdf_skips_tiny_images(pdf_path: Path, tmp_path: Path) -> None:
    units = ingest_pdf(pdf_path, figures_dir=tmp_path / "f", min_figure_area_px=10**9)
    assert not [u for u in units if u.modality == "figure"]


def test_ingest_docx_chunks_text(docx_path: Path) -> None:
    units = ingest_docx(docx_path, chunk_tokens=40, overlap_tokens=10)
    assert units and all(u.modality == "text" for u in units)
    assert all(u.location.page is None for u in units), "docx has no page geometry"
    assert "attention" in " ".join(u.content for u in units).lower()


# --------------------------------------------------------------------- image

@requires_tesseract
def test_ingest_image_ocrs_text(png_path: Path) -> None:
    (unit,) = ingest_file(png_path)
    assert unit.modality == "figure"
    assert "ATTENTION" in unit.content.upper()
    assert unit.metadata["ocr"] is True


# --------------------------------------------------------------------- audio

def _words(spans: list[tuple[float, float]]) -> list[Word]:
    return [Word(s, e, f"w{i}") for i, (s, e) in enumerate(spans)]


def test_segment_words_cuts_at_the_window() -> None:
    words = _words([(t, t + 1.0) for t in range(0, 70)])
    buckets = segment_words(words, window_seconds=30.0)
    assert len(buckets) == 3
    assert [len(b) for b in buckets] == [30, 30, 10]
    # units end on a word boundary, and are contiguous
    assert buckets[0][-1].end == buckets[1][0].start


def test_segment_words_keeps_a_word_longer_than_the_window() -> None:
    (bucket,) = segment_words(_words([(0.0, 45.0)]), window_seconds=30.0)
    assert len(bucket) == 1


def test_segment_words_on_empty_input() -> None:
    assert segment_words([]) == []


def test_wav_fixture_is_valid_audio(wav_path: Path) -> None:
    import wave

    with wave.open(str(wav_path)) as handle:
        assert handle.getnframes() / handle.getframerate() == pytest.approx(5.0)


@pytest.mark.slow
def test_ingest_audio_end_to_end(wav_path: Path) -> None:
    """Opt-in: downloads the whisper model. `pytest -m slow` to run."""
    from linkrag.ingest.audio import ingest_audio

    units = ingest_audio(wav_path, model_size="tiny", window_seconds=30.0)
    assert all(u.modality == "audio" for u in units)


# ------------------------------------------------------------------ dispatch

def test_ingest_file_rejects_unknown_types(tmp_path: Path) -> None:
    bad = tmp_path / "notes.xyz"
    bad.write_text("nope")
    with pytest.raises(ValueError, match="unsupported file type"):
        ingest_file(bad)


# ------------------------------------------------------------- id uniqueness

def test_dedupe_ids_disambiguates_same_stem_files() -> None:
    """a/notes.pdf and b/notes.pdf mint identical ids; ids are what the LLM
    cites, so a silent collision mis-attributes evidence."""
    from linkrag.core import EvidenceUnit
    from linkrag.ingest import dedupe_ids

    units = [
        EvidenceUnit(id="notes:p1:t0", modality="text", content=c, source_file=f"{c}/notes.pdf")
        for c in "abc"
    ]
    assert [u.id for u in dedupe_ids(units)] == ["notes:p1:t0", "notes:p1:t0#2", "notes:p1:t0#3"]


def test_dedupe_ids_leaves_distinct_ids_alone() -> None:
    from linkrag.core import EvidenceUnit
    from linkrag.ingest import dedupe_ids

    units = [
        EvidenceUnit(id=f"notes:p1:t{i}", modality="text", content="x", source_file="notes.pdf")
        for i in range(3)
    ]
    assert [u.id for u in dedupe_ids(units)] == ["notes:p1:t0", "notes:p1:t1", "notes:p1:t2"]


# ------------------------------------------------- sentence-aware segmentation

def _stream(text: str, per_word: float = 1.0) -> list[Word]:
    return [Word(i * per_word, (i + 1) * per_word, w) for i, w in enumerate(text.split())]


def test_split_sentences_cuts_after_terminal_punctuation() -> None:
    got = split_sentences(_stream("One two. Three four five! Six?"))
    assert [[w.text for w in s] for s in got] == [
        ["One", "two."], ["Three", "four", "five!"], ["Six?"]
    ]


def test_split_sentences_keeps_an_unterminated_tail() -> None:
    got = split_sentences(_stream("Done. Trailing words"))
    assert [w.text for w in got[-1]] == ["Trailing", "words"]


def test_split_sentences_ignores_mid_word_periods() -> None:
    """Decimals and abbreviations must not become sentence boundaries."""
    got = split_sentences(_stream("It costs 3.50 today. Next"))
    assert len(got) == 2 and [w.text for w in got[0]][-1] == "today."


def test_split_sentences_on_empty_input() -> None:
    assert split_sentences([]) == []


def test_pack_merges_whole_sentences_up_to_the_window() -> None:
    sentences = [_stream("a b."), [Word(2, 12, "c.")], [Word(12, 40, "d.")]]
    buckets = pack(sentences, window_seconds=30.0)
    assert len(buckets) == 2
    assert [w.text for w in buckets[0]] == ["a", "b.", "c."]
    assert [w.text for w in buckets[1]] == ["d."]


def test_pack_never_splits_a_sentence() -> None:
    """The property the whole fix exists for: a unit may only end where a
    sentence ends. Fixed windows cut 70% of pilot boundaries mid-sentence;
    packing whole whisper segments cut 74%; this cuts 0%."""
    import re

    sentences = split_sentences(_stream(" ".join(f"w{i} w{i}b." for i in range(60))))
    for bucket in pack(sentences, window_seconds=30.0):
        assert re.search(r"[.!?]$", bucket[-1].text)


def test_pack_isolates_a_sentence_longer_than_the_window() -> None:
    buckets = pack([[Word(0, 45, "long.")]], window_seconds=30.0)
    assert len(buckets) == 1


def test_pack_on_empty_input() -> None:
    assert pack([]) == [] and pack([[]]) == []


def test_ingest_audio_rejects_an_unknown_segmentation(wav_path: Path) -> None:
    from linkrag.ingest.audio import ingest_audio

    with pytest.raises(ValueError, match="unknown segmentation"):
        ingest_audio(wav_path, segmentation="magic")


# ------------------------------------------------------ frozen transcripts

def test_ingest_audio_from_a_frozen_transcript_never_calls_whisper(
    wav_path: Path, tmp_path: Path, monkeypatch
) -> None:
    """Regression: the frozen path once raised UnboundLocalError on a debug counter
    bound only in the whisper branch, and ingest_files swallowed it -- producing an
    index silently missing an entire modality."""
    import json

    from linkrag.ingest import audio as audio_mod

    def explode(*a, **k):
        raise AssertionError("whisper must not run when a transcript is frozen")

    monkeypatch.setattr(audio_mod, "transcribe_segments", explode)

    frozen = tmp_path / "tone.frozen.json"
    frozen.write_text(json.dumps({"words": [
        [0.0, 0.5, "Attention"], [0.5, 1.0, "is"], [1.0, 1.6, "weighting."],
        [1.6, 2.2, "The"], [2.2, 2.9, "second"], [2.9, 3.5, "point."],
    ]}))

    units = audio_mod.ingest_audio(wav_path, transcript=frozen, window_seconds=30.0)
    assert len(units) == 1
    assert units[0].content == "Attention is weighting. The second point."
    assert units[0].location.start_s == 0.0 and units[0].location.end_s == 3.5
    assert len(units[0].metadata["words"]) == 6


def test_frozen_transcript_lookup(tmp_path: Path) -> None:
    from linkrag.ingest.audio import frozen_transcript_for

    assert frozen_transcript_for("a/lecture.mp3", tmp_path) is None
    (tmp_path / "lecture.frozen.json").write_text("{}")
    assert frozen_transcript_for("a/lecture.mp3", tmp_path) == tmp_path / "lecture.frozen.json"
    assert frozen_transcript_for("a/lecture.mp3", None) is None


def test_ingest_files_records_failures(tmp_path: Path) -> None:
    """A skipped file must be recoverable by the caller, not just logged."""
    from linkrag.ingest import ingest_files

    bad = tmp_path / "broken.pdf"
    bad.write_text("not a pdf")
    assert ingest_files([bad]) == []
    assert len(getattr(ingest_files, "last_failures")) == 1
