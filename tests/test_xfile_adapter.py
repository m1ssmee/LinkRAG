"""XFile adapter: MaViLS word-level transcript parsing, the topic table, page cutting, and
that a frozen transcript it writes is read back by the real ingest path."""

from __future__ import annotations

import importlib.util
import json
import sys

import pytest

spec = importlib.util.spec_from_file_location("xfile_adapter", "scripts/adapters/xfile.py")
x = importlib.util.module_from_spec(spec)
sys.modules["xfile_adapter"] = x        # @dataclass resolves its module through sys.modules
spec.loader.exec_module(x)

SRT = """[00:00:06.160 --> 00:00:06.760] Welcome
[00:00:07.300 --> 00:00:07.640] high
[00:00:07.640 --> 00:00:07.980]-quality

[01:02:03.500 --> 01:02:04.000] lectures.
not a timestamped line
"""


def test_parse_word_srt_joins_unspaced_pieces_and_skips_junk():
    words = x.parse_word_srt(SRT)
    assert words == [
        [6.16, 6.76, "Welcome"],
        [7.3, 7.98, "high-quality"],            # tail joined, end time extended
        [3723.5, 3724.0, "lectures."],          # hours parsed
    ]


def test_unspaced_piece_at_start_is_kept_as_a_word():
    assert x.parse_word_srt("[00:00:01.000 --> 00:00:01.500]-dash\n") == [[1.0, 1.5, "-dash"]]


def test_load_topics_reads_pages_and_optional_notes(tmp_path):
    f = tmp_path / "topics.yaml"
    f.write_text("topics:\n"
                 "  a: {srt: a.srt, slides: a.pdf, notes: {file: n.pdf, pages: [3, 5]}}\n"
                 "  b: {srt: b.srt, slides: b.pdf}\n")
    a, b = x.load_topics(f)
    assert (a.name, a.notes, a.notes_pages) == ("a", "n.pdf", (3, 5))
    assert b.notes is None and b.notes_pages is None
    assert a.talk().name == "a_talk.wav" and a.deck().name == "a_slides.pdf"


def test_repo_topic_table_has_unique_basenames():
    # gold matching is by basename across the pooled corpus
    topics = x.load_topics()
    names = [p.name for t in topics for p in (t.talk(), t.deck(), t.notes_out())]
    assert len(names) == len(set(names)) and len(topics) >= 7


def test_cut_pages_range_and_bounds(tmp_path):
    fitz = pytest.importorskip("fitz")
    src = tmp_path / "src.pdf"
    with fitz.open() as doc:
        for i in range(5):
            doc.new_page().insert_text((72, 72), f"page {i + 1}")
        doc.save(src)
    dst = tmp_path / "out.pdf"
    assert x.cut_pages(src, dst, (2, 4)) == 3
    with fitz.open(dst) as out:
        assert [p.get_text().strip() for p in out] == ["page 2", "page 3", "page 4"]
    with pytest.raises(ValueError):
        x.cut_pages(src, dst, (4, 9))


def test_frozen_transcript_is_read_by_ingest(tmp_path, monkeypatch):
    from linkrag.ingest.audio import frozen_transcript_for, load_frozen_transcript

    (tmp_path / "audioscripts").mkdir()
    (tmp_path / "audioscripts" / "t.srt").write_text(SRT)
    topic = x.Topic(name="demo", srt="t.srt", slides="unused.pdf")
    out = x.freeze(topic, frozen_dir=tmp_path / "frozen", mavils=tmp_path)
    assert json.loads(out.read_text())["asr_backend"] == "mavils_shipped"
    # ingest resolves the transcript by the stub's stem and never opens the audio
    assert frozen_transcript_for(topic.talk(), tmp_path / "frozen") == out
    words = load_frozen_transcript(out)
    assert [w.text for w in words] == ["Welcome", "high-quality", "lectures."]


def test_apply_review_needs_every_decision_and_applies_edits(tmp_path):
    import csv
    proposed = tmp_path / "p.jsonl"
    proposed.write_text("\n".join(json.dumps(r) for r in [
        {"_meta": {"corpus": "xfile"}},
        {"qid": "A1", "question": "q1", "expected_answer": "a1", "gold_units": []},
        {"qid": "A2", "question": "q2", "expected_answer": "a2", "gold_units": []},
        {"qid": "A3", "question": "q3", "expected_answer": "a3", "gold_units": []}]) + "\n")
    cols = ["qid", "kind", "reviewer", x.DECISION, "edited_question", "edited_answer"]

    def sheet(decisions):
        f = tmp_path / "r.csv"
        with f.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            for qid, (d, eq, ea) in decisions.items():
                w.writerow({"qid": qid, "kind": "bridge", "reviewer": "R", x.DECISION: d,
                            "edited_question": eq, "edited_answer": ea})
        return f

    with pytest.raises(SystemExit):
        x.apply_review(proposed, sheet({"A1": ("accept", "", ""), "A2": ("", "", ""), "A3": ("reject", "", "")}),
                       tmp_path / "o.jsonl")
    counts = x.apply_review(proposed, sheet({"A1": ("accept", "", ""), "A2": ("edit", "", "better a2"),
                                             "A3": ("Reject", "", "")}), tmp_path / "o.jsonl")
    assert counts == {"accept": 1, "edit": 1, "reject": 1}
    rows = [json.loads(l) for l in (tmp_path / "o.jsonl").read_text().splitlines()]
    assert rows[0]["_meta"]["status"] == "human-reviewed"
    assert [(r["qid"], r["question"], r["expected_answer"], r["kind"]) for r in rows[1:]] == [
        ("A1", "q1", "a1", "bridge"), ("A2", "q2", "better a2", "bridge")]


def test_paired_bootstrap_resamples_topics_and_pairs_by_question(tmp_path):
    qs = [{"qid": f"{t}{i}", "topic": t, "type": "cross_modal_split"} for t in "ABC" for i in range(4)]
    a, b = "linkrag|complementarity", "baseline|complementarity"
    correct = {}
    for q in qs:            # a wins every question on topic A only, by one run in three
        correct[f"{q['qid']}|{a}"] = [True, True, q["topic"] == "A"]
        correct[f"{q['qid']}|{b}"] = [True, True, False]
    p = tmp_path / "r.questions.json"
    p.write_text(json.dumps({"questions": qs, "correct": correct}))
    r = x.paired(p, a, b, types={"cross_modal_split"})
    assert r["n"] == 12 and r["topics"] == 3
    assert abs(r["diff"] - (4 / 3) / 12) < 1e-9           # 4 questions x 1/3 run, over 12
    assert r["lo"] == 0 and r["verdict"] == "no measurable difference"   # resamples without A give 0


def test_paired_interval_touching_zero_is_not_a_difference(tmp_path):
    # thirds summed in floating point give -1e-17, not 0; that must not read as "b higher"
    qs = [{"qid": f"{t}{i}", "topic": t, "type": "cross_modal_split"} for t in "AB" for i in range(3)]
    a, b = "linkrag|complementarity", "baseline|complementarity"
    correct = {f"{q['qid']}|{a}": [False, False, q["topic"] == "B" or q["qid"] == "A0"] for q in qs}
    correct |= {f"{q['qid']}|{b}": [False, False, True] for q in qs}
    p = tmp_path / "r.questions.json"
    p.write_text(json.dumps({"questions": qs, "correct": correct}))
    r = x.paired(p, a, b, types={"cross_modal_split"})
    assert r["hi"] == 0 and r["verdict"] == "no measurable difference"
