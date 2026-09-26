"""Lectern's HTTP API (linkrag.ui.api) over a three-file lecture built through the real
endpoints: upload -> streamed build -> pairs / stats -> ask. Whisper is replaced by a scripted
transcript and bge-m3 by the bag-of-words stub; the answerer and judge are the offline mocks,
so nothing here can bill."""

from __future__ import annotations

import json
from pathlib import Path

import pymupdf
import pytest
from fastapi.testclient import TestClient

from linkrag.core import load_config
from linkrag.generate.verify import ABSTENTION
from linkrag.index import build_index
from linkrag.ingest.audio import Segment, Word
from linkrag.link.align import save_links
from linkrag.manifest import build_manifest, write_manifest
from linkrag.ui import api

from conftest import BODY, CAPTION, CROSS_REF, TABLE_REF

ROOT = Path(__file__).resolve().parents[1]
SLIDES = ["Focus cheap ingest time indexing",          # terse, as slides are: a question about
          "Query time clustering",                      # what was *said* seeds on the speech and
          "Results 57 times cheaper"]                   # reaches the slide through its link
SPEECH = ["Focus indexes video with cheap classifiers at ingest time.",
          "At query time clustering cuts the expensive detector calls.",
          "So Focus is 57 times cheaper than the ingest only baseline."]
SID = "test-session-0001"
QUESTION = "How does Focus compare with the ingest only baseline?"


def _cfg() -> dict:
    cfg = load_config(ROOT / "configs" / "default.yaml")
    cfg["ingest"].update(audio_segment_seconds=1.6, ocr_figures=False)
    cfg["retrieve"]["linkrag"]["k_final"] = 3       # well below the corpus, so links can add units
    cfg["retrieve"]["rerank"]["pool"] = 3
    return cfg


def _transcript(*_args, **_kwargs) -> list[Segment]:
    """Whisper's output shape: 1 s sentences with 0.8 s pauses, one audio unit each."""
    segments, t = [], 0.0
    for sentence in SPEECH:
        words = []
        for w in sentence.split():
            words.append(Word(round(t, 2), round(t + 0.1, 2), w))
            t += 0.1
        segments.append(Segment(words[0].start, words[-1].end, sentence, tuple(words)))
        t += 0.8
    return segments


def _deck(path: Path) -> Path:
    doc = pymupdf.open()
    for text in SLIDES:
        doc.new_page(width=720, height=405).insert_text((40, 80), text, fontsize=20)   # landscape: a deck
    doc.save(path)
    return path


def _stream(response) -> list[dict]:
    return [json.loads(line) for line in response.iter_lines() if line]


@pytest.fixture
def lecture(tmp_path, monkeypatch, pdf_path, wav_path, stub_encoder):
    """A built lecture: deck.pdf (3 slides), notes.pdf (conftest's 2-page note with a figure)
    and talk.wav narrating the deck."""
    encode = stub_encoder([*SLIDES, *SPEECH, BODY, CAPTION, CROSS_REF, TABLE_REF,
                           "Attention weights concentrate on the subject token."])
    monkeypatch.setattr("linkrag.ingest.audio.transcribe_segments", _transcript)
    app = api.create_app(_cfg(), workdir=tmp_path, answerer="mock", encoder=encode)
    client = TestClient(app, headers={"X-Session": SID})
    files = [("files", ("deck.pdf", _deck(tmp_path / "deck.pdf").read_bytes(), "application/pdf")),
             ("files", ("notes.pdf", pdf_path.read_bytes(), "application/pdf")),
             ("files", ("talk.wav", wav_path.read_bytes(), "audio/wav"))]
    assert client.post("/upload", files=files).json()["pending"] == ["deck.pdf", "notes.pdf", "talk.wav"]
    with client.stream("POST", "/build") as r:
        events = _stream(r)
    return client, events, tmp_path


def test_build_streams_every_stage_and_ends_ready(lecture):
    client, events, _ = lecture
    done = [e["stage"] for e in events if e.get("status") == "done"]
    assert done == ["ingest", "index", "link", "pairs"]
    assert events[-1] == {"status": "ready"}
    assert any(e.get("log", "").startswith("ingest.audio") for e in events)   # the pipeline's own lines
    state = client.get("/state").json()
    assert [(f["name"], f["kind"]) for f in state["corpus"]["files"]] == [
        ("talk.wav", "speech"), ("deck.pdf", "slides"), ("notes.pdf", "notes")]
    assert state["pending"] == [] and state["answerer"]["mock"] and state["cost_usd"] == 0


def test_pairs_follow_the_gate_and_overrides_change_the_links(lecture):
    client, _, _ = lecture
    pairs = {(p["a"], p["b"]): p for p in client.get("/pairs").json()["pairs"]}
    speech = pairs[("deck.pdf", "talk.wav")]
    assert speech["kind"] == "audio_slide" and speech["z"] is not None and speech["setting"] == "auto"
    # auto is the pipeline's own verdict: links exist exactly when the gate passed
    assert (speech["links"] > 0) == (speech["gate"] == "related")

    def audio_slide(body):
        return body["stats"]["by_type"].get("audio_slide", 0)

    off = client.put("/pairs", json={"a": "talk.wav", "b": "deck.pdf", "setting": "unrelated"}).json()
    assert audio_slide(off) == 0 and off["stats"]["by_type"].get("deictic", 0) == 0
    assert next(p for p in off["pairs"] if p["b"] == "talk.wav")["state"] == "unlinked"
    on = client.put("/pairs", json={"a": "talk.wav", "b": "deck.pdf", "setting": "related"}).json()
    assert audio_slide(on) == len(SPEECH)            # ungated: one link per aligned segment
    stats = client.get("/stats").json()
    assert stats["total"] + stats["dropped"] == stats["links"]      # measurement rule 2
    assert sum(stats["by_type"].values()) == stats["total"]
    assert client.put("/pairs", json={"a": "talk.wav", "b": "nope.pdf", "setting": "related"}).status_code == 404


def test_ask_returns_verified_claims_citing_numbered_evidence(lecture):
    client, _, _ = lecture
    client.put("/pairs", json={"a": "talk.wav", "b": "deck.pdf", "setting": "related"})
    body = client.post("/ask", json={"question": QUESTION}).json()
    assert body["links"] > 0 and body["verification_note"] == api.MOCK_NOTE
    evidence = {e["n"]: e for e in body["evidence"]}
    assert sorted(evidence) == list(range(1, len(evidence) + 1))
    assert not body["abstained"] and body["answer_claims"]
    for claim in body["answer_claims"]:
        assert claim["verdict"] == "supported" and claim["citations"]
        assert all(n in evidence for n in claim["citations"])
    cited = evidence[body["answer_claims"][0]["citations"][0]]
    assert {"unit_id", "modality", "file", "location", "origin", "link_type", "excerpt",
            "crop_url", "clip_url"} <= set(cited)
    speech = next(e for e in body["evidence"] if e["modality"] == "speech")
    assert speech["location"]["end_s"] > speech["location"]["start_s"] and speech["clip_url"]
    slide = next(e for e in body["evidence"] if e["modality"] == "slides")
    assert slide["location"]["page"] in (1, 2, 3) and slide["crop_url"]
    # link-following: with k (3) below the corpus size, a linked unit enters through its link
    linked = [e for e in body["evidence"] if e["origin"] == "via_link"]
    assert linked and all(e["link_type"] and e["via_unit"] for e in linked)


def test_compare_numbers_both_answers_in_one_space_and_names_what_baseline_missed(lecture):
    client, _, _ = lecture
    body = client.post("/ask", json={"question": QUESTION, "compare_baseline": True}).json()
    ours, base = body["evidence"], body["baseline"]["evidence"]
    assert [e["n"] for e in ours] == list(range(1, len(ours) + 1))
    same = {e["unit_id"]: e["n"] for e in ours}
    assert all(e["n"] == same[e["unit_id"]] for e in base if e["unit_id"] in same)
    assert all(e["n"] > len(ours) for e in base if e["unit_id"] not in same)
    assert all(e["origin"] == "direct" for e in base)          # baseline follows no links
    base_ids = {e["unit_id"] for e in base}
    assert body["baseline"]["missed_evidence"] == [e["unit_id"] for e in ours if e["unit_id"] not in base_ids]


def test_abstention_when_nothing_in_the_material_answers(lecture):
    client, _, _ = lecture
    body = client.post("/ask", json={"question": "What is the recipe for sourdough bread?"}).json()
    assert body["abstained"] and body["answer_claims"] == [] and body["summary"] == ABSTENTION


def test_claim_verdicts_map_supported_weak_unsupported(lecture, monkeypatch):
    client, _, _ = lecture

    def scripted(system, user):
        head = api.EVIDENCE_HEAD.search(user)
        text = user[head.end():].strip().split("\n")[0]
        return json.dumps({"answer": "x", "claims": [
            {"claim": text, "unit_ids": [head.group(1)]},
            {"claim": "Something asserted without evidence", "unit_ids": []},
            {"claim": "The moon is made of green cheese", "unit_ids": [head.group(1)]}]})
    monkeypatch.setattr(api, "mock_answerer", scripted)
    body = client.post("/ask", json={"question": QUESTION}).json()
    assert [c["verdict"] for c in body["answer_claims"]] == ["supported", "weak", "unsupported"]
    assert body["answer_claims"][1]["citations"] == []


def test_graph_timeline_and_media(lecture):
    client, _, _ = lecture
    client.put("/pairs", json={"a": "talk.wav", "b": "deck.pdf", "setting": "related"})
    body = client.post("/ask", json={"question": QUESTION}).json()
    ids = [e["unit_id"] for e in body["evidence"]] + [e["via_unit"] for e in body["evidence"] if e["via_unit"]]
    graph = client.get("/graph", params=[("units", i) for i in ids]).json()
    assert {n["unit_id"] for n in graph["nodes"]} <= set(ids) and graph["edges"]
    assert all(e["src"] in ids and e["dst"] in ids for e in graph["edges"])

    tl = client.get("/timeline").json()
    assert tl["file"] == "talk.wav" and tl["audio_url"] and tl["peaks"] and max(tl["peaks"]) == 1.0
    starts = [i["start"] for i in tl["intervals"]]
    assert tl["intervals"] and starts == sorted(starts) and {i["page"] for i in tl["intervals"]} <= {1, 2, 3}

    slide = next(e for e in body["evidence"] if e["crop_url"])
    png = client.get(slide["crop_url"], params={"s": SID})
    assert png.status_code == 200 and png.content[:4] == b"\x89PNG"
    audio = client.get("/media/audio", headers={"Range": "bytes=0-99"})
    assert audio.status_code == 206 and len(audio.content) == 100
    speech = next(e for e in body["evidence"] if e["clip_url"])
    clip = client.get(speech["clip_url"], params={"s": SID})              # generate.citations.clip_audio
    assert clip.status_code == 200 and clip.headers["content-type"] == "audio/mp4" and len(clip.content) > 100


def test_uploads_are_checked_and_sessions_are_separate(lecture):
    client, _, work = lecture
    other = TestClient(client.app, headers={"X-Session": "another-session-02"})
    assert other.get("/state").json()["corpus"] is None
    assert other.get("/stats").status_code == 409
    assert TestClient(client.app).get("/state").status_code == 400           # no session id
    assert other.post("/upload", files=[("files", ("run.exe", b"MZ", "application/octet-stream"))]).status_code == 415
    saved = other.post("/upload", files=[("files", ("../../my notes.pdf", b"%PDF-1.4", "application/pdf"))]).json()
    assert saved["files"] == ["my_notes.pdf"]
    assert (work / "sessions" / "another-session-02" / "uploads" / "my_notes.pdf").exists()


def test_zero_cost_mode_refuses_before_anything_is_sent(tmp_path, monkeypatch, stub_encoder):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-never-sent")
    monkeypatch.setattr("requests.post", lambda *a, **k: pytest.fail("an LLM request was sent"))
    sample = _frozen_sample(tmp_path / "frozen", stub_encoder)
    app = api.create_app(_cfg(), workdir=tmp_path / "work", sample=sample, answerer="llm", max_cost=0.0,
                         ledger=None, encoder=stub_encoder(SLIDES + SPEECH))
    live = TestClient(app, headers={"X-Session": SID})
    with live.stream("POST", "/sample") as r:
        assert _stream(r)[-1] == {"status": "ready"}
    refused = live.post("/ask", json={"question": QUESTION})
    assert refused.status_code == 402 and "budget" in refused.json()["detail"]


def test_strong_model_only_with_the_demo_flag(tmp_path, stub_encoder):
    cfg = _cfg()
    cfg["models"]["llm"]["model"] = api.STRONG_MODEL
    with pytest.raises(SystemExit):
        api.create_app(cfg, workdir=tmp_path, answerer="llm", encoder=stub_encoder(["x"]))
    app = api.create_app(_cfg(), workdir=tmp_path, answerer="llm", strong=True, encoder=stub_encoder(["x"]))
    state = TestClient(app, headers={"X-Session": SID}).get("/state").json()
    assert state["answerer"]["model"] == api.STRONG_MODEL and state["answerer"]["strong"]


def _frozen_sample(root: Path, stub_encoder) -> Path:
    """A frozen corpus in the layout scripts/ingest.py + build_links.py write, sources relative
    to the project root (root/data/processed, root/data/raw) like pilot01's."""
    from linkrag.core import EvidenceUnit, Link, Location
    processed = root / "data" / "processed"
    audio = [EvidenceUnit(f"talk:a{i}", "audio", s, "data/raw/talk.wav", Location(start_s=2.0 * i, end_s=2.0 * i + 1.5))
             for i, s in enumerate(SPEECH)]
    slides = [EvidenceUnit(f"deck:p{i + 1}:t0", "text", s, "data/raw/deck.pdf", Location(page=i + 1),
                           metadata={"slide_deck": True}) for i, s in enumerate(SLIDES)]
    figure = EvidenceUnit("deck:p2:g0", "figure", "clustering diagram", "data/raw/deck.pdf",
                          Location(page=2, bbox=(40.0, 120.0, 360.0, 380.0)), metadata={"slide_deck": True})
    units = audio + slides + [figure]
    build_index(units, encoder=stub_encoder(SLIDES + SPEECH), embedding_model="BAAI/bge-m3").save(processed / "index")
    manifest = write_manifest(build_manifest([], units), processed / "manifest.json")
    links = [Link(a.id, s.id, "audio_slide", 0.8) for a, s in zip(audio, slides)]
    save_links(links, processed / "links.jsonl", manifest_hash=json.loads(manifest.read_text())["hash"])
    return processed


def test_sample_is_frozen_read_only_and_shared_per_session(tmp_path, stub_encoder):
    processed = _frozen_sample(tmp_path / "pilot", stub_encoder)
    before = sorted((p.relative_to(processed), p.stat().st_mtime) for p in processed.rglob("*"))
    app = api.create_app(_cfg(), workdir=tmp_path / "work", sample=processed, sample_title="Pilot",
                         answerer="mock", encoder=stub_encoder(SLIDES + SPEECH))
    client = TestClient(app, headers={"X-Session": SID})
    with client.stream("POST", "/sample") as r:
        assert _stream(r)[-1] == {"status": "ready"}
    state = client.get("/state").json()["corpus"]
    assert state["sample"] and state["title"] == "Pilot"
    assert client.get("/stats").json()["by_type"] == {"audio_slide": 3}
    client.put("/pairs", json={"a": "talk.wav", "b": "deck.pdf", "setting": "unrelated"})
    other = TestClient(app, headers={"X-Session": "another-session-02"})
    with other.stream("POST", "/sample") as r:
        _stream(r)
    assert other.get("/stats").json()["total"] == 3            # one session's setting is not another's
    assert client.post("/upload", files=[("files", ("x.pdf", b"%PDF", "application/pdf"))]).status_code == 409
    assert client.post("/ask", json={"question": QUESTION}).json()["answer_claims"]
    assert sorted((p.relative_to(processed), p.stat().st_mtime) for p in processed.rglob("*")) == before


def test_each_recording_deck_pair_gets_its_own_gate(tmp_path, monkeypatch, pdf_path, wav_path, stub_encoder):
    """Two recordings and one deck: the pipeline links them as one sequence, but each
    connector's z must come from that recording's own alignment, not be copied."""
    monkeypatch.setattr("linkrag.ingest.audio.transcribe_segments", _transcript)
    app = api.create_app(_cfg(), workdir=tmp_path, answerer="mock", encoder=stub_encoder(SLIDES + SPEECH))
    client = TestClient(app, headers={"X-Session": SID})
    client.post("/upload", files=[("files", ("deck.pdf", _deck(tmp_path / "deck.pdf").read_bytes(), "application/pdf")),
                                  ("files", ("notes.pdf", pdf_path.read_bytes(), "application/pdf")),
                                  ("files", ("part1.wav", wav_path.read_bytes(), "audio/wav")),
                                  ("files", ("part2.wav", wav_path.read_bytes(), "audio/wav"))])
    with client.stream("POST", "/build") as r:
        events = _stream(r)
    assert events[-1] == {"status": "ready"} and any("2 recordings x 1 decks" in e.get("warning", "") for e in events)
    pairs = {(p["a"], p["b"]): p for p in client.get("/pairs").json()["pairs"]}
    assert pairs[("deck.pdf", "part1.wav")]["kind"] == pairs[("deck.pdf", "part2.wav")]["kind"] == "audio_slide"
    assert pairs[("deck.pdf", "part1.wav")]["z"] is not None and pairs[("deck.pdf", "part2.wav")]["z"] is not None
    hindi = client.post("/ask", json={"question": QUESTION, "lang": "hi"}).json()
    assert hindi["verification_note"] == api.MOCK_NOTE          # the mock outranks the Hindi caveat
