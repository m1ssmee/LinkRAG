"""Lectern's HTTP API (linkrag.ui.api) over a three-file lecture built through the real
endpoints: upload -> streamed build -> pairs / stats -> ask. Whisper is replaced by a scripted
transcript and bge-m3 by the bag-of-words stub; the answerer and judge are the offline mocks,
so nothing here can bill."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pymupdf
import pytest
from fastapi.testclient import TestClient

from linkrag.core import load_config
from linkrag.generate.verify import ABSTENTION
from linkrag.index import build_index
from linkrag.ingest.audio import Segment, Word
from linkrag.link.align import save_links
from linkrag.manifest import build_manifest, manifest_hash, write_manifest
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


def test_compare_numbers_both_answers_in_one_space_and_lists_what_baseline_did_not_retrieve(lecture):
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
    paper = EvidenceUnit("paper:p1:t0", "text", "The paper evaluates the system on fourteen streams.",
                         "data/raw/paper.pdf", Location(page=1))          # the notes: pilot01's paper
    units = audio + slides + [figure, paper]
    build_index(units, encoder=stub_encoder(SLIDES + SPEECH), embedding_model="BAAI/bge-m3").save(processed / "index")
    manifest = write_manifest(build_manifest([], units), processed / "manifest.json")
    links = [Link(a.id, s.id, "audio_slide", 0.8) for a, s in zip(audio, slides)]
    # the relatedness judge failed talk:a1 -> deck:p2:t0; load_links(gated=True) drops it
    links[1].metadata = {"relatedness": {"passed": False, "votes": ["no", "no", "yes"], "subject": ""}}
    save_links(links, processed / "links.jsonl", manifest_hash=json.loads(manifest.read_text())["hash"])
    return processed


def test_sample_loads_recording_and_deck_by_default_and_the_paper_on_request(tmp_path, stub_encoder, monkeypatch):
    processed = _frozen_sample(tmp_path / "pilot", stub_encoder)
    before = sorted((p.relative_to(processed), p.stat().st_mtime) for p in processed.rglob("*"))
    make = lambda: api.create_app(_cfg(), workdir=tmp_path / "work", sample=processed, sample_title="Pilot",
                                  answerer="mock", encoder=stub_encoder(SLIDES + SPEECH))
    client = TestClient(make(), headers={"X-Session": SID})
    assert client.get("/state").json()["sample_notes"] == ["paper.pdf"]

    # default: recording + deck, linked again by the pipeline for those two files, gates ready at once
    with client.stream("POST", "/sample") as r:
        assert _stream(r)[-1] == {"status": "ready"}
    state = client.get("/state").json()["corpus"]
    assert state["sample"] and state["title"] == "Pilot"
    assert [f["name"] for f in state["files"]] == ["talk.wav", "deck.pdf"]
    pairs = client.get("/pairs").json()
    assert not pairs["pending"] and pairs["pairs"][0]["kind"] == "audio_slide" and pairs["pairs"][0]["z"] is not None
    folder, = (tmp_path / "work").glob("sample_lecture_*")
    remade = [json.loads(l) for l in (folder / "links.jsonl").read_text().splitlines()[1:]]
    failed = [r for r in remade if (r["src_id"], r["dst_id"]) == ("talk:a1", "deck:p2:t0")]
    assert failed and failed[0]["metadata"]["relatedness"]["passed"] is False    # the stored verdict carried over
    edges = client.get("/graph", params=[("units", "talk:a1"), ("units", "deck:p2:t0")]).json()["edges"]
    assert not [e for e in edges if e["link_type"] == "audio_slide"]            # and was applied

    # with the paper: the frozen corpus as stored (3 audio_slide links, 1 gated out); gates in the background
    other = TestClient(client.app, headers={"X-Session": "another-session-02"})
    with other.stream("POST", "/sample?paper=true") as r:
        assert _stream(r)[-1] == {"status": "ready"}
    assert [f["name"] for f in other.get("/state").json()["corpus"]["files"]] == ["talk.wav", "deck.pdf", "paper.pdf"]
    assert other.get("/stats").json()["by_type"] == {"audio_slide": 2}
    for _ in range(100):
        full = other.get("/pairs").json()
        if not full["pending"]:
            break
        time.sleep(0.05)
    assert not full["pending"] and list((tmp_path / "work").glob("sample_gates_*.json"))

    # one session's setting is not another's; the sample is read-only; answers work
    total = client.get("/stats").json()["total"]
    other.put("/pairs", json={"a": "talk.wav", "b": "deck.pdf", "setting": "unrelated"})
    assert client.get("/stats").json()["total"] == total
    assert client.post("/upload", files=[("files", ("x.pdf", b"%PDF", "application/pdf"))]).status_code == 409
    assert client.post("/ask", json={"question": QUESTION}).json()["answer_claims"]

    # a restart reads the rebuilt sample from its cache: nothing is linked again
    monkeypatch.setattr(api, "link_corpus", lambda *a, **k: pytest.fail("the cached sample was linked again"))
    again = TestClient(make(), headers={"X-Session": SID})
    with again.stream("POST", "/sample") as r:
        assert _stream(r)[-1] == {"status": "ready"}
    assert again.get("/stats").json()["total"] == total
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


def test_a_linked_pair_below_the_unsure_band_asks_for_confirmation(lecture, stub_encoder, tmp_path, monkeypatch,
                                                                   pdf_path, wav_path):
    """The lecture's audio x deck pair passes its gate (z above 1.27); with the unsure band
    raised above its z (config ui.unsure_below_z) it shows as unsure until confirmed."""
    client, _, _ = lecture
    z = next(p for p in client.get("/pairs").json()["pairs"] if p["kind"] == "audio_slide")["z"]
    cfg = _cfg()
    cfg["ui"] = {"unsure_below_z": z + 1.0}
    monkeypatch.setattr("linkrag.ingest.audio.transcribe_segments", _transcript)
    app = api.create_app(cfg, workdir=tmp_path / "band", answerer="mock",
                         encoder=stub_encoder([*SLIDES, *SPEECH, BODY, CAPTION, CROSS_REF, TABLE_REF]))
    other = TestClient(app, headers={"X-Session": SID})
    other.post("/upload", files=[("files", ("deck.pdf", _deck(tmp_path / "deck.pdf").read_bytes(), "application/pdf")),
                                 ("files", ("notes.pdf", pdf_path.read_bytes(), "application/pdf")),
                                 ("files", ("talk.wav", wav_path.read_bytes(), "audio/wav"))])
    with other.stream("POST", "/build") as r:
        _stream(r)
    speech = next(p for p in other.get("/pairs").json()["pairs"] if p["kind"] == "audio_slide")
    assert speech["gate"] == "related" and speech["state"] == "unsure"
    confirmed = other.put("/pairs", json={"a": speech["a"], "b": speech["b"], "setting": "related"}).json()["pairs"]
    assert next(p for p in confirmed if p["kind"] == "audio_slide")["state"] == "linked"


def test_linked_context_takes_neighbours_in_turn_and_stops_at_the_cap():
    """u0 and u1 are cited. u0 has four uncited neighbours, u1 two (one along an incoming link);
    the cited u1 is never listed as context for u0. Rows alternate between the cited units,
    strongest link first, and stop at LINKED_ROWS."""
    from linkrag.core import EvidenceUnit, Link, Location
    from linkrag.link.graph import build_graph
    units = [EvidenceUnit(f"u{i}", "text", "x", "deck.pdf", Location(page=i + 1), metadata={"slide_deck": True})
             for i in range(8)]
    links = [Link("u0", "u1", "same_slide", 1.0), Link("u1", "u6", "figure_text", 0.9), Link("u7", "u1", "deictic", 0.5),
             *(Link("u0", f"u{i}", "figure_text", 1.0 - i / 10) for i in (2, 3, 4, 5))]
    rows = api.linked_context(build_graph(units, links), {u.id: u for u in units},
                              [{"citations": [1]}, {"citations": [2]}], {"u0": 1, "u1": 2})
    assert [(r["unit_id"], r["from_n"], r["link_type"]) for r in rows] == [
        ("u2", 1, "figure_text"), ("u6", 2, "figure_text"), ("u3", 1, "figure_text"), ("u7", 2, "deictic")]
    assert len(rows) == api.LINKED_ROWS and {"modality", "file", "location"} <= set(rows[0])


def test_ask_returns_linked_context_beside_the_answer_not_in_it(lecture):
    client, _, _ = lecture
    client.put("/pairs", json={"a": "talk.wav", "b": "deck.pdf", "setting": "related"})
    body = client.post("/ask", json={"question": QUESTION, "compare_baseline": True}).json()
    cited = {n for c in body["answer_claims"] for n in c["citations"]}
    id_of = {e["n"]: e["unit_id"] for e in body["evidence"]}
    rows = body["linked_context"]
    assert 0 < len(rows) <= api.LINKED_ROWS
    for row in rows:
        assert row["unit_id"] not in {id_of[n] for n in cited} and row["from_n"] in cited
        edges = client.get("/graph", params=[("units", row["unit_id"]), ("units", id_of[row["from_n"]])]).json()["edges"]
        assert row["link_type"] in {e["link_type"] for e in edges}               # a real 1-hop link
    assert body["baseline"]["linked_context"] == []                                # the baseline follows no links
    assert client.post("/ask", json={"question": "What is the recipe for sourdough bread?"}).json()["linked_context"] == []


def test_the_rebuilt_sample_is_cached_per_parent_corpus(tmp_path, stub_encoder):
    """Two frozen corpora over the same files that differ only in `derived` (their frozen
    transcript) must not share the rebuilt sample: the cut would otherwise lose what tells them
    apart (manifest.build_manifest)."""
    folders = []
    for name, transcript in (("a", "one.frozen.json"), ("b", "two.frozen.json")):
        processed = _frozen_sample(tmp_path / name, stub_encoder)
        manifest = json.loads((processed / "manifest.json").read_text())
        manifest["derived"] = [{"name": transcript, "sha256": transcript, "role": "frozen_transcript"}]
        manifest.pop("hash")
        manifest["hash"] = manifest_hash(manifest)
        (processed / "manifest.json").write_text(json.dumps(manifest))
        links = (processed / "links.jsonl").read_text().splitlines()
        (processed / "links.jsonl").write_text("\n".join([json.dumps({"_meta": {"manifest_hash": manifest["hash"]}}), *links[1:]]) + "\n")
        app = api.create_app(_cfg(), workdir=tmp_path / "work", sample=processed, answerer="mock",
                             encoder=stub_encoder(SLIDES + SPEECH))
        with TestClient(app, headers={"X-Session": SID}).stream("POST", "/sample") as r:
            assert _stream(r)[-1] == {"status": "ready"}
        folders = sorted(p.name for p in (tmp_path / "work").glob("sample_lecture_*"))
    assert len(folders) == 2
