"""compare_retrieval end to end with a mocked model: the ledger row and --out are written.

Regression for the review finding: `n, per_run, llm = got` rebound the model config to a
float, so every run that called a model crashed at `llm.get("model")` -- before the ledger
row and the report were written, i.e. it could spend without a record.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from linkrag.core import EvidenceUnit, Location
from linkrag.index import build_index

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import compare_retrieval  # noqa: E402

TEXTS = {"t1": "attention weights on the subject token", "t2": "optimizer schedule with warmup",
         "a1": "the speaker explains attention weights here"}


def test_model_run_writes_ledger_and_report(tmp_path, monkeypatch, stub_encoder) -> None:
    units = [EvidenceUnit(id=uid, modality="audio" if uid.startswith("a") else "text", content=text,
                          source_file="talk.mp3" if uid.startswith("a") else "deck.pdf",
                          location=Location(start_s=0.0, end_s=30.0) if uid.startswith("a") else Location(page=1))
             for uid, text in TEXTS.items()]
    question = "what are the attention weights"
    encode = stub_encoder(list(TEXTS.values()) + [question, "attention weights follow up"])
    build_index(units, encoder=encode, embedding_model="stub").save(tmp_path / "index")
    (tmp_path / "links.jsonl").write_text("")
    (tmp_path / "q.jsonl").write_text(json.dumps(
        {"qid": "Q1", "type": "single_modality", "question": question, "gold_unit_ids": ["t1"]}) + "\n")

    def complete(system, user):                  # the iterative mode's follow-up query
        complete.usage["calls"] += 1
        complete.usage["prompt_tokens"] += 10
        complete.usage["completion_tokens"] += 5
        return "attention weights follow up"
    complete.usage = {"calls": 0, "cached_calls": 0, "prompt_tokens": 0, "completion_tokens": 0}

    monkeypatch.setattr(compare_retrieval, "default_encoder", lambda *a, **k: encode)
    monkeypatch.setattr(compare_retrieval, "http_completer", lambda llm: complete)
    monkeypatch.chdir(tmp_path)                  # the ledger path is relative: reports/llm_ledger.jsonl
    out = tmp_path / "out.md"
    rc = compare_retrieval.main([
        "--questions", str(tmp_path / "q.jsonl"), "--config", str(ROOT / "configs/default.yaml"),
        "--max-cost", "0", "--index", str(tmp_path / "index"), "--links", str(tmp_path / "links.jsonl"),
        "--repeats", "3", "--k", "2", "--out", str(out)])

    assert rc == 0
    assert complete.usage["calls"] > 0          # a model was really called
    rows = [json.loads(l) for l in (tmp_path / "reports/llm_ledger.jsonl").read_text().splitlines()]
    assert [(r["script"], r["model"], r["calls"]) for r in rows] == [
        ("scripts/compare_retrieval.py", "gpt-5.4-mini-2026-03-17", complete.usage["calls"])]
    report = out.read_text()
    assert "| iterative | complementarity |" in report and "LLM cost (this run):" in report


def test_only_link_modes_hand_the_graph_to_the_reranker() -> None:
    # baseline / iterative + complementarity must be a no-links control (2026-11-17 fix)
    g = object()
    assert [compare_retrieval.rerank_graph(m, g) is g for m in compare_retrieval.MODES] == [
        False, False, True, True]


def _setup(tmp_path, stub_encoder, rows):
    units = [EvidenceUnit(id=uid, modality="audio" if uid.startswith("a") else "text", content=text,
                          source_file="talk.mp3" if uid.startswith("a") else "deck.pdf",
                          location=Location(start_s=0.0, end_s=30.0) if uid.startswith("a") else Location(page=1))
             for uid, text in TEXTS.items()]
    encode = stub_encoder(list(TEXTS.values()) + [r["question"] for r in rows])
    build_index(units, encoder=encode, embedding_model="stub").save(tmp_path / "index")
    (tmp_path / "links.jsonl").write_text("")
    (tmp_path / "q.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return encode


ROWS = [{"qid": "Q1", "topic": "t1", "kind": "bridge", "type": "cross_modal_split",
         "question": "what are the attention weights", "expected_answer": "attention weights",
         "gold_unit_ids": ["t1"]},
        {"qid": "Q2", "topic": "t2", "kind": "control", "type": "slides_only",
         "question": "optimizer schedule", "expected_answer": "warmup", "gold_unit_ids": ["t2"]}]


def test_correctness_is_graded_and_reported_by_kind(tmp_path, monkeypatch, stub_encoder) -> None:
    encode = _setup(tmp_path, stub_encoder, ROWS)

    def model(system, user):
        if "Candidate answer:" in user:                    # the judge's grade prompt
            ok = "attention" in user.split("Candidate answer:")[1]
            return json.dumps({"facts": [], "verdict": "PASS" if ok else "FAIL", "reason": "r"})
        return json.dumps({"answer": "the attention weights", "claims": []})
    model.usage = {"calls": 0, "cached_calls": 0, "prompt_tokens": 0, "completion_tokens": 0}

    monkeypatch.setattr(compare_retrieval, "default_encoder", lambda *a, **k: encode)
    monkeypatch.setattr(compare_retrieval, "http_completer", lambda llm: model)
    monkeypatch.setattr(compare_retrieval, "judge_completer", lambda cfg, backend: model)
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "out.md"
    rc = compare_retrieval.main([
        "--questions", str(tmp_path / "q.jsonl"), "--config", str(ROOT / "configs/default.yaml"),
        "--max-cost", "0", "--index", str(tmp_path / "index"), "--links", str(tmp_path / "links.jsonl"),
        "--k", "2", "--repeats", "3", "--correctness", "--grade-runs", "1", "--rerank", "none",
        "--out", str(out)])
    assert rc == 0
    report = out.read_text()
    assert "### Answer correctness" in report and "| kind: bridge | 1 | baseline | none |" in report
    assert (tmp_path / "data/processed/verify_cache").exists()      # replies went through the cache


def test_topics_filter_and_pending_run_writes_no_report(tmp_path, monkeypatch, stub_encoder) -> None:
    from linkrag.agent_queue import queue_completer
    encode = _setup(tmp_path, stub_encoder, ROWS)
    monkeypatch.setenv("LINKRAG_AGENT_QUEUE", str(tmp_path / "queue.jsonl"))
    monkeypatch.setattr(compare_retrieval, "default_encoder", lambda *a, **k: encode)
    monkeypatch.setattr(compare_retrieval, "http_completer", lambda llm: queue_completer({"model": "m"}))
    monkeypatch.setattr(compare_retrieval, "judge_completer", lambda cfg, backend: queue_completer({"model": "j"}))
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "out.md"
    rc = compare_retrieval.main([
        "--questions", str(tmp_path / "q.jsonl"), "--config", str(ROOT / "configs/default.yaml"),
        "--max-cost", "0", "--index", str(tmp_path / "index"), "--links", str(tmp_path / "links.jsonl"),
        "--k", "2", "--repeats", "3", "--correctness", "--rerank", "none", "--topics", "t1",
        "--out", str(out)])
    assert rc == 3 and not out.exists()
    queued = [json.loads(l) for l in (tmp_path / "queue.jsonl").read_text().splitlines()]
    assert queued and all("Question: optimizer schedule" not in q["user"] for q in queued)   # t2 filtered out
