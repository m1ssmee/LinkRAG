"""An open-ended run that reaches its budget stops and reports what ran.

Real here: `open_modes`, `open_report`, `answer_json`, `cached_completer`, `http_completer`,
`SpendGuard`, `record_run`. Stubbed: the models, the corpus, and the HTTP call.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

from linkrag.core import EvidenceUnit, Location, load_config, set_max_cost

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "adapters"))
from lectqa import open_ended as oe  # noqa: E402

UNIT = EvidenceUnit(id="u0", modality="audio", content="the answer text", source_file="v.m4a",
                    location=Location(start_s=0.0, end_s=1.0))
REPLY = json.dumps({"answer": "the answer text", "claims": [{"claim": "the answer text", "unit_ids": ["u0"]}]})
QUESTIONS, REPEATS = 30, 2
TASKS = QUESTIONS * REPEATS * len(oe.OPEN_MODES)
CAP = 0.15          # above the estimate, so the run starts; below the real cost, so the guard stops it


def test_a_run_stopped_by_the_spend_guard_reports_what_ran(tmp_path, monkeypatch):
    cfg = load_config(ROOT / "configs" / "default.yaml")
    set_max_cost(cfg, CAP)
    monkeypatch.setenv("OPENAI_API_KEY", "not-a-key")
    monkeypatch.chdir(tmp_path)                     # the ledger path is relative
    monkeypatch.setattr(oe, "PROCESSED", tmp_path)
    monkeypatch.setattr(oe, "EVIDENCE", tmp_path / "evidence.jsonl")

    levels = [oe.LEVELS[i % len(oe.LEVELS)] for i in range(QUESTIONS)]
    items = [{"vid": "video_1", "qi": i, "level": lv, "question": f"q{i}", "ref": "the answer text", "q": {}}
             for i, lv in enumerate(levels)]
    monkeypatch.setattr(oe, "open_items", lambda ids: (items, set(range(QUESTIONS))))
    monkeypatch.setattr(oe, "default_encoder", lambda *a: lambda texts: None)
    monkeypatch.setattr(oe, "t1_encoder", lambda device: None)
    monkeypatch.setattr(oe, "t1_chunks", lambda vid, index, enc: ([], None))
    monkeypatch.setattr(oe, "t1_retrieve", lambda *a: [UNIT])
    monkeypatch.setattr(oe, "gold_interval", lambda vid, q: ((0.0, 1.0), "ok"))
    monkeypatch.setattr(oe, "Index", SimpleNamespace(load=lambda path: SimpleNamespace(units=[UNIT])))
    monkeypatch.setattr("sentence_transformers.CrossEncoder", lambda *a, **k: None)
    monkeypatch.setattr("tiktoken.get_encoding", lambda name: SimpleNamespace(encode=str.split))
    monkeypatch.setattr("linkrag.retrieve.iterative.retrieve_iterative",
                        lambda *a, **k: SimpleNamespace(units=[SimpleNamespace(unit=UNIT)]))
    monkeypatch.setattr("linkrag.eval.lectqa_metrics.similarity", lambda preds, refs, device="cpu": [1.0] * len(preds))

    sent = []

    def post(url, json=None, headers=None, timeout=None):
        sent.append(url)
        body = {"choices": [{"message": {"content": REPLY}}], "usage": {"prompt_tokens": 1000, "completion_tokens": 400}}
        return SimpleNamespace(status_code=200, text="", headers={}, json=lambda: body)

    monkeypatch.setattr("linkrag.generate.answer.requests.post", post)

    out = tmp_path / "open.md"
    stored = {path: f"complete {path.name}" for path in (out, out.with_suffix(".jsonl"), oe.EVIDENCE)}
    for path, text in stored.items():
        path.write_text(text)

    assert oe.open_modes(["video_1"], cfg, {}, out, max_cost=CAP, dry_run=False, repeats=REPEATS) == 0

    partial = tmp_path / "open_partial.md"
    rows = [json.loads(line) for line in partial.with_suffix(".jsonl").read_text().splitlines()]
    assert 0 < len(rows) < TASKS
    assert len(rows) == len(sent), "retrieve_iterative is stubbed here, so one call per answer"
    assert {r["mode"] for r in rows} == set(oe.OPEN_MODES)
    report = partial.read_text()
    assert f"{len(rows)} of {TASKS} answers" in report
    assert report.index("Not reportable") < report.index("## Subset"), "the caveat comes before the numbers"
    assert "**Scope.**" not in report and "LLM cost (this run):" in report
    assert {path: path.read_text() for path in stored} == stored, "a stopped run replaces nothing complete"
