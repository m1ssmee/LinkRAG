"""scripts/eval/judge_agreement.py with a mocked judge run: the report, the cache-only
answerer, the pinned llm backend, and the billing refusal."""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts" / "eval"


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_judge_agreement_writes_a_kappa_report_from_a_mocked_judge(tmp_path, monkeypatch):
    monkeypatch.delenv("LINKRAG_JUDGE_BACKEND", raising=False)
    ja = load("judge_agreement")
    seen = {}
    stored = json.loads(Path("reports/gold_verified_pilot01.json").read_text())

    def fake_verify(argv):                                 # stands in for the candidate judge's run
        import os
        seen["argv"], seen["judge_env"] = argv, os.environ.get("LINKRAG_JUDGE_BACKEND")
        seen["llm_env"] = os.environ.get("LINKRAG_LLM_BACKEND")
        out = Path(argv[argv.index("--reports-dir") + 1])
        flipped, n = json.loads(json.dumps(stored)), 0
        for q in flipped:                                  # a judge that disagrees on 10 unit verdicts
            for u in q["units"]:
                if n < 10:
                    u["kept"], n = not u["kept"], n + 1
        (out / "gold_verified_pilot01.json").write_text(json.dumps(flipped))
        return 0

    monkeypatch.setitem(sys.modules, "verify_gold", types.SimpleNamespace(main=fake_verify))
    monkeypatch.setenv("LINKRAG_LLM_BACKEND", "colab")   # must be dropped: the answerer is the stored one
    assert ja.main(["--judge", "groq", "--max-cost", "0", "--out-dir", str(tmp_path)]) == 0
    assert seen["judge_env"] == "groq" and seen["llm_env"] is None
    import os                                              # and the caller's environment is back
    assert "LINKRAG_JUDGE_BACKEND" not in os.environ and os.environ["LINKRAG_LLM_BACKEND"] == "colab"
    assert seen["argv"][seen["argv"].index("--answerer-model") + 1] == "gpt-5.4-2026-03-05"
    assert "--answerer-cache-only" in seen["argv"] and seen["argv"][seen["argv"].index("--entailment") + 1] == "llm"
    report = (tmp_path / "judge_agreement_groq.md").read_text()
    assert "groq:openai/gpt-oss-120b" in report and "llm:gpt-4.1-mini" in report
    row = next(line for line in report.splitlines() if line.startswith("| llm:gpt-4.1-mini | groq:"))
    assert "137/147" in row                                # 10 flipped verdicts out of 147 pairs
    assert "Questions whose type differs (llm:gpt-4.1-mini vs groq:openai/gpt-oss-120b)" in report


def test_judge_agreement_refuses_a_billing_judge_without_a_budget(tmp_path, monkeypatch):
    monkeypatch.delenv("LINKRAG_JUDGE_BACKEND", raising=False)
    ja = load("judge_agreement")
    monkeypatch.setitem(sys.modules, "verify_gold",
                        types.SimpleNamespace(main=lambda argv: pytest.fail("must not run")))
    with pytest.raises(SystemExit, match="bills"):
        ja.main(["--judge", "default", "--max-cost", "0", "--out-dir", str(tmp_path)])      # gpt-4.1-mini, metered
    ja.refuse_billing({"billing": "metered"}, 0.5, False)               # an explicit budget allows it
    ja.refuse_billing({"billing": "metered"}, 0.0, True)                # replay-only never bills
