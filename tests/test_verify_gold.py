"""Automated gold verification: majority logic, span check, relabel rules, sampler."""

from __future__ import annotations

import json

from linkrag.core import EvidenceUnit, Location
from linkrag.eval.verify_gold import (
    SourceRun, entail_unit, relabel, span_in_text, verified_gold_rows, verify_gold,
)


def _unit(uid, modality, content, *, page=None, start=None, end=None, deck=False):
    src = "deck.pdf" if deck else ("talk.mp3" if modality == "audio" else "paper.pdf")
    return EvidenceUnit(id=uid, modality=modality, content=content, source_file=src,
                        location=Location(page=page, start_s=start, end_s=end),
                        metadata={"slide_deck": deck})


def test_span_must_be_quotable():
    text = "We use YOLOv2 as the ground truth CNN, trained on COCO."
    assert span_in_text("YOLOv2 as the ground truth", text)
    assert span_in_text("We use YOLOv2 ... trained on COCO", text)
    assert not span_in_text("ResNet152", text)
    assert not span_in_text("", text)


def test_entail_majority_and_unquoted_yes_counts_as_no():
    u = _unit("p:p1:t0", "text", "Poster number 47 is ours.", page=1)
    fact = lambda span, ok=True: {"facts": [{"fact": "47", "supported": ok, "span": span}],
                                  "verdict": "yes" if ok else "no"}
    replies = iter([
        json.dumps(fact("Poster number 47")),
        json.dumps(fact("booth 12")),            # unquotable -> no
        json.dumps(fact("", ok=False)),
    ])
    v = entail_unit("Which poster?", "47", u, lambda s, p: next(replies), runs=3, backend="llm")
    assert v.votes == ["yes", "yes-unquoted", "no"]
    assert not v.kept


def _run(source, ok):
    return SourceRun(source, "a", ok, ["PASS" if ok else "FAIL"] * 3, "", [])


def test_relabel_rules():
    runs = {"audio": _run("audio", False), "deck": _run("deck", False),
            "paper": _run("paper", False), "all": _run("all", True)}
    assert relabel("cross_modal_split", runs)[0] == "cross_modal_split"
    assert relabel("slides_only", runs)[0] == "cross_modal_split"     # only all passed
    runs["deck"] = _run("deck", True)
    t, status, src = relabel("cross_modal_deictic", runs)
    assert (t, src) == ("slides_only", ["deck"]) and status.startswith("relabelled")
    runs["paper"] = _run("paper", True)
    assert relabel("cross_modal_split", runs)[0] == "single_modality"
    runs = {s: _run(s, False) for s in ("audio", "deck", "paper", "all")}
    assert relabel("audio_only", runs)[0] is None


def test_end_to_end_with_scripted_judge():
    """Two-unit cross-modal question: one unit passes, one fails; deck alone answers
    it -> relabelled slides_only, dropped unit logged, gold narrowed to the survivor."""
    units = [
        _unit("deck:p3:t0", "text", "Focus is 57x cheaper and 162x faster.", page=3, deck=True),
        _unit("deck:p3:g0", "figure", "Loading... [| PRR a aR", page=3, deck=True),
        _unit("talk:a1", "audio", "Our poster is number 47.", start=10.0, end=20.0),
    ]
    row = {"qid": "X1", "type": "cross_modal_split", "question": "How much cheaper is Focus?",
           "expected_answer": "57x cheaper.",
           "gold_units": [{"source": "deck.pdf", "page": 3}], "gold_terms": {"slide": [], "audio": []}}

    def judge(system, prompt):
        if prompt.startswith("Context ("):                       # answering
            return "57x cheaper" if "57x" in prompt else "NOT ANSWERABLE"
        if "Candidate answer:" in prompt:                        # grading
            ok = "57x" in prompt.split("Candidate answer:")[1]
            return json.dumps({"facts": [{"fact": "57x", "present": ok}],
                               "verdict": "PASS" if ok else "FAIL", "reason": "r"})
        ok = "57x cheaper" in prompt                             # entailment
        return json.dumps({"facts": [{"fact": "57x", "supported": ok, "span": "57x cheaper" if ok else ""}],
                           "verdict": "yes" if ok else "no"})

    out = verify_gold([row], units, judge, judge=judge, deck_files={"deck.pdf"}, runs=3, workers=2,
                      backend="llm")
    v = out[0]
    assert v.verified_type == "slides_only" and v.status.startswith("relabelled")
    assert [u.kept for u in v.units] == [True, False]
    rows = verified_gold_rows(out, units)
    assert rows[0]["gold_unit_ids"] == ["deck:p3:t0"]
    assert rows[0]["gold_units"][0]["page"] == 3 and rows[0]["gold_modalities"] == ["text"]


def test_audit_sampler_stratifies_with_minimum():
    import importlib.util, sys
    spec = importlib.util.spec_from_file_location("audit_sample", "scripts/eval/audit_sample.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    rows = [{"type": "a", "i": i} for i in range(100)] + [{"type": "b", "i": i} for i in range(3)]
    s = mod.stratified_sample(rows, fraction=0.1, minimum=5, seed=1)
    from collections import Counter
    c = Counter(r["type"] for r in s)
    assert c == {"a": 10, "b": 3}


def test_nli_is_an_ablation_that_cannot_write_gold_or_stored_reports(tmp_path, monkeypatch):
    """DESIGN.md finding 11: NLI never decides gold or intake, and its reports say what it is."""
    import pytest
    from linkrag.core import load_config, set_max_cost
    from linkrag.eval.verify_gold import refuse_nli_decisions, verifier_label
    with pytest.raises(SystemExit, match="ablation"):
        refuse_nli_decisions("nli", "tests/regression/pilot01_questions.jsonl", tmp_path)
    with pytest.raises(SystemExit, match="reports"):
        refuse_nli_decisions("nli", "reports")
    refuse_nli_decisions("nli", tmp_path / "g.jsonl", tmp_path)          # scratch paths are fine
    refuse_nli_decisions("llm", "tests/regression/pilot01_questions.jsonl", "reports")
    for var in ("LINKRAG_JUDGE_BACKEND", "LINKRAG_LLM_BACKEND"):        # the default judge, whatever ran before
        monkeypatch.delenv(var, raising=False)
    cfg = load_config("configs/default.yaml")
    assert verifier_label(cfg, "nli", 3) == ("nli:cross-encoder/nli-deberta-v3-base", 1)
    assert verifier_label(cfg, "llm", 3) == ("gpt-4.1-mini-2025-04-14", 3)
    set_max_cost(cfg, 0.5)                                                # reaches the whisper-1 check too
    assert cfg["cost"]["max_usd"] == 0.5 and cfg["eval"]["judge"]["max_cost_usd"] == 0.5


def test_parse_json_ignores_trailing_content_but_still_rejects_malformed():
    from linkrag.eval.verify_gold import parse_json
    ok = '{"facts": [{"fact": "x", "supported": true, "span": "x"}], "verdict": "yes"}'
    assert parse_json(ok + "}")["verdict"] == "yes"                    # the stray brace judges add
    assert parse_json("Here you go:\n" + ok + "\nThanks!")["verdict"] == "yes"
    assert parse_json("```json\n" + ok + "\n```")["verdict"] == "yes"
    assert parse_json('{"facts": []} , "verdict": "yes"}') == {"facts": []}   # brace mid-reply: verdict lost
    assert parse_json('{"verdict": }') == {}                           # genuinely malformed
    assert parse_json("no json here") == {}
    assert parse_json('[{"verdict": "yes"}]') == {"verdict": "yes"}   # the first object, as before the fix
