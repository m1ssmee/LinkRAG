"""Phase 6: structured answers, claim verification, citations.

The named case is Q3 from the pilot: an answer that cites a real unit and is still
false. The unit text is the persisted transcript, not a paraphrase.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from linkrag.core import EvidenceUnit, Location
from linkrag.generate.answer import Answer, Claim, answer_json, parse_answer_json
from linkrag.generate.citations import cite, locator, mmss
from linkrag.generate.verify import ABSTENTION, citation_correctness, hallucination_rate, verify_answer

# hsieh:a42, data/processed/transcripts/hsieh.frozen.json -- the closing line runs
# straight into an audience member's self-introduction, which is how Q3 produced a
# false author with a valid citation (DESIGN.md, known issue (a)).
A42 = ("querying objects in massive videos is very challenging. And our approach is we enable low latency "
       "query using low cost ingest. Our result shows that focus is 57 times cheaper than ingest time only "
       "solution, and it's 162 times faster than state -of -the -art query time only solution. Our poster is "
       "number 47, and you are welcome to see. Saurabh Bakhti from Purdue. So you do some clustering to speed "
       "up your processing.")


def _audio(uid="hsieh:a42", text=A42, start=1109.7, end=1147.4):
    return EvidenceUnit(id=uid, modality="audio", content=text, source_file="data/raw/pilot01/hsieh.mp3",
                        location=Location(start_s=start, end_s=end))


def _judge(yes_when):
    """Scripted judge in verify_gold's JSON shape: says yes iff `yes_when(prompt)`."""
    def judge(system, prompt):
        ok = yes_when(prompt)
        span = "Our poster is number 47" if ok else ""
        return json.dumps({"facts": [{"fact": "f", "supported": ok, "span": span}],
                           "verdict": "yes" if ok else "no"})
    return judge


# ------------------------------------------------------------------ JSON answers

def test_parse_tolerates_fences_and_prefixes_and_rejects_garbage():
    assert parse_answer_json('```json\n{"answer":"a","claims":[{"claim":"c","unit_ids":["u1"]}]}\n```')[0] == "a"
    assert parse_answer_json('Sure: {"answer":"a","claims":[]}')[1] == []
    assert parse_answer_json("no json here") is None
    assert parse_answer_json('{"claims":[]}') is None          # no answer key


def test_malformed_output_is_retried_once_then_flagged():
    units = [_audio()]
    calls = []

    def bad(system, user):
        calls.append(user)
        return "I think the answer is 47."
    ans = answer_json("q", units, complete=bad)
    assert len(calls) == 2 and "not valid JSON" in calls[1]
    assert ans.malformed and ans.answer.startswith("I think")

    good = iter(["oops", '{"answer":"47","claims":[{"claim":"poster 47","unit_ids":["hsieh:a42"]}]}'])
    ans = answer_json("q", units, complete=lambda s, u: next(good))
    assert not ans.malformed and ans.claims[0].unit_ids == ["hsieh:a42"]


def test_fabricated_unit_ids_are_dropped_before_verification():
    units = [_audio()]
    reply = '{"answer":"x","claims":[{"claim":"c","unit_ids":["hsieh:a42","made:up:id"]}]}'
    ans = answer_json("q", units, complete=lambda s, u: reply)
    assert ans.claims[0].unit_ids == ["hsieh:a42"]


# ------------------------------------------------------------------ the Q3 case

def test_q3_false_author_claim_is_caught_by_claim_level_verification():
    """The citation is valid (hsieh:a42 really is in the evidence and really does
    contain 'Saurabh Bakhti from Purdue'), so an id-level check passes. The claim
    'Saurabh Bakshi is an author of Focus' is not entailed by that unit, and the
    claim-level check must mark it unsupported."""
    units = [_audio()]
    ans = Answer(answer="The authors include Kevin Hsieh and Saurabh Bakshi from Purdue.", raw="",
                 claims=[Claim(claim="The poster number is 47.", unit_ids=["hsieh:a42"]),
                         Claim(claim="Saurabh Bakshi from Purdue is an author of Focus.", unit_ids=["hsieh:a42"])])
    judge = _judge(lambda prompt: "poster number is 47" in prompt)
    out = verify_answer(ans, units, judge, runs=3)
    verdicts = {c["claim"]: c["verdict"] for c in out["claims"]}
    assert verdicts["The poster number is 47."] == "supported"
    assert verdicts["Saurabh Bakshi from Purdue is an author of Focus."] == "unsupported"
    assert out["hallucination_rate"] == 0.5
    # id-level checking would have passed this answer:
    assert all(uid in {u.id for u in units} for c in out["claims"] for uid in c["unit_ids"])


def test_strict_hides_unsupported_claims_and_abstains_when_none_survive():
    units = [_audio()]
    ans = Answer(answer="Saurabh Bakshi is an author.", raw="",
                 claims=[Claim(claim="Saurabh Bakshi is an author of Focus.", unit_ids=["hsieh:a42"])])
    out = verify_answer(ans, units, _judge(lambda p: False), runs=3, strict=True)
    assert out["answer"] == ABSTENTION and out["abstained"] and out["hallucination_rate"] == 1.0
    ans2 = Answer(answer="The poster is number 47.", raw="",
                  claims=[Claim(claim="The poster number is 47.", unit_ids=["hsieh:a42"])])
    out2 = verify_answer(ans2, units, _judge(lambda p: True), runs=3, strict=True)
    assert out2["answer"] == "The poster is number 47." and not out2["abstained"]


def test_a_claim_with_no_citation_is_weak_not_supported():
    out = verify_answer(Answer(answer="x", raw="", claims=[Claim(claim="unsourced", unit_ids=[])]),
                        [_audio()], _judge(lambda p: True), runs=3)
    assert out["claims"][0]["verdict"] == "weak" and out["unsupported"] == 0


# ------------------------------------------------------------------ metrics

def test_metrics_pool_over_answers_and_match_gold_by_locator_not_id():
    results = [{"claims": [{"unit_ids": ["hsieh:a42"], "verdict": "supported"},
                           {"unit_ids": ["hsieh:a42"], "verdict": "unsupported"}], "unsupported": 1},
               {"claims": [{"unit_ids": ["hsieh:a99"], "verdict": "unsupported"}], "unsupported": 1}]
    assert hallucination_rate(results) == pytest.approx(2 / 3)
    units = [_audio(), _audio("hsieh:a99", "unrelated", 10.0, 20.0)]
    gold = [{"source": "hsieh.mp3", "start_s": 1100.0, "end_s": 1150.0}]
    # 2 of 3 cited units overlap the gold time span; the id itself is never compared
    assert citation_correctness(results, units, gold) == pytest.approx(2 / 3)
    renamed = [_audio("totally:different:id"), _audio("hsieh:a99", "unrelated", 10.0, 20.0)]
    results2 = [{"claims": [{"unit_ids": ["totally:different:id"], "verdict": "supported"}], "unsupported": 0}]
    assert citation_correctness(results2, renamed, gold) == 1.0


# ------------------------------------------------------------------ citations

def test_locator_and_mmss():
    assert mmss(804.0) == "13:24"
    assert locator(_audio()) == "hsieh.mp3 18:30-19:07"
    page = EvidenceUnit(id="d:p3:t0", modality="text", content="x", source_file="deck.pdf",
                        location=Location(page=3))
    assert locator(page) == "deck.pdf p.3"


def test_citation_reports_a_missing_source_instead_of_raising():
    c = cite(EvidenceUnit(id="d:p3:t0", modality="text", content="x", source_file="nope.pdf",
                          location=Location(page=3)))
    assert c.path is None and c.note and c.where == "nope.pdf p.3"


@pytest.mark.skipif(not Path("data/raw/pilot01/osdi18_slides_hsieh.pdf").exists(), reason="pilot01 corpus absent")
def test_page_crop_is_written_once(tmp_path):
    unit = EvidenceUnit(id="osdi18_slides_hsieh:p20:t0", modality="text", content="x",
                        source_file="data/raw/pilot01/osdi18_slides_hsieh.pdf",
                        location=Location(page=20, bbox=(40.0, 60.0, 500.0, 300.0)))
    c = cite(unit, tmp_path)
    assert c.path and c.path.exists() and c.path.stat().st_size > 1000
    before = c.path.stat().st_mtime_ns
    assert cite(unit, tmp_path).path.stat().st_mtime_ns == before, "an existing crop must not be re-rendered"


def test_regression_runner_exposes_grounding_metrics():
    """The runner must compute both Phase-6 metrics, not just print an answer."""
    src = Path("scripts/run_regression.py").read_text()
    assert "hallucination_rate(verified)" in src and "citation_correctness(verified" in src
    assert "--strict" in src


def test_verify_module_reuses_the_gold_entailment_primitive_not_a_copy():
    """Same prompt, same 3-run majority, same quotable-span rule as gold verification."""
    import inspect
    from linkrag.generate import verify
    assert "from linkrag.eval.verify_gold import entail_unit" in inspect.getsource(verify)
    assert "ENTAIL_PROMPT" not in inspect.getsource(verify), "the prompt must not be duplicated here"


def test_compare_retrieval_grounding_table_renders_without_an_api_key():
    """The per-cell block crashed once after the matrix had already printed, leaving no
    ledger row and no table. Exercised here with stub verdicts so it cannot regress."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("cmp_retr", "scripts/compare_retrieval.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    units = [_audio()]
    verdict = {"claims": [{"claim": "c", "unit_ids": ["hsieh:a42"], "verdict": "supported"},
                          {"claim": "d", "unit_ids": ["hsieh:a42"], "verdict": "unsupported"}],
               "unsupported": 1, "weak": 0, "supported": 1, "abstained": False,
               "gold_units": [{"source": "hsieh.mp3", "start_s": 1100.0, "end_s": 1150.0}]}
    out = m.grounding_table({("linkrag", "complementarity"): [verdict]}, units)
    assert any("Grounding (claim-level" in line for line in out)
    row = [line for line in out if line.startswith("| linkrag |")][0]
    assert "50.0%" in row and "100.0%" in row        # 1 of 2 claims unsupported; both citations on gold
    assert m.grounding_table({}, units) == []
