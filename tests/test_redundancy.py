"""Modality redundancy metric: sentence splitting, role assignment, majority, report."""

from __future__ import annotations

import json

import numpy as np

from linkrag.core import EvidenceUnit, Location
from linkrag.eval.redundancy import redundancy, sentences_by_role, split_sentences, summarise


def test_split_sentences_and_bullets():
    assert split_sentences("Focus is 57x cheaper. It is also 162x faster than NoScope today.") == [
        "Focus is 57x cheaper.", "It is also 162x faster than NoScope today."][1:]  # first has 4 words
    deck = "Key Takeaways • Querying objects in massive videos is challenging • Low-latency query with low-cost ingest"
    items = split_sentences(deck, bullets=True)
    assert items == ["Querying objects in massive videos is challenging",
                     "Low-latency query with low-cost ingest"]


def _u(uid, modality, content, deck=False, start=None, page=None):
    return EvidenceUnit(id=uid, modality=modality, content=content,
                        source_file="deck.pdf" if deck else ("talk.mp3" if modality == "audio" else "paper.pdf"),
                        location=Location(start_s=start, end_s=(start or 0) + 10, page=page),
                        metadata={"slide_deck": deck})


def test_roles_and_dedup():
    units = [_u("a0", "audio", "Focus is fifty seven times cheaper than the baseline. Focus is fifty seven times cheaper than the baseline.", start=0),
             _u("d1", "text", "• Focus is 57x cheaper than Ingest-heavy", deck=True, page=1),
             _u("g1", "figure", "OCR noise that must not become a deck claim here", deck=True, page=1),
             _u("p1", "text", "Focus reduces ingest cost by 57x on average across videos.", page=3)]
    by = sentences_by_role(units)
    assert len(by["transcript"]) == 1          # duplicate sentence collapsed
    assert [s for s, _ in by["deck"]] == ["Focus is 57x cheaper than Ingest-heavy"]
    assert len(by["paper"]) == 1


def test_redundancy_majority_and_summary():
    units = [_u("a0", "audio", "Focus is fifty seven times cheaper than the baseline.", start=0),
             _u("a1", "audio", "Our poster is number forty seven, please come by.", start=10),
             _u("d1", "text", "• Focus is 57x cheaper than Ingest-heavy", deck=True, page=1)]
    encoder = lambda texts: np.ones((len(texts), 4), dtype=np.float32)
    calls = []

    def judge(system, prompt):
        calls.append(prompt)
        ok = "fifty seven" in prompt
        return json.dumps({"entailed": ok, "span": "57x cheaper" if ok else ""})

    v = redundancy(units, encoder, judge, pairs=[("transcript", "deck")], k=2, runs=3, workers=2,
                   backend="llm")
    assert [x.entailed for x in v] == [True, False]
    assert v[0].votes == ["yes", "yes", "yes"] and len(calls) == 6
    s = summarise(v)
    assert s["pairs"]["transcript->deck"]["fraction"] == 0.5 and s["overall"]["n"] == 2


def test_unquotable_yes_counts_as_no():
    units = [_u("a0", "audio", "Focus is fifty seven times cheaper than the baseline.", start=0),
             _u("d1", "text", "• Focus is 57x cheaper than Ingest-heavy", deck=True, page=1)]
    encoder = lambda texts: np.ones((len(texts), 4), dtype=np.float32)
    judge = lambda s, p: json.dumps({"entailed": True, "span": "not in any passage"})
    v = redundancy(units, encoder, judge, pairs=[("transcript", "deck")], k=1, runs=3, workers=1,
                   backend="llm")
    assert not v[0].entailed and v[0].votes == ["yes-unquoted"] * 3


def test_nli_backend_scores_redundancy_without_a_judge():
    """Zero-cost default: the local cross-encoder decides, deterministically."""
    units = [_u("a0", "audio", "Focus is fifty seven times cheaper than the ingest-heavy baseline.", start=0),
             _u("a1", "audio", "Our poster is number forty seven, please come by.", start=10),
             _u("d1", "text", "Focus is 57x cheaper than Ingest-heavy.", deck=True, page=1)]
    encoder = lambda texts: np.ones((len(texts), 4), dtype=np.float32)

    def exploding(system, prompt):
        raise AssertionError("the nli backend must not call the judge")

    v = redundancy(units, encoder, exploding, pairs=[("transcript", "deck")], k=2, workers=1, backend="nli")
    by_sentence = {x.sentence.split()[1]: x.entailed for x in v}   # "Focus" / "poster"
    assert by_sentence["is"] is True and by_sentence["poster"] is False
    assert all(x.votes in (["yes"], ["no"]) for x in v), "one deterministic run"


def test_stratified_sample_and_wilson_ci():
    from linkrag.eval.redundancy import intake_gate, sample_positions, wilson
    idx = sample_positions(167, 50, "s:a->b")
    assert len(idx) == 50 and len(set(idx)) == 50 and idx == sorted(idx)
    assert idx == sample_positions(167, 50, "s:a->b")                    # seeded: reproducible
    assert all(i * 167 // 50 - 4 <= x <= (i + 1) * 167 // 50 + 4 for i, x in enumerate(idx))  # one per stratum
    assert sample_positions(52, 150, "x") == list(range(52))              # N >= population: census
    lo, hi = wilson(49, 52)
    assert 0.84 < lo < 0.85 and 0.97 < hi < 0.99
    row = {"sampled": True, "fraction": 0.94, "ci95": (0.84, 0.98)}
    assert intake_gate(row, 0.65) == "REJECT"
    assert intake_gate({**row, "ci95": (0.40, 0.60)}, 0.65) == "KEEP"
    assert intake_gate({**row, "ci95": (0.55, 0.75)}, 0.65) == "BORDERLINE"
    assert intake_gate({"sampled": False, "fraction": 0.60}, 0.65) == "KEEP"


def test_sampled_redundancy_agrees_with_the_full_run_on_pilot01_cached_verdicts():
    """DESIGN.md design change (sampled redundancy): on the stored LLM verdicts, a
    stratified sample puts the full-run fraction inside its 95% CI for every pair."""
    import json
    from collections import defaultdict
    from pathlib import Path
    from linkrag.eval.redundancy import sample_positions, wilson
    by = defaultdict(list)
    for v in json.loads(Path("reports/redundancy_pilot01.json").read_text()):
        by[(v["source"], v["target"])].append(v["entailed"])
    assert sum(len(x) for x in by.values()) == 438
    for n in (150, 50):
        for (s, t), xs in by.items():
            idx = sample_positions(len(xs), n, f"20260923:{s}->{t}")
            lo, hi = wilson(sum(xs[i] for i in idx), len(idx))
            assert lo <= sum(xs) / len(xs) <= hi, (n, s, t)
