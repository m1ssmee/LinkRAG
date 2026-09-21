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

    v = redundancy(units, encoder, judge, pairs=[("transcript", "deck")], k=2, runs=3, workers=2)
    assert [x.entailed for x in v] == [True, False]
    assert v[0].votes == ["yes", "yes", "yes"] and len(calls) == 6
    s = summarise(v)
    assert s["pairs"]["transcript->deck"]["fraction"] == 0.5 and s["overall"]["n"] == 2


def test_unquotable_yes_counts_as_no():
    units = [_u("a0", "audio", "Focus is fifty seven times cheaper than the baseline.", start=0),
             _u("d1", "text", "• Focus is 57x cheaper than Ingest-heavy", deck=True, page=1)]
    encoder = lambda texts: np.ones((len(texts), 4), dtype=np.float32)
    judge = lambda s, p: json.dumps({"entailed": True, "span": "not in any passage"})
    v = redundancy(units, encoder, judge, pairs=[("transcript", "deck")], k=1, runs=3, workers=1)
    assert not v[0].entailed and v[0].votes == ["yes-unquoted"] * 3
