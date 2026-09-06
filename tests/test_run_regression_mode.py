"""The regression runner must dispatch on --mode.

Regression for the audit's highest-risk finding: `--mode linkrag` reached only the
generation prompt suffix while retrieval stayed baseline, so a report could label
itself `linkrag` over baseline retrieval and nothing would say so.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_regression  # noqa: E402

CFG = {
    "retrieve": {
        "top_k": 8, "candidates": 50, "rrf_k": 60,
        "linkrag": {"k_seed": 5, "k_final": 8, "hops": 1,
                    "link_types": ["audio_slide"], "min_link_score": 0.0, "decay": 0.5},
    }
}


class _Unit:
    def __init__(self, uid): self.id = uid


class _Result:
    def __init__(self, uid, origin): self.unit, self.id, self.origin = _Unit(uid), uid, origin


@pytest.fixture
def spies(monkeypatch):
    calls = {"baseline": 0, "linkrag": 0}

    def fake_scored(question, index, **kw):
        calls["baseline"] += 1
        return [(_Unit("b1"), 0.03)]

    def fake_linkrag(question, index, graph, **kw):
        calls["linkrag"] += 1
        return [_Result("s1", "seed"), _Result("e1", "expanded")]

    monkeypatch.setattr(run_regression, "retrieve_scored", fake_scored)
    monkeypatch.setattr(run_regression, "retrieve_linkrag", fake_linkrag)
    monkeypatch.setattr(run_regression, "expansion_report", lambda results, g: (1, 2))
    return calls


def test_baseline_mode_calls_only_the_baseline_retriever(spies) -> None:
    units, expanded, seeded = run_regression.retrieve_for_mode(
        "baseline", "q", index=None, encoder=None, graph=None, cfg=CFG)
    assert spies == {"baseline": 1, "linkrag": 0}
    assert [u.id for u in units] == ["b1"]
    assert (expanded, seeded) == (0, 0)


def test_linkrag_mode_calls_only_the_linkrag_retriever(spies) -> None:
    units, expanded, seeded = run_regression.retrieve_for_mode(
        "linkrag", "q", index=None, encoder=None, graph=object(), cfg=CFG)
    assert spies == {"baseline": 0, "linkrag": 1}, "linkrag mode must not use baseline retrieval"
    assert [u.id for u in units] == ["s1", "e1"]
    assert (expanded, seeded) == (1, 2)


def test_linkrag_mode_passes_the_configured_knobs(monkeypatch) -> None:
    seen = {}

    def fake_linkrag(question, index, graph, **kw):
        seen.update(kw)
        return []

    monkeypatch.setattr(run_regression, "retrieve_linkrag", fake_linkrag)
    monkeypatch.setattr(run_regression, "expansion_report", lambda r, g: (0, 0))
    run_regression.retrieve_for_mode("linkrag", "q", index=None, encoder=None,
                                     graph=object(), cfg=CFG)
    assert seen["mode"] == "linkrag"
    assert seen["k_seed"] == 5 and seen["k_final"] == 8
    assert seen["link_types"] == ["audio_slide"] and seen["decay"] == 0.5
