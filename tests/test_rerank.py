"""Complementarity-aware reranking and its two ablations."""

from __future__ import annotations

import numpy as np
import pytest

from linkrag.core import EvidenceUnit, Link, Location
from linkrag.index import build_index
from linkrag.link.graph import build_graph
from linkrag.retrieve.linkrag import RetrievedUnit
from linkrag.retrieve.rerank import (
    METHODS,
    cross_encoder_input,
    modality_tag,
    rerank,
    set_diagnostics,
)

# Five near-duplicate audio segments, each slightly more relevant than the figure,
# plus one figure that is *linked* to the top audio unit. Plain top-k fills the set
# with restatements; complementarity must reach the figure.
AUDIO_TEXT = "the attention weights concentrate on the subject token in this example"
FIG_TEXT = "heatmap of attention weights layer six head three"


@pytest.fixture
def scenario(stub_encoder):
    units = [
        EvidenceUnit(id=f"a{i}", modality="audio", content=AUDIO_TEXT,
                     source_file="talk.mp3",
                     location=Location(start_s=i * 30.0, end_s=(i + 1) * 30.0))
        for i in range(5)
    ]
    units.append(EvidenceUnit(id="fig", modality="figure", content=FIG_TEXT,
                              source_file="deck.pdf", location=Location(page=12)))
    index = build_index(units, encoder=stub_encoder([AUDIO_TEXT, FIG_TEXT]),
                        embedding_model="stub")
    graph = build_graph(units, [Link("a0", "fig", "deictic", 0.8)])
    # audio scores strictly above the figure, so relevance alone never picks it
    results = [RetrievedUnit(unit=u, score=0.90 - 0.01 * i, origin="seed")
               for i, u in enumerate(units[:5])]
    results.append(RetrievedUnit(unit=units[5], score=0.50, origin="expanded",
                                 via_seed="a0", via_link="deictic", link_score=0.8))
    return results, index, graph


# --------------------------------------------------- the claim, as a test

def test_top_k_fills_the_set_with_near_duplicates(scenario) -> None:
    results, index, graph = scenario
    picked = rerank(results, k=5, method="none", index=index, graph=graph)
    assert [r.id for r in picked] == ["a0", "a1", "a2", "a3", "a4"]
    assert "fig" not in {r.id for r in picked}, "top-k must miss the figure"
    assert set_diagnostics(picked, index)["modalities"] == 1


def test_complementarity_reaches_the_linked_figure(scenario) -> None:
    results, index, graph = scenario
    picked = rerank(results, k=5, method="complementarity", index=index, graph=graph,
                    alpha=0.3, beta=0.2, gamma=0.4)
    ids = [r.id for r in picked]
    assert "fig" in ids, f"complementarity must include the figure, got {ids}"
    assert set_diagnostics(picked, index)["modalities"] == 2


def test_complementarity_lowers_redundancy_versus_top_k(scenario) -> None:
    results, index, graph = scenario
    plain = set_diagnostics(rerank(results, k=5, method="none", index=index), index)
    comp = set_diagnostics(
        rerank(results, k=5, method="complementarity", index=index, graph=graph), index)
    assert comp["redundancy"] <= plain["redundancy"]
    assert comp["modalities"] > plain["modalities"]


# ------------------------------------------------------------- each term

def test_alpha_is_what_buys_the_modality(scenario) -> None:
    """With no modality bonus and no link bonus the figure loses on relevance."""
    results, index, graph = scenario
    without = rerank(results, k=3, method="complementarity", index=index, graph=graph,
                     alpha=0.0, beta=0.0, gamma=0.0)
    assert "fig" not in {r.id for r in without}
    with_alpha = rerank(results, k=3, method="complementarity", index=index, graph=graph,
                        alpha=1.0, beta=0.0, gamma=0.0)
    assert "fig" in {r.id for r in with_alpha}


def test_beta_rewards_a_link_into_the_selected_set(scenario) -> None:
    results, index, graph = scenario
    with_beta = rerank(results, k=3, method="complementarity", index=index, graph=graph,
                       alpha=0.0, beta=1.0, gamma=0.0)
    assert "fig" in {r.id for r in with_beta}, "the a0->fig edge should pull it in"
    no_graph = rerank(results, k=3, method="complementarity", index=index, graph=None,
                      alpha=0.0, beta=1.0, gamma=0.0)
    assert "fig" not in {r.id for r in no_graph}, "no graph, no edge bonus"


def test_gamma_penalises_only_within_a_modality(scenario) -> None:
    """An audio segment and the figure it describes are supposed to be similar;
    charging for that would defeat the objective."""
    results, index, graph = scenario
    heavy = rerank(results, k=3, method="complementarity", index=index, graph=graph,
                   alpha=0.0, beta=0.0, gamma=5.0)
    assert "fig" in {r.id for r in heavy}, "cross-modal similarity must not be charged"


# ---------------------------------------------------------------- ablations

def test_mmr_is_modality_blind(scenario) -> None:
    """MMR diversifies on text alone, so it cannot be credited with the modality
    coverage the complementarity objective produces."""
    results, index, graph = scenario
    picked = rerank(results, k=3, method="mmr", index=index, graph=graph, mmr_lambda=0.5)
    assert len(picked) == 3
    assert all(r.id in {u.id for u in [x.unit for x in results]} for r in picked)


def test_all_methods_return_k_and_no_duplicates(scenario) -> None:
    results, index, graph = scenario
    for method in METHODS:
        picked = rerank(results, k=4, method=method, index=index, graph=graph)
        assert len(picked) == 4 and len({r.id for r in picked}) == 4


def test_unknown_method_raises(scenario) -> None:
    results, index, graph = scenario
    with pytest.raises(ValueError, match="unknown rerank method"):
        rerank(results, k=3, method="magic", index=index, graph=graph)


def test_empty_and_undersized_candidate_lists(scenario) -> None:
    results, index, graph = scenario
    assert rerank([], k=5, index=index) == []
    assert len(rerank(results[:2], k=8, method="complementarity", index=index,
                      graph=graph)) == 2


# ------------------------------------------------ cross-encoder input format

def test_modality_tag_carries_kind_and_location(scenario) -> None:
    results, _index, _graph = scenario
    audio = next(r for r in results if r.unit.modality == "audio" and r.id == "a1")
    figure = next(r for r in results if r.id == "fig")
    assert modality_tag(audio) == "[AUDIO 0:30-1:00]"
    assert modality_tag(figure) == "[FIGURE p.12]"
    assert cross_encoder_input(figure).startswith("[FIGURE p.12] heatmap")


def test_missing_cross_encoder_degrades_to_retrieval_scores(scenario, monkeypatch) -> None:
    """A missing reranker model must not abort a query."""
    import linkrag.retrieve.rerank as rr

    monkeypatch.setattr(rr, "_cross_encoder",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("no model")))
    results, index, graph = scenario
    picked = rerank(results, k=3, method="complementarity", index=index, graph=graph,
                    question="q", cross_encoder="nonexistent/model")
    assert len(picked) == 3


def test_set_diagnostics_reports_coverage_and_redundancy(scenario) -> None:
    results, index, _graph = scenario
    d = set_diagnostics(results[:5], index)
    assert d["units"] == 5 and d["modalities"] == 1
    assert d["redundancy"] == pytest.approx(1.0, abs=1e-6), "identical audio text"
    assert set_diagnostics([], index)["units"] == 0


# ------------------------------------------------ modality-need gate (2026-09-23)

def _unit(uid, modality, text="x"):
    from linkrag.core import EvidenceUnit, Location
    return EvidenceUnit(id=uid, modality=modality, content=text, source_file="f.pdf",
                        location=Location(page=1))


def _res(pairs):
    from linkrag.retrieve.linkrag import RetrievedUnit
    return [RetrievedUnit(unit=_unit(f"u{i}", m), score=s, origin="seed")
            for i, (m, s) in enumerate(pairs)]


def test_gate_calls_a_single_modality_pool_concentrated():
    from linkrag.retrieve.rerank import modality_concentration
    g = modality_concentration(_res([("text", 1.0 - 0.05 * i) for i in range(8)]))
    assert g["concentrated"] and g["share"] == 1.0 and g["dominant"] == "text"


def test_gate_calls_a_mixed_pool_spread():
    from linkrag.retrieve.rerank import modality_concentration
    mix = [("text", 0.9), ("audio", 0.88), ("figure", 0.86), ("text", 0.84),
           ("audio", 0.82), ("figure", 0.80), ("text", 0.78), ("audio", 0.76)]
    g = modality_concentration(_res(mix))
    assert not g["concentrated"] and g["share"] < 0.75 and g["margin"] < 0.5


def test_gate_zeroes_alpha_only_when_concentrated():
    """Concentrated pool: gated selection == plain top-k by score (alpha 0, no links,
    and the redundancy penalty cannot reorder identical-modality ties here).
    Spread pool: the gate leaves alpha alone, so the set still covers modalities."""
    from linkrag.retrieve.rerank import rerank
    conc = _res([("text", 0.90 - 0.01 * i) for i in range(7)] + [("figure", 0.70)])
    gated = [r.id for r in rerank(conc, 4, method="complementarity", alpha=0.3, beta=0.0, gamma=0.0,
                                  modality_gate=True)]
    assert rerank.last_gate["concentrated"]
    assert gated == [r.id for r in rerank(conc, 4, method="none")]
    ungated = [r.id for r in rerank(conc, 4, method="complementarity", alpha=0.3, beta=0.0, gamma=0.0)]
    assert ungated != gated, "without the gate, alpha pulls in the low-scoring other modalities"

    mix = _res([("text", 0.90), ("text", 0.89), ("text", 0.88), ("text", 0.87),
                ("audio", 0.86), ("audio", 0.85), ("figure", 0.84), ("figure", 0.83)])
    with_gate = [r.id for r in rerank(mix, 3, method="complementarity", alpha=0.3, beta=0.0, gamma=0.0,
                                      modality_gate=True)]
    assert not rerank.last_gate["concentrated"]
    assert with_gate == [r.id for r in rerank(mix, 3, method="complementarity", alpha=0.3, beta=0.0, gamma=0.0)]


def test_gate_is_inert_for_other_methods():
    from linkrag.retrieve.rerank import rerank
    conc = _res([("text", 1.0 - 0.05 * i) for i in range(8)])
    assert ([r.id for r in rerank(conc, 3, method="mmr", modality_gate=True)]
            == [r.id for r in rerank(conc, 3, method="mmr")])
    assert rerank.last_gate is None
