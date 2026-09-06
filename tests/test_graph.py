"""The link graph: traversal, filtering and GraphML export."""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import pytest

from linkrag.core import EvidenceUnit, Link, Location
from linkrag.link.graph import (
    build_graph,
    export_graphml,
    neighbors,
    subgraph_around,
    summarise,
)


@pytest.fixture
def graph():
    units = [
        EvidenceUnit(id="a1", modality="audio", content="this arrow here",
                     source_file="talk.mp3", location=Location(start_s=10.0, end_s=40.0)),
        EvidenceUnit(id="s5", modality="text", content="slide five text",
                     source_file="deck.pdf", location=Location(page=5)),
        EvidenceUnit(id="f5", modality="figure", content="heatmap",
                     source_file="deck.pdf", location=Location(page=5)),
        EvidenceUnit(id="s9", modality="text", content="slide nine text",
                     source_file="deck.pdf", location=Location(page=9)),
    ]
    links = [
        Link("a1", "s5", "audio_slide", 0.90),
        Link("a1", "f5", "deictic", 0.70, {"phrase": "this arrow here"}),
        Link("f5", "s9", "figure_text", 0.55, {"reference": "figure 1"}),
    ]
    return build_graph(units, links)


def test_nodes_carry_provenance(graph) -> None:
    assert graph.number_of_nodes() == 4 and graph.number_of_edges() == 3
    assert graph.nodes["a1"]["modality"] == "audio"
    assert graph.nodes["s5"]["page"] == 5
    assert graph.nodes["a1"]["start_s"] == pytest.approx(10.0)


def test_link_metadata_survives_onto_the_edge(graph) -> None:
    # edges are auto-keyed now, so look the edge up by its data, not by key
    data = next(d for _u, _v, d in graph.edges(data=True) if d["link_type"] == "deictic")
    assert data["meta_phrase"] == "this arrow here"
    assert data["score"] == pytest.approx(0.70)


def test_links_to_unknown_units_are_dropped_not_invented() -> None:
    units = [EvidenceUnit(id="a1", modality="audio", content="x", source_file="t.mp3")]
    g = build_graph(units, [Link("a1", "ghost", "deictic", 0.9)])
    assert g.number_of_nodes() == 1, "a dangling id must not create a phantom node"
    assert g.graph["dropped_links"] == 1


def test_neighbors_follows_both_directions_by_default(graph) -> None:
    """A slide must be reachable from the speech about it and vice versa."""
    assert [n for n, _t, _s in neighbors(graph, "a1")] == ["s5", "f5"]
    assert [n for n, _t, _s in neighbors(graph, "s5")] == ["a1"]
    assert neighbors(graph, "s5", direction="out") == []
    assert [n for n, _t, _s in neighbors(graph, "s5", direction="in")] == ["a1"]


def test_neighbors_filters_by_type_and_score(graph) -> None:
    assert [n for n, _t, _s in neighbors(graph, "a1", link_types=["deictic"])] == ["f5"]
    assert [n for n, _t, _s in neighbors(graph, "a1", min_score=0.8)] == ["s5"]
    assert neighbors(graph, "a1", link_types=["same_topic"]) == []


def test_neighbors_sorted_by_score_descending(graph) -> None:
    scores = [s for _n, _t, s in neighbors(graph, "a1")]
    assert scores == sorted(scores, reverse=True)


def test_neighbors_of_an_unknown_unit_is_empty_not_an_error(graph) -> None:
    assert neighbors(graph, "nope") == []


def test_neighbors_rejects_a_bad_direction(graph) -> None:
    with pytest.raises(ValueError, match="unknown direction"):
        neighbors(graph, "a1", direction="sideways")


def test_subgraph_expands_by_hops(graph) -> None:
    one = subgraph_around(graph, "a1", hops=1)
    assert set(one.nodes()) == {"a1", "s5", "f5"}
    two = subgraph_around(graph, "a1", hops=2)
    assert set(two.nodes()) == {"a1", "s5", "f5", "s9"}, "second hop reaches via the figure"
    assert set(subgraph_around(graph, "a1", hops=0).nodes()) == {"a1"}


def test_subgraph_filter_applies_to_edges_too(graph) -> None:
    sub = subgraph_around(graph, "a1", hops=2, min_score=0.8)
    assert set(sub.nodes()) == {"a1", "s5"}
    assert all(d["score"] >= 0.8 for _u, _v, d in sub.edges(data=True))


def test_subgraph_of_an_unknown_unit_raises(graph) -> None:
    with pytest.raises(KeyError):
        subgraph_around(graph, "nope")


def test_summarise_counts_and_averages_by_type(graph) -> None:
    stats = summarise(graph)
    assert stats["audio_slide"]["count"] == 1
    assert stats["deictic"]["mean_score"] == pytest.approx(0.70)
    assert set(stats) == {"audio_slide", "deictic", "figure_text"}


def test_graphml_round_trip(graph, tmp_path: Path) -> None:
    out = export_graphml(graph, tmp_path / "g.graphml")
    assert out.exists()
    reloaded = nx.read_graphml(str(out))
    assert reloaded.number_of_nodes() == graph.number_of_nodes()
    assert reloaded.number_of_edges() == graph.number_of_edges()


def test_graphml_stringifies_non_scalar_attributes(tmp_path: Path) -> None:
    """GraphML holds only scalars; a dict attribute must not blow up the writer
    halfway through and leave a truncated file."""
    units = [EvidenceUnit(id="a", modality="audio", content="x", source_file="t.mp3")]
    g = build_graph(units, [])
    g.nodes["a"]["awkward"] = {"nested": [1, 2]}
    assert export_graphml(g, tmp_path / "g.graphml").exists()


def test_parallel_links_of_the_same_type_are_all_kept() -> None:
    """Regression: keying edges by link_type collapsed two cues in one segment
    pointing at the same figure, silently losing 19 of 41 deictic links."""
    units = [
        EvidenceUnit(id="a1", modality="audio", content="x", source_file="t.mp3"),
        EvidenceUnit(id="f1", modality="figure", content="y", source_file="d.pdf"),
    ]
    links = [
        Link("a1", "f1", "deictic", 0.8, {"phrase": "this arrow here"}),
        Link("a1", "f1", "deictic", 0.6, {"phrase": "as you can see"}),
    ]
    g = build_graph(units, links)
    assert g.number_of_edges() == 2, "both cues must survive"
    assert summarise(g)["deictic"]["count"] == 2
    phrases = {d["meta_phrase"] for _u, _v, d in g.edges(data=True)}
    assert phrases == {"this arrow here", "as you can see"}


def test_neighbors_dedupes_parallel_edges_to_the_best_score() -> None:
    units = [
        EvidenceUnit(id="a1", modality="audio", content="x", source_file="t.mp3"),
        EvidenceUnit(id="f1", modality="figure", content="y", source_file="d.pdf"),
    ]
    g = build_graph(units, [Link("a1", "f1", "deictic", 0.8),
                            Link("a1", "f1", "deictic", 0.6)])
    assert neighbors(g, "a1") == [("f1", "deictic", 0.8)]
