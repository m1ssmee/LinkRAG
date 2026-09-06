"""The Evidence Linking Layer as a graph.

Links are produced per type by `align`, `figure_text` and `deictic`; retrieval needs
them as one traversable structure. This module is that structure and nothing more --
no scoring lives here, so a change to how links are *made* cannot silently change how
they are *followed*.

A MultiDiGraph, because two units can legitimately be joined by more than one
relation (a figure aligned to a slide and also referenced by its prose).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import networkx as nx

from linkrag.core import EvidenceUnit, Link

PREVIEW_CHARS = 160


def build_graph(units: Sequence[EvidenceUnit], links: Sequence[Link]) -> nx.MultiDiGraph:
    """Units become nodes, links become typed directed edges.

    A link whose endpoint is not in `units` is dropped rather than creating a
    phantom node -- a dangling id means a stale links file, and inventing an empty
    node would hide that.
    """
    graph = nx.MultiDiGraph()
    for unit in units:
        graph.add_node(
            unit.id,
            modality=unit.modality,
            source_file=unit.source_file,
            location=unit.location.cite(),
            page=unit.location.page if unit.location.page is not None else -1,
            start_s=unit.location.start_s if unit.location.start_s is not None else -1.0,
            preview=" ".join(unit.content.split())[:PREVIEW_CHARS],
        )
    dropped = 0
    for link in links:
        if link.src_id not in graph or link.dst_id not in graph:
            dropped += 1
            continue
        # Auto key, NOT key=link_type: one audio segment can point at the same
        # figure with two different cues ("this arrow here" and "as you can see"),
        # and keying by type silently kept only the last -- 19 of 41 deictic links
        # vanished before this was caught by the per-type counts disagreeing with
        # links.jsonl.
        graph.add_edge(link.src_id, link.dst_id,
                       link_type=link.link_type, score=float(link.score),
                       **{f"meta_{k}": v for k, v in (link.metadata or {}).items()})
    graph.graph["dropped_links"] = dropped
    return graph


def neighbors(
    graph: nx.MultiDiGraph,
    unit_id: str,
    link_types: Iterable[str] | None = None,
    min_score: float = 0.0,
    direction: str = "both",
) -> list[tuple[str, str, float]]:
    """(neighbour_id, link_type, score), best first.

    `direction="both"` by default: an audio segment reaches its slide along an
    outgoing edge, but a slide reaches the speech about it along an incoming one,
    and link-following retrieval needs both.

    Parallel edges are deduped to the best-scoring one per (neighbour, type): the
    graph keeps every cue, but a traversal wants each neighbour once.
    """
    if unit_id not in graph:
        return []
    wanted = set(link_types) if link_types else None
    out: list[tuple[str, str, float]] = []

    if direction in ("both", "out"):
        for _s, dst, data in graph.out_edges(unit_id, data=True):
            if (wanted is None or data["link_type"] in wanted) and data["score"] >= min_score:
                out.append((dst, data["link_type"], data["score"]))
    if direction in ("both", "in"):
        for src, _d, data in graph.in_edges(unit_id, data=True):
            if (wanted is None or data["link_type"] in wanted) and data["score"] >= min_score:
                out.append((src, data["link_type"], data["score"]))
    if direction not in ("both", "out", "in"):
        raise ValueError(f"unknown direction {direction!r}: use 'both', 'out' or 'in'")

    best: dict[tuple[str, str], float] = {}
    for node, link_type, score in out:
        key = (node, link_type)
        if score > best.get(key, float("-inf")):
            best[key] = score
    merged = [(node, link_type, score) for (node, link_type), score in best.items()]
    merged.sort(key=lambda t: (-t[2], t[0]))
    return merged


def subgraph_around(
    graph: nx.MultiDiGraph,
    unit_id: str,
    hops: int = 1,
    link_types: Iterable[str] | None = None,
    min_score: float = 0.0,
) -> nx.MultiDiGraph:
    """Everything within `hops` link-follows of `unit_id`, edges filtered the same way."""
    if unit_id not in graph:
        raise KeyError(f"{unit_id!r} is not in the graph")
    seen = {unit_id}
    frontier = [unit_id]
    for _ in range(max(0, hops)):
        nxt = []
        for node in frontier:
            for neighbour, _t, _s in neighbors(graph, node, link_types, min_score):
                if neighbour not in seen:
                    seen.add(neighbour)
                    nxt.append(neighbour)
        frontier = nxt
        if not frontier:
            break
    sub = graph.subgraph(seen).copy()
    stale = [(u, v, k) for u, v, k, d in sub.edges(keys=True, data=True)
             if (link_types and d["link_type"] not in set(link_types)) or d["score"] < min_score]
    sub.remove_edges_from(stale)
    return sub


def export_graphml(graph: nx.MultiDiGraph, path: str | Path) -> Path:
    """GraphML for Gephi/yEd/Cytoscape.

    GraphML holds only scalars, so any non-scalar attribute is stringified rather
    than allowed to raise halfway through writing the file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    clean = graph.copy()
    for _n, data in clean.nodes(data=True):
        for k, v in list(data.items()):
            if not isinstance(v, (str, int, float, bool)):
                data[k] = str(v)
    for _u, _v, data in clean.edges(data=True):
        for k, val in list(data.items()):
            if not isinstance(val, (str, int, float, bool)):
                data[k] = str(val)
    clean.graph.pop("dropped_links", None)
    nx.write_graphml(clean, str(path))
    return path


def summarise(graph: nx.MultiDiGraph) -> dict[str, dict[str, float]]:
    """Per-link-type count and mean score."""
    stats: dict[str, dict[str, float]] = {}
    for _u, _v, data in graph.edges(data=True):
        row = stats.setdefault(data["link_type"], {"count": 0, "total": 0.0})
        row["count"] += 1
        row["total"] += data["score"]
    for row in stats.values():
        row["mean_score"] = row["total"] / row["count"] if row["count"] else 0.0
        del row["total"]
    return stats
