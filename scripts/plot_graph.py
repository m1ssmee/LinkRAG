#!/usr/bin/env python3
"""Render the link neighbourhood around one unit to reports/.

    python scripts/plot_graph.py --unit hsieh:a30 --hops 2

matplotlib rather than pyvis: matplotlib is already a dependency for the alignment
heatmap, and a static PNG is what goes in the paper. GraphML is written alongside the
links for anyone who wants to explore interactively in Gephi.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from linkrag.core import Link, load_config, setup_logging
from linkrag.index import Index
from linkrag.link.graph import build_graph, neighbors, subgraph_around

EDGE_STYLE = {
    "audio_slide": ("#00E676", "-"),
    "figure_text": ("#FFD54F", "-"),
    "deictic": ("#4FC3F7", "--"),
    "same_topic": ("#BDBDBD", ":"),
}
NODE_COLOUR = {"audio": "#1565C0", "text": "#2E7D32", "figure": "#EF6C00", "table": "#6A1B9A"}


def load_links(path: str | Path) -> list[Link]:
    out = []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            out.append(Link(**json.loads(line)))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--unit", required=True, help="unit id at the centre, e.g. hsieh:a30")
    ap.add_argument("--hops", type=int, default=1)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--index", default=None)
    ap.add_argument("--links", default=None)
    ap.add_argument("--link-types", nargs="*", default=None)
    ap.add_argument("--min-score", type=float, default=0.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    setup_logging()
    cfg = load_config(args.config)
    index = Index.load(args.index or cfg["index"]["store_dir"])
    links = load_links(args.links or cfg["link"]["align"]["links_path"])
    graph = build_graph(list(index.units), links)

    if args.unit not in graph:
        raise SystemExit(f"{args.unit!r} not in the graph ({graph.number_of_nodes()} units)")

    sub = subgraph_around(graph, args.unit, hops=args.hops,
                          link_types=args.link_types, min_score=args.min_score)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import networkx as nx

    pos = nx.spring_layout(sub, seed=11, k=1.4 / max(1, sub.number_of_nodes() ** 0.5))
    fig, ax = plt.subplots(figsize=(11, 8))

    for modality, colour in NODE_COLOUR.items():
        nodes = [n for n, d in sub.nodes(data=True) if d.get("modality") == modality]
        if nodes:
            nx.draw_networkx_nodes(sub, pos, nodelist=nodes, node_color=colour,
                                   node_size=[900 if n == args.unit else 380 for n in nodes],
                                   edgecolors="white", linewidths=1.2, ax=ax, label=modality)
    for link_type, (colour, style) in EDGE_STYLE.items():
        edges = [(u, v) for u, v, d in sub.edges(data=True) if d["link_type"] == link_type]
        if edges:
            nx.draw_networkx_edges(sub, pos, edgelist=edges, edge_color=colour,
                                   style=style, width=1.5, alpha=0.85,
                                   connectionstyle="arc3,rad=0.08", ax=ax)
    labels = {n: (n if n == args.unit else n.split(":")[-1]) for n in sub.nodes()}
    nx.draw_networkx_labels(sub, pos, labels, font_size=7, ax=ax)

    handles = [plt.Line2D([], [], color=c, ls=s, label=t)
               for t, (c, s) in EDGE_STYLE.items()
               if any(d["link_type"] == t for _u, _v, d in sub.edges(data=True))]
    handles += [plt.Line2D([], [], marker="o", ls="", color=c, label=m)
                for m, c in NODE_COLOUR.items()
                if any(d.get("modality") == m for _n, d in sub.nodes(data=True))]
    ax.legend(handles=handles, loc="lower right", fontsize=8, framealpha=0.9)
    ax.set_title(f"Evidence links around {args.unit}  ({args.hops} hop(s), "
                 f"{sub.number_of_nodes()} units, {sub.number_of_edges()} links)")
    ax.axis("off")
    fig.tight_layout()

    out = Path(args.out or f"reports/graph_{args.unit.replace(':', '_')}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    print(f"wrote {out}")
    for neighbour, link_type, score in neighbors(graph, args.unit,
                                                 args.link_types, args.min_score):
        print(f"  {link_type:14} {score:.4f}  {neighbour}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
