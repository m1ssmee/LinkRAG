#!/usr/bin/env python3
"""Ask a question against an ingested index.

    python scripts/ask.py "what did the lecturer say about attention?" --mode baseline
"""

from __future__ import annotations

import argparse
from pathlib import Path

from linkrag.core import load_config, setup_logging, stage_timer
from linkrag.eval import format_modality_distribution
from linkrag.costs import record_run
from linkrag.generate.answer import answer, cited_ids, http_completer
from linkrag.index import Index, default_encoder
from linkrag.link.align import load_links
from linkrag.manifest import MANIFEST_NAME, load_manifest
from linkrag.link.graph import build_graph
from linkrag.retrieve.baseline import retrieve_scored
from linkrag.retrieve.linkrag import RetrievedUnit, retrieve_linkrag
from linkrag.retrieve.rerank import METHODS, rerank, set_diagnostics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Question answering over an ingested corpus.")
    parser.add_argument("question")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--mode", choices=["baseline", "linkrag"], default="baseline")
    parser.add_argument("--index", default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--links", default=None, help="links.jsonl for --mode linkrag")
    parser.add_argument("--rerank", choices=list(METHODS), default=None,
                        help="override retrieve.rerank.method")
    parser.add_argument("--show-evidence", action="store_true")
    args = parser.parse_args(argv)

    setup_logging()
    cfg = load_config(args.config)
    index_dir = args.index or cfg["index"]["store_dir"]
    if not Path(index_dir).exists():
        parser.error(f"no index at {index_dir} -- run scripts/ingest.py first")

    with stage_timer("index.load", dir=index_dir) as t:
        index = Index.load(index_dir)
        t["units"] = len(index)

    encoder = default_encoder(index.embedding_model, cfg["device"], index.normalize)
    # Load the embedder outside the retrieval timer, or the first query reports
    # the model load as retrieval latency.
    with stage_timer("encoder.warmup", model=index.embedding_model):
        encoder([""])

    manifest_hash = (load_manifest(Path(index_dir).parent / MANIFEST_NAME) or {}).get("hash")
    lcfg = cfg["retrieve"]["linkrag"]
    rcfg = cfg["retrieve"]["rerank"]
    method = args.rerank or rcfg["method"]
    k = args.top_k or lcfg["k_final"]
    pool = max(int(rcfg.get("pool", k)), k) if method != "none" else k
    graph = None
    if args.mode == "linkrag":
        links_path = args.links or cfg["link"]["align"]["links_path"]
        if not Path(links_path).exists():
            parser.error(f"no links at {links_path} -- run scripts/build_links.py first")
        graph = build_graph(list(index.units),
                            load_links(links_path, expect_manifest=manifest_hash))
        results = retrieve_linkrag(
            args.question, index, graph, encoder=encoder, mode="linkrag",
            k_seed=lcfg["k_seed"], k_final=pool,
            hops=lcfg["hops"], link_types=lcfg["link_types"],
            min_link_score=lcfg["min_link_score"], decay=lcfg["decay"],
            candidates=cfg["retrieve"]["candidates"], rrf_k=cfg["retrieve"]["rrf_k"],
            normalise_seeds=lcfg.get("normalise_seeds", False),
            expansion=cfg["retrieve"].get("expansion", "additive"),
        )
    else:
        results = [
            RetrievedUnit(unit=u, score=s, origin="seed")
            for u, s in retrieve_scored(
                args.question, index, encoder=encoder, top_k=pool,
                candidates=cfg["retrieve"]["candidates"],
                rrf_k=cfg["retrieve"]["rrf_k"])
        ]
    results = rerank(results, k, method=method, index=index, graph=graph,
                     alpha=rcfg["alpha"], beta=rcfg["beta"], gamma=rcfg["gamma"],
                     mmr_lambda=rcfg["mmr_lambda"], question=args.question,
                     cross_encoder=rcfg.get("cross_encoder"), device=cfg["device"],
                     modality_gate=rcfg.get("modality_gate", False))
    retrieved = [(r.unit, r.score) for r in results]
    units = [r.unit for r in results]

    # Instrument #1 (DESIGN.md): always report what the retrieved set is made of.
    print(f"retrieved {len(units)} units — modality: {format_modality_distribution(units)}")

    if args.mode == "linkrag":
        n_expanded = sum(1 for r in results if r.origin == "expanded")
        print(f"  {len(results) - n_expanded} seed + {n_expanded} expanded via links")

    if args.show_evidence:
        print("--- evidence ---")
        for r in results:
            print(f"  {r.score:.4f}  [{r.unit.id}, {r.unit.modality}]  <{r.explain()}>")
            print(f"           {' '.join(r.unit.content.split())[:88]}")
        print()

    complete = http_completer(cfg["models"]["llm"])
    text = answer(args.question, units, cfg, mode=args.mode, complete=complete)
    print(text)

    valid = {u.id for u in units}
    cited = cited_ids(text)
    unknown = [c for c in cited if c not in valid]
    print(f"\ncited {len(cited)} of {len(units)} units" + (f"; NOT IN EVIDENCE: {unknown}" if unknown else ""))
    print("\n".join(record_run("scripts/ask.py", args.question[:60],
                              [(str(cfg["models"]["llm"].get("model")), complete.usage)],
                              cfg["models"].get("pricing"))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
