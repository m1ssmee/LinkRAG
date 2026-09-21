"""File-pair relatedness gate (shuffled-slide-order null), its composition with
per-segment abstention, the cross-document figure_text gate, and the retrieval
fallback when a pair is unrelated."""

from __future__ import annotations

import numpy as np

from linkrag.core import EvidenceUnit, Location
from linkrag.link.align import (Alignment, align_monotonic, build_links, path_score,
                                relatedness_gate)
from linkrag.link.figure_text import document_pair_gate, link_figures_to_text
from linkrag.link.graph import build_graph
from linkrag.retrieve.baseline import retrieve_scored
from linkrag.retrieve.linkrag import expansion_report, retrieve_linkrag

DP = lambda S: align_monotonic(S, jump_penalty=0.05, skip_penalty=0.02, max_back=2)


def _related_matrix(n=30, m=10, seed=0):
    """A monotone diagonal band: segment i talks about slide i*m//n."""
    rng = np.random.default_rng(seed)
    S = rng.random((n, m)) * 0.2
    for i in range(n):
        S[i, i * m // n] += 0.6
    return S


def _unrelated_matrix(n=30, m=10, seed=1):
    return np.random.default_rng(seed).random((n, m)) * 0.3


def test_gate_passes_related_and_rejects_unrelated():
    g = relatedness_gate(_related_matrix(), DP)
    assert g["related"] and g["z"] > 2.0 and g["score"] > g["null_mean"]
    u = relatedness_gate(_unrelated_matrix(), DP)
    assert not u["related"], u
    assert len(u["null_scores"]) == 5


def _units(n, m):
    audio = [EvidenceUnit(id=f"a{i}", modality="audio", content="x", source_file="talk.mp3",
                          location=Location(start_s=i * 10.0, end_s=i * 10.0 + 9)) for i in range(n)]
    slides = [EvidenceUnit(id=f"s{j}", modality="text", content="y", source_file="deck.pdf",
                           location=Location(page=j + 1)) for j in range(m)]
    return audio, slides


def test_gate_and_abstention_compose():
    """Related pair with two flat segments: gate passes, abstention drops exactly
    those two. Unrelated pair: gate rejects, abstention never gets to run, zero links."""
    S = _related_matrix()
    S[5, :] = 0.05          # nothing on screen is recognisable for segments 5 and 6
    S[6, :] = 0.05
    audio, slides = _units(*S.shape)
    aln = Alignment(path=DP(S), similarity=S, method="monotonic", total_score=0.0)
    links = build_links(audio, slides, aln, relatedness_z=2.0, decode=DP, min_segment_sim=0.3)
    assert build_links.gate["related"]
    assert len(links) == S.shape[0] - 2
    assert {l.src_id for l in links}.isdisjoint({"a5", "a6"})

    U = _unrelated_matrix()
    audio, slides = _units(*U.shape)
    aln = Alignment(path=DP(U), similarity=U, method="monotonic", total_score=0.0)
    assert build_links(audio, slides, aln, relatedness_z=2.0, decode=DP, min_segment_sim=0.3) == []
    assert build_links.gate["related"] is False
    # gate off, abstention on: the unrelated pair would still leak links -> the gate is load-bearing
    assert len(build_links(audio, slides, aln, relatedness_z=None, min_segment_sim=0.0)) == U.shape[0]


def test_path_score_and_objective():
    from linkrag.link.align import path_objective
    S = np.array([[0.2, 0.8], [0.6, 0.1]])
    assert abs(path_score(S, [1, 0]) - 0.7) < 1e-9
    # back-jump 1 -> 0 costs beta; objective = (0.8 + 0.6 - 0.15) / 2
    assert abs(path_objective(S, [1, 0], jump_penalty=0.05, skip_penalty=0.02, back_penalty=0.15) - 0.625) < 1e-9
    # forward jump 0 -> 3 over two skipped slides: lam*3 + sig*2
    S4 = np.zeros((2, 4)); S4[0, 0] = S4[1, 3] = 1.0
    assert abs(path_objective(S4, [0, 3], jump_penalty=0.1, skip_penalty=0.01, back_penalty=0.2) - (2 - 0.32) / 2) < 1e-9


# ---------------------------------------------------------------- figure_text cross-doc gate

def _bag_encoder(vocab):
    """Deterministic bag-of-words encoder: shared words -> cosine, disjoint -> 0."""
    idx = {w: i for i, w in enumerate(vocab)}

    def enc(texts):
        out = np.zeros((len(texts), len(vocab) + 1), dtype=np.float32)
        for r, t in enumerate(texts):
            for w in t.lower().split():
                if w in idx:
                    out[r, idx[w]] += 1
            out[r, -1] = 1e-3   # never a zero vector
        return out
    return enc


def test_synthetic_unrelated_deck_gets_no_cross_document_semantic_links():
    vocab = ("recall", "precision", "cnn", "video", "ingest", "query", "mitosis", "cell", "enzyme", "protein", "membrane", "dna")
    enc = _bag_encoder(vocab)
    figs = [EvidenceUnit(id=f"paper:p{i}:c0", modality="figure", content=c, source_file="paper.pdf",
                         location=Location(page=i))
            for i, c in enumerate(["recall precision cnn", "video ingest query", "cnn query recall"], start=1)]
    related = [EvidenceUnit(id=f"deck:p{i}:t0", modality="text", content=c, source_file="deck.pdf",
                            location=Location(page=i))
               for i, c in enumerate(["recall precision of the cnn", "ingest the video then query", "query recall"], start=1)]
    unrelated = [EvidenceUnit(id=f"bio:p{i}:t0", modality="text", content=c, source_file="bio.pdf",
                              location=Location(page=i))
                 for i, c in enumerate(["mitosis cell enzyme", "protein membrane dna", "enzyme dna cell membrane"], start=1)]
    assert document_pair_gate(figs, related, encoder=enc)["related"]
    assert not document_pair_gate(figs, unrelated, encoder=enc)["related"]
    links = link_figures_to_text(figs, related + unrelated, encoder=enc, threshold=0.0,
                                 weights={"reference": 0.3, "layout": 0.15, "page": 0.15, "dense": 0.25, "overlap": 0.15})
    dst_docs = {l.dst_id.split(":")[0] for l in links}
    assert "deck" in dst_docs and "bio" not in dst_docs
    assert link_figures_to_text.unrelated_pairs[0]["text_from"] == "bio.pdf"


# ---------------------------------------------------------------- retrieval fallback

def test_unrelated_files_fall_back_to_plain_hybrid_search():
    """An unrelated pair leaves the graph without audio_slide edges; link-following
    must then equal plain hybrid retrieval exactly, and expansion_report must say so."""
    from linkrag.index import build_index
    vocab = ("recall", "precision", "cnn", "video", "ingest", "query", "mitosis", "cell", "enzyme", "protein")
    enc = _bag_encoder(vocab)
    audio = [EvidenceUnit(id=f"talk:a{i}", modality="audio", content=c, source_file="talk.mp3",
                          location=Location(start_s=i * 10.0, end_s=i * 10.0 + 9))
             for i, c in enumerate(["recall precision cnn", "video ingest query", "cnn query recall precision"])]
    deck = [EvidenceUnit(id=f"bio:p{i}:t0", modality="text", content=c, source_file="bio.pdf",
                         location=Location(page=i))
            for i, c in enumerate(["mitosis cell enzyme", "protein enzyme cell", "cell protein"], start=1)]
    index = build_index(audio + deck, encoder=enc, embedding_model="bag", normalize=True)
    graph = build_graph(audio + deck, [])              # what build_links emits for an unrelated pair
    q = "recall of the cnn"
    plain = [u.id for u, _ in retrieve_scored(q, index, encoder=enc, top_k=3)]
    res = retrieve_linkrag(q, index, graph, encoder=enc, k_final=3, expansion="additive", normalise_seeds=True)
    assert [r.id for r in res][:3] == plain
    assert all(r.origin == "seed" for r in res)
    assert expansion_report(res, graph) == (0, 0)
