"""Link-following retrieval, and the iterative (P1) baseline it is measured against."""

from __future__ import annotations

import pytest

from linkrag.core import EvidenceUnit, Link, Location
from linkrag.index import build_index
from linkrag.link.graph import build_graph
from linkrag.retrieve.baseline import retrieve_scored
from linkrag.retrieve.iterative import _clean_query, retrieve_iterative
from linkrag.retrieve.linkrag import retrieve_linkrag

# The gold unit shares no vocabulary with the question: similarity cannot reach it,
# only the audio_slide link can. This is the whole claim of link-following.
QUESTION = "what did the speaker say about attention weights"
# Enough distractors that top-k is a real selection: with only a handful of units,
# top-5 would include GOLD simply because it covers most of the corpus.
CORPUS = [
    ("a1", "audio", "so the attention weights concentrate on the subject token here"),
    ("t1", "text", "attention weights discussion in prose"),
    ("t2", "text", "optimizer cosine schedule with warmup"),
    ("t3", "text", "dataset statistics and collection methodology"),
    ("t4", "text", "related work on convolutional architectures"),
    ("t5", "text", "evaluation protocol and accuracy targets"),
    ("t6", "text", "hardware configuration and throughput measurements"),
    ("t7", "text", "limitations and threats to validity"),
    ("t8", "text", "background on neural network inference cost"),
    ("t9", "text", "clustering of similar feature vectors"),
    ("t10", "text", "index construction and storage layout"),
    ("t11", "text", "query planning and execution order"),
    ("GOLD", "text", "zzz qqq vvv xxx"),   # deliberately unreachable by similarity
]


@pytest.fixture
def corpus(stub_encoder):
    units = []
    for i, (uid, modality, content) in enumerate(CORPUS):
        loc = (Location(start_s=0.0, end_s=30.0) if modality == "audio"
               else Location(page=i + 1))
        units.append(EvidenceUnit(id=uid, modality=modality, content=content,
                                  source_file="deck.pdf" if modality == "text" else "talk.mp3",
                                  location=loc))
    encode = stub_encoder([c for _, _, c in CORPUS] + [QUESTION])
    index = build_index(units, encoder=encode, embedding_model="stub")
    return index, encode, units


@pytest.fixture
def graph(corpus):
    _index, _encode, units = corpus
    return build_graph(units, [Link("a1", "GOLD", "audio_slide", 0.9)])


# ------------------------------------------------ the claim, stated as a test

def test_baseline_misses_a_gold_unit_only_reachable_by_link(corpus, graph) -> None:
    index, encode, _ = corpus
    base = [u.id for u, _ in retrieve_scored(QUESTION, index, encoder=encode, top_k=5)]
    assert "GOLD" not in base, "similarity must not reach GOLD, or the test proves nothing"
    assert "a1" in base, "the seed that links to GOLD must itself be retrievable"


def test_linkrag_retrieves_it_through_the_link(corpus, graph) -> None:
    index, encode, _ = corpus
    results = retrieve_linkrag(QUESTION, index, graph, encoder=encode,
                               k_seed=3, k_final=8, decay=0.5)
    ids = [r.id for r in results]
    assert "GOLD" in ids
    gold = next(r for r in results if r.id == "GOLD")
    assert gold.origin == "expanded"
    assert gold.via_seed == "a1" and gold.via_link == "audio_slide"
    assert gold.link_score == pytest.approx(0.9)
    assert "expanded from a1 via audio_slide" in gold.explain()


def test_baseline_mode_is_plain_top_k(corpus, graph) -> None:
    """The ablation must be exactly the baseline retriever, not a crippled variant."""
    index, encode, _ = corpus
    got = [r.id for r in retrieve_linkrag(QUESTION, index, graph, encoder=encode,
                                          mode="baseline", k_final=5)]
    want = [u.id for u, _ in retrieve_scored(QUESTION, index, encoder=encode, top_k=5)]
    assert got == want
    assert "GOLD" not in got


# ------------------------------------------------------------- filters/scoring

def test_link_type_filter_blocks_expansion(corpus, graph) -> None:
    index, encode, _ = corpus
    ids = [r.id for r in retrieve_linkrag(QUESTION, index, graph, encoder=encode,
                                          k_seed=3, link_types=["figure_text"])]
    assert "GOLD" not in ids, "audio_slide was excluded, so GOLD is unreachable"


def test_min_link_score_blocks_weak_links(corpus, graph) -> None:
    index, encode, _ = corpus
    assert "GOLD" not in [r.id for r in retrieve_linkrag(
        QUESTION, index, graph, encoder=encode, k_seed=3, min_link_score=0.95)]
    assert "GOLD" in [r.id for r in retrieve_linkrag(
        QUESTION, index, graph, encoder=encode, k_seed=3, min_link_score=0.5)]


def test_expanded_score_is_seed_times_link_times_decay(corpus, graph) -> None:
    index, encode, _ = corpus
    results = retrieve_linkrag(QUESTION, index, graph, encoder=encode, k_seed=3, decay=0.5)
    seed = next(r for r in results if r.id == "a1")
    gold = next(r for r in results if r.id == "GOLD")
    assert gold.score == pytest.approx(seed.score * 0.9 * 0.5)


def test_decay_zero_disables_expansion_by_score(corpus, graph) -> None:
    index, encode, _ = corpus
    results = retrieve_linkrag(QUESTION, index, graph, encoder=encode, k_seed=3, decay=0.0)
    gold = next((r for r in results if r.id == "GOLD"), None)
    assert gold is None or gold.score == pytest.approx(0.0)


def test_hops_zero_is_no_expansion(corpus, graph) -> None:
    index, encode, _ = corpus
    assert "GOLD" not in [r.id for r in retrieve_linkrag(
        QUESTION, index, graph, encoder=encode, k_seed=3, hops=0)]


def test_a_seed_is_never_demoted_to_expanded(corpus, stub_encoder) -> None:
    """a1 is a seed and also link-reachable from t1; it must stay a seed."""
    index, encode, units = corpus
    g = build_graph(units, [Link("t1", "a1", "figure_text", 0.9),
                            Link("a1", "GOLD", "audio_slide", 0.9)])
    results = retrieve_linkrag(QUESTION, index, g, encoder=encode, k_seed=3)
    assert next(r for r in results if r.id == "a1").origin == "seed"


def test_stale_link_to_an_unindexed_unit_is_ignored(corpus, stub_encoder) -> None:
    index, encode, units = corpus
    g = build_graph(units + [EvidenceUnit(id="ghost", modality="text", content="x",
                                          source_file="d.pdf")],
                    [Link("a1", "ghost", "audio_slide", 0.9)])
    ids = [r.id for r in retrieve_linkrag(QUESTION, index, g, encoder=encode, k_seed=3)]
    assert "ghost" not in ids, "a link into a unit the index lacks must not be returned"


def test_k_final_caps_the_set(corpus, graph) -> None:
    index, encode, _ = corpus
    assert len(retrieve_linkrag(QUESTION, index, graph, encoder=encode,
                                k_seed=3, k_final=4)) == 4


def test_results_are_sorted_and_deduped(corpus, graph) -> None:
    index, encode, _ = corpus
    results = retrieve_linkrag(QUESTION, index, graph, encoder=encode, k_seed=3)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)
    assert len({r.id for r in results}) == len(results)


def test_unknown_mode_raises(corpus, graph) -> None:
    index, encode, _ = corpus
    with pytest.raises(ValueError, match="unknown mode"):
        retrieve_linkrag(QUESTION, index, graph, encoder=encode, mode="magic")


# ------------------------------------------------------- iterative (P1) baseline

def test_clean_query_strips_model_preamble() -> None:
    assert _clean_query('"attention heatmap"') == "attention heatmap"
    assert _clean_query("Follow-up query: ingest cost") == "ingest cost"
    assert _clean_query("first line\nsecond line") == "first line"
    assert _clean_query("") == ""


def test_iterative_makes_one_llm_call_per_extra_round(corpus) -> None:
    index, encode, _ = corpus
    calls = []

    def complete(system, user):
        calls.append(user)
        return "zzz qqq vvv xxx"   # steers round 2 at GOLD

    result = retrieve_iterative(QUESTION, index, encoder=encode, complete=complete,
                                rounds=2, k_per_round=4, k_final=8)
    assert result.llm_calls == 1 and len(calls) == 1
    assert len(result.queries) == 2
    assert "GOLD" in result.ids, "the follow-up query should reach the missed unit"
    assert result.latency_s > 0


def test_iterative_without_a_completer_runs_one_round(corpus) -> None:
    """The harness must be exercisable without spending API calls."""
    index, encode, _ = corpus
    result = retrieve_iterative(QUESTION, index, encoder=encode, complete=None, rounds=2)
    assert result.llm_calls == 0 and len(result.queries) == 1


def test_iterative_stops_when_the_follow_up_repeats_the_question(corpus) -> None:
    """A round that re-asks the same thing is pure cost."""
    index, encode, _ = corpus
    result = retrieve_iterative(QUESTION, index, encoder=encode,
                                complete=lambda s, u: QUESTION, rounds=3, k_per_round=4)
    assert result.llm_calls == 1 and len(result.queries) == 1


def test_iterative_records_round_provenance(corpus) -> None:
    index, encode, _ = corpus
    result = retrieve_iterative(QUESTION, index, encoder=encode,
                                complete=lambda s, u: "zzz qqq vvv xxx",
                                rounds=2, k_per_round=3, k_final=8)
    assert {r.origin for r in result.units} <= {"round1", "round2"}
    assert next(r for r in result.units if r.id == "GOLD").origin == "round2"


def test_expansion_report_distinguishes_empty_graph_from_filtered_links(corpus, graph) -> None:
    """Zero expansion with seeds that have edges means filters ate everything;
    zero with no edges means the graph is empty or stale. Both make linkrag equal
    baseline, and they need different fixes."""
    from linkrag.link.graph import build_graph as _bg
    from linkrag.retrieve.linkrag import expansion_report

    index, encode, units = corpus
    results = retrieve_linkrag(QUESTION, index, graph, encoder=encode, k_seed=3)
    expanded, seeded = expansion_report(results, graph)
    assert expanded >= 1 and seeded >= 1

    empty = _bg(units, [])
    results = retrieve_linkrag(QUESTION, index, empty, encoder=encode, k_seed=3)
    assert expansion_report(results, empty) == (0, 0), "empty graph: nothing to expand"

    results = retrieve_linkrag(QUESTION, index, graph, encoder=encode, k_seed=3,
                               link_types=["figure_text"])
    expanded, seeded = expansion_report(results, graph)
    assert expanded == 0 and seeded >= 1, "filtered out, but the edges existed"


# ------------------------------------------------ seed normalisation / combo mode

def test_normalise_seeds_puts_seeds_and_links_on_one_scale(corpus, graph) -> None:
    """Raw RRF seeds (~0.03) against link weights (~0.9) leave expanded scores two
    orders of magnitude down, so expansion can only ever fill the tail."""
    index, encode, _ = corpus
    raw = retrieve_linkrag(QUESTION, index, graph, encoder=encode, k_seed=3)
    norm = retrieve_linkrag(QUESTION, index, graph, encoder=encode, k_seed=3,
                            normalise_seeds=True)
    assert max(r.score for r in raw if r.origin == "seed") < 0.5, "RRF scale"
    assert max(r.score for r in norm if r.origin == "seed") == pytest.approx(1.0)
    assert min(r.score for r in norm if r.origin == "seed") > 0.0


def test_normalisation_lets_an_expanded_unit_outrank_a_weak_seed(corpus, graph) -> None:
    index, encode, _ = corpus
    raw = retrieve_linkrag(QUESTION, index, graph, encoder=encode, k_seed=3, decay=1.0)
    norm = retrieve_linkrag(QUESTION, index, graph, encoder=encode, k_seed=3, decay=1.0,
                            normalise_seeds=True)
    rank = lambda rs: [r.origin for r in rs]
    assert rank(raw)[:3] == ["seed"] * 3, "unnormalised: seeds always come first"
    assert "expanded" in rank(norm)[:3], "normalised: expansion can beat a weak seed"


def test_seed_results_override_lets_another_retriever_supply_seeds(corpus, graph) -> None:
    index, encode, units = corpus
    gold_unit = next(u for u in units if u.id == "GOLD")
    a1 = next(u for u in units if u.id == "a1")
    results = retrieve_linkrag(QUESTION, index, graph, encoder=encode, k_seed=2,
                               seed_results=[(a1, 0.9), (gold_unit, 0.8)])
    assert {r.id for r in results if r.origin == "seed"} == {"a1", "GOLD"}


def test_linkrag_iter_seeds_from_the_iterative_retriever(corpus, graph) -> None:
    """Q3's failure mode: the a0->p1 edge exists but a0 is never a seed. A second
    query can supply the missing entry point."""
    from linkrag.retrieve.iterative import retrieve_linkrag_iter

    index, encode, _ = corpus
    calls = []

    def complete(system, user):
        calls.append(user)
        return "optimizer cosine schedule with warmup"

    results, it = retrieve_linkrag_iter(QUESTION, index, graph, encoder=encode,
                                        complete=complete, rounds=2, k_seed=3, k_final=8)
    assert it.llm_calls == 1 and len(calls) == 1
    assert any(r.origin == "expanded" for r in results), "expansion must still run"
    assert "t2" in {r.id for r in results}, "the follow-up query's hit must be seeded"
