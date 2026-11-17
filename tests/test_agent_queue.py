"""agent_queue: subagent answers reach scripts through the ordinary reply cache."""

from __future__ import annotations

import importlib.util
import json
import sys

import pytest

from linkrag.agent_queue import PENDING, batches, load_queue, queue_completer
from linkrag.costs import cached_completer
from linkrag.generate.answer import http_completer

spec = importlib.util.spec_from_file_location("agent_queue_cli", "scripts/agent_queue.py")
cli = importlib.util.module_from_spec(spec)
sys.modules["agent_queue_cli"] = cli
spec.loader.exec_module(cli)


@pytest.fixture
def queue(tmp_path, monkeypatch):
    q = tmp_path / "pending.jsonl"
    monkeypatch.setenv("LINKRAG_AGENT_QUEUE", str(q))
    return q


def test_http_completer_routes_agent_queue_without_a_key(queue):
    inner = http_completer({"provider": "agent_queue", "model": "claude-haiku-5-5",
                            "api_key_env": "SURELY_UNSET_KEY_XYZ"})
    assert inner.deferred
    with pytest.raises(RuntimeError):            # its cache slot is unknown without the cache
        inner("s", "u")


def test_round_trip_collect_batch_import_replay(tmp_path, queue):
    cache = tmp_path / "cache"
    llm = cached_completer(queue_completer({"model": "m"}), cache)
    # three repeats of one prompt plus one other prompt
    assert [llm("sys", "q1") for _ in range(3)] + [llm("sys", "q2")] == [PENDING] * 4
    # a judge prompt quoting a placeholder depends on an unanswered reply: not queued
    assert llm("judge", f"answer: {PENDING}") == PENDING
    assert not list(cache.glob("*.txt"))          # placeholders are never cached
    rows = load_queue(queue)
    assert len(rows) == 4

    out = tmp_path / "round1"
    assert cli.main(["batches", "--out", str(out)]) == 0
    dirs = sorted(out.iterdir())
    # repeats of q1 can't share a batch: each batch has one row per prompt hash
    assert len(dirs) == 3
    for d in dirs:
        manifest = json.loads((d / "manifest.json").read_text())
        for i in manifest:
            user = (d / f"{i}.user.txt").read_text()
            (d / "replies" / f"{i}.txt").write_text(f"reply to {user} in {d.name}")
    assert cli.main(["import", "--batches", str(out)]) == 0
    assert load_queue(queue) == []

    replay = cached_completer(queue_completer({"model": "m"}), cache, cache_only=True)
    got = [replay("sys", "q1") for _ in range(3)] + [replay("sys", "q2")]
    assert all(g.startswith("reply to q") for g in got)
    assert len(set(got[:3])) == 3                 # three independent samples, one per slot


def test_import_refuses_placeholder_and_reports_missing(tmp_path, queue):
    llm = cached_completer(queue_completer({"model": "m"}), tmp_path / "cache")
    llm("s", "a"), llm("s", "b")
    out = tmp_path / "r"
    cli.main(["batches", "--out", str(out)])
    (d,) = list(out.iterdir())
    (d / "replies" / "0.txt").write_text(PENDING)
    assert cli.main(["import", "--batches", str(out)]) == 3
    assert len(load_queue(queue)) == 2


def test_batches_respect_size_budget():
    rows = [{"model": "m", "cache_dir": "c", "stem": f"h{i}.0", "system": "", "user": "x" * 100}
            for i in range(5)]
    assert [len(b) for b in batches(rows, max_chars=250, max_items=10)] == [2, 2, 1]
    big = [{"model": "m", "cache_dir": "c", "stem": "hbig.0", "system": "", "user": "x" * 999}]
    assert batches(big, max_chars=250, max_items=10) == [big]


def test_shared_context_batches_rebuild_the_exact_prompt(tmp_path, queue):
    from linkrag.agent_queue import shared_context
    llm = cached_completer(queue_completer({"model": "m"}), tmp_path / "cache")
    ctx = "Context (transcript):\n\n" + "words " * 5000
    users = [f"{ctx}\n\nQuestion: q{i}?\n\nAnswer:" for i in range(3)]
    for u in users:
        llm("sys", u)
    llm("sys", "short prompt")
    out = tmp_path / "r"
    assert cli.main(["batches", "--out", str(out), "--share-min-chars", "1000"]) == 0
    shared = [d for d in out.iterdir() if d.name.startswith("c")]
    assert len(shared) == 1 and len(list(out.iterdir())) == 2
    d = shared[0]
    manifest = json.loads((d / "manifest.json").read_text())
    rebuilt = {(d / "context.txt").read_text() + (d / f"{i}.question.txt").read_text() for i in manifest}
    assert rebuilt == set(users) and (d / "system.txt").read_text() == "sys"
    assert shared_context("no marker here", 10) is None
