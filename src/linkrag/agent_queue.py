"""Answer LLM calls with Claude subagents, through the reply cache.

Hosted credit is gone and the free tiers' daily caps are too slow for the XFile benchmark
(DESIGN.md priority (vi)). This backend lets subagents answer the exact prompts the
pipeline builds, without touching any script's logic:

1. **Collect.** A script runs with a backend whose provider is ``agent_queue``. On a cache
   miss, ``cached_completer`` calls ``defer``, which appends the exact (system, user) prompt
   and its cache slot to the queue file. It returns ``PENDING``, a placeholder that is never
   cached. Prompts that contain ``PENDING`` are not queued: they depend on a reply that does
   not exist yet (a judge prompt quoting a placeholder answer).
2. **Answer.** ``scripts/agent_queue.py batches`` splits the queue into batch folders. A
   subagent answers each prompt as the model would, one reply file per prompt.
3. **Import.** ``scripts/agent_queue.py import`` writes each reply into its cache slot. The
   script is run again. Dependent stages take one round each, until nothing is pending. The
   final run is then an ordinary cache replay ($0, repeatable).

A run that returned any ``PENDING`` is **not a result**. Scripts' reports from such a run
are discarded; ``pending_count`` says whether a run was clean.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

PENDING = "[[AGENT_QUEUE_PENDING]]"
DEFAULT_QUEUE = Path("data/processed/agent_queue/pending.jsonl")
_lock = threading.Lock()


def queue_path(llm_cfg: dict[str, Any] | None = None) -> Path:
    env = os.environ.get("LINKRAG_AGENT_QUEUE")
    return Path(env or (llm_cfg or {}).get("queue_file") or DEFAULT_QUEUE)


def queue_completer(llm_cfg: dict[str, Any]):
    """The inner completer for provider ``agent_queue``. Called directly, without the reply
    cache in front of it, it cannot know its cache slot, so it refuses."""
    qpath = queue_path(llm_cfg)
    model = str(llm_cfg.get("model"))
    usage = {"calls": 0, "cached_calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
             "backend": str(llm_cfg.get("backend") or "agent_queue"), "deferred": 0}

    def complete(system: str, user: str) -> str:
        raise RuntimeError("agent_queue backend needs cached_completer in front of it "
                           "(the reply cache is where subagent answers land)")

    def defer(cache_dir: Path, stem: str, system: str, user: str) -> str:
        with _lock:
            usage["deferred"] += 1
            complete.pending += 1                       # type: ignore[attr-defined]
        if PENDING in system or PENDING in user:
            return PENDING                               # depends on an unanswered reply
        row = {"cache_dir": str(cache_dir), "stem": stem, "model": model,
               "system": system, "user": user}
        with _lock:
            qpath.parent.mkdir(parents=True, exist_ok=True)
            with qpath.open("a") as fh:
                fh.write(json.dumps(row) + "\n")
        return PENDING

    complete.deferred = True        # type: ignore[attr-defined]
    complete.defer = defer          # type: ignore[attr-defined]
    complete.pending = 0            # type: ignore[attr-defined]
    complete.usage = usage          # type: ignore[attr-defined]
    complete.last_usage = {}        # type: ignore[attr-defined]
    complete.model = model          # type: ignore[attr-defined]
    return complete


def load_queue(path: Path) -> list[dict]:
    """Unique (cache_dir, stem) rows still missing a reply, first occurrence kept."""
    if not path.exists():
        return []
    seen, out = set(), []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        slot = (row["cache_dir"], row["stem"])
        if slot in seen or (Path(row["cache_dir"]) / f"{row['stem']}.txt").exists():
            continue
        seen.add(slot)
        out.append(row)
    return out


QUESTION_MARK = "\n\nQuestion: "


def shared_context(user: str, min_chars: int) -> tuple[str, str] | None:
    """(context, rest) when ``user`` is a long context followed by a question, else None.

    verify_gold's whole-source runs send one topic's entire transcript, deck or notes,
    followed by ``\\n\\nQuestion: ...``. Every question on that topic repeats the same tens
    of thousands of characters. Splitting at the last question marker lets a batch store the
    context once. ``context + rest`` is exactly the original user message.
    """
    i = user.rfind(QUESTION_MARK)
    if i < min_chars:
        return None
    return user[:i], user[i:]


def context_batches(rows: list[dict], *, min_chars: int, max_items: int) -> tuple[list[list[dict]], list[dict]]:
    """Group long shared-context prompts by (model, system, context). Returns (groups, rest).

    One prompt hash per batch still holds: each batch takes at most one row per hash.
    """
    groups: dict[tuple, list[list[dict]]] = {}
    rest = []
    for row in sorted(rows, key=lambda r: (r["model"], r["cache_dir"], r["stem"])):
        split = shared_context(row["user"], min_chars)
        if split is None:
            rest.append(row)
            continue
        key = (row["model"], row["system"], split[0])
        h = row["stem"].split(".")[0]
        bins = groups.setdefault(key, [])
        for b in bins:
            if len(b) < max_items and all(r["stem"].split(".")[0] != h for r in b):
                b.append(row)
                break
        else:
            bins.append([row])
    return [b for bins in groups.values() for b in bins], rest


def batches(rows: list[dict], *, max_chars: int, max_items: int) -> list[list[dict]]:
    """Split rows into batches under a size budget, keeping repeats of one prompt apart.

    Rows that differ only in the run index share the prompt hash (the stem's first field).
    A subagent that saw run 0 and run 1 of the same prompt would answer them alike, so
    repeats must not be independent samples from one context. Each batch therefore holds at
    most one row per prompt hash. A prompt bigger than ``max_chars`` gets a batch of its own.
    """
    out: list[list[dict]] = []
    sizes: list[int] = []
    hashes: list[set[str]] = []
    for row in sorted(rows, key=lambda r: (r["model"], r["cache_dir"], r["stem"])):
        h = row["stem"].split(".")[0]
        n = len(row["system"]) + len(row["user"])
        for i, b in enumerate(out):
            if (b[0]["model"] == row["model"] and h not in hashes[i] and len(b) < max_items
                    and sizes[i] + n <= max_chars):
                b.append(row)
                sizes[i] += n
                hashes[i].add(h)
                break
        else:
            out.append([row])
            sizes.append(n)
            hashes.append({h})
    return out


def write_reply(cache_dir: Path, stem: str, text: str) -> Path:
    """Store a subagent reply where ``cached_completer`` will find it. No usage file: hits
    are then reported as estimated tokens, which is true -- nothing was metered."""
    path = Path(cache_dir) / f"{stem}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path
