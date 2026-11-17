#!/usr/bin/env python3
"""Hand queued LLM prompts to Claude subagents and load their replies into the reply cache.

    python scripts/agent_queue.py status
    python scripts/agent_queue.py batches --out data/processed/agent_queue/round1 [--max-chars 240000]
    python scripts/agent_queue.py import  --batches data/processed/agent_queue/round1
    python scripts/agent_queue.py clear              # after every slot is answered

A batch folder holds one subfolder per batch. In a ``b*`` batch each prompt is
``<i>.system.txt`` plus ``<i>.user.txt``. In a ``c*`` (shared-context) batch the system prompt
is ``system.txt`` and the user message of item i is ``context.txt`` followed by
``<i>.question.txt``. Each batch has an empty ``replies/`` folder. The subagent writes ``replies/<i>.txt``,
the raw text the model would have returned. ``manifest.json`` maps each ``<i>`` to its cache
slot. Import refuses a reply that is empty or still the placeholder. See
``linkrag.agent_queue`` for the full loop.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from linkrag.agent_queue import (PENDING, batches, context_batches, load_queue, queue_path,
                                 shared_context, write_reply)


def cmd_status(qpath: Path) -> int:
    rows = load_queue(qpath)
    by = Counter((r["model"], Path(r["cache_dir"]).name) for r in rows)
    chars = sum(len(r["system"]) + len(r["user"]) for r in rows)
    print(f"{len(rows)} prompts pending in {qpath} (~{chars // 4:,} prompt tokens)")
    for (model, cdir), n in sorted(by.items()):
        print(f"  {n:5d}  {model}  ({cdir})")
    return 0


def cmd_batches(qpath: Path, out: Path, max_chars: int, max_items: int,
                share_min_chars: int, share_max_items: int) -> int:
    rows = load_queue(qpath)
    if not rows:
        print("queue empty: nothing to batch")
        return 0
    shared, rest = (context_batches(rows, min_chars=share_min_chars, max_items=share_max_items)
                    if share_min_chars > 0 else ([], rows))
    groups = [(g, True) for g in shared] + [(g, False) for g in batches(rest, max_chars=max_chars,
                                                                        max_items=max_items)]
    out.mkdir(parents=True, exist_ok=True)
    for b, (group, is_shared) in enumerate(groups):
        d = out / f"{'c' if is_shared else 'b'}{b:03d}"
        (d / "replies").mkdir(parents=True, exist_ok=True)
        manifest = {}
        if is_shared:
            # user message of item N = context.txt + N.question.txt, exactly
            (d / "system.txt").write_text(group[0]["system"])
            (d / "context.txt").write_text(shared_context(group[0]["user"], share_min_chars)[0])
        for i, row in enumerate(group):
            if is_shared:
                (d / f"{i}.question.txt").write_text(shared_context(row["user"], share_min_chars)[1])
            else:
                (d / f"{i}.system.txt").write_text(row["system"])
                (d / f"{i}.user.txt").write_text(row["user"])
            manifest[str(i)] = {"cache_dir": row["cache_dir"], "stem": row["stem"], "model": row["model"]}
        (d / "manifest.json").write_text(json.dumps(manifest, indent=1))
    models = Counter((g[0]["model"], s) for g, s in groups)
    print(f"{len(rows)} prompts -> {len(groups)} batches in {out}  " +
          ", ".join(f"{m}{' shared-context' if s else ''}: {n}" for (m, s), n in sorted(models.items())))
    return 0


def cmd_import(folder: Path) -> int:
    written, missing, bad = 0, [], []
    for d in sorted(p for p in folder.iterdir() if (p / "manifest.json").exists()):
        manifest = json.loads((d / "manifest.json").read_text())
        for i, slot in manifest.items():
            reply = d / "replies" / f"{i}.txt"
            if not reply.exists():
                missing.append(f"{d.name}/{i}")
                continue
            text = reply.read_text()
            if not text.strip() or PENDING in text:
                bad.append(f"{d.name}/{i}")
                continue
            write_reply(Path(slot["cache_dir"]), slot["stem"], text)
            written += 1
    print(f"imported {written} replies; missing {len(missing)}; refused {len(bad)}")
    for name in (missing + bad)[:20]:
        print(f"  {name}")
    return 0 if not (missing or bad) else 3


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--queue", type=Path, default=None, help="default: $LINKRAG_AGENT_QUEUE or "
                    "data/processed/agent_queue/pending.jsonl")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    b = sub.add_parser("batches")
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--max-chars", type=int, default=240_000, help="prompt characters per batch (~60k tokens)")
    b.add_argument("--max-items", type=int, default=25)
    b.add_argument("--share-min-chars", type=int, default=20_000,
                   help="store a context this long once per batch (0 = off)")
    b.add_argument("--share-max-items", type=int, default=20)
    i = sub.add_parser("import")
    i.add_argument("--batches", type=Path, required=True)
    sub.add_parser("clear")
    args = ap.parse_args(argv)
    qpath = args.queue or queue_path()
    if args.cmd == "status":
        return cmd_status(qpath)
    if args.cmd == "batches":
        return cmd_batches(qpath, args.out, args.max_chars, args.max_items,
                           args.share_min_chars, args.share_max_items)
    if args.cmd == "import":
        return cmd_import(args.batches)
    left = load_queue(qpath)
    if left:
        print(f"refusing to clear: {len(left)} prompts still unanswered")
        return 3
    qpath.unlink(missing_ok=True)
    print(f"cleared {qpath}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
