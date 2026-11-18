#!/usr/bin/env python3
"""Did each agent_queue subagent see every line of its prompt files?

    python scripts/agent_read_audit.py data/processed/agent_queue/<round> ~/.claude/projects/<proj>/<session>/subagents/workflows/wf_<id> [...]

Reads the subagent transcripts (Claude Code workflow folders). Counts only Read calls whose
result is file text, not an error; later workflow folders override earlier ones for the same
batch. Prints the Haiku batches whose context.txt / N.user.txt were not covered line by line.
Prompts read with cat are invisible to it and are listed as not covered. See DESIGN.md (o).
"""
import json, glob, sys
from pathlib import Path
def failed(c, err):
    t = c if isinstance(c, str) else " ".join(x.get("text", "") for x in c if isinstance(x, dict))
    return bool(err) or t.lstrip().startswith(("File content (", "<tool_use_error>", "Error", "<persisted-output>"))

if __name__ == "__main__":
    batch = sys.argv[1].rstrip("/") + "/"
    final = {}
    for wf in sys.argv[2:]:
        for p in glob.glob(f"{wf}/agent-*.jsonl"):
            calls, res = {}, {}
            for l in open(p):
                cont = json.loads(l).get("message", {}).get("content")
                if not isinstance(cont, list): continue
                for x in cont:
                    if x.get("type") == "tool_use" and x["name"] == "Read" and x["input"].get("file_path", "").startswith(batch) \
                            and x["input"]["file_path"].endswith(("context.txt", "user.txt")):
                        calls[x["id"]] = x["input"]
                    if x.get("type") == "tool_result": res[x["tool_use_id"]] = (x.get("content"), x.get("is_error"))
            cov = {}
            for cid, i in calls.items():
                c, err = res.get(cid, ("", True))
                if failed(c, err): continue
                o = i.get("offset") or 1
                cov.setdefault(i["file_path"], set()).update(range(o, o + (i.get("limit") or 2000)))
            for fp in {i["file_path"] for i in calls.values() if Path(i["file_path"]).exists()}:
                n = len(Path(fp).read_text().splitlines())
                final[fp[len(batch):].split("/")[0]] = all(k in cov.get(fp, set()) for k in range(1, n + 1))
    items = [d.name for d in Path(batch).iterdir() if (d / "manifest.json").exists()
             and "haiku" in next(iter(json.load(open(d / "manifest.json")).values()))["model"]]
    bad = sorted(n for n in items if not final.get(n, False))
    print(f"{len(items) - len(bad)}/{len(items)} fully read; not: {bad}")
