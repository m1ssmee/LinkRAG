#!/usr/bin/env python3
"""Link types in every stored links file, against the closed set (`linkrag.core.LINK_TYPES`).

    python scripts/check_link_types.py               # data/processed and results
    python scripts/check_link_types.py some/dir ...

Prints the counts per type per file and exits 1 when a file holds a type outside the set.
Run it before switching validation on in `Link` / `load_links`."""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

from linkrag.core import LINK_TYPES, check_link_type


def main(argv: list[str] | None = None) -> int:
    roots = [Path(a) for a in (argv if argv is not None else sys.argv[1:])] or [Path("data/processed"), Path("results")]
    files = sorted(p for root in roots for p in root.rglob("*.jsonl") if "link" in p.name)
    bad = 0
    for p in files:
        counts = collections.Counter(r.get("link_type") for r in map(json.loads, filter(str.strip, p.read_text().splitlines()))
                                     if "_meta" not in r)
        retired = {}
        for t, n in counts.items():
            try:
                check_link_type(str(t))
            except ValueError:
                retired[t] = n
        bad += bool(retired)
        print(f"{p}: {dict(sorted(counts.items()))}" + (f"  OUTSIDE THE SET: {retired}" if retired else ""))
    print(f"{len(files)} files, {bad} with a type outside {sorted(LINK_TYPES)}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
