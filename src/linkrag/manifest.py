"""Corpus manifest: what was indexed, and a hash that identifies it.

Gold sets are written against a specific corpus. When the corpus changes, the gold
can silently stop meaning what it meant -- adding the OSDI paper made Q3 answerable
from a source its gold locators do not mention, so the run scored 4/4 gold terms and
"no" gold coverage at the same time. Nothing in the output said the corpus had moved.

The manifest records the source files (with content hashes) and the unit counts they
produced. Its `hash` covers exactly that, so a gold file stamped with a hash can be
checked against the corpus it is being run on.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from linkrag.core import EvidenceUnit

MANIFEST_NAME = "manifest.json"


def file_digest(path: str | Path, chunk: int = 1 << 20) -> str:
    """sha256 of a file's bytes, streamed -- corpus media runs to tens of MB."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def build_manifest(paths: Sequence[str | Path], units: Sequence[EvidenceUnit]) -> dict[str, Any]:
    by_source: dict[str, Counter] = {}
    for unit in units:
        by_source.setdefault(Path(unit.source_file).name, Counter())[unit.modality] += 1

    files = []
    for path in sorted({str(p) for p in paths}):
        p = Path(path)
        if not p.exists():
            continue
        files.append({
            "name": p.name,
            "bytes": p.stat().st_size,
            "sha256": file_digest(p),
            "units": dict(sorted(by_source.get(p.name, Counter()).items())),
        })

    manifest: dict[str, Any] = {
        "files": files,
        "total_units": len(units),
        "units_by_modality": dict(sorted(Counter(u.modality for u in units).items())),
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    manifest["hash"] = manifest_hash(manifest)
    return manifest


def manifest_hash(manifest: dict[str, Any]) -> str:
    """Hash of the corpus content only.

    `created_utc` and any previous `hash` are excluded, so re-ingesting the same
    files gives the same hash and a gold stamp does not go stale on a rebuild.
    """
    payload = {k: v for k, v in manifest.items() if k not in ("created_utc", "hash")}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]


def write_manifest(manifest: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return path


def load_manifest(path: str | Path) -> dict[str, Any] | None:
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def check_gold_manifest(gold_hash: str | None, manifest: dict[str, Any] | None) -> str | None:
    """Warning string when a gold set does not match the corpus, else None.

    A warning, never an error: running old gold against a new corpus is a legitimate
    thing to do deliberately. It just must not happen silently.
    """
    if manifest is None:
        return "no corpus manifest found -- run scripts/ingest.py to write one"
    current = manifest.get("hash")
    if not gold_hash:
        return (f"gold set carries no manifest stamp; current corpus is {current}. "
                f"Results cannot be attributed to a known corpus.")
    if gold_hash != current:
        names = ", ".join(f["name"] for f in manifest.get("files", []))
        return (f"gold set was written against corpus {gold_hash}, but the current "
                f"corpus is {current} ({manifest.get('total_units')} units: {names}). "
                f"Gold coverage may be understated -- a newly added document can answer "
                f"a question its gold locators do not mention.")
    return None
