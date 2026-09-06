"""DOCX -> text EvidenceUnits via python-docx.

baseline: paragraphs chunked by size, nothing else.
linkrag:  same units; the linker relates them to figures and audio later.
"""

from __future__ import annotations

from pathlib import Path

import docx

from linkrag.core import EvidenceUnit, Location, stage_timer
from linkrag.ingest.pdf import WORDS_PER_TOKEN


def ingest_docx(
    path: str | Path,
    *,
    chunk_tokens: int = 300,
    overlap_tokens: int = 60,
) -> list[EvidenceUnit]:
    """DOCX has no page geometry until it is rendered, so units carry no
    page/bbox -- Location stays empty and citations fall back to the filename.

    Note: paragraph text only. Tables and headers are skipped; add them when
    a corpus actually needs them.
    """
    path = Path(path)
    with stage_timer("ingest.docx", file=path.name) as t:
        document = docx.Document(str(path))
        words = [w for p in document.paragraphs for w in p.text.split()]

        size = max(1, int(chunk_tokens * WORDS_PER_TOKEN))
        overlap = min(int(overlap_tokens * WORDS_PER_TOKEN), size - 1)
        step = size - overlap

        units = []
        for start in range(0, len(words), step):
            window = words[start : start + size]
            if not window or (start > 0 and len(window) <= overlap):
                break
            units.append(
                EvidenceUnit(
                    id=f"{path.stem}:t{len(units)}",
                    modality="text",
                    content=" ".join(window),
                    source_file=str(path),
                    location=Location(),
                )
            )
        t["units"] = len(units)
    return units
