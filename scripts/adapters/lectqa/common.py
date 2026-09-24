"""Shared LectQA-Vid data access: paths, QA files, video ids, T1's published numbers,
the strict gold-timestamp parser, prompts and evidence formatting."""

from __future__ import annotations

import collections
import json
import re
from pathlib import Path


from linkrag.core import EvidenceUnit, Link
from linkrag.retrieve.linkrag import RetrievedUnit



RAW = Path("data/raw/lectqa_vid")


PROCESSED = Path("data/processed/lectqa_vid")


LEVELS = ("simple", "hard", "very hard")


# Their published numbers, all in %: CMC 88(2) art. 96, full-text HTML, Tables 4-6 (images),
# read 2026-09-23. Corrected then: earlier values here (sim 0.71, MCQ 56.30 %, and most
# per-difficulty cells) did not match the published tables.
THEIRS = {
    "open": {  # Table 4, "Temporal-Aware RAG (Ours)"
        "simple":    {"f1": 29.47, "sim": 77.23, "bleu": 8.61, "meteor": 38.15, "rouge1": 36.82},
        "hard":      {"f1": 24.38, "sim": 74.56, "bleu": 5.29, "meteor": 34.71, "rouge1": 30.94},
        "very hard": {"f1": 16.72, "sim": 71.48, "bleu": 2.83, "meteor": 24.36, "rouge1": 22.51},
        "overall":   {"f1": 23.52, "sim": 74.42, "bleu": 5.58, "meteor": 32.41, "rouge1": 29.76},
    },
    "mcq": {"simple": 57.29, "hard": 52.34, "very hard": 50.67, "overall": 53.43},  # Table 5, ACC
    # Table 6, "Multimodal RAG (ASR + captions, no timestamps)" -- their no-temporal baseline
    "open_no_timestamps": {"overall": {"f1": 14.80, "sim": 61.00, "bleu": 2.95, "meteor": 21.65}},
}


def video_links() -> dict[str, str]:
    import docx
    doc = docx.Document(RAW / "video_links.docx")
    out = {}
    for row in doc.tables[0].rows[1:]:
        vid, url = (c.text.strip() for c in row.cells[:2])
        out[vid.replace(" ", "_").rstrip(".")] = url      # the docx labels one row "video_94."
    return out


def load_qa() -> dict[str, list[dict]]:
    mcq = json.loads((RAW / "mcq_questions.json").read_text())
    opn = json.loads((RAW / "open_ended_questions.json").read_text())
    out: dict[str, list[dict]] = collections.defaultdict(list)
    for vid, qs in mcq.items():
        for q in qs:
            out[vid].append({**q, "kind": "mcq", "level": q["level"].replace("_", " ")})
    for vid, qs in opn.items():
        for q in qs:
            out[vid].append({**q, "kind": "open", "level": q["level"].replace("_", " ")})
    return out


def video_ids(n: int | None, only: str | None) -> list[str]:
    ids = [f"video_{i}" for i in range(1, 101)]
    if only:
        ids = [v for v in ids if v in set(only.split(","))]
    return ids[:n] if n else ids


def temporal_links(units: list[EvidenceUnit]) -> list[Link]:
    """audio segment <-> every frame on screen during it. This is T1's own signal;
    on a single video it is the only alignment there is."""
    audio = [u for u in units if u.modality == "audio"]
    frames = [u for u in units if u.modality == "figure"]
    links = []
    for a in audio:
        for f in frames:
            if f.location.start_s < a.location.end_s and f.location.end_s > a.location.start_s:
                overlap = min(a.location.end_s, f.location.end_s) - max(a.location.start_s, f.location.start_s)
                links.append(Link(a.id, f.id, "audio_slide", round(min(1.0, overlap / max(1.0, f.location.end_s - f.location.start_s)), 4)))
    return links


MCQ_SYSTEM = ("You answer a multiple-choice question about a lecture video using ONLY the evidence. "
              "Reply with the single letter of the best option and nothing else.")


OPEN_SYSTEM = ("You answer a question about a lecture video using ONLY the evidence. Answer in one or "
               "two sentences. If the evidence does not contain the answer, say: NOT ANSWERABLE.")


def evidence_block(results: list[RetrievedUnit]) -> str:
    lines = []
    for r in results:
        u = r.unit
        tag = f"[{u.modality.upper()} {int(u.location.start_s // 60)}:{int(u.location.start_s % 60):02d}]"
        lines.append(f"{tag} {' '.join(u.content.split()) or '(no text on frame)'}")
    return "\n".join(lines)


def mcq_choice(reply: str, options: list[str]) -> int | None:
    m = re.match(r"\s*\(?([A-Da-d])\)?[\s.:)]*", reply)
    if m:
        return "ABCD".index(m.group(1).upper())
    low = reply.strip().lower()
    for i, o in enumerate(options):
        if low == o.strip().lower():
            return i
    return None


def strict_seconds(t: str) -> float | None:
    """A timestamp only when its format is unambiguous: HH:MM:SS or MM:SS with minute and
    second fields below 60, or plain seconds. The published files also contain SS:cc
    ('00:12:70'), SSS:cc ('258:36') and M:SSS:cc ('01:91:60') stamps whose meaning differs
    between videos, so those are not guessed at."""
    parts = str(t).strip().split(":")
    try:
        vals = [float(x) for x in parts]
    except ValueError:
        return None
    if len(vals) == 1:
        return vals[0]
    if len(vals) > 3 or any(v >= 60 for v in vals[1:]) or (len(vals) == 3 and vals[0] >= 24):
        return None
    while len(vals) < 3:
        vals.insert(0, 0.0)
    return vals[0] * 3600 + vals[1] * 60 + vals[2]


_DURATION: dict[str, float] = {}


def gold_interval(vid: str, q: dict) -> tuple[tuple[float, float] | None, str]:
    """(interval, reason). None when the stamp is ambiguous or outside the fetched video."""
    if vid not in _DURATION:
        import av
        with av.open(str(RAW / "videos" / f"{vid}.m4a")) as c:
            _DURATION[vid] = float(c.duration) / 1e6
    g0, g1 = strict_seconds(q["timestamp_start"]), strict_seconds(q["timestamp_end"])
    if g0 is None or g1 is None:
        return None, "ambiguous format"
    if g1 < g0:
        return None, "end before start"
    if g1 > _DURATION[vid] + 2.0:
        return None, "beyond the fetched video's end"
    return (g0, g1 if g1 > g0 else g0 + 1.0), "ok"


def iou(a0, a1, b0, b1) -> float:
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    return inter / union if union > 0 else 0.0
