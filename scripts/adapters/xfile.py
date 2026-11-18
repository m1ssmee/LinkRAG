#!/usr/bin/env python3
"""XFile adapter: build the cross-file benchmark corpus (DESIGN.md priority (vi)).

    python scripts/adapters/xfile.py prepare            # MaViLS + notes -> data/raw/xfile/
    python scripts/adapters/xfile.py files              # corpus paths, one per line, for ingest.py

Each topic is three separately authored files:

* ``<topic>_talk.wav``: a **zero-byte stub**. The words come from the transcript MaViLS
  ships (``data/audioscripts/*.srt``, word-level faster-whisper output), frozen to
  ``<frozen_transcript_dir>/<topic>_talk.frozen.json``. Ingest finds a frozen transcript by
  file stem and never opens the audio (``linkrag.ingest.audio.frozen_transcript_for``), so
  nothing is re-transcribed. There is no media to cut, which is why ``configs/xfile.yaml``
  turns citation media off.
* ``<topic>_slides.pdf``: the lecture's deck, copied from MaViLS.
* ``<topic>_notes.pdf``: a notes, textbook or paper source by different authors, cut to the
  pages in ``configs/xfile_topics.yaml``. It is fetched by hand into ``data/raw/xfile/_sources/``.
  Nothing under ``data/raw/`` is committed (see ``data/raw/README.md``).

Basenames carry the topic, so they are unique across the pooled corpus: gold matching is by
basename (``linkrag.eval.metrics.matches_locator``).
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

TOPICS_FILE = Path("configs/xfile_topics.yaml")
MAVILS = Path("data/raw/mavils/data")
RAW = Path("data/raw/xfile")
SOURCES = RAW / "_sources"
FROZEN = Path("data/processed/xfile/transcripts")

# "[00:01:02.340 --> 00:01:02.900] word"; a piece with no space after "]" ("-based",
# ".99") continues the previous word.
LINE = re.compile(r"^\[(\d+):(\d+):(\d+(?:\.\d+)?) --> (\d+):(\d+):(\d+(?:\.\d+)?)\]( ?)(.*)$")


@dataclass
class Topic:
    name: str
    srt: str
    slides: str
    notes: str | None = None
    notes_pages: tuple[int, int] | None = None   # 1-based, inclusive

    def talk(self) -> Path:
        return RAW / self.name / f"{self.name}_talk.wav"

    def deck(self) -> Path:
        return RAW / self.name / f"{self.name}_slides.pdf"

    def notes_out(self) -> Path:
        return RAW / self.name / f"{self.name}_notes.pdf"


def load_topics(path: Path = TOPICS_FILE) -> list[Topic]:
    raw = yaml.safe_load(path.read_text()) or {}
    out = []
    for name, spec in (raw.get("topics") or {}).items():
        notes = spec.get("notes") or {}
        pages = notes.get("pages")
        out.append(Topic(name=name, srt=spec["srt"], slides=spec["slides"],
                         notes=notes.get("file"),
                         notes_pages=tuple(pages) if pages else None))
    return out


def _seconds(h: str, m: str, s: str) -> float:
    return int(h) * 3600 + int(m) * 60 + float(s)


def parse_word_srt(text: str) -> list[list]:
    """MaViLS's word-per-line transcript -> ``[[start, end, word], ...]``.

    A line whose text follows ``]`` with no space is the tail of the previous word, as
    whisper splits "high-quality" into "high" and "-quality". It is joined onto the previous
    word and extends that word's end time. Lines that are not timestamped words (blank
    lines) are skipped.
    """
    words: list[list] = []
    for line in text.splitlines():
        m = LINE.match(line.rstrip("\n"))
        if not m:
            continue
        start, end = _seconds(*m.group(1, 2, 3)), _seconds(*m.group(4, 5, 6))
        spaced, token = m.group(7), m.group(8).strip()
        if not token:
            continue
        if not spaced and words:
            words[-1][1] = end
            words[-1][2] += token
        else:
            words.append([start, end, token])
    return words


def freeze(topic: Topic, frozen_dir: Path = FROZEN, mavils: Path = MAVILS) -> Path:
    words = parse_word_srt((mavils / "audioscripts" / topic.srt).read_text())
    if not words:
        raise ValueError(f"{topic.srt}: no timestamped words parsed")
    out = frozen_dir / f"{topic.talk().stem}.frozen.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "audio": str(topic.talk()),
        "words": words,
        "asr_backend": "mavils_shipped",
        "model": "faster-whisper (MaViLS data/audioscripts, as shipped)",
        "initial_prompt": None,
        "source": f"{mavils / 'audioscripts' / topic.srt}",
    }))
    return out


def cut_pages(src: Path, dst: Path, pages: tuple[int, int] | None) -> int:
    import fitz  # PyMuPDF, already a pipeline dependency

    with fitz.open(src) as doc:
        if pages is None:
            shutil.copyfile(src, dst)
            return doc.page_count
        first, last = pages
        if not (1 <= first <= last <= doc.page_count):
            raise ValueError(f"{src.name}: pages {pages} outside 1..{doc.page_count}")
        with fitz.open() as out:
            out.insert_pdf(doc, from_page=first - 1, to_page=last - 1)
            out.save(dst)
        return last - first + 1


def prepare(topics: list[Topic]) -> list[dict]:
    rows = []
    for t in topics:
        (RAW / t.name).mkdir(parents=True, exist_ok=True)
        t.talk().touch()
        frozen = freeze(t)
        shutil.copyfile(MAVILS / "lectures" / t.slides, t.deck())
        notes_pages = None
        if t.notes:
            src = SOURCES / t.notes
            if not src.exists():
                print(f"  {t.name}: notes source missing: {src}", file=sys.stderr)
            else:
                notes_pages = cut_pages(src, t.notes_out(), t.notes_pages)
        words = len(json.loads(frozen.read_text())["words"])
        rows.append({"topic": t.name, "words": words, "notes_pages": notes_pages})
        print(f"  {t.name}: {words} words, notes pages: {notes_pages}")
    return rows


def corpus_files(topics: list[Topic]) -> list[Path]:
    files = []
    for t in topics:
        files += [t.deck(), t.talk()]
        if t.notes_out().exists():
            files.append(t.notes_out())
    return files


DECISION = "decision (accept/edit/reject)"


def apply_review(proposed: Path, review: Path, out: Path) -> dict[str, int]:
    """Human review sheet -> the proposal verify_gold reads.

    Every row needs a decision. ``accept`` keeps the draft. ``edit`` keeps it with the
    reviewer's question and/or answer (blank keeps the draft's). ``reject`` drops it. The
    question ``kind`` (bridge / conjunction / deictic / control) and the reviewer ride along,
    so results can be split by kind later.
    """
    import csv

    raw = [json.loads(l) for l in proposed.read_text().splitlines() if l.strip()]
    meta = next((r["_meta"] for r in raw if "_meta" in r), {})
    drafts = {r["qid"]: r for r in raw if "_meta" not in r}
    sheet = list(csv.DictReader(review.open(newline="")))
    undecided = [r["qid"] for r in sheet if r[DECISION].strip().lower() not in ("accept", "edit", "reject")]
    if undecided:
        raise SystemExit(f"{len(undecided)} rows have no decision yet: {', '.join(undecided[:10])}")
    if missing := set(drafts) - {r["qid"] for r in sheet}:
        raise SystemExit(f"review sheet lacks {sorted(missing)}")
    counts = {"accept": 0, "edit": 0, "reject": 0}
    kept = []
    for r in sheet:
        d = r[DECISION].strip().lower()
        counts[d] += 1
        if d == "reject":
            continue
        row = dict(drafts[r["qid"]])
        if d == "edit":
            row["question"] = r["edited_question"].strip() or row["question"]
            row["expected_answer"] = r["edited_answer"].strip() or row["expected_answer"]
        row.update(kind=r["kind"], reviewed_by=r["reviewer"], review=d)
        kept.append(row)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        fh.write(json.dumps({"_meta": {**meta, "status": "human-reviewed", "review_sheet": str(review),
                                       "review_counts": counts}}) + "\n")
        for row in kept:
            fh.write(json.dumps(row) + "\n")
    return counts


def paired(results: Path, a: str, b: str, *, types: set[str] | None = None, metric: str = "correct") -> dict:
    """Cell a minus cell b ("mode|rerank"), per question averaged over runs, with a 95 % interval
    from resampling whole topics (the sampled unit, DESIGN.md reporting convention)."""
    sys.path.insert(0, str(Path(__file__).parent))
    from lectqa.recheck import by_video      # cluster bootstrap: resample groups, pool their questions

    d = json.loads(results.read_text())
    qs = [q for q in d["questions"] if types is None or q["type"] in types]
    m = lambda q, cell: sum(map(float, d[metric][f"{q['qid']}|{cell}"])) / len(d[metric][f"{q['qid']}|{cell}"])
    diffs = [m(q, a) - m(q, b) for q in qs]
    topics = [q["topic"] for q in qs]
    lo, hi = by_video(diffs, topics, seed=20261118)
    mean = sum(diffs) / len(diffs)
    return {"n": len(qs), "topics": len(set(topics)), "a": sum(m(q, a) for q in qs) / len(qs),
            "b": sum(m(q, b) for q in qs) / len(qs), "diff": mean, "lo": lo, "hi": hi,
            "verdict": "no measurable difference" if lo <= 0 <= hi else ("a higher" if lo > 0 else "b higher")}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare", help="freeze transcripts, copy decks, cut notes")
    sub.add_parser("files", help="print corpus file paths for scripts/ingest.py")
    ar = sub.add_parser("apply-review", help="filled review sheet -> reviewed proposal for verify_gold")
    ar.add_argument("--proposed", type=Path, default=Path("tests/regression/xfile_drafts.jsonl"))
    ar.add_argument("--review", type=Path, default=Path("reports/xfile_review.csv"))
    ar.add_argument("--out", type=Path, default=Path("tests/regression/xfile_proposed.jsonl"))
    bp = sub.add_parser("paired", help="paired difference of two compare_retrieval cells, bootstrap over topics")
    bp.add_argument("results", type=Path, help="<out>.questions.json written by compare_retrieval.py")
    bp.add_argument("--a", default="linkrag|complementarity")
    bp.add_argument("--b", default="baseline|complementarity")
    bp.add_argument("--metric", choices=["correct", "recall"], default="correct")
    parser.add_argument("--topics", type=Path, default=TOPICS_FILE)
    args = parser.parse_args(argv)
    if args.cmd == "apply-review":
        counts = apply_review(args.proposed, args.review, args.out)
        print(f"{counts} -> {args.out}")
        return 0
    if args.cmd == "paired":
        cross = {"cross_modal_split", "cross_modal_deictic"}
        single = {"slides_only", "audio_only", "paper_only"}
        print(f"| questions | n | topics | {args.a} | {args.b} | difference [95 % CI, topics resampled] | |")
        print("|---|---:|---:|---:|---:|---|---|")
        for name, types in (("cross-file", cross), ("single-source", single), ("all", None)):
            r = paired(args.results, args.a, args.b, types=types, metric=args.metric)
            print(f"| {name} | {r['n']} | {r['topics']} | {r['a']:.1%} | {r['b']:.1%} | "
                  f"{r['diff']:+.1%} [{r['lo']:+.1%}, {r['hi']:+.1%}] | {r['verdict']} |")
        return 0
    topics = load_topics(args.topics)
    if args.cmd == "prepare":
        prepare(topics)
    else:
        print("\n".join(str(p) for p in corpus_files(topics)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
