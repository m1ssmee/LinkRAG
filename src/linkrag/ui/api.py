"""Lectern -- the web UI's HTTP API.

Wraps the pipeline and decides nothing the pipeline decides. Every step is the library's own
call with configs/default.yaml's values, made the way the scripts make it:

  ingest    linkrag.ingest.ingest_files                            (scripts/ingest.py)
  index     linkrag.index.build_index
  links     linkrag.link.pipeline.link_corpus over scripts/build_links.split_units
  retrieve  linkrag.retrieve.iterative.retrieve_pool + retrieve.rerank.rerank
                                                                   (scripts/compare_retrieval.py)
  answer    generate.answer.answer_json + generate.verify.verify_answer  (scripts/ask.py)

What the UI adds is a say over cross-file links, one file pair at a time. Each pair shows its
file-pair gate: audio x deck from `link_corpus`'s own gate (`align.relatedness_gate`), run on
that recording and that deck; document x document from `figure_text.document_pair_gate`, in
the directions and with the arguments `link_figures_to_text` uses. The links themselves are
the pipeline's single run over the whole lecture ("auto"). A pair can be forced related (its
cross-file links from the same run with the gate off are added) or marked unrelated (every
link between the two files is dropped). For a frozen corpus the links are the stored ones and
the gates are computed by the current code and config, for display (cached under the workdir,
keyed by both).

The sample lecture loads as its recording and slide deck; its notes (pilot01's paper) are an
option. Without them the two files are linked again by the pipeline (`sample_lecture`), from the
frozen units and vectors: no re-transcription, no re-embedding.

Each answer also carries `linked_context`: the cited units' 1-hop neighbours in the link graph
that are not cited, as context beside the answer, never as citations (`linked_context`).

An answer is `linkrag` mode with the configured reranker (complementarity over a pool of 20).
"Compare with baseline" adds `baseline` mode beside it -- plain top-k, no links, no reranker --
with the same config, answerer and judge. The two differ in links *and* reranking: the full
system against plain retrieval. `missed_evidence` (tagged "not retrieved by baseline" on the page)
lists units in Lectern's evidence that the baseline's does not hold. There is no gold behind it:
it says the baseline did not retrieve a unit, not that the unit was needed.
Nothing the UI shows is reportable (DESIGN.md reporting convention): mock verdicts are true by
construction, and no timing here meets the measurement rules.

Spend follows DESIGN.md's cost policy: the cheap tier by default, the strong model only with
--demo-strong, and --max-cost caps the answerer and the judge together (default 0: nothing
that bills is sent). `--answerer mock` is an offline stand-in for both, for tests and for
building without spend.

    python -m linkrag.ui.api --sample data/processed --answerer mock
    python -m linkrag.ui.api --sample data/processed --max-cost 0.50
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import importlib.util
import json
import logging
import math
import os
import queue
import re
import shutil
import sys
import tempfile
import threading
import time
from collections import Counter, OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Literal
from urllib.parse import quote

import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import linkrag
import linkrag.link.align as align_module
import linkrag.link.figure_text as figure_text_module
import linkrag.link.pipeline as pipeline_module
from linkrag.core import EvidenceUnit, Link, load_config, refuse_strong_in_batch, set_max_cost, setup_logging
from linkrag.costs import LEDGER, BillingRefused, cached_completer, price_for, record_run, usage_cost
from linkrag.eval.verify_gold import entailment_opts, span_in_text, verifier_label
from linkrag.generate.answer import MissingApiKey, answer_json, http_completer, judge_completer
from linkrag.generate.citations import clip_audio
from linkrag.generate.verify import ABSTENTION, verify_answer
from linkrag.index import Encoder, Index, build_index, default_encoder, embeddable_text, tokenize
from linkrag.ingest import SUFFIXES, ingest_files
from linkrag.link.align import load_links, save_links
from linkrag.link.figure_text import document_pair_gate
from linkrag.link.graph import build_graph, neighbors
from linkrag.link.pipeline import link_corpus
from linkrag.link.same_slide import is_slide_deck
from linkrag.manifest import MANIFEST_NAME, load_manifest, manifest_hash, write_manifest
from linkrag.retrieve.iterative import retrieve_pool
from linkrag.retrieve.linkrag import RetrievedUnit
from linkrag.retrieve.rerank import rerank

log = logging.getLogger("linkrag")

REPO = Path(linkrag.__file__).resolve().parents[2]
WEB = Path(__file__).resolve().parent / "web" / "dist"


def _script(name: str):
    """A module from scripts/, loaded the way the tests load one: the UI splits a corpus with
    scripts/build_links.py's own `split_units`, so there is one split, not two."""
    spec = importlib.util.spec_from_file_location(f"lectern_{name}", REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


split_units = _script("build_links").split_units

STRONG_MODEL = "gpt-5.4-2026-03-05"   # DESIGN.md cost policy: the strong model, for the demo only
# Note: a display rule; it changes no link. Linked audio x deck pairs whose z is below 3.51
# are shown as "unsure": 3.51 is the highest z any of 380 unrelated MaViLS pairs reached
# (results/external/mavils_gate_v2.md) -- topical neighbours, which the gate cannot tell from
# the right deck (DESIGN.md finding 7). The document-pair gate has no measured band, so its
# pairs are never shown as unsure. Config `ui.unsure_below_z` overrides it.
UNSURE_BELOW_Z = 3.51
# Note: sessions are an in-memory LRU in one process; a shared store if the Space scales out.
MAX_SESSIONS = 32
MAX_SESSION_BYTES = 500 << 20  # uploads per session; /tmp is the only writable disk on a Space
EXCERPT_CHARS = 360
LINKED_ROWS = 4                # the rail's "Linked to this evidence" rows (as requested); config ui.linked_rows
PEAK_BINS = 1200               # waveform resolution; the player resamples to the bars that fit
PAGE_PX = 720                  # page renders: twice the 360 px evidence rail
# A figure in the notes is cropped to its own region (padded, at least FIGURE_MIN_PT wide) so it
# stays legible in the rail. Text keeps its whole page: a chunk's box is the union of its blocks,
# which on a two-column page spans both columns, so a partial crop could show the wrong part.
FIGURE_PAD_PT, FIGURE_MIN_PT = 18.0, 240.0
HIGHLIGHT = {"slides": (0xB4, 0x53, 0x09), "notes": (0x40, 0x40, 0x40)}
IMAGE_SUFFIXES = {s for s, kind in SUFFIXES.items() if kind == "image"}
HINDI = ("\n- Write the answer and every claim in Hindi, in Devanagari script. Copy the unit ids "
         "exactly as given.")
HINDI_NOTE = ("Hindi claims are checked against English evidence; that judge setup has not been "
              "measured, so these verdicts are unvalidated.")
MOCK_NOTE = "Mock answerer and judge: claims are copied from the evidence, so every verdict holds by construction."
SESSION_ID = re.compile(r"^[A-Za-z0-9_-]{16,64}$")


# ----------------------------------------------------------------- units and files

def fname(unit: EvidenceUnit) -> str:
    return Path(unit.source_file).name


def kind(unit: EvidenceUnit) -> str:
    """speech | slides | notes: the three kinds of material a lecture comes in."""
    if unit.modality == "audio":
        return "speech"
    return "slides" if is_slide_deck(unit) else "notes"


def pair_key(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def finite(x: float | None) -> float | None:
    return round(float(x), 2) if x is not None and math.isfinite(x) else None


def excerpt(text: str, limit: int = EXCERPT_CHARS) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + "…"


def unit_view(unit: EvidenceUnit) -> dict[str, Any]:
    loc = unit.location
    if loc.start_s is not None:
        where: dict[str, Any] = {"start_s": round(loc.start_s, 2), "end_s": round(loc.end_s or loc.start_s, 2)}
    elif loc.page is not None:
        where = {"page": loc.page, "bbox": [round(v, 1) for v in loc.bbox] if loc.bbox else None}
    else:
        where = {}
    return {"unit_id": unit.id, "modality": kind(unit), "figure": unit.modality in ("figure", "table"),
            "file": fname(unit), "location": where}


def memo_encoder(encoder: Encoder, index: Index | None = None) -> Encoder:
    """`encoder` with a text -> vector memo, seeded from the index's stored vectors. The linker
    and the gates re-embed texts the index already embedded, and the gates' nulls are seeded,
    so a second run shuffles identically: through the memo each distinct text is embedded once."""
    memo = {} if index is None else {embeddable_text(u): v for u, v in zip(index.units, index.vectors)}

    def encode(texts):
        texts = list(texts)
        if not texts:
            return np.asarray(encoder(texts), dtype="float32")
        missing = [t for t in dict.fromkeys(texts) if t not in memo]
        if missing:
            memo.update(zip(missing, np.asarray(encoder(missing), dtype="float32")))
        return np.stack([memo[t] for t in texts])
    return encode


def linked_context(graph, by_id: dict[str, EvidenceUnit], claims: list[dict], n_of: dict[str, int],
                   limit: int = LINKED_ROWS) -> list[dict]:
    """Context for the cited evidence: each cited unit's 1-hop neighbours in the link graph
    (`link.graph.neighbors`, strongest link first) that are not cited themselves, taken in turn
    from each cited unit so every citation is represented, at most `limit`. Shown beside the
    answer, never part of it: no evidence number, no verdict, nothing is regenerated. A
    neighbour that is also on the rail as uncited evidence is listed too (the rule is "not
    cited", not "not shown")."""
    id_of = {n: uid for uid, n in n_of.items()}
    cited = sorted({n for claim in claims for n in claim["citations"]})
    cited_ids = {id_of[n] for n in cited}
    queues = [(n, [(v, t) for v, t, _s in neighbors(graph, id_of[n]) if v not in cited_ids and v in by_id])
              for n in cited]
    rows: list[dict] = []
    seen: set[str] = set()
    for depth in range(max((len(q) for _n, q in queues), default=0)):
        for n, queue_ in queues:
            if depth < len(queue_) and queue_[depth][0] not in seen:
                unit_id, link_type = queue_[depth]
                seen.add(unit_id)
                rows.append({**unit_view(by_id[unit_id]), "link_type": link_type, "from_n": n})
                if len(rows) == limit:
                    return rows
    return rows


def dump_gates(gates: dict[tuple[str, str], dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([{"pair": list(p), "gate": g} for p, g in gates.items()]))


def read_gates(path: Path) -> dict[tuple[str, str], dict]:
    return {tuple(row["pair"]): row["gate"] for row in json.loads(path.read_text())}


# ----------------------------------------------------------------- the offline answerer

MOCK_STOP = frozenset("what which when where does into from that this with about there their they "
                      "have were been will would could should than then them these those your more "
                      "most also only such some many much".split())
MOCK_CLAIMS, MOCK_CLAIM_CHARS = 3, 220
EVIDENCE_HEAD = re.compile(r"^\[([^,\]\n]+), \w+, [^\]\n]*\]$", re.M)   # generate.answer.format_evidence
SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def mock_answerer(system: str, user: str) -> str:
    """Offline stand-in for the answerer: tests, the end-to-end run, building without spend.
    Each claim is the sentence of an evidence unit that shares the most content words with the
    question, copied verbatim and citing that unit (at most three, in evidence order). A
    question that shares no content word with any unit gets the pipeline's abstention."""
    body, _, rest = user.rpartition("\n\nQuestion: ")
    wanted = {w for w in tokenize(rest.split("\n\nAnswer with citations:")[0])
              if len(w) > 3 and w not in MOCK_STOP}
    heads = list(EVIDENCE_HEAD.finditer(body))
    claims = []
    for head, nxt in zip(heads, heads[1:] + [None]):
        text = " ".join(body[head.end(): nxt.start() if nxt else len(body)].split())
        best = max(SENTENCE_END.split(text), key=lambda s: len(wanted & set(tokenize(s))))
        if wanted & set(tokenize(best)):
            clipped = best if len(best) <= MOCK_CLAIM_CHARS else best[:MOCK_CLAIM_CHARS].rsplit(" ", 1)[0]
            claims.append({"claim": clipped, "unit_ids": [head.group(1)]})
        if len(claims) == MOCK_CLAIMS:
            break
    if not claims:
        return json.dumps({"answer": ABSTENTION, "claims": []})
    return json.dumps({"answer": claims[0]["claim"], "claims": claims})


def mock_judge(system: str, user: str) -> str:
    """Offline stand-in for the judge, in verify_gold's reply shape: a claim is supported iff
    it can be quoted from the unit (`span_in_text`, the check the real judge's span must pass)."""
    claim = user.partition("Reference answer: ")[2].partition("\n\nText:")[0]
    text = user.partition('Text:\n"""')[2].partition('"""\n\nStep 1')[0]
    ok = span_in_text(claim, text)
    return json.dumps({"facts": [{"fact": claim, "supported": ok, "span": claim if ok else ""}],
                       "verdict": "yes" if ok else "no"})


# ----------------------------------------------------------------- corpus state

@dataclass
class Corpus:
    """One lecture: its index, its links as the pipeline made them, and the file-pair settings."""

    title: str
    root: Path                              # relative source paths resolve here
    index: Index
    auto: list[Link]                        # the pipeline's links, file-pair gate on
    ungated: list[Link]                     # the same run with the gate off: what "related" adds
    gates: dict[tuple[str, str], dict]      # file pair -> gate verdict and z
    sample: bool = False
    gates_ready: threading.Event = field(default_factory=lambda: _done())
    unsure_below_z: float = UNSURE_BELOW_Z
    overrides: dict[tuple[str, str], str] = field(default_factory=dict)
    links: list[Link] = field(default_factory=list)
    graph: Any = None
    by_id: dict[str, EvidenceUnit] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.by_id = {u.id: u for u in self.index.units}
        self.relink()

    def pair_of(self, link: Link) -> tuple[str, str] | None:
        a, b = self.by_id.get(link.src_id), self.by_id.get(link.dst_id)
        if a is None or b is None or fname(a) == fname(b):
            return None
        return pair_key(fname(a), fname(b))

    def relink(self) -> None:
        """Effective links: the pipeline's own, minus every link between files marked
        unrelated, plus the ungated run's cross-file links for pairs forced related."""
        def keep(link: Link, source: str) -> bool:
            pair = self.pair_of(link)
            setting = self.overrides.get(pair, "auto") if pair else "auto"
            if setting == "unrelated":
                return False
            return source == ("ungated" if setting == "related" else "auto")
        self.links = [l for l in self.auto if keep(l, "auto")] + [l for l in self.ungated if keep(l, "ungated")]
        self.graph = build_graph(list(self.index.units), self.links)

    def source(self, unit: EvidenceUnit) -> Path | None:
        path = Path(unit.source_file)
        path = path if path.is_absolute() else self.root / path
        return path if path.exists() else None

    def files(self) -> list[dict[str, Any]]:
        by_file: dict[str, list[EvidenceUnit]] = {}
        for u in self.index.units:
            by_file.setdefault(fname(u), []).append(u)
        rows = []
        for name, units in by_file.items():
            pages = [u.location.page for u in units if u.location.page is not None]
            ends = [u.location.end_s for u in units if u.location.end_s is not None]
            rows.append({"name": name, "kind": kind(units[0]), "units": len(units),
                         "pages": max(pages) if pages else None,
                         "duration": round(max(ends), 1) if ends else None})
        order = {"speech": 0, "slides": 1, "notes": 2}
        return sorted(rows, key=lambda r: (order[r["kind"]], r["name"]))

    def pairs(self) -> list[dict[str, Any]]:
        counts = Counter(p for l in self.links if (p := self.pair_of(l)))
        keys = set(self.gates) | {p for l in (*self.auto, *self.ungated) if (p := self.pair_of(l))}
        position = {f["name"]: i for i, f in enumerate(self.files())}
        rows = []
        for key in sorted(keys, key=lambda k: sorted(position.get(n, 99) for n in k)):
            gate = self.gates.get(key) or {}
            setting, n, z = self.overrides.get(key, "auto"), counts[key], finite(gate.get("z"))
            unsure = (setting == "auto" and n > 0 and gate.get("kind") == "audio_slide"
                      and z is not None and z < self.unsure_below_z)
            rows.append({"a": key[0], "b": key[1], "kind": gate.get("kind"), "z": z,
                         "threshold_z": gate.get("threshold_z"),
                         "gate": ("related" if gate["related"] else "unrelated") if gate else None,
                         "setting": setting, "links": n,
                         "state": "unsure" if unsure else "linked" if n else "unlinked"})
        return rows

    def stats(self) -> dict[str, Any]:
        """Link counts from the graph retrieval walks. Edges plus the links build_graph dropped
        must equal the links it was given (DESIGN.md measurement rule 2: the check that caught
        edges keyed by link type)."""
        by_type = Counter(d["link_type"] for _u, _v, d in self.graph.edges(data=True))
        total, dropped = self.graph.number_of_edges(), self.graph.graph.get("dropped_links", 0)
        if total + dropped != len(self.links):
            raise RuntimeError(f"{len(self.links)} links became {total} edges + {dropped} dropped")
        return {"by_type": dict(sorted(by_type.items())), "total": total, "dropped": dropped,
                "links": len(self.links), "cross_file": sum(1 for l in self.links if self.pair_of(l)),
                "units": len(self.index), "units_by_kind": dict(Counter(kind(u) for u in self.index.units))}

    def main_audio(self) -> list[EvidenceUnit]:
        """The recording the player plays: the first audio file, its units in time order."""
        audio = [u for u in self.index.units if u.modality == "audio"]
        if not audio:
            return []
        first = min(fname(u) for u in audio)
        return sorted((u for u in audio if fname(u) == first), key=lambda u: u.location.start_s or 0.0)


def _done() -> threading.Event:
    event = threading.Event()
    event.set()
    return event


@dataclass
class Session:
    dir: Path
    uploads: dict[str, Path] = field(default_factory=dict)
    corpus: Corpus | None = None
    stale: bool = False             # uploads changed since the corpus was built
    busy: bool = False
    cost: float = 0.0
    cost_known: bool = True


# ----------------------------------------------------------------- media

def render_page(unit: EvidenceUnit, src: Path, out: Path) -> Path:
    """The unit's page with its box drawn in the modality colour: the whole page, except a
    figure in the notes, which is cropped to its region (see FIGURE_*)."""
    import pymupdf
    out.parent.mkdir(parents=True, exist_ok=True)
    rgb = tuple(c / 255 for c in HIGHLIGHT[kind(unit)])
    with pymupdf.open(src) as doc:
        page = doc[unit.location.page - 1]
        clip = page.rect
        if unit.location.bbox:
            box = pymupdf.Rect(unit.location.bbox) & page.rect
            page.draw_rect(box, color=rgb, fill=rgb, fill_opacity=0.10, width=1.5)
            if not is_slide_deck(unit) and unit.modality in ("figure", "table"):
                grow = max(FIGURE_PAD_PT, (FIGURE_MIN_PT - box.width) / 2)
                clip = pymupdf.Rect(box.x0 - grow, box.y0 - FIGURE_PAD_PT, box.x1 + grow,
                                    box.y1 + FIGURE_PAD_PT) & page.rect
        zoom = PAGE_PX / clip.width
        page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), clip=clip).save(out)
    return out


@lru_cache(maxsize=8)
def audio_profile(path: str, mtime: float, bins: int = PEAK_BINS) -> tuple[float, tuple[float, ...]]:
    """(duration in seconds, peak level per bin in [0, 1]): the waveform the player draws.
    One peak per decoded frame, then per bin; `mtime` keys the cache to the file's version."""
    import av
    peaks, end = [], 0.0
    with av.open(path) as container:
        stream = container.streams.audio[0]
        for frame in container.decode(stream):
            peaks.append(float(np.abs(frame.to_ndarray().astype("float32")).max()))
            if frame.time is not None:
                end = max(end, float(frame.time) + frame.samples / float(frame.sample_rate or 1))
    if not peaks:
        return end, ()
    binned = [float(b.max()) for b in np.array_split(np.asarray(peaks), min(bins, len(peaks)))]
    top = max(binned) or 1.0
    return end, tuple(round(p / top, 3) for p in binned)


# ----------------------------------------------------------------- jobs (build, sample)

class _Forward(logging.Handler):
    """The pipeline's own stage timings (`core.stage_timer` lines), from the job's thread only,
    forwarded as progress."""

    def __init__(self, emit: Callable[..., None], thread: int):
        super().__init__(logging.INFO)
        self.send, self.thread = emit, thread

    def emit(self, record: logging.LogRecord) -> None:
        if record.thread == self.thread:
            self.send(log=" ".join(record.getMessage().split()))


@contextmanager
def stage(job, name: str):
    start = time.perf_counter()
    job.emit(stage=name, status="running")
    yield
    job.emit(stage=name, status="done", seconds=round(time.perf_counter() - start, 1))


def stream_job(session: Session, work: Callable[[Any], None]) -> StreamingResponse:
    """Run `work` in a thread and stream its events as NDJSON; the last line is
    {"status": "ready"} or {"status": "error", "detail": ...}. A client that disconnects
    leaves the job running."""
    events: queue.Queue = queue.Queue()
    job = SimpleNamespace(emit=lambda **event: events.put(event))

    def run() -> None:
        forward = _Forward(job.emit, threading.get_ident())
        log.addHandler(forward)
        try:
            work(job)
            job.emit(status="ready")
        except (Exception, SystemExit) as exc:          # MissingApiKey is a SystemExit
            log.exception("lectern job failed")
            job.emit(status="error", detail=str(exc) or type(exc).__name__)
        finally:
            log.removeHandler(forward)
            session.busy = False
            events.put(None)

    session.busy = True
    threading.Thread(target=run, name="lectern-job", daemon=True).start()

    def lines():
        while (event := events.get()) is not None:
            yield json.dumps(event) + "\n"
    return StreamingResponse(lines(), media_type="application/x-ndjson")


# ----------------------------------------------------------------- the app

def _first_call_alone(encoder: Encoder) -> Encoder:
    """The first call loads the model; lru_cache does not stop two threads loading it at once
    (start-up warm-up and a sample load), which would hold bge-m3 in memory twice."""
    lock, warm = threading.Lock(), threading.Event()

    def encode(texts):
        if not warm.is_set():
            with lock:
                out = encoder(texts)
                warm.set()
                return out
        return encoder(texts)
    return encode


class AskBody(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    compare_baseline: bool = False
    lang: Literal["en", "hi"] = "en"


class PairBody(BaseModel):
    a: str
    b: str
    setting: Literal["auto", "related", "unrelated"]


class Answerers:
    """The answerer and the judge. The real ones are made once per process, so their spend
    guards see every call; each question gets fresh caches over them, so its own usage can be
    read off. --max-cost caps the two together: before each call, a completer's budget is what
    the other has not spent (each guard reads its budget live)."""

    def __init__(self, cfg: dict[str, Any], kind: str, max_cost: float, cache: Path):
        self.cfg, self.kind, self.max_cost, self.cache = cfg, kind, max_cost, cache
        self.backend = entailment_opts(cfg)["backend"]
        self._pair: tuple[Callable, Callable] | None = None
        self._lock = threading.Lock()

    def names(self) -> tuple[str, str]:
        if self.kind == "mock":
            return "mock", "mock"
        runs = int((self.cfg.get("generation") or {}).get("verify_runs", 3))
        return str(self.cfg["models"]["llm"]["model"]), verifier_label(self.cfg, self.backend, runs)[0]

    def _make(self) -> tuple[Callable, Callable]:
        answer_block = self.cfg["models"]["llm"]
        judge_block = (self.cfg.get("eval") or {}).get("judge") or answer_block
        answer, judge = http_completer(answer_block), judge_completer(self.cfg, self.backend)
        pricing = self.cfg["models"].get("pricing")

        def capped(inner, block, other, other_model):
            price = price_for(str(other_model), pricing)

            def complete(system: str, user: str) -> str:
                block["max_cost_usd"] = max(0.0, self.max_cost - (usage_cost(other.usage, price) or 0.0))
                text = inner(system, user)
                complete.last_usage = getattr(inner, "last_usage", {})  # type: ignore[attr-defined]
                return text
            complete.usage, complete.last_usage = inner.usage, {}  # type: ignore[attr-defined]
            return complete
        return (capped(answer, answer_block, judge, judge_block.get("model")),
                capped(judge, judge_block, answer, answer_block.get("model")))

    def pair(self) -> tuple[Callable, Callable]:
        if self.kind == "mock":
            return mock_answerer, mock_judge
        with self._lock:
            if self._pair is None:
                self._pair = self._make()
        answer, judge = self._pair
        return cached_completer(answer, self.cache / "answer"), cached_completer(judge, self.cache / "judge")


def create_app(cfg: dict[str, Any], *, workdir: Path, sample: Path | None = None,
               sample_title: str = "Sample lecture", answerer: str = "llm", max_cost: float = 0.0,
               strong: bool = False, ledger: Path | None = LEDGER, encoder: Encoder | None = None) -> FastAPI:
    """`sample`: a frozen corpus directory (index/, links.jsonl, manifest.json -- the layout
    scripts/ingest.py and scripts/build_links.py write), read and never written. `encoder` is
    injectable so tests run without bge-m3. Everything written goes under `workdir`."""
    if answerer not in ("llm", "mock"):
        raise ValueError(f"unknown answerer {answerer!r}: use 'llm' or 'mock'")
    cfg = copy.deepcopy(cfg)
    set_max_cost(cfg, max_cost)
    cfg["cost"]["max_usd"] = 0.0        # whisper-1's budget: an upload form never pays for ASR
    if answerer == "llm":
        if strong:
            cfg["models"]["llm"].update(model=STRONG_MODEL, allow_strong_in_batch=True)
        refuse_strong_in_batch(cfg)
    encoder = encoder or _first_call_alone(default_encoder(cfg["models"]["embedding"], cfg["device"],
                                                           cfg["index"]["normalize_embeddings"]))
    workdir = Path(workdir)
    models = Answerers(cfg, answerer, max_cost, workdir / "llm_cache")
    sessions: OrderedDict[str, Session] = OrderedDict()
    sessions_lock, sample_lock = threading.Lock(), threading.Lock()
    build_lock = threading.Lock()       # one upload build at a time: ASR and embedding take the CPU
    # link_corpus hands results through function attributes (build_links.gate,
    # link_figures_to_text.unrelated_pairs), so two runs at once could swap them
    link_lock = threading.Lock()
    live_lock = threading.Lock()        # billed questions one at a time, so --max-cost holds
    sample_bases: dict[bool, dict[str, Any]] = {}     # the sample with / without its notes
    rcfg = cfg["retrieve"]["rerank"]
    k = int(cfg["retrieve"]["linkrag"]["k_final"])
    unsure_below_z = float((cfg.get("ui") or {}).get("unsure_below_z", UNSURE_BELOW_Z))
    linked_rows = int((cfg.get("ui") or {}).get("linked_rows", LINKED_ROWS))
    if log.getEffectiveLevel() > logging.INFO:
        log.setLevel(logging.INFO)          # stage timings are the build's progress lines

    app = FastAPI(title="Lectern", docs_url=None, redoc_url=None, openapi_url=None)
    app.state.encoder = encoder
    app.state.preload = lambda: sample and [ensure_sample(SimpleNamespace(emit=lambda **_e: None), paper)
                                            for paper in (False, True)]

    @lru_cache(maxsize=1)
    def sample_notes() -> list[str]:
        """The sample's notes files (for pilot01, the paper): optional when it is loaded."""
        return sorted({fname(u) for u in Index.load(sample / "index").units if kind(u) == "notes"})

    def session(request: Request) -> Session:
        """Sessions are keyed by an id the page sends (X-Session header, or ?s= on media URLs),
        not a cookie: a Space runs in a cross-site iframe, where cookies are not sent."""
        sid = request.headers.get("x-session") or request.query_params.get("s") or ""
        if not SESSION_ID.match(sid):
            raise HTTPException(400, "missing or malformed session id (X-Session header)")
        with sessions_lock:
            current = sessions.pop(sid, None) or Session(dir=workdir / "sessions" / sid)
            sessions[sid] = current
            while len(sessions) > MAX_SESSIONS:
                idle = next((s for s, v in sessions.items() if not v.busy and s != sid), None)
                if idle is None:
                    break
                shutil.rmtree(sessions.pop(idle).dir, ignore_errors=True)
        return current

    def corpus_of(s: Session) -> Corpus:
        if s.corpus is None:
            raise HTTPException(409, "no lecture loaded: upload files and build, or load the sample")
        return s.corpus

    def link_pipeline(job, index: Index, frozen: list[Link] | None = None):
        """(auto, ungated, gates) for an index. Needs speech and slides, like build_links.py;
        without them there is nothing to align, and answers come from plain retrieval.
        `frozen`: stored links, used as they are -- the run is then only for the gates' z."""
        audio, slides, texts, figures = split_units(index, None, None)
        missing = ("speech and slides" if not audio or not slides else
                   # Note: link_corpus raises without a figure (figure_text_scores needs one);
                   # a guard belongs in link_figures_to_text, which is outside the UI
                   "at least one figure" if not figures else None)
        if missing:
            job.emit(stage="link", warning=f"links need {missing}: answering from plain retrieval")
            return frozen or [], frozen or [], {}
        with hold(link_lock, job):
            return link_runs(job, index, frozen, audio, slides, texts, figures)

    def link_runs(job, index, frozen, audio, slides, texts, figures):
        memo = memo_encoder(encoder, index)
        with stage(job, "link"):
            run = link_corpus(audio, slides, texts, figures, encoder=memo, cfg=cfg)
        with stage(job, "pairs"):
            z = cfg["link"]["align"].get("relatedness_z")
            gates: dict[tuple[str, str], dict] = {}
            recordings, decks = sorted({fname(u) for u in audio}), sorted({fname(u) for u in slides})
            if len(recordings) * len(decks) > 1:
                job.emit(stage="pairs", warning=f"{len(recordings)} recordings x {len(decks)} decks are linked "
                                                "as one sequence, as scripts/build_links.py does; each pair's "
                                                "z is from its own alignment")
            if run.gate is not None:
                for a in recordings:
                    for d in decks:     # one recording and one deck: the run's own gate
                        g = run.gate if len(recordings) * len(decks) == 1 else link_corpus(
                            [u for u in audio if fname(u) == a], [u for u in slides if fname(u) == d],
                            texts, figures, encoder=memo, cfg=cfg).gate
                        gates[pair_key(a, d)] = {"kind": "audio_slide", "z": g["z"], "related": g["related"],
                                                 "threshold_z": z}
            if z is not None:           # the directions link_figures_to_text gates, with its arguments
                for fd in sorted({fname(f) for f in figures}):
                    for td in sorted({fname(t) for t in texts} - {fd}):
                        g = document_pair_gate([f for f in figures if fname(f) == fd],
                                               [t for t in texts if fname(t) == td], encoder=memo, z=z)
                        pair = gates.setdefault(pair_key(fd, td), {"kind": "figure_text", "z": -math.inf,
                                                                   "related": False, "threshold_z": z})
                        pair["z"], pair["related"] = max(pair["z"], g["z"]), pair["related"] or g["related"]
            if frozen is not None:
                return frozen, frozen, gates
            off = copy.deepcopy(cfg)
            off["link"]["align"]["relatedness_z"] = None
            ungated = link_corpus(audio, slides, texts, figures, encoder=memo, cfg=off).links
        return run.links, ungated, gates

    @contextmanager
    def hold(lock: threading.Lock, job):
        """`lock`, telling the page when it has to wait for it."""
        if not lock.acquire(blocking=False):
            job.emit(status="queued")
            lock.acquire()
        try:
            yield
        finally:
            lock.release()

    def build(job, s: Session) -> None:
        bcfg = copy.deepcopy(cfg)       # ingest writes figures and transcripts: into the session
        bcfg["ingest"]["figures_dir"] = str(s.dir / "figures")
        bcfg["ingest"]["frozen_transcript_dir"] = str(s.dir / "transcripts")
        with hold(build_lock, job):
            with stage(job, "ingest"):
                units = ingest_files(list(s.uploads.values()), bcfg)
                for path, exc in getattr(ingest_files, "last_failures", []):
                    job.emit(stage="ingest", warning=f"skipped {Path(path).name}: {type(exc).__name__}: {exc}")
                if not units:
                    raise ValueError("nothing could be read from these files")
            with stage(job, "index"):
                index = build_index(units, encoder=encoder, embedding_model=cfg["models"]["embedding"],
                                    device=cfg["device"], normalize=cfg["index"]["normalize_embeddings"])
            auto, ungated, gates = link_pipeline(job, index)
        decks = sorted({fname(u) for u in index.units if kind(u) == "slides"})
        title = re.sub(r"[_-]+", " ", Path((decks or sorted({fname(u) for u in index.units}))[0]).stem).strip()
        s.corpus = Corpus(title=title, root=s.dir, index=index, auto=auto, ungated=ungated, gates=gates,
                          unsure_below_z=unsure_below_z)
        s.stale = False

    def cache_key(corpus_hash: str | None, embedding_model: str) -> str | None:
        """What a cached derivation of the sample stands for: the corpus, the embedder, the link
        config and the gate code. An unstamped corpus is not identified by anything: no key."""
        if not corpus_hash:
            return None
        code = b"".join(Path(m.__file__).read_bytes() for m in (align_module, figure_text_module,
                                                                  pipeline_module, sys.modules[__name__]))
        return hashlib.sha256(json.dumps([corpus_hash, embedding_model, cfg["link"]], sort_keys=True,
                                         default=str).encode() + code).hexdigest()[:16]

    def sample_gates(index: Index, frozen: list[Link], gates: dict, ready: threading.Event, key: str | None) -> None:
        """The full sample's file-pair gates, in the background: the document-pair gate re-embeds
        shuffled text (minutes of CPU), and the sample is usable without them. Cached under
        the workdir by `cache_key`, so a restart does not pay again."""
        cache = workdir / f"sample_gates_{key}.json" if key else None
        try:
            if cache is not None and cache.exists():
                found = read_gates(cache)
            else:
                found = link_pipeline(SimpleNamespace(emit=lambda **_e: None), index, frozen=frozen)[2]
                if cache is not None:
                    dump_gates(found, cache)
            gates.update(found)
        except Exception:
            log.exception("sample gates failed; pairs show links without a gate")
        finally:
            ready.set()

    def sample_lecture(job, index: Index, manifest: dict[str, Any]) -> dict[str, Any]:
        """The sample without its notes: the recording and the deck, linked by the pipeline for
        those files alone. Units and vectors are the frozen index's own (no re-transcription, no
        re-embedding). Linking two files instead of three can move scores (figure_text's IDF and
        per-figure cap depend on the texts present) and could change which links exist; on
        pilot01 the links are the frozen ones without the paper, and only scores move. Stored
        relatedness verdicts (the LLM judge on pilot01) carry over to every link judged before,
        keyed by (src, dst, type), so parallel links share one; load_links(gated=True) drops the
        failures, as it does for the frozen links, and a link never judged is kept. Built once,
        into the standard corpus layout under the workdir, and read from there after."""
        units = [u for u in index.units if kind(u) != "notes"]
        names = {fname(u) for u in units}
        # the parent's `derived` (its frozen transcript) stays: two corpora over the same files with
        # different transcripts must not share a hash (manifest.build_manifest)
        sub = {"files": [f for f in manifest.get("files", []) if f["name"] in names],
               "derived": manifest.get("derived", []), "total_units": len(units),
               "units_by_modality": dict(sorted(Counter(u.modality for u in units).items()))}
        sub["hash"] = manifest_hash(sub)
        # keyed by the parent corpus itself, so nothing about it can be lost in the cut
        key = cache_key(manifest.get("hash") and f"{manifest['hash']}/lecture", index.embedding_model)
        folder = workdir / f"sample_lecture_{key or 'unstamped'}"
        if key is None or not (folder / "gates.json").exists():
            lecture = build_index(units, encoder=memo_encoder(encoder, index), embedding_model=index.embedding_model,
                                  normalize=index.normalize)
            auto, ungated, gates = link_pipeline(job, lecture)
            judged = {(l.src_id, l.dst_id, l.link_type): l.metadata["relatedness"]
                      for l in load_links(sample / "links.jsonl", gated=False) if (l.metadata or {}).get("relatedness")}
            for link in (*auto, *ungated):
                verdict = judged.get((link.src_id, link.dst_id, link.link_type))
                if verdict:
                    link.metadata = {**(link.metadata or {}), "relatedness": verdict}
            lecture.save(folder / "index")
            write_manifest(sub, folder / MANIFEST_NAME)
            save_links(auto, folder / "links.jsonl", manifest_hash=sub["hash"])
            save_links(ungated, folder / "ungated.jsonl", manifest_hash=sub["hash"])
            dump_gates(gates, folder / "gates.json")      # last: it marks the folder complete
        return {"index": Index.load(folder / "index"),
                "auto": load_links(folder / "links.jsonl", expect_manifest=sub["hash"]),
                "ungated": load_links(folder / "ungated.jsonl", expect_manifest=sub["hash"]),
                "gates": read_gates(folder / "gates.json")}

    def ensure_sample(job, paper: bool) -> dict[str, Any]:
        """The sample, loaded once per process and variant; main() loads both at start-up.
        With its notes (`paper`): the frozen corpus as stored, gates in the background.
        Without: the recording and the deck (`sample_lecture`)."""
        with sample_lock:
            if paper not in sample_bases:
                with stage(job, "index"):
                    index = Index.load(sample / "index")
                    if index.embedding_model != cfg["models"]["embedding"]:
                        raise ValueError(f"the sample was embedded with {index.embedding_model}; "
                                         f"questions would be embedded with {cfg['models']['embedding']}")
                    manifest = load_manifest(sample / MANIFEST_NAME) or {}
                    # frozen as stored: gated=True drops per-link relatedness failures, as scripts/ask.py does
                    frozen = load_links(sample / "links.jsonl", expect_manifest=manifest.get("hash"))
                    encoder([""])           # load the embedder now, not on the first question
                if paper:
                    gates, ready = {}, threading.Event()
                    threading.Thread(target=sample_gates,
                                     args=(index, frozen, gates, ready, cache_key(manifest.get("hash"), index.embedding_model)),
                                     name="lectern-sample-gates", daemon=True).start()
                    part = {"index": index, "auto": frozen, "ungated": frozen, "gates": gates, "gates_ready": ready}
                else:
                    part = sample_lecture(job, index, manifest)
                # the stored source paths are relative to the project root
                sample_bases[paper] = {"title": sample_title, "root": sample.parent.parent, "sample": True,
                                       "unsure_below_z": unsure_below_z, **part}
            return sample_bases[paper]

    def load_sample(job, s: Session, paper: bool) -> None:
        s.corpus = Corpus(**ensure_sample(job, paper))
        s.uploads.clear()
        s.stale = False

    def answer_one(c: Corpus, graph, question: str, mode: str, method: str, answer, judge):
        """One answer, as scripts/ask.py makes it: retrieve, rerank to k, answer, verify."""
        pool = max(int(rcfg.get("pool", k)), k) if method != "none" else k
        results, _calls = retrieve_pool(mode, question, c.index, graph, encoder=encoder, cfg=cfg,
                                        complete=None, pool=pool)
        results = rerank(results, k, method=method, index=c.index, graph=graph if mode == "linkrag" else None,
                         alpha=rcfg["alpha"], beta=rcfg["beta"], gamma=rcfg["gamma"],
                         mmr_lambda=rcfg["mmr_lambda"], question=question,
                         cross_encoder=rcfg.get("cross_encoder"), device=cfg["device"],
                         modality_gate=rcfg.get("modality_gate", False))
        units = [r.unit for r in results]
        ans = answer_json(question, units, cfg, mode=mode, complete=answer)
        if ans.malformed:
            raise RuntimeError("the answerer did not return valid JSON after one retry")
        gcfg = cfg.get("generation") or {}
        if gcfg.get("verify", True):
            verify_answer(ans, units, judge, runs=int(gcfg.get("verify_runs", 3)),
                          strict=bool(gcfg.get("strict", False)), **entailment_opts(cfg))
        return results, ans

    def evidence_item(c: Corpus, n: int, r: RetrievedUnit) -> dict[str, Any]:
        u, src = r.unit, c.source(r.unit)
        uid = quote(u.id, safe="")
        pageable = src is not None and (src.suffix.lower() in IMAGE_SUFFIXES
                                        or (src.suffix.lower() == ".pdf" and u.location.page is not None))
        return {"n": n, **unit_view(u),
                "crop_url": f"/media/page/{uid}" if pageable else None,
                "clip_url": f"/media/clip/{uid}" if u.modality == "audio" and src is not None else None,
                "origin": "via_link" if r.origin == "expanded" else "direct",
                "link_type": r.via_link, "via_unit": r.via_seed, "excerpt": excerpt(u.content)}

    def pack(c: Corpus, results: list[RetrievedUnit], ans, n_of: dict[str, int]) -> dict[str, Any]:
        strict = bool((cfg.get("generation") or {}).get("strict", False))
        claims = [{"text": cl.claim, "verdict": cl.verdict,
                   "citations": sorted({n_of[i] for i in cl.unit_ids if i in n_of})}
                  for cl in ans.claims if not (strict and cl.verdict == "unsupported")]
        return {"answer_claims": claims, "summary": ans.answer if claims else ABSTENTION,
                "evidence": [evidence_item(c, n_of[r.unit.id], r) for r in results],
                "abstained": not claims}

    def account(s: Session, answer, judge, question: str) -> None:
        if models.kind == "mock":
            return
        pricing = cfg["models"].get("pricing")
        parts = list(zip(models.names(), (answer.usage, judge.usage)))
        for model, usage in parts:
            billed = usage.get("calls", 0) - usage.get("cached_calls", 0)
            cost = usage_cost(usage, price_for(model, pricing)) if billed else 0.0
            if cost is None:
                s.cost_known = False
            else:
                s.cost += cost
        if ledger is not None and any(u.get("calls") for _m, u in parts):
            record_run("linkrag.ui", question[:60], parts, pricing, ledger=ledger)

    # ------------------------------------------------------------- endpoints

    @app.get("/state")
    def state(request: Request):
        s = session(request)
        c = s.corpus
        return {"sample_available": sample is not None, "sample_notes": sample_notes() if sample else [],
                "building": s.busy,
                "corpus": None if c is None else {"title": c.title, "sample": c.sample, "files": c.files(),
                                                   "links": len(c.links), "stale": s.stale},
                "pending": list(s.uploads) if (c is None or s.stale) else [],
                "cost_usd": round(s.cost, 6), "cost_known": s.cost_known, "max_cost_usd": max_cost,
                "answerer": dict(zip(("model", "judge"), models.names()), mock=models.kind == "mock",
                                 strong=bool(strong and answerer == "llm"))}

    @app.post("/upload")
    def upload(request: Request, files: list[UploadFile] = File(...)):
        s = session(request)
        if s.busy:
            raise HTTPException(409, "a build is running")
        if s.corpus is not None and s.corpus.sample:
            raise HTTPException(409, "the sample lecture is read-only")
        names = [re.sub(r"[^A-Za-z0-9._-]+", "_", Path(f.filename or "").name).strip("._")[:120] or "file"
                 for f in files]
        bad = [n for n in names if Path(n).suffix.lower() not in SUFFIXES]
        if bad:
            raise HTTPException(415, f"not a lecture file: {', '.join(bad)} (accepted: {' '.join(sorted(SUFFIXES))})")
        used = sum(p.stat().st_size for p in s.uploads.values() if p.exists())
        (s.dir / "uploads").mkdir(parents=True, exist_ok=True)
        for f, name in zip(files, names):
            dest = s.dir / "uploads" / name
            with dest.open("wb") as out:
                shutil.copyfileobj(f.file, out, 1 << 20)
            used += dest.stat().st_size
            if used > MAX_SESSION_BYTES:
                dest.unlink()
                raise HTTPException(413, f"uploads are limited to {MAX_SESSION_BYTES >> 20} MB per lecture")
            s.uploads[name] = dest
        s.stale = True
        return {"files": names, "pending": list(s.uploads)}

    @app.post("/build")
    def build_endpoint(request: Request):
        s = session(request)
        if not s.uploads:
            raise HTTPException(400, "upload files first")
        if s.busy:
            raise HTTPException(409, "a build is already running")
        return stream_job(s, lambda job: build(job, s))

    @app.post("/sample")
    def sample_endpoint(request: Request, paper: bool = False):
        s = session(request)
        if sample is None:
            raise HTTPException(404, "no sample lecture on this server")
        if s.busy:
            raise HTTPException(409, "a build is running")
        return stream_job(s, lambda job: load_sample(job, s, paper))

    @app.post("/reset")
    def reset(request: Request):
        s = session(request)
        if s.busy:
            raise HTTPException(409, "a build is running")
        s.corpus, s.stale = None, False
        s.uploads.clear()
        shutil.rmtree(s.dir / "uploads", ignore_errors=True)
        return {"ok": True}

    @app.get("/pairs")
    def get_pairs(request: Request):
        c = corpus_of(session(request))
        return {"pairs": c.pairs(), "pending": not c.gates_ready.is_set()}

    @app.put("/pairs")
    def put_pairs(request: Request, body: PairBody):
        c = corpus_of(session(request))
        key = pair_key(body.a, body.b)
        if key not in {(p["a"], p["b"]) for p in c.pairs()}:
            raise HTTPException(404, f"no file pair {body.a} / {body.b}")
        if body.setting == "auto":
            c.overrides.pop(key, None)
        else:
            c.overrides[key] = body.setting
        c.relink()
        return {"pairs": c.pairs(), "pending": not c.gates_ready.is_set(), "stats": c.stats()}

    @app.get("/stats")
    def stats(request: Request):
        return corpus_of(session(request)).stats()

    @app.post("/ask")
    def ask(request: Request, body: AskBody):
        s = session(request)
        c = corpus_of(s)
        graph = c.graph                 # one snapshot, even if a pair is toggled meanwhile
        answer, judge = models.pair()
        speak = (lambda system, user: answer(system + HINDI, user)) if body.lang == "hi" else answer
        try:
            with live_lock if models.kind == "llm" else contextlib.nullcontext():
                lectern = answer_one(c, graph, body.question, "linkrag", rcfg["method"], speak, judge)
                base = (answer_one(c, graph, body.question, "baseline", "none", speak, judge)
                        if body.compare_baseline else None)
        except BillingRefused as exc:
            raise HTTPException(402, str(exc))
        except MissingApiKey as exc:
            raise HTTPException(503, str(exc))
        except RuntimeError as exc:
            raise HTTPException(502, str(exc))
        finally:
            account(s, answer, judge, body.question)
        order = [r.unit.id for r in lectern[0]]
        if base is not None:
            order += [r.unit.id for r in base[0] if r.unit.id not in set(order)]
        n_of = {uid: i + 1 for i, uid in enumerate(order)}
        # a lecture without links answers from plain retrieval: say so rather than label it linkrag
        packed = pack(c, *lectern, n_of)
        out = {"question": body.question, **packed, "links": graph.number_of_edges(),
               "linked_context": linked_context(graph, c.by_id, packed["answer_claims"], n_of, linked_rows),
               "cost_usd": round(s.cost, 6), "cost_known": s.cost_known,
               "verification_note": MOCK_NOTE if models.kind == "mock" else HINDI_NOTE if body.lang == "hi" else None}
        if base is not None:
            seen = {r.unit.id for r in base[0]}
            # in Lectern's evidence, not retrieved by the baseline (no gold: not "needed")
            out["baseline"] = {**pack(c, *base, n_of), "linked_context": [],     # the baseline follows no links
                               "missed_evidence": [uid for uid in order[:len(lectern[0])] if uid not in seen]}
        return out

    @app.get("/graph")
    def graph_endpoint(request: Request, units: list[str] = Query(default=[])):
        c = corpus_of(session(request))
        ids = [i for i in dict.fromkeys(units) if i in c.graph]
        best: dict[tuple[str, str, str], float] = {}
        for src, dst, data in c.graph.subgraph(ids).edges(data=True):
            key = (src, dst, data["link_type"])
            best[key] = max(best.get(key, -math.inf), float(data["score"]))
        return {"nodes": [unit_view(c.by_id[i]) for i in ids],
                "edges": [{"src": a, "dst": b, "link_type": t, "score": round(v, 3)}
                          for (a, b, t), v in best.items()]}

    @app.get("/timeline")
    def timeline(request: Request):
        c = corpus_of(session(request))
        audio = c.main_audio()
        if not audio:
            return {"file": None, "duration": 0.0, "intervals": [], "peaks": [], "audio_url": None}
        src = c.source(audio[0])
        duration, peaks = max(u.location.end_s or 0.0 for u in audio), ()
        if src is not None:
            duration, peaks = audio_profile(str(src), src.stat().st_mtime)
        slide = {l.src_id: c.by_id[l.dst_id] for l in c.links
                 if l.link_type == "audio_slide" and l.dst_id in c.by_id}
        intervals: list[dict[str, Any]] = []
        previous = None
        for u in audio:     # consecutive segments on one slide make one interval
            s = slide.get(u.id)
            if s is not None and previous is s:
                intervals[-1]["end"] = round(u.location.end_s or 0.0, 2)
            elif s is not None:
                intervals.append({"start": round(u.location.start_s or 0.0, 2), "end": round(u.location.end_s or 0.0, 2),
                                  "page": s.location.page, "file": fname(s), "unit_id": s.id})
            previous = s
        return {"file": fname(audio[0]), "duration": round(duration, 2), "intervals": intervals,
                "peaks": list(peaks), "audio_url": "/media/audio" if src is not None else None}

    def media_unit(request: Request, unit_id: str) -> tuple[Session, Corpus, EvidenceUnit, Path]:
        s = session(request)
        c = corpus_of(s)
        u = c.by_id.get(unit_id)
        src = c.source(u) if u is not None else None
        if u is None or src is None:
            raise HTTPException(404, "no such unit, or its source file is not on this server")
        return s, c, u, src

    def media_dir(s: Session, c: Corpus) -> Path:
        return workdir / "sample" if c.sample else s.dir

    @app.get("/media/page/{unit_id}")
    def media_page(request: Request, unit_id: str):
        s, c, u, src = media_unit(request, unit_id)
        if src.suffix.lower() in IMAGE_SUFFIXES:
            return FileResponse(src)
        if u.location.page is None:
            raise HTTPException(404, "this unit has no page")
        out = media_dir(s, c) / "pages" / f"{hashlib.sha256(u.id.encode()).hexdigest()[:20]}.png"
        if not out.exists():
            render_page(u, src, out)
        return FileResponse(out, media_type="image/png")

    @app.get("/media/clip/{unit_id}")
    def media_clip(request: Request, unit_id: str):
        s, c, u, src = media_unit(request, unit_id)
        if u.modality != "audio":
            raise HTTPException(404, "not an audio unit")
        path = clip_audio(replace(u, source_file=str(src)), media_dir(s, c) / "clips")
        return FileResponse(path, media_type="audio/mp4")

    @app.get("/media/audio")
    def media_audio(request: Request):
        c = corpus_of(session(request))
        audio = c.main_audio()
        src = c.source(audio[0]) if audio else None
        if src is None:
            raise HTTPException(404, "the recording is not on this server")
        return FileResponse(src)

    if (WEB / "index.html").exists():
        app.mount("/assets", StaticFiles(directory=WEB / "assets"), name="assets")

        @app.get("/")
        def page():
            return FileResponse(WEB / "index.html")

        @app.get("/favicon.svg")
        def favicon():
            return FileResponse(WEB / "favicon.svg")
    else:
        @app.get("/")
        def page():
            return PlainTextResponse("frontend not built: cd src/linkrag/ui/web && npm ci && npm run build")
    return app


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Lectern, the LinkRAG web UI.")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--sample", default=os.environ.get("LECTERN_SAMPLE"),
                    help="frozen corpus dir (a data/processed with index/, links.jsonl, manifest.json), read-only")
    ap.add_argument("--sample-title", default=os.environ.get("LECTERN_SAMPLE_TITLE", "Sample lecture"))
    ap.add_argument("--workdir", default=os.environ.get("LECTERN_WORKDIR",
                                                        str(Path(tempfile.gettempdir()) / "lectern")),
                    help="everything the server writes goes here")
    ap.add_argument("--answerer", choices=["llm", "mock"], default=os.environ.get("LECTERN_ANSWERER", "llm"),
                    help="mock: offline extractive stand-in for answerer and judge (no spend)")
    ap.add_argument("--max-cost", type=float, default=float(os.environ.get("LECTERN_MAX_COST", "0")),
                    help="USD cap on answerer + judge for this process; 0 = zero-cost mode")
    ap.add_argument("--demo-strong", action="store_true",
                    help=f"answer with the strong model {STRONG_MODEL} (demo only, DESIGN.md cost policy)")
    ap.add_argument("--ledger", default=os.environ.get("LECTERN_LEDGER", str(LEDGER)),
                    help="spend ledger (linkrag.costs); '' to keep none")
    ap.add_argument("--host", default=os.environ.get("LECTERN_HOST", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    args = ap.parse_args(argv)

    setup_logging()
    sample = Path(args.sample) if args.sample else None
    if sample is not None and not (sample / "index" / "units.json").exists():
        log.warning("no frozen corpus at %s: the sample lecture is off", sample)   # an image built without it
        sample = None
    app = create_app(load_config(args.config), workdir=Path(args.workdir),
                     sample=sample, sample_title=args.sample_title,
                     answerer=args.answerer, max_cost=args.max_cost, strong=args.demo_strong,
                     ledger=Path(args.ledger) if args.ledger else None)
    # load the embedder and the sample while the page loads, not when someone first needs them
    threading.Thread(target=lambda: (app.state.encoder([""]), app.state.preload()), name="lectern-warmup",
                     daemon=True).start()
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
