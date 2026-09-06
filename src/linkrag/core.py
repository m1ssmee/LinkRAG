"""Core data model shared by every LinkRAG stage.

An EvidenceUnit is the atom of retrieval: one chunk of one modality, with
enough provenance to cite it. A Link is a typed, scored edge between two
EvidenceUnits -- the Evidence Linking Layer's output and the thing that
separates LinkRAG from the per-modality-independent baselines.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Literal

import yaml

Mode = Literal["baseline", "linkrag"]
"""Every stage takes a `mode`. "baseline" reproduces what P1/P2/P3 do,
"linkrag" uses our links. Keeping both in one code path is what makes the
ablations honest -- see DESIGN.md."""

Modality = Literal["text", "figure", "table", "audio"]

LinkType = Literal[
    "audio_slide",   # spoken segment <-> the slide/page it covers
    "figure_text",   # figure/table <-> the prose that explains it
    "deictic",       # "this arrow here" <-> the visual element referred to
    "same_slide",    # figure <-> the text of the deck page it sits on
]


@dataclass
class Location:
    """Where a unit came from. Docs use page/bbox, audio uses start_s/end_s."""

    page: int | None = None
    bbox: tuple[float, float, float, float] | None = None  # (x0, y0, x1, y1)
    start_s: float | None = None
    end_s: float | None = None

    def cite(self) -> str:
        if self.start_s is not None:
            return f"{self.start_s:.1f}s-{self.end_s:.1f}s"
        return f"p.{self.page}" if self.page is not None else "?"


@dataclass
class EvidenceUnit:
    id: str
    modality: Modality
    content: str
    source_file: str
    location: Location = field(default_factory=Location)
    # Note: plain list, not np.ndarray -- keeps dataclasses.asdict/json
    # round-tripping free. Switch to a shared matrix if RAM becomes the limit.
    embedding: list[float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Link:
    src_id: str
    dst_id: str
    link_type: LinkType
    score: float
    # Why the link was made: the deictic phrase for "deictic", the matched
    # reference for "figure_text". Retrieval needs it to explain an evidence set,
    # and the eval needs it to score whether the *reason* was right, not just the
    # endpoints.
    metadata: dict[str, Any] = field(default_factory=dict)


def load_config(path: str | Path = "configs/default.yaml") -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text())


# ---------------------------------------------------------------- timing

log = logging.getLogger("linkrag")


@contextmanager
def stage_timer(stage: str, **fields: Any) -> Iterator[dict[str, Any]]:
    """Log wall-clock for one pipeline stage.

    Yields a dict the caller can drop counts into; they get logged with the
    timing, so every stage reports "what it did" alongside "how long it took".

        with stage_timer("ingest.pdf", file=path) as t:
            units = ...
            t["units"] = len(units)
    """
    fields = dict(fields)
    start = time.perf_counter()
    try:
        yield fields
    finally:
        elapsed = time.perf_counter() - start
        extra = " ".join(f"{k}={v}" for k, v in fields.items())
        log.info("%-22s %7.2fs  %s", stage, elapsed, extra)


# These log every HTTP request at INFO and bury the stage timings.
NOISY_LOGGERS = ("httpx", "httpcore", "urllib3", "filelock", "sentence_transformers", "faster_whisper")


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)-5s %(message)s", datefmt="%H:%M:%S")
    for name in NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
