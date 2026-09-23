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


ENDPOINT_KEYS = ("provider", "base_url", "base_url_env", "api_key_env", "billing")


def load_config(path: str | Path = "configs/default.yaml") -> dict[str, Any]:
    """YAML config with the active LLM backend resolved into `models.llm`.

    `models.llm` is the working backend (OpenAI). `models.llm_backends.<name>`
    holds alternatives -- the Colab-served open model for Phase 8 -- selected by
    `models.llm.backend: <name>` or the env var LINKRAG_LLM_BACKEND. Selecting one
    overlays its keys onto `models.llm`, so every script keeps one code path.
    """
    import os

    cfg = yaml.safe_load(Path(path).read_text())
    backends = cfg.get("models", {}).get("llm_backends") or {}
    # the judge can sit on a backend too (e.g. Groq judging a Colab answerer): the
    # zero-cost route for `eval.entailment.backend: llm`
    for parent, key, env in ((cfg.get("models", {}), "llm", "LINKRAG_LLM_BACKEND"),
                             (cfg.get("eval") or {}, "judge", "LINKRAG_JUDGE_BACKEND")):
        block = parent.get(key) or {}
        name = os.environ.get(env) or block.get("backend")
        if name and name != "default":
            if name not in backends:
                raise KeyError(f"models.llm_backends has no entry {name!r} "
                               f"(have: {sorted(backends)})")
            # Endpoint, credential and billing come only from the selected backend: never
            # send the base block's key (e.g. OPENAI_API_KEY) to another backend's URL.
            base = {k: v for k, v in block.items() if k not in ENDPOINT_KEYS}
            parent[key] = {**base, **backends[name], "backend": name}
    # Zero-cost mode: every completer built from this config is budgeted at
    # `cost.max_usd` (default 0 -- refuse anything that would bill). A script's
    # --max-cost raises it via `set_max_cost`. Pricing rides along so the guard can
    # price a call without the caller threading it through.
    budget = float((cfg.get("cost") or {}).get("max_usd", 0.0) or 0.0)
    pricing = cfg.get("models", {}).get("pricing") or {}
    for block in (cfg.get("models", {}).get("llm"), (cfg.get("eval") or {}).get("judge")):
        if isinstance(block, dict):
            block.setdefault("max_cost_usd", budget)
            block["_pricing"] = pricing
    return cfg


def set_max_cost(cfg: dict[str, Any], usd: float) -> None:
    """Raise (or lower) the run budget: every LLM block, and `cost.max_usd`, which the
    whisper-1 pre-upload check reads."""
    cfg.setdefault("cost", {})["max_usd"] = float(usd)
    for block in (cfg.get("models", {}).get("llm"), (cfg.get("eval") or {}).get("judge")):
        if isinstance(block, dict):
            block["max_cost_usd"] = float(usd)


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
