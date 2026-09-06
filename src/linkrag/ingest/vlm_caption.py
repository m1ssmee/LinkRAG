"""Optional VLM descriptions for extracted figures, via a local Ollama vision model.

Off by default. A 7B VLM on CPU takes tens of seconds per image, which is fine for a
27-slide deck and unusable for a large corpus -- so this is opt-in
(`ingest.vlm_captions.enabled`) and every result is cached by image content hash, so
re-ingesting never pays twice.

Why it exists: on pilot01, 13 of 18 extracted figures had no caption and OCR
recovered only 5, mostly institution logos, because decks embed diagrams as raster
fragments with the labels in the PDF text layer. A VLM description is the only way to
give those figures any text of their own -- which is what `figure_text`'s semantic
term and `deictic`'s keyword overlap both need.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import requests

log = logging.getLogger("linkrag")

PROMPT = (
    "Describe this figure from a technical talk in at most {max_words} words. "
    "State what kind of visual it is (plot, diagram, screenshot, logo, photo), what "
    "it shows, and any axis labels, series names or numbers you can read. "
    "Do not speculate beyond what is visible. Reply with the description only."
)


def _cache_key(image: Path, model: str, max_words: int) -> str:
    digest = hashlib.sha256(image.read_bytes()).hexdigest()[:24]
    return f"{digest}-{model.replace(':', '_')}-{max_words}"


def describe_image(
    image_path: str | Path,
    *,
    model: str = "llava:7b",
    base_url: str = "http://localhost:11434",
    max_words: int = 60,
    cache_dir: str | Path | None = "data/processed/vlm_captions",
    timeout_s: int = 300,
) -> str:
    """One figure -> one short description. Returns "" if the VLM is unreachable.

    Never raises on a transport error: a missing Ollama must degrade ingestion to
    "no VLM text", not abort a corpus build halfway through.
    """
    image_path = Path(image_path)
    cache_file = None
    if cache_dir:
        cache_file = Path(cache_dir) / f"{_cache_key(image_path, model, max_words)}.json"
        if cache_file.exists():
            return json.loads(cache_file.read_text())["description"]

    payload = {
        "model": model,
        "prompt": PROMPT.format(max_words=max_words),
        "images": [base64.b64encode(image_path.read_bytes()).decode()],
        "stream": False,
    }
    try:
        response = requests.post(f"{base_url.rstrip('/')}/api/generate",
                                 json=payload, timeout=timeout_s)
    except requests.RequestException as exc:
        log.warning("VLM unreachable at %s (%s); continuing without a description",
                    base_url, type(exc).__name__)
        return ""
    if response.status_code != 200:
        log.warning("VLM returned %s: %s", response.status_code, response.text[:200])
        return ""

    text = " ".join(str(response.json().get("response", "")).split())
    words = text.split()
    if len(words) > max_words:
        text = " ".join(words[:max_words])

    if cache_file is not None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(
            {"image": str(image_path), "model": model, "description": text}))
    return text


def vlm_config(cfg: dict[str, Any] | None) -> dict[str, Any]:
    section = ((cfg or {}).get("ingest", {}) or {}).get("vlm_captions", {}) or {}
    return {
        "enabled": bool(section.get("enabled", False)),
        "model": section.get("model", "llava:7b"),
        "base_url": section.get("base_url", "http://localhost:11434"),
        "max_words": int(section.get("max_words", 60)),
        "cache_dir": section.get("cache_dir", "data/processed/vlm_captions"),
        "timeout_s": int(section.get("timeout_s", 300)),
    }


def describe_if_enabled(image_path: str | Path, cfg: dict[str, Any] | None) -> str:
    settings = vlm_config(cfg)
    if not settings.pop("enabled"):
        return ""
    return describe_image(image_path, **settings)
