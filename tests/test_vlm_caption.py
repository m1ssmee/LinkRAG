"""Optional VLM figure descriptions. The model is never called in tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import requests

from linkrag.ingest.vlm_caption import describe_if_enabled, describe_image, vlm_config


class _Resp:
    def __init__(self, status: int, payload: dict) -> None:
        self.status_code, self._p, self.text = status, payload, json.dumps(payload)

    def json(self) -> dict:
        return self._p


@pytest.fixture
def image(tmp_path: Path) -> Path:
    from PIL import Image

    p = tmp_path / "fig.png"
    Image.new("RGB", (40, 20), "white").save(p)
    return p


def test_disabled_by_default_and_never_calls_the_model(image, monkeypatch) -> None:
    def explode(*a, **k):
        raise AssertionError("the VLM must not be called when disabled")

    monkeypatch.setattr(requests, "post", explode)
    assert describe_if_enabled(image, {"ingest": {}}) == ""
    assert vlm_config(None)["enabled"] is False


def test_description_is_truncated_to_max_words(image, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(requests, "post",
                        lambda *a, **k: _Resp(200, {"response": " ".join(["word"] * 200)}))
    text = describe_image(image, max_words=60, cache_dir=tmp_path / "c")
    assert len(text.split()) == 60


def test_result_is_cached_by_image_content(image, tmp_path, monkeypatch) -> None:
    calls = []

    def once(*a, **k):
        calls.append(1)
        return _Resp(200, {"response": "a bar chart of recall against K"})

    monkeypatch.setattr(requests, "post", once)
    cache = tmp_path / "c"
    first = describe_image(image, cache_dir=cache)
    second = describe_image(image, cache_dir=cache)
    assert first == second == "a bar chart of recall against K"
    assert len(calls) == 1, "second call must be served from cache"


def test_unreachable_model_degrades_to_empty_not_an_exception(image, tmp_path, monkeypatch) -> None:
    """A missing Ollama must not abort a corpus build halfway through."""
    def refuse(*a, **k):
        raise requests.ConnectionError("refused")

    monkeypatch.setattr(requests, "post", refuse)
    assert describe_image(image, cache_dir=tmp_path / "c") == ""


def test_error_status_degrades_to_empty(image, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(requests, "post", lambda *a, **k: _Resp(500, {"error": "boom"}))
    assert describe_image(image, cache_dir=tmp_path / "c") == ""
