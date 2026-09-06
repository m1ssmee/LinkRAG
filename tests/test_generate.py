from __future__ import annotations

import pytest

from linkrag.core import EvidenceUnit, Location
from linkrag.generate.answer import (
    LINKRAG_SYSTEM_SUFFIX,
    answer,
    build_prompt,
    cited_ids,
    format_evidence,
    http_completer,
)

UNITS = [
    EvidenceUnit(
        id="notes:p2:t0", modality="text",
        content="Self-attention assigns each token a weight over all other tokens.",
        source_file="data/raw/notes.pdf", location=Location(page=2, bbox=(1.0, 2.0, 3.0, 4.0)),
    ),
    EvidenceUnit(
        id="lecture03:a5", modality="audio",
        content="as you can see here the weights concentrate on the subject token",
        source_file="data/raw/lecture03.wav", location=Location(start_s=612.0, end_s=628.5),
    ),
    EvidenceUnit(
        id="slides03:p14:f0", modality="figure", content="",
        source_file="data/raw/slides03.pdf", location=Location(page=14),
    ),
]


@pytest.fixture
def spy():
    """Mock LLM: records the prompt it was given, returns a canned citation."""
    calls: list[tuple[str, str]] = []

    def complete(system: str, user: str) -> str:
        calls.append((system, user))
        return "Attention weights each token [notes:p2:t0], as the lecturer says [lecture03:a5]."

    complete.calls = calls
    return complete


# ------------------------------------------------------------------ prompting

def test_format_evidence_tags_id_modality_and_location() -> None:
    text = format_evidence(UNITS)
    assert "[notes:p2:t0, text, notes.pdf p.2]" in text
    assert "[lecture03:a5, audio, lecture03.wav 612.0s-628.5s]" in text
    assert "[slides03:p14:f0, figure, slides03.pdf p.14]" in text


def test_format_evidence_marks_an_empty_unit_rather_than_dropping_it() -> None:
    assert "(no text extracted)" in format_evidence([UNITS[2]])


def test_build_prompt_carries_the_question_and_every_id() -> None:
    prompt = build_prompt("What is attention?", UNITS)
    assert "What is attention?" in prompt
    assert all(u.id in prompt for u in UNITS)


# ------------------------------------------------------------------ citations

def test_cited_ids_dedupes_and_preserves_order() -> None:
    text = "First [b:p1:t0] then [a:p2:t9] then [b:p1:t0] again."
    assert cited_ids(text) == ["b:p1:t0", "a:p2:t9"]


def test_cited_ids_on_an_uncited_answer() -> None:
    assert cited_ids("The evidence does not answer this question.") == []


# --------------------------------------------------------------------- answer

def test_answer_calls_the_llm_and_returns_its_text(spy) -> None:
    text = answer("What is attention?", UNITS, complete=spy)
    assert "[notes:p2:t0]" in text
    assert len(spy.calls) == 1


def test_answer_instructs_the_model_to_cite(spy) -> None:
    answer("What is attention?", UNITS, complete=spy)
    system, user = spy.calls[0]
    assert "[id]" in system and "ONLY" in system
    assert "notes:p2:t0" in user


def test_baseline_mode_omits_the_cross_modal_instruction(spy) -> None:
    answer("q", UNITS, mode="baseline", complete=spy)
    assert LINKRAG_SYSTEM_SUFFIX not in spy.calls[0][0]


def test_linkrag_mode_adds_the_cross_modal_instruction(spy) -> None:
    answer("q", UNITS, mode="linkrag", complete=spy)
    assert LINKRAG_SYSTEM_SUFFIX in spy.calls[0][0]


def test_answer_short_circuits_on_empty_evidence(spy) -> None:
    text = answer("What is attention?", [], complete=spy)
    assert "cannot answer" in text
    assert spy.calls == [], "no evidence must not cost an LLM call"


def test_every_cited_id_is_in_the_evidence(spy) -> None:
    """The check scripts/ask.py runs: a citation outside the evidence set is a
    hallucinated source, which is exactly what P2 does not check for."""
    text = answer("q", UNITS, complete=spy)
    assert set(cited_ids(text)) <= {u.id for u in UNITS}


# --------------------------------------------------------------------- config

def test_http_completer_accepts_an_ollama_config() -> None:
    assert http_completer({"provider": "ollama", "model": "llama3.1:8b"}) is not None


def test_http_completer_requires_an_explicit_model() -> None:
    """No guessed default: silently querying the wrong model against a paid API
    would corrupt the pilot numbers without erroring."""
    with pytest.raises(ValueError, match="models.llm.model is not set"):
        http_completer({"provider": "openai", "base_url": "https://api.openai.com/v1"})


def test_http_completer_rejects_an_unknown_provider() -> None:
    with pytest.raises(ValueError, match="no base_url"):
        http_completer({"provider": "mystery"})


def test_http_completer_requires_the_named_api_key(monkeypatch) -> None:
    monkeypatch.delenv("TEST_LLM_KEY", raising=False)
    with pytest.raises(RuntimeError, match="TEST_LLM_KEY is not set"):
        http_completer({"provider": "openai", "model": "m", "api_key_env": "TEST_LLM_KEY"})


def test_http_completer_reports_an_unreachable_server(monkeypatch) -> None:
    import requests

    def boom(*a, **k):
        raise requests.ConnectionError("refused")

    monkeypatch.setattr(requests, "post", boom)
    complete = http_completer(
        {"provider": "ollama", "model": "llama3.1:8b", "base_url": "http://localhost:11434/v1"}
    )
    with pytest.raises(RuntimeError, match="ollama serve"):
        complete("sys", "user")


# ------------------------------------------------- OpenAI parameter drift

class _Resp:
    def __init__(self, status: int, payload: dict | None = None, text: str = "") -> None:
        self.status_code = status
        self._payload = payload or {}
        self.text = text or str(payload)

    def json(self) -> dict:
        return self._payload


OK = {"choices": [{"message": {"content": "Blue [n:p1:t0]."}}]}
REJECT = _Resp(400, {"error": {"param": "max_tokens"}},
               "Unsupported parameter: 'max_tokens' is not supported with this model. "
               "Use 'max_completion_tokens' instead.")


def _capture(monkeypatch, responses: list[_Resp]) -> list[dict]:
    """Record each payload posted; reply with the queued responses in order."""
    import requests

    sent: list[dict] = []

    def fake_post(url, json, headers, timeout):
        # snapshot: the retry mutates the payload in place, so a stored
        # reference would show the final state for every recorded call
        sent.append(dict(json))
        return responses[len(sent) - 1]

    monkeypatch.setattr(requests, "post", fake_post)
    return sent


def test_completer_retries_with_max_completion_tokens(monkeypatch) -> None:
    """Newer OpenAI models reject max_tokens. Model names are not a usable signal
    for which, so the parameter is discovered from the server's own 400."""
    sent = _capture(monkeypatch, [REJECT, _Resp(200, OK)])
    complete = http_completer({"provider": "openai", "model": "gpt-5.4", "max_tokens": 64,
                               "base_url": "https://api.openai.com/v1"})
    assert complete("sys", "user") == "Blue [n:p1:t0]."
    assert "max_tokens" in sent[0] and "max_completion_tokens" not in sent[0]
    assert sent[1]["max_completion_tokens"] == 64 and "max_tokens" not in sent[1]


def test_completer_remembers_the_parameter_after_one_probe(monkeypatch) -> None:
    sent = _capture(monkeypatch, [REJECT, _Resp(200, OK), _Resp(200, OK)])
    complete = http_completer({"provider": "openai", "model": "gpt-5.4", "max_tokens": 64,
                               "base_url": "https://api.openai.com/v1"})
    complete("s", "u")
    complete("s", "u")
    assert len(sent) == 3, "the second call must not re-probe"
    assert "max_completion_tokens" in sent[2]


def test_completer_omits_a_null_temperature(monkeypatch) -> None:
    """Some models allow only their default temperature. Omitting is correct;
    silently substituting one would break determinism the eval depends on."""
    sent = _capture(monkeypatch, [_Resp(200, OK)])
    http_completer({"provider": "openai", "model": "m", "temperature": None,
                    "base_url": "https://x/v1"})("s", "u")
    assert "temperature" not in sent[0]


def test_completer_sends_an_explicit_temperature(monkeypatch) -> None:
    sent = _capture(monkeypatch, [_Resp(200, OK)])
    http_completer({"provider": "openai", "model": "m", "temperature": 0.0,
                    "base_url": "https://x/v1"})("s", "u")
    assert sent[0]["temperature"] == 0.0


def test_completer_surfaces_a_non_token_400(monkeypatch) -> None:
    """A temperature rejection must fail loudly, not get retried into silence."""
    _capture(monkeypatch, [_Resp(400, {"error": {"param": "temperature"}},
                                 "'temperature' does not support 0.0 with this model")])
    complete = http_completer({"provider": "openai", "model": "gpt-5.5", "temperature": 0.0,
                               "base_url": "https://api.openai.com/v1"})
    with pytest.raises(RuntimeError, match="temperature"):
        complete("s", "u")
