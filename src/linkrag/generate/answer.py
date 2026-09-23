"""Cited answer synthesis over a retrieved evidence set.

baseline: the top-k units are pasted into the prompt in rank order and the model
          answers. No statement that they relate to one another.
linkrag:  the link-expanded, complementarity-reranked set, with the traversed
          links described so the model composes across modalities.

Both Ollama and any OpenAI-compatible endpoint are driven through the same
/v1/chat/completions call -- Ollama implements that shape, so "provider" only
selects a default base_url.
"""

from __future__ import annotations

import json
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import requests

from linkrag.core import EvidenceUnit, Mode, stage_timer

Completer = Callable[[str, str], str]
"""(system, user) -> assistant text. Injectable so tests never hit a server."""

PROVIDER_BASE_URLS = {
    "ollama": "http://localhost:11434/v1",
    "openai": "https://api.openai.com/v1",
    # "openai_compatible" (vLLM, Ollama behind a tunnel, ...) has no default: the
    # Colab backend sets models.llm_backends.<name>.base_url explicitly.
}

SYSTEM = """You answer questions using ONLY the numbered evidence given to you.

Rules:
- Cite the evidence for every claim, using its exact id in square brackets: [id].
- An id looks like [notes:p2:t0] or [lecture03:a5]. Copy it exactly.
- If the evidence does not answer the question, say so plainly. Do not guess and
  do not use outside knowledge -- an uncited or unsupported claim is a failure.
- Be concise."""

JSON_SYSTEM = """You answer questions using ONLY the evidence given to you, as JSON.

Output EXACTLY this shape and nothing else -- no prose before or after, no code fence:

{"answer": "<one or two sentences>",
 "claims": [{"claim": "<a single checkable statement>", "unit_ids": ["<id>", ...]}, ...]}

Rules:
- Every claim must be supported by the evidence, and `unit_ids` must list the units
  that state it, copied exactly (they look like notes:p2:t0 or lecture03:a5).
- Split the answer into one claim per checkable fact; do not merge two facts.
- Never use outside knowledge. If the evidence does not answer the question, return
  {"answer": "not found in the provided material", "claims": []}."""

LINKRAG_SYSTEM_SUFFIX = """
- The evidence spans several files and modalities and has been linked as covering
  the same content. Prefer an answer that combines them over one that picks a
  single unit."""

CITATION_RE = re.compile(r"\[([A-Za-z0-9_.:\-]+)\]")


def format_evidence(units: list[EvidenceUnit]) -> str:
    """Each unit tagged [id, modality, location] so the model can cite it."""
    lines = []
    for unit in units:
        where = f"{Path(unit.source_file).name} {unit.location.cite()}"
        content = unit.content.strip() or "(no text extracted)"
        lines.append(f"[{unit.id}, {unit.modality}, {where}]\n{content}")
    return "\n\n".join(lines)


def build_prompt(question: str, units: list[EvidenceUnit]) -> str:
    return f"Evidence:\n\n{format_evidence(units)}\n\nQuestion: {question}\n\nAnswer with citations:"


def cited_ids(answer_text: str) -> list[str]:
    """Ids the answer actually cited, in order of first appearance."""
    seen: dict[str, None] = {}
    for match in CITATION_RE.findall(answer_text):
        seen.setdefault(match, None)
    return list(seen)


class MissingApiKey(SystemExit):
    """A backend's key -- or, for `base_url_env` backends such as colab, its URL -- is
    not in the environment. SystemExit, so a script stops with this
    one line on stderr (exit 1, no traceback). Nothing falls back to another backend --
    least of all a billing one."""


def judge_completer(cfg: dict[str, Any], backend: str) -> Completer:
    """`eval.judge` when entailment runs on the llm backend. Under nli (the ablation) the
    judge is never called, so no key is demanded; a call anyway is a bug and raises."""
    if backend == "llm":
        return http_completer((cfg.get("eval") or {}).get("judge") or cfg["models"]["llm"])

    def unused(system: str, user: str) -> str:
        raise RuntimeError(f"judge called while eval.entailment.backend is {backend!r}")
    unused.usage = {"calls": 0, "cached_calls": 0, "prompt_tokens": 0, "completion_tokens": 0,  # type: ignore[attr-defined]
                    "estimated_tokens": 0}
    return unused


def http_completer(llm_cfg: dict[str, Any], pricing: dict[str, Any] | None = None) -> Completer:
    """OpenAI-compatible /chat/completions. Works for Ollama, vLLM, OpenAI, Groq,
    Google AI Studio, etc.

    Every request passes two gates first: the spend guard (`linkrag.costs.SpendGuard`
    -- zero-cost mode refuses anything that would bill past `max_cost_usd`, default 0)
    and, when the backend declares `rate_limit_rpm` / `rate_limit_tpm`, a shared
    one-minute rate limiter."""
    import os

    from linkrag.costs import RateLimiter, SpendGuard

    provider = llm_cfg.get("provider", "ollama")
    backend = str(llm_cfg.get("backend") or provider)
    url_env = llm_cfg.get("base_url_env")
    if url_env:                       # e.g. colab: the tunnel URL changes every session
        env_url = os.environ.get(url_env, "").strip()
        if not env_url:
            raise MissingApiKey(f"{url_env} is not set: backend {backend!r} needs the URL printed by "
                                f"notebooks/colab_serve.ipynb (export {url_env}=https://.../v1). Not falling back.")
        llm_cfg = {**llm_cfg, "base_url": env_url}
    base_url = (llm_cfg.get("base_url") or PROVIDER_BASE_URLS.get(provider, "")).rstrip("/")
    if not base_url:
        raise ValueError(f"no base_url for provider {provider!r}; set models.llm.base_url")
    model = llm_cfg.get("model")
    if not model:
        raise ValueError("models.llm.model is not set in the config")

    headers = {"Content-Type": "application/json"}
    key_env = llm_cfg.get("api_key_env")
    if key_env:
        key = os.environ.get(key_env)
        if not key:
            raise MissingApiKey(f"{key_env} is not set: backend {llm_cfg.get('backend', llm_cfg.get('provider'))!r} "
                                f"({llm_cfg.get('model')}) needs it -- export it in ~/.zshenv. Not falling back.")
        headers["Authorization"] = f"Bearer {key}"

    limit = llm_cfg.get("max_tokens", 1024)
    usage = {"calls": 0, "cached_calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
             "estimated_tokens": 0, "backend": backend}
    # Older models take `max_tokens`; newer ones reject it and require
    # `max_completion_tokens`. Model *names* are not a usable signal for which --
    # the families change faster than any prefix list survives -- so discover it
    # once from the server's own error and remember the answer.
    token_param = "max_tokens"
    guard = SpendGuard(llm_cfg, pricing if pricing is not None else llm_cfg.get("_pricing"))
    limiter = RateLimiter.for_backend(base_url, str(model), llm_cfg.get("rate_limit_rpm"),
                                      llm_cfg.get("rate_limit_tpm"))

    def post(payload: dict[str, Any]):
        return requests.post(
            f"{base_url}/chat/completions",
            json=payload,
            headers=headers,
            timeout=llm_cfg.get("timeout_s", 120),
        )

    def complete(system: str, user: str) -> str:
        nonlocal token_param
        est_prompt = (len(system) + len(user)) // 4 + 8
        guard.check(est_prompt, int(limit))            # refuse BEFORE anything is sent
        if limiter is not None:
            limiter.acquire(est_prompt + int(limit))
        payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            token_param: limit,
        }
        # temperature: null omits the field, for models that only allow their
        # default. Never silently substitute a different temperature -- this is an
        # eval harness and determinism is a property the results depend on.
        if llm_cfg.get("temperature") is not None:
            payload["temperature"] = llm_cfg["temperature"]
        # seed is best-effort on the OpenAI API, not a guarantee: the server returns
        # system_fingerprint to signal backend stability and this account gets None.
        # Send it anyway -- it is the only reproducibility lever available -- but
        # never claim determinism from its presence. Measure it.
        if llm_cfg.get("seed") is not None:
            payload["seed"] = llm_cfg["seed"]

        try:
            # A read timeout is transient too: retry it a few times (1s, 2s, 4s) before
            # giving up, so one slow reply does not kill a 1,000-call batch.
            timeout_retries = int(llm_cfg.get("timeout_retries", 3))
            for attempt in range(timeout_retries + 1):
                try:
                    response = post(payload)
                    break
                except requests.Timeout:
                    if attempt == timeout_retries:
                        raise
                    time.sleep(2 ** attempt + random.uniform(0, 0.5))
            # Test the payload, not the shared `token_param`: with concurrent callers
            # another thread may already have flipped it, and this request -- built
            # before the flip -- would then raise instead of retrying.
            if (
                response.status_code == 400
                and "max_tokens" in payload
                and "max_completion_tokens" in response.text
            ):
                token_param = "max_completion_tokens"
                payload[token_param] = payload.pop("max_tokens")
                response = post(payload)
            # 429 / 5xx: back off and retry. Hosted judges have per-minute token
            # limits and the verifier runs eight workers; without this one throttled
            # call killed a 1,000-call batch.
            for attempt in range(llm_cfg.get("max_retries", 12)):
                if response.status_code not in (429, 500, 502, 503, 504):
                    break
                retry_after = response.headers.get("Retry-After")
                wait = float(retry_after) if retry_after else min(2 ** attempt, 60)
                time.sleep(wait + random.uniform(0, 1.0))
                response = post(payload)
        except requests.RequestException as exc:
            # RequestException, not ConnectionError: a read timeout is just as fatal
            # to the call and was previously an unhandled traceback that killed a
            # whole batch comparison mid-run.
            hint = " Is `ollama serve` running?" if provider == "ollama" else ""
            raise RuntimeError(
                f"LLM call to {base_url} failed ({type(exc).__name__}).{hint}") from exc

        if response.status_code != 200:
            raise RuntimeError(f"LLM returned {response.status_code}: {response.text[:300]}")
        body = response.json()
        u = body.get("usage") or {}
        last = {"prompt_tokens": int(u.get("prompt_tokens", 0)),
                "completion_tokens": int(u.get("completion_tokens", 0))}
        complete.last_usage = last  # type: ignore[attr-defined]
        usage["calls"] += 1
        usage["prompt_tokens"] += last["prompt_tokens"]
        usage["completion_tokens"] += last["completion_tokens"]
        guard.add(last["prompt_tokens"], last["completion_tokens"])
        return body["choices"][0]["message"]["content"]

    complete.usage = usage        # type: ignore[attr-defined]
    complete.last_usage = {}      # type: ignore[attr-defined]
    complete.model = model        # type: ignore[attr-defined]
    return complete


@dataclass
class Claim:
    """One checkable statement from the answer, with the units it cites."""

    claim: str
    unit_ids: list[str]
    verdict: str = "unchecked"          # supported | weak | unsupported | unchecked
    span: str = ""
    votes: list[str] = field(default_factory=list)


@dataclass
class Answer:
    answer: str
    claims: list[Claim]
    raw: str
    malformed: bool = False             # JSON could not be parsed even after one retry

    @property
    def unsupported(self) -> list[Claim]:
        return [c for c in self.claims if c.verdict == "unsupported"]

    def text(self, *, strict: bool = False) -> str:
        """The answer as shown to a reader. `strict` drops unsupported claims and
        abstains when nothing survives."""
        if strict and self.claims and not [c for c in self.claims if c.verdict in ("supported", "weak", "unchecked")]:
            return "not found in the provided material"
        return self.answer


def parse_answer_json(text: str) -> tuple[str, list[Claim]] | None:
    """Tolerate a fenced or prefixed reply; return None when it is not usable."""
    body = text.strip()
    if body.startswith("```"):
        body = re.sub(r"^```[a-z]*\s*|\s*```$", "", body)
    start, end = body.find("{"), body.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        data = json.loads(body[start:end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict) or "answer" not in data:
        return None
    claims = []
    for c in data.get("claims") or []:
        if not isinstance(c, dict) or not str(c.get("claim", "")).strip():
            continue
        ids = c.get("unit_ids") or c.get("units") or []
        if isinstance(ids, str):
            ids = [ids]
        claims.append(Claim(claim=str(c["claim"]).strip(), unit_ids=[str(i).strip() for i in ids if str(i).strip()]))
    return str(data["answer"]).strip(), claims


def answer_json(
    question: str,
    units: list[EvidenceUnit],
    cfg: dict[str, Any] | None = None,
    *,
    mode: Mode = "baseline",
    complete: Completer | None = None,
) -> Answer:
    """Structured answer: prose plus one claim per checkable fact, each citing units.

    A malformed reply is retried once with the parse failure fed back; a second
    failure is flagged (`Answer.malformed`) and the raw text is kept as the answer,
    so a downstream verifier can still see what was said."""
    if not units:
        return Answer(answer="No evidence was retrieved for this question, so I cannot answer it.",
                      claims=[], raw="")
    llm_cfg = (cfg or {}).get("models", {}).get("llm", {})
    complete = complete or http_completer(llm_cfg)
    system = JSON_SYSTEM + (LINKRAG_SYSTEM_SUFFIX if mode == "linkrag" else "")
    prompt = build_prompt(question, units)
    with stage_timer("generate.answer_json", mode=mode, units=len(units)) as t:
        raw = complete(system, prompt)
        parsed = parse_answer_json(raw)
        t["retried"] = 0
        if parsed is None:
            t["retried"] = 1
            raw = complete(system, prompt + "\n\nYour previous reply was not valid JSON. "
                                            "Reply with the JSON object only.")
            parsed = parse_answer_json(raw)
        if parsed is None:
            t["malformed"] = 1
            return Answer(answer=raw.strip(), claims=[], raw=raw, malformed=True)
        text_, claims = parsed
        t["claims"] = len(claims)
        # keep only ids that are actually in the evidence set: a cited id that was
        # never shown is a fabrication, and must not become a "supported" claim.
        valid = {u.id for u in units}
        for c in claims:
            c.unit_ids = [i for i in c.unit_ids if i in valid]
        return Answer(answer=text_, claims=claims, raw=raw)


def answer(
    question: str,
    units: list[EvidenceUnit],
    cfg: dict[str, Any] | None = None,
    *,
    mode: Mode = "baseline",
    complete: Completer | None = None,
) -> str:
    if not units:
        return "No evidence was retrieved for this question, so I cannot answer it."

    llm_cfg = (cfg or {}).get("models", {}).get("llm", {})
    complete = complete or http_completer(llm_cfg)

    system = SYSTEM + (LINKRAG_SYSTEM_SUFFIX if mode == "linkrag" else "")
    with stage_timer("generate.answer", mode=mode, units=len(units)) as t:
        text = complete(system, build_prompt(question, units))
        t["chars"] = len(text)
        t["citations"] = len(cited_ids(text))
    return text
