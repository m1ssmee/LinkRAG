"""Token and cost accounting for every LLM-touching run, plus the reply cache.

Every script that calls a model wraps its completer(s) with `cached_completer` and
ends with `record_run(...)`, which appends one row to `reports/llm_ledger.jsonl` and
returns the footer lines (this run + cumulative) that go under the report table.

Token counts come from the server's `usage` block and are exact. A cache hit whose
usage was never stored (calls made before this module existed) is *estimated* at
4 characters per token and flagged as such, so a backfilled row is never mistaken
for a measured one.
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

Completer = Callable[[str, str], str]

LEDGER = Path("reports/llm_ledger.jsonl")
CHARS_PER_TOKEN = 4.0        # estimate for backfilled cache hits only


USAGE_KEYS = ("calls", "cached_calls", "prompt_tokens", "completion_tokens",
              "cached_prompt_tokens", "cached_completion_tokens", "estimated_tokens",
              "audio_seconds", "cached_audio_seconds")


def empty_usage() -> dict[str, Any]:
    return {k: 0 for k in USAGE_KEYS}


def add_usage(total: dict[str, Any], one: dict[str, Any]) -> None:
    for k in USAGE_KEYS:
        total[k] = total.get(k, 0) + int(one.get(k, 0))


def estimate_tokens(text: str) -> int:
    return int(round(len(text) / CHARS_PER_TOKEN))


class CacheMiss(RuntimeError):
    """Raised by a cache-only completer when a prompt has never been answered."""


def cached_completer(inner: Completer, cache_dir: Path, *, cache_only: bool = False,
                     max_cost_usd: float | None = None, price: dict[str, float] | None = None) -> Completer:
    """Disk cache keyed by (system, user, n-th identical call). Thread-safe.

    The n-th index keeps repeated judge runs distinct calls while letting an
    interrupted batch resume for free. Under threads it is order-dependent: two
    callers with the same prompt can each get the other's stored reply. A caller
    that knows who is asking passes `key=(context, run)` instead: the reply is
    stored per context and run, so a replay is deterministic. A keyed miss first
    reads the pre-keying entry for that run (`<prompt>.<run>`), so replies paid for
    before keying stay usable, and the same entry is read for every context. Usage is stored next to each reply
    (`<key>.<n>.usage.json`) so a later hit reports exact tokens; hits without it
    are estimated and counted under `estimated_tokens`.

    `cache_only=True` never calls the model: a miss raises `CacheMiss`, so a report
    can be filled from what was already paid for and nothing else.
    `max_cost_usd` (with `price`) stops the run with a RuntimeError once the
    uncached spend of this process crosses the cap -- the guard that was missing
    when a batch cost more than it should have (2026-09-22).
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    seen: Counter = Counter()
    lock = threading.Lock()
    usage = empty_usage()
    if getattr(inner, "usage", {}).get("backend"):    # carried through to the ledger row
        usage["backend"] = inner.usage["backend"]       # type: ignore[attr-defined]

    def complete(system: str, user: str, key: tuple[str, int] | None = None) -> str:
        base = hashlib.sha256((system + "\x00" + user).encode()).hexdigest()
        if key is None:
            with lock:
                n = seen[base]
                seen[base] += 1
            stem, legacy = f"{base}.{n}", None
        else:
            ctx, n = key
            stem, legacy = f"{base}.{hashlib.sha256(ctx.encode()).hexdigest()[:12]}.{n}", f"{base}.{n}"
        path = cache_dir / f"{stem}.txt"
        upath = cache_dir / f"{stem}.usage.json"
        hit = stem if path.exists() else legacy if legacy and (cache_dir / f"{legacy}.txt").exists() else None
        if hit is not None:
            text = (cache_dir / f"{hit}.txt").read_text()
            uhit = cache_dir / f"{hit}.usage.json"
            if uhit.exists():
                u = json.loads(uhit.read_text())
                pt, ct, est = u.get("prompt_tokens", 0), u.get("completion_tokens", 0), 0
            else:
                pt, ct = estimate_tokens(system + user), estimate_tokens(text)
                est = pt + ct
            one = {"calls": 1, "cached_calls": 1, "prompt_tokens": pt, "completion_tokens": ct,
                   "cached_prompt_tokens": pt, "cached_completion_tokens": ct,
                   "estimated_tokens": est}
            with lock:
                add_usage(usage, one)
            return text
        if cache_only:
            raise CacheMiss(f"no cached reply for this prompt ({base[:12]}…); cache-only mode makes no calls")
        if max_cost_usd is not None:
            spent = usage_cost(usage, price) or 0.0
            if spent >= max_cost_usd:
                raise RuntimeError(f"cost cap reached: ${spent:.2f} uncached spend >= max_cost_usd ${max_cost_usd:.2f}; "
                                   f"raise --max-cost to continue (everything so far is cached)")
        text = inner(system, user)
        last = getattr(inner, "last_usage", None) or {}
        path.write_text(text)
        if last:
            upath.write_text(json.dumps(last))
        with lock:
            add_usage(usage, {"calls": 1, "prompt_tokens": last.get("prompt_tokens", 0),
                              "completion_tokens": last.get("completion_tokens", 0)})
        return text

    complete.usage = usage        # type: ignore[attr-defined]
    complete.calls = seen         # type: ignore[attr-defined]
    return complete


# ----------------------------------------------------------------- pricing

def price_for(model: str, pricing: dict[str, Any] | None) -> dict[str, float] | None:
    """Longest-prefix match of `model` against the config pricing table."""
    if not pricing or not model:
        return None
    best = None
    for prefix, price in pricing.items():
        if model.startswith(prefix) and (best is None or len(prefix) > len(best[0])):
            best = (prefix, price)
    return None if best is None else best[1]


def usage_cost(usage: dict[str, Any], price: dict[str, float] | None,
               charge_cached: bool = False) -> float | None:
    """Spend for this run. Cache hits cost nothing unless `charge_cached` (used only
    when replaying a cache to backfill the spend of the original run)."""
    if price is None:
        return None
    pt = usage.get("prompt_tokens", 0) - (0 if charge_cached else usage.get("cached_prompt_tokens", 0))
    ct = usage.get("completion_tokens", 0) - (0 if charge_cached else usage.get("cached_completion_tokens", 0))
    secs = usage.get("audio_seconds", 0.0) - (0.0 if charge_cached else usage.get("cached_audio_seconds", 0.0))
    return ((pt * float(price.get("input_per_m", 0)) + ct * float(price.get("output_per_m", 0))) / 1e6
            + (secs / 60.0) * float(price.get("per_minute", 0)))


def fmt_cost(cost: float | None) -> str:
    return "price unknown" if cost is None else f"${cost:,.4f}"


# ----------------------------------------------------------------- ledger

def record_run(script: str, label: str, parts: list[tuple[str, dict[str, Any]]],
               pricing: dict[str, Any] | None, ledger: Path = LEDGER,
               charge_cached: bool = False) -> list[str]:
    """Append one ledger row per (model, usage) part; return footer lines.

    `parts` is [(model, usage), ...] -- a run may use an answerer and a judge.
    """
    ledger.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    lines = ["", "LLM cost (this run):", ""]
    run_total = 0.0
    run_known = True
    for model, usage in parts:
        price = price_for(model, pricing)
        cost = usage_cost(usage, price, charge_cached)
        row = {"utc": stamp, "script": script, "label": label, "model": model,
               **({"backend": usage["backend"]} if usage.get("backend") else {}),
               **{k: usage.get(k, 0) for k in USAGE_KEYS}, "cost_usd": cost}
        with ledger.open("a") as fh:
            fh.write(json.dumps(row) + "\n")
        est = usage.get("estimated_tokens", 0)
        note = f" (of which {est:,} tokens estimated from cache)" if est else ""
        cached_note = ""
        if usage.get("cached_calls") and not charge_cached and price is not None:
            would = usage_cost(usage, price, charge_cached=True)
            cached_note = f" (cache replays free; would have been {fmt_cost(would)})"
        body = (f"{usage['audio_seconds'] / 60:.1f} audio minutes" if usage.get("audio_seconds")
                else f"{usage.get('prompt_tokens', 0):,} in / {usage.get('completion_tokens', 0):,} out")
        lines.append(f"- `{model}`: {usage.get('calls', 0)} calls "
                     f"({usage.get('cached_calls', 0)} cached) · {body}{note} · {fmt_cost(cost)}{cached_note}")
        if cost is None:
            run_known = False
        else:
            run_total += cost
    lines.append(f"- run total: {fmt_cost(run_total) if run_known else fmt_cost(None)}")
    cum = cumulative(ledger)
    lines.append(f"- cumulative (all recorded runs, `{ledger}`): {cum['calls']:,} calls · "
                 f"{cum['prompt_tokens']:,} in / {cum['completion_tokens']:,} out · "
                 f"${cum['cost_usd']:,.2f}"
                 + (f" · {cum['unpriced_rows']} row(s) unpriced" if cum["unpriced_rows"] else "")
                 + (f" · {cum['estimated_tokens']:,} tokens estimated" if cum["estimated_tokens"] else ""))
    return lines


def cumulative(ledger: Path = LEDGER) -> dict[str, Any]:
    out = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0,
           "unpriced_rows": 0, "estimated_tokens": 0, "audio_seconds": 0.0, "rows": 0}
    if not ledger.exists():
        return out
    for line in ledger.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        out["rows"] += 1
        for k in ("calls", "prompt_tokens", "completion_tokens", "estimated_tokens"):
            out[k] += int(r.get(k, 0))
        out["audio_seconds"] += float(r.get("audio_seconds", 0.0))
        if r.get("cost_usd") is None:
            out["unpriced_rows"] += 1
        else:
            out["cost_usd"] += float(r["cost_usd"])
    return out


# ----------------------------------------------------------------- zero-cost guard

class BillingRefused(RuntimeError):
    """A call would bill and the budget for this run is exhausted (default budget: $0)."""


class SpendGuard:
    """Per-completer budget, checked BEFORE every request.

    `max_cost_usd` is the run's budget (config `cost.max_usd`, default 0, or a
    script's `--max-cost`). A backend declared `billing: free` (a free-tier key:
    Groq, Google AI Studio, a Colab-served open model) always passes, as does a
    local server (provider ollama or a localhost base_url) that declares nothing. Anything else
    is priced from `models.pricing`; with no price row it is treated as billing,
    because a $0 budget must not be spent on an unknown price.
    """

    def __init__(self, llm_cfg: dict[str, Any], pricing: dict[str, Any] | None):
        self.cfg = llm_cfg
        # a local server (ollama, or anything on localhost) cannot bill; say so explicitly
        # with `billing:` to override
        local = llm_cfg.get("provider") == "ollama" or any(
            h in str(llm_cfg.get("base_url", "")) for h in ("://localhost", "://127.0.0.1"))
        self.free = str(llm_cfg.get("billing", "free" if local else "metered")) == "free"
        self.price = None if self.free else price_for(str(llm_cfg.get("model", "")), pricing)
        self.spent = 0.0
        self.lock = threading.Lock()

    @property
    def budget(self) -> float:
        return float(self.cfg.get("max_cost_usd", 0.0) or 0.0)   # read live: a script may raise it

    def check(self, est_prompt_tokens: int, max_out_tokens: int) -> None:
        if self.free:
            return
        if self.price is None:
            if self.budget <= 0:
                raise BillingRefused(f"{self.cfg.get('model')}: no price row and a $0 budget; "
                                     f"declare `billing: free` for a free-tier key or pass --max-cost")
            return
        worst = usage_cost({"prompt_tokens": est_prompt_tokens, "completion_tokens": max_out_tokens}, self.price) or 0.0
        with self.lock:
            if self.spent + worst > self.budget:
                raise BillingRefused(f"{self.cfg.get('model')}: this call could cost ${worst:.4f}; "
                                     f"${self.spent:.4f} of the ${self.budget:.2f} budget is spent "
                                     f"(cost.max_usd / --max-cost; zero-cost mode defaults to $0)")

    def add(self, prompt_tokens: int, completion_tokens: int) -> None:
        if self.free or self.price is None:
            return
        with self.lock:
            self.spent += usage_cost({"prompt_tokens": prompt_tokens,
                                      "completion_tokens": completion_tokens}, self.price) or 0.0


class RateLimiter:
    """Sliding one-minute window on requests and (estimated) tokens, shared by every
    completer that talks to the same (base_url, model) in this process. Free tiers
    429 hard when exceeded; waiting is cheaper than a retry storm."""

    _shared: dict[tuple[str, str], "RateLimiter"] = {}
    _shared_lock = threading.Lock()

    def __init__(self, rpm: int | None, tpm: int | None):
        self.rpm, self.tpm = rpm, tpm
        self.events: list[tuple[float, int]] = []
        self.lock = threading.Lock()

    @classmethod
    def for_backend(cls, base_url: str, model: str, rpm: int | None, tpm: int | None) -> "RateLimiter | None":
        if not rpm and not tpm:
            return None
        key = (base_url, model)
        with cls._shared_lock:
            if key not in cls._shared:
                cls._shared[key] = cls(rpm, tpm)
            return cls._shared[key]

    def acquire(self, tokens: int) -> None:
        import time
        while True:
            with self.lock:
                now = time.monotonic()
                self.events = [(t, n) for t, n in self.events if now - t < 60.0]
                reqs, toks = len(self.events), sum(n for _, n in self.events)
                ok = (not self.rpm or reqs < self.rpm) and (not self.tpm or toks + tokens <= self.tpm or not self.events)
                if ok:
                    self.events.append((now, tokens))
                    return
                wait = 60.0 - (now - self.events[0][0]) + 0.05
            time.sleep(max(wait, 0.05))
