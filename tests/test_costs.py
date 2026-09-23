"""Token/cost accounting: cache stores usage, hits without it are flagged estimates,
prices match by longest prefix, the ledger accumulates."""

from __future__ import annotations

import json

from linkrag.costs import cached_completer, cumulative, price_for, record_run, usage_cost


def test_cache_records_exact_usage_and_flags_estimates(tmp_path):
    def inner(system, user):
        inner.last_usage = {"prompt_tokens": 100, "completion_tokens": 10}
        return "reply"
    c = cached_completer(inner, tmp_path)
    assert c("s", "u") == "reply" and c.usage["prompt_tokens"] == 100
    assert c("s", "u") == "reply"                       # 2nd identical call = new key .1 -> miss
    c2 = cached_completer(inner, tmp_path)              # fresh process: hits with stored usage
    c2("s", "u"); c2("s", "u")
    from linkrag.costs import empty_usage
    assert c2.usage == empty_usage() | {"calls": 2, "cached_calls": 2, "prompt_tokens": 200,
                                        "completion_tokens": 20, "cached_prompt_tokens": 200,
                                        "cached_completion_tokens": 20}
    assert usage_cost(c2.usage, {"input_per_m": 1e6, "output_per_m": 1e6}) == 0.0
    assert usage_cost(c2.usage, {"input_per_m": 1e6, "output_per_m": 1e6}, charge_cached=True) == 220.0
    # a legacy cache file with no usage sidecar is estimated and flagged
    (tmp_path / "legacy").mkdir()
    import hashlib
    key = hashlib.sha256(("s" + "\x00" + "u" * 40).encode()).hexdigest()
    (tmp_path / "legacy" / f"{key}.0.txt").write_text("x" * 80)
    c3 = cached_completer(inner, tmp_path / "legacy")
    c3("s", "u" * 40)
    assert c3.usage["estimated_tokens"] == c3.usage["prompt_tokens"] + c3.usage["completion_tokens"] > 0


def test_price_longest_prefix_and_unknown():
    pricing = {"gpt-5.4": {"input_per_m": 2.5, "output_per_m": 15.0},
               "gpt-5.4-mini": {"input_per_m": 0.75, "output_per_m": 4.5}}
    assert price_for("gpt-5.4-mini-2026", pricing)["input_per_m"] == 0.75
    assert price_for("gpt-5.4-2026-03-05", pricing)["input_per_m"] == 2.5
    assert price_for("unknown-model-x", pricing) is None
    assert usage_cost({"prompt_tokens": 1_000_000, "completion_tokens": 0},
                      price_for("gpt-5.4-2026", pricing)) == 2.5
    assert usage_cost({"prompt_tokens": 5}, None) is None


def test_ledger_accumulates_and_footer_mentions_unpriced(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    pricing = {"m": {"input_per_m": 1.0, "output_per_m": 2.0}}
    u = {"calls": 3, "cached_calls": 0, "prompt_tokens": 1_000_000, "completion_tokens": 500_000,
         "estimated_tokens": 0}
    lines = record_run("x.py", "run1", [("m-1", u), ("other", u)], pricing, ledger=ledger)
    assert any("$2.0000" in l for l in lines) and any("price unknown" in l for l in lines)
    record_run("x.py", "run2", [("m-1", u)], pricing, ledger=ledger)
    cum = cumulative(ledger)
    assert cum["rows"] == 3 and cum["cost_usd"] == 4.0 and cum["unpriced_rows"] == 1
    assert len(ledger.read_text().splitlines()) == 3


def test_cache_only_never_calls_and_cost_cap_stops(tmp_path):
    from linkrag.costs import CacheMiss
    import pytest
    calls = {"n": 0}

    def inner(system, user):
        calls["n"] += 1
        inner.last_usage = {"prompt_tokens": 400_000, "completion_tokens": 0}
        return "r"
    ro = cached_completer(inner, tmp_path, cache_only=True)
    with pytest.raises(CacheMiss):
        ro("s", "u")
    assert calls["n"] == 0
    capped = cached_completer(inner, tmp_path, max_cost_usd=1.5, price={"input_per_m": 2.5, "output_per_m": 0})
    capped("s", "u1")                       # $1.00 spent
    capped("s", "u2")                       # $2.00 spent -> next call refused
    with pytest.raises(RuntimeError, match="cost cap"):
        capped("s", "u3")
    assert calls["n"] == 2


def test_spend_guard_zero_cost_mode():
    import pytest
    from linkrag.costs import BillingRefused, SpendGuard
    pricing = {"gpt-4.1-mini": {"input_per_m": 0.4, "output_per_m": 1.6}}
    # free backend passes at $0
    SpendGuard({"model": "openai/gpt-oss-120b", "billing": "free", "max_cost_usd": 0.0}, pricing).check(10_000, 1_000)
    # a local server with nothing declared cannot bill
    SpendGuard({"provider": "ollama", "model": "llama3.1:8b", "max_cost_usd": 0.0}, pricing).check(10, 10)
    # metered, no price row, $0 budget: refused before sending
    with pytest.raises(BillingRefused, match="no price row"):
        SpendGuard({"model": "mystery-model", "max_cost_usd": 0.0}, pricing).check(10, 10)
    # metered and priced: $0 refuses; a budget admits until the worst case would exceed it
    cfg = {"model": "gpt-4.1-mini", "max_cost_usd": 0.0}
    g = SpendGuard(cfg, pricing)
    with pytest.raises(BillingRefused):
        g.check(1000, 100)
    cfg["max_cost_usd"] = 0.001                       # read live, as set_max_cost does
    g.check(1000, 100)                                # worst 0.00056 fits
    g.add(1000, 100)
    with pytest.raises(BillingRefused, match="budget"):
        g.check(1000, 100)                            # 0.00056 + 0.00056 > 0.001


def test_set_max_cost_reaches_every_llm_block():
    from linkrag.core import load_config, set_max_cost
    cfg = load_config("configs/default.yaml")
    assert cfg["models"]["llm"]["max_cost_usd"] == 0.0 and cfg["eval"]["judge"]["max_cost_usd"] == 0.0
    set_max_cost(cfg, 3)
    assert cfg["models"]["llm"]["max_cost_usd"] == 3.0 and cfg["eval"]["judge"]["max_cost_usd"] == 3.0


def test_rate_limiter_waits_out_the_minute(monkeypatch):
    import time
    from linkrag.costs import RateLimiter
    clock, slept = [1000.0], []
    monkeypatch.setattr(time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(time, "sleep", lambda s: (slept.append(s), clock.__setitem__(0, clock[0] + s)))
    assert RateLimiter.for_backend("u", "m", None, None) is None
    lim = RateLimiter(rpm=2, tpm=None)
    lim.acquire(1); lim.acquire(1)
    assert not slept
    lim.acquire(1)                                    # third in the same minute waits ~60 s
    assert len(slept) == 1 and 59.9 < slept[0] < 60.2
    assert RateLimiter.for_backend("u", "m2", 5, None) is RateLimiter.for_backend("u", "m2", 5, None)
