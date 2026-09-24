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


def test_keyed_cache_keeps_identical_prompts_apart_and_replays_in_any_order(tmp_path):
    import hashlib
    calls = []

    def inner(system, user):
        calls.append(user)
        inner.last_usage = {"prompt_tokens": 1, "completion_tokens": 1}
        return f"reply{len(calls)}"
    c = cached_completer(inner, tmp_path)
    assert c("s", "same", key=("answerA|u1", 0)) == "reply1"
    assert c("s", "same", key=("answerB|u1", 0)) == "reply2"          # same prompt, its own reply

    def boom(system, user):
        raise AssertionError("must replay from cache")
    replay = cached_completer(boom, tmp_path, cache_only=True)
    assert replay("s", "same", key=("answerB|u1", 0)) == "reply2"     # order reversed, same mapping
    assert replay("s", "same", key=("answerA|u1", 0)) == "reply1"
    # an entry written before keying (<prompt>.<run>) is read for every context, by run index
    base = hashlib.sha256(("s" + "\x00" + "old").encode()).hexdigest()
    (tmp_path / f"{base}.0.txt").write_text("legacy0")
    (tmp_path / f"{base}.1.txt").write_text("legacy1")
    assert replay("s", "old", key=("any", 1)) == "legacy1" and replay("s", "old", key=("other", 1)) == "legacy1"
    assert replay("s", "old", key=("any", 0)) == "legacy0"


def test_verify_answer_keys_the_judge_by_answer_and_unit():
    from linkrag.core import EvidenceUnit
    from linkrag.generate.answer import Answer, Claim
    from linkrag.generate.verify import verify_answer
    seen = []

    def judge(system, user, key=None):
        seen.append(key)
        return '{"facts": [{"fact": "x", "supported": true, "span": "x"}], "verdict": "yes"}'
    unit = EvidenceUnit(id="u1", modality="text", content="x", source_file="f")
    for who in ("q1:baseline", "q2:baseline"):
        verify_answer(Answer(answer="x", raw="", claims=[Claim(claim="x", unit_ids=["u1"])]), [unit], judge,
                      runs=2, backend="llm", answer_key=who)
    assert seen == [("q1:baseline|u1", 0), ("q1:baseline|u1", 1), ("q2:baseline|u1", 0), ("q2:baseline|u1", 1)]


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


def test_verifier_defaults_to_the_cheap_tier_llm_judge_and_never_falls_back(monkeypatch):
    """Dataset policy (2026-09-23): the judge is gpt-4.1-mini (cheap tier, the stored
    reference) and the batch answerer a different cheap model. A free judge backend
    selected without its key stops the run in one line and sends nothing anywhere."""
    import pytest
    import requests
    from linkrag.core import load_config
    from linkrag.eval.verify_gold import entailment_opts
    from linkrag.generate.answer import MissingApiKey, http_completer, judge_completer
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("LINKRAG_JUDGE_BACKEND", raising=False)
    monkeypatch.delenv("LINKRAG_LLM_BACKEND", raising=False)
    monkeypatch.setattr(requests, "post", lambda *a, **k: pytest.fail("no request may be sent"))
    cfg = load_config("configs/default.yaml")
    assert entailment_opts(cfg)["backend"] == "llm" and entailment_opts({})["backend"] == "llm"
    judge, llm = cfg["eval"]["judge"], cfg["models"]["llm"]
    assert judge["model"].startswith("gpt-4.1-mini") and judge["api_key_env"] == "OPENAI_API_KEY"
    assert llm["model"] != judge["model"]                       # separate judge (DESIGN.md)
    monkeypatch.setenv("LINKRAG_JUDGE_BACKEND", "groq")
    groq = load_config("configs/default.yaml")
    with pytest.raises(MissingApiKey) as err:
        judge_completer(groq, "llm")
    assert "GROQ_API_KEY" in str(err.value) and "\n" not in str(err.value)
    with pytest.raises(MissingApiKey):
        http_completer(groq["eval"]["judge"])
    stub = judge_completer(groq, "nli")                         # the nli ablation needs no judge key
    assert stub.usage["calls"] == 0
    with pytest.raises(RuntimeError, match="nli"):
        stub("s", "u")


def test_batch_runs_refuse_the_strong_model_unless_allowed(monkeypatch):
    import pytest
    from linkrag.core import load_config, refuse_strong_in_batch
    monkeypatch.delenv("LINKRAG_LLM_BACKEND", raising=False)
    monkeypatch.delenv("LINKRAG_JUDGE_BACKEND", raising=False)
    cfg = load_config("configs/default.yaml")
    refuse_strong_in_batch(cfg)                                 # defaults are cheap tier
    cfg["models"]["llm"]["model"] = "gpt-5.4-2026-03-05"
    with pytest.raises(SystemExit, match="allow_strong_in_batch"):
        refuse_strong_in_batch(cfg)
    refuse_strong_in_batch(cfg, answerer_cache_only=True)       # a replay cannot spend
    cfg["models"]["llm"]["allow_strong_in_batch"] = True
    refuse_strong_in_batch(cfg)                                 # explicit reference-row opt-in
    cfg["models"]["llm"].update(allow_strong_in_batch=False, model="gpt-5.4-mini-2026-03-17")
    refuse_strong_in_batch(cfg)                                 # dated cheap snapshots count
    cfg["eval"]["judge"]["model"] = "gpt-4.1"
    with pytest.raises(SystemExit, match="eval.judge"):
        refuse_strong_in_batch(cfg)

def test_colab_backend_against_a_local_openai_compatible_server(monkeypatch, tmp_path):
    """backend colab: URL from LINKRAG_COLAB_BASE_URL, no key sent (not even the base
    block's OPENAI_API_KEY), $0 in the ledger tagged backend=colab, one-line failure
    when the URL is unset -- for both answerer and judge."""
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer
    import pytest
    from linkrag.core import load_config
    from linkrag.costs import cached_completer, record_run
    from linkrag.generate.answer import MissingApiKey, http_completer
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append((self.path, self.headers.get("Authorization"), body))
            out = json.dumps({"choices": [{"message": {"content": "ok"}}],
                              "usage": {"prompt_tokens": 7, "completion_tokens": 1}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

        def log_message(self, *a):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-must-not-leak")
        monkeypatch.setenv("LINKRAG_LLM_BACKEND", "colab")
        monkeypatch.setenv("LINKRAG_JUDGE_BACKEND", "colab")
        monkeypatch.delenv("LINKRAG_COLAB_BASE_URL", raising=False)
        cfg = load_config("configs/default.yaml")
        for block in (cfg["models"]["llm"], cfg["eval"]["judge"]):
            with pytest.raises(MissingApiKey, match="LINKRAG_COLAB_BASE_URL"):
                http_completer(block)
        assert not seen
        monkeypatch.setenv("LINKRAG_COLAB_BASE_URL", f"http://127.0.0.1:{server.server_port}/v1")
        for block in (cfg["models"]["llm"], cfg["eval"]["judge"]):
            complete = cached_completer(http_completer(block), tmp_path / block["backend"] / str(len(seen)))
            assert complete("sys", "user") == "ok"
            path, auth, body = seen[-1]
            assert path == "/v1/chat/completions" and auth is None
            assert body["model"] == "qwen2.5:7b-instruct" and body["temperature"] == 0.0
            ledger = tmp_path / "ledger.jsonl"
            record_run("test", "colab", [(block["model"], complete.usage)], cfg["models"]["pricing"], ledger=ledger)
            row = json.loads(ledger.read_text().splitlines()[-1])
            assert row["backend"] == "colab" and row["cost_usd"] == 0.0 and row["calls"] == 1
    finally:
        server.shutdown()
