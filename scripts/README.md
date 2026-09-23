# scripts/ — how to run things, and the LLM backends

Every script reads `configs/default.yaml` (`--config` to override) and appends to
`reports/`. Every LLM-touching script prints a cost footer and appends a row to
`reports/llm_ledger.jsonl` (exact tokens from the server's `usage`; cache replays are
flagged *estimated*).

| script | what | LLM |
|---|---|---|
| `ingest.py` | files → `EvidenceUnit`s → index + corpus manifest | no |
| `build_links.py` | Evidence Linking Layer → `links.jsonl` stamped with the manifest | no |
| `ask.py` | one question, `--mode`, `--rerank`, `--show-evidence` | answerer |
| `run_regression.py` | the verified question set, one mode, appended to `regression.md` | answerer |
| `compare_retrieval.py` | mode × rerank matrix, per-type and per-question tables | answerer (iterative modes) |
| `eval/verify_gold.py` | machine-verifies a proposed gold set; writes the stamped gold | answerer + judge |
| `eval/audit_sample.py` | stratified human-audit sheet for the verification; `--score` | no |
| `dataset/redundancy.py` | modality redundancy of an ingested corpus | judge |
| `dataset/candidate.py` | intake: ingest a candidate lecture, measure, keep/reject | judge |
| `eval_alignment.py`, `eval_deictic.py` | alignment / deixis against ear labels | no |
| `gate_links.py` | relatedness gate over `links.jsonl`; flags failed links in place | judge |
| `adapters/mavils.py` | target T2: their 20 lectures, their micro-F1, transcript + PDF only | no |
| `adapters/lectqa_vid.py` | target T1: fetch / prepare / run on their QA pairs and levels | answerer |

## Zero-cost mode (default)

Nothing bills unless a run says so.

- **Budget.** `cost.max_usd: 0` in the config; every LLM script takes `--max-cost USD`
  (default 0). The spend guard prices each request *before* sending it and refuses
  (`BillingRefused`) when it could exceed the budget. A metered model with no price
  row is refused outright at $0. whisper-1 is priced on the file's duration before
  the first upload.
- **Free backends pass.** A backend with `billing: free` (below), or a local server
  (provider `ollama`, or a `localhost` base URL), is never refused.
- **Entailment is local.** Gold verification, claim verification and redundancy use
  `eval.entailment.backend: nli` (cross-encoder/nli-deberta-v3-base on the CPU,
  deterministic, one run). `--entailment llm` or `backend: llm` uses the judge
  instead. **Read `results/nli_vs_llm_pilot01.md` first:** on pilot01 the two agree
  far less than two LLM judges agree with each other. The free way to keep the LLM
  judge is to put it on a free backend: `LINKRAG_JUDGE_BACKEND=groq` (or
  `eval.judge.backend: groq`).
- **ASR is local.** faster-whisper; `ingest.py --device cuda` / `candidate.py` on a
  Colab GPU (`scripts/dataset/README.md`). whisper-1 is `--asr openai`.

Keys go in `~/.zshenv` (never in the repo): `export GROQ_API_KEY=...`,
`export GEMINI_API_KEY=...`, `export COLAB_LLM_KEY=...`.

| backend | select with | model | free-tier limit (enforced client-side) |
|---|---|---|---|
| `colab_ollama` | `LINKRAG_LLM_BACKEND=colab_ollama` | qwen2.5:7b-instruct | 60 rpm (one GPU) |
| `colab_vllm` | `LINKRAG_LLM_BACKEND=colab_vllm` | Qwen/Qwen2.5-7B-Instruct | 120 rpm |
| `groq` | `LINKRAG_LLM_BACKEND=groq` or `LINKRAG_JUDGE_BACKEND=groq` | openai/gpt-oss-120b | 28 rpm, 7,500 tpm, 1,000 req/day |
| `google_ai_studio` | `LINKRAG_LLM_BACKEND=google_ai_studio` | gemini-2.5-flash | 10 rpm |

The judge must stay a different model from the answerer (e.g. Gemini answers, Groq
judges). The rate limiter is one sliding minute per (base URL, model), shared across
threads, so a batch waits rather than hitting 429s.

## LLM backends

Two roles, two config blocks:

- **answerer** — `models.llm`. Writes answers, follow-up queries, modality-only answers.
- **judge** — `eval.judge`. Every yes/no decision: gold entailment, answer grading,
  redundancy. **Must be a different model from the answerer.**

`models.llm` is the working backend (OpenAI, `gpt-5.4-2026-03-05`, temperature 0,
seed sent). Alternatives live under `models.llm_backends` and are selected with
`models.llm.backend: <name>` or `LINKRAG_LLM_BACKEND=<name>`; the selected block is
overlaid onto `models.llm`, so nothing else changes.

### `colab_openai_compatible` — an open model served from a Colab GPU

This is the backend for **Phase 8 reported numbers**: an open-weights model whose
weights and decoding are fixed, so a result can be reproduced without an account.
The code path is identical to OpenAI's (`/v1/chat/completions`); only `base_url`,
`model` and the key differ.

**A. Serve with Ollama (simplest).** In a Colab notebook with a GPU runtime:

```bash
!curl -fsSL https://ollama.com/install.sh | sh
# long contexts: the redundancy judge sends ~2k tokens, the modality-only
# answerer up to ~30k. Ollama's default context is 4k -- raise it.
!OLLAMA_HOST=0.0.0.0:11434 OLLAMA_CONTEXT_LENGTH=32768 nohup ollama serve > ollama.log 2>&1 &
!ollama pull qwen2.5:7b-instruct        # answerer
!ollama pull llama3.1:8b                # judge (a different family)
# expose it: cloudflared needs no account
!wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O cloudflared && chmod +x cloudflared
!nohup ./cloudflared tunnel --url http://localhost:11434 > tunnel.log 2>&1 &
!sleep 5 && grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' tunnel.log
```

**B. Serve with vLLM (faster, exact `max_tokens`/`seed` semantics).**

```bash
!pip -q install vllm
!nohup python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct --dtype half --max-model-len 32768 \
    --api-key colab-secret --port 8000 > vllm.log 2>&1 &
# then the same cloudflared line with http://localhost:8000
```

**C. Point the config at it** (or export the env vars):

```yaml
models:
  llm:
    backend: colab_openai_compatible
  llm_backends:
    colab_openai_compatible:
      model: qwen2.5:7b-instruct          # Ollama tag / vLLM --model value
      base_url: https://<your-tunnel>.trycloudflare.com/v1
      api_key_env: COLAB_LLM_KEY          # export COLAB_LLM_KEY=colab-secret (any string for Ollama)
eval:
  judge:
    provider: openai_compatible
    model: llama3.1:8b
    base_url: https://<your-tunnel>.trycloudflare.com/v1
    api_key_env: COLAB_LLM_KEY
```

**D. Checks before trusting a session** (the tunnel URL changes every runtime):

```bash
curl -s $BASE/v1/models -H "Authorization: Bearer $COLAB_LLM_KEY" | head -c 300
python scripts/ask.py "what is the poster number?" --mode baseline      # one call, prints cost footer
python scripts/eval/verify_gold.py --only A4 --reports-dir /tmp/x --out /tmp/x/g.jsonl
```

Rules that still apply: temperature 0 and a seed are sent; LLM-touching cells run
3× and report mean ± std; a session's model tag, tunnel and date go in the report
header (compare_retrieval prints them). Pricing for self-hosted tags is 0 in
`models.pricing` — the ledger still counts tokens.

## Cost ledger

`reports/llm_ledger.jsonl`, one row per (run, model). `linkrag.costs.cumulative()`
sums it; every footer prints the cumulative line. Rows from before the counter
existed were backfilled by replaying the reply cache (tokens *estimated*, flagged) or
recorded with call counts only and `cost_usd: null` where tokens were never captured.
