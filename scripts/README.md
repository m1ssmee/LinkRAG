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
| `eval/judge_agreement.py` | a candidate judge (`--judge colab\|groq\|...`) vs the stored pilot01 verdicts: kappa report; refuses a billing judge without `--max-cost` | judge (answerer cache-only) |
| `eval/compare_entailment.py` | agreement tables between stored verification / redundancy runs | no |
| `dataset/redundancy.py` | modality redundancy of an ingested corpus | judge |
| `dataset/candidate.py` | intake: ingest a candidate lecture, measure, keep/reject | judge |
| `eval_alignment.py`, `eval_deictic.py` | alignment / deixis against ear labels | no |
| `gate_links.py` | relatedness gate over `links.jsonl`; flags failed links in place | judge |
| `adapters/mavils.py` | target T2: their 20 lectures, their micro-F1, transcript + PDF only | no |
| `adapters/lectqa_vid.py` | target T1: entry point for fetch / prepare / run / audit / mcq / open / faith / v2 / frameslides; the steps live in `adapters/lectqa/` | answerer |

## Cost policy (DESIGN.md *Cost policy*)

Nothing bills without an explicit budget, and spend goes to the cheap tier.

- **Budget.** Every batch script **requires** `--max-cost USD`. `0` allows only cache
  replays and free backends. The spend guard prices each request *before* sending it and
  refuses (`BillingRefused`) anything that could exceed the budget. A metered model with
  no price row is refused outright at $0. whisper-1 is priced on the file's duration
  before the first upload.
- **Cheap tier.** Judge gpt-4.1-mini, batch answerer gpt-5.4-mini (they must differ). Any
  other metered model (the strong gpt-5.4) is refused in a batch run unless
  `models.llm.allow_strong_in_batch: true`, which is for the final reference row and the
  demo only.
- **Free backends pass.** A backend with `billing: free` (below), or a local server
  (provider `ollama`, or a `localhost` base URL), is never refused. A free judge (`groq`,
  `colab`) is used only after `scripts/eval/judge_agreement.py --judge <name>` has measured
  it. Without its key or URL, a run stops with one line and never falls back.
- **NLI** (`--entailment nli`) is an **ablation only**: on pilot01 it agrees with the LLM
  judges at kappa 0.31/0.36, where they agree with each other at 0.85
  (`results/nli_vs_llm_pilot01.md`, DESIGN.md finding 11).
- **ASR is local.** faster-whisper; `ingest.py --device cuda` / `candidate.py` on a
  Colab GPU (`scripts/dataset/README.md`). whisper-1 is `--asr openai`.

Keys go in `~/.zshenv` (never in the repo): `export GROQ_API_KEY=...`,
`export GEMINI_API_KEY=...`, `export COLAB_LLM_KEY=...`.

| backend | select with | model | free-tier limit (enforced client-side) |
|---|---|---|---|
| `colab` | `LINKRAG_LLM_BACKEND=colab` / `LINKRAG_JUDGE_BACKEND=colab` + `LINKRAG_COLAB_BASE_URL` | qwen2.5:7b-instruct | 60 rpm (one GPU) |
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

### `colab` — an open model served from a free Colab GPU

The backend for Phase 8 reported numbers: open weights with fixed decoding, reproducible
without an account. Same code path as every other backend (`/v1/chat/completions`).

1. **Serve:** open `notebooks/colab_serve.ipynb` in Colab (T4 GPU runtime) and run it top
   to bottom. It installs Ollama, pulls `qwen2.5:7b-instruct` (the quantised 14B option
   and how to check VRAM are in the notebook), opens a Cloudflare quick tunnel, and
   prints one line.
2. **Point LinkRAG at it**, in the shell where LinkRAG runs:
   ```bash
   export LINKRAG_COLAB_BASE_URL=https://<random>.trycloudflare.com/v1   # the printed line
   export LINKRAG_LLM_BACKEND=colab       # answerer
   export LINKRAG_JUDGE_BACKEND=colab     # judge -- only after it passes step 3
   ```
   No key: Ollama needs none, and the colab backend sends none (a selected backend never
   inherits another backend's key). Without `LINKRAG_COLAB_BASE_URL`, any run stops with one
   line; it never falls back to a paid backend. Cost is recorded as $0 with `backend: colab`
   in `reports/llm_ledger.jsonl`. The rate limit is `rate_limit_rpm` in the `colab` block.
3. **Measure before trusting it as judge:**
   `python scripts/eval/judge_agreement.py --judge colab --max-cost 0` writes
   `results/judge_agreement_colab.md` (kappa vs the stored gpt-4.1-mini and gpt-5.4 verdicts).
   It must come close to the LLM-LLM 0.85 before it decides gold or intake (DESIGN.md finding 11).
4. **Disconnects:** free sessions end after a few hours. Re-run the notebook, export the new
   URL, and re-run the same command. Finished calls replay from the reply cache.

vLLM instead of Ollama: backend `colab_vllm` (start vLLM with `--api-key` and export that
key as `COLAB_LLM_KEY`; same `LINKRAG_COLAB_BASE_URL`). The old names `colab_ollama` /
`colab_openai_compatible` are aliases of `colab`.

Rules that still apply: temperature 0 and a seed are sent; LLM-touching cells run
3× and report mean ± std; a session's model tag, tunnel and date go in the report
header.

## Cost ledger

`reports/llm_ledger.jsonl`, one row per (run, model). `linkrag.costs.cumulative()`
sums it; every footer prints the cumulative line. Rows from before the counter
existed were backfilled by replaying the reply cache (tokens *estimated*, flagged) or
recorded with call counts only and `cost_usd: null` where tokens were never captured.
