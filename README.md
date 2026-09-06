# LinkRAG

LinkRAG is a question-answering chatbot over a mixed corpus of documents (PDF/DOCX),
images (diagrams, scanned notes) and audio (lecture recordings), built on
Retrieval-Augmented Generation. Where standard multimodal RAG indexes each modality
independently and answers from a plain top-k of similar chunks, LinkRAG adds an
**Evidence Linking Layer**: at indexing time it materialises explicit, typed, scored
links across modalities and across files — an audio segment to the slide it covers, a
figure to the paragraph that explains it, a piece of deictic speech ("as you can see
here") to the visual element it points at. Retrieval then *follows* those links out
from its top-k seeds instead of stopping at them, and a complementarity-aware reranker
scores the resulting evidence **set** for coverage rather than scoring each chunk in
isolation. The result is an answer that can be grounded jointly in what was said, what
was shown, and what was written — with a citation for each. Everything runs CPU-first;
a GPU only makes it faster.

## How this differs from existing work

| | **P1 — MI-RAG** (ICLR 2026) | **P2 — Intra-Video Temporal-Aware RAG** (CMC 2026) | **P3 — MARA** (ACM MM 2025) | **LinkRAG (ours)** |
|---|---|---|---|---|
| Modalities | text (+ multimodal queries) | audio + visual, one video | document text + figures | text, figures, tables, audio |
| Cross-modal links | none | timestamp co-occurrence only | none | **typed + scored: audio↔slide, figure↔paragraph, deictic↔visual** |
| Links across separate files | no | **no** — stops at the video boundary | no (pages independent) | **yes** |
| Retrieval strategy | iterative re-querying | top-k over fused segments | top-k over pages | **top-k seeds + link traversal** |
| Ranking | per-chunk similarity | per-segment similarity | per-page similarity | **complementarity-aware set reranking** |
| Known weakness | high recall, **low precision** | no link to external slides; **no hallucination check** | **no audio**; pages isolated | — |
| Grounding check | — | none | — | **citation faithfulness scored in eval** |
| Ablatable baseline | — | — | — | **every module ships `baseline` + `linkrag` mode** |

## Quickstart

```bash
make setup                 # venv (Python 3.11) + pinned deps + editable install
make test                  # pytest (model-downloading tests: pytest -m slow)
brew install tesseract     # only if you ingest standalone images (OCR)

# index a corpus, then ask it questions
python scripts/ingest.py data/raw/*.pdf data/raw/lecture.wav data/raw/notes.docx
python scripts/ask.py "what did the lecturer say about attention?" --mode baseline --show-evidence
```

Answering needs an LLM. The default config points at Ollama
(`ollama serve`, then `ollama pull llama3.1:8b`); any OpenAI-compatible endpoint
works by setting `models.llm.base_url`, `model`, and `api_key_env` in
`configs/default.yaml`. Only `--mode baseline` is implemented so far.

## Layout

```
src/linkrag/
  core.py       EvidenceUnit / Link / Location / Mode  — the only currency between stages
  ingest/       PDF, DOCX, image and audio  -> EvidenceUnits
  index/        bge-m3 embeddings, exact numpy dense search + BM25, persisted
  link/         Evidence Linking Layer                      [novel 1]
  retrieve/     baseline.py: RRF(dense, BM25) top-k. Link-following + complementarity  [novel 2, 3]
  generate/     cited answer synthesis (Ollama / OpenAI-compatible)
  eval/         metrics, both modes, ablations
  ui/           Streamlit chat + evidence panel
configs/        default.yaml — models, chunk sizes, thresholds, top-k
scripts/        ingest.py, ask.py
data/raw/       corpus in
data/processed/ chunks, embeddings, index, links
```

See `DESIGN.md` for conventions and the two-mode rule.
