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

LinkRAG is measured against two published systems, each on its own benchmark.

| Target | Their method | Their published numbers | Limitation we address | Metric we will report |
|---|---|---|---|---|
| **T1 — Intra-Video Temporal-Aware RAG** (Shafiq, Ejaz, Shah, Kamal, Sohail, Aslam; *CMC* 88(2):96, 2026; [doi:10.32604/cmc.2026.081534](https://doi.org/10.32604/cmc.2026.081534)). Benchmark **LectQA-Vid**: 100 CS lecture videos (2–5 min), 3,000 QA pairs (½ MCQ, ½ open-ended), 80/10/10 split, 1,000-pair eval subset. | Whisper transcript + frame captions cut into timestamped segments; retrieval restricted to a temporal window inside the *same video*; cross-encoder rerank; LLM answer. | Open-ended, Table 4 *Overall* row: **F1 23.52 %, semantic similarity 74.42 %**, ROUGE-1 29.76 %, BLEU 5.58 %, METEOR 32.41 %. MCQ, Table 5 *Overall* row: **accuracy 53.43 %**. Their multimodal-RAG-without-timestamps baseline, Table 6 *Overall*: F1 14.80 %, similarity 61.00 %. (Corrected 2026-09-23 from 0.71 / 56.30 % / 19.62 %.) Ablation, Table 7: without the temporal filter F1 falls to 11.40 %. | Evidence is linked only by *timestamp* and only *inside the video*: a separate slide deck or paper is unreachable, no retrieved unit can pull in another it depends on, and the evidence set is chosen unit-by-unit so it can be eight near-duplicate segments. | Their metric suite (eqs. 29–35), per difficulty and Overall, on a stratified 300-question open-ended subset (their split is unpublished), 3 repeats, one answerer (gpt-5.4-mini) for all three modes. **Overall F1: T1 pipeline replicated 31.23, ours (iterative) 36.70, whole transcript 38.74.** The whole transcript beats both retrieval pipelines, because every LectQA-Vid transcript fits in the context. MCQ accuracy is not reportable: the answer is option A in 99.7 % of MCQs and the unique longest option in 94.6 % (`results/external/lectqa_audit.md`, `DESIGN.md` finding 13). |
| **T2 — MaViLS** (Anderer, Reich, Wölfel; *Interspeech 2024*, pp. 1375–1379; [doi:10.21437/Interspeech.2024-978](https://doi.org/10.21437/Interspeech.2024-978); [arXiv:2409.16765](https://arxiv.org/abs/2409.16765)). Benchmark **MaViLS**: 20 lectures (MIT OCW, Tübingen, DeepMind), >22 h, 12,830 segments, every spoken sentence hand-labelled with its slide (−1 = none). | OCR text, transcript text and image embeddings each give a frame×slide similarity matrix; dynamic programming with a slide-jump penalty (λ_jump = 0.1) picks the slide sequence for a video. | Per-frame F1, Table 1 *Average* row: **audio transcript only 0.53**, OCR text only 0.76, image only 0.64, SIFT baseline 0.56. All three features combined, λ_jump = 0.1, Table 2 *Average* / §4.2: **0.82**. | Needs the *video frames*: OCR and image features carry the accuracy and transcript alone reaches 0.53. Alignment is the end product — nothing downstream retrieves or answers over it. | Their per-frame F1 with their definition, per lecture and average. From **transcript + slide PDF**, their audio column (0.53) is the like-for-like comparison. **With the video frames** (19 of 20 lectures; `results/external/mavils_visual.md`, DESIGN.md finding 14), test half, 9 lectures with video: **image + frame OCR + speech-to-slide text, from the frame at each sentence's timestamp: 0.853 vs their all-features 0.80 on the same lectures** (6 wins, 3 losses; paired difference +0.055, 95 % CI −0.050 to +0.161): **parity** (DESIGN.md finding 16). With 320 px representative frames 0.810 (finding 15); image and text 0.765 (finding 14); text only 0.49. **First run (`reports/mavils_final.md`, `mavils_fused.md`): ours 0.46; a matrix × decoder decomposition using their public code shows the gap is in the similarity features; our decoder ≥ theirs (their matrix + our DP 0.520 > their own 0.513). Fusing the two similarities, weight set on a tune half, reaches 0.520 on the test half vs their 0.51 there.** |

Components and related work, not targets: **MI-RAG** (Choi et al., [arXiv:2509.00798](https://arxiv.org/abs/2509.00798); its
iterative re-querying is our `retrieve/iterative.py`), **MARA** (Wu et al., ACM MM 2025,
[doi:10.1145/3746027.3755390](https://doi.org/10.1145/3746027.3755390); page-independent document QA), and
the literature-review set in `paper/bibliography.bib`. See `DESIGN.md` → *Targets* for the full positioning
and the binding priority order.

## Quickstart

```bash
make setup                 # venv (Python 3.11) + pinned deps + editable install
make test                  # pytest (model-downloading tests: pytest -m slow)
brew install tesseract     # only if you ingest standalone images (OCR)

# index a corpus, then ask it questions
python scripts/ingest.py data/raw/*.pdf data/raw/lecture.wav data/raw/notes.docx
python scripts/build_links.py      # Evidence Linking Layer -> data/processed/links.jsonl
python scripts/ask.py "what did the lecturer say about attention?" --mode linkrag --show-evidence
```

Answering needs an LLM. The default config points at Ollama
(`ollama serve`, then `ollama pull llama3.1:8b`); any OpenAI-compatible endpoint
works by setting `models.llm.base_url`, `model`, and `api_key_env` in
`configs/default.yaml`. `--mode baseline|linkrag|iterative|linkrag_iter` and `--rerank none|mmr|complementarity` are all implemented; `--mode linkrag` needs `python scripts/build_links.py` to have run first.

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
scripts/        ingest.py, build_links.py, ask.py, run_regression.py, compare_retrieval.py, eval_*.py
docs/           pilot01_history.md — the v0-pilot narrative and numbers
paper/          bibliography.bib
data/raw/       corpus in
data/processed/ chunks, embeddings, index, links
```

See `DESIGN.md` for conventions and the two-mode rule.
