# Lectern, the LinkRAG web UI, for a Hugging Face Space (Docker SDK, CPU, 16 GB).
#
#   docker build -t lectern .            # context: the repository root
#   docker run -p 7860:7860 -e OPENAI_API_KEY=... -e LECTERN_MAX_COST=0.50 lectern
#
# Spend. With the secret OPENAI_API_KEY and the variable LECTERN_MAX_COST (USD) the Space answers
# with the cheap tier (models.llm, judged by eval.judge). The cap covers answerer + judge together
# but is per process: it starts again at every restart of the Space, and the spend is written to
# a ledger inside the container (/tmp), not to the project's reports/llm_ledger.jsonl. Unset, it
# is 0 and every question is refused before anything is sent. For a public Space without anyone
# watching the spend, LECTERN_ANSWERER=mock runs the offline stand-in (its verdicts say so).
# The strong model is never on here: --demo-strong is for a supervised demo run only.
#
# The sample lecture ("Try the sample lecture") is the frozen pilot01 corpus, baked in when it is
# in the build context. It is not tracked in git (data/processed is ignored), so copy it in first:
#   mkdir -p src/linkrag/ui/demo/data/processed
#   cp -r data/processed/index data/processed/links.jsonl data/processed/manifest.json \
#         src/linkrag/ui/demo/data/processed/
# Without them the image builds and the sample button is simply absent. Licence first: the index
# (index/units.json) holds the text of the talk, the slide deck and the paper verbatim, and the
# deck and recording are third-party material whose licence is unverified (data/raw/README.md).
# Building the image publishes whatever it contains, so settle that before a public Space
# (a private Space does not publish it). On a Space the files have to be in the Space repository.
# The recording and PDFs (data/raw/pilot01) add audio playback, the waveform and page images;
# without them the sample answers from text alone.

FROM node:22-slim AS web
WORKDIR /web
COPY src/linkrag/ui/web/package.json src/linkrag/ui/web/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY src/linkrag/ui/web/ ./
RUN npm run build

FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt requirements-ui.txt ./
# CPU torch first: the PyPI build for Linux pulls CUDA libraries a CPU Space cannot use. The
# pinned torch==2.14.0 in requirements.txt is then already satisfied by 2.14.0+cpu.
# playwright is a test-only dependency (tests/test_ui_e2e.py).
RUN pip install --no-cache-dir torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu \
    && grep -v '^playwright' requirements-ui.txt > /tmp/requirements-ui.txt \
    && pip install --no-cache-dir -r requirements.txt -r /tmp/requirements-ui.txt

# Models are baked in and read offline: the running container writes only under /tmp.
ENV HF_HOME=/opt/hf
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3')" \
    && python -c "from faster_whisper import WhisperModel; WhisperModel('small', device='cpu', compute_type='int8')"

COPY configs/ configs/
COPY scripts/build_links.py scripts/build_links.py
COPY src/ src/
COPY --from=web /web/dist src/linkrag/ui/web/dist

RUN useradd --uid 1000 --no-create-home user
USER user
ENV PYTHONPATH=/app/src \
    HOME=/tmp \
    XDG_CACHE_HOME=/tmp/cache \
    MPLCONFIGDIR=/tmp/matplotlib \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    LECTERN_HOST=0.0.0.0 \
    PORT=7860 \
    LECTERN_WORKDIR=/tmp/lectern \
    LECTERN_LEDGER=/tmp/lectern/llm_ledger.jsonl \
    LECTERN_SAMPLE=/app/src/linkrag/ui/demo/data/processed \
    LECTERN_SAMPLE_TITLE="Focus: Querying Large Video Datasets (OSDI ’18)"
EXPOSE 7860
CMD ["python", "-m", "linkrag.ui.api"]
