# RAG-BME — Biomedical RAG Pipeline

An end-to-end, production-oriented **Retrieval-Augmented Generation (RAG)** pipeline for
biomedical literature — e-books, research papers, and articles in PDF. It turns a PDF
into a queryable knowledge base and answers questions using an LLM, grounded in the
document's own content.

The pipeline is built with a clean **ports-and-adapters** architecture: every stage
(parsing, OCR, normalization, chunking, embedding, storage, retrieval) is a pluggable
component selected from a single validated configuration. It ships with deterministic
mock adapters for testing and real adapters (pypdf, sentence-transformers, pgvector)
for production use.

---

## Table of contents

- [Features](#features)
- [Architecture](#architecture)
- [Project structure](#project-structure)
- [Installation](#installation)
- [Quick start](#quick-start)
  - [1. Command-line RAG](#1-command-line-rag)
  - [2. Web app (Streamlit)](#2-web-app-streamlit)
- [LLM answer generation](#llm-answer-generation)
- [Persistence](#persistence)
- [Configuration](#configuration)
- [Testing](#testing)
- [How it works](#how-it-works)
- [Limitations](#limitations)

---

## Features

- **PDF ingestion** — accepts PDF, EPUB, DOCX, and HTML with a strict, content-sniffing
  validation gate (size, format, corruption, and SHA-256 de-duplication).
- **Pluggable parsing** — text-layer extraction via `pypdf` (no OCR) or layout-aware
  parsing via Docling; engine chosen by configuration.
- **Normalization** — artifact removal (headers/footers/page numbers), dictionary-based
  de-hyphenation, and a canonical, serializable document model.
- **Token-bounded chunking** — configurable size and overlap, with table- and
  figure-aware splitting and full metadata propagation.
- **Real embeddings** — `sentence-transformers` (local, free) behind a pluggable
  embedding port, with batch embedding for speed.
- **Durable storage** — in-memory, disk-backed (pickle), or **pgvector** (PostgreSQL).
- **Semantic retrieval** — top-k similarity search with metadata filtering and
  deterministic ranking.
- **LLM generation (true RAG)** — retrieved chunks are sent to an LLM to produce a
  grounded, cited answer. Supports **freemodel.dev**, **OpenAI**, **Ollama**, or a
  zero-setup **extractive** fallback.
- **Web UI** — a Streamlit app for uploading PDFs and asking questions.
- **433 passing tests** including 30 Hypothesis property-based tests.

---

## Architecture

```
                          PipelineConfig (single source of truth)
                                       │
  submit ──► Ingestion ──► Parsing ──► Normalization ──► Chunking ──► Embedding ──► Storage
  (validate)              (engine)    (clean text)     (token-bound)  (vectors)    (vector DB)
                                                                                      │
  retrieve ◄──────────────────────────────────── Retriever ◄──────────────────────────┘
                                                      │
                                                   LLM (freemodel / OpenAI / Ollama)
                                                      │
                                                   grounded answer
```

Every component implements a stable **port** (interface) and is built from the config by
a **registry**, so backends can be swapped without touching pipeline logic.

---

## Project structure

```
RAG-BME/
├── src/biomed_rag/             # The library
│   ├── config/                 # Validated PipelineConfig
│   ├── ingestion/              # Submission + validation gate
│   ├── parsing/                # Parser port + Docling/LlamaParse/mock adapters
│   ├── ocr/                    # OCR port + processor
│   ├── normalization/          # Artifact removal, de-hyphenation, serialization
│   ├── chunking/               # Token-bounded chunker + tokenizer
│   ├── embedding/              # Embedding port + registry + mock model
│   ├── storage/                # VectorStore port + in-memory & pgvector adapters
│   ├── retrieval/              # Retriever (validation, filtering, ranking)
│   ├── orchestration/          # Stage orchestrator (retry/resume/progress)
│   ├── models/                 # Core data models
│   └── pipeline.py             # Top-level facade (submit → process → retrieve)
├── webapp/                     # Streamlit web app + reusable RAG engine
│   ├── app.py                  # UI
│   ├── rag_engine.py           # Ingest (batch embed) + retrieve + generate
│   ├── llm.py                  # Pluggable LLM providers
│   └── README.md               # Web app docs
├── run_real.py                 # CLI: real PDF → RAG (pypdf + sentence-transformers)
├── tests/                      # 433 tests (unit + property-based)
└── pyproject.toml
```

---

## Installation

Requires **Python 3.10+**.

```bash
# Clone
git clone https://github.com/hariomaggarwal11/RAG-BME.git
cd RAG-BME

# Create a virtual environment
python -m venv .venv
# Windows:  .\.venv\Scripts\Activate.ps1
# macOS/Linux:  source .venv/bin/activate

# Install the library (editable) + dev/test tools
pip install -e ".[dev]"
```

For the real pipeline and web app, also install:

```bash
pip install pypdf sentence-transformers streamlit
# Optional providers:
pip install openai                       # freemodel.dev / OpenAI generation
pip install "psycopg[binary]" pgvector   # pgvector persistence
```

---

## Quick start

### 1. Command-line RAG

Process a PDF and ask questions from the terminal:

```bash
python run_real.py "mybook.pdf" "What is the main contribution?"
# or interactive mode:
python run_real.py "mybook.pdf"
```

Uses `pypdf` for text extraction (no OCR) and local `sentence-transformers` embeddings —
no API keys required.

### 2. Web app (Streamlit)

```bash
pip install -r webapp/requirements.txt
streamlit run webapp/app.py
```

Open the printed URL (usually http://localhost:8501), upload one or more PDFs, click
**Process**, then ask questions. You'll get an LLM-generated answer plus the exact source
passages (with page numbers and similarity scores).

See [`webapp/README.md`](webapp/README.md) for full web app documentation.

---

## LLM answer generation

The app turns retrieved chunks into a grounded, cited answer. It auto-selects a provider,
or you can force one in the sidebar / via `RAG_LLM_PROVIDER`.

| Provider | Enable with | Cost |
|----------|-------------|------|
| **Extractive** (fallback) | nothing — works out of the box | free, no LLM |
| **freemodel.dev** | `FREEMODEL_API_KEY` (OpenAI-compatible) | per your plan |
| **OpenAI** | `OPENAI_API_KEY` | paid API |
| **Ollama** (local) | install Ollama + `ollama pull llama3.2` | free, local |

> freemodel.dev and OpenAI use the OpenAI SDK: `pip install openai`.

```bash
# freemodel.dev (OpenAI-compatible)
export FREEMODEL_API_KEY=your_key
export FREEMODEL_BASE_URL=https://api.freemodel.dev/v1   # optional (default)
export FREEMODEL_MODEL=claude-t0                          # your model id

# OpenAI
export OPENAI_API_KEY=sk-...

# Ollama
export OLLAMA_MODEL=llama3.2

# Force a provider
export RAG_LLM_PROVIDER=freemodel   # freemodel | openai | ollama | extractive
```

---

## Persistence

Embeddings are stored so each PDF is processed only once.

- **Disk (default)** — pickled to `.rag_store.pkl`; no setup. Change with `RAG_STORE_PATH`.
- **pgvector (PostgreSQL)** — set `BIOMED_RAG_PGVECTOR_DSN` to switch automatically. The
  table schema (embedding dimension `384` for `all-MiniLM-L6-v2`) is documented in
  [`webapp/README.md`](webapp/README.md).

---

## Configuration

All bounded parameters live in a single immutable `PipelineConfig` (validated at
construction). Highlights:

| Setting | Default | Notes |
|---------|---------|-------|
| `parsing_engine` | `docling` | engine key resolved by the registry |
| `max_chunk_tokens` | `512` | 128–2048 |
| `chunk_overlap_tokens` | `64` | `< max_chunk_tokens` |
| `embedding_dimension` | `1536` | 64–4096 (use `384` for MiniLM) |
| `default_top_k` | `5` | 1–100 |
| `vector_store_backend` | `pgvector` | `pgvector` or `qdrant` |
| `stage_retry_limit` | `3` | 0–10 |

```python
from biomed_rag.config import PipelineConfig

config = PipelineConfig(
    parsing_engine="docling",
    embedding_model="st-minilm",
    embedding_dimension=384,
)
```

---

## Testing

```bash
pytest                 # run all 433 tests
pytest -v              # verbose
pytest tests/pipeline/ # a single area
pytest --hypothesis-show-statistics
```

The suite includes deterministic unit/smoke tests and 30 Hypothesis property tests
(≥100–300 examples each). One pgvector integration test is skipped unless a real
PostgreSQL DSN is provided.

---

## How it works

```
PDF → parse (pypdf/Docling) → normalize → chunk → batch-embed (sentence-transformers)
    → store (disk / pgvector) → retrieve top-k → LLM → grounded, cited answer
```

Library usage (programmatic):

```python
from biomed_rag import Pipeline
from biomed_rag.ingestion.service import FileInput, Accepted
from biomed_rag.retrieval.retriever import QueryRequest

pipeline = Pipeline()                                # deterministic defaults
job = pipeline.submit(FileInput(filename="doc.html",
                                content=b"<html><body>...</body></html>"))
pipeline.process(job.jobId)                          # parse → … → store
result = pipeline.retrieve(QueryRequest(text="your question", topK=5))
for chunk in result.chunks:
    print(chunk.similarity, chunk.content[:120])
```

For real PDFs with real embeddings, see `run_real.py` and `webapp/rag_engine.py`.

---

## Limitations

- The `pypdf` engine reads **text-layer PDFs** (most e-books, papers, articles).
  Scanned/image-only PDFs need OCR (Docling + a working OCR backend).
- Default embeddings run on CPU; large books take longer to embed.
- The extractive fallback returns passages, not a synthesized answer — configure an LLM
  provider for natural-language answers.

---

## License

MIT
