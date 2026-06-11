# Biomedical RAG — Web App

A Streamlit UI for the RAG pipeline: upload PDFs (e-books, papers, articles),
ask questions, and get LLM-generated answers grounded in the document, with the
exact source passages shown.

## Features

- **PDF upload & processing** — drag-and-drop one or more PDFs.
- **Batch embedding** — all chunks of a document are embedded in one batched
  call to `sentence-transformers` (fast).
- **Persistence** — embeddings are stored on disk (or pgvector), so each PDF is
  processed only once and survives restarts.
- **True RAG generation** — retrieved chunks are fed to an LLM to produce a
  natural-language answer (not just raw passages).
- **Pluggable LLM** — OpenAI, local Ollama, or a zero-setup extractive fallback.

## Quick start

```bash
pip install -r webapp/requirements.txt
streamlit run webapp/app.py
```

Then open the browser tab Streamlit prints (usually http://localhost:8501),
upload a PDF, click **Process**, and ask questions.

## Choosing an answer generator (LLM)

The app auto-selects, but you can force a choice in the sidebar or via env vars:

| Provider | How to enable | Cost |
|----------|---------------|------|
| **Extractive** (default fallback) | nothing — works out of the box | free, no LLM |
| **freemodel.dev** | set `FREEMODEL_API_KEY` (OpenAI-compatible) | per your plan |
| **OpenAI** | `pip install openai` and set `OPENAI_API_KEY` | paid API |
| **Ollama** (local) | install [Ollama](https://ollama.com), run `ollama pull llama3.2` | free, local |

> Note: freemodel.dev and OpenAI both use the OpenAI Python SDK, so install it once: `pip install openai`.

Environment variables:

```bash
# freemodel.dev (OpenAI-compatible)
export FREEMODEL_API_KEY=your_key_here
export FREEMODEL_BASE_URL=https://api.freemodel.dev/v1   # optional (default)
export FREEMODEL_MODEL=claude-t0                          # optional, your model id

# OpenAI
export OPENAI_API_KEY=sk-...
export OPENAI_MODEL=gpt-4o-mini        # optional
export OPENAI_BASE_URL=...             # optional, for other OpenAI-compatible hosts

# Ollama
export OLLAMA_HOST=http://localhost:11434   # optional
export OLLAMA_MODEL=llama3.2                 # optional

# Force a provider
export RAG_LLM_PROVIDER=freemodel   # freemodel | openai | ollama | extractive
```

## Persistence options

### Default: disk (no setup)
Embeddings are pickled to `.rag_store.pkl` in the working directory. Change the
path with `RAG_STORE_PATH=/some/path.pkl`.

### Optional: pgvector (PostgreSQL)
Set a DSN and the app uses pgvector instead of disk:

```bash
export BIOMED_RAG_PGVECTOR_DSN="postgresql://user:pass@localhost:5432/ragdb"
export BIOMED_RAG_PGVECTOR_TABLE=vector_records   # optional
```

Create the table once (the embedding dimension is 384 for `all-MiniLM-L6-v2`):

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE vector_records (
    chunk_id            TEXT PRIMARY KEY,
    document_id         TEXT NOT NULL,
    content             TEXT NOT NULL,
    token_count         INTEGER NOT NULL,
    order_index         INTEGER NOT NULL,
    overlap_token_count INTEGER NOT NULL,
    page_number         INTEGER,
    heading_path        TEXT,
    is_table_part       BOOLEAN NOT NULL,
    model_id            TEXT NOT NULL,
    status              TEXT NOT NULL,
    attempts            INTEGER NOT NULL,
    embedding           vector(384)
);

CREATE INDEX ON vector_records (document_id);
```

## How it works

```
PDF → pypdf (parse) → normalize → chunk → batch-embed (sentence-transformers)
    → store (disk / pgvector) → retrieve (top-k) → LLM → grounded answer
```

Notes:
- Uses `pypdf` for text extraction (no OCR). Works for text-based PDFs; scanned
  PDFs are not supported by this engine.
- The embedding model downloads once on first run, then is cached locally.
