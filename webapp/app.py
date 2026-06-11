"""Streamlit web UI for the Biomedical RAG pipeline.

Run:
    pip install -r webapp/requirements.txt
    streamlit run webapp/app.py

Features:
- Upload one or more PDFs (e-books, papers, articles) and process them.
- Ask questions and get an LLM-generated, source-grounded answer.
- See the exact passages used, with page numbers and similarity scores.
- Embeddings persist on disk, so a PDF is only processed once.
"""

from __future__ import annotations

import os
import sys

# Make imports work regardless of how Streamlit launches this script. Streamlit
# puts the script's own folder (webapp/) on sys.path, not the repo root, so add
# the repo root (for the `webapp` package) and `src` (for `biomed_rag` when not
# pip-installed) explicitly.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_REPO_ROOT, os.path.join(_REPO_ROOT, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import streamlit as st

from webapp.rag_engine import RagEngine
from webapp.book_metadata import BookMeta, extract_pdf_metadata


st.set_page_config(page_title="Biomedical RAG", page_icon="📚", layout="wide")


@st.cache_resource(show_spinner="Loading models (first run downloads the embedding model)...")
def load_engine(llm_provider: str) -> RagEngine:
    """Build the engine once and reuse it across reruns."""
    provider = None if llm_provider == "auto" else llm_provider
    return RagEngine(llm_provider=provider)


# ---------------------------------------------------------------------------
# Sidebar: settings + ingestion
# ---------------------------------------------------------------------------

st.sidebar.title("📚 Biomedical RAG")
st.sidebar.caption("Upload PDFs, ask questions, get grounded answers.")

llm_choice = st.sidebar.selectbox(
    "Answer generator (LLM)",
    options=["auto", "freemodel", "openai", "ollama", "extractive"],
    index=0,
    help=(
        "auto: freemodel.dev if FREEMODEL_API_KEY set, else OpenAI, else Ollama, "
        "else extractive (no LLM). 'extractive' just returns the top passages."
    ),
)

top_k = st.sidebar.slider("Passages to retrieve (top-k)", 1, 20, 5)

engine = load_engine(llm_choice)
st.sidebar.success(f"LLM: {engine.llm.name}")
st.sidebar.info(f"Embeddings: {engine.model.model_id()} ({engine.model.dimension()}-dim)")

st.sidebar.divider()
st.sidebar.subheader("1. Upload BME books")

uploaded = st.sidebar.file_uploader(
    "Drop PDF files here",
    type=["pdf"],
    accept_multiple_files=True,
)

if "docs" not in st.session_state:
    st.session_state.docs = {}  # filename -> document_id

# For each uploaded file, show editable book metadata (auto-filled from the PDF).
book_inputs = {}
if uploaded:
    st.sidebar.caption("Confirm or correct the book details, then process:")
    for uf in uploaded:
        auto = extract_pdf_metadata(uf.getvalue(), uf.name)
        with st.sidebar.expander(f"📖 {uf.name}", expanded=True):
            title = st.text_input("Title", value=auto.title, key=f"title_{uf.name}")
            author = st.text_input("Author", value=auto.author, key=f"author_{uf.name}")
            edition = st.text_input("Edition (optional)", value=auto.edition, key=f"ed_{uf.name}")
            year = st.text_input("Year (optional)", value=auto.year, key=f"yr_{uf.name}")
        book_inputs[uf.name] = BookMeta(
            title=title.strip(),
            author=author.strip(),
            edition=edition.strip(),
            year=year.strip(),
            source_filename=uf.name,
        )

if st.sidebar.button("Process uploaded PDFs", type="primary", disabled=not uploaded):
    for uf in uploaded:
        with st.spinner(f"Processing {uf.name}..."):
            result = engine.ingest(uf.getvalue(), uf.name, book=book_inputs.get(uf.name))
        if result.error:
            st.sidebar.error(f"{uf.name}: {result.error}")
        elif result.already_existed:
            st.session_state.docs[uf.name] = result.document_id
            st.sidebar.info(f"{uf.name}: already processed ({result.chunk_count} chunks).")
        else:
            st.session_state.docs[uf.name] = result.document_id
            label = result.book.display() if result.book else uf.name
            st.sidebar.success(f"{label}: {result.chunk_count} chunks stored.")

if st.session_state.docs:
    st.sidebar.divider()
    st.sidebar.subheader("Library")
    for name, doc_id in st.session_state.docs.items():
        meta = engine.books.get(doc_id)
        st.sidebar.write(f"• {meta.display() if meta else name}")


# ---------------------------------------------------------------------------
# Main: ask questions
# ---------------------------------------------------------------------------

st.title("Ask your documents")

if not st.session_state.docs:
    st.info("Upload and process one or more PDFs from the sidebar to begin.")
else:
    # Optional: restrict the answer to a single document.
    doc_names = ["All documents"] + list(st.session_state.docs.keys())
    selected = st.selectbox("Search within", doc_names)
    doc_id = None if selected == "All documents" else st.session_state.docs[selected]

    question = st.text_input(
        "Your question",
        placeholder="e.g. What is the main contribution of this paper?",
    )

    if st.button("Ask", type="primary", disabled=not question.strip()):
        with st.spinner("Retrieving and generating answer..."):
            res = engine.ask(question, top_k=top_k, document_id=doc_id)

        st.subheader("Answer")
        st.markdown(res.answer)
        st.caption(f"Generated by: {res.llm_name} · retrieval status: {res.status}")

        if res.sources:
            st.subheader("📚 References")
            for s in res.sources:
                pages = ", ".join(str(p) for p in s.pages)
                author = f" — {s.author}" if s.author else ""
                st.markdown(f"- **{s.book}**{author}  *(pages: {pages})*")

        if res.chunks:
            st.subheader("Source passages")
            for i, c in enumerate(res.chunks, 1):
                header = f"[{i}] {c.book}"
                if c.author:
                    header += f" — {c.author}"
                header += f" · page {c.page} · similarity {c.similarity:.3f}"
                with st.expander(header):
                    if c.section:
                        st.caption(f"Section: {c.section}")
                    st.write(c.content)


st.divider()
st.caption(
    "Pipeline: pypdf (parse) → normalize → chunk → batch-embed "
    "(sentence-transformers) → store (disk/pgvector) → retrieve → LLM answer."
)
