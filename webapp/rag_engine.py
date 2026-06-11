"""Reusable RAG engine for the web app.

Wraps the biomed_rag pipeline components into a simple, fast, production-friendly
engine with three capabilities the demo scripts lacked:

1. Batch embedding   - all chunks of a document are embedded in one batched call
                       to sentence-transformers (much faster than per-chunk).
2. Persistence       - embeddings are stored durably so a PDF is processed once
                       and survives across runs/sessions. Uses pgvector when a
                       DSN is configured, otherwise a local disk-backed store.
3. LLM generation    - retrieved chunks are fed to an LLM to produce a true
                       natural-language answer (see webapp/llm.py).

Ingestion uses a direct flow (parse -> normalize -> chunk -> batch-embed ->
store) so batching and skip-if-already-stored both work cleanly.
"""

from __future__ import annotations

import hashlib
import io
import os
import pickle
from dataclasses import dataclass
from typing import List, Optional

from sentence_transformers import SentenceTransformer

from biomed_rag.config import PipelineConfig
from biomed_rag.embedding.model import EmbeddingModel
from biomed_rag.models import (
    Embedding,
    Format,
    VectorRecord,
    document_id_from_hash,
)
from biomed_rag.normalization.normalizer import Normalizer
from biomed_rag.normalization.result import Malformed
from biomed_rag.chunking.chunker import Chunker
from biomed_rag.parsing.engine import ParseError, ParsingEngine
from biomed_rag.parsing.parser import ParseFailure, Parser
from biomed_rag.parsing.raw_result import RawBlock, RawPage, RawParseResult, SourceDocument
from biomed_rag.parsing.registry import ParsingEngineRegistry
from biomed_rag.retrieval.retriever import QueryRequest, Retriever, RetrievalStatus
from biomed_rag.storage.in_memory import InMemoryVectorStore
from biomed_rag.storage.port import VectorStore

from .llm import ContextChunk, get_llm
from .book_metadata import BookMeta, BookRegistry, extract_pdf_metadata

# Embedding model config (local, free, 384-dim, good quality).
MODEL_ID = "st-minilm"
MODEL_NAME = os.environ.get("RAG_EMBED_MODEL", "all-MiniLM-L6-v2")
DIMENSION = 384

# Where the local disk-backed vector store persists (used when no pgvector DSN).
DEFAULT_STORE_PATH = os.environ.get("RAG_STORE_PATH", ".rag_store.pkl")


# ---------------------------------------------------------------------------
# PDF parsing engine (pypdf; no OCR, no Docling)
# ---------------------------------------------------------------------------

class PyPDFEngine(ParsingEngine):
    """Reliable text-layer PDF parser built on pypdf (no OCR)."""

    def engine_id(self) -> str:
        return "pypdf"

    def is_available(self) -> bool:
        try:
            import pypdf  # noqa: F401
            return True
        except ImportError:
            return False

    def parse(self, doc: SourceDocument, deadline: Optional[float] = None) -> RawParseResult:
        from pypdf import PdfReader

        try:
            reader = PdfReader(io.BytesIO(doc.raw_bytes))
        except Exception as exc:
            raise ParseError(f"pypdf could not open the PDF: {exc}") from exc

        blocks: List[RawBlock] = []
        pages: List[RawPage] = []
        for page_index, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception:
                text = ""
            has_text = bool(text.strip())
            pages.append(RawPage(page_number=page_index, has_text_layer=has_text))
            paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
            if not paragraphs and has_text:
                paragraphs = [text.strip()]
            for para in paragraphs:
                blocks.append(RawBlock(text=para, page_number=page_index, kind="paragraph"))

        return RawParseResult(engine_id="pypdf", blocks=blocks, pages=pages)


# ---------------------------------------------------------------------------
# Embedding model (sentence-transformers) with a batch path
# ---------------------------------------------------------------------------

class SentenceTransformerModel(EmbeddingModel):
    """Real embedding model; loaded once and reused for the whole session."""

    def __init__(self, model_name: str = MODEL_NAME, model_id: str = MODEL_ID) -> None:
        self._model_id = model_id
        self._model = SentenceTransformer(model_name)
        if hasattr(self._model, "get_embedding_dimension"):
            self._dimension = self._model.get_embedding_dimension()
        else:
            self._dimension = self._model.get_sentence_embedding_dimension()

    def model_id(self) -> str:
        return self._model_id

    def dimension(self) -> int:
        return self._dimension

    def embed(self, text: str, deadline: Optional[float] = None) -> List[float]:
        return self._model.encode(text, normalize_embeddings=True).tolist()

    def embed_batch(self, texts: List[str], batch_size: int = 64) -> List[List[float]]:
        """Embed many texts in one batched call (true batch embedding)."""
        if not texts:
            return []
        vectors = self._model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=batch_size,
            show_progress_bar=False,
        )
        return [v.tolist() for v in vectors]


# ---------------------------------------------------------------------------
# Disk-backed persistent vector store (no external DB required)
# ---------------------------------------------------------------------------

class DiskVectorStore(InMemoryVectorStore):
    """In-memory store that persists its contents to a pickle file on disk.

    Gives durable storage with zero database setup: the document's chunks and
    embeddings are written to ``path`` after every change and reloaded on start,
    so a PDF processed once is reused across runs.
    """

    def __init__(self, path: str = DEFAULT_STORE_PATH, **kwargs) -> None:
        super().__init__(**kwargs)
        self._path = path
        self._load()

    def _load(self) -> None:
        if os.path.exists(self._path):
            try:
                with open(self._path, "rb") as f:
                    self._by_document = pickle.load(f)
            except Exception:
                # Corrupt/old cache: start fresh rather than crash.
                self._by_document = {}

    def _save(self) -> None:
        with open(self._path, "wb") as f:
            pickle.dump(self._by_document, f)

    def _commit(self, document_id, new_records) -> None:  # type: ignore[override]
        super()._commit(document_id, new_records)
        self._save()

    def delete_document(self, document_id):  # type: ignore[override]
        result = super().delete_document(document_id)
        self._save()
        return result


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

@dataclass
class IngestResult:
    document_id: str
    filename: str
    chunk_count: int
    already_existed: bool
    book: Optional["BookMeta"] = None
    error: Optional[str] = None


@dataclass
class Source:
    """A unique source book cited in an answer."""

    book: str
    author: str
    pages: List[object]


@dataclass
class AnswerResult:
    answer: str
    chunks: List[ContextChunk]
    llm_name: str
    status: str
    sources: List[Source] = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# The engine
# ---------------------------------------------------------------------------

class RagEngine:
    """End-to-end RAG engine: ingest PDFs, retrieve, and generate answers."""

    def __init__(
        self,
        store_path: str = DEFAULT_STORE_PATH,
        top_k: int = 5,
        llm_provider: Optional[str] = None,
        books_path: Optional[str] = None,
    ) -> None:
        self.config = PipelineConfig(
            parsing_engine="docling",  # config key; we register pypdf under it
            embedding_model=MODEL_ID,
            embedding_dimension=DIMENSION,
        )
        self.top_k = top_k

        # Parser with our pypdf engine.
        self._parsing_registry = ParsingEngineRegistry()
        self._parsing_registry.register(self.config.parsing_engine, lambda: PyPDFEngine())
        self.parser = Parser(config=self.config, registry=self._parsing_registry)

        self.normalizer = Normalizer()
        self.chunker = Chunker()

        # Embedding model (loaded once).
        self.model = SentenceTransformerModel()

        # Vector store: pgvector if a DSN is configured, else disk-backed.
        self.vector_store: VectorStore = self._build_store(store_path)

        # Book metadata registry (for citations: book title + author).
        self.books = BookRegistry(books_path) if books_path else BookRegistry()

        # Retriever (uses the same embedding model for queries).
        self.retriever = Retriever(self.vector_store, self.model, self.config)

        # LLM for answer generation.
        self.llm = get_llm(llm_provider)

    @staticmethod
    def _build_store(store_path: str) -> VectorStore:
        """Use pgvector when BIOMED_RAG_PGVECTOR_DSN is set; otherwise disk."""
        if os.environ.get("BIOMED_RAG_PGVECTOR_DSN"):
            from biomed_rag.storage.pgvector_adapter import pgvector_store_from_env

            return pgvector_store_from_env()
        return DiskVectorStore(store_path)

    # -- ingestion (parse -> normalize -> chunk -> batch-embed -> store) --
    def ingest(
        self,
        data: bytes,
        filename: str,
        book: Optional["BookMeta"] = None,
    ) -> IngestResult:
        """Process a PDF and store its chunk embeddings (idempotent).

        Book metadata (title/author) is captured for citations: auto-extracted
        from the PDF when ``book`` is not supplied, then recorded in the
        registry keyed by documentId. If the same document (by content hash) is
        already stored, processing is skipped.
        """
        content_hash = hashlib.sha256(data).hexdigest()
        document_id = document_id_from_hash(content_hash)

        # Resolve book metadata: explicit override, else auto from the PDF.
        if book is None:
            book = extract_pdf_metadata(data, filename)
        book.source_filename = book.source_filename or filename
        self.books.set(document_id, book)

        # Skip if already ingested (persistence => no re-processing).
        existing = self.vector_store.get_document(document_id)
        if existing:
            return IngestResult(
                document_id=document_id,
                filename=filename,
                chunk_count=len(existing),
                already_existed=True,
                book=book,
            )

        source = SourceDocument(
            document_id=document_id,
            raw_bytes=data,
            doc_format=Format.PDF,
        )

        # Parse.
        try:
            parsed = self.parser.parse(None, source)
        except ParseFailure as exc:
            return IngestResult(document_id, filename, 0, False, book=book, error=f"parse failed: {exc.reason}")

        # Normalize.
        norm = self.normalizer.normalize(parsed)
        if isinstance(norm, Malformed):
            return IngestResult(document_id, filename, 0, False, book=book, error=f"normalize failed: {norm.error}")
        normalized_doc = norm.document

        # Chunk.
        chunks = self.chunker.chunk(normalized_doc, self.config)
        if not chunks:
            return IngestResult(document_id, filename, 0, False, book=book, error="no extractable text found")

        # Batch-embed all chunks at once.
        vectors = self.model.embed_batch([c.content for c in chunks])

        # Build records and store.
        records: List[VectorRecord] = []
        for chunk, vector in zip(chunks, vectors):
            embedding = Embedding(chunkId=chunk.chunkId, vector=vector, modelId=MODEL_ID)
            records.append(VectorRecord(documentId=document_id, chunk=chunk, embedding=embedding))

        self.vector_store.upsert_batch(document_id, records)

        return IngestResult(
            document_id=document_id,
            filename=filename,
            chunk_count=len(records),
            already_existed=False,
            book=book,
        )

    # -- query + generate -------------------------------------------------
    def ask(
        self,
        question: str,
        top_k: Optional[int] = None,
        document_id: Optional[str] = None,
    ) -> AnswerResult:
        """Retrieve relevant chunks and generate a natural-language answer."""
        filt = {"documentId": document_id} if document_id else None
        result = self.retriever.retrieve(
            QueryRequest(text=question, topK=top_k or self.top_k, filter=filt)
        )

        chunks: List[ContextChunk] = []
        for c in result.chunks:
            meta = self.books.get(str(c.documentId))
            chunks.append(
                ContextChunk(
                    content=c.content,
                    similarity=c.similarity,
                    page=c.pageNumber,
                    document_id=c.documentId,
                    book=(meta.title if meta and meta.title else "Unknown source"),
                    author=(meta.author if meta else ""),
                    section=" > ".join(c.headingPath) if c.headingPath else "",
                )
            )

        if result.status is not RetrievalStatus.OK or not chunks:
            return AnswerResult(
                answer=_status_message(result.status),
                chunks=chunks,
                llm_name=self.llm.name,
                status=result.status.value,
                sources=[],
            )

        answer = self.llm.generate(question, chunks)
        return AnswerResult(
            answer=answer,
            chunks=chunks,
            llm_name=self.llm.name,
            status=result.status.value,
            sources=_collect_sources(chunks),
        )


def _status_message(status: RetrievalStatus) -> str:
    return {
        RetrievalStatus.LIBRARY_EMPTY: "No documents have been ingested yet. Upload a PDF first.",
        RetrievalStatus.NO_MATCH: "No matching passages were found for that query/filter.",
        RetrievalStatus.INVALID_QUERY: "The query was empty or too long.",
        RetrievalStatus.TOPK_OUT_OF_RANGE: "The requested number of results is out of range (1-100).",
    }.get(status, "No results.")


def _collect_sources(chunks: List[ContextChunk]) -> List[Source]:
    """Group retrieved chunks into unique (book, author) sources with pages."""
    grouped: dict = {}
    for c in chunks:
        key = (c.book, c.author)
        if key not in grouped:
            grouped[key] = []
        if c.page not in grouped[key]:
            grouped[key].append(c.page)
    sources: List[Source] = []
    for (book, author), pages in grouped.items():
        # Sort pages when they are all comparable (ints); leave as-is otherwise.
        try:
            pages = sorted(pages, key=lambda p: (isinstance(p, str), p))
        except TypeError:
            pass
        sources.append(Source(book=book, author=author, pages=pages))
    return sources
