"""The Retriever: query the Knowledge_Library for grounded LLM context (Req 9).

The :class:`Retriever` turns a natural-language :class:`QueryRequest` into a
ranked set of stored chunks. It depends only on two injected ports — the
pluggable :class:`~biomed_rag.storage.VectorStore` and the pluggable
:class:`~biomed_rag.embedding.EmbeddingModel` — so it is independent of any
concrete backend (design: ports-and-adapters).

Behaviour (design Retriever section, Req 9):

* Validate the query text: reject empty or longer than ``maxQueryChars`` (4000)
  with an ``INVALID_QUERY`` status and no chunks (Req 9.2).
* Validate the requested count: reject ``topK`` < 1 or > 100 with a
  ``TOPK_OUT_OF_RANGE`` status (Req 9.3); ``topK`` defaults to
  ``config.default_top_k`` (5) when omitted (Req 9.1).
* Embed the query text and ask the store for the top-K most similar records,
  each carrying a similarity score in [0.0, 1.0] (Req 9.1).
* Attach source metadata (documentId, pageNumber) to every returned chunk,
  substituting a placeholder when a field is unavailable (Req 9.4, 9.5).
* Apply an optional metadata filter (Req 9.7); a filter that matches nothing
  yields an empty result with a ``NO_MATCH`` status (Req 9.8).
* An empty Knowledge_Library yields an empty result with a ``LIBRARY_EMPTY``
  status (Req 9.6).
* Order results by descending similarity, breaking ties by ascending
  documentId (Req 9.9).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Mapping, Optional, Sequence

from biomed_rag.config import PipelineConfig
from biomed_rag.embedding.model import EmbeddingModel
from biomed_rag.models import DocumentId, ScoredRecord
from biomed_rag.storage.port import MetadataFilter, VectorStore

# Placeholder values returned when a chunk's source metadata is unavailable
# (Req 9.5). documentId is structurally always present on a stored record, but
# the placeholder is defined for completeness and defensive backend handling.
PLACEHOLDER_DOCUMENT_ID = "UNKNOWN_DOCUMENT"
PLACEHOLDER_PAGE_NUMBER = "UNKNOWN_PAGE"

# The Retriever's own latency budget (Req 9.1: results within 2 seconds). Used
# as the embedding deadline so a slow model surfaces as a timeout rather than
# blowing the budget silently.
_RETRIEVAL_BUDGET_SECONDS = 2.0


class RetrievalStatus(str, Enum):
    """Outcome status carried by every :class:`RetrievalResult` (Req 9)."""

    OK = "ok"
    INVALID_QUERY = "invalid_query"          # empty/oversized text (Req 9.2)
    TOPK_OUT_OF_RANGE = "topk_out_of_range"  # topK < 1 or > 100 (Req 9.3)
    LIBRARY_EMPTY = "library_empty"          # no chunks stored (Req 9.6)
    NO_MATCH = "no_match"                    # filter matched nothing (Req 9.8)


@dataclass(frozen=True)
class QueryRequest:
    """A retrieval request: query text with an optional count and filter.

    ``topK`` defaults to the configured ``default_top_k`` when ``None``
    (Req 9.1). ``filter`` is a mapping of metadata field to required value
    applied by the store (Req 9.7).
    """

    text: str
    topK: Optional[int] = None
    filter: Optional[Mapping[str, object]] = None


@dataclass(frozen=True)
class RetrievedChunk:
    """A single ranked result with its content, score, and source metadata.

    ``documentId`` and ``pageNumber`` always carry a value: the real metadata
    when available, otherwise a placeholder (Req 9.4, 9.5).
    """

    chunkId: str
    content: str
    similarity: float
    documentId: object
    pageNumber: object
    headingPath: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class RetrievalResult:
    """The result of a retrieval: a status plus the (possibly empty) chunks.

    ``chunks`` is empty whenever ``status`` is anything other than ``OK``
    (Req 9.2, 9.3, 9.6, 9.8).
    """

    status: RetrievalStatus
    chunks: List[RetrievedChunk] = field(default_factory=list)

    @property
    def is_ok(self) -> bool:
        return self.status is RetrievalStatus.OK


class Retriever:
    """Ranks stored chunks for a query against the Vector_Store (Req 9)."""

    # Bounds for the requested result count (Req 9.1, 9.3).
    MIN_TOP_K = 1
    MAX_TOP_K = 100

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_model: EmbeddingModel,
        config: Optional[PipelineConfig] = None,
    ) -> None:
        """Inject the storage and embedding ports plus the pipeline config.

        ``config`` supplies ``default_top_k`` (Req 9.1) and ``max_query_chars``
        (Req 9.2); it defaults to a stock :class:`PipelineConfig`.
        """
        self._store = vector_store
        self._model = embedding_model
        self._config = config if config is not None else PipelineConfig()

    def retrieve(self, query: QueryRequest) -> RetrievalResult:
        """Return the top-K chunks most similar to ``query`` (Req 9)."""
        # 1. Validate query text: empty or oversized -> invalid (Req 9.2).
        if not self._is_valid_text(query.text) or not self._is_valid_length(query.text):
            return RetrievalResult(status=RetrievalStatus.INVALID_QUERY, chunks=[])

        # 2. Resolve and validate the requested count (Req 9.1, 9.3).
        top_k = query.topK if query.topK is not None else self._config.default_top_k
        if not self._is_valid_top_k(top_k):
            return RetrievalResult(status=RetrievalStatus.TOPK_OUT_OF_RANGE, chunks=[])

        # 3. Embed the query text within the retrieval budget (Req 9.1).
        deadline = time.monotonic() + _RETRIEVAL_BUDGET_SECONDS
        embedding = self._model.embed(query.text, deadline=deadline)

        # 4. Ask the store for candidates (with the optional filter, Req 9.7).
        scored = self._store.query(embedding, top_k=top_k, filter=query.filter)

        if scored:
            ordered = self._order(scored)[:top_k]
            return RetrievalResult(
                status=RetrievalStatus.OK,
                chunks=[self._to_retrieved(s) for s in ordered],
            )

        # 5. Empty result: distinguish an empty library from a no-match filter
        #    by probing the store without the filter (Req 9.6, 9.8).
        if self._library_is_empty(embedding):
            return RetrievalResult(status=RetrievalStatus.LIBRARY_EMPTY, chunks=[])
        return RetrievalResult(status=RetrievalStatus.NO_MATCH, chunks=[])

    # -- validation -------------------------------------------------------
    @staticmethod
    def _is_valid_text(text: object) -> bool:
        """True when ``text`` is a non-empty string within the length limit."""
        return isinstance(text, str) and len(text.strip()) > 0

    def _is_valid_length(self, text: str) -> bool:
        return len(text) <= self._config.max_query_chars

    def _is_valid_top_k(self, top_k: object) -> bool:
        return (
            isinstance(top_k, int)
            and not isinstance(top_k, bool)
            and self.MIN_TOP_K <= top_k <= self.MAX_TOP_K
        )

    # -- ordering & shaping ----------------------------------------------
    @staticmethod
    def _order(scored: Sequence[ScoredRecord]) -> List[ScoredRecord]:
        """Order by descending similarity, ties broken by ascending documentId.

        Re-applied here so the contract (Req 9.9) holds regardless of how a
        particular Vector_Store backend orders its own results.
        """
        return sorted(
            scored,
            key=lambda s: (-s.similarity, s.record.documentId),
        )

    @staticmethod
    def _to_retrieved(scored: ScoredRecord) -> RetrievedChunk:
        """Build a :class:`RetrievedChunk`, filling placeholders for missing
        source metadata (Req 9.4, 9.5)."""
        record = scored.record
        chunk = record.chunk
        document_id: object = record.documentId or PLACEHOLDER_DOCUMENT_ID
        page_number: object = (
            chunk.pageNumber
            if chunk.pageNumber is not None
            else PLACEHOLDER_PAGE_NUMBER
        )
        return RetrievedChunk(
            chunkId=chunk.chunkId,
            content=chunk.content,
            similarity=scored.similarity,
            documentId=document_id,
            pageNumber=page_number,
            headingPath=list(chunk.headingPath),
        )

    # -- library emptiness probe -----------------------------------------
    def _library_is_empty(self, embedding: Sequence[float]) -> bool:
        """True when the store holds no records at all.

        Probes with an unfiltered query: if even an unfiltered query returns
        nothing, the Knowledge_Library is empty (Req 9.6); otherwise the empty
        result was caused by the supplied filter (Req 9.8).
        """
        probe = self._store.query(embedding, top_k=1, filter=None)
        return len(probe) == 0
