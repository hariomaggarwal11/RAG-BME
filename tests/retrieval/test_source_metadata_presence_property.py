"""Property test for source metadata presence on returned chunks (Task 12.4).

Feature: biomedical-rag-pipeline, Property 24: Returned chunks always carry source metadata

This exercises Req 9.4 and 9.5 against the deterministic ``InMemoryVectorStore``
adapter and ``MockEmbeddingModel`` (per the design Testing Strategy: retrieval
properties run against in-memory/mock backends for fast 100+ iteration runs).

The property: for any returned chunk, the result carries a non-None document
identifier and page number, substituting the defined placeholder exactly when
the underlying source metadata is unavailable (here: a missing ``pageNumber``).
"""

from __future__ import annotations

from typing import Dict, List, Optional

from hypothesis import given, settings
from hypothesis import strategies as st

from biomed_rag.config import PipelineConfig
from biomed_rag.embedding.mock import MockEmbeddingModel
from biomed_rag.models import Chunk, Embedding, VectorRecord
from biomed_rag.retrieval import (
    PLACEHOLDER_PAGE_NUMBER,
    QueryRequest,
    RetrievalStatus,
    Retriever,
)
from biomed_rag.storage import InMemoryVectorStore

_EMBED_DIM = 8

# Safe, non-empty identifier text (Chunk/VectorRecord require non-empty ids).
_ID_TEXT = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_",
    min_size=1,
    max_size=12,
)

# pageNumber is either present (a real page) or absent (None -> placeholder).
_PAGE_NUMBER = st.one_of(st.none(), st.integers(min_value=0, max_value=5000))


@st.composite
def _library(draw: st.DrawFn):
    """Generate a non-empty stored library plus the expected pageNumber per chunk.

    Some chunks carry a real ``pageNumber`` and some carry ``None`` so the
    placeholder substitution path (Req 9.5) is exercised alongside the
    real-metadata path (Req 9.4). Chunk ids are globally unique so the expected
    map is unambiguous.
    """
    model = MockEmbeddingModel(dimension=_EMBED_DIM)
    document_ids = draw(st.lists(_ID_TEXT, unique=True, min_size=1, max_size=4))

    records_by_doc: Dict[str, List[VectorRecord]] = {}
    # expected[chunkId] = (documentId, pageNumber-as-stored)
    expected: Dict[str, tuple] = {}
    total = 0

    for document_id in document_ids:
        count = draw(st.integers(min_value=0, max_value=5))
        records: List[VectorRecord] = []
        for order in range(count):
            page: Optional[int] = draw(_PAGE_NUMBER)
            content = f"content-{document_id}-{order}"
            chunk_id = f"{document_id}::chunk-{order}"
            chunk = Chunk(
                documentId=document_id,
                content=content,
                tokenCount=3,
                orderIndex=order,
                pageNumber=page,
                chunkId=chunk_id,
            )
            vector = model.embed(content)
            embedding = Embedding(
                chunkId=chunk_id, vector=vector, modelId=model.model_id()
            )
            records.append(
                VectorRecord(documentId=document_id, chunk=chunk, embedding=embedding)
            )
            expected[chunk_id] = (document_id, page)
            total += 1
        records_by_doc[document_id] = records

    # Ensure at least one stored record so retrieval can return OK with chunks.
    if total == 0:
        document_id = document_ids[0]
        chunk_id = f"{document_id}::chunk-0"
        page = draw(_PAGE_NUMBER)
        chunk = Chunk(
            documentId=document_id,
            content="content-seed",
            tokenCount=3,
            orderIndex=0,
            pageNumber=page,
            chunkId=chunk_id,
        )
        vector = model.embed("content-seed")
        embedding = Embedding(
            chunkId=chunk_id, vector=vector, modelId=model.model_id()
        )
        records_by_doc[document_id] = [
            VectorRecord(documentId=document_id, chunk=chunk, embedding=embedding)
        ]
        expected[chunk_id] = (document_id, page)
        total = 1

    return records_by_doc, expected, total


@settings(max_examples=200, deadline=None)
@given(data=_library())
def test_returned_chunks_always_carry_source_metadata(data) -> None:
    # Feature: biomedical-rag-pipeline, Property 24: Returned chunks always carry source metadata
    records_by_doc, expected, total = data

    store = InMemoryVectorStore()
    for document_id, records in records_by_doc.items():
        if records:
            store.upsert_batch(document_id, records)

    retriever = Retriever(
        store, MockEmbeddingModel(dimension=_EMBED_DIM), PipelineConfig()
    )
    # Request all stored chunks (topK is capped at 100; total is always <= 20).
    result = retriever.retrieve(QueryRequest(text="query", topK=min(100, total)))

    assert result.status is RetrievalStatus.OK
    assert result.chunks, "a non-empty library must yield at least one chunk"

    for chunk in result.chunks:
        # Every returned chunk carries source metadata (Req 9.4): never None.
        assert chunk.documentId is not None
        assert chunk.pageNumber is not None

        source_document_id, source_page = expected[chunk.chunkId]

        # documentId is always the real, present source identifier (Req 9.4).
        assert chunk.documentId == source_document_id

        # pageNumber uses the placeholder exactly when the source is missing it,
        # and the real value otherwise (Req 9.5).
        if source_page is None:
            assert chunk.pageNumber == PLACEHOLDER_PAGE_NUMBER
        else:
            assert chunk.pageNumber == source_page
            assert chunk.pageNumber != PLACEHOLDER_PAGE_NUMBER
