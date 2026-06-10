"""Unit tests for Retriever query validation and empty-library status (Req 9).

Task 12.6 focuses on the input-validation gate and the empty-library outcome:

* empty / whitespace / oversized query text -> ``INVALID_QUERY`` (Req 9.2)
* ``topK`` outside [1, 100] -> ``TOPK_OUT_OF_RANGE`` (Req 9.3)
* accepted boundaries (4000-char query; ``topK`` 1 and 100)
* an empty Knowledge_Library -> ``LIBRARY_EMPTY`` (Req 9.6)

These complement the broader behaviour exercised in ``test_retriever_smoke``
by pinning down the exact accept/reject boundaries.
"""

from __future__ import annotations

import pytest

from biomed_rag.config import PipelineConfig
from biomed_rag.embedding.mock import MockEmbeddingModel
from biomed_rag.models import Chunk, Embedding, VectorRecord
from biomed_rag.retrieval import (
    QueryRequest,
    RetrievalStatus,
    Retriever,
)
from biomed_rag.storage.in_memory import InMemoryVectorStore

_MAX_QUERY_CHARS = 4000


def _make_record(document_id, *, content="hello"):
    model = MockEmbeddingModel(dimension=8)
    chunk = Chunk(
        documentId=document_id,
        content=content,
        tokenCount=3,
        orderIndex=0,
        pageNumber=1,
    )
    vector = model.embed(content)
    embedding = Embedding(chunkId=chunk.chunkId, vector=vector, modelId=model.model_id())
    return VectorRecord(documentId=document_id, chunk=chunk, embedding=embedding)


def _populated_store():
    store = InMemoryVectorStore()
    # Each record is upserted under its own document id (the store requires the
    # batch id to match every record's documentId).
    for i in range(5):
        doc_id = f"doc-{i}"
        store.upsert_batch(doc_id, [_make_record(doc_id, content=f"c{i}")])
    return store


def _retriever(store):
    return Retriever(store, MockEmbeddingModel(dimension=8), PipelineConfig())


# -- query text validation (Req 9.2) --------------------------------------
@pytest.mark.parametrize("bad_text", ["", " ", "   ", "\t", "\n", "  \t \n "])
def test_empty_or_whitespace_query_rejected(bad_text):
    result = _retriever(_populated_store()).retrieve(QueryRequest(text=bad_text))
    assert result.status is RetrievalStatus.INVALID_QUERY
    assert result.chunks == []


def test_query_at_max_length_is_accepted():
    # Exactly maxQueryChars (4000) is within the limit and must not be rejected.
    text = "x" * _MAX_QUERY_CHARS
    result = _retriever(_populated_store()).retrieve(QueryRequest(text=text))
    assert result.status is not RetrievalStatus.INVALID_QUERY


def test_query_one_over_max_length_is_rejected():
    text = "x" * (_MAX_QUERY_CHARS + 1)
    result = _retriever(_populated_store()).retrieve(QueryRequest(text=text))
    assert result.status is RetrievalStatus.INVALID_QUERY
    assert result.chunks == []


# -- topK range validation (Req 9.3) ---------------------------------------
@pytest.mark.parametrize("bad_top_k", [0, -1, -100, 101, 1000])
def test_out_of_range_top_k_rejected(bad_top_k):
    result = _retriever(_populated_store()).retrieve(
        QueryRequest(text="query", topK=bad_top_k)
    )
    assert result.status is RetrievalStatus.TOPK_OUT_OF_RANGE
    assert result.chunks == []


@pytest.mark.parametrize("good_top_k", [1, 100])
def test_boundary_top_k_accepted(good_top_k):
    # 1 and 100 are the inclusive bounds and must pass validation.
    result = _retriever(_populated_store()).retrieve(
        QueryRequest(text="query", topK=good_top_k)
    )
    assert result.status is not RetrievalStatus.TOPK_OUT_OF_RANGE


def test_top_k_validation_runs_after_query_validation():
    # An invalid query short-circuits before topK is even considered.
    result = _retriever(_populated_store()).retrieve(QueryRequest(text="", topK=0))
    assert result.status is RetrievalStatus.INVALID_QUERY


# -- empty library status (Req 9.6) ----------------------------------------
def test_empty_library_returns_library_empty():
    store = InMemoryVectorStore()
    result = _retriever(store).retrieve(QueryRequest(text="query", topK=5))
    assert result.status is RetrievalStatus.LIBRARY_EMPTY
    assert result.chunks == []


def test_empty_library_with_filter_still_reports_empty():
    # With no chunks stored, even a filtered query reports the library is empty
    # rather than a no-match (Req 9.6 takes precedence over Req 9.8).
    store = InMemoryVectorStore()
    result = _retriever(store).retrieve(
        QueryRequest(text="query", filter={"documentId": "doc-a"})
    )
    assert result.status is RetrievalStatus.LIBRARY_EMPTY
    assert result.chunks == []
