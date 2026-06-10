"""Smoke / unit tests for the Retriever (Req 9).

These exercise the Retriever end-to-end against the deterministic
InMemoryVectorStore and MockEmbeddingModel. The formal property tests for
retrieval live in separate tasks (12.2-12.6).
"""

from __future__ import annotations

import pytest

from biomed_rag.config import PipelineConfig
from biomed_rag.embedding.mock import MockEmbeddingModel
from biomed_rag.models import Chunk, Embedding, VectorRecord
from biomed_rag.retrieval import (
    PLACEHOLDER_PAGE_NUMBER,
    QueryRequest,
    RetrievalStatus,
    Retriever,
)
from biomed_rag.storage.in_memory import InMemoryVectorStore


def _make_record(document_id, *, page=1, content="hello", page_set=True):
    model = MockEmbeddingModel(dimension=8)
    chunk = Chunk(
        documentId=document_id,
        content=content,
        tokenCount=3,
        orderIndex=0,
        pageNumber=page if page_set else None,
    )
    vector = model.embed(content)
    embedding = Embedding(chunkId=chunk.chunkId, vector=vector, modelId=model.model_id())
    return VectorRecord(documentId=document_id, chunk=chunk, embedding=embedding)


def _store_with(records):
    store = InMemoryVectorStore()
    # group by document id for upsert
    by_doc = {}
    for r in records:
        by_doc.setdefault(r.documentId, []).append(r)
    for doc_id, recs in by_doc.items():
        store.upsert_batch(doc_id, recs)
    return store


def _retriever(store):
    return Retriever(store, MockEmbeddingModel(dimension=8), PipelineConfig())


def test_returns_top_k_scored_chunks():
    store = _store_with([_make_record(f"doc-{i}", content=f"c{i}") for i in range(10)])
    result = _retriever(store).retrieve(QueryRequest(text="query", topK=3))
    assert result.status is RetrievalStatus.OK
    assert len(result.chunks) == 3
    for c in result.chunks:
        assert 0.0 <= c.similarity <= 1.0


def test_default_top_k_is_five():
    store = _store_with([_make_record(f"doc-{i}", content=f"c{i}") for i in range(10)])
    result = _retriever(store).retrieve(QueryRequest(text="query"))
    assert result.status is RetrievalStatus.OK
    assert len(result.chunks) == 5


def test_empty_query_rejected():
    store = _store_with([_make_record("doc-1")])
    result = _retriever(store).retrieve(QueryRequest(text="   "))
    assert result.status is RetrievalStatus.INVALID_QUERY
    assert result.chunks == []


def test_oversized_query_rejected():
    store = _store_with([_make_record("doc-1")])
    result = _retriever(store).retrieve(QueryRequest(text="x" * 4001))
    assert result.status is RetrievalStatus.INVALID_QUERY
    assert result.chunks == []


@pytest.mark.parametrize("bad_top_k", [0, -1, 101, 1000])
def test_out_of_range_top_k_rejected(bad_top_k):
    store = _store_with([_make_record("doc-1")])
    result = _retriever(store).retrieve(QueryRequest(text="query", topK=bad_top_k))
    assert result.status is RetrievalStatus.TOPK_OUT_OF_RANGE
    assert result.chunks == []


def test_empty_library_status():
    store = InMemoryVectorStore()
    result = _retriever(store).retrieve(QueryRequest(text="query"))
    assert result.status is RetrievalStatus.LIBRARY_EMPTY
    assert result.chunks == []


def test_filter_restricts_results():
    store = _store_with(
        [_make_record("doc-a", content="a"), _make_record("doc-b", content="b")]
    )
    result = _retriever(store).retrieve(
        QueryRequest(text="query", filter={"documentId": "doc-a"})
    )
    assert result.status is RetrievalStatus.OK
    assert {c.documentId for c in result.chunks} == {"doc-a"}


def test_filter_no_match_status():
    store = _store_with([_make_record("doc-a", content="a")])
    result = _retriever(store).retrieve(
        QueryRequest(text="query", filter={"documentId": "missing"})
    )
    assert result.status is RetrievalStatus.NO_MATCH
    assert result.chunks == []


def test_missing_page_number_uses_placeholder():
    store = _store_with([_make_record("doc-a", page_set=False, content="a")])
    result = _retriever(store).retrieve(QueryRequest(text="query"))
    assert result.status is RetrievalStatus.OK
    assert result.chunks[0].pageNumber == PLACEHOLDER_PAGE_NUMBER
    assert result.chunks[0].documentId == "doc-a"


def test_ordering_descending_with_documentid_tie_break():
    # Build records whose vectors tie on similarity by reusing identical content
    # so ordering falls back to ascending documentId.
    store = _store_with(
        [
            _make_record("doc-c", content="same"),
            _make_record("doc-a", content="same"),
            _make_record("doc-b", content="same"),
        ]
    )
    result = _retriever(store).retrieve(QueryRequest(text="query", topK=3))
    sims = [c.similarity for c in result.chunks]
    assert sims == sorted(sims, reverse=True)
    # equal similarities -> ascending documentId
    ids = [c.documentId for c in result.chunks]
    assert ids == sorted(ids)
