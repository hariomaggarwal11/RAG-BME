"""Unit tests for the VectorStore port and in-memory adapter (Task 11.1).

These cover the contract introduced in task 11.1 only: the port shape, the
in-memory adapter's store/replace/delete/query behaviour, atomic reprocess
replacement (Req 8.4), retrievability by document id (Req 8.2), persistence
failure retaining prior records (Req 8.6), and not-found removal (Req 8.8).
Property-based tests and the pgvector adapter are added in their own tasks.
"""

from __future__ import annotations

import pytest

from biomed_rag.config import PipelineConfig
from biomed_rag.config import VectorStoreBackend as VectorStoreBackendChoice
from biomed_rag.models import Chunk, Embedding, VectorRecord
from biomed_rag.storage import (
    DocumentNotFoundError,
    InMemoryVectorStore,
    PersistenceError,
    VectorStore,
    VectorStoreNotRegisteredError,
    VectorStoreRegistry,
)


def _record(
    document_id: str,
    *,
    order: int,
    vector: list[float],
    page: int | None = None,
    heading: list[str] | None = None,
) -> VectorRecord:
    chunk = Chunk(
        documentId=document_id,
        content=f"content-{order}",
        tokenCount=len(vector),
        orderIndex=order,
        pageNumber=page,
        headingPath=heading or [],
    )
    embedding = Embedding(chunkId=chunk.chunkId, vector=vector, modelId="mock-model")
    return VectorRecord(documentId=document_id, chunk=chunk, embedding=embedding)


def test_in_memory_implements_port() -> None:
    assert isinstance(InMemoryVectorStore(), VectorStore)


def test_upsert_then_retrieve_by_document_id() -> None:
    store = InMemoryVectorStore()
    r1 = _record("doc-1", order=0, vector=[1.0, 0.0])
    r2 = _record("doc-1", order=1, vector=[0.0, 1.0])
    result = store.upsert_batch("doc-1", [r1, r2])

    assert result.replaced is False
    assert set(result.stored_chunk_ids) == {r1.chunk.chunkId, r2.chunk.chunkId}
    # Req 8.2: exactly the records stored under the id, no more, no fewer.
    stored = store.get_document("doc-1")
    assert {rec.chunk.chunkId for rec in stored} == {r1.chunk.chunkId, r2.chunk.chunkId}


def test_upsert_rejects_record_for_other_document() -> None:
    store = InMemoryVectorStore()
    foreign = _record("doc-2", order=0, vector=[1.0])
    with pytest.raises(PersistenceError):
        store.upsert_batch("doc-1", [foreign])
    assert store.get_document("doc-1") == []


def test_replace_document_is_atomic_swap() -> None:
    store = InMemoryVectorStore()
    old = _record("doc-1", order=0, vector=[1.0, 0.0])
    store.upsert_batch("doc-1", [old])

    new1 = _record("doc-1", order=0, vector=[0.0, 1.0])
    new2 = _record("doc-1", order=1, vector=[1.0, 1.0])
    result = store.replace_document("doc-1", [new1, new2])

    assert result.replaced is True
    stored_ids = {rec.chunk.chunkId for rec in store.get_document("doc-1")}
    # Old chunk is gone; only the new set remains (Req 8.4).
    assert stored_ids == {new1.chunk.chunkId, new2.chunk.chunkId}
    assert old.chunk.chunkId not in stored_ids


def test_persistence_failure_retains_prior_records() -> None:
    # commit_hook fires on the *next* write; install prior records first with a
    # store that never fails, then point the failing store at the same data.
    store = InMemoryVectorStore()
    prior = _record("doc-1", order=0, vector=[1.0, 0.0])
    store.upsert_batch("doc-1", [prior])

    # Now make subsequent writes fail mid-commit.
    store._commit_hook = lambda: (_ for _ in ()).throw(RuntimeError("disk full"))
    newrec = _record("doc-1", order=1, vector=[0.0, 1.0])
    with pytest.raises(PersistenceError):
        store.replace_document("doc-1", [newrec])

    # Req 8.6: prior records retained unchanged after the failure.
    stored_ids = {rec.chunk.chunkId for rec in store.get_document("doc-1")}
    assert stored_ids == {prior.chunk.chunkId}


def test_delete_document_removes_all_records() -> None:
    store = InMemoryVectorStore()
    r1 = _record("doc-1", order=0, vector=[1.0])
    r2 = _record("doc-1", order=1, vector=[2.0])
    store.upsert_batch("doc-1", [r1, r2])

    result = store.delete_document("doc-1")
    assert set(result.deleted_chunk_ids) == {r1.chunk.chunkId, r2.chunk.chunkId}
    assert store.get_document("doc-1") == []  # Req 8.5


def test_delete_unknown_document_raises_not_found() -> None:
    store = InMemoryVectorStore()
    with pytest.raises(DocumentNotFoundError):
        store.delete_document("missing")  # Req 8.8


def test_query_orders_by_descending_similarity() -> None:
    store = InMemoryVectorStore()
    near = _record("doc-a", order=0, vector=[1.0, 0.0])
    far = _record("doc-b", order=0, vector=[-1.0, 0.0])
    store.upsert_batch("doc-a", [near])
    store.upsert_batch("doc-b", [far])

    results = store.query([1.0, 0.0], top_k=2)
    assert [s.record.chunk.chunkId for s in results] == [
        near.chunk.chunkId,
        far.chunk.chunkId,
    ]
    assert all(0.0 <= s.similarity <= 1.0 for s in results)
    assert results[0].similarity >= results[1].similarity


def test_query_top_k_limits_results() -> None:
    store = InMemoryVectorStore()
    for i in range(5):
        rec = _record("doc-1", order=i, vector=[float(i + 1), 0.0])
        store.upsert_batch("doc-1", [rec])
    assert len(store.query([1.0, 0.0], top_k=3)) == 3


def test_query_filter_by_document_id() -> None:
    store = InMemoryVectorStore()
    a = _record("doc-a", order=0, vector=[1.0, 0.0])
    b = _record("doc-b", order=0, vector=[1.0, 0.0])
    store.upsert_batch("doc-a", [a])
    store.upsert_batch("doc-b", [b])

    results = store.query([1.0, 0.0], top_k=10, filter={"documentId": "doc-a"})
    assert [s.record.chunk.chunkId for s in results] == [a.chunk.chunkId]


def test_registry_selects_in_memory_from_config() -> None:
    registry = VectorStoreRegistry()
    registry.register(VectorStoreBackendChoice.PGVECTOR, InMemoryVectorStore)
    config = PipelineConfig(vector_store_backend=VectorStoreBackendChoice.PGVECTOR)
    assert isinstance(registry.select(config), InMemoryVectorStore)


def test_registry_unregistered_choice_raises() -> None:
    registry = VectorStoreRegistry()
    with pytest.raises(VectorStoreNotRegisteredError):
        registry.create(VectorStoreBackendChoice.QDRANT)
