"""Unit/integration tests for persistence failure, not-found, and latency (Task 11.6).

This module targets three Requirement 8 behaviours that the property tests do not
cover directly, exercising both the in-memory adapter and the pgvector adapter:

* **Persistence failure retains prior chunks and returns an error (Req 8.6).**
  A simulated mid-write failure must leave the previously stored chunks intact
  and surface a :class:`PersistenceError`. The in-memory adapter is driven via
  its ``commit_hook`` fault knob; the pgvector adapter via a DB-API fake whose
  ``fail_on`` triggers a driver error after the delete half of a swap has
  staged.
* **Removal of an unknown id returns not-found (Req 8.8).** Deleting a document
  id with no stored records raises :class:`DocumentNotFoundError` and commits
  nothing.
* **pgvector persistence latency (Req 8.1).** Persistence must complete within
  5 seconds of generation. A real PostgreSQL is not generally available, so the
  budget is asserted against the fake-connection pgvector write path (the same
  SQL/transaction code an embedding store call exercises). A real-database
  variant is provided but skips cleanly unless ``BIOMED_RAG_PGVECTOR_DSN`` names
  a reachable database.

The pgvector tests use a self-contained DB-API 2.0 fake mirroring the
``FakeConnection`` pattern in ``test_pgvector_adapter.py``: writes buffer in a
working copy until ``commit`` and are discarded on ``rollback``.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional, Tuple

import pytest

from biomed_rag.models import Chunk, Embedding, VectorRecord
from biomed_rag.storage import (
    DocumentNotFoundError,
    InMemoryVectorStore,
    PersistenceError,
    PgVectorStore,
)
from biomed_rag.storage import pgvector_adapter
from biomed_rag.storage.pgvector_adapter import _COLUMNS, DSN_ENV_VAR

# The persistence budget from Req 8.1: an embedding must be persisted within
# 5 seconds of generation.
PERSISTENCE_BUDGET_SECONDS = 5.0


# --------------------------------------------------------------------------- #
# Shared record builder
# --------------------------------------------------------------------------- #
def _record(
    document_id: str,
    *,
    order: int,
    vector: List[float],
    page: Optional[int] = None,
    heading: Optional[List[str]] = None,
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


# --------------------------------------------------------------------------- #
# A minimal transactional DB-API 2.0 fake (writes buffer until commit).
# --------------------------------------------------------------------------- #
class _FakeCursor:
    def __init__(self, conn: "_FakeConnection") -> None:
        self._conn = conn
        self.rowcount = 0
        self._result: List[Tuple[Any, ...]] = []

    def __enter__(self) -> "_FakeCursor":
        return self

    def __exit__(self, *exc: Any) -> None:
        return None

    def execute(self, sql: str, params: Optional[Tuple[Any, ...]] = None) -> None:
        params = params or ()
        self._conn.fail_next_if_configured(sql)
        table = self._conn.working_table()
        if sql.startswith("INSERT INTO"):
            row = dict(zip(_COLUMNS, params))
            table[row["chunk_id"]] = row
            self.rowcount = 1
        elif sql.startswith("DELETE FROM"):
            doc = params[0]
            victims = [cid for cid, r in table.items() if r["document_id"] == doc]
            for cid in victims:
                del table[cid]
            self.rowcount = len(victims)
        elif sql.startswith("SELECT chunk_id FROM"):
            doc = params[0]
            self._result = [
                (r["chunk_id"],)
                for r in self._conn.committed.values()
                if r["document_id"] == doc
            ]
        elif sql.startswith("SELECT"):
            doc = params[0]
            rows = [
                r for r in self._conn.committed.values() if r["document_id"] == doc
            ]
            rows.sort(key=lambda r: r["order_index"])
            self._result = [tuple(r[c] for c in _COLUMNS) for r in rows]
        else:  # pragma: no cover - defensive
            raise AssertionError(f"unexpected SQL: {sql}")

    def executemany(self, sql: str, seq: List[Tuple[Any, ...]]) -> None:
        for params in seq:
            self.execute(sql, params)

    def fetchall(self) -> List[Tuple[Any, ...]]:
        return list(self._result)


class _FakeConnection:
    def __init__(self) -> None:
        self.committed: Dict[str, Dict[str, Any]] = {}
        self._working: Optional[Dict[str, Dict[str, Any]]] = None
        self.commits = 0
        self.rollbacks = 0
        self._fail_pattern: Optional[str] = None
        self.closed = False

    def fail_on(self, sql_substring: str) -> None:
        self._fail_pattern = sql_substring

    def fail_next_if_configured(self, sql: str) -> None:
        if self._fail_pattern and self._fail_pattern in sql:
            self._fail_pattern = None
            raise RuntimeError("simulated driver error")

    def working_table(self) -> Dict[str, Dict[str, Any]]:
        if self._working is None:
            self._working = {cid: dict(r) for cid, r in self.committed.items()}
        return self._working

    def cursor(self) -> _FakeCursor:
        return _FakeCursor(self)

    def commit(self) -> None:
        self.commits += 1
        if self._working is not None:
            self.committed = self._working
            self._working = None

    def rollback(self) -> None:
        self.rollbacks += 1
        self._working = None

    def close(self) -> None:
        self.closed = True


def _pg_store(conn: _FakeConnection) -> PgVectorStore:
    return PgVectorStore(connection=conn, table_name="vector_records")


# --------------------------------------------------------------------------- #
# Persistence failure retains prior chunks and returns an error (Req 8.6)
# --------------------------------------------------------------------------- #
def test_in_memory_persistence_failure_retains_prior_and_errors() -> None:
    store = InMemoryVectorStore()
    prior = _record("doc-1", order=0, vector=[1.0, 0.0])
    store.upsert_batch("doc-1", [prior])

    # Make the next commit fail mid-write via the fault hook.
    store._commit_hook = lambda: (_ for _ in ()).throw(RuntimeError("disk full"))
    newrec = _record("doc-1", order=1, vector=[0.0, 1.0])

    with pytest.raises(PersistenceError):
        store.replace_document("doc-1", [newrec])

    # Req 8.6: the prior chunk survives the failed replacement unchanged.
    stored_ids = {rec.chunk.chunkId for rec in store.get_document("doc-1")}
    assert stored_ids == {prior.chunk.chunkId}


def test_in_memory_upsert_failure_retains_prior_and_errors() -> None:
    store = InMemoryVectorStore()
    prior = _record("doc-1", order=0, vector=[1.0, 0.0])
    store.upsert_batch("doc-1", [prior])

    store._commit_hook = lambda: (_ for _ in ()).throw(IOError("write error"))
    addition = _record("doc-1", order=1, vector=[0.5, 0.5])

    with pytest.raises(PersistenceError):
        store.upsert_batch("doc-1", [addition])

    # The failed batch added nothing; the prior chunk remains the only record.
    stored_ids = {rec.chunk.chunkId for rec in store.get_document("doc-1")}
    assert stored_ids == {prior.chunk.chunkId}


def test_pgvector_persistence_failure_rolls_back_and_retains_prior() -> None:
    conn = _FakeConnection()
    store = _pg_store(conn)
    prior = _record("doc-1", order=0, vector=[1.0, 0.0])
    store.upsert_batch("doc-1", [prior])

    # Fail on the INSERT half of the swap, after the DELETE has staged.
    conn.fail_on("INSERT INTO")
    newrec = _record("doc-1", order=1, vector=[0.0, 1.0])
    with pytest.raises(PersistenceError):
        store.replace_document("doc-1", [newrec])

    # Req 8.6: failure rolls back and the prior chunk is retained unchanged.
    assert conn.rollbacks >= 1
    stored_ids = {rec.chunk.chunkId for rec in store.get_document("doc-1")}
    assert stored_ids == {prior.chunk.chunkId}


# --------------------------------------------------------------------------- #
# Removal of an unknown id returns not-found (Req 8.8)
# --------------------------------------------------------------------------- #
def test_in_memory_delete_unknown_id_returns_not_found() -> None:
    store = InMemoryVectorStore()
    with pytest.raises(DocumentNotFoundError):
        store.delete_document("missing")


def test_in_memory_delete_after_delete_returns_not_found() -> None:
    # A second removal of an already-deleted id must also be not-found.
    store = InMemoryVectorStore()
    rec = _record("doc-1", order=0, vector=[1.0])
    store.upsert_batch("doc-1", [rec])
    store.delete_document("doc-1")
    with pytest.raises(DocumentNotFoundError):
        store.delete_document("doc-1")


def test_pgvector_delete_unknown_id_returns_not_found() -> None:
    conn = _FakeConnection()
    store = _pg_store(conn)
    with pytest.raises(DocumentNotFoundError):
        store.delete_document("missing")
    # The not-found signal rolls back; nothing is committed.
    assert conn.commits == 0


# --------------------------------------------------------------------------- #
# pgvector persistence latency (integration, Req 8.1)
# --------------------------------------------------------------------------- #
def test_pgvector_upsert_persists_within_budget_fake_connection() -> None:
    # Stand-in for a real DB: assert the adapter's full persist path (validate ->
    # transaction -> commit) completes well within the 5-second budget (Req 8.1).
    conn = _FakeConnection()
    store = _pg_store(conn)
    records = [_record("doc-1", order=i, vector=[float(i), 1.0]) for i in range(50)]

    start = time.perf_counter()
    result = store.upsert_batch("doc-1", records)
    elapsed = time.perf_counter() - start

    assert conn.commits == 1
    assert len(result.stored_chunk_ids) == len(records)
    assert elapsed < PERSISTENCE_BUDGET_SECONDS


@pytest.mark.skipif(
    not os.environ.get(DSN_ENV_VAR),
    reason=f"real-DB pgvector integration test requires {DSN_ENV_VAR} to be set",
)
def test_pgvector_upsert_persists_within_budget_real_db() -> None:
    # Real-database integration: only runs when a DSN is configured. Persistence
    # of a generated embedding must complete within 5 seconds (Req 8.1).
    if not pgvector_adapter.is_driver_available():  # pragma: no cover - env dependent
        pytest.skip("psycopg driver is not installed")

    store = pgvector_adapter.pgvector_store_from_env()
    document_id = "task-11-6-latency-probe"
    record = _record(document_id, order=0, vector=[0.1, 0.2, 0.3])

    try:
        start = time.perf_counter()
        store.upsert_batch(document_id, [record])
        elapsed = time.perf_counter() - start
        assert elapsed < PERSISTENCE_BUDGET_SECONDS
        stored = store.get_document(document_id)
        assert {r.chunk.chunkId for r in stored} == {record.chunk.chunkId}
    finally:
        # Best-effort cleanup so reruns stay clean; ignore if already gone.
        try:
            store.delete_document(document_id)
        except DocumentNotFoundError:  # pragma: no cover - cleanup race
            pass
