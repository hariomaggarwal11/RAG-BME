"""Unit tests for the pgvector adapter (Task 11.2).

These exercise the SQL and transaction logic of :class:`PgVectorStore` against a
DB-API 2.0 *fake* connection, with no real PostgreSQL required (a real-database
integration test is task 11.6). The fake buffers writes until ``commit`` and
discards them on ``rollback`` so the adapter's transactional/versioned swap
(Req 8.4) and persistence-failure retention (Req 8.6) can be asserted directly.

Covered behaviour:
* module imports cleanly without the psycopg driver and reports unavailability;
* ``upsert_batch`` persists/updates rows and validates batch ownership (Req 8.1);
* ``replace_document`` is an atomic swap and reports ``replaced`` correctly (8.4);
* a mid-write failure rolls back and retains prior rows (Req 8.6);
* ``delete_document`` removes all rows and errors on an unknown id (8.5, 8.8);
* ``query`` orders by descending similarity with bounded scores and filters.
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Tuple

import pytest

from biomed_rag.config import PipelineConfig
from biomed_rag.config import VectorStoreBackend as VectorStoreBackendChoice
from biomed_rag.models import Chunk, Embedding, VectorRecord
from biomed_rag.storage import (
    DocumentNotFoundError,
    PersistenceError,
    PgVectorDriverUnavailableError,
    PgVectorStore,
    VectorStore,
    VectorStoreError,
    default_registry,
)
from biomed_rag.storage import pgvector_adapter
from biomed_rag.storage.pgvector_adapter import (
    _COLUMNS,
    _distance_to_similarity,
    _format_vector,
    pgvector_store_from_env,
)


# --------------------------------------------------------------------------- #
# A DB-API 2.0 fake that buffers writes per-transaction.
# --------------------------------------------------------------------------- #
def _parse_vector_literal(literal: str) -> List[float]:
    text = literal.strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    if not text:
        return []
    return [float(p) for p in text.split(",")]


def _cosine_distance(a: List[float], b: List[float]) -> float:
    if len(a) != len(b) or not a:
        return 1.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 1.0
    return 1.0 - (dot / (na * nb))


class FakeCursor:
    def __init__(self, conn: "FakeConnection") -> None:
        self._conn = conn
        self.rowcount = 0
        self._result: List[Tuple[Any, ...]] = []
        self.executed: List[Tuple[str, Any]] = []

    def __enter__(self) -> "FakeCursor":
        return self

    def __exit__(self, *exc: Any) -> None:
        return None

    # -- helpers ----------------------------------------------------------
    def _table(self) -> Dict[str, Dict[str, Any]]:
        return self._conn._working_table()

    def execute(self, sql: str, params: Optional[Tuple[Any, ...]] = None) -> None:
        params = params or ()
        self.executed.append((sql, params))
        self._conn.fail_next_if_configured(sql)

        if sql.startswith("INSERT INTO"):
            self._apply_insert(params)
        elif sql.startswith("DELETE FROM"):
            self._apply_delete(params)
        elif "embedding <=>" in sql:
            self._apply_query(sql, params)
        elif sql.startswith("SELECT chunk_id FROM"):
            doc = params[0]
            self._result = [
                (row["chunk_id"],)
                for row in self._conn.committed.values()
                if row["document_id"] == doc
            ]
        elif sql.startswith("SELECT"):
            self._apply_select_document(params)
        else:  # pragma: no cover - defensive
            raise AssertionError(f"unexpected SQL: {sql}")

    def executemany(self, sql: str, seq: List[Tuple[Any, ...]]) -> None:
        for params in seq:
            self.execute(sql, params)

    def fetchall(self) -> List[Tuple[Any, ...]]:
        return list(self._result)

    # -- statement handlers ----------------------------------------------
    def _apply_insert(self, params: Tuple[Any, ...]) -> None:
        row = dict(zip(_COLUMNS, params))
        self._table()[row["chunk_id"]] = row
        self.rowcount = 1

    def _apply_delete(self, params: Tuple[Any, ...]) -> None:
        doc = params[0]
        table = self._table()
        victims = [cid for cid, row in table.items() if row["document_id"] == doc]
        for cid in victims:
            del table[cid]
        self.rowcount = len(victims)

    def _apply_select_document(self, params: Tuple[Any, ...]) -> None:
        doc = params[0]
        rows = [
            row
            for row in self._conn.committed.values()
            if row["document_id"] == doc
        ]
        rows.sort(key=lambda r: r["order_index"])
        self._result = [tuple(r[c] for c in _COLUMNS) for r in rows]

    def _apply_query(self, sql: str, params: Tuple[Any, ...]) -> None:
        query_vec = _parse_vector_literal(params[0])
        limit = params[-1]
        # Extract filter columns from the WHERE clause (in order) and zip with
        # the middle params. " WHERE FALSE" yields no rows.
        if " WHERE FALSE" in sql:
            self._result = []
            return
        where_cols = re.findall(r"(\w+) = %s", sql)
        filter_params = list(params[1:-1])
        filters = dict(zip(where_cols, filter_params))

        scored = []
        for row in self._conn.committed.values():
            if not all(str(row.get(c)) == str(v) for c, v in filters.items()):
                continue
            stored_vec = _parse_vector_literal(row["embedding"])
            distance = _cosine_distance(query_vec, stored_vec)
            scored.append((distance, row))
        scored.sort(key=lambda t: (t[0], t[1]["document_id"]))
        scored = scored[:limit]
        self._result = [
            tuple(r[c] for c in _COLUMNS) + (dist,) for dist, r in scored
        ]


class FakeConnection:
    """A transactional DB-API fake: writes buffer until commit."""

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

    def _working_table(self) -> Dict[str, Dict[str, Any]]:
        if self._working is None:
            self._working = {cid: dict(r) for cid, r in self.committed.items()}
        return self._working

    def cursor(self) -> FakeCursor:
        return FakeCursor(self)

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


# --------------------------------------------------------------------------- #
# Test helpers
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


def _store(conn: FakeConnection) -> PgVectorStore:
    return PgVectorStore(connection=conn, table_name="vector_records")


# --------------------------------------------------------------------------- #
# Construction / driver availability
# --------------------------------------------------------------------------- #
def test_implements_port_with_injected_connection() -> None:
    assert isinstance(_store(FakeConnection()), VectorStore)


def test_dsn_only_without_driver_raises_clear_error(monkeypatch) -> None:
    monkeypatch.setattr(pgvector_adapter, "is_driver_available", lambda: False)
    with pytest.raises(PgVectorDriverUnavailableError):
        PgVectorStore(dsn="postgresql://localhost/db")


def test_dsn_only_with_driver_constructs(monkeypatch) -> None:
    # Driver present -> construction succeeds without opening a connection.
    monkeypatch.setattr(pgvector_adapter, "is_driver_available", lambda: True)
    store = PgVectorStore(dsn="postgresql://localhost/db")
    assert isinstance(store, PgVectorStore)


def test_requires_a_connection_source() -> None:
    with pytest.raises(ValueError):
        PgVectorStore()


def test_invalid_table_name_rejected() -> None:
    with pytest.raises(ValueError):
        PgVectorStore(connection=FakeConnection(), table_name="bad name; DROP TABLE x")


# --------------------------------------------------------------------------- #
# Write / read path
# --------------------------------------------------------------------------- #
def test_upsert_then_retrieve_by_document_id() -> None:
    conn = FakeConnection()
    store = _store(conn)
    r1 = _record("doc-1", order=0, vector=[1.0, 0.0])
    r2 = _record("doc-1", order=1, vector=[0.0, 1.0])

    result = store.upsert_batch("doc-1", [r1, r2])

    assert result.replaced is False
    assert set(result.stored_chunk_ids) == {r1.chunk.chunkId, r2.chunk.chunkId}
    assert conn.commits == 1
    stored = store.get_document("doc-1")
    assert {rec.chunk.chunkId for rec in stored} == {r1.chunk.chunkId, r2.chunk.chunkId}
    # Round-trip preserves metadata.
    by_id = {rec.chunk.chunkId: rec for rec in stored}
    assert by_id[r1.chunk.chunkId].chunk.orderIndex == 0
    assert by_id[r1.chunk.chunkId].embedding.vector == [1.0, 0.0]


def test_upsert_updates_existing_chunk_in_place() -> None:
    conn = FakeConnection()
    store = _store(conn)
    rec = _record("doc-1", order=0, vector=[1.0, 0.0])
    store.upsert_batch("doc-1", [rec])

    # Re-store the same chunk id with new content/vector.
    updated_chunk = Chunk(
        documentId="doc-1",
        content="updated",
        tokenCount=2,
        orderIndex=0,
        chunkId=rec.chunk.chunkId,
    )
    updated = VectorRecord(
        documentId="doc-1",
        chunk=updated_chunk,
        embedding=Embedding(
            chunkId=rec.chunk.chunkId, vector=[0.0, 1.0], modelId="mock-model"
        ),
    )
    store.upsert_batch("doc-1", [updated])

    stored = store.get_document("doc-1")
    assert len(stored) == 1
    assert stored[0].chunk.content == "updated"
    assert stored[0].embedding.vector == [0.0, 1.0]


def test_upsert_rejects_record_for_other_document() -> None:
    conn = FakeConnection()
    store = _store(conn)
    foreign = _record("doc-2", order=0, vector=[1.0])
    with pytest.raises(PersistenceError):
        store.upsert_batch("doc-1", [foreign])
    # Validation happens before any transaction opens.
    assert conn.commits == 0
    assert store.get_document("doc-1") == []


def test_replace_document_is_atomic_swap() -> None:
    conn = FakeConnection()
    store = _store(conn)
    old = _record("doc-1", order=0, vector=[1.0, 0.0])
    store.upsert_batch("doc-1", [old])

    new1 = _record("doc-1", order=0, vector=[0.0, 1.0])
    new2 = _record("doc-1", order=1, vector=[1.0, 1.0])
    result = store.replace_document("doc-1", [new1, new2])

    assert result.replaced is True
    stored_ids = {rec.chunk.chunkId for rec in store.get_document("doc-1")}
    assert stored_ids == {new1.chunk.chunkId, new2.chunk.chunkId}
    assert old.chunk.chunkId not in stored_ids


def test_replace_on_empty_document_reports_not_replaced() -> None:
    conn = FakeConnection()
    store = _store(conn)
    rec = _record("doc-1", order=0, vector=[1.0])
    result = store.replace_document("doc-1", [rec])
    assert result.replaced is False


def test_persistence_failure_rolls_back_and_retains_prior_records() -> None:
    conn = FakeConnection()
    store = _store(conn)
    prior = _record("doc-1", order=0, vector=[1.0, 0.0])
    store.upsert_batch("doc-1", [prior])

    # Fail on the INSERT half of the replace, after the DELETE has staged.
    conn.fail_on("INSERT INTO")
    newrec = _record("doc-1", order=1, vector=[0.0, 1.0])
    with pytest.raises(PersistenceError):
        store.replace_document("doc-1", [newrec])

    assert conn.rollbacks >= 1
    # Req 8.6: the prior records survive the failed replacement unchanged.
    stored_ids = {rec.chunk.chunkId for rec in store.get_document("doc-1")}
    assert stored_ids == {prior.chunk.chunkId}


# --------------------------------------------------------------------------- #
# Delete path
# --------------------------------------------------------------------------- #
def test_delete_document_removes_all_records() -> None:
    conn = FakeConnection()
    store = _store(conn)
    r1 = _record("doc-1", order=0, vector=[1.0])
    r2 = _record("doc-1", order=1, vector=[2.0])
    store.upsert_batch("doc-1", [r1, r2])

    result = store.delete_document("doc-1")
    assert set(result.deleted_chunk_ids) == {r1.chunk.chunkId, r2.chunk.chunkId}
    assert store.get_document("doc-1") == []


def test_delete_unknown_document_raises_not_found() -> None:
    conn = FakeConnection()
    store = _store(conn)
    with pytest.raises(DocumentNotFoundError):
        store.delete_document("missing")
    # The not-found signal rolls back; nothing is committed.
    assert conn.commits == 0


# --------------------------------------------------------------------------- #
# Query path
# --------------------------------------------------------------------------- #
def test_query_orders_by_descending_similarity() -> None:
    conn = FakeConnection()
    store = _store(conn)
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
    conn = FakeConnection()
    store = _store(conn)
    for i in range(5):
        store.upsert_batch("doc-1", [_record("doc-1", order=i, vector=[float(i + 1), 0.0])])
    assert len(store.query([1.0, 0.0], top_k=3)) == 3


def test_query_non_positive_top_k_returns_empty() -> None:
    conn = FakeConnection()
    store = _store(conn)
    store.upsert_batch("doc-1", [_record("doc-1", order=0, vector=[1.0, 0.0])])
    assert store.query([1.0, 0.0], top_k=0) == []
    assert store.query([1.0, 0.0], top_k=-3) == []


def test_query_filter_by_document_id() -> None:
    conn = FakeConnection()
    store = _store(conn)
    a = _record("doc-a", order=0, vector=[1.0, 0.0])
    b = _record("doc-b", order=0, vector=[1.0, 0.0])
    store.upsert_batch("doc-a", [a])
    store.upsert_batch("doc-b", [b])

    results = store.query([1.0, 0.0], top_k=10, filter={"documentId": "doc-a"})
    assert [s.record.chunk.chunkId for s in results] == [a.chunk.chunkId]


def test_query_unknown_filter_key_matches_nothing() -> None:
    conn = FakeConnection()
    store = _store(conn)
    store.upsert_batch("doc-a", [_record("doc-a", order=0, vector=[1.0, 0.0])])
    assert store.query([1.0, 0.0], top_k=10, filter={"nope": "x"}) == []


# --------------------------------------------------------------------------- #
# SQL builders / helpers
# --------------------------------------------------------------------------- #
def test_upsert_sql_uses_on_conflict_and_vector_cast() -> None:
    store = _store(FakeConnection())
    sql = store._upsert_sql()
    assert "ON CONFLICT (chunk_id) DO UPDATE SET" in sql
    assert "%s::vector" in sql
    assert sql.startswith("INSERT INTO vector_records (")


def test_format_vector_literal() -> None:
    assert _parse_vector_literal(_format_vector([1.0, 2.5, -3.0])) == [1.0, 2.5, -3.0]


def test_distance_to_similarity_is_bounded_and_monotonic() -> None:
    assert _distance_to_similarity(0.0) == 1.0  # identical direction
    assert _distance_to_similarity(2.0) == 0.0  # opposite direction
    assert _distance_to_similarity(1.0) == 0.5
    # Clamped for floating-point drift outside [0, 2].
    assert _distance_to_similarity(-0.1) == 1.0
    assert _distance_to_similarity(2.5) == 0.0


# --------------------------------------------------------------------------- #
# Registry / env factory wiring
# --------------------------------------------------------------------------- #
def test_pgvector_registered_as_default_backend() -> None:
    assert default_registry.is_registered(VectorStoreBackendChoice.PGVECTOR)


def test_env_factory_requires_dsn(monkeypatch) -> None:
    monkeypatch.delenv(pgvector_adapter.DSN_ENV_VAR, raising=False)
    with pytest.raises(VectorStoreError):
        pgvector_store_from_env()


def test_env_factory_builds_from_dsn(monkeypatch) -> None:
    monkeypatch.setattr(pgvector_adapter, "is_driver_available", lambda: True)
    monkeypatch.setenv(pgvector_adapter.DSN_ENV_VAR, "postgresql://localhost/db")
    store = pgvector_store_from_env()
    assert isinstance(store, PgVectorStore)


def test_registry_select_for_pgvector_without_dsn_errors(monkeypatch) -> None:
    monkeypatch.delenv(pgvector_adapter.DSN_ENV_VAR, raising=False)
    config = PipelineConfig(vector_store_backend=VectorStoreBackendChoice.PGVECTOR)
    with pytest.raises(VectorStoreError):
        default_registry.select(config)
