"""A pgvector-backed :class:`VectorStore` adapter (Req 8.1, 8.4, 8.6).

This adapter implements the backend-neutral storage port against PostgreSQL +
the `pgvector <https://github.com/pgvector/pgvector>`_ extension. It mirrors the
behavioural contract of the in-memory adapter while persisting durably:

* ``upsert_batch`` is all-or-nothing inside a single transaction (Req 8.1, 8.6).
* ``replace_document`` performs an atomic, transactional swap — the prior rows
  for a document are deleted and the new rows inserted inside one transaction,
  so a failure rolls back and leaves the previously stored chunks untouched
  (Req 8.4). This is the "transactional/versioned swap" called for by the
  design.
* a persistence failure rolls the transaction back, retains the prior records,
  and raises :class:`~biomed_rag.storage.port.PersistenceError` (Req 8.6).
* the synchronous commit path keeps persistence well within the 5-second budget
  measured from embedding generation (Req 8.1).

Driver independence and testability
------------------------------------
``psycopg`` (and the ``pgvector`` Python helpers) are *optional* third-party
dependencies that may not be installed. To keep this module importable in any
environment:

* the driver is imported lazily, only when a real connection must be opened;
* the module never imports ``psycopg`` at module scope;
* the adapter accepts an injected connection or connection factory so the SQL
  and transaction logic can be unit-tested against a DB-API 2.0 fake without a
  running PostgreSQL (real-database integration coverage is task 11.6);
* constructing the adapter with only a DSN raises a clear
  :class:`PgVectorDriverUnavailableError` when the driver is missing.

The adapter speaks plain DB-API 2.0 (``cursor()``, ``execute``, ``executemany``,
``fetchall``, ``commit``, ``rollback``) and casts vectors with ``%s::vector`` so
it works whether or not the ``pgvector`` type adapters are registered.
"""

from __future__ import annotations

import json
import os
import re
from typing import (
    Any,
    Callable,
    List,
    Optional,
    Sequence,
    Tuple,
)

from biomed_rag.models import (
    Chunk,
    DocumentId,
    Embedding,
    EmbeddingStatus,
    ScoredRecord,
    VectorRecord,
)

from .port import (
    DeleteResult,
    DocumentNotFoundError,
    MetadataFilter,
    PersistenceError,
    StoreResult,
    VectorStore,
    VectorStoreError,
)

# A connection is any DB-API 2.0 connection (commit/rollback/cursor). A factory
# builds a fresh one on demand. Typed loosely so a test fake satisfies it.
Connection = Any
ConnectionFactory = Callable[[], Connection]

# Valid SQL identifier for the backing table. Interpolating an identifier into
# SQL text is unavoidable (parameters cannot name a table), so we constrain it
# strictly to an unqualified identifier to keep interpolation injection-safe.
_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# The ordered persisted columns. ``chunk_id`` is the primary key so an upsert
# replaces the matching chunk rather than duplicating it (mirrors in-memory).
_COLUMNS: Tuple[str, ...] = (
    "chunk_id",
    "document_id",
    "content",
    "token_count",
    "order_index",
    "overlap_token_count",
    "page_number",
    "heading_path",
    "is_table_part",
    "model_id",
    "status",
    "attempts",
    "embedding",
)


class PgVectorDriverUnavailableError(VectorStoreError):
    """Raised when the pgvector adapter needs the driver but it is not installed.

    The adapter imports ``psycopg`` lazily; this error surfaces at construction
    (when only a DSN is supplied) or when an operation must open a connection,
    with a clear message on how to make the driver available.
    """

    def __init__(self, detail: str = "") -> None:
        message = (
            "the pgvector adapter requires the 'psycopg' driver, which is not "
            "installed; install the optional dependency (e.g. "
            "`pip install \"psycopg[binary]\" pgvector`) or construct the "
            "adapter with an injected connection/connection_factory"
        )
        if detail:
            message = f"{message} ({detail})"
        super().__init__(message)


def is_driver_available() -> bool:
    """Return whether the optional ``psycopg`` driver can be imported."""
    try:  # pragma: no cover - import availability depends on environment
        import psycopg  # noqa: F401
    except Exception:
        return False
    return True


def _format_vector(vector: Sequence[float]) -> str:
    """Render an embedding vector as a pgvector literal, e.g. ``[0.1,0.2]``.

    Using the textual literal with a ``::vector`` cast keeps the adapter working
    whether or not the ``pgvector`` Python type adapters are registered.
    """
    return "[" + ",".join(repr(float(v)) for v in vector) + "]"


def _parse_vector(value: Any) -> List[float]:
    """Parse a stored embedding back into a list of floats.

    Accepts either a pgvector-adapted sequence or the textual ``[..]`` literal
    returned when type adapters are not registered.
    """
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("[") and text.endswith("]"):
            text = text[1:-1]
        if not text:
            return []
        return [float(part) for part in text.split(",")]
    # Already a sequence of numbers (pgvector adapter registered).
    return [float(v) for v in value]


class PgVectorStore(VectorStore):
    """A PostgreSQL + pgvector adapter implementing the :class:`VectorStore` port.

    Connection sourcing (one of):

    * ``connection`` — a single long-lived DB-API connection reused per call.
    * ``connection_factory`` — a zero-arg callable returning a fresh connection
      per operation (the connection is closed afterwards when it exposes
      ``close``).
    * ``dsn`` — a libpq connection string; the adapter lazily imports
      ``psycopg`` and opens a fresh connection per operation. Supplying only a
      DSN with the driver missing raises :class:`PgVectorDriverUnavailableError`
      at construction time.

    ``table_name`` names the backing table and must be a plain SQL identifier.
    """

    def __init__(
        self,
        *,
        dsn: Optional[str] = None,
        connection: Optional[Connection] = None,
        connection_factory: Optional[ConnectionFactory] = None,
        table_name: str = "vector_records",
    ) -> None:
        if not _IDENTIFIER_RE.match(table_name):
            raise ValueError(
                f"table_name {table_name!r} is not a valid SQL identifier "
                "(letters, digits, and underscores only; must not start with a digit)"
            )
        self._table = table_name

        provided = [
            name
            for name, value in (
                ("dsn", dsn),
                ("connection", connection),
                ("connection_factory", connection_factory),
            )
            if value is not None
        ]
        if not provided:
            raise ValueError(
                "PgVectorStore requires one of dsn, connection, or "
                "connection_factory to source a database connection"
            )

        self._connection = connection
        self._dsn = dsn

        if connection_factory is not None:
            self._connection_factory: Optional[ConnectionFactory] = connection_factory
        elif dsn is not None and connection is None:
            # Only a DSN was supplied: we must be able to open connections, which
            # needs the driver. Fail fast and clearly if it is missing (Req 8.x:
            # report unavailability at construction time).
            if not is_driver_available():
                raise PgVectorDriverUnavailableError("constructed with dsn only")
            self._connection_factory = self._open_with_driver
        else:
            self._connection_factory = None

    # -- connection management -------------------------------------------
    def _open_with_driver(self) -> Connection:
        """Open a fresh connection via the lazily-imported driver."""
        try:
            import psycopg
        except Exception as exc:  # pragma: no cover - environment dependent
            raise PgVectorDriverUnavailableError(str(exc)) from exc
        return psycopg.connect(self._dsn)

    def _acquire(self) -> Tuple[Connection, bool]:
        """Return ``(connection, owned)``.

        ``owned`` is True when the caller must close the connection after use
        (it came from a factory) and False when it is a shared, reused handle.
        """
        if self._connection is not None:
            return self._connection, False
        if self._connection_factory is not None:
            return self._connection_factory(), True
        # Should be unreachable given constructor validation.
        raise PgVectorDriverUnavailableError("no connection source configured")

    @staticmethod
    def _release(connection: Connection, owned: bool) -> None:
        if owned:
            close = getattr(connection, "close", None)
            if callable(close):
                close()

    # -- write path -------------------------------------------------------
    def upsert_batch(
        self,
        document_id: DocumentId,
        records: Sequence[VectorRecord],
    ) -> StoreResult:
        # Validate before touching the database so a bad batch never opens a
        # transaction (all-or-nothing, Req 8.1/8.6).
        rows = self._stage(document_id, records)
        chunk_ids = [row[0] for row in rows]

        def work(cursor: Any) -> None:
            if rows:
                cursor.executemany(self._upsert_sql(), rows)

        self._run_in_transaction(work)
        return StoreResult(
            document_id=document_id,
            stored_chunk_ids=chunk_ids,
            replaced=False,
        )

    def replace_document(
        self,
        document_id: DocumentId,
        records: Sequence[VectorRecord],
    ) -> StoreResult:
        rows = self._stage(document_id, records)
        chunk_ids = [row[0] for row in rows]
        had_prior_box = {"value": False}

        def work(cursor: Any) -> None:
            # Atomic swap: delete prior rows then insert the new set inside one
            # transaction. On any failure the transaction rolls back and the
            # prior rows survive (Req 8.4). rowcount tells us whether prior rows
            # existed so ``replaced`` is reported accurately.
            cursor.execute(self._delete_sql(), (document_id,))
            deleted = getattr(cursor, "rowcount", 0) or 0
            had_prior_box["value"] = deleted > 0
            if rows:
                cursor.executemany(self._insert_sql(), rows)

        self._run_in_transaction(work)
        return StoreResult(
            document_id=document_id,
            stored_chunk_ids=chunk_ids,
            replaced=had_prior_box["value"],
        )

    def delete_document(self, document_id: DocumentId) -> DeleteResult:
        deleted_box: dict = {"ids": None}

        def work(cursor: Any) -> None:
            cursor.execute(self._select_chunk_ids_sql(), (document_id,))
            ids = [row[0] for row in cursor.fetchall()]
            if not ids:
                # No rows for this id: signal not-found (Req 8.8). Raising inside
                # the transaction triggers a rollback; nothing was changed.
                raise DocumentNotFoundError(document_id)
            cursor.execute(self._delete_sql(), (document_id,))
            deleted_box["ids"] = ids

        self._run_in_transaction(work)
        return DeleteResult(
            document_id=document_id,
            deleted_chunk_ids=deleted_box["ids"] or [],
        )

    # -- read path --------------------------------------------------------
    def get_document(self, document_id: DocumentId) -> List[VectorRecord]:
        connection, owned = self._acquire()
        try:
            with connection.cursor() as cursor:
                cursor.execute(self._select_document_sql(), (document_id,))
                rows = cursor.fetchall()
            return [self._row_to_record(row) for row in rows]
        finally:
            self._release(connection, owned)

    def query(
        self,
        embedding: Sequence[float],
        top_k: int,
        filter: Optional[MetadataFilter] = None,
    ) -> List[ScoredRecord]:
        if top_k < 0:
            top_k = 0
        if top_k == 0:
            return []
        sql, params = self._query_sql(embedding, top_k, filter)
        connection, owned = self._acquire()
        try:
            with connection.cursor() as cursor:
                cursor.execute(sql, params)
                rows = cursor.fetchall()
        finally:
            self._release(connection, owned)

        results: List[ScoredRecord] = []
        for row in rows:
            # Each row is the persisted columns plus a trailing cosine distance.
            distance = float(row[len(_COLUMNS)])
            record = self._row_to_record(row)
            results.append(
                ScoredRecord(record=record, similarity=_distance_to_similarity(distance))
            )
        return results

    # -- transaction helper ----------------------------------------------
    def _run_in_transaction(self, work: Callable[[Any], None]) -> None:
        """Execute ``work(cursor)`` inside a single commit/rollback boundary.

        On success the transaction is committed. On any failure it is rolled
        back so previously stored chunks are retained (Req 8.4, 8.6). Driver
        errors are surfaced as :class:`PersistenceError`; the
        :class:`DocumentNotFoundError` control-flow signal is re-raised as-is
        after rollback.
        """
        connection, owned = self._acquire()
        try:
            try:
                with connection.cursor() as cursor:
                    work(cursor)
                connection.commit()
            except DocumentNotFoundError:
                _safe_rollback(connection)
                raise
            except PersistenceError:
                _safe_rollback(connection)
                raise
            except Exception as exc:
                _safe_rollback(connection)
                raise PersistenceError(
                    f"failed to persist to pgvector table {self._table!r}: {exc}"
                ) from exc
        finally:
            self._release(connection, owned)

    # -- staging / validation --------------------------------------------
    def _stage(
        self,
        document_id: DocumentId,
        records: Sequence[VectorRecord],
    ) -> List[Tuple[Any, ...]]:
        """Validate ``records`` and build insert/upsert parameter rows.

        Validation mirrors the in-memory adapter: every record must be a
        :class:`VectorRecord` belonging to ``document_id``. Failures raise
        :class:`PersistenceError` before any transaction opens, so the store
        retains its prior contents (Req 8.6). Later records with the same chunk
        id win, matching the dict-keyed in-memory semantics.
        """
        staged: dict = {}
        for record in records:
            if not isinstance(record, VectorRecord):
                raise PersistenceError(
                    f"expected VectorRecord, got {type(record).__name__}"
                )
            if record.documentId != document_id:
                raise PersistenceError(
                    f"record for chunk {record.chunk.chunkId!r} has document id "
                    f"{record.documentId!r}, expected {document_id!r}"
                )
            staged[record.chunk.chunkId] = self._record_to_params(record)
        return list(staged.values())

    @staticmethod
    def _record_to_params(record: VectorRecord) -> Tuple[Any, ...]:
        chunk = record.chunk
        embedding = record.embedding
        return (
            chunk.chunkId,
            record.documentId,
            chunk.content,
            chunk.tokenCount,
            chunk.orderIndex,
            chunk.overlapTokenCount,
            chunk.pageNumber,
            json.dumps(list(chunk.headingPath)),
            chunk.isTablePart,
            embedding.modelId,
            embedding.status.value,
            embedding.attempts,
            _format_vector(embedding.vector),
        )

    def _row_to_record(self, row: Sequence[Any]) -> VectorRecord:
        (
            chunk_id,
            document_id,
            content,
            token_count,
            order_index,
            overlap_token_count,
            page_number,
            heading_path,
            is_table_part,
            model_id,
            status,
            attempts,
            embedding,
        ) = row[: len(_COLUMNS)]

        chunk = Chunk(
            documentId=document_id,
            content=content,
            tokenCount=int(token_count),
            orderIndex=int(order_index),
            overlapTokenCount=int(overlap_token_count),
            pageNumber=None if page_number is None else int(page_number),
            headingPath=_load_heading_path(heading_path),
            isTablePart=bool(is_table_part),
            chunkId=chunk_id,
        )
        emb = Embedding(
            chunkId=chunk_id,
            vector=_parse_vector(embedding),
            modelId=model_id,
            status=EmbeddingStatus(status),
            attempts=int(attempts),
        )
        return VectorRecord(documentId=document_id, chunk=chunk, embedding=emb)

    # -- SQL builders -----------------------------------------------------
    def _insert_sql(self) -> str:
        columns = ", ".join(_COLUMNS)
        placeholders = ", ".join(
            "%s::vector" if col == "embedding" else "%s" for col in _COLUMNS
        )
        return f"INSERT INTO {self._table} ({columns}) VALUES ({placeholders})"

    def _upsert_sql(self) -> str:
        # Upsert keyed on chunk_id so re-storing a chunk updates it in place.
        updates = ", ".join(
            f"{col} = EXCLUDED.{col}" for col in _COLUMNS if col != "chunk_id"
        )
        return (
            f"{self._insert_sql()} "
            f"ON CONFLICT (chunk_id) DO UPDATE SET {updates}"
        )

    def _delete_sql(self) -> str:
        return f"DELETE FROM {self._table} WHERE document_id = %s"

    def _select_chunk_ids_sql(self) -> str:
        return f"SELECT chunk_id FROM {self._table} WHERE document_id = %s"

    def _select_document_sql(self) -> str:
        columns = ", ".join(_COLUMNS)
        return (
            f"SELECT {columns} FROM {self._table} "
            "WHERE document_id = %s ORDER BY order_index ASC"
        )

    def _query_sql(
        self,
        embedding: Sequence[float],
        top_k: int,
        filter: Optional[MetadataFilter],
    ) -> Tuple[str, Tuple[Any, ...]]:
        columns = ", ".join(_COLUMNS)
        vector_literal = _format_vector(embedding)
        params: List[Any] = [vector_literal]
        where_sql, where_params = self._build_filter(filter)
        params.extend(where_params)
        # The trailing distance column drives both ordering and the similarity
        # score. Cosine distance (<=>) ascending == similarity descending; the
        # document_id tie-break matches the in-memory adapter's determinism.
        sql = (
            f"SELECT {columns}, (embedding <=> %s::vector) AS distance "
            f"FROM {self._table}"
            f"{where_sql} "
            "ORDER BY distance ASC, document_id ASC "
            "LIMIT %s"
        )
        # The distance expression's vector param is the first placeholder; the
        # ORDER BY reuses the alias so only one vector param is needed.
        params.append(top_k)
        return sql, tuple(params)

    @staticmethod
    def _build_filter(
        filter: Optional[MetadataFilter],
    ) -> Tuple[str, List[Any]]:
        """Translate a metadata filter into a SQL WHERE clause and params.

        Supports the same fields as the in-memory adapter: ``documentId``,
        ``pageNumber``, ``headingPath``. An unknown key matches nothing, so the
        whole query is short-circuited to return no rows (``WHERE FALSE``).
        """
        if not filter:
            return "", []
        clauses: List[str] = []
        params: List[Any] = []
        for key, expected in filter.items():
            if key == "documentId":
                clauses.append("document_id = %s")
                params.append(expected)
            elif key == "pageNumber":
                clauses.append("page_number = %s")
                params.append(expected)
            elif key == "headingPath":
                clauses.append("heading_path = %s")
                params.append(json.dumps(list(expected)))  # type: ignore[arg-type]
            else:
                # Unknown filter key matches nothing.
                return " WHERE FALSE", []
        return " WHERE " + " AND ".join(clauses), params


# Environment variables read by the registry factory so the backend choice
# stays configuration-driven without threading a DSN through PipelineConfig.
DSN_ENV_VAR = "BIOMED_RAG_PGVECTOR_DSN"
TABLE_ENV_VAR = "BIOMED_RAG_PGVECTOR_TABLE"


def pgvector_store_from_env() -> "PgVectorStore":
    """Build a :class:`PgVectorStore` from environment configuration.

    The registry uses this zero-argument factory (Req 8.x: configuration-driven
    backend selection). The libpq DSN is read from :data:`DSN_ENV_VAR` and the
    optional table name from :data:`TABLE_ENV_VAR`. A missing DSN raises a clear
    error; a missing driver surfaces as :class:`PgVectorDriverUnavailableError`
    from the constructor.
    """
    dsn = os.environ.get(DSN_ENV_VAR)
    if not dsn:
        raise VectorStoreError(
            f"pgvector backend selected but {DSN_ENV_VAR} is not set; export a "
            "libpq connection string to use the pgvector vector store"
        )
    table_name = os.environ.get(TABLE_ENV_VAR, "vector_records")
    return PgVectorStore(dsn=dsn, table_name=table_name)


def _safe_rollback(connection: Connection) -> None:
    """Roll back ``connection``, swallowing any secondary rollback failure.

    The original error is the meaningful one to surface; a rollback that itself
    fails (e.g. a broken connection) must not mask it.
    """
    rollback = getattr(connection, "rollback", None)
    if callable(rollback):
        try:
            rollback()
        except Exception:  # pragma: no cover - secondary failure path
            pass


def _distance_to_similarity(distance: float) -> float:
    """Map a pgvector cosine distance in [0, 2] to a similarity in [0, 1].

    Cosine distance ``d = 1 - cos``; the in-memory adapter maps cosine into
    ``(cos + 1) / 2``. Substituting ``cos = 1 - d`` gives ``1 - d / 2``, so both
    adapters report scores on the same [0, 1] scale. Values are clamped for
    floating-point drift.
    """
    similarity = 1.0 - (distance / 2.0)
    if similarity < 0.0:
        return 0.0
    if similarity > 1.0:
        return 1.0
    return similarity


def _load_heading_path(value: Any) -> List[str]:
    """Decode the stored heading path (JSON text or a native list)."""
    if value is None:
        return []
    if isinstance(value, str):
        if not value:
            return []
        return [str(h) for h in json.loads(value)]
    return [str(h) for h in value]
