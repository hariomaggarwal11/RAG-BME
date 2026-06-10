"""Vector storage and the pluggable Vector_Store port (Req 8).

This package defines the backend-neutral storage port and supporting shapes:

* :class:`VectorStore` — the pluggable port (Req 8.x).
* :class:`StoreResult` / :class:`DeleteResult` — port result shapes.
* :class:`VectorStoreError` / :class:`PersistenceError` /
  :class:`DocumentNotFoundError` — port error contract (Req 8.6, 8.8).
* :class:`InMemoryVectorStore` — a deterministic in-memory adapter for fast
  property and unit tests.
* :class:`PgVectorStore` — the PostgreSQL + pgvector adapter (Req 8.1, 8.4, 8.6),
  registered as the default backend for the ``PGVECTOR`` choice.
* :class:`VectorStoreRegistry` / :data:`default_registry` — config-driven
  backend selection.
"""

from __future__ import annotations

from biomed_rag.config import VectorStoreBackend as _VectorStoreBackend

from .in_memory import InMemoryVectorStore
from .pgvector_adapter import (
    DSN_ENV_VAR,
    TABLE_ENV_VAR,
    PgVectorDriverUnavailableError,
    PgVectorStore,
    is_driver_available,
    pgvector_store_from_env,
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
from .registry import (
    VectorStoreFactory,
    VectorStoreNotRegisteredError,
    VectorStoreRegistry,
    default_registry,
)

# Register the pgvector adapter as the backend for the PGVECTOR choice. The
# factory is lazy: it only reads connection config and imports the driver when a
# store is actually created, so importing this package never requires psycopg.
default_registry.register(
    _VectorStoreBackend.PGVECTOR,
    pgvector_store_from_env,
    replace=True,
)

__all__ = [
    # port
    "VectorStore",
    "MetadataFilter",
    # result shapes
    "StoreResult",
    "DeleteResult",
    # errors
    "VectorStoreError",
    "PersistenceError",
    "DocumentNotFoundError",
    "PgVectorDriverUnavailableError",
    # adapters
    "InMemoryVectorStore",
    "PgVectorStore",
    "pgvector_store_from_env",
    "is_driver_available",
    "DSN_ENV_VAR",
    "TABLE_ENV_VAR",
    # registry
    "VectorStoreRegistry",
    "VectorStoreNotRegisteredError",
    "VectorStoreFactory",
    "default_registry",
]
