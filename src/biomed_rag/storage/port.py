"""The pluggable Vector_Store port (Req 8).

``VectorStore`` is the stable interface every storage backend implements.
Concrete adapters (the in-memory adapter here; the pgvector adapter in a later
task) sit behind this port so the Orchestrator, Embedder, and Retriever depend
only on the contract, never on a specific backend (design: ports-and-adapters,
Req 8.x "pluggable vector store").

This module defines the port, its result shapes, and the errors a backend may
raise. The job-state policy that consumes these results (mark job completed /
failed, report unstored chunk ids) lives in the Orchestrator, not here.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Mapping, Optional, Sequence

from biomed_rag.models import DocumentId, ScoredRecord, VectorRecord


class VectorStoreError(Exception):
    """Base class for all Vector_Store failures."""


class PersistenceError(VectorStoreError):
    """Raised when persisting embeddings/chunks fails (Req 8.6).

    When this is raised the store must have left any previously stored chunks
    for the document identifier intact (Req 8.6).
    """


class DocumentNotFoundError(VectorStoreError):
    """Raised when removal targets a document with no stored records (Req 8.8)."""

    def __init__(self, document_id: DocumentId) -> None:
        self.document_id = document_id
        super().__init__(
            f"no chunks or embeddings found for document id {document_id!r}"
        )


@dataclass(frozen=True)
class StoreResult:
    """Outcome of an ``upsert_batch`` / ``replace_document`` call.

    ``stored_chunk_ids`` lists the chunk ids now persisted for the document as a
    result of the call. ``replaced`` is True when prior records for the document
    were atomically swapped out (Req 8.4).
    """

    document_id: DocumentId
    stored_chunk_ids: List[str] = field(default_factory=list)
    replaced: bool = False


@dataclass(frozen=True)
class DeleteResult:
    """Outcome of a ``delete_document`` call (Req 8.5).

    ``deleted_chunk_ids`` lists every chunk id removed for the document.
    """

    document_id: DocumentId
    deleted_chunk_ids: List[str] = field(default_factory=list)


# A metadata filter is a mapping of field name to required value. Supported
# fields are the source-metadata fields carried by a stored record:
# ``documentId``, ``pageNumber``, and ``headingPath`` (design: Retriever filter).
MetadataFilter = Mapping[str, object]


class VectorStore(ABC):
    """Port for a pluggable vector storage backend (Req 8.x).

    Implementations persist :class:`VectorRecord` units addressable by their
    source ``documentId`` (Req 8.2) and answer similarity ``query`` requests.
    Reprocess replacement is atomic (Req 8.4); removal of an unknown id is an
    error (Req 8.8); a persistence failure retains prior records (Req 8.6).
    """

    @abstractmethod
    def upsert_batch(
        self,
        document_id: DocumentId,
        records: Sequence[VectorRecord],
    ) -> StoreResult:
        """Insert or update ``records`` for ``document_id``.

        Every record must belong to ``document_id``. The call is all-or-nothing:
        on a persistence failure no record from the batch is stored and any
        previously stored records for the document are retained, and
        :class:`PersistenceError` is raised (Req 8.1, 8.6).
        """
        raise NotImplementedError

    @abstractmethod
    def replace_document(
        self,
        document_id: DocumentId,
        records: Sequence[VectorRecord],
    ) -> StoreResult:
        """Atomically swap the stored records for ``document_id`` (Req 8.4).

        The previously stored chunks are replaced by ``records`` only after the
        new records are successfully staged; if staging fails the prior records
        are retained unchanged and :class:`PersistenceError` is raised.
        """
        raise NotImplementedError

    @abstractmethod
    def delete_document(self, document_id: DocumentId) -> DeleteResult:
        """Remove all chunks/embeddings for ``document_id`` (Req 8.5).

        Raises:
            DocumentNotFoundError: ``document_id`` has no stored records (Req 8.8).
        """
        raise NotImplementedError

    @abstractmethod
    def query(
        self,
        embedding: Sequence[float],
        top_k: int,
        filter: Optional[MetadataFilter] = None,
    ) -> List[ScoredRecord]:
        """Return up to ``top_k`` records most similar to ``embedding``.

        Results are ordered by descending similarity. An optional ``filter``
        restricts candidates to records whose metadata matches every entry
        (design: Retriever metadata filter).
        """
        raise NotImplementedError

    @abstractmethod
    def get_document(self, document_id: DocumentId) -> List[VectorRecord]:
        """Return exactly the records stored under ``document_id`` (Req 8.2).

        Returns an empty list when nothing is stored for the identifier. This is
        the retrievability-by-identifier accessor underpinning Req 8.2.
        """
        raise NotImplementedError
