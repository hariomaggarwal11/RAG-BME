"""A deterministic in-memory :class:`VectorStore` adapter (Req 8.x).

This adapter keeps every :class:`VectorRecord` in process memory, indexed by
source ``documentId``. It implements the full port contract so stage logic and
property/unit tests can run fast without a real database (design Testing
Strategy: "Vector store properties run against an in-memory ``VectorStore``
adapter for fast 100+ iteration runs").

Behavioural guarantees mirror the port:

* ``upsert_batch`` is all-or-nothing (Req 8.1, 8.6).
* ``replace_document`` performs an atomic swap (Req 8.4).
* ``delete_document`` removes all records and errors on an unknown id (8.5, 8.8).
* a staging failure retains the prior records and raises ``PersistenceError`` (8.6).

A ``commit_hook`` constructor knob lets tests simulate a mid-write persistence
failure: it runs after new state is staged but before it is committed, so a
raised exception leaves the prior contents untouched.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Sequence

from biomed_rag.models import DocumentId, ScoredRecord, VectorRecord

from .port import (
    DeleteResult,
    DocumentNotFoundError,
    MetadataFilter,
    PersistenceError,
    StoreResult,
    VectorStore,
)


class InMemoryVectorStore(VectorStore):
    """In-memory vector store for fast property and unit tests."""

    def __init__(
        self,
        *,
        commit_hook: Optional[Callable[[], None]] = None,
    ) -> None:
        # documentId -> (chunkId -> VectorRecord). An inner dict per document
        # keeps records addressable by source identifier (Req 8.2) and keyed by
        # chunk id so upserts replace the matching chunk rather than duplicate it.
        self._by_document: Dict[DocumentId, Dict[str, VectorRecord]] = {}
        self._commit_hook = commit_hook

    # -- write path -------------------------------------------------------
    def upsert_batch(
        self,
        document_id: DocumentId,
        records: Sequence[VectorRecord],
    ) -> StoreResult:
        staged = self._stage(document_id, records)
        # Merge onto a copy of the existing records so a failure leaves the
        # live state untouched (all-or-nothing, Req 8.1/8.6).
        existing = dict(self._by_document.get(document_id, {}))
        existing.update(staged)
        self._commit(document_id, existing)
        return StoreResult(
            document_id=document_id,
            stored_chunk_ids=list(staged.keys()),
            replaced=False,
        )

    def replace_document(
        self,
        document_id: DocumentId,
        records: Sequence[VectorRecord],
    ) -> StoreResult:
        staged = self._stage(document_id, records)
        had_prior = document_id in self._by_document
        # Atomic swap: the staged set fully replaces the prior set, and only
        # after it is committed (Req 8.4). A failure in _commit leaves the prior
        # records in place.
        self._commit(document_id, staged)
        return StoreResult(
            document_id=document_id,
            stored_chunk_ids=list(staged.keys()),
            replaced=had_prior,
        )

    def _stage(
        self,
        document_id: DocumentId,
        records: Sequence[VectorRecord],
    ) -> Dict[str, VectorRecord]:
        """Validate ``records`` and build the staged chunkId -> record map.

        Validation failures raise :class:`PersistenceError` before any live
        state is touched, so the store retains its prior contents (Req 8.6).
        """
        staged: Dict[str, VectorRecord] = {}
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
            staged[record.chunk.chunkId] = record
        return staged

    def _commit(
        self,
        document_id: DocumentId,
        new_records: Dict[str, VectorRecord],
    ) -> None:
        """Run the optional fault hook, then atomically install ``new_records``.

        If the hook raises, nothing is installed and the prior contents are
        retained (Req 8.6). An empty ``new_records`` removes the document entry.
        """
        if self._commit_hook is not None:
            try:
                self._commit_hook()
            except Exception as exc:  # surface as a persistence failure (Req 8.6)
                raise PersistenceError(str(exc)) from exc
        if new_records:
            self._by_document[document_id] = new_records
        else:
            self._by_document.pop(document_id, None)

    # -- delete path ------------------------------------------------------
    def delete_document(self, document_id: DocumentId) -> DeleteResult:
        records = self._by_document.get(document_id)
        if not records:
            raise DocumentNotFoundError(document_id)
        deleted_ids = list(records.keys())
        del self._by_document[document_id]
        return DeleteResult(document_id=document_id, deleted_chunk_ids=deleted_ids)

    # -- read path --------------------------------------------------------
    def get_document(self, document_id: DocumentId) -> List[VectorRecord]:
        return list(self._by_document.get(document_id, {}).values())

    def query(
        self,
        embedding: Sequence[float],
        top_k: int,
        filter: Optional[MetadataFilter] = None,
    ) -> List[ScoredRecord]:
        query_vector = [float(v) for v in embedding]
        scored: List[ScoredRecord] = []
        for record in self._iter_records():
            if filter is not None and not self._matches_filter(record, filter):
                continue
            similarity = self._cosine_similarity(query_vector, record.embedding.vector)
            scored.append(ScoredRecord(record=record, similarity=similarity))
        # Descending similarity; ascending documentId tie-break for determinism
        # (mirrors the Retriever's ordering rule, Req 9.9).
        scored.sort(key=lambda s: (-s.similarity, s.record.documentId))
        if top_k < 0:
            top_k = 0
        return scored[:top_k]

    # -- internals --------------------------------------------------------
    def _iter_records(self):
        for records in self._by_document.values():
            yield from records.values()

    @staticmethod
    def _matches_filter(record: VectorRecord, filter: MetadataFilter) -> bool:
        for key, expected in filter.items():
            if key == "documentId":
                if record.documentId != expected:
                    return False
            elif key == "pageNumber":
                if record.chunk.pageNumber != expected:
                    return False
            elif key == "headingPath":
                if list(record.chunk.headingPath) != list(expected):  # type: ignore[arg-type]
                    return False
            else:
                # Unknown filter key matches nothing.
                return False
        return True

    @staticmethod
    def _cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
        """Cosine similarity mapped from [-1, 1] into the [0.0, 1.0] range.

        Vectors of differing dimension or a zero-magnitude vector yield 0.0.
        The [0, 1] mapping keeps results valid for :class:`ScoredRecord`.
        """
        if len(a) != len(b) or not a:
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        cosine = dot / (norm_a * norm_b)
        # Clamp for floating-point drift, then map [-1, 1] -> [0, 1].
        cosine = max(-1.0, min(1.0, cosine))
        return (cosine + 1.0) / 2.0
