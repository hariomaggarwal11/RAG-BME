"""Unit tests for the in-memory JobStateStore (Task 4.1).

Covers the two Job-State-Store responsibilities from the design:
* unique job identifier allocation and lookup by id  - Requirement 1.1
* content-hash dedup index and lookup by hash         - Requirement 1.5
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from biomed_rag.ingestion import (
    DuplicateContentHashError,
    JobStateStore,
    UnknownJobError,
)
from biomed_rag.models import (
    DocumentMetadata,
    Format,
    OverallStatus,
    Stage,
    document_id_from_hash,
)


def _metadata(content_hash: str, filename: str = "paper.pdf") -> DocumentMetadata:
    return DocumentMetadata(
        filename=filename,
        format=Format.PDF,
        byteSize=1024,
        contentHash=content_hash,
        submittedAtUtc=datetime.now(timezone.utc),
    )


class TestJobCreation:
    def test_create_job_persists_and_initializes(self) -> None:
        store = JobStateStore()
        hash_a = "a" * 64
        job = store.create_job(
            document_id=document_id_from_hash(hash_a),
            metadata=_metadata(hash_a),
        )

        assert job.jobId
        assert job.documentId == hash_a
        assert job.currentStage is Stage.PARSING
        assert job.overallStatus is OverallStatus.QUEUED
        # Persisted and retrievable.
        assert store.get(job.jobId) is job
        assert len(store) == 1

    def test_generated_job_ids_are_unique(self) -> None:
        store = JobStateStore()
        ids = set()
        for i in range(500):
            h = f"{i:064d}"
            job = store.create_job(
                document_id=document_id_from_hash(h), metadata=_metadata(h)
            )
            ids.add(job.jobId)
        assert len(ids) == 500


class TestLookups:
    def test_find_by_id_and_content_hash(self) -> None:
        store = JobStateStore()
        h = "b" * 64
        job = store.create_job(document_id=document_id_from_hash(h), metadata=_metadata(h))

        assert store.find_by_id(job.jobId) is job
        assert store.find_by_content_hash(h) is job
        assert store.contains_content_hash(h)

    def test_missing_lookups_return_none(self) -> None:
        store = JobStateStore()
        assert store.find_by_id("nope") is None
        assert store.find_by_content_hash("c" * 64) is None
        assert not store.contains_content_hash("c" * 64)

    def test_get_unknown_raises(self) -> None:
        store = JobStateStore()
        with pytest.raises(UnknownJobError):
            store.get("missing")


class TestDedupIndex:
    def test_duplicate_content_hash_rejected_with_existing_id(self) -> None:
        store = JobStateStore()
        h = "d" * 64
        first = store.create_job(
            document_id=document_id_from_hash(h), metadata=_metadata(h)
        )
        with pytest.raises(DuplicateContentHashError) as exc:
            store.create_job(document_id=document_id_from_hash(h), metadata=_metadata(h))
        assert exc.value.existing_job_id == first.jobId
        # No second job created.
        assert len(store) == 1


class TestSave:
    def test_save_updates_in_place(self) -> None:
        store = JobStateStore()
        h = "e" * 64
        job = store.create_job(document_id=document_id_from_hash(h), metadata=_metadata(h))
        job.overallStatus = OverallStatus.RUNNING
        store.save(job)
        assert store.get(job.jobId).overallStatus is OverallStatus.RUNNING
        assert len(store) == 1
