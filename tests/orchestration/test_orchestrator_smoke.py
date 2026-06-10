"""Smoke tests for the Orchestrator sequential stage execution (Task 13.1).

These drive a real Processing_Job end-to-end through the five stages using the
deterministic mock / in-memory adapters, asserting:

* strictly sequential execution to completion (Req 10.1),
* every transition recorded as ``{stage, status, timestamp}`` (Req 10.6),
* current stage + bounded, monotonic integer progress exposed (Req 10.7),
* the job marked completed once every chunk is stored (Req 8.3), and
* the job marked failed reporting unstored chunk ids on a storage failure
  (Req 8.7).

Retry and resume (task 13.2) are out of scope here.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from biomed_rag.chunking.chunker import Chunker
from biomed_rag.config import ParsingEngine as ParsingEngineChoice
from biomed_rag.config import PipelineConfig
from biomed_rag.embedding.embedder import Embedder
from biomed_rag.embedding.mock import MockEmbeddingModel
from biomed_rag.embedding.registry import EmbeddingModelRegistry
from biomed_rag.ingestion.job_state_store import JobStateStore
from biomed_rag.models import (
    DocumentMetadata,
    Format,
    OverallStatus,
    Stage,
    StageStatus,
)
from biomed_rag.normalization.normalizer import Normalizer
from biomed_rag.orchestration import (
    JobCompleted,
    JobFailed,
    Orchestrator,
)
from biomed_rag.parsing import MockParsingEngine, Parser, ParsingEngineRegistry
from biomed_rag.parsing.raw_result import SourceDocument
from biomed_rag.storage.in_memory import InMemoryVectorStore

_EMBED_MODEL_ID = "mock-emb"
_EMBED_DIM = 64
_SOURCE_TEXT = b"Heart tissue analysis.\n\nThe ventricle contracts rhythmically."


# -- builders -------------------------------------------------------------


def _config() -> PipelineConfig:
    return PipelineConfig(
        parsing_engine=ParsingEngineChoice.DOCLING,
        embedding_model=_EMBED_MODEL_ID,
        embedding_dimension=_EMBED_DIM,
    )


def _parser(config: PipelineConfig) -> Parser:
    registry = ParsingEngineRegistry()
    registry.register(ParsingEngineChoice.DOCLING, lambda: MockParsingEngine(engine_id="docling"))
    return Parser(config=config, registry=registry)


def _embedder() -> Embedder:
    registry = EmbeddingModelRegistry()
    registry.register(
        _EMBED_MODEL_ID,
        lambda: MockEmbeddingModel(model_id=_EMBED_MODEL_ID, dimension=_EMBED_DIM),
    )
    return Embedder(registry)


def _create_job(store: JobStateStore, document_id: str = "doc-hash-1"):
    metadata = DocumentMetadata(
        filename="paper.pdf",
        format=Format.PDF,
        byteSize=len(_SOURCE_TEXT),
        contentHash=document_id,
        submittedAtUtc=datetime.now(timezone.utc),
    )
    return store.create_job(document_id=document_id, metadata=metadata)


def _orchestrator(config: PipelineConfig, store: JobStateStore, vector_store):
    return Orchestrator(
        config,
        store,
        parser=_parser(config),
        normalizer=Normalizer(),
        chunker=Chunker(),
        embedder=_embedder(),
        vector_store=vector_store,
        source_resolver=lambda job: SourceDocument(
            document_id=job.documentId, raw_bytes=_SOURCE_TEXT
        ),
    )


# -- completion path (Req 8.3, 10.1, 10.6, 10.7) --------------------------


def test_run_drives_job_to_completion() -> None:
    config = _config()
    store = JobStateStore()
    vector_store = InMemoryVectorStore()
    job = _create_job(store)
    orch = _orchestrator(config, store, vector_store)

    outcome = orch.run(job.jobId)

    assert isinstance(outcome, JobCompleted)
    assert outcome.jobId == job.jobId
    assert len(outcome.storedChunkIds) >= 1

    # Job marked completed (Req 8.3) and every stage succeeded.
    saved = store.get(job.jobId)
    assert saved.overallStatus is OverallStatus.COMPLETED
    assert saved.failingStage is None
    for stage in Stage.ordered():
        assert saved.stageStates[stage].status is StageStatus.SUCCEEDED
        assert saved.stageStates[stage].artifactRef is not None

    # Stored records are retrievable by document id (Req 8.2 via storage).
    stored = vector_store.get_document(job.documentId)
    assert {r.chunk.chunkId for r in stored} == set(outcome.storedChunkIds)


def test_status_reports_full_progress_after_completion() -> None:
    config = _config()
    store = JobStateStore()
    job = _create_job(store)
    orch = _orchestrator(config, store, InMemoryVectorStore())

    orch.run(job.jobId)
    status = orch.status(job.jobId)

    assert status.currentStage is Stage.STORAGE
    assert status.progressPercent == 100
    assert all(s is StageStatus.SUCCEEDED for s in status.stageStatuses.values())


def test_every_transition_is_recorded_in_order() -> None:
    config = _config()
    store = JobStateStore()
    job = _create_job(store)
    orch = _orchestrator(config, store, InMemoryVectorStore())

    orch.run(job.jobId)
    transitions = orch.transitions(job.jobId)

    # Each stage records RUNNING then SUCCEEDED, in stage order (Req 10.1, 10.6).
    expected = []
    for stage in Stage.ordered():
        expected.append((stage, StageStatus.RUNNING))
        expected.append((stage, StageStatus.SUCCEEDED))
    assert [(t.stage, t.status) for t in transitions] == expected
    assert all(isinstance(t.timestamp, datetime) for t in transitions)


def test_progress_is_bounded_and_monotonic() -> None:
    # A clock-free way to observe progress: capture progressPercent at each
    # transition by reading the saved job. Here we assert the recorded progress
    # sequence over the transition log is non-decreasing and within [0, 100].
    config = _config()
    store = JobStateStore()
    job = _create_job(store)

    progresses: list[int] = []

    orch = _orchestrator(config, store, InMemoryVectorStore())
    # Wrap _save to snapshot progress on every persisted transition.
    original_save = orch._save

    def _snapshooting_save(j):
        progresses.append(j.progressPercent)
        return original_save(j)

    orch._save = _snapshooting_save  # type: ignore[assignment]

    orch.run(job.jobId)

    assert progresses, "expected progress snapshots"
    assert all(0 <= p <= 100 for p in progresses)
    assert progresses == sorted(progresses)
    assert progresses[-1] == 100


# -- storage failure path (Req 8.7) ---------------------------------------


def test_storage_failure_marks_job_failed_with_unstored_chunk_ids() -> None:
    config = _config()
    store = JobStateStore()
    job = _create_job(store)

    # A commit hook that always raises simulates a mid-write persistence failure;
    # the in-memory store surfaces it as PersistenceError (Req 8.6).
    def _boom() -> None:
        raise RuntimeError("disk full")

    vector_store = InMemoryVectorStore(commit_hook=_boom)
    orch = _orchestrator(config, store, vector_store)

    outcome = orch.run(job.jobId)

    assert isinstance(outcome, JobFailed)
    assert outcome.failingStage is Stage.STORAGE
    assert outcome.unstoredChunkIds, "expected unstored chunk ids reported"

    saved = store.get(job.jobId)
    assert saved.overallStatus is OverallStatus.FAILED
    assert saved.failingStage is Stage.STORAGE
    assert saved.stageStates[Stage.STORAGE].status is StageStatus.FAILED
    # Upstream stages still succeeded and their outputs are preserved (Req 10.3).
    for stage in (Stage.PARSING, Stage.NORMALIZATION, Stage.CHUNKING, Stage.EMBEDDING):
        assert saved.stageStates[stage].status is StageStatus.SUCCEEDED
    # Nothing was persisted (all-or-nothing).
    assert vector_store.get_document(job.documentId) == []


def test_storage_failure_progress_never_exceeds_completed_stages() -> None:
    config = _config()
    store = JobStateStore()
    job = _create_job(store)
    vector_store = InMemoryVectorStore(commit_hook=lambda: (_ for _ in ()).throw(RuntimeError("x")))
    orch = _orchestrator(config, store, vector_store)

    orch.run(job.jobId)
    status = orch.status(job.jobId)

    # Four of five stages succeeded; storage failed → 80%.
    assert status.progressPercent == 80
    assert status.currentStage is Stage.STORAGE
