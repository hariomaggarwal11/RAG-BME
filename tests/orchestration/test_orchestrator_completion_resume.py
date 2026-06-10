"""Unit tests for non-resumable resume and job completion/failure (Task 13.8).

These focused unit tests pin down three terminal behaviours of the
:class:`Orchestrator`, each driven through the five stages with the
deterministic mock / in-memory adapters:

* ``resume`` is rejected for a job with no recorded failing stage — both a
  never-run job and an already-completed job (Req 10.5);
* a normal ``run`` stores every chunk, marks the job ``COMPLETED``, and reports
  ``JobCompleted.storedChunkIds`` that exactly match the persisted records
  (Req 8.3); and
* a storage failure (an always-raising commit hook with ``stage_retry_limit=0``
  so there are no retries) marks the job ``FAILED`` at the storage stage,
  reports the unstored chunk ids, and persists nothing (Req 8.7).

The builders mirror the existing orchestration smoke tests so the pipeline is
assembled identically; only the assertions differ in focus.
"""

from __future__ import annotations

from datetime import datetime, timezone

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
    ResumeRejected,
)
from biomed_rag.parsing import MockParsingEngine, Parser, ParsingEngineRegistry
from biomed_rag.parsing.raw_result import SourceDocument
from biomed_rag.storage.in_memory import InMemoryVectorStore

_EMBED_MODEL_ID = "mock-emb"
_EMBED_DIM = 64
_SOURCE_TEXT = b"Heart tissue analysis.\n\nThe ventricle contracts rhythmically."


# -- builders -------------------------------------------------------------


def _config(stage_retry_limit: int = 0) -> PipelineConfig:
    return PipelineConfig(
        parsing_engine=ParsingEngineChoice.DOCLING,
        embedding_model=_EMBED_MODEL_ID,
        embedding_dimension=_EMBED_DIM,
        stage_retry_limit=stage_retry_limit,
    )


def _parser(config: PipelineConfig) -> Parser:
    registry = ParsingEngineRegistry()
    registry.register(
        ParsingEngineChoice.DOCLING, lambda: MockParsingEngine(engine_id="docling")
    )
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


# -- resume rejected when not resumable (Req 10.5) ------------------------


def test_resume_rejected_for_never_run_job() -> None:
    config = _config()
    store = JobStateStore()
    job = _create_job(store)
    orch = _orchestrator(config, store, InMemoryVectorStore())

    # A freshly created job has never failed, so it has no recorded failing
    # stage and is not in a resumable state (Req 10.5).
    outcome = orch.resume(job.jobId)

    assert isinstance(outcome, ResumeRejected)
    assert outcome.jobId == job.jobId
    assert "resumable" in outcome.reason

    # The rejected request must not mutate the job into a running/failed state.
    saved = store.get(job.jobId)
    assert saved.failingStage is None
    assert saved.overallStatus is not OverallStatus.RUNNING


def test_resume_rejected_for_completed_job() -> None:
    config = _config()
    store = JobStateStore()
    job = _create_job(store)
    orch = _orchestrator(config, store, InMemoryVectorStore())

    completed = orch.run(job.jobId)
    assert isinstance(completed, JobCompleted)

    # A completed job cleared its failing stage, so resume is rejected (Req 10.5).
    outcome = orch.resume(job.jobId)

    assert isinstance(outcome, ResumeRejected)
    assert outcome.jobId == job.jobId
    assert "resumable" in outcome.reason

    # Resume rejection leaves the completed job untouched.
    saved = store.get(job.jobId)
    assert saved.overallStatus is OverallStatus.COMPLETED
    assert saved.failingStage is None


# -- job completion on full storage (Req 8.3) -----------------------------


def test_run_marks_job_completed_with_stored_chunk_ids_matching_records() -> None:
    config = _config()
    store = JobStateStore()
    vector_store = InMemoryVectorStore()
    job = _create_job(store)
    orch = _orchestrator(config, store, vector_store)

    outcome = orch.run(job.jobId)

    # Every chunk stored → JobCompleted reporting the stored chunk ids (Req 8.3).
    assert isinstance(outcome, JobCompleted)
    assert outcome.jobId == job.jobId
    assert len(outcome.storedChunkIds) >= 1
    # No duplicate ids reported.
    assert len(outcome.storedChunkIds) == len(set(outcome.storedChunkIds))

    # Job marked completed with no failing stage and every stage succeeded.
    saved = store.get(job.jobId)
    assert saved.overallStatus is OverallStatus.COMPLETED
    assert saved.failingStage is None
    for stage in Stage.ordered():
        assert saved.stageStates[stage].status is StageStatus.SUCCEEDED

    # The reported ids exactly match the records actually persisted (Req 8.3).
    stored = vector_store.get_document(job.documentId)
    assert {r.chunk.chunkId for r in stored} == set(outcome.storedChunkIds)
    assert len(stored) == len(outcome.storedChunkIds)


# -- job failure reports unstored chunk ids (Req 8.7) ---------------------


def test_storage_failure_marks_failed_and_reports_unstored_chunk_ids() -> None:
    # stage_retry_limit=0 → a single attempt with no retries, so an always-
    # failing commit hook fails the storage stage immediately (Req 8.7).
    config = _config(stage_retry_limit=0)
    store = JobStateStore()
    job = _create_job(store)

    def _always_fail() -> None:
        raise RuntimeError("disk full")

    vector_store = InMemoryVectorStore(commit_hook=_always_fail)
    orch = _orchestrator(config, store, vector_store)

    outcome = orch.run(job.jobId)

    # Storage failure → JobFailed at STORAGE reporting unstored chunk ids (Req 8.7).
    assert isinstance(outcome, JobFailed)
    assert outcome.jobId == job.jobId
    assert outcome.failingStage is Stage.STORAGE
    assert outcome.unstoredChunkIds, "expected unstored chunk ids to be reported"

    # The job is marked failed at the storage stage, with only a single attempt
    # since retries are disabled (stage_retry_limit=0).
    saved = store.get(job.jobId)
    assert saved.overallStatus is OverallStatus.FAILED
    assert saved.failingStage is Stage.STORAGE
    assert saved.stageStates[Stage.STORAGE].status is StageStatus.FAILED
    assert saved.stageStates[Stage.STORAGE].attempts == 1

    # The reported unstored ids are exactly the chunks the embedding stage
    # produced — an all-or-nothing failure stores none of them (Req 8.7).
    embedded_ids = {
        r.chunk.chunkId
        for r in orch.artifacts.get(
            saved.stageStates[Stage.EMBEDDING].artifactRef
        )
    }
    assert set(outcome.unstoredChunkIds) == embedded_ids

    # Nothing was persisted to the knowledge library (all-or-nothing storage).
    assert vector_store.get_document(job.documentId) == []
