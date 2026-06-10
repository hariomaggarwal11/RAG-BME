"""Unit tests for resume rejection and job completion/failure (Task 13.8).

These focus on three terminal Orchestrator behaviours with deterministic
mock / in-memory adapters:

* ``resume`` is rejected with :class:`ResumeRejected` when the job has no
  recorded failing stage — both a never-run job and an already-completed job
  are not in a resumable state (Req 10.5).
* a normal ``run`` stores every chunk and marks the job ``COMPLETED``, with
  :attr:`JobCompleted.storedChunkIds` matching exactly the chunk ids of the
  records persisted to the vector store (Req 8.3).
* a storage failure (always-failing commit hook, ``stage_retry_limit=0`` so
  there is a single attempt) marks the job ``FAILED`` at the storage stage,
  reports every chunk id in :attr:`JobFailed.unstoredChunkIds`, and persists
  nothing (Req 8.7).
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


# -- resume rejection when not resumable (Req 10.5) -----------------------


def test_resume_rejected_for_never_run_job() -> None:
    config = _config()
    store = JobStateStore()
    job = _create_job(store)
    orch = _orchestrator(config, store, InMemoryVectorStore())

    # A freshly created job has never run, so no failing stage was recorded.
    outcome = orch.resume(job.jobId)

    assert isinstance(outcome, ResumeRejected)
    assert outcome.jobId == job.jobId
    assert "resumable" in outcome.reason

    # Resume must not have mutated the job into a running/terminal state.
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

    # Completion clears any failing stage, so the job is not resumable (Req 10.5).
    outcome = orch.resume(job.jobId)

    assert isinstance(outcome, ResumeRejected)
    assert outcome.jobId == job.jobId
    saved = store.get(job.jobId)
    assert saved.overallStatus is OverallStatus.COMPLETED


# -- completion on full storage (Req 8.3) ---------------------------------


def test_run_completes_with_stored_chunk_ids_matching_records() -> None:
    config = _config()
    store = JobStateStore()
    vector_store = InMemoryVectorStore()
    job = _create_job(store)
    orch = _orchestrator(config, store, vector_store)

    outcome = orch.run(job.jobId)

    assert isinstance(outcome, JobCompleted)
    assert outcome.jobId == job.jobId
    assert outcome.storedChunkIds, "expected at least one stored chunk id"

    # Job marked COMPLETED with no failing stage (Req 8.3).
    saved = store.get(job.jobId)
    assert saved.overallStatus is OverallStatus.COMPLETED
    assert saved.failingStage is None

    # The reported stored chunk ids match exactly the persisted records.
    stored_records = vector_store.get_document(job.documentId)
    assert {r.chunk.chunkId for r in stored_records} == set(outcome.storedChunkIds)
    # No duplicate ids reported.
    assert len(outcome.storedChunkIds) == len(set(outcome.storedChunkIds))
    assert len(outcome.storedChunkIds) == len(stored_records)


# -- storage failure reporting unstored chunk ids (Req 8.7) ---------------


def test_storage_failure_reports_all_unstored_chunk_ids() -> None:
    # stage_retry_limit=0 => a single storage attempt, which fails.
    config = _config(stage_retry_limit=0)
    store = JobStateStore()
    job = _create_job(store)

    def _always_fail() -> None:
        raise RuntimeError("disk full")

    vector_store = InMemoryVectorStore(commit_hook=_always_fail)
    orch = _orchestrator(config, store, vector_store)

    outcome = orch.run(job.jobId)

    assert isinstance(outcome, JobFailed)
    assert outcome.failingStage is Stage.STORAGE
    assert outcome.unstoredChunkIds, "expected unstored chunk ids reported"
    # No duplicates in the reported unstored ids.
    assert len(outcome.unstoredChunkIds) == len(set(outcome.unstoredChunkIds))

    saved = store.get(job.jobId)
    assert saved.overallStatus is OverallStatus.FAILED
    assert saved.failingStage is Stage.STORAGE
    assert saved.stageStates[Stage.STORAGE].status is StageStatus.FAILED
    # A single attempt was made (no retries) (Req 10.2 with limit 0).
    assert saved.stageStates[Stage.STORAGE].attempts == 1

    # Nothing was persisted: all-or-nothing storage (Req 8.7).
    assert vector_store.get_document(job.documentId) == []
