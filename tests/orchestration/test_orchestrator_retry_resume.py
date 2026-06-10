"""Smoke tests for Orchestrator retry, failure preservation, and resume (Task 13.2).

These drive a real Processing_Job through the five stages with deterministic
mock / in-memory adapters, injecting transient stage failures to assert:

* a failing stage is retried up to ``stage_retry_limit`` extra attempts and the
  job still completes when a retry succeeds (Req 10.2),
* exhausting retries marks the job failed, records the failing stage, and
  preserves the artifacts of every completed upstream stage (Req 10.3),
* ``resume`` re-enters at the recorded failing stage, reusing the preserved
  upstream artifacts (including the deserialized normalized document) rather
  than re-running completed stages (Req 10.4), and
* ``resume`` on a job with no recorded failing stage is rejected (Req 10.5).
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


# -- fault-injection helpers ----------------------------------------------


class ControllableCommit:
    """A storage commit hook whose failures can be toggled on/off.

    Raising from the in-memory store's commit hook simulates a mid-write
    persistence failure that is surfaced as a ``PersistenceError`` (Req 8.6),
    failing the storage stage. ``should_fail`` lets a test fail a bounded number
    of attempts and then let storage succeed.
    """

    def __init__(self, *, should_fail: bool = True) -> None:
        self.should_fail = should_fail
        self.calls = 0

    def __call__(self) -> None:
        self.calls += 1
        if self.should_fail:
            raise RuntimeError(f"transient storage failure #{self.calls}")


class FlakyCommit:
    """A storage commit hook that fails the first ``fail_times`` calls."""

    def __init__(self, fail_times: int) -> None:
        self.fail_times = fail_times
        self.calls = 0

    def __call__(self) -> None:
        self.calls += 1
        if self.calls <= self.fail_times:
            raise RuntimeError(f"transient storage failure #{self.calls}")


class FlakyChunker:
    """Wraps a real :class:`Chunker`, failing its first ``fail_times`` calls."""

    def __init__(self, inner: Chunker, fail_times: int) -> None:
        self._inner = inner
        self.fail_times = fail_times
        self.calls = 0

    def chunk(self, normalized, config):  # noqa: ANN001 - mirrors Chunker.chunk
        self.calls += 1
        if self.calls <= self.fail_times:
            raise RuntimeError(f"transient chunk failure #{self.calls}")
        return self._inner.chunk(normalized, config)


# -- builders -------------------------------------------------------------


def _config(stage_retry_limit: int = 1) -> PipelineConfig:
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


def _orchestrator(config, store, vector_store, *, chunker=None) -> Orchestrator:
    return Orchestrator(
        config,
        store,
        parser=_parser(config),
        normalizer=Normalizer(),
        chunker=chunker if chunker is not None else Chunker(),
        embedder=_embedder(),
        vector_store=vector_store,
        source_resolver=lambda job: SourceDocument(
            document_id=job.documentId, raw_bytes=_SOURCE_TEXT
        ),
    )


# -- retry then succeed (Req 10.2) ----------------------------------------


def test_retry_then_succeed_completes_job() -> None:
    config = _config(stage_retry_limit=2)
    store = JobStateStore()
    job = _create_job(store)
    # Storage fails once, then the first retry succeeds.
    commit = FlakyCommit(fail_times=1)
    vector_store = InMemoryVectorStore(commit_hook=commit)
    orch = _orchestrator(config, store, vector_store)

    outcome = orch.run(job.jobId)

    assert isinstance(outcome, JobCompleted)
    saved = store.get(job.jobId)
    assert saved.overallStatus is OverallStatus.COMPLETED
    assert saved.failingStage is None
    # Storage was attempted twice (initial attempt + one retry) (Req 10.2).
    assert saved.stageStates[Stage.STORAGE].attempts == 2
    assert saved.stageStates[Stage.STORAGE].status is StageStatus.SUCCEEDED
    # Every record was persisted.
    stored = vector_store.get_document(job.documentId)
    assert {r.chunk.chunkId for r in stored} == set(outcome.storedChunkIds)


def test_retry_attempts_do_not_exceed_limit_plus_one() -> None:
    limit = 3
    config = _config(stage_retry_limit=limit)
    store = JobStateStore()
    job = _create_job(store)
    vector_store = InMemoryVectorStore(commit_hook=ControllableCommit(should_fail=True))
    orch = _orchestrator(config, store, vector_store)

    outcome = orch.run(job.jobId)

    assert isinstance(outcome, JobFailed)
    saved = store.get(job.jobId)
    # Initial attempt + ``limit`` retries == limit + 1 (Req 10.2).
    assert saved.stageStates[Stage.STORAGE].attempts == limit + 1


# -- retry exhaustion preserves upstream artifacts (Req 10.3) -------------


def test_retry_exhaustion_fails_and_preserves_upstream_artifacts() -> None:
    config = _config(stage_retry_limit=1)
    store = JobStateStore()
    job = _create_job(store)
    vector_store = InMemoryVectorStore(commit_hook=ControllableCommit(should_fail=True))
    orch = _orchestrator(config, store, vector_store)

    outcome = orch.run(job.jobId)

    assert isinstance(outcome, JobFailed)
    assert outcome.failingStage is Stage.STORAGE
    assert outcome.unstoredChunkIds

    saved = store.get(job.jobId)
    assert saved.overallStatus is OverallStatus.FAILED
    # Failing stage recorded (Req 10.3).
    assert saved.failingStage is Stage.STORAGE
    assert saved.stageStates[Stage.STORAGE].status is StageStatus.FAILED
    # Every completed upstream stage succeeded and its artifact is preserved.
    for stage in (Stage.PARSING, Stage.NORMALIZATION, Stage.CHUNKING, Stage.EMBEDDING):
        state = saved.stageStates[stage]
        assert state.status is StageStatus.SUCCEEDED
        assert state.artifactRef is not None
        assert orch.artifacts.has(state.artifactRef)
    # Nothing was persisted (all-or-nothing storage).
    assert vector_store.get_document(job.documentId) == []


# -- successful resume reusing preserved upstream artifacts (Req 10.4) ----


def test_resume_from_storage_reuses_embedding_artifact() -> None:
    config = _config(stage_retry_limit=1)
    store = JobStateStore()
    job = _create_job(store)
    commit = ControllableCommit(should_fail=True)
    vector_store = InMemoryVectorStore(commit_hook=commit)
    orch = _orchestrator(config, store, vector_store)

    failed = orch.run(job.jobId)
    assert isinstance(failed, JobFailed)
    assert failed.failingStage is Stage.STORAGE

    transitions_before = len(orch.transitions(job.jobId))

    # Storage recovers; resume re-enters only at the failing stage (Req 10.4).
    commit.should_fail = False
    resumed = orch.resume(job.jobId)

    assert isinstance(resumed, JobCompleted)
    saved = store.get(job.jobId)
    assert saved.overallStatus is OverallStatus.COMPLETED
    assert saved.failingStage is None

    # Resume re-ran only the storage stage: every new transition is STORAGE.
    new_transitions = orch.transitions(job.jobId)[transitions_before:]
    assert new_transitions, "resume should record storage transitions"
    assert all(t.stage is Stage.STORAGE for t in new_transitions)

    stored = vector_store.get_document(job.documentId)
    assert {r.chunk.chunkId for r in stored} == set(resumed.storedChunkIds)


def test_resume_from_chunking_deserializes_preserved_normalized_doc() -> None:
    config = _config(stage_retry_limit=1)
    store = JobStateStore()
    job = _create_job(store)
    vector_store = InMemoryVectorStore()
    # Chunking fails on every attempt during the initial run, then recovers.
    flaky_chunker = FlakyChunker(Chunker(), fail_times=99)
    orch = _orchestrator(config, store, vector_store, chunker=flaky_chunker)

    failed = orch.run(job.jobId)
    assert isinstance(failed, JobFailed)
    assert failed.failingStage is Stage.CHUNKING

    saved = store.get(job.jobId)
    # Parsing + normalization artifacts preserved (normalization is serialized).
    assert saved.stageStates[Stage.NORMALIZATION].status is StageStatus.SUCCEEDED
    assert saved.stageStates[Stage.NORMALIZATION].artifactRef is not None

    transitions_before = len(orch.transitions(job.jobId))

    # Let chunking recover and resume: it must reuse the preserved (serialized)
    # normalized document, not re-run parsing/normalization (Req 10.4, 5.6).
    flaky_chunker.fail_times = 0
    resumed = orch.resume(job.jobId)

    assert isinstance(resumed, JobCompleted)
    saved = store.get(job.jobId)
    assert saved.overallStatus is OverallStatus.COMPLETED
    assert saved.failingStage is None

    # No parsing/normalization transitions were recorded during resume.
    new_transitions = orch.transitions(job.jobId)[transitions_before:]
    assert new_transitions
    assert all(
        t.stage in (Stage.CHUNKING, Stage.EMBEDDING, Stage.STORAGE)
        for t in new_transitions
    )
    stored = vector_store.get_document(job.documentId)
    assert {r.chunk.chunkId for r in stored} == set(resumed.storedChunkIds)


# -- resume rejection when not resumable (Req 10.5) -----------------------


def test_resume_rejected_when_no_failing_stage() -> None:
    config = _config()
    store = JobStateStore()
    job = _create_job(store)
    orch = _orchestrator(config, store, InMemoryVectorStore())

    # A freshly created, never-run job has no recorded failing stage.
    outcome = orch.resume(job.jobId)

    assert isinstance(outcome, ResumeRejected)
    assert outcome.jobId == job.jobId
    assert "resumable" in outcome.reason


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
