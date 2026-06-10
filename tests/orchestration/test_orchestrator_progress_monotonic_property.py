"""Property test for bounded, monotonic progress (Task 13.7, Req 10.7).

Feature: biomedical-rag-pipeline, Property 30: Progress is bounded and monotonic

Statement: while a Processing_Job is running, the completion progress value is
an integer in [0, 100] and never decreases.

The Orchestrator advances ``job.progressPercent`` on every recorded transition
(``_record`` -> ``_save``), where each succeeded stage contributes an equal
share of 100. To observe the value at *every* persisted transition we wrap
``orch._save`` to snapshot ``job.progressPercent`` (the snapshot pattern from the
smoke test). We sweep the configured ``stage_retry_limit`` and inject transient
failures at various stages (which recover after a bounded number of attempts),
then drive the job through ``run`` and, when it fails, ``resume``. Across the
combined snapshot sequence we assert every value is an ``int`` in [0, 100] and
the sequence is non-decreasing.
"""

from __future__ import annotations

from datetime import datetime, timezone

from hypothesis import given, settings
from hypothesis import strategies as st

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
    Stage,
)
from biomed_rag.normalization.normalizer import Normalizer
from biomed_rag.orchestration import JobCompleted, JobFailed, Orchestrator
from biomed_rag.parsing import MockParsingEngine, Parser, ParsingEngineRegistry
from biomed_rag.parsing.raw_result import SourceDocument
from biomed_rag.storage.in_memory import InMemoryVectorStore

_EMBED_MODEL_ID = "mock-emb"
_EMBED_DIM = 64
_SOURCE_TEXT = b"Heart tissue analysis.\n\nThe ventricle contracts rhythmically."


# -- transient (recovering) fault injectors -------------------------------


class FlakyCommit:
    """A storage commit hook that fails its first ``fail_times`` calls."""

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


def _config(stage_retry_limit: int) -> PipelineConfig:
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


def _install_progress_snapshot(orch: Orchestrator, sink: list) -> None:
    """Wrap ``orch._save`` to snapshot ``progressPercent`` on every transition."""
    original_save = orch._save

    def _snapshotting_save(job):
        sink.append(job.progressPercent)
        return original_save(job)

    orch._save = _snapshotting_save  # type: ignore[assignment]


# -- property -------------------------------------------------------------


# Feature: biomedical-rag-pipeline, Property 30: Progress is bounded and monotonic
@settings(max_examples=200)
@given(
    stage_retry_limit=st.integers(min_value=0, max_value=6),
    failing_stage=st.sampled_from([Stage.CHUNKING, Stage.STORAGE]),
    # How many initial attempts the flaky component fails. When this exceeds the
    # retry budget the run fails and we exercise resume; otherwise it recovers
    # in-run. Covers both the in-run-retry and fail-then-resume code paths.
    fail_times=st.integers(min_value=0, max_value=8),
)
def test_progress_is_bounded_and_monotonic(
    stage_retry_limit: int, failing_stage: Stage, fail_times: int
) -> None:
    """Validates: Requirements 10.7"""
    config = _config(stage_retry_limit=stage_retry_limit)
    store = JobStateStore()
    job = _create_job(store)

    # Inject a transient failure at the chosen stage that recovers after
    # ``fail_times`` attempts.
    if failing_stage is Stage.STORAGE:
        commit = FlakyCommit(fail_times=fail_times)
        vector_store = InMemoryVectorStore(commit_hook=commit)
        flaky = commit
        orch = _orchestrator(config, store, vector_store)
    else:  # Stage.CHUNKING
        vector_store = InMemoryVectorStore()
        flaky = FlakyChunker(Chunker(), fail_times=fail_times)
        orch = _orchestrator(config, store, vector_store, chunker=flaky)

    snapshots: list[int] = []
    _install_progress_snapshot(orch, snapshots)

    outcome = orch.run(job.jobId)

    # If the transient failure outlasted the in-run retry budget the job fails;
    # let the component recover and resume from the recorded failing stage so we
    # also observe progress across resume.
    if isinstance(outcome, JobFailed):
        flaky.fail_times = 0
        orch.resume(job.jobId)
    else:
        assert isinstance(outcome, JobCompleted)

    # Progress was observed at every persisted transition across run (+ resume).
    assert snapshots, "expected progress snapshots"
    # Bounded integers in [0, 100] (Req 10.7).
    assert all(isinstance(p, int) for p in snapshots)
    assert all(0 <= p <= 100 for p in snapshots)
    # Never decreases as the job advances through stages (Req 10.7).
    assert snapshots == sorted(snapshots)
