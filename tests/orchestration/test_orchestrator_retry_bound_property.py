"""Property test for bounded retry attempts (Task 13.4, Req 10.2).

Feature: biomedical-rag-pipeline, Property 27: Retry attempts are bounded by the configured limit

Statement: for any stage that fails with configured retry limit L (in [0, 10]),
the number of execution attempts for that stage does not exceed L + 1.

The Orchestrator's ``_run_stage`` attempts a stage up to ``stage_retry_limit + 1``
times (the initial attempt plus the configured retries). We force a chosen stage
to fail on *every* attempt (storage via an always-failing commit hook, or
chunking via an always-failing chunker), sweep L across its full [0, 10] range,
and assert that:

* the job fails at exactly the forced stage,
* the failing stage's recorded ``attempts`` equals L + 1 (never more), and
* the number of RUNNING transitions recorded for that stage equals L + 1.
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
    OverallStatus,
    Stage,
    StageStatus,
)
from biomed_rag.normalization.normalizer import Normalizer
from biomed_rag.orchestration import JobFailed, Orchestrator
from biomed_rag.parsing import MockParsingEngine, Parser, ParsingEngineRegistry
from biomed_rag.parsing.raw_result import SourceDocument
from biomed_rag.storage.in_memory import InMemoryVectorStore

_EMBED_MODEL_ID = "mock-emb"
_EMBED_DIM = 64
_SOURCE_TEXT = b"Heart tissue analysis.\n\nThe ventricle contracts rhythmically."


# -- always-failing fault injectors ---------------------------------------


class AlwaysFailingCommit:
    """A storage commit hook that fails on every call (storage always fails)."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self) -> None:
        self.calls += 1
        raise RuntimeError(f"persistent storage failure #{self.calls}")


class AlwaysFailingChunker:
    """A chunker that raises on every call (chunking always fails)."""

    def __init__(self) -> None:
        self.calls = 0

    def chunk(self, normalized, config):  # noqa: ANN001 - mirrors Chunker.chunk
        self.calls += 1
        raise RuntimeError(f"persistent chunk failure #{self.calls}")


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


# -- property -------------------------------------------------------------


# Feature: biomedical-rag-pipeline, Property 27: Retry attempts are bounded by the configured limit
@settings(max_examples=200)
@given(
    limit=st.integers(min_value=0, max_value=10),
    failing_stage=st.sampled_from([Stage.STORAGE, Stage.CHUNKING]),
)
def test_retry_attempts_are_bounded_by_configured_limit(
    limit: int, failing_stage: Stage
) -> None:
    """Validates: Requirements 10.2"""
    config = _config(stage_retry_limit=limit)
    store = JobStateStore()
    job = _create_job(store)

    if failing_stage is Stage.STORAGE:
        vector_store = InMemoryVectorStore(commit_hook=AlwaysFailingCommit())
        orch = _orchestrator(config, store, vector_store)
    else:  # Stage.CHUNKING
        vector_store = InMemoryVectorStore()
        orch = _orchestrator(
            config, store, vector_store, chunker=AlwaysFailingChunker()
        )

    outcome = orch.run(job.jobId)

    # The forced stage fails on every attempt, so the job must fail there.
    assert isinstance(outcome, JobFailed)
    assert outcome.failingStage is failing_stage

    saved = store.get(job.jobId)
    assert saved.overallStatus is OverallStatus.FAILED
    assert saved.failingStage is failing_stage

    failing_state = saved.stageStates[failing_stage]
    assert failing_state.status is StageStatus.FAILED

    # Core bound: attempts == L + 1 (initial attempt + L retries) and, in
    # particular, never exceeds the configured limit plus one (Req 10.2).
    assert failing_state.attempts == limit + 1
    assert failing_state.attempts <= limit + 1

    # Each attempt records exactly one RUNNING transition for the stage, so the
    # observable RUNNING-transition count for the failing stage is also L + 1.
    running_transitions = [
        t
        for t in orch.transitions(job.jobId)
        if t.stage is failing_stage and t.status is StageStatus.RUNNING
    ]
    assert len(running_transitions) == limit + 1
