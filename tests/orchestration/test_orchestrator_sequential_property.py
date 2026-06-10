"""Property test for strictly sequential stage execution (Task 13.3).

Feature: biomedical-rag-pipeline, Property 26: Stage execution is strictly sequential

Property 26 (Validates Requirements 10.1): for any Processing_Job, the recorded
stage transitions show that each stage begins only after its immediately
preceding stage has reached SUCCEEDED, in the order

    parsing -> normalization -> chunking -> embedding -> storage.

The test varies ``stage_retry_limit`` and injects *transient* (recoverable)
failures at arbitrary stages -- each stage is made to fail no more often than
the retry limit allows, so the job always runs to completion. The injected
failures interleave RUNNING/FAILED transitions, which makes the ordering
guarantee non-trivial: regardless of how many times a stage is retried, the
*first* RUNNING transition of every stage must still follow the SUCCEEDED
transition of its immediate predecessor, and the stages must reach SUCCEEDED in
the canonical order.
"""

from __future__ import annotations

from datetime import datetime, timezone

from hypothesis import HealthCheck, given, settings
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
from biomed_rag.orchestration import JobCompleted, Orchestrator
from biomed_rag.parsing import MockParsingEngine, Parser, ParsingEngineRegistry
from biomed_rag.parsing.raw_result import SourceDocument
from biomed_rag.storage.in_memory import InMemoryVectorStore

_EMBED_MODEL_ID = "mock-emb"
_EMBED_DIM = 64
_SOURCE_TEXT = b"Heart tissue analysis.\n\nThe ventricle contracts rhythmically."


# -- transient fault injectors -------------------------------------------
#
# Each wrapper fails the *stage-level* operation on its first ``fail_times``
# attempts and then delegates to the real component. Because the Orchestrator
# retries a failing stage as a whole, and each stage's key operation is invoked
# at least once per attempt (parse/normalize/chunk once; embed once per chunk,
# but the first chunk's failure aborts the attempt before later chunks), a
# single failure counter per wrapper corresponds to one failed stage attempt.


class _FlakyParser:
    def __init__(self, inner: Parser, fail_times: int) -> None:
        self._inner = inner
        self._fail_times = fail_times
        self._failures = 0

    def parse(self, job, source):  # noqa: ANN001 - mirrors Parser.parse
        if self._failures < self._fail_times:
            self._failures += 1
            raise RuntimeError("transient parse failure")
        return self._inner.parse(job, source)


class _FlakyNormalizer:
    def __init__(self, inner: Normalizer, fail_times: int) -> None:
        self._inner = inner
        self._fail_times = fail_times
        self._failures = 0

    def normalize(self, parsed):  # noqa: ANN001 - mirrors Normalizer.normalize
        if self._failures < self._fail_times:
            self._failures += 1
            raise RuntimeError("transient normalize failure")
        return self._inner.normalize(parsed)

    def serialize(self, document):  # noqa: ANN001
        return self._inner.serialize(document)

    def deserialize(self, data):  # noqa: ANN001
        return self._inner.deserialize(data)


class _FlakyChunker:
    def __init__(self, inner: Chunker, fail_times: int) -> None:
        self._inner = inner
        self._fail_times = fail_times
        self._failures = 0

    def chunk(self, normalized, config):  # noqa: ANN001 - mirrors Chunker.chunk
        if self._failures < self._fail_times:
            self._failures += 1
            raise RuntimeError("transient chunk failure")
        return self._inner.chunk(normalized, config)


class _FlakyEmbedder:
    def __init__(self, inner: Embedder, fail_times: int) -> None:
        self._inner = inner
        self._fail_times = fail_times
        self._failures = 0

    def embed(self, chunk, config):  # noqa: ANN001 - mirrors Embedder.embed
        # Failing on the first chunk of a stage attempt aborts that attempt
        # before later chunks are reached, so one increment == one failed attempt.
        if self._failures < self._fail_times:
            self._failures += 1
            raise RuntimeError("transient embed failure")
        return self._inner.embed(chunk, config)


class _FlakyCommit:
    """Storage commit hook that fails its first ``fail_times`` invocations."""

    def __init__(self, fail_times: int) -> None:
        self._fail_times = fail_times
        self._calls = 0

    def __call__(self) -> None:
        self._calls += 1
        if self._calls <= self._fail_times:
            raise RuntimeError("transient storage failure")


# -- builders -------------------------------------------------------------


def _config(stage_retry_limit: int) -> PipelineConfig:
    return PipelineConfig(
        parsing_engine=ParsingEngineChoice.DOCLING,
        embedding_model=_EMBED_MODEL_ID,
        embedding_dimension=_EMBED_DIM,
        stage_retry_limit=stage_retry_limit,
    )


def _base_parser(config: PipelineConfig) -> Parser:
    registry = ParsingEngineRegistry()
    registry.register(
        ParsingEngineChoice.DOCLING, lambda: MockParsingEngine(engine_id="docling")
    )
    return Parser(config=config, registry=registry)


def _base_embedder() -> Embedder:
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


# -- property -------------------------------------------------------------

# stage_retry_limit is bounded by config to [0, 10]; keep it small for speed
# while still exercising multi-retry interleaving.
_RETRY_LIMIT = st.integers(min_value=0, max_value=4)


@settings(max_examples=150, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    stage_retry_limit=_RETRY_LIMIT,
    data=st.data(),
)
def test_stage_execution_is_strictly_sequential(stage_retry_limit, data) -> None:
    # Feature: biomedical-rag-pipeline, Property 26: Stage execution is strictly sequential
    # For every stage, inject 0..stage_retry_limit transient failures so the
    # stage always recovers within its retry budget and the job completes.
    fails = data.draw(
        st.fixed_dictionaries(
            {
                stage: st.integers(min_value=0, max_value=stage_retry_limit)
                for stage in Stage.ordered()
            }
        ),
        label="transient_failures_per_stage",
    )

    config = _config(stage_retry_limit)
    store = JobStateStore()
    job = _create_job(store)

    parser = _FlakyParser(_base_parser(config), fails[Stage.PARSING])
    normalizer = _FlakyNormalizer(Normalizer(), fails[Stage.NORMALIZATION])
    chunker = _FlakyChunker(Chunker(), fails[Stage.CHUNKING])
    embedder = _FlakyEmbedder(_base_embedder(), fails[Stage.EMBEDDING])
    vector_store = InMemoryVectorStore(commit_hook=_FlakyCommit(fails[Stage.STORAGE]))

    orch = Orchestrator(
        config,
        store,
        parser=parser,
        normalizer=normalizer,
        chunker=chunker,
        embedder=embedder,
        vector_store=vector_store,
        source_resolver=lambda j: SourceDocument(
            document_id=j.documentId, raw_bytes=_SOURCE_TEXT
        ),
    )

    outcome = orch.run(job.jobId)

    # With every stage's failures bounded by the retry limit, the job completes.
    assert isinstance(outcome, JobCompleted)
    assert store.get(job.jobId).overallStatus is OverallStatus.COMPLETED

    transitions = orch.transitions(job.jobId)
    ordered_stages = Stage.ordered()

    # (a) Stages reach SUCCEEDED in the canonical order. Each stage succeeds
    # exactly once, and the sequence of SUCCEEDED transitions equals the
    # canonical stage order.
    succeeded_sequence = [
        t.stage for t in transitions if t.status is StageStatus.SUCCEEDED
    ]
    assert succeeded_sequence == list(ordered_stages)

    # (b) Each stage's FIRST RUNNING transition occurs only after the SUCCEEDED
    # transition of its immediately preceding stage (Req 10.1). The first stage
    # (parsing) has no predecessor.
    first_running_index = {}
    succeeded_index = {}
    for idx, t in enumerate(transitions):
        if t.status is StageStatus.RUNNING and t.stage not in first_running_index:
            first_running_index[t.stage] = idx
        if t.status is StageStatus.SUCCEEDED:
            succeeded_index[t.stage] = idx

    for stage in ordered_stages:
        assert stage in first_running_index, f"{stage} never started"
        if stage.order == 0:
            continue
        prior = ordered_stages[stage.order - 1]
        assert succeeded_index[prior] < first_running_index[stage], (
            f"{stage.name} started before {prior.name} succeeded"
        )

    # (c) Timestamps are non-decreasing across the recorded transitions, a
    # corollary of strictly sequential execution.
    timestamps = [t.timestamp for t in transitions]
    assert timestamps == sorted(timestamps)
