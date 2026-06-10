"""Property test for Orchestrator stage-transition recording (Task 13.6).

Feature: biomedical-rag-pipeline, Property 29: Every stage transition is fully recorded

Validates: Requirements 10.6

For any Processing_Job execution, every stage transition the Orchestrator
records carries the stage identifier, the resulting stage status (one of
``pending | running | succeeded | failed``), and a timestamp. We drive a real
job end-to-end through the five stages with deterministic mock / in-memory
adapters, varying ``stage_retry_limit`` and injecting transient failures at
various stages, and assert:

* every recorded :class:`StageTransition` has a valid ``stage``, a ``status`` in
  the allowed set, and a non-``None`` timestamp;
* each stage that ran records its transitions as ``RUNNING`` immediately
  followed by a terminal (``SUCCEEDED`` or ``FAILED``) status; and
* the per-stage ``attempts`` count on the saved job equals the number of
  ``RUNNING`` transitions recorded for that stage.
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
    StageStatus,
)
from biomed_rag.normalization.normalizer import Normalizer
from biomed_rag.orchestration import Orchestrator
from biomed_rag.parsing import MockParsingEngine, Parser, ParsingEngineRegistry
from biomed_rag.parsing.raw_result import SourceDocument
from biomed_rag.storage.in_memory import InMemoryVectorStore

_EMBED_MODEL_ID = "mock-emb"
_EMBED_DIM = 64
_SOURCE_TEXT = b"Heart tissue analysis.\n\nThe ventricle contracts rhythmically."

# The complete set of stage statuses the model defines (Req 10.6).
_ALLOWED_STATUSES = {
    StageStatus.PENDING,
    StageStatus.RUNNING,
    StageStatus.SUCCEEDED,
    StageStatus.FAILED,
}
_TERMINAL_STATUSES = {StageStatus.SUCCEEDED, StageStatus.FAILED}


# -- fault injection ------------------------------------------------------


class FailFirstN:
    """Trips (raises) on the first ``fail_times`` calls, then succeeds.

    One ``trip()`` is issued per stage *attempt* by every wrapper below, so the
    controller's call count tracks the number of attempts of the stage it guards
    and it fails exactly the first ``fail_times`` of them.
    """

    def __init__(self, fail_times: int) -> None:
        self.fail_times = fail_times
        self.calls = 0

    def trip(self) -> None:
        self.calls += 1
        if self.calls <= self.fail_times:
            raise RuntimeError(f"injected transient failure #{self.calls}")


class _ParserWrapper:
    def __init__(self, inner: Parser, controller: FailFirstN) -> None:
        self._inner = inner
        self._controller = controller

    def parse(self, job, source):  # noqa: ANN001 - mirrors Parser.parse
        self._controller.trip()
        return self._inner.parse(job, source)


class _NormalizerWrapper:
    def __init__(self, inner: Normalizer, controller: FailFirstN) -> None:
        self._inner = inner
        self._controller = controller

    def normalize(self, parsed):  # noqa: ANN001 - mirrors Normalizer.normalize
        self._controller.trip()
        return self._inner.normalize(parsed)

    def serialize(self, doc):  # noqa: ANN001
        return self._inner.serialize(doc)

    def deserialize(self, data):  # noqa: ANN001
        return self._inner.deserialize(data)


class _ChunkerWrapper:
    def __init__(self, inner: Chunker, controller: FailFirstN) -> None:
        self._inner = inner
        self._controller = controller

    def chunk(self, normalized, config):  # noqa: ANN001 - mirrors Chunker.chunk
        self._controller.trip()
        return self._inner.chunk(normalized, config)


class _EmbedderWrapper:
    def __init__(self, inner: Embedder, controller: FailFirstN) -> None:
        self._inner = inner
        self._controller = controller

    def embed(self, chunk, config):  # noqa: ANN001 - mirrors Embedder.embed
        # Each embedding stage attempt re-embeds from the first chunk
        # (orderIndex 0); trip once per attempt to fail the whole stage attempt.
        if chunk.orderIndex == 0:
            self._controller.trip()
        return self._inner.embed(chunk, config)


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


def _build_orchestrator(config, store, vector_store, failing_stage, controller):
    """Assemble an Orchestrator, injecting ``controller`` into ``failing_stage``.

    The four non-storage stages are guarded by wrapping their component; the
    storage stage is guarded via the in-memory store's commit hook.
    """
    parser = _parser(config)
    normalizer = Normalizer()
    chunker = Chunker()
    embedder = _embedder()

    if failing_stage is Stage.PARSING:
        parser = _ParserWrapper(parser, controller)
    elif failing_stage is Stage.NORMALIZATION:
        normalizer = _NormalizerWrapper(normalizer, controller)
    elif failing_stage is Stage.CHUNKING:
        chunker = _ChunkerWrapper(chunker, controller)
    elif failing_stage is Stage.EMBEDDING:
        embedder = _EmbedderWrapper(embedder, controller)

    return Orchestrator(
        config,
        store,
        parser=parser,
        normalizer=normalizer,
        chunker=chunker,
        embedder=embedder,
        vector_store=vector_store,
        source_resolver=lambda job: SourceDocument(
            document_id=job.documentId, raw_bytes=_SOURCE_TEXT
        ),
    )


# -- the property ---------------------------------------------------------


@settings(max_examples=150, deadline=None)
@given(
    stage_retry_limit=st.integers(min_value=0, max_value=4),
    failing_stage=st.one_of(st.none(), st.sampled_from(list(Stage.ordered()))),
    fail_times=st.integers(min_value=1, max_value=6),
)
def test_every_stage_transition_is_fully_recorded(
    stage_retry_limit: int, failing_stage, fail_times: int
) -> None:
    # Feature: biomedical-rag-pipeline, Property 29: Every stage transition is fully recorded
    config = _config(stage_retry_limit)
    store = JobStateStore()
    job = _create_job(store)
    controller = FailFirstN(fail_times)

    if failing_stage is Stage.STORAGE:
        vector_store = InMemoryVectorStore(commit_hook=controller.trip)
    else:
        vector_store = InMemoryVectorStore()

    orch = _build_orchestrator(config, store, vector_store, failing_stage, controller)

    orch.run(job.jobId)

    transitions = orch.transitions(job.jobId)
    saved = store.get(job.jobId)

    # At least one transition is always recorded (parsing always runs first).
    assert transitions, "expected at least one recorded transition"

    # (1) Every transition is fully recorded: a valid stage, a status in the
    # allowed set, and a non-None timestamp (Req 10.6).
    for t in transitions:
        assert isinstance(t.stage, Stage)
        assert t.status in _ALLOWED_STATUSES
        assert t.timestamp is not None
        assert isinstance(t.timestamp, datetime)

    # (2) Per stage, transitions form RUNNING -> terminal pairs in order: each
    # stage attempt records a RUNNING followed immediately by a terminal status.
    for stage in Stage.ordered():
        stage_transitions = [t for t in transitions if t.stage is stage]
        if not stage_transitions:
            continue  # stage never ran (an earlier stage failed terminally)
        assert len(stage_transitions) % 2 == 0, (
            f"{stage.name} transitions should be RUNNING/terminal pairs"
        )
        for i in range(0, len(stage_transitions), 2):
            assert stage_transitions[i].status is StageStatus.RUNNING
            assert stage_transitions[i + 1].status in _TERMINAL_STATUSES

    # (3) The saved per-stage attempt count equals the number of RUNNING
    # transitions recorded for that stage.
    for stage in Stage.ordered():
        running_count = sum(
            1
            for t in transitions
            if t.stage is stage and t.status is StageStatus.RUNNING
        )
        if running_count == 0:
            continue
        assert saved.stageStates[stage].attempts == running_count, (
            f"{stage.name}: attempts={saved.stageStates[stage].attempts} "
            f"!= RUNNING transitions={running_count}"
        )
