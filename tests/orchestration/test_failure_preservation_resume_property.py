"""Property test for failure preservation and correct resume (Task 13.5).

Feature: biomedical-rag-pipeline, Property 28: Failure preserves completed-stage outputs and enables correct resume

Property 28 (design): *For any* Processing_Job that fails at a stage S after
exhausting retries, the failing stage S is recorded, the persisted outputs of
all stages completed before S remain intact, and resuming the job restarts
execution at S without re-executing any stage before S.

Validates: Requirements 10.3, 10.4

Strategy
--------
For a generated failing stage S in {chunking, embedding, storage} and a
generated source document, we drive two runs of the *same* deterministic
pipeline:

* an **uninterrupted** run (no fault) that records the canonical stored result;
  and
* a **faulted** run whose stage S fails on every initial attempt (exhausting
  ``stageRetryLimit`` retries) and then *clears* before resume.

After the faulted run we assert the job is FAILED at S, every stage completed
before S is SUCCEEDED with its artifact preserved in the ArtifactStore (Req
10.3). We then clear the fault and ``resume``: the job completes, every
transition recorded after the failure is for a stage at or after S (no earlier
stage is re-executed, Req 10.4), and the content the resumed run stored is
identical to what the uninterrupted run stored.

Chunk identifiers are freshly minted UUIDs per run, so equality is asserted on
the stable *content* projection of each stored record (order, text, metadata,
and the deterministic embedding vector) rather than on the random chunk ids.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Tuple

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
from biomed_rag.orchestration import JobCompleted, JobFailed, Orchestrator
from biomed_rag.parsing import MockParsingEngine, Parser, ParsingEngineRegistry
from biomed_rag.parsing.raw_result import SourceDocument
from biomed_rag.storage.in_memory import InMemoryVectorStore

_EMBED_MODEL_ID = "mock-emb"
_EMBED_DIM = 64

# The three non-initial stages that have at least one completed upstream stage
# whose preserved output a resume must reuse (parsing has no upstream artifact,
# normalization's only upstream is parsing). These are the failing stages this
# property exercises.
_FAILING_STAGES = [Stage.CHUNKING, Stage.EMBEDDING, Stage.STORAGE]


# -- toggleable fault injectors -------------------------------------------
#
# Each injector fails while ``should_fail`` is set, so it fails every attempt of
# the initial run (exhausting retries) and then, once cleared, lets the stage
# succeed on resume.


class ControllableCommit:
    """Storage commit hook that fails while ``should_fail`` is set (Req 8.6)."""

    def __init__(self, *, should_fail: bool = True) -> None:
        self.should_fail = should_fail
        self.calls = 0

    def __call__(self) -> None:
        self.calls += 1
        if self.should_fail:
            raise RuntimeError(f"transient storage failure #{self.calls}")


class ControllableChunker:
    """Wraps a real :class:`Chunker`, failing while ``should_fail`` is set."""

    def __init__(self, inner: Chunker, *, should_fail: bool = True) -> None:
        self._inner = inner
        self.should_fail = should_fail
        self.calls = 0

    def chunk(self, normalized, config):  # noqa: ANN001 - mirrors Chunker.chunk
        self.calls += 1
        if self.should_fail:
            raise RuntimeError(f"transient chunk failure #{self.calls}")
        return self._inner.chunk(normalized, config)


class ControllableEmbedder:
    """Wraps a real :class:`Embedder`, failing while ``should_fail`` is set."""

    def __init__(self, inner: Embedder, *, should_fail: bool = True) -> None:
        self._inner = inner
        self.should_fail = should_fail
        self.calls = 0

    def embed(self, chunk, config):  # noqa: ANN001 - mirrors Embedder.embed
        self.calls += 1
        if self.should_fail:
            raise RuntimeError(f"transient embed failure #{self.calls}")
        return self._inner.embed(chunk, config)


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


def _create_job(store: JobStateStore, source: bytes, document_id: str):
    metadata = DocumentMetadata(
        filename="paper.pdf",
        format=Format.PDF,
        byteSize=len(source),
        contentHash=document_id,
        submittedAtUtc=datetime.now(timezone.utc),
    )
    return store.create_job(document_id=document_id, metadata=metadata)


def _orchestrator(config, store, vector_store, *, chunker=None, embedder=None):
    return Orchestrator(
        config,
        store,
        parser=_parser(config),
        normalizer=Normalizer(),
        chunker=chunker if chunker is not None else Chunker(),
        embedder=embedder if embedder is not None else _embedder(),
        vector_store=vector_store,
        source_resolver=lambda job, src=None: SourceDocument(
            document_id=job.documentId, raw_bytes=_SOURCE_BY_DOC[job.documentId]
        ),
    )


# Maps documentId -> source bytes so the injected resolver can serve the exact
# bytes generated for each example (the resolver receives only the job).
_SOURCE_BY_DOC: dict = {}


# -- stored-result projection ---------------------------------------------


def _stored_projection(vector_store, document_id: str) -> List[Tuple]:
    """Content-stable projection of every stored record, ordered deterministically.

    Excludes the randomly-minted ``chunkId`` and compares the durable content:
    order, text, source metadata, and the deterministic embedding vector.
    """
    projection = []
    for record in vector_store.get_document(document_id):
        chunk = record.chunk
        projection.append(
            (
                chunk.orderIndex,
                chunk.content,
                chunk.pageNumber,
                tuple(chunk.headingPath),
                chunk.isTablePart,
                chunk.tokenCount,
                tuple(record.embedding.vector),
                record.embedding.modelId,
            )
        )
    projection.sort(key=lambda row: (row[0], row[1]))
    return projection


# -- generators -----------------------------------------------------------

_words = st.text(alphabet="abcdefghijklmnopqrstuvwxyz", min_size=1, max_size=8)
_paragraph = st.lists(_words, min_size=1, max_size=6).map(" ".join)
# One or more non-empty paragraphs joined on blank lines; the mock parser splits
# on "\n\n", so this yields a document with extractable content (>= 1 chunk).
_source_text = st.lists(_paragraph, min_size=1, max_size=4).map(
    lambda paras: ("\n\n".join(paras)).encode("utf-8")
)


def _make_fault(failing_stage: Stage):
    """Return ``(chunker, embedder, commit)`` with exactly one toggled fault."""
    chunker = None
    embedder = None
    commit = ControllableCommit(should_fail=False)
    if failing_stage is Stage.CHUNKING:
        chunker = ControllableChunker(Chunker(), should_fail=True)
    elif failing_stage is Stage.EMBEDDING:
        embedder = ControllableEmbedder(_embedder(), should_fail=True)
    elif failing_stage is Stage.STORAGE:
        commit = ControllableCommit(should_fail=True)
    return chunker, embedder, commit


def _clear_fault(failing_stage: Stage, chunker, embedder, commit) -> None:
    if failing_stage is Stage.CHUNKING:
        chunker.should_fail = False
    elif failing_stage is Stage.EMBEDDING:
        embedder.should_fail = False
    elif failing_stage is Stage.STORAGE:
        commit.should_fail = False


# -- the property ---------------------------------------------------------


@settings(max_examples=120, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    source=_source_text,
    failing_stage=st.sampled_from(_FAILING_STAGES),
    retry_limit=st.integers(min_value=0, max_value=3),
    seed=st.integers(min_value=0, max_value=10_000),
)
def test_failure_preserves_outputs_and_resume_is_correct(
    source: bytes, failing_stage: Stage, retry_limit: int, seed: int
) -> None:
    # Feature: biomedical-rag-pipeline, Property 28: Failure preserves completed-stage outputs and enables correct resume
    config = _config(stage_retry_limit=retry_limit)

    # 1) Uninterrupted reference run: records the canonical stored result.
    ref_doc = f"doc-ref-{seed}"
    _SOURCE_BY_DOC[ref_doc] = source
    ref_store = JobStateStore()
    ref_job = _create_job(ref_store, source, ref_doc)
    ref_vs = InMemoryVectorStore()
    ref_orch = _orchestrator(config, ref_store, ref_vs)
    ref_outcome = ref_orch.run(ref_job.jobId)
    assert isinstance(ref_outcome, JobCompleted)
    expected_projection = _stored_projection(ref_vs, ref_doc)

    # 2) Faulted run: stage S fails on every initial attempt, then clears.
    doc = f"doc-resume-{seed}"
    _SOURCE_BY_DOC[doc] = source
    store = JobStateStore()
    job = _create_job(store, source, doc)
    chunker, embedder, commit = _make_fault(failing_stage)
    vector_store = InMemoryVectorStore(commit_hook=commit)
    orch = _orchestrator(
        config, store, vector_store, chunker=chunker, embedder=embedder
    )

    failed = orch.run(job.jobId)

    # The job failed at the generated stage, and the failing stage is recorded.
    assert isinstance(failed, JobFailed)
    assert failed.failingStage is failing_stage
    saved = store.get(job.jobId)
    assert saved.overallStatus is OverallStatus.FAILED
    assert saved.failingStage is failing_stage
    assert saved.stageStates[failing_stage].status is StageStatus.FAILED

    # Every stage completed before S succeeded and its artifact is preserved
    # in the ArtifactStore (Req 10.3).
    upstream = [s for s in Stage.ordered() if s.order < failing_stage.order]
    assert upstream, "failing stage must have at least one completed upstream stage"
    for stage in upstream:
        state = saved.stageStates[stage]
        assert state.status is StageStatus.SUCCEEDED
        assert state.artifactRef is not None
        assert orch.artifacts.has(state.artifactRef)

    transitions_before = len(orch.transitions(job.jobId))

    # 3) Clear the fault and resume.
    _clear_fault(failing_stage, chunker, embedder, commit)
    resumed = orch.resume(job.jobId)

    # Resume completes the job.
    assert isinstance(resumed, JobCompleted)
    saved = store.get(job.jobId)
    assert saved.overallStatus is OverallStatus.COMPLETED
    assert saved.failingStage is None

    # Resume re-entered at S: every transition recorded after the failure is for
    # a stage at or after S, so no earlier (already-completed) stage ran again
    # (Req 10.4).
    new_transitions = orch.transitions(job.jobId)[transitions_before:]
    assert new_transitions, "resume should record at least one transition"
    assert all(t.stage.order >= failing_stage.order for t in new_transitions)

    # The resumed run stored exactly what the uninterrupted run stored.
    assert _stored_projection(vector_store, doc) == expected_projection
