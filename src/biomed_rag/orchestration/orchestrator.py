"""The Orchestrator: sequential stage execution, progress, and transitions (Req 10).

This module implements task 13.1: the Orchestrator drives a single
Processing_Job through the five resumable stages

    parsing → normalization → chunking → embedding → storage

strictly in order, starting each stage only after the prior one has succeeded
(Req 10.1). On every transition it records the ``{stage, status, timestamp}``
triple on the job's :class:`~biomed_rag.models.StageState` and exposes the
current stage plus an integer progress percent in ``[0, 100]`` (Req 10.6,
10.7). Each stage's output is persisted to an :class:`ArtifactStore` and the
job's ``StageState.artifactRef`` points at it (Req 10.3, 10.4 — the resume that
consumes these refs is task 13.2).

Terminal outcomes:

* every chunk stored → the job is marked ``COMPLETED`` (Req 8.3); and
* a storage failure → the job is marked ``FAILED`` and the outcome reports the
  chunk ids that were not stored (Req 8.7).

The five stage components (:class:`Parser`, :class:`Normalizer`,
:class:`Chunker`, :class:`Embedder`, :class:`VectorStore`) plus the
:class:`PipelineConfig` and :class:`JobStateStore` are injected, so the
Orchestrator is exercised end-to-end with deterministic mock / in-memory
adapters.

Retry and resume policy (task 13.2) is implemented here: :meth:`_run_stage`
retries a failing stage up to ``stage_retry_limit`` additional attempts before
the job is failed (Req 10.2), failure preserves the artifacts of completed
stages (Req 10.3), and :meth:`resume` re-enters the pipeline at the recorded
failing stage reusing those preserved upstream artifacts (Req 10.4), rejecting
jobs that have no recorded failing stage (Req 10.5).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional, Tuple

from biomed_rag.chunking.chunker import Chunker
from biomed_rag.config import PipelineConfig
from biomed_rag.embedding.embedder import Embedder, EmbedFailed
from biomed_rag.ingestion.job_state_store import JobStateStore
from biomed_rag.models import (
    Chunk,
    OverallStatus,
    ProcessingJob,
    Stage,
    StageState,
    StageStatus,
    VectorRecord,
)
from biomed_rag.normalization.normalizer import Normalizer
from biomed_rag.normalization.result import Malformed
from biomed_rag.parsing.parser import ParseFailure, Parser
from biomed_rag.parsing.raw_result import SourceDocument
from biomed_rag.storage.port import PersistenceError, VectorStore

from .artifact_store import ArtifactRef, ArtifactStore
from .results import (
    JobCompleted,
    JobFailed,
    JobOutcome,
    JobStatus,
    ResumeOutcome,
    ResumeRejected,
    StageTransition,
)

# Resolves the raw Source_Document bytes for a job. The Ingestion_Service does
# not retain document bytes, so the Orchestrator is handed a resolver that maps
# a job to its SourceDocument. Injecting it keeps the Orchestrator decoupled
# from where the bytes live (and trivially mockable in tests).
SourceResolver = Callable[[ProcessingJob], SourceDocument]

# A clock returning timezone-aware UTC timestamps, injectable for determinism.
Clock = Callable[[], datetime]

# Progress is distributed evenly across the five stages; each completed stage
# contributes an equal share so progress is bounded in [0, 100] and monotonic.
_TOTAL_STAGES = len(Stage.ordered())


class StageFailure(Exception):
    """Internal signal that a stage could not complete.

    Carries the human-readable ``reason`` recorded on the stage state and, for a
    storage failure, the ids of the chunks that were not stored (Req 8.7). The
    enclosing stage is supplied by :meth:`Orchestrator._run_stage`, so callers
    raise this with just the reason.
    """

    def __init__(
        self,
        reason: str,
        *,
        unstored_chunk_ids: Optional[List[str]] = None,
    ) -> None:
        self.reason = reason
        self.unstored_chunk_ids = list(unstored_chunk_ids or [])
        super().__init__(reason)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Orchestrator:
    """Coordinates a Processing_Job through the sequential stage pipeline (Req 10)."""

    def __init__(
        self,
        config: PipelineConfig,
        job_store: JobStateStore,
        *,
        parser: Parser,
        normalizer: Normalizer,
        chunker: Chunker,
        embedder: Embedder,
        vector_store: VectorStore,
        source_resolver: SourceResolver,
        artifact_store: Optional[ArtifactStore] = None,
        clock: Clock = _utc_now,
    ) -> None:
        if not isinstance(config, PipelineConfig):
            raise TypeError("config must be a PipelineConfig")
        if not isinstance(job_store, JobStateStore):
            raise TypeError("job_store must be a JobStateStore")
        if not callable(source_resolver):
            raise TypeError("source_resolver must be callable")

        self.config = config
        self.job_store = job_store
        self.parser = parser
        self.normalizer = normalizer
        self.chunker = chunker
        self.embedder = embedder
        self.vector_store = vector_store
        self.source_resolver = source_resolver
        self.artifacts = artifact_store if artifact_store is not None else ArtifactStore()
        self.clock = clock

        # Full transition history per job, appended on every transition
        # (Req 10.6). The job's StageState holds the *latest* state; this log
        # retains the ordered sequence for observability and the resume/property
        # work in later tasks.
        self._transitions: Dict[str, List[StageTransition]] = {}

    # -- public API -------------------------------------------------------
    def run(self, job_id: str) -> JobOutcome:
        """Execute the full pipeline for ``job_id`` and return its outcome.

        Stages run strictly in order, each starting only after the prior stage
        has succeeded (Req 10.1). A failing stage is retried up to
        ``stage_retry_limit`` additional times before the job is failed (Req
        10.2). Returns :class:`JobCompleted` when every chunk is stored (Req
        8.3), or :class:`JobFailed` identifying the failing stage (and, for
        storage, the unstored chunk ids) once retries are exhausted (Req 8.7,
        10.3).
        """
        job = self.job_store.get(job_id)
        self._transitions.setdefault(job.jobId, [])
        job.overallStatus = OverallStatus.RUNNING
        job.failingStage = None
        self._save(job)

        try:
            stored_chunk_ids = self._run_pipeline_from(job, Stage.PARSING, None)
        except StageFailure as failure:
            return self._mark_failed(job, failure)

        return self._mark_completed(job, stored_chunk_ids)

    def resume(self, job_id: str) -> ResumeOutcome:
        """Resume a previously failed job from its recorded failing stage.

        Restarts execution at ``job.failingStage`` rather than at the first
        stage, reusing the artifacts persisted by the stages that completed
        before it — the parsed document, the durable serialized normalized
        document (deserialized back into a :class:`NormalizedDocument`), the
        chunk set, or the embedding set — so no already-completed stage is
        re-executed (Req 10.4).

        If the job has no recorded ``failingStage`` it is not in a resumable
        state (it never failed, or already completed); the request is rejected
        with a :class:`ResumeRejected` carrying the reason (Req 10.5).
        """
        job = self.job_store.get(job_id)
        failing_stage = job.failingStage
        if failing_stage is None:
            return ResumeRejected(
                jobId=job.jobId,
                reason=(
                    "job has no recorded failing stage and is not in a resumable "
                    f"state (overallStatus={job.overallStatus.name})"
                ),
            )

        self._transitions.setdefault(job.jobId, [])
        # Reconstruct the input to the failing stage from the preserved upstream
        # artifact before mutating job state (Req 10.4).
        resume_input = self._resume_input(job, failing_stage)

        job.overallStatus = OverallStatus.RUNNING
        job.failingStage = None
        self._save(job)

        try:
            stored_chunk_ids = self._run_pipeline_from(
                job, failing_stage, resume_input
            )
        except StageFailure as failure:
            return self._mark_failed(job, failure)

        return self._mark_completed(job, stored_chunk_ids)

    def status(self, job_id: str) -> JobStatus:
        """Return the live ``{currentStage, stageStatuses, progressPercent}`` view.

        Every stage has a status; stages not yet reached report ``PENDING``.
        ``progressPercent`` is the integer percent in ``[0, 100]`` (Req 10.7).
        """
        job = self.job_store.get(job_id)
        stage_statuses = {
            stage: (
                job.stageStates[stage].status
                if stage in job.stageStates
                else StageStatus.PENDING
            )
            for stage in Stage.ordered()
        }
        return JobStatus(
            currentStage=job.currentStage,
            stageStatuses=stage_statuses,
            progressPercent=job.progressPercent,
        )

    def transitions(self, job_id: str) -> List[StageTransition]:
        """Return the ordered transition history recorded for ``job_id`` (Req 10.6)."""
        return list(self._transitions.get(job_id, []))

    # -- pipeline runner --------------------------------------------------
    def _run_pipeline_from(
        self,
        job: ProcessingJob,
        start_stage: Stage,
        initial_input: object,
    ) -> List[str]:
        """Run stages from ``start_stage`` to the end, threading each result.

        Stages before ``start_stage`` are skipped — for a fresh ``run`` this is
        :data:`Stage.PARSING` (nothing skipped); for a ``resume`` it is the
        recorded failing stage, with ``initial_input`` carrying the
        reconstructed upstream artifact so completed stages are not re-executed
        (Req 10.4). Returns the stored chunk ids produced by the storage stage,
        which always runs (it is the terminal stage).
        """
        current = initial_input
        stored_chunk_ids: List[str] = []
        for stage in Stage.ordered():
            if stage.order < start_stage.order:
                continue
            current = self._run_stage_with_input(job, stage, current)
            if stage is Stage.STORAGE:
                stored_chunk_ids = list(current)  # type: ignore[arg-type]
        return stored_chunk_ids

    def _run_stage_with_input(
        self, job: ProcessingJob, stage: Stage, stage_input: object
    ) -> object:
        """Execute ``stage`` (with retries) on ``stage_input`` via :meth:`_run_stage`."""
        executor = self._stage_executor(job, stage)
        return self._run_stage(job, stage, lambda: executor(stage_input))

    def _stage_executor(
        self, job: ProcessingJob, stage: Stage
    ) -> Callable[[object], Tuple[object, object]]:
        """Return the ``(input) -> (result, payload)`` executor for ``stage``.

        Parsing ignores its input and resolves the Source_Document bytes itself;
        every other stage consumes the previous stage's result (or, on resume,
        the reconstructed upstream artifact).
        """
        if stage is Stage.PARSING:
            return lambda _input: self._do_parse(job)
        if stage is Stage.NORMALIZATION:
            return lambda parsed: self._do_normalize(parsed)
        if stage is Stage.CHUNKING:
            return lambda normalized: self._do_chunk(normalized)
        if stage is Stage.EMBEDDING:
            return lambda chunks: self._do_embed(job, chunks)
        if stage is Stage.STORAGE:
            return lambda records: self._do_store(job, records)
        raise ValueError(f"no executor for stage {stage!r}")  # defensive

    def _resume_input(self, job: ProcessingJob, failing_stage: Stage) -> object:
        """Reconstruct the input to ``failing_stage`` from preserved artifacts.

        The failing stage's input is the output persisted by the immediately
        preceding stage (Req 10.4). Parsing has no upstream artifact — it
        re-resolves the Source_Document. The normalization artifact is the
        durable *serialized* form, so it is deserialized back into a
        :class:`NormalizedDocument` before chunking consumes it (Req 5.6, 10.4).
        """
        if failing_stage is Stage.PARSING:
            return None
        prior_stage = Stage.ordered()[failing_stage.order - 1]
        prior_state = job.stageStates.get(prior_stage)
        if prior_state is None or prior_state.artifactRef is None:
            raise StageFailure(
                f"cannot resume from {failing_stage.name}: no preserved artifact "
                f"for completed stage {prior_stage.name}"
            )
        artifact = self.artifacts.get(prior_state.artifactRef)
        if prior_stage is Stage.NORMALIZATION:
            # Stored in durable serialized byte form; rebuild the document.
            return self.normalizer.deserialize(artifact)
        return artifact

    # -- stage runner -----------------------------------------------------
    def _run_stage(
        self,
        job: ProcessingJob,
        stage: Stage,
        execute: Callable[[], Tuple[object, object]],
    ) -> object:
        """Record transitions around one stage, retrying on failure (Req 10.2).

        ``execute`` runs the stage and returns ``(result, artifact_payload)``:
        the logical result handed to the next stage, and the payload persisted
        to the :class:`ArtifactStore`. The stage is attempted up to
        ``stage_retry_limit + 1`` times (the initial attempt plus the configured
        number of retries). Each attempt records a RUNNING transition and, on
        failure, a FAILED transition. If every attempt fails the last
        :class:`StageFailure` is re-raised so :meth:`run` / :meth:`resume` can
        finalize the job (Req 10.3). On success the artifact is persisted and a
        SUCCEEDED transition recorded.
        """
        max_attempts = self.config.stage_retry_limit + 1
        last_failure: Optional[StageFailure] = None
        for _attempt in range(max_attempts):
            self._record(job, stage, StageStatus.RUNNING)
            try:
                result, payload = execute()
            except StageFailure as failure:
                self._record(
                    job, stage, StageStatus.FAILED, failure_reason=failure.reason
                )
                last_failure = failure
                continue
            except Exception as exc:  # defensive: an unexpected error fails the stage
                failure = StageFailure(f"unexpected error during {stage.name}: {exc}")
                self._record(
                    job, stage, StageStatus.FAILED, failure_reason=failure.reason
                )
                last_failure = failure
                continue

            ref = self._persist_artifact(job, stage, payload)
            self._record(job, stage, StageStatus.SUCCEEDED, artifact_ref=ref)
            return result

        # Retries exhausted: surface the final failure (Req 10.3).
        assert last_failure is not None  # loop runs >= 1 time, so always set on failure
        raise last_failure

    # -- stage implementations -------------------------------------------
    def _do_parse(self, job: ProcessingJob) -> Tuple[object, object]:
        """Parsing stage: Source_Document → Parsed_Document (Req 2)."""
        try:
            source = self.source_resolver(job)
        except Exception as exc:
            raise StageFailure(f"could not resolve source document: {exc}") from exc
        if not isinstance(source, SourceDocument):
            raise StageFailure(
                "source_resolver did not return a SourceDocument; got "
                f"{type(source).__name__}"
            )
        try:
            parsed = self.parser.parse(job, source)
        except ParseFailure as failure:
            # The Parser fails closed and records its own reason; surface it as a
            # stage failure so the Orchestrator owns the recorded transition.
            raise StageFailure(failure.reason) from failure
        return parsed, parsed

    def _do_normalize(self, parsed: object) -> Tuple[object, object]:
        """Normalization stage: Parsed_Document → Normalized_Document (Req 5).

        A :class:`Malformed` result fails the stage (Req 5.8). An ``Empty``
        result is a valid (empty) normalized document and proceeds — chunking
        will yield zero chunks. The artifact is persisted in its durable
        serialized byte form for resume (Req 5.6).
        """
        result = self.normalizer.normalize(parsed)
        if isinstance(result, Malformed):
            raise StageFailure(f"normalization rejected malformed input: {result.error}")
        document = result.document
        payload = self.normalizer.serialize(document)
        return document, payload

    def _do_chunk(self, normalized: object) -> Tuple[object, object]:
        """Chunking stage: Normalized_Document → Chunk set (Req 6)."""
        chunks = self.chunker.chunk(normalized, self.config)
        return chunks, chunks

    def _do_embed(
        self, job: ProcessingJob, chunks: List[Chunk]
    ) -> Tuple[object, object]:
        """Embedding stage: Chunk set → Embedding set (Req 7).

        Each chunk is embedded through the injected Embedder (which owns the
        dimension/timeout/retry policy). If any chunk cannot be embedded the
        stage fails, reporting the affected chunk ids — storage requires a
        complete embedding set.
        """
        records: List[VectorRecord] = []
        failed_chunk_ids: List[str] = []
        for chunk in chunks:
            result = self.embedder.embed(chunk, self.config)
            if isinstance(result, EmbedFailed):
                failed_chunk_ids.append(chunk.chunkId)
                continue
            records.append(
                VectorRecord(
                    documentId=chunk.documentId,
                    chunk=chunk,
                    embedding=result,
                )
            )
        if failed_chunk_ids:
            raise StageFailure(
                "embedding failed for chunks: " + ", ".join(failed_chunk_ids)
            )
        return records, records

    def _do_store(
        self, job: ProcessingJob, records: List[VectorRecord]
    ) -> Tuple[object, object]:
        """Storage stage: persist every record for the document (Req 8.1, 8.3, 8.7).

        ``upsert_batch`` is all-or-nothing: a :class:`PersistenceError` stores
        nothing, so every chunk is reported unstored (Req 8.6, 8.7). On success
        the stored chunk ids are compared against the expected set; any
        shortfall fails the stage reporting exactly the unstored ids (Req 8.7).
        An empty record set stores nothing and trivially completes.
        """
        expected_ids = [record.chunk.chunkId for record in records]
        try:
            store_result = self.vector_store.upsert_batch(job.documentId, records)
        except PersistenceError as exc:
            raise StageFailure(
                f"storage failed to persist chunks: {exc}",
                unstored_chunk_ids=expected_ids,
            ) from exc

        stored = set(store_result.stored_chunk_ids)
        unstored = [cid for cid in expected_ids if cid not in stored]
        if unstored:
            raise StageFailure(
                "storage did not persist all chunks",
                unstored_chunk_ids=unstored,
            )
        stored_chunk_ids = list(store_result.stored_chunk_ids)
        return stored_chunk_ids, stored_chunk_ids

    # -- transition recording (Req 10.6, 10.7) ---------------------------
    def _record(
        self,
        job: ProcessingJob,
        stage: Stage,
        status: StageStatus,
        *,
        artifact_ref: Optional[ArtifactRef] = None,
        failure_reason: Optional[str] = None,
    ) -> None:
        """Record one ``{stage, status, timestamp}`` transition (Req 10.6).

        Updates the stage's :class:`StageState` (incrementing ``attempts`` when a
        stage starts running), advances ``currentStage`` and the integer
        ``progressPercent`` (Req 10.7), appends to the transition log, and
        persists the job.
        """
        now = self.clock()
        previous = job.stageStates.get(stage)
        attempts = previous.attempts if previous is not None else 0
        if status is StageStatus.RUNNING:
            attempts += 1
        # Preserve a previously recorded artifact ref unless this transition
        # supplies a new one (e.g. a RUNNING transition keeps the prior ref).
        ref = artifact_ref if artifact_ref is not None else (
            previous.artifactRef if previous is not None else None
        )

        job.stageStates[stage] = StageState(
            stage=stage,
            status=status,
            attempts=attempts,
            lastTransitionAt=now,
            failureReason=failure_reason,
            artifactRef=ref,
        )
        job.currentStage = stage
        job.progressPercent = self._progress(job)
        self._transitions.setdefault(job.jobId, []).append(
            StageTransition(stage=stage, status=status, timestamp=now)
        )
        self._save(job)

    def _progress(self, job: ProcessingJob) -> int:
        """Integer completion percent in ``[0, 100]`` (Req 10.7).

        Each succeeded stage contributes an equal share of the total. Because
        stages succeed in order and a SUCCEEDED status is never rolled back
        within a run, the value is monotonically non-decreasing and bounded.
        """
        succeeded = sum(
            1
            for stage in Stage.ordered()
            if (state := job.stageStates.get(stage)) is not None
            and state.status is StageStatus.SUCCEEDED
        )
        return (succeeded * 100) // _TOTAL_STAGES

    # -- artifact persistence (Req 10.3, 10.4) ---------------------------
    def _persist_artifact(
        self, job: ProcessingJob, stage: Stage, payload: object
    ) -> ArtifactRef:
        """Persist ``payload`` for ``stage`` and return its :data:`ArtifactRef`."""
        ref = f"{job.jobId}::{stage.name}"
        return self.artifacts.put(ref, payload)

    # -- finalization -----------------------------------------------------
    def _mark_completed(
        self, job: ProcessingJob, stored_chunk_ids: List[str]
    ) -> JobCompleted:
        """Mark the job COMPLETED once every chunk is stored (Req 8.3)."""
        job.overallStatus = OverallStatus.COMPLETED
        job.failingStage = None
        self._save(job)
        return JobCompleted(jobId=job.jobId, storedChunkIds=list(stored_chunk_ids))

    def _mark_failed(self, job: ProcessingJob, failure: StageFailure) -> JobFailed:
        """Mark the job FAILED at its current stage (Req 8.7, 10.3)."""
        failing_stage = job.currentStage
        job.failingStage = failing_stage
        job.overallStatus = OverallStatus.FAILED
        self._save(job)
        return JobFailed(
            jobId=job.jobId,
            failingStage=failing_stage,
            reason=failure.reason,
            unstoredChunkIds=list(failure.unstored_chunk_ids),
        )

    # -- persistence helper ----------------------------------------------
    def _save(self, job: ProcessingJob) -> None:
        self.job_store.save(job)
