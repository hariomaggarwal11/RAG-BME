"""Orchestrator outcome and observability types (Req 8.3, 8.7, 10.6, 10.7).

These are the values the Orchestrator returns and exposes:

* :class:`JobOutcome` — the terminal result of :meth:`Orchestrator.run`:
  :class:`JobCompleted` when every chunk was stored (Req 8.3) or
  :class:`JobFailed` carrying the failing stage and, for a storage failure, the
  chunk ids that were not stored (Req 8.7, 10.3).
* :class:`StageTransition` — an immutable record of one ``{stage, status,
  timestamp}`` transition, appended on every transition (Req 10.6).
* :class:`JobStatus` — the live ``{currentStage, stageStatuses,
  progressPercent}`` view exposed while a job runs (Req 10.7).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Union

from biomed_rag.models import JobId, Stage, StageStatus


@dataclass(frozen=True)
class StageTransition:
    """One recorded stage transition: ``{stage, status, timestamp}`` (Req 10.6)."""

    stage: Stage
    status: StageStatus
    timestamp: datetime


@dataclass(frozen=True)
class JobCompleted:
    """A job whose every chunk was successfully stored (Req 8.3)."""

    jobId: JobId
    storedChunkIds: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class JobFailed:
    """A job that failed at ``failingStage`` (Req 8.7, 10.3).

    ``unstoredChunkIds`` is populated for a storage-stage failure to report
    exactly which chunks were not stored (Req 8.7); it is empty for failures of
    other stages.
    """

    jobId: JobId
    failingStage: Stage
    reason: str
    unstoredChunkIds: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class ResumeRejected:
    """A rejected resume request for a non-resumable job (Req 10.5).

    Returned by :meth:`Orchestrator.resume` when the job has no recorded
    ``failingStage`` — i.e. it never failed (it is queued, running, or already
    completed). ``reason`` is a human-readable explanation of why the job is not
    in a resumable state.
    """

    jobId: JobId
    reason: str


# JobOutcome = JobCompleted | JobFailed  (design: Orchestrator.run -> JobOutcome).
JobOutcome = Union[JobCompleted, JobFailed]

# resume(jobId) -> JobOutcome | Rejected(notResumable)  (design: Orchestrator).
ResumeOutcome = Union[JobCompleted, JobFailed, ResumeRejected]


@dataclass(frozen=True)
class JobStatus:
    """The live observability view of a job (Req 10.7).

    ``stageStatuses`` carries a status for every stage (PENDING for stages not
    yet reached). ``progressPercent`` is an integer in ``[0, 100]``.
    """

    currentStage: Stage
    stageStatuses: Dict[Stage, StageStatus]
    progressPercent: int


__all__ = [
    "StageTransition",
    "JobCompleted",
    "JobFailed",
    "JobOutcome",
    "ResumeRejected",
    "ResumeOutcome",
    "JobStatus",
]
