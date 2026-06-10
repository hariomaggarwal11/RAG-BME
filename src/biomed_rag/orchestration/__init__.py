"""Pipeline orchestration (Req 10).

Exposes the :class:`Orchestrator`, which drives a Processing_Job through the
five sequential stages (parsing → normalization → chunking → embedding →
storage), persists each stage artifact, records every stage transition, and
exposes live progress. Supporting types: the :class:`ArtifactStore` for
per-stage artifacts and the outcome/observability values
(:class:`JobCompleted`, :class:`JobFailed`, :class:`JobStatus`,
:class:`StageTransition`).
"""

from __future__ import annotations

from .artifact_store import ArtifactRef, ArtifactStore, UnknownArtifactError
from .orchestrator import Orchestrator, SourceResolver, StageFailure
from .results import (
    JobCompleted,
    JobFailed,
    JobOutcome,
    JobStatus,
    ResumeOutcome,
    ResumeRejected,
    StageTransition,
)

__all__ = [
    "Orchestrator",
    "SourceResolver",
    "StageFailure",
    "ArtifactStore",
    "ArtifactRef",
    "UnknownArtifactError",
    "JobCompleted",
    "JobFailed",
    "JobOutcome",
    "ResumeRejected",
    "ResumeOutcome",
    "JobStatus",
    "StageTransition",
]
