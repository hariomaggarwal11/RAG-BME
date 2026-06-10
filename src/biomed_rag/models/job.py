"""Processing_Job and related state models (Req 1, 10)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Optional

from ._validation import (
    require,
    require_int_in_range,
    require_non_empty_str,
)
from .enums import Format, OverallStatus, Stage, StageStatus
from .identifiers import DocumentId, JobId

# 500 MB inclusive upper bound on document size (Req 1.2, 1.6).
MAX_BYTE_SIZE = 524_288_000
MAX_FILENAME_LENGTH = 255


@dataclass
class DocumentMetadata:
    """Metadata recorded for an accepted submission (Req 1.4)."""

    filename: str
    format: Format
    byteSize: int
    contentHash: str
    submittedAtUtc: datetime

    def __post_init__(self) -> None:
        # Filename present and within the 255-char limit (Req 1.8).
        require(isinstance(self.filename, str), "DocumentMetadata.filename must be a str")
        require(
            1 <= len(self.filename) <= MAX_FILENAME_LENGTH,
            f"DocumentMetadata.filename length must be in [1, {MAX_FILENAME_LENGTH}]",
        )
        require(
            isinstance(self.format, Format), "DocumentMetadata.format must be a Format"
        )
        # Byte size in [1, 500 MB] inclusive (Req 1.2, 1.6, 1.7).
        require_int_in_range(
            self.byteSize, "DocumentMetadata.byteSize", minimum=1, maximum=MAX_BYTE_SIZE
        )
        require_non_empty_str(self.contentHash, "DocumentMetadata.contentHash")
        require(
            isinstance(self.submittedAtUtc, datetime),
            "DocumentMetadata.submittedAtUtc must be a datetime",
        )
        # Recorded in UTC (Req 1.4): reject a non-UTC tz-aware timestamp.
        if self.submittedAtUtc.tzinfo is not None:
            require(
                self.submittedAtUtc.utcoffset() == timezone.utc.utcoffset(None),
                "DocumentMetadata.submittedAtUtc must be in UTC",
            )


@dataclass
class StageState:
    """Recorded state of a single stage, updated on every transition (Req 10.6)."""

    stage: Stage
    status: StageStatus = StageStatus.PENDING
    attempts: int = 0
    lastTransitionAt: Optional[datetime] = None
    failureReason: Optional[str] = None
    artifactRef: Optional[str] = None

    def __post_init__(self) -> None:
        require(isinstance(self.stage, Stage), "StageState.stage must be a Stage")
        require(
            isinstance(self.status, StageStatus),
            "StageState.status must be a StageStatus",
        )
        require_int_in_range(self.attempts, "StageState.attempts", minimum=0)
        if self.lastTransitionAt is not None:
            require(
                isinstance(self.lastTransitionAt, datetime),
                "StageState.lastTransitionAt must be a datetime or None",
            )
        if self.failureReason is not None:
            require(
                isinstance(self.failureReason, str),
                "StageState.failureReason must be a str or None",
            )
        if self.artifactRef is not None:
            require(
                isinstance(self.artifactRef, str),
                "StageState.artifactRef must be a str or None",
            )


@dataclass
class ProcessingJob:
    """A tracked unit of work through the pipeline (Req 1, 10)."""

    jobId: JobId
    documentId: DocumentId
    metadata: DocumentMetadata
    currentStage: Stage = Stage.PARSING
    stageStates: Dict[Stage, StageState] = field(default_factory=dict)
    failingStage: Optional[Stage] = None
    progressPercent: int = 0
    overallStatus: OverallStatus = OverallStatus.QUEUED

    def __post_init__(self) -> None:
        require_non_empty_str(self.jobId, "ProcessingJob.jobId")
        require_non_empty_str(self.documentId, "ProcessingJob.documentId")
        require(
            isinstance(self.metadata, DocumentMetadata),
            "ProcessingJob.metadata must be a DocumentMetadata",
        )
        require(
            isinstance(self.currentStage, Stage),
            "ProcessingJob.currentStage must be a Stage",
        )
        require(
            all(
                isinstance(k, Stage) and isinstance(v, StageState)
                for k, v in self.stageStates.items()
            ),
            "ProcessingJob.stageStates must map Stage -> StageState",
        )
        require(
            all(k == v.stage for k, v in self.stageStates.items()),
            "ProcessingJob.stageStates keys must match their StageState.stage",
        )
        if self.failingStage is not None:
            require(
                isinstance(self.failingStage, Stage),
                "ProcessingJob.failingStage must be a Stage or None",
            )
        # Progress is an integer percent in [0, 100] (Req 10.7).
        require_int_in_range(
            self.progressPercent, "ProcessingJob.progressPercent", minimum=0, maximum=100
        )
        require(
            isinstance(self.overallStatus, OverallStatus),
            "ProcessingJob.overallStatus must be an OverallStatus",
        )
