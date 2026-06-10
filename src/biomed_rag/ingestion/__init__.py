"""Ingestion stage package (Req 1).

This package contains the :class:`IngestionService` and its supporting
:class:`JobStateStore`. The store owns the two pieces of state the design
assigns to the "Job State Store": the unique job-identifier space (Req 1.1)
and the content-hash dedup index (Req 1.5).
"""

from __future__ import annotations

from .job_state_store import (
    DuplicateContentHashError,
    JobStateStore,
    UnknownJobError,
)
from .service import (
    Accepted,
    Duplicate,
    FileInput,
    IngestionResult,
    IngestionService,
    Rejected,
    RejectionCode,
)

__all__ = [
    "JobStateStore",
    "DuplicateContentHashError",
    "UnknownJobError",
    # service
    "IngestionService",
    "FileInput",
    "IngestionResult",
    "Accepted",
    "Duplicate",
    "Rejected",
    "RejectionCode",
]
