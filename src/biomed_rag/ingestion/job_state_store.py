"""In-memory Job State Store with a content-hash dedup index (Req 1.1, 1.5).

The design's high-level architecture places a "Job State Store" between the
Ingestion_Service and the Orchestrator. It owns two responsibilities that the
rest of the ingestion logic builds on:

* **Unique job identifiers (Req 1.1).** Every persisted :class:`ProcessingJob`
  is assigned a UUID job identifier that is guaranteed distinct from every
  other job the store has ever issued. :meth:`JobStateStore.generate_job_id`
  regenerates on the astronomically-unlikely event of a UUID collision so the
  uniqueness guarantee is total rather than merely probabilistic.
* **Content-hash dedup index (Req 1.5).** Jobs are indexed by the SHA-256
  content hash recorded in their :class:`DocumentMetadata`, so a byte-identical
  resubmission can be mapped back to the originally assigned job identifier
  without creating a second job.

The store is deliberately backend-agnostic: it keeps everything in plain
dictionaries so it can back property and unit tests deterministically. A
persistent adapter could later implement the same surface.
"""

from __future__ import annotations

from typing import Dict, Iterator, List, Optional

from ..models import (
    DocumentId,
    DocumentMetadata,
    JobId,
    ProcessingJob,
    new_job_id,
)


class UnknownJobError(KeyError):
    """Raised when a lookup or update targets a job id the store does not hold."""

    def __init__(self, job_id: JobId) -> None:
        self.job_id = job_id
        super().__init__(f"No ProcessingJob registered for job id {job_id!r}")


class DuplicateContentHashError(ValueError):
    """Raised when a *new* job is created for a content hash already on file.

    The carried :attr:`existing_job_id` lets the caller (the IngestionService)
    honor the dedup contract by returning the original job identifier instead
    of creating a second Processing_Job (Req 1.5).
    """

    def __init__(self, content_hash: str, existing_job_id: JobId) -> None:
        self.content_hash = content_hash
        self.existing_job_id = existing_job_id
        super().__init__(
            f"Content hash {content_hash!r} is already registered to job "
            f"{existing_job_id!r}"
        )


class JobStateStore:
    """An in-memory registry of :class:`ProcessingJob` records.

    Invariants maintained at all times:

    * Every job id in the store is unique (it is the dictionary key).
    * The content-hash index maps each distinct content hash to exactly one
      job id, and that job id always resolves to a stored job.
    """

    def __init__(self) -> None:
        # Primary store: job id -> job. The key set is the authoritative set of
        # issued identifiers used to guarantee uniqueness (Req 1.1).
        self._jobs: Dict[JobId, ProcessingJob] = {}
        # Dedup index: content hash -> job id (Req 1.5).
        self._by_content_hash: Dict[str, JobId] = {}

    # -- identifier allocation (Req 1.1) ---------------------------------

    def generate_job_id(self) -> JobId:
        """Return a UUID job id guaranteed unique across all jobs in the store.

        ``new_job_id`` draws a random UUIDv4; in the vanishingly rare case it
        collides with an id already issued, we draw again so the guarantee is
        absolute rather than probabilistic.
        """
        job_id = new_job_id()
        while job_id in self._jobs:  # pragma: no cover - collisions are ~impossible
            job_id = new_job_id()
        return job_id

    # -- creation --------------------------------------------------------

    def create_job(
        self,
        *,
        document_id: DocumentId,
        metadata: DocumentMetadata,
    ) -> ProcessingJob:
        """Create, persist, and return a fresh job for an accepted submission.

        A unique job id is allocated, a :class:`ProcessingJob` is constructed in
        its initial (queued, parsing) state, and the job is registered in both
        the primary store and the content-hash index.

        Raises:
            DuplicateContentHashError: if a job already exists for
                ``metadata.contentHash``. Callers performing dedup should look
                the hash up first via :meth:`find_by_content_hash`.
        """
        if not isinstance(metadata, DocumentMetadata):
            raise TypeError("metadata must be a DocumentMetadata")

        existing = self._by_content_hash.get(metadata.contentHash)
        if existing is not None:
            raise DuplicateContentHashError(metadata.contentHash, existing)

        job_id = self.generate_job_id()
        job = ProcessingJob(
            jobId=job_id,
            documentId=document_id,
            metadata=metadata,
        )
        self._jobs[job_id] = job
        self._by_content_hash[metadata.contentHash] = job_id
        return job

    # -- persistence of updates ------------------------------------------

    def save(self, job: ProcessingJob) -> ProcessingJob:
        """Persist a job record, inserting or replacing by job id.

        Used by the Orchestrator to write back stage-state transitions. The
        content-hash index is kept in sync with the saved job's metadata.
        """
        if not isinstance(job, ProcessingJob):
            raise TypeError("job must be a ProcessingJob")

        self._jobs[job.jobId] = job
        self._by_content_hash[job.metadata.contentHash] = job.jobId
        return job

    # -- lookups ---------------------------------------------------------

    def get(self, job_id: JobId) -> ProcessingJob:
        """Return the job for ``job_id`` or raise :class:`UnknownJobError`."""
        try:
            return self._jobs[job_id]
        except KeyError:
            raise UnknownJobError(job_id) from None

    def find_by_id(self, job_id: JobId) -> Optional[ProcessingJob]:
        """Return the job for ``job_id`` or ``None`` if absent (Req 1.1 lookup)."""
        return self._jobs.get(job_id)

    def find_by_content_hash(self, content_hash: str) -> Optional[ProcessingJob]:
        """Return the job registered for ``content_hash`` or ``None`` (Req 1.5)."""
        job_id = self._by_content_hash.get(content_hash)
        if job_id is None:
            return None
        return self._jobs.get(job_id)

    # -- introspection ---------------------------------------------------

    def contains_content_hash(self, content_hash: str) -> bool:
        """Return whether a job is registered for ``content_hash``."""
        return content_hash in self._by_content_hash

    def all_jobs(self) -> List[ProcessingJob]:
        """Return a snapshot list of every stored job."""
        return list(self._jobs.values())

    def __contains__(self, job_id: object) -> bool:
        return job_id in self._jobs

    def __len__(self) -> int:
        return len(self._jobs)

    def __iter__(self) -> Iterator[ProcessingJob]:
        return iter(self._jobs.values())
