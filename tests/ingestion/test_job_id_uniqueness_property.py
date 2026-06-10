"""Property-based test for job identifier uniqueness (Task 4.4).

Feature: biomedical-rag-pipeline, Property 1: Job identifier uniqueness

Property statement (design.md, Property 1):
    For any sequence of distinct document submissions, every Processing_Job
    created by the Ingestion_Service is assigned a job identifier that is
    distinct from all other assigned job identifiers.

**Validates: Requirements 1.1**

Strategy: generate sequences of distinct document payloads (distinct content
=> distinct SHA-256 hashes => no dedup), wrap each in a minimal well-formed PDF,
submit them through a single :class:`IngestionService`, and assert that every
:class:`Accepted` result carries a job id distinct from all others.
"""

from __future__ import annotations

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from biomed_rag.ingestion import Accepted, FileInput, IngestionService, JobStateStore


def _well_formed_pdf(payload: bytes) -> bytes:
    """Wrap ``payload`` into a minimal, well-formed PDF.

    The Ingestion_Service well-formedness check requires the ``%PDF-`` header
    and a ``%%EOF`` trailer; embedding the unique ``payload`` in the body keeps
    every generated document's bytes (and therefore its content hash) distinct.
    """
    return b"%PDF-1.4\n" + payload + b"\n%%EOF"


# Distinct content payloads => distinct content hashes => distinct submissions.
_distinct_payloads = st.lists(
    st.binary(min_size=0, max_size=32),
    min_size=0,
    max_size=25,
    unique=True,
)


@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
@given(payloads=_distinct_payloads)
def test_job_identifiers_are_unique_across_distinct_submissions(payloads) -> None:
    # A fresh store per example so each sequence starts from an empty id space.
    service = IngestionService(JobStateStore())

    job_ids = []
    for index, payload in enumerate(payloads):
        result = service.submit(
            FileInput(filename=f"doc_{index}.pdf", content=_well_formed_pdf(payload))
        )
        # Distinct content must be accepted (never deduplicated).
        assert isinstance(result, Accepted), f"expected Accepted, got {result!r}"
        job_ids.append(result.jobId)

    # Every created job carries an identifier distinct from all others.
    assert len(set(job_ids)) == len(job_ids)
    # And the store agrees: one stored job per distinct submission.
    assert len(service._store) == len(payloads)
