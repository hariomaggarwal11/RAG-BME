"""Smoke test for the full pipeline assembly (Task 14.1, Req 8.3, 10.1).

Drives a single document through the facade's three entry points using the
default deterministic in-memory wiring:

* :meth:`Pipeline.submit` accepts a document (Req 1),
* :meth:`Pipeline.process` runs every stage to completion (Req 8.3, 10.1), and
* :meth:`Pipeline.retrieve` returns stored chunks for a query (Req 9).

The full end-to-end integration tests (reprocess/replace, multiple documents)
are task 14.2 and out of scope here.
"""

from __future__ import annotations

from biomed_rag.ingestion.service import Accepted, Duplicate, FileInput
from biomed_rag.models import OverallStatus, Stage, StageStatus
from biomed_rag.orchestration import JobCompleted
from biomed_rag.pipeline import Pipeline
from biomed_rag.retrieval.retriever import QueryRequest, RetrievalStatus

# A minimal, well-formed HTML document that passes the ingestion gate and yields
# extractable text through the deterministic mock parsing engine.
_HTML = (
    b"<html><body>\n"
    b"Heart tissue analysis of the left ventricle.\n\n"
    b"The ventricle contracts rhythmically during systole.\n"
    b"</body></html>"
)


def _submit_html(pipeline: Pipeline) -> str:
    result = pipeline.submit(FileInput(filename="paper.html", content=_HTML))
    assert isinstance(result, Accepted)
    return result.jobId


def test_submit_process_retrieve_round_trip() -> None:
    pipeline = Pipeline()

    # submit -> accepted (Req 1).
    job_id = _submit_html(pipeline)

    # process -> completed with every stage succeeded (Req 8.3, 10.1).
    outcome = pipeline.process(job_id)
    assert isinstance(outcome, JobCompleted)
    assert outcome.jobId == job_id
    assert len(outcome.storedChunkIds) >= 1

    saved = pipeline.job_store.get(job_id)
    assert saved.overallStatus is OverallStatus.COMPLETED
    for stage in Stage.ordered():
        assert saved.stageStates[stage].status is StageStatus.SUCCEEDED

    # The stored chunks are addressable by document id (Req 8.2).
    stored = pipeline.vector_store.get_document(saved.documentId)
    assert {r.chunk.chunkId for r in stored} == set(outcome.storedChunkIds)

    # retrieve -> OK with at least one ranked chunk carrying source metadata.
    result = pipeline.retrieve("How does the ventricle contract?")
    assert result.status is RetrievalStatus.OK
    assert len(result.chunks) >= 1
    for chunk in result.chunks:
        assert 0.0 <= chunk.similarity <= 1.0
        assert chunk.documentId == saved.documentId


def test_retrieve_accepts_query_request() -> None:
    pipeline = Pipeline()
    job_id = _submit_html(pipeline)
    pipeline.process(job_id)

    result = pipeline.retrieve(QueryRequest(text="ventricle", topK=1))
    assert result.status is RetrievalStatus.OK
    assert len(result.chunks) == 1


def test_retrieve_on_empty_library_reports_status() -> None:
    pipeline = Pipeline()
    result = pipeline.retrieve("anything")
    assert result.status is RetrievalStatus.LIBRARY_EMPTY
    assert result.chunks == []


def test_duplicate_submission_returns_existing_job() -> None:
    pipeline = Pipeline()
    first = pipeline.submit(FileInput(filename="paper.html", content=_HTML))
    assert isinstance(first, Accepted)

    second = pipeline.submit(FileInput(filename="paper.html", content=_HTML))
    assert isinstance(second, Duplicate)
    assert second.existingJobId == first.jobId


def test_status_reports_completion_progress() -> None:
    pipeline = Pipeline()
    job_id = _submit_html(pipeline)
    pipeline.process(job_id)

    status = pipeline.status(job_id)
    assert status.currentStage is Stage.STORAGE
    assert status.progressPercent == 100
