"""End-to-end integration tests for the full pipeline (Task 14.2).

These drive the assembled :class:`Pipeline` facade with its default deterministic
in-memory wiring (mock parsing engine, mock embedding model, in-memory vector
store) across whole-system scenarios that the single-document smoke test
(task 14.1) does not cover:

* a complete ``submit -> process -> retrieve`` round trip asserted at the
  storage boundary, confirming the job reaches ``JobCompleted`` once every chunk
  is stored (Req 8.3) and the orchestrator runs the stages in order (Req 10.1);
* several distinct documents ingested into one Knowledge_Library and retrieved
  independently, with a ``documentId`` metadata filter restricting results to a
  single source (Req 8.2, 9.7); and
* a reprocess/replace flow where re-storing a document's chunks atomically swaps
  the prior set for the new one — only the new chunks remain and are retrievable
  (Req 8.4).
"""

from __future__ import annotations

from biomed_rag.ingestion.service import Accepted, FileInput
from biomed_rag.models import Chunk, Embedding, OverallStatus, Stage, StageStatus, VectorRecord
from biomed_rag.orchestration import JobCompleted
from biomed_rag.pipeline import (
    DEFAULT_EMBEDDING_DIMENSION,
    DEFAULT_EMBEDDING_MODEL_ID,
    Pipeline,
)
from biomed_rag.embedding.mock import MockEmbeddingModel
from biomed_rag.retrieval.retriever import QueryRequest, RetrievalStatus

# Three well-formed HTML documents with distinct biomedical content. Distinct
# bytes give distinct content hashes, so none is rejected as a duplicate
# (Req 1.5) and each yields its own document identifier.
_HTML_HEART = (
    b"<html><body>\n"
    b"Heart tissue analysis of the left ventricle.\n\n"
    b"The ventricle contracts rhythmically during systole.\n"
    b"</body></html>"
)
_HTML_NEURON = (
    b"<html><body>\n"
    b"Neuronal signalling across the synaptic cleft.\n\n"
    b"Action potentials propagate along the myelinated axon.\n"
    b"</body></html>"
)
_HTML_BONE = (
    b"<html><body>\n"
    b"Cortical bone remodelling by osteoblasts and osteoclasts.\n\n"
    b"Trabecular density responds to mechanical loading.\n"
    b"</body></html>"
)


def _ingest(pipeline: Pipeline, filename: str, content: bytes) -> str:
    """Submit one document, asserting it is accepted, and return its job id."""
    result = pipeline.submit(FileInput(filename=filename, content=content))
    assert isinstance(result, Accepted)
    return result.jobId


def _process_to_completion(pipeline: Pipeline, job_id: str) -> JobCompleted:
    """Run every stage for ``job_id`` and assert it completes (Req 8.3, 10.1)."""
    outcome = pipeline.process(job_id)
    assert isinstance(outcome, JobCompleted)
    return outcome


def test_full_round_trip_completes_and_retrieves_with_source_metadata() -> None:
    """submit -> process (all stages) -> storage -> retrieve (Req 8.3, 10.1)."""
    pipeline = Pipeline()

    job_id = _ingest(pipeline, "heart.html", _HTML_HEART)
    outcome = _process_to_completion(pipeline, job_id)

    job = pipeline.job_store.get(job_id)
    document_id = job.documentId

    # Job completed only once every chunk was stored (Req 8.3) and every stage
    # ran in order to SUCCEEDED (Req 10.1).
    assert job.overallStatus is OverallStatus.COMPLETED
    assert len(outcome.storedChunkIds) >= 1
    for stage in Stage.ordered():
        assert job.stageStates[stage].status is StageStatus.SUCCEEDED

    # The reported stored chunk ids exactly match what the store holds for the
    # document identifier (Req 8.2): nothing dropped, nothing extra.
    stored = pipeline.vector_store.get_document(document_id)
    assert {record.chunk.chunkId for record in stored} == set(outcome.storedChunkIds)

    # Retrieval returns ranked chunks that carry their source metadata back to
    # the caller (Req 9.1, 9.4).
    result = pipeline.retrieve(QueryRequest(text="How does the ventricle contract?", topK=5))
    assert result.status is RetrievalStatus.OK
    assert len(result.chunks) >= 1
    stored_chunk_ids = {record.chunk.chunkId for record in stored}
    for chunk in result.chunks:
        assert chunk.chunkId in stored_chunk_ids
        assert chunk.content != ""
        assert chunk.documentId == document_id
        assert 0.0 <= chunk.similarity <= 1.0


def test_multiple_documents_are_independently_retrievable_with_filter() -> None:
    """Distinct documents coexist in one library; a documentId filter isolates
    one source (Req 8.2, 9.7)."""
    pipeline = Pipeline()

    heart_job = _ingest(pipeline, "heart.html", _HTML_HEART)
    neuron_job = _ingest(pipeline, "neuron.html", _HTML_NEURON)
    bone_job = _ingest(pipeline, "bone.html", _HTML_BONE)

    _process_to_completion(pipeline, heart_job)
    _process_to_completion(pipeline, neuron_job)
    _process_to_completion(pipeline, bone_job)

    heart_doc = pipeline.job_store.get(heart_job).documentId
    neuron_doc = pipeline.job_store.get(neuron_job).documentId
    bone_doc = pipeline.job_store.get(bone_job).documentId
    assert len({heart_doc, neuron_doc, bone_doc}) == 3

    # Each document's chunks are stored under its own identifier and disjoint.
    heart_chunks = {r.chunk.chunkId for r in pipeline.vector_store.get_document(heart_doc)}
    neuron_chunks = {r.chunk.chunkId for r in pipeline.vector_store.get_document(neuron_doc)}
    bone_chunks = {r.chunk.chunkId for r in pipeline.vector_store.get_document(bone_doc)}
    assert heart_chunks and neuron_chunks and bone_chunks
    assert heart_chunks.isdisjoint(neuron_chunks)
    assert heart_chunks.isdisjoint(bone_chunks)
    assert neuron_chunks.isdisjoint(bone_chunks)

    # An unfiltered query can surface chunks from more than one document.
    unfiltered = pipeline.retrieve(QueryRequest(text="biomedical tissue", topK=100))
    assert unfiltered.status is RetrievalStatus.OK
    assert {c.chunkId for c in unfiltered.chunks} == heart_chunks | neuron_chunks | bone_chunks

    # A documentId filter restricts results to exactly that source (Req 9.7).
    filtered = pipeline.retrieve(
        QueryRequest(text="biomedical tissue", topK=100, filter={"documentId": neuron_doc})
    )
    assert filtered.status is RetrievalStatus.OK
    assert len(filtered.chunks) >= 1
    assert all(c.documentId == neuron_doc for c in filtered.chunks)
    assert {c.chunkId for c in filtered.chunks} == neuron_chunks
    assert {c.chunkId for c in filtered.chunks}.isdisjoint(heart_chunks | bone_chunks)


def test_reprocess_replaces_prior_chunks_atomically() -> None:
    """Reprocessing a document atomically swaps its chunks: only the new set
    remains and is retrievable (Req 8.4)."""
    pipeline = Pipeline()

    job_id = _ingest(pipeline, "heart.html", _HTML_HEART)
    outcome = _process_to_completion(pipeline, job_id)
    document_id = pipeline.job_store.get(job_id).documentId

    original_chunk_ids = set(outcome.storedChunkIds)
    assert original_chunk_ids

    # Build a fresh chunk set for the SAME document, as a reprocess would. Its
    # embedding is produced by the same model family the retriever queries with,
    # so the new content is retrievable after the swap.
    model = MockEmbeddingModel(
        model_id=DEFAULT_EMBEDDING_MODEL_ID,
        dimension=DEFAULT_EMBEDDING_DIMENSION,
    )
    new_content = "Revised study of mitral valve regurgitation under load."
    new_chunk = Chunk(
        documentId=document_id,
        content=new_content,
        tokenCount=8,
        orderIndex=0,
        pageNumber=1,
    )
    new_record = VectorRecord(
        documentId=document_id,
        chunk=new_chunk,
        embedding=Embedding(
            chunkId=new_chunk.chunkId,
            vector=model.embed(new_content),
            modelId=DEFAULT_EMBEDDING_MODEL_ID,
        ),
    )

    swap = pipeline.vector_store.replace_document(document_id, [new_record])
    assert swap.replaced is True
    assert swap.stored_chunk_ids == [new_chunk.chunkId]

    # The atomic swap leaves only the new chunk under the identifier; every old
    # chunk is gone (Req 8.4).
    remaining = {r.chunk.chunkId for r in pipeline.vector_store.get_document(document_id)}
    assert remaining == {new_chunk.chunkId}
    assert remaining.isdisjoint(original_chunk_ids)

    # The replacement chunk is retrievable and none of the prior chunks are.
    result = pipeline.retrieve(QueryRequest(text=new_content, topK=5))
    assert result.status is RetrievalStatus.OK
    retrieved_ids = {c.chunkId for c in result.chunks}
    assert new_chunk.chunkId in retrieved_ids
    assert retrieved_ids.isdisjoint(original_chunk_ids)
