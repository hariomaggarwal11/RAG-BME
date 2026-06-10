"""Property test for metadata filtering in the Retriever (Task 12.3).

Feature: biomedical-rag-pipeline, Property 23: Returned chunks satisfy the metadata filter

Exercises Req 9.7 and 9.8 against the deterministic ``InMemoryVectorStore`` and
``MockEmbeddingModel`` (per the design Testing Strategy: retrieval properties run
against in-memory adapters for fast 100+ iteration runs).

The property: for any query supplied with a metadata filter, every returned
chunk satisfies the filter predicate; and if no stored chunk satisfies the
filter, the result set is empty with status ``NO_MATCH``.
"""

from __future__ import annotations

from typing import List, Mapping, Optional, Tuple

from hypothesis import given, settings
from hypothesis import strategies as st

from biomed_rag.config import PipelineConfig
from biomed_rag.embedding.mock import MockEmbeddingModel
from biomed_rag.models import Chunk, Embedding, VectorRecord
from biomed_rag.retrieval import QueryRequest, RetrievalStatus, Retriever
from biomed_rag.storage.in_memory import InMemoryVectorStore

# Small fixed pools so generated filters have a realistic chance of matching
# (and of deterministically missing) the stored records.
_DOC_IDS = ["doc-a", "doc-b", "doc-c"]
_PAGE_NUMBERS = [1, 2, 3]

# A sentinel document id / page that is never assigned to any record, used to
# force the no-match (Req 9.8) branch.
_MISSING_DOC_ID = "doc-never-stored"
_MISSING_PAGE = 999

_DIMENSION = 8


def _make_record(document_id: str, page: Optional[int], order: int) -> VectorRecord:
    """Build a VectorRecord with the given source metadata and a unique chunk id."""
    model = MockEmbeddingModel(dimension=_DIMENSION)
    content = f"{document_id}-p{page}-{order}"
    chunk = Chunk(
        documentId=document_id,
        content=content,
        tokenCount=3,
        orderIndex=order,
        pageNumber=page,
        chunkId=f"{document_id}::chunk-{order}",
    )
    vector = model.embed(content)
    embedding = Embedding(
        chunkId=chunk.chunkId, vector=vector, modelId=model.model_id()
    )
    return VectorRecord(documentId=document_id, chunk=chunk, embedding=embedding)


def _record_matches(record: VectorRecord, filter: Mapping[str, object]) -> bool:
    """Mirror the store's filter predicate for the keys this test generates."""
    for key, expected in filter.items():
        if key == "documentId":
            if record.documentId != expected:
                return False
        elif key == "pageNumber":
            if record.chunk.pageNumber != expected:
                return False
        else:  # pragma: no cover - test only generates the keys above
            return False
    return True


@st.composite
def _library_and_filter(
    draw: st.DrawFn,
) -> Tuple[List[VectorRecord], Mapping[str, object]]:
    """Generate a non-empty record set plus a metadata filter.

    Records draw their documentId/pageNumber from small pools so a generated
    filter may match several, exactly one, or (via sentinel values) zero
    records, exercising both the satisfied (Req 9.7) and no-match (Req 9.8)
    branches.
    """
    count = draw(st.integers(min_value=1, max_value=8))
    records: List[VectorRecord] = []
    for order in range(count):
        document_id = draw(st.sampled_from(_DOC_IDS))
        page = draw(st.sampled_from(_PAGE_NUMBERS))
        records.append(_make_record(document_id, page, order))

    # Build a filter over documentId and/or pageNumber, occasionally using a
    # sentinel value guaranteed to match nothing.
    filter: dict = {}
    if draw(st.booleans()):
        filter["documentId"] = draw(
            st.sampled_from(_DOC_IDS + [_MISSING_DOC_ID])
        )
    if draw(st.booleans()) or not filter:
        filter["pageNumber"] = draw(
            st.sampled_from(_PAGE_NUMBERS + [_MISSING_PAGE])
        )
    return records, filter


@settings(max_examples=200, deadline=None)
@given(data=_library_and_filter())
def test_returned_chunks_satisfy_the_metadata_filter(
    data: Tuple[List[VectorRecord], Mapping[str, object]],
) -> None:
    # Feature: biomedical-rag-pipeline, Property 23: Returned chunks satisfy the metadata filter
    records, filter = data

    store = InMemoryVectorStore()
    by_doc: dict = {}
    for record in records:
        by_doc.setdefault(record.documentId, []).append(record)
    for document_id, recs in by_doc.items():
        store.upsert_batch(document_id, recs)

    retriever = Retriever(store, MockEmbeddingModel(dimension=_DIMENSION), PipelineConfig())
    # Request the maximum topK so the result is bounded only by what matches.
    result = retriever.retrieve(
        QueryRequest(text="probe query", topK=Retriever.MAX_TOP_K, filter=filter)
    )

    matching = [r for r in records if _record_matches(r, filter)]

    if not matching:
        # Req 9.8: a filter matching nothing yields an empty NO_MATCH result.
        assert result.status is RetrievalStatus.NO_MATCH
        assert result.chunks == []
    else:
        # Req 9.7: every returned chunk satisfies the filter predicate.
        assert result.status is RetrievalStatus.OK
        assert result.chunks
        matching_chunk_ids = {r.chunk.chunkId for r in matching}
        for chunk in result.chunks:
            assert chunk.chunkId in matching_chunk_ids
            if "documentId" in filter:
                assert chunk.documentId == filter["documentId"]
            if "pageNumber" in filter:
                assert chunk.pageNumber == filter["pageNumber"]
