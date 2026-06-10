"""Property test for Retriever cardinality and score range (Req 9.1).

Feature: biomedical-rag-pipeline, Property 22: Retrieval cardinality and score range

Statement: for any non-empty Knowledge_Library and valid Query with requested
count K, the Retriever returns at most K chunks, each with a similarity score
in [0.0, 1.0].
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from biomed_rag.config import PipelineConfig
from biomed_rag.embedding.mock import MockEmbeddingModel
from biomed_rag.models import Chunk, Embedding, VectorRecord
from biomed_rag.retrieval import QueryRequest, RetrievalStatus, Retriever
from biomed_rag.storage.in_memory import InMemoryVectorStore

_DIMENSION = 8


def _make_record(document_id: str, content: str) -> VectorRecord:
    model = MockEmbeddingModel(dimension=_DIMENSION)
    chunk = Chunk(
        documentId=document_id,
        content=content,
        tokenCount=3,
        orderIndex=0,
        pageNumber=1,
    )
    vector = model.embed(content)
    embedding = Embedding(
        chunkId=chunk.chunkId, vector=vector, modelId=model.model_id()
    )
    return VectorRecord(documentId=document_id, chunk=chunk, embedding=embedding)


def _store_with(library_size: int) -> InMemoryVectorStore:
    store = InMemoryVectorStore()
    records = [
        _make_record(f"doc-{i}", content=f"content-{i}") for i in range(library_size)
    ]
    by_doc: dict[str, list[VectorRecord]] = {}
    for r in records:
        by_doc.setdefault(r.documentId, []).append(r)
    for doc_id, recs in by_doc.items():
        store.upsert_batch(doc_id, recs)
    return store


# Feature: biomedical-rag-pipeline, Property 22: Retrieval cardinality and score range
@settings(max_examples=200)
@given(
    library_size=st.integers(min_value=1, max_value=50),
    top_k=st.integers(min_value=1, max_value=100),
    query_text=st.text(min_size=1, max_size=64).filter(lambda s: s.strip() != ""),
)
def test_retrieval_cardinality_and_score_range(
    library_size: int, top_k: int, query_text: str
) -> None:
    """Validates: Requirements 9.1"""
    store = _store_with(library_size)
    retriever = Retriever(store, MockEmbeddingModel(dimension=_DIMENSION), PipelineConfig())

    result = retriever.retrieve(QueryRequest(text=query_text, topK=top_k))

    # A non-empty library with a valid query yields an OK result.
    assert result.status is RetrievalStatus.OK

    # Cardinality: at most K chunks are returned, and never more than the
    # library can supply.
    assert len(result.chunks) <= top_k
    assert len(result.chunks) <= library_size

    # Score range: every similarity lies within [0.0, 1.0].
    for chunk in result.chunks:
        assert 0.0 <= chunk.similarity <= 1.0
