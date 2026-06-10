"""Property test for Retriever result ordering and tie-break (Req 9.9).

Feature: biomedical-rag-pipeline, Property 25: Result ordering is by descending similarity with deterministic tie-break

Statement: results are ordered by non-increasing similarity score, and chunks
with equal similarity scores are ordered by ascending document identifier.
"""

from __future__ import annotations

from typing import List, Tuple

from hypothesis import given, settings
from hypothesis import strategies as st

from biomed_rag.config import PipelineConfig
from biomed_rag.embedding.mock import MockEmbeddingModel
from biomed_rag.models import Chunk, Embedding, VectorRecord
from biomed_rag.retrieval import QueryRequest, RetrievalStatus, Retriever
from biomed_rag.storage.in_memory import InMemoryVectorStore

_DIMENSION = 8

# A small content pool: reusing identical content across distinct documentIds
# yields identical mock embeddings (the mock derives its vector from the text),
# hence identical similarity scores. That forces ties so the deterministic
# ascending-documentId tie-break (Req 9.9) is actually exercised.
_CONTENT_POOL = ["alpha", "beta", "gamma", "delta"]


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


@st.composite
def _libraries(draw) -> List[Tuple[str, str]]:
    """Generate a library as a list of (documentId, content) pairs.

    Document ids are distinct (one record per document) and contents are drawn
    from a small pool so identical-content records across different documents
    tie on similarity.
    """
    doc_ids = draw(
        st.lists(
            st.text(alphabet="abcdefghij0123456789", min_size=1, max_size=6),
            min_size=1,
            max_size=20,
            unique=True,
        )
    )
    return [(doc_id, draw(st.sampled_from(_CONTENT_POOL))) for doc_id in doc_ids]


def _store_with(library: List[Tuple[str, str]]) -> InMemoryVectorStore:
    store = InMemoryVectorStore()
    for doc_id, content in library:
        store.upsert_batch(doc_id, [_make_record(doc_id, content)])
    return store


# Feature: biomedical-rag-pipeline, Property 25: Result ordering is by descending similarity with deterministic tie-break
@settings(max_examples=200)
@given(
    library=_libraries(),
    top_k=st.integers(min_value=1, max_value=100),
    query_text=st.text(min_size=1, max_size=64).filter(lambda s: s.strip() != ""),
)
def test_result_ordering_and_tie_break(
    library: List[Tuple[str, str]], top_k: int, query_text: str
) -> None:
    """Validates: Requirements 9.9"""
    store = _store_with(library)
    retriever = Retriever(
        store, MockEmbeddingModel(dimension=_DIMENSION), PipelineConfig()
    )

    result = retriever.retrieve(QueryRequest(text=query_text, topK=top_k))

    # A non-empty library with a valid query yields an OK result.
    assert result.status is RetrievalStatus.OK

    keys = [(c.similarity, c.documentId) for c in result.chunks]

    # Non-increasing similarity, ascending documentId tie-break: the sort key
    # (-similarity, documentId) must be non-decreasing across the sequence.
    sort_keys = [(-sim, doc_id) for sim, doc_id in keys]
    assert sort_keys == sorted(sort_keys)

    # Equivalently, the returned order matches sorting by (-similarity, docId).
    assert keys == sorted(keys, key=lambda k: (-k[0], k[1]))
