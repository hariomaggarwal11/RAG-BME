"""Property test for retrievability by document identifier (Task 11.3).

Feature: biomedical-rag-pipeline, Property 19: Stored embeddings are retrievable by document identifier

This exercises Req 8.2 against the deterministic ``InMemoryVectorStore`` adapter
(per the design Testing Strategy: vector store properties run against the
in-memory adapter for fast 100+ iteration runs).
"""

from __future__ import annotations

from typing import Dict, List

from hypothesis import given, settings
from hypothesis import strategies as st

from biomed_rag.models import Chunk, Embedding, VectorRecord
from biomed_rag.storage import InMemoryVectorStore

# A library is a mapping of documentId -> the list of records stored under it.
Library = Dict[str, List[VectorRecord]]

# Safe, non-empty identifier text (Chunk/VectorRecord require non-empty ids).
_ID_TEXT = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_",
    min_size=1,
    max_size=12,
)

# Finite floats keep the generated embedding vectors valid for ``Embedding``.
_VECTOR = st.lists(
    st.floats(allow_nan=False, allow_infinity=False, width=32),
    min_size=1,
    max_size=8,
)


@st.composite
def _library(draw: st.DrawFn) -> Library:
    """Generate a set of documents, each with a (possibly empty) record set.

    Chunk ids are made unique within a document so the in-memory adapter keys
    each record distinctly, faithfully modelling a real stored record set.
    """
    document_ids = draw(st.lists(_ID_TEXT, unique=True, min_size=1, max_size=4))
    library: Library = {}
    for document_id in document_ids:
        count = draw(st.integers(min_value=0, max_value=5))
        records: List[VectorRecord] = []
        for order in range(count):
            vector = draw(_VECTOR)
            chunk_id = f"{document_id}::chunk-{order}"
            chunk = Chunk(
                documentId=document_id,
                content=f"content-{order}",
                tokenCount=len(vector),
                orderIndex=order,
                chunkId=chunk_id,
            )
            embedding = Embedding(
                chunkId=chunk_id, vector=vector, modelId="mock-model"
            )
            records.append(
                VectorRecord(documentId=document_id, chunk=chunk, embedding=embedding)
            )
        library[document_id] = records
    return library


@settings(max_examples=200, deadline=None)
@given(library=_library())
def test_stored_embeddings_are_retrievable_by_document_identifier(
    library: Library,
) -> None:
    # Feature: biomedical-rag-pipeline, Property 19: Stored embeddings are retrievable by document identifier
    store = InMemoryVectorStore()
    for document_id, records in library.items():
        store.upsert_batch(document_id, records)

    for document_id, records in library.items():
        retrieved = store.get_document(document_id)
        expected_ids = {rec.chunk.chunkId for rec in records}
        retrieved_ids = [rec.chunk.chunkId for rec in retrieved]

        # No duplicates leaked in.
        assert len(retrieved_ids) == len(set(retrieved_ids))
        # Exactly the records stored under this id: no more, no fewer.
        assert set(retrieved_ids) == expected_ids
        # The returned records are the very ones stored (full equality), and no
        # record from another document leaked across the identifier boundary.
        assert {id(r) for r in retrieved} == {id(r) for r in records} or all(
            r in records for r in retrieved
        )
        assert all(r.documentId == document_id for r in retrieved)

    # An identifier that was never stored returns nothing.
    unseen = "::".join(library.keys()) + "::never-stored"
    assert store.get_document(unseen) == []
