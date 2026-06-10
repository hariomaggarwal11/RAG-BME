"""Property-based test for complete document removal (Task 11.5).

Feature: biomedical-rag-pipeline, Property 21: Document removal is complete

Property statement (design.md, Property 21):
    For any document with stored records, removing it deletes all chunks and
    embeddings for its source document id, leaving none retrievable by that id.

**Validates: Requirements 8.5**

Strategy: generate a library of documents (each with a non-empty record set,
so every id is genuinely removable), store them in an ``InMemoryVectorStore``,
then delete one chosen document. The deleted id must return nothing via
``get_document`` and its chunk ids must be exactly the ones reported deleted;
every other document must remain byte-for-byte unaffected (per the design
Testing Strategy: vector store properties run against the in-memory adapter
for fast 100+ iteration runs).
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
    """Generate documents, each with a non-empty (hence removable) record set.

    Chunk ids are made unique within a document so the in-memory adapter keys
    each record distinctly, faithfully modelling a real stored record set.
    """
    document_ids = draw(st.lists(_ID_TEXT, unique=True, min_size=1, max_size=4))
    library: Library = {}
    for document_id in document_ids:
        count = draw(st.integers(min_value=1, max_value=5))
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


def _snapshot(store: InMemoryVectorStore, document_id: str):
    """Capture (chunkId -> vector) for a document, for exact comparison."""
    return {
        rec.chunk.chunkId: list(rec.embedding.vector)
        for rec in store.get_document(document_id)
    }


@st.composite
def _library_and_target(draw: st.DrawFn):
    library = draw(_library())
    target = draw(st.sampled_from(sorted(library.keys())))
    return library, target


@settings(max_examples=200, deadline=None)
@given(payload=_library_and_target())
def test_document_removal_is_complete(payload) -> None:
    # Feature: biomedical-rag-pipeline, Property 21: Document removal is complete
    library, target = payload

    store = InMemoryVectorStore()
    for document_id, records in library.items():
        store.upsert_batch(document_id, records)

    # Snapshot every other document so we can prove they are unaffected.
    others = {
        document_id: _snapshot(store, document_id)
        for document_id in library
        if document_id != target
    }
    expected_deleted = {rec.chunk.chunkId for rec in library[target]}

    result = store.delete_document(target)

    # The delete reports exactly the chunk ids that were stored under the id.
    assert set(result.deleted_chunk_ids) == expected_deleted
    assert result.document_id == target

    # Req 8.5: nothing for that source document id is retrievable afterwards.
    assert store.get_document(target) == []

    # Every other document remains byte-for-byte unaffected.
    for document_id, before in others.items():
        assert _snapshot(store, document_id) == before
