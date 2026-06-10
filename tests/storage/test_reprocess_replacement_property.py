"""Property-based test for atomic reprocess replacement (Task 11.4).

Feature: biomedical-rag-pipeline, Property 20: Reprocess replacement is atomic

Property statement (design.md, Property 20):
    For any reprocessing of a document, the previously stored Chunks are
    replaced only after the newly generated Chunks are stored successfully; if
    the new storage does not complete, the previously stored Chunks remain
    unchanged.

**Validates: Requirements 8.4**

Strategy: generate a prior set of records and a new set of records for one
document (each :class:`Chunk` gets an auto-generated unique chunkId, so the two
sets never collide). Exercise both arms of the property against an
:class:`InMemoryVectorStore`:

* successful swap -- ``replace_document`` with no fault installed must leave
  exactly the new set and none of the prior chunk ids; and
* failing swap -- a ``commit_hook`` that raises must surface a
  ``PersistenceError`` and leave the prior set byte-for-byte unchanged.
"""

from __future__ import annotations

from typing import List

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from biomed_rag.models import Chunk, Embedding, VectorRecord
from biomed_rag.storage import InMemoryVectorStore, PersistenceError

_DOCUMENT_ID = "doc-under-reprocess"


def _record(document_id: str, order: int, vector: List[float]) -> VectorRecord:
    """Build a VectorRecord with an auto-generated unique chunkId."""
    chunk = Chunk(
        documentId=document_id,
        content=f"content-{order}",
        tokenCount=len(vector),
        orderIndex=order,
    )
    embedding = Embedding(chunkId=chunk.chunkId, vector=vector, modelId="mock-model")
    return VectorRecord(documentId=document_id, chunk=chunk, embedding=embedding)


# A small float vector; bounded values keep generation fast and valid.
_vectors = st.lists(
    st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
    min_size=1,
    max_size=4,
)

# A record set is modelled as a list of vectors -> one record per element. The
# prior set may be empty (first-time store), the new set must be non-empty so a
# successful swap leaves a document present and the two arms stay comparable.
_record_sets = st.builds(
    lambda vectors: [_record(_DOCUMENT_ID, i, v) for i, v in enumerate(vectors)],
    st.lists(_vectors, min_size=0, max_size=6),
)
_non_empty_record_sets = st.builds(
    lambda vectors: [_record(_DOCUMENT_ID, i, v) for i, v in enumerate(vectors)],
    st.lists(_vectors, min_size=1, max_size=6),
)


def _ids(records) -> set:
    return {rec.chunk.chunkId for rec in records}


def _snapshot(store: InMemoryVectorStore):
    """Capture (chunkId -> vector) for the document, for exact comparison."""
    return {
        rec.chunk.chunkId: list(rec.embedding.vector)
        for rec in store.get_document(_DOCUMENT_ID)
    }


@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
@given(prior=_record_sets, new=_non_empty_record_sets)
def test_reprocess_replacement_is_atomic(prior, new) -> None:
    # --- successful swap: new set fully replaces the prior set (Req 8.4) ---
    store = InMemoryVectorStore()
    if prior:
        store.upsert_batch(_DOCUMENT_ID, prior)
    prior_ids = _ids(prior)

    result = store.replace_document(_DOCUMENT_ID, new)

    assert result.replaced is bool(prior)
    stored_ids = _ids(store.get_document(_DOCUMENT_ID))
    # Only the new set remains; every prior-only id is gone.
    assert stored_ids == _ids(new)
    assert not (prior_ids - _ids(new)) & stored_ids

    # --- failing swap: prior set remains unchanged (Req 8.4 / 8.6) ---
    failing_store = InMemoryVectorStore()
    if prior:
        failing_store.upsert_batch(_DOCUMENT_ID, prior)
    before = _snapshot(failing_store)

    # Install a fault that raises mid-commit, after the new set is staged but
    # before it is installed.
    failing_store._commit_hook = lambda: (_ for _ in ()).throw(
        RuntimeError("simulated mid-write failure")
    )

    try:
        failing_store.replace_document(_DOCUMENT_ID, new)
        raise AssertionError("expected PersistenceError on a failed swap")
    except PersistenceError:
        pass

    # The prior records remain byte-for-byte unchanged; none of the new-only
    # ids leaked into the store.
    after = _snapshot(failing_store)
    assert after == before
    assert not (_ids(new) - prior_ids) & set(after)
