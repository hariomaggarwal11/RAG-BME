"""Core identifier types for the pipeline.

Per the design:

* ``JobId`` is a UUID string that is unique across all Processing_Jobs (Req 1.1).
* ``DocumentId`` is a stable id derived from a document's content hash.
"""

from __future__ import annotations

import uuid

# Identifiers are represented as opaque strings. Type aliases document intent at
# call sites without imposing a wrapper class on every model field.
JobId = str
DocumentId = str


def new_job_id() -> JobId:
    """Generate a fresh, globally-unique job identifier (Req 1.1)."""
    return str(uuid.uuid4())


def document_id_from_hash(content_hash: str) -> DocumentId:
    """Derive a stable :data:`DocumentId` from a document's content hash.

    The same content hash always maps to the same document id, which is what
    makes content-addressed ingestion and dedup work (Req 1.5).
    """
    from ._validation import require_non_empty_str

    require_non_empty_str(content_hash, "content_hash")
    return content_hash
