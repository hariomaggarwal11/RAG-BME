"""Property-based test for content-hash deduplication idempotency (Task 4.5).

Feature: biomedical-rag-pipeline, Property 3: Content-hash deduplication is idempotent

Property statement (design.md, Property 3):
    For any document, submitting a byte-identical document a second time
    returns the originally assigned job identifier and creates no new
    Processing_Job, while recorded metadata for an accepted submission equals
    the values derived from its input.

**Validates: Requirements 1.4, 1.5**

Strategy
--------
Generate a well-formed document across every supported format together with the
ground-truth format the content sniffer should detect. For each document:

* Submit it once and assert it is :class:`Accepted`. Independently derive the
  expected metadata (filename, format, byte size, SHA-256 content hash, derived
  document id) from the input and assert the recorded metadata equals it,
  exercising the "recorded metadata equals values derived from input" half of
  the property (Req 1.4).
* Re-submit the byte-identical content one or more times -- under arbitrary
  (possibly different) filenames -- and assert each result is a
  :class:`Duplicate` carrying the originally assigned job id, with the store's
  job count never growing past one, exercising the idempotent-dedup half
  (Req 1.5).
"""

from __future__ import annotations

import hashlib
import io
import zipfile

from hypothesis import given, settings
from hypothesis import strategies as st

from biomed_rag.ingestion import (
    Accepted,
    Duplicate,
    FileInput,
    IngestionService,
    JobStateStore,
)
from biomed_rag.models import Format, document_id_from_hash

_MAX_FILENAME_LENGTH = 255


# ---------------------------------------------------------------------------
# Well-formed document builders, one per supported format. Each embeds a unique
# `body` so distinct draws produce distinct content hashes.
# ---------------------------------------------------------------------------


def _build_pdf(body: bytes) -> bytes:
    return b"%PDF-1.4\n" + body + b"\n%%EOF"


def _build_html(body: bytes) -> bytes:
    return b"<!DOCTYPE html><html><body>" + body + b"</body></html>"


def _build_epub(body: bytes) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(
            zipfile.ZipInfo("mimetype"),
            "application/epub+zip",
            compress_type=zipfile.ZIP_STORED,
        )
        zf.writestr("META-INF/container.xml", "<?xml version='1.0'?><container/>")
        zf.writestr("OEBPS/content.opf", "<package/>" + body.hex())
    return buf.getvalue()


def _build_docx(body: bytes) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("[Content_Types].xml", "<Types/>")
        zf.writestr("word/document.xml", "<document/>" + body.hex())
    return buf.getvalue()


@st.composite
def _well_formed_document(draw):
    """Draw a (content_bytes, expected_format) pair for a supported document."""
    kind = draw(st.sampled_from(["pdf", "html", "epub", "docx"]))
    body = draw(st.binary(min_size=0, max_size=128))
    if kind == "pdf":
        return _build_pdf(body), Format.PDF
    if kind == "html":
        return _build_html(body), Format.HTML
    if kind == "epub":
        return _build_epub(body), Format.EPUB
    return _build_docx(body), Format.DOCX


# Filenames are valid (1..255 chars) so the only difference between an original
# submission and a resubmission is allowed to be the name -- dedup must key on
# content, never the filename.
_valid_filename = st.text(min_size=1, max_size=_MAX_FILENAME_LENGTH)


@settings(max_examples=200, deadline=None)
@given(
    document=_well_formed_document(),
    original_name=_valid_filename,
    resubmit_names=st.lists(_valid_filename, min_size=1, max_size=4),
)
def test_dedup_is_idempotent_and_metadata_matches_input(
    document, original_name, resubmit_names
) -> None:
    content, expected_format = document
    store = JobStateStore()
    service = IngestionService(store)

    # -- first submission is accepted (Req 1.1) ---------------------------
    first = service.submit(FileInput(filename=original_name, content=content))
    assert isinstance(first, Accepted), f"expected Accepted, got {first!r}"
    original_job_id = first.jobId

    # -- recorded metadata equals values derived from input (Req 1.4) -----
    expected_hash = hashlib.sha256(content).hexdigest()
    job = store.get(original_job_id)
    assert job.metadata.filename == original_name
    assert job.metadata.format is expected_format
    assert job.metadata.byteSize == len(content)
    assert job.metadata.contentHash == expected_hash
    assert job.documentId == document_id_from_hash(expected_hash)
    # Submission timestamp is recorded in UTC.
    assert job.metadata.submittedAtUtc.tzinfo is not None
    assert job.metadata.submittedAtUtc.utcoffset() is not None

    # -- byte-identical resubmissions are idempotent duplicates (Req 1.5) --
    for name in resubmit_names:
        again = service.submit(FileInput(filename=name, content=content))
        assert isinstance(again, Duplicate), f"expected Duplicate, got {again!r}"
        # Returns the originally assigned job id, regardless of filename.
        assert again.existingJobId == original_job_id
        # No new Processing_Job is ever created.
        assert len(store) == 1
