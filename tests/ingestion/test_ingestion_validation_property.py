"""Property-based test for ingestion validation totality and bound-correctness.

Feature: biomedical-rag-pipeline, Property 2: Ingestion validation is total and bound-correct

This test exercises Property 2 from the design's Correctness Properties:

    *For any* submission, the Ingestion_Service accepts it if and only if the
    filename length is in [1, 255], the byte size is in [1, 524,288,000], and
    the format is one of {PDF, EPUB, DOCX, HTML} and well-formed; otherwise it
    is rejected with the corresponding error and no Processing_Job is created.

**Validates: Requirements 1.2, 1.3, 1.6, 1.7, 1.8**

Strategy
--------
Each generated submission carries *known ground truth* about itself:

* the filename and whether its length is valid (1..255 chars),
* the document bytes,
* the format the gate's content sniffer should detect (or ``None`` when the
  content is not a supported format), and
* whether those bytes are a complete, well-formed instance of that format.

Because the generator constructs the bytes deliberately, the expected outcome
is computed independently of the implementation by replaying the design's
*ordered* validation gate (filename -> size -> format -> well-formedness). The
test then asserts the service's actual result matches that expectation, that
the accept/reject decision is exactly the bound-correct "iff", and that no
Processing_Job is ever created on rejection.
"""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from typing import Optional

from hypothesis import given, settings
from hypothesis import strategies as st

from biomed_rag.config import PipelineConfig
from biomed_rag.ingestion import (
    Accepted,
    FileInput,
    IngestionService,
    JobStateStore,
    Rejected,
    RejectionCode,
)
from biomed_rag.models import Format

# The design's fixed bounds (Configuration Model table).
_MAX_FILENAME_LENGTH = 255
_MAX_FILE_SIZE_BYTES = 524_288_000  # 500 MB


# ---------------------------------------------------------------------------
# A generated submission together with its known ground truth.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Submission:
    filename: str
    content: bytes
    # The format the content sniffer should detect, or None if unsupported.
    sniffed_format: Optional[Format]
    # Whether `content` is a complete, well-formed instance of `sniffed_format`.
    well_formed: bool


# ---------------------------------------------------------------------------
# Content builders: each returns (bytes, sniffed_format, well_formed).
# ---------------------------------------------------------------------------


def _build_pdf(body: bytes) -> Submission:
    content = b"%PDF-1.4\n" + body + b"\n%%EOF"
    return Submission("", content, Format.PDF, True)


def _build_html(body: bytes) -> Submission:
    content = b"<!DOCTYPE html><html><body>" + body + b"</body></html>"
    return Submission("", content, Format.HTML, True)


def _build_epub(extra: bytes) -> Submission:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(
            zipfile.ZipInfo("mimetype"),
            "application/epub+zip",
            compress_type=zipfile.ZIP_STORED,
        )
        zf.writestr("META-INF/container.xml", "<?xml version='1.0'?><container/>")
        zf.writestr("OEBPS/content.opf", "<package/>" + extra.hex())
    return Submission("", buf.getvalue(), Format.EPUB, True)


def _build_docx(extra: bytes) -> Submission:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("[Content_Types].xml", "<Types/>")
        zf.writestr("word/document.xml", "<document/>" + extra.hex())
    return Submission("", buf.getvalue(), Format.DOCX, True)


def _build_corrupt_pdf() -> Submission:
    # PDF header present (sniffs as PDF) but no %%EOF trailer -> not well-formed.
    return Submission("", b"%PDF-1.4\nbroken content with no trailer", Format.PDF, False)


def _build_corrupt_html() -> Submission:
    # Looks like HTML but is missing the closing </html> tag -> not well-formed.
    return Submission("", b"<!DOCTYPE html><html><body>truncated", Format.HTML, False)


def _build_corrupt_docx() -> Submission:
    # ZIP magic + a word/document.xml marker -> sniffs DOCX, but the archive
    # body is truncated so it is not a well-formed ZIP.
    return Submission(
        "", b"PK\x03\x04word/document.xml truncated archive", Format.DOCX, False
    )


# Unsupported content: deterministic blobs the sniffer must map to None.
_UNSUPPORTED_BLOBS = [
    b"\x89PNG\r\n\x1a\n" + b"\x00" * 32,  # PNG image
    b"\xff\xd8\xff\xe0" + b"\x00" * 32,  # JPEG image
    b"GIF89a" + b"\x00" * 32,  # GIF image
    b"{\\rtf1\\ansi truncated rtf}",  # RTF document
    b"just some plain prose, definitely not markup at all",  # plain text
    b"\x1f\x8b\x08\x00gzip-ish-bytes",  # gzip archive
]


@st.composite
def _supported_content(draw) -> Submission:
    kind = draw(st.sampled_from(["pdf", "html", "epub", "docx"]))
    body = draw(st.binary(min_size=0, max_size=128))
    if kind == "pdf":
        return _build_pdf(body)
    if kind == "html":
        return _build_html(body)
    if kind == "epub":
        return _build_epub(body)
    return _build_docx(body)


@st.composite
def _content(draw) -> Submission:
    category = draw(
        st.sampled_from(
            ["supported", "empty", "unsupported", "corrupt_pdf", "corrupt_html", "corrupt_docx"]
        )
    )
    if category == "supported":
        return draw(_supported_content())
    if category == "empty":
        return Submission("", b"", None, False)
    if category == "unsupported":
        return Submission("", draw(st.sampled_from(_UNSUPPORTED_BLOBS)), None, False)
    if category == "corrupt_pdf":
        return _build_corrupt_pdf()
    if category == "corrupt_html":
        return _build_corrupt_html()
    return _build_corrupt_docx()


@st.composite
def _filename(draw) -> str:
    category = draw(st.sampled_from(["valid", "empty", "too_long"]))
    if category == "valid":
        return draw(st.text(min_size=1, max_size=_MAX_FILENAME_LENGTH))
    if category == "empty":
        return ""
    return draw(st.text(min_size=_MAX_FILENAME_LENGTH + 1, max_size=_MAX_FILENAME_LENGTH + 64))


@st.composite
def _submission(draw) -> Submission:
    content = draw(_content())
    filename = draw(_filename())
    return Submission(filename, content.content, content.sniffed_format, content.well_formed)


@st.composite
def _case(draw):
    submission = draw(_submission())
    # Vary the configured size bound across a tiny regime (exercises the
    # FILE_TOO_LARGE branch with real well-formed bytes) and the full 500 MB
    # default (exercises the lower bound only).
    max_size = draw(st.one_of(st.just(_MAX_FILE_SIZE_BYTES), st.integers(min_value=1, max_value=2000)))
    return submission, max_size


def _expected_rejection_code(sub: Submission, max_size: int) -> Optional[RejectionCode]:
    """Replay the design's ordered gate; return the code, or None to accept."""
    if len(sub.filename) < 1 or len(sub.filename) > _MAX_FILENAME_LENGTH:
        return RejectionCode.INVALID_FILENAME
    if len(sub.content) < 1:
        return RejectionCode.EMPTY_FILE
    if len(sub.content) > max_size:
        return RejectionCode.FILE_TOO_LARGE
    if sub.sniffed_format is None:
        return RejectionCode.UNSUPPORTED_FORMAT
    if not sub.well_formed:
        return RejectionCode.CORRUPTED_FILE
    return None  # all gates pass -> accept


@settings(max_examples=300, deadline=None)
@given(_case())
def test_ingestion_validation_is_total_and_bound_correct(case) -> None:
    submission, max_size = case
    store = JobStateStore()
    service = IngestionService(store, PipelineConfig(max_file_size_bytes=max_size))

    expected_code = _expected_rejection_code(submission, max_size)
    result = service.submit(
        FileInput(filename=submission.filename, content=submission.content)
    )

    if expected_code is None:
        # iff (accept): all bound conditions satisfied -> a job is created.
        assert isinstance(result, Accepted), (
            f"expected acceptance but got {result!r}"
        )
        assert len(store) == 1
        job = store.get(result.jobId)
        assert job.metadata.byteSize == len(submission.content)
        assert job.metadata.format is submission.sniffed_format
    else:
        # iff (reject): at least one bound violated -> rejected with the
        # corresponding error and no Processing_Job created.
        assert isinstance(result, Rejected), (
            f"expected rejection {expected_code} but got {result!r}"
        )
        assert result.code is expected_code, (
            f"expected {expected_code} but got {result.code} "
            f"(filename_len={len(submission.filename)}, size={len(submission.content)}, "
            f"fmt={submission.sniffed_format}, well_formed={submission.well_formed}, "
            f"max_size={max_size})"
        )
        assert len(store) == 0
