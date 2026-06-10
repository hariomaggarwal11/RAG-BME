"""Unit tests for IngestionService.submit and its ordered validation gate (Task 4.2).

Covers the gate's per-step rejections, the accept path with UTC metadata
recording, and content-hash dedup:
* filename present and <= 255 chars  - Requirement 1.8
* byte size in [1, 500 MB]           - Requirements 1.2, 1.6, 1.7
* format by content sniffing         - Requirement 1.3
* well-formed-file check             - Requirement 1.7
* SHA-256 dedup                       - Requirements 1.4, 1.5
* job creation + unique id           - Requirement 1.1
"""

from __future__ import annotations

import hashlib
import io
import zipfile
from datetime import timezone

import pytest

from biomed_rag.ingestion import (
    Accepted,
    Duplicate,
    FileInput,
    IngestionService,
    JobStateStore,
    Rejected,
    RejectionCode,
)
from biomed_rag.models import Format


# -- fixtures: minimal well-formed documents per format --------------------


def _pdf_bytes() -> bytes:
    return b"%PDF-1.4\n1 0 obj<<>>endobj\ntrailer<<>>\n%%EOF"


def _html_bytes() -> bytes:
    return b"<!DOCTYPE html><html><head><title>t</title></head><body>hi</body></html>"


def _epub_bytes() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        # OCF requires the mimetype entry be stored first, uncompressed.
        zf.writestr(
            zipfile.ZipInfo("mimetype"),
            "application/epub+zip",
            compress_type=zipfile.ZIP_STORED,
        )
        zf.writestr(
            "META-INF/container.xml",
            "<?xml version='1.0'?><container/>",
        )
        zf.writestr("OEBPS/content.opf", "<package/>")
    return buf.getvalue()


def _docx_bytes() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("[Content_Types].xml", "<Types/>")
        zf.writestr("word/document.xml", "<document/>")
    return buf.getvalue()


@pytest.fixture
def service() -> IngestionService:
    return IngestionService(JobStateStore())


# -- accept path -----------------------------------------------------------


class TestAcceptPath:
    @pytest.mark.parametrize(
        "filename,maker,expected_format",
        [
            ("paper.pdf", _pdf_bytes, Format.PDF),
            ("book.epub", _epub_bytes, Format.EPUB),
            ("article.docx", _docx_bytes, Format.DOCX),
            ("page.html", _html_bytes, Format.HTML),
        ],
    )
    def test_supported_formats_accepted(
        self, service, filename, maker, expected_format
    ) -> None:
        data = maker()
        result = service.submit(FileInput(filename=filename, content=data))
        assert isinstance(result, Accepted)

        job = service._store.get(result.jobId)
        assert job.metadata.filename == filename
        assert job.metadata.format is expected_format
        assert job.metadata.byteSize == len(data)
        assert job.metadata.contentHash == hashlib.sha256(data).hexdigest()
        # Metadata timestamp recorded in UTC (Req 1.4).
        assert job.metadata.submittedAtUtc.tzinfo is not None
        assert job.metadata.submittedAtUtc.utcoffset() == timezone.utc.utcoffset(None)

    def test_distinct_documents_get_distinct_job_ids(self, service) -> None:
        r1 = service.submit(FileInput("a.pdf", _pdf_bytes()))
        r2 = service.submit(FileInput("b.html", _html_bytes()))
        assert isinstance(r1, Accepted) and isinstance(r2, Accepted)
        assert r1.jobId != r2.jobId


# -- dedup -----------------------------------------------------------------


class TestDedup:
    def test_byte_identical_resubmission_returns_existing_id(self, service) -> None:
        data = _pdf_bytes()
        first = service.submit(FileInput("first.pdf", data))
        assert isinstance(first, Accepted)
        # Same bytes, different filename -> still a duplicate by content hash.
        again = service.submit(FileInput("renamed.pdf", data))
        assert isinstance(again, Duplicate)
        assert again.existingJobId == first.jobId
        assert len(service._store) == 1


# -- rejection gate (order matters) ----------------------------------------


class TestRejectionGate:
    def test_empty_filename_rejected(self, service) -> None:
        result = service.submit(FileInput("", _pdf_bytes()))
        assert isinstance(result, Rejected)
        assert result.code is RejectionCode.INVALID_FILENAME

    def test_overlong_filename_rejected(self, service) -> None:
        result = service.submit(FileInput("x" * 256, _pdf_bytes()))
        assert isinstance(result, Rejected)
        assert result.code is RejectionCode.INVALID_FILENAME

    def test_empty_file_rejected(self, service) -> None:
        result = service.submit(FileInput("empty.pdf", b""))
        assert isinstance(result, Rejected)
        assert result.code is RejectionCode.EMPTY_FILE

    def test_oversized_file_rejected(self, service) -> None:
        # Patch config bound via a small max to avoid allocating 500 MB.
        from biomed_rag.config import PipelineConfig

        svc = IngestionService(JobStateStore(), PipelineConfig(max_file_size_bytes=10))
        result = svc.submit(FileInput("big.pdf", _pdf_bytes()))
        assert isinstance(result, Rejected)
        assert result.code is RejectionCode.FILE_TOO_LARGE

    def test_unsupported_format_rejected(self, service) -> None:
        png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64
        result = service.submit(FileInput("img.png", png))
        assert isinstance(result, Rejected)
        assert result.code is RejectionCode.UNSUPPORTED_FORMAT
        assert "PNG" in result.message

    def test_corrupt_pdf_rejected(self, service) -> None:
        # Looks like a PDF (header) but is truncated: missing %%EOF trailer.
        result = service.submit(FileInput("bad.pdf", b"%PDF-1.4\nbroken"))
        assert isinstance(result, Rejected)
        assert result.code is RejectionCode.CORRUPTED_FILE

    def test_corrupt_zip_docx_rejected(self, service) -> None:
        # ZIP magic but truncated archive body.
        result = service.submit(
            FileInput("bad.docx", b"PK\x03\x04word/document.xml broken")
        )
        assert isinstance(result, Rejected)
        # Either unsupported (cannot determine subformat) or corrupted is fine;
        # the point is no job is created.
        assert result.code in (
            RejectionCode.CORRUPTED_FILE,
            RejectionCode.UNSUPPORTED_FORMAT,
        )
        assert len(service._store) == 0

    def test_filename_checked_before_format(self, service) -> None:
        # Empty filename with unsupported bytes -> filename rejection wins.
        result = service.submit(FileInput("", b"\x89PNG\r\n\x1a\n"))
        assert isinstance(result, Rejected)
        assert result.code is RejectionCode.INVALID_FILENAME
