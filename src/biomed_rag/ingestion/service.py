"""Ingestion_Service with the ordered validation gate (Req 1).

The :class:`IngestionService` is the pipeline's front door. It accepts a
submitted :class:`FileInput`, runs it through a strictly ordered validation
gate, and either registers a new :class:`~biomed_rag.models.ProcessingJob`
(returning its identifier), reports a duplicate submission (returning the
originally assigned identifier), or rejects the submission with a distinct
error code and message.

Validation gate, applied in order (design: Ingestion_Service section). Each
step produces its own distinct rejection so callers can tell failures apart:

1. **Filename present and ``<= 255`` chars** (Req 1.8).
2. **Byte size in ``[1, 500 MB]``** (Req 1.2, 1.6, and the empty-file case of
   Req 1.7).
3. **Format in {PDF, EPUB, DOCX, HTML} via content sniffing**, not just the
   extension (Req 1.3).
4. **Well-formed-file check** -- the bytes are a complete instance of the
   sniffed format (Req 1.7 corruption case).
5. **SHA-256 dedup** -- a byte-identical resubmission returns the existing job
   identifier and creates no new job (Req 1.5).

On success the service computes the SHA-256 content hash, derives the stable
:data:`~biomed_rag.models.DocumentId`, records metadata
``{filename, format, byteSize, contentHash, submittedAtUtc}`` in UTC (Req 1.4),
creates a :class:`ProcessingJob` with a unique identifier (Req 1.1), and returns
that identifier.
"""

from __future__ import annotations

import hashlib
import io
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Optional, Union

from ..config import PipelineConfig
from ..models import (
    DocumentMetadata,
    Format,
    JobId,
    document_id_from_hash,
)
from .job_state_store import JobStateStore

# 500 MB inclusive upper bound, surfaced in the user-facing message (Req 1.6).
_MAX_BYTE_SIZE = 524_288_000
_FIVE_HUNDRED_MB_LABEL = "500 MB"


# ---------------------------------------------------------------------------
# Inputs and results
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FileInput:
    """A submitted document.

    ``declaredFormat`` (e.g. derived from a file extension) is advisory only;
    the gate determines the true format by sniffing ``content`` so a misleading
    extension cannot smuggle in an unsupported or corrupt file (Req 1.3).
    """

    filename: str
    content: bytes
    declaredFormat: Optional[Format] = None


class RejectionCode(Enum):
    """Distinct rejection codes, one per failure case of the gate (Req 1.3, 1.6, 1.7, 1.8)."""

    INVALID_FILENAME = "invalid_filename"  # Req 1.8
    EMPTY_FILE = "empty_file"  # Req 1.7 (0-byte case)
    FILE_TOO_LARGE = "file_too_large"  # Req 1.6
    UNSUPPORTED_FORMAT = "unsupported_format"  # Req 1.3
    CORRUPTED_FILE = "corrupted_file"  # Req 1.7 (malformed case)


@dataclass(frozen=True)
class Accepted:
    """A new :class:`ProcessingJob` was created (Req 1.1)."""

    jobId: JobId


@dataclass(frozen=True)
class Duplicate:
    """The submission matched a previously ingested document (Req 1.5)."""

    existingJobId: JobId


@dataclass(frozen=True)
class Rejected:
    """The submission failed the validation gate (Req 1.3, 1.6, 1.7, 1.8)."""

    code: RejectionCode
    message: str


IngestionResult = Union[Accepted, Duplicate, Rejected]


# ---------------------------------------------------------------------------
# Content sniffing helpers
# ---------------------------------------------------------------------------

_ZIP_MAGICS = (b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08")
_BOM = b"\xef\xbb\xbf"


def _has_zip_magic(data: bytes) -> bool:
    return any(data.startswith(magic) for magic in _ZIP_MAGICS)


def _looks_like_html(data: bytes) -> bool:
    """Best-effort HTML detection from leading text content (Req 1.3)."""
    head = data
    if head.startswith(_BOM):
        head = head[len(_BOM):]
    head = head.lstrip()
    lowered = head[:1024].lower()
    return (
        lowered.startswith(b"<!doctype html")
        or lowered.startswith(b"<html")
        or b"<html" in lowered
    )


def _sniff_zip_subformat(data: bytes) -> Optional[Format]:
    """Distinguish an EPUB from a DOCX (or neither) for a ZIP container.

    EPUB's OCF layout stores an uncompressed ``mimetype`` entry as the very
    first archive member, so the literal bytes ``mimetypeapplication/epub+zip``
    appear near the start; DOCX (OOXML) always carries a ``word/document.xml``
    member. These markers are detectable in the raw bytes without a full unzip,
    which keeps detection separate from the well-formedness check.
    """
    if b"mimetypeapplication/epub+zip" in data[:4096] or b"META-INF/container.xml" in data:
        return Format.EPUB
    if b"word/document.xml" in data:
        return Format.DOCX
    return None


def _sniff_format(data: bytes) -> Optional[Format]:
    """Return the supported :class:`Format` detected from ``data`` or ``None``.

    ``None`` means the content is not one of the supported formats (Req 1.3).
    """
    if data.startswith(b"%PDF-"):
        return Format.PDF
    if _has_zip_magic(data):
        return _sniff_zip_subformat(data)
    if _looks_like_html(data):
        return Format.HTML
    return None


def _describe_detected(data: bytes) -> str:
    """Produce a short label for the detected (unsupported) content (Req 1.3)."""
    prefixes = {
        b"\x89PNG\r\n\x1a\n": "PNG image",
        b"\xff\xd8\xff": "JPEG image",
        b"GIF87a": "GIF image",
        b"GIF89a": "GIF image",
        b"%!PS": "PostScript",
        b"{\\rtf": "RTF document",
        b"\x1f\x8b": "gzip archive",
        b"\x50\x4b": "ZIP-based archive (not EPUB or DOCX)",
    }
    for prefix, label in prefixes.items():
        if data.startswith(prefix):
            return label
    return "an unrecognized/unsupported format"


def _is_well_formed(data: bytes, fmt: Format) -> bool:
    """Return whether ``data`` is a complete, well-formed instance of ``fmt`` (Req 1.7)."""
    if fmt is Format.PDF:
        # A complete PDF carries the %PDF- header and a closing %%EOF trailer.
        return data.startswith(b"%PDF-") and b"%%EOF" in data[-2048:]
    if fmt in (Format.EPUB, Format.DOCX):
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                if zf.testzip() is not None:
                    return False
                names = set(zf.namelist())
        except (zipfile.BadZipFile, OSError):
            return False
        if fmt is Format.DOCX:
            return "word/document.xml" in names and "[Content_Types].xml" in names
        # EPUB: OCF requires the mimetype entry and a container manifest.
        return "mimetype" in names or "META-INF/container.xml" in names
    if fmt is Format.HTML:
        # Require a closing root tag so a truncated fragment is rejected.
        return b"</html>" in data.lower()
    return False  # pragma: no cover - all supported formats handled above


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class IngestionService:
    """Accept, validate, and register submitted documents (Req 1)."""

    def __init__(
        self,
        store: JobStateStore,
        config: Optional[PipelineConfig] = None,
    ) -> None:
        if not isinstance(store, JobStateStore):
            raise TypeError("store must be a JobStateStore")
        self._store = store
        self._config = config if config is not None else PipelineConfig()

    def submit(self, file: FileInput) -> IngestionResult:
        """Run the ordered validation gate and register the document.

        Returns:
            * :class:`Accepted` carrying the new job id on success (Req 1.1).
            * :class:`Duplicate` carrying the existing job id for a byte-identical
              resubmission (Req 1.5).
            * :class:`Rejected` carrying a distinct code/message for each gate
              failure (Req 1.3, 1.6, 1.7, 1.8).
        """
        if not isinstance(file, FileInput):
            raise TypeError("file must be a FileInput")

        max_filename = self._config.max_filename_length
        max_size = self._config.max_file_size_bytes

        # 1. Filename present and within the length limit (Req 1.8).
        filename = file.filename
        if not isinstance(filename, str) or len(filename) == 0:
            return Rejected(
                RejectionCode.INVALID_FILENAME,
                "filename is missing; a filename of 1 to "
                f"{max_filename} characters is required",
            )
        if len(filename) > max_filename:
            return Rejected(
                RejectionCode.INVALID_FILENAME,
                f"filename exceeds the {max_filename}-character limit "
                f"(got {len(filename)} characters)",
            )

        # 2. Byte size in [1, 500 MB] (Req 1.2, 1.6, and empty case of 1.7).
        data = file.content
        if not isinstance(data, (bytes, bytearray)):
            raise TypeError("file.content must be bytes")
        data = bytes(data)
        size = len(data)
        if size < 1:
            return Rejected(
                RejectionCode.EMPTY_FILE,
                "document is empty (0 bytes) and cannot be processed",
            )
        if size > max_size:
            return Rejected(
                RejectionCode.FILE_TOO_LARGE,
                f"document exceeds the {_FIVE_HUNDRED_MB_LABEL} size limit "
                f"({size} bytes > {max_size} bytes)",
            )

        # 3. Format detected by content sniffing (Req 1.3).
        fmt = _sniff_format(data)
        if fmt is None:
            return Rejected(
                RejectionCode.UNSUPPORTED_FORMAT,
                "unsupported document format: detected "
                f"{_describe_detected(data)}; supported formats are "
                "PDF, EPUB, DOCX, and HTML",
            )

        # 4. Well-formed instance of the detected format (Req 1.7).
        if not _is_well_formed(data, fmt):
            return Rejected(
                RejectionCode.CORRUPTED_FILE,
                f"document is empty or corrupted: not a complete, well-formed "
                f"{fmt.name} file",
            )

        # 5. Content-hash dedup (Req 1.5).
        content_hash = hashlib.sha256(data).hexdigest()
        existing = self._store.find_by_content_hash(content_hash)
        if existing is not None:
            return Duplicate(existing.jobId)

        # Success: record metadata in UTC (Req 1.4) and create the job (Req 1.1).
        metadata = DocumentMetadata(
            filename=filename,
            format=fmt,
            byteSize=size,
            contentHash=content_hash,
            submittedAtUtc=datetime.now(timezone.utc),
        )
        job = self._store.create_job(
            document_id=document_id_from_hash(content_hash),
            metadata=metadata,
        )
        return Accepted(job.jobId)
