"""Enumerations shared across the pipeline data models."""

from __future__ import annotations

from enum import Enum


class Stage(Enum):
    """The five resumable pipeline stages, in execution order (Req 10.1)."""

    PARSING = 0
    NORMALIZATION = 1
    CHUNKING = 2
    EMBEDDING = 3
    STORAGE = 4

    @property
    def order(self) -> int:
        """Zero-based sequential position of this stage in the pipeline."""
        return self.value

    @classmethod
    def ordered(cls) -> tuple["Stage", ...]:
        """The stages in their strict execution order."""
        return tuple(sorted(cls, key=lambda s: s.value))


class StageStatus(Enum):
    """Lifecycle status recorded for each stage transition (Req 10.6)."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class OverallStatus(Enum):
    """Top-level status of a Processing_Job."""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class Format(Enum):
    """Supported source document formats (Req 1.2, 1.3)."""

    PDF = "pdf"
    EPUB = "epub"
    DOCX = "docx"
    HTML = "html"


class BlockType(Enum):
    """Type of a parsed text block (Req 2.1)."""

    PARAGRAPH = "paragraph"
    HEADING = "heading"
    CAPTION = "caption"
    OCR_TEXT = "ocr_text"
    LIST_ITEM = "list_item"
    FOOTNOTE = "footnote"
    OTHER = "other"


class BlockSource(Enum):
    """Origin of a text block's text content."""

    TEXT_LAYER = "text_layer"
    OCR = "ocr"


class ElementKind(Enum):
    """Kind of a normalized content element (Req 5.4)."""

    TEXT = "text"
    HEADING = "heading"
    TABLE = "table"
    FIGURE = "figure"


class EmbeddingStatus(Enum):
    """Outcome status of an embedding attempt (Req 7.7)."""

    OK = "ok"
    FAILED = "failed"


class OCRErrorKind(Enum):
    """Reason an image could not be processed by OCR (Req 4.5, 4.6)."""

    UNREADABLE = "unreadable"
    CORRUPT = "corrupt"
    UNSUPPORTED = "unsupported"
    TIMEOUT = "timeout"
