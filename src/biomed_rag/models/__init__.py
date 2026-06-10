"""Shared data models for the biomedical RAG pipeline.

This package defines the core identifiers, enums, and dataclasses that flow
between pipeline stages. Field-level invariants from the design's Data Models
section are enforced at construction time; violations raise
:class:`ModelValidationError`.
"""

from __future__ import annotations

from ._validation import ModelValidationError
from .chunk import Chunk, Embedding, ScoredRecord, VectorRecord
from .enums import (
    BlockSource,
    BlockType,
    ElementKind,
    EmbeddingStatus,
    Format,
    OCRErrorKind,
    OverallStatus,
    Stage,
    StageStatus,
)
from .identifiers import (
    DocumentId,
    JobId,
    document_id_from_hash,
    new_job_id,
)
from .job import (
    MAX_BYTE_SIZE,
    MAX_FILENAME_LENGTH,
    DocumentMetadata,
    ProcessingJob,
    StageState,
)
from .normalized import (
    ContentElement,
    FigurePayload,
    NormalizedDocument,
    Payload,
    TablePayload,
    TextPayload,
)
from .parsed import (
    Cell,
    Figure,
    Heading,
    ImageOCRError,
    ParsedDocument,
    Table,
    TextBlock,
)

__all__ = [
    # validation
    "ModelValidationError",
    # identifiers
    "JobId",
    "DocumentId",
    "new_job_id",
    "document_id_from_hash",
    # enums
    "Stage",
    "StageStatus",
    "OverallStatus",
    "Format",
    "BlockType",
    "BlockSource",
    "ElementKind",
    "EmbeddingStatus",
    "OCRErrorKind",
    # job
    "DocumentMetadata",
    "StageState",
    "ProcessingJob",
    "MAX_BYTE_SIZE",
    "MAX_FILENAME_LENGTH",
    # parsed
    "TextBlock",
    "Cell",
    "Table",
    "Figure",
    "Heading",
    "ImageOCRError",
    "ParsedDocument",
    # normalized
    "TextPayload",
    "TablePayload",
    "FigurePayload",
    "Payload",
    "ContentElement",
    "NormalizedDocument",
    # chunk / embedding / storage / retrieval
    "Chunk",
    "Embedding",
    "VectorRecord",
    "ScoredRecord",
]
