"""Chunk, Embedding, and storage/retrieval record models (Req 6, 7, 8, 9)."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from numbers import Real
from typing import List, Optional

from ._validation import (
    require,
    require_float_in_range,
    require_int_in_range,
    require_non_empty_str,
)
from .enums import EmbeddingStatus
from .identifiers import DocumentId


def _new_uuid() -> str:
    return str(uuid.uuid4())


@dataclass
class Chunk:
    """A bounded segment of normalized content with its source metadata
    (Req 6.1, 6.2, 6.3, 6.7).

    ``documentId`` is always present. ``pageNumber`` is ``None`` and
    ``headingPath`` is an empty list when the respective metadata is unavailable.
    """

    documentId: DocumentId
    content: str
    tokenCount: int
    orderIndex: int
    overlapTokenCount: int = 0
    pageNumber: Optional[int] = None
    headingPath: List[str] = field(default_factory=list)
    isTablePart: bool = False
    chunkId: str = field(default_factory=_new_uuid)

    def __post_init__(self) -> None:
        # documentId is mandatory for every chunk (Req 6.3, 6.7).
        require_non_empty_str(self.documentId, "Chunk.documentId")
        require(isinstance(self.content, str), "Chunk.content must be a str")
        require_int_in_range(self.tokenCount, "Chunk.tokenCount", minimum=0)
        require_int_in_range(self.orderIndex, "Chunk.orderIndex", minimum=0)
        # Overlap is shared with the previous chunk; it cannot exceed this
        # chunk's own token count (Req 6.2, 6.5).
        require_int_in_range(
            self.overlapTokenCount,
            "Chunk.overlapTokenCount",
            minimum=0,
            maximum=self.tokenCount,
        )
        # pageNumber is empty (None) when unavailable (Req 6.7).
        if self.pageNumber is not None:
            require_int_in_range(self.pageNumber, "Chunk.pageNumber", minimum=0)
        require(
            all(isinstance(h, str) for h in self.headingPath),
            "Chunk.headingPath must be a list of str",
        )
        require(isinstance(self.isTablePart, bool), "Chunk.isTablePart must be a bool")
        require_non_empty_str(self.chunkId, "Chunk.chunkId")


@dataclass
class Embedding:
    """A numeric vector embedding for a chunk plus its attempt accounting
    (Req 7.1, 7.5, 7.6, 7.7)."""

    chunkId: str
    vector: List[float]
    modelId: str
    status: EmbeddingStatus = EmbeddingStatus.OK
    attempts: int = 0

    def __post_init__(self) -> None:
        require_non_empty_str(self.chunkId, "Embedding.chunkId")
        require(isinstance(self.vector, list), "Embedding.vector must be a list")
        require(
            all(isinstance(v, Real) and not isinstance(v, bool) for v in self.vector),
            "Embedding.vector must contain only numbers",
        )
        self.vector = [float(v) for v in self.vector]
        require_non_empty_str(self.modelId, "Embedding.modelId")
        require(
            isinstance(self.status, EmbeddingStatus),
            "Embedding.status must be an EmbeddingStatus",
        )
        require_int_in_range(self.attempts, "Embedding.attempts", minimum=0)

    @property
    def dimension(self) -> int:
        """The dimension of the embedding vector."""
        return len(self.vector)


@dataclass
class VectorRecord:
    """The stored unit: a chunk together with its embedding, addressable by the
    source document identifier (Req 8.1, 8.2)."""

    documentId: DocumentId
    chunk: Chunk
    embedding: Embedding

    def __post_init__(self) -> None:
        require_non_empty_str(self.documentId, "VectorRecord.documentId")
        require(isinstance(self.chunk, Chunk), "VectorRecord.chunk must be a Chunk")
        require(
            isinstance(self.embedding, Embedding),
            "VectorRecord.embedding must be an Embedding",
        )
        # Keep the stored unit internally consistent.
        require(
            self.chunk.documentId == self.documentId,
            "VectorRecord.documentId must match chunk.documentId",
        )
        require(
            self.embedding.chunkId == self.chunk.chunkId,
            "VectorRecord.embedding.chunkId must match chunk.chunkId",
        )


@dataclass
class ScoredRecord:
    """A retrieval result: a stored record paired with its similarity score
    (Req 9.1)."""

    record: VectorRecord
    similarity: float

    def __post_init__(self) -> None:
        require(
            isinstance(self.record, VectorRecord),
            "ScoredRecord.record must be a VectorRecord",
        )
        self.similarity = require_float_in_range(
            self.similarity, "ScoredRecord.similarity", minimum=0.0, maximum=1.0
        )
