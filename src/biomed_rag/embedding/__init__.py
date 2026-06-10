"""Embedder and pluggable Embedding_Model port (Req 7).

This package defines the model-neutral embedding port and supporting pieces:

* :class:`EmbeddingModel` — the pluggable port (Req 7.3).
* :class:`EmbeddingModelRegistry` / :data:`default_registry` — config-driven
  model selection (Req 7.3).
* :class:`MockEmbeddingModel` — a deterministic adapter for tests.
* :class:`Embedder` — dimension/timeout enforcement and the retry policy,
  returning an :data:`EmbedResult` (``Embedding | EmbedFailed``) (Req 7.1–7.7).
"""

from __future__ import annotations

from .embedder import (
    EmbedFailed,
    EmbedFailureReason,
    Embedder,
    EmbedResult,
)
from .mock import MockEmbeddingModel
from .model import (
    EmbeddingError,
    EmbeddingModel,
    EmbeddingTimeoutError,
)
from .registry import (
    EmbeddingModelNotRegisteredError,
    EmbeddingModelRegistry,
    ModelFactory,
    default_registry,
)

__all__ = [
    # port
    "EmbeddingModel",
    "EmbeddingError",
    "EmbeddingTimeoutError",
    # registry
    "EmbeddingModelRegistry",
    "EmbeddingModelNotRegisteredError",
    "ModelFactory",
    "default_registry",
    # embedder
    "Embedder",
    "EmbedResult",
    "EmbedFailed",
    "EmbedFailureReason",
    # test support
    "MockEmbeddingModel",
]
