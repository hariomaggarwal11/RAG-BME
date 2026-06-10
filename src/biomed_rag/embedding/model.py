"""The pluggable Embedding_Model port (Req 7.3).

``EmbeddingModel`` is the stable interface every embedding backend implements.
Concrete adapters (sentence-transformers, hosted embedding APIs, etc.) sit
behind this port so the Embedder and the rest of the pipeline depend only on
the contract, never on a specific model (design: ports-and-adapters, Req 7.3).

This module defines the port and the embed-time exceptions a model may raise.
The Embedder (implemented in a later task) translates these into the retry
policy and ``EmbedFailed`` outcomes (Req 7.6, 7.7); the failure-handling policy
itself lives in the Embedder, not here.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional


class EmbeddingError(Exception):
    """Raised by a model when it cannot embed a piece of text (Req 7.6)."""


class EmbeddingTimeoutError(EmbeddingError):
    """Raised by a model when embedding exceeds the supplied deadline (Req 7.2)."""


class EmbeddingModel(ABC):
    """Port for a pluggable text embedding backend (Req 7.3).

    Implementations are expected to be deterministic with respect to their
    inputs where possible and side-effect free with respect to the pipeline's
    state: they read text and return a numeric vector, leaving chunk-state
    transitions and retries to the Embedder.

    The vector returned by :meth:`embed` MUST have exactly :meth:`dimension`
    elements, and that dimension MUST be identical across every call for a
    given model instance (Req 7.1, 7.5).
    """

    @abstractmethod
    def model_id(self) -> str:
        """Return the stable identifier of this model (e.g. ``"mock-emb"``).

        Used for configuration-driven selection and for recording the model in
        :class:`~biomed_rag.models.Embedding` records (Req 7.3).
        """
        raise NotImplementedError

    @abstractmethod
    def dimension(self) -> int:
        """Return the fixed dimension of vectors produced by this model.

        Constant for the lifetime of the model instance (Req 7.1, 7.5).
        """
        raise NotImplementedError

    @abstractmethod
    def embed(
        self,
        text: str,
        deadline: Optional[float] = None,
    ) -> List[float]:
        """Embed ``text`` into a vector of length :meth:`dimension`.

        ``deadline`` is an optional monotonic-clock timestamp (seconds, as from
        :func:`time.monotonic`) past which the model should abort and raise
        :class:`EmbeddingTimeoutError` (Req 7.2). ``None`` means no deadline.

        Raises:
            EmbeddingError: the text could not be embedded (Req 7.6).
            EmbeddingTimeoutError: the deadline was exceeded (Req 7.2).
        """
        raise NotImplementedError
