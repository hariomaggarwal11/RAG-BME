"""The Embedder: retry policy and dimension/timeout enforcement (Req 7).

The :class:`Embedder` turns a :class:`~biomed_rag.models.Chunk` into an
:class:`~biomed_rag.models.Embedding`, mediating between the pipeline and a
pluggable :class:`~biomed_rag.embedding.model.EmbeddingModel` selected from a
:class:`~biomed_rag.config.PipelineConfig`. It owns the *policy* the port
deliberately leaves out:

* **Dimension enforcement** — the produced vector must have exactly the
  configured dimension (an integer in ``64..4096``) and that dimension is
  identical for every chunk embedded with the same model (Req 7.1, 7.5).
* **Timeout enforcement** — each attempt is given a fresh deadline of
  ``config.embedding_timeout_seconds`` and must complete within it (Req 7.2).
* **Configuration validation** — a missing or unrecognized model (or a model
  whose declared dimension disagrees with the config) is rejected without
  touching the chunk (Req 7.3, 7.4).
* **Retry policy** — a transient embed failure is retried up to
  ``config.embedding_max_retries`` (3) additional times; once those retries are
  exhausted the chunk is marked failed with its original content retained
  (Req 7.6, 7.7).

The result is an :data:`EmbedResult`, a closed union of a successful
:class:`~biomed_rag.models.Embedding` (``status = OK``) or an
:class:`EmbedFailed` carrying the cause, the retained chunk, and the number of
attempts made.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional, Union

from biomed_rag.config import PipelineConfig
from biomed_rag.models import Chunk, Embedding, EmbeddingStatus

from .model import EmbeddingError, EmbeddingModel, EmbeddingTimeoutError
from .registry import (
    EmbeddingModelNotRegisteredError,
    EmbeddingModelRegistry,
    default_registry,
)

# A monotonic clock, injectable for deterministic timeout tests.
Clock = Callable[[], float]


class EmbedFailureReason(Enum):
    """Why the Embedder could not produce a usable embedding for a chunk."""

    #: ``config.embedding_model`` is ``None`` — no model configured (Req 7.4).
    MODEL_NOT_CONFIGURED = "model_not_configured"
    #: The configured model id is not registered (Req 7.4).
    MODEL_UNRECOGNIZED = "model_unrecognized"
    #: The model's declared dimension disagrees with the configured dimension
    #: (Req 7.1, 7.5) — a configuration error, surfaced before any embed call.
    DIMENSION_MISCONFIGURED = "dimension_misconfigured"
    #: The model repeatedly failed to embed and retries were exhausted
    #: (Req 7.6, 7.7).
    EMBED_FAILED = "embed_failed"

    @property
    def is_misconfiguration(self) -> bool:
        """True when the failure is a configuration error (Req 7.4).

        Misconfiguration rejects the request before any embed attempt, so the
        chunk is left completely unmodified.
        """
        return self in (
            EmbedFailureReason.MODEL_NOT_CONFIGURED,
            EmbedFailureReason.MODEL_UNRECOGNIZED,
            EmbedFailureReason.DIMENSION_MISCONFIGURED,
        )


@dataclass(frozen=True)
class EmbedFailed:
    """The unsuccessful arm of :data:`EmbedResult` (Req 7.4, 7.6, 7.7).

    ``chunk`` is the original, unmodified chunk (retained per Req 7.4/7.7).
    ``cause`` is a human-readable error indication describing why embedding
    failed (Req 7.6). ``attempts`` is the number of embed calls made: ``0`` for
    a configuration rejection (no embed was attempted) and ``1 +
    embedding_max_retries`` when transient failures exhausted the retry budget.
    """

    chunk: Chunk
    reason: EmbedFailureReason
    cause: str
    attempts: int = 0

    def as_failed_embedding(self, model_id: str) -> Embedding:
        """Build a ``status = FAILED`` :class:`Embedding` marker for this chunk.

        Provided so a caller (e.g. the Orchestrator) can persist the canonical
        "chunk failed" marker (Req 7.7) using the existing
        :class:`~biomed_rag.models.EmbeddingStatus`. The vector is empty because
        no valid embedding was produced; the chunk's content is retained on
        :attr:`chunk` unmodified.
        """
        return Embedding(
            chunkId=self.chunk.chunkId,
            vector=[],
            modelId=model_id,
            status=EmbeddingStatus.FAILED,
            attempts=self.attempts,
        )


# EmbedResult = Embedding | EmbedFailed(cause)  (design: Embedder section).
EmbedResult = Union[Embedding, EmbedFailed]


class Embedder:
    """Generates embeddings for chunks with dimension/timeout/retry policy (Req 7)."""

    def __init__(
        self,
        registry: EmbeddingModelRegistry = default_registry,
        *,
        clock: Clock = time.monotonic,
    ) -> None:
        """Create an Embedder.

        Args:
            registry: the model registry used to select a model from config
                (Req 7.3). Defaults to the process-wide ``default_registry``.
            clock: a monotonic clock returning seconds; injectable so timeout
                behaviour (Req 7.2) is deterministically testable.
        """
        self._registry = registry
        self._clock = clock

    def embed(self, chunk: Chunk, config: PipelineConfig) -> EmbedResult:
        """Embed ``chunk`` according to ``config`` (Req 7).

        Returns a successful :class:`~biomed_rag.models.Embedding` whose
        dimension equals ``config.embedding_dimension``, or an
        :class:`EmbedFailed` describing why no embedding could be produced. The
        chunk is never mutated.
        """
        if not isinstance(chunk, Chunk):
            raise TypeError("chunk must be a Chunk")
        if not isinstance(config, PipelineConfig):
            raise TypeError("config must be a PipelineConfig")

        # --- model selection / configuration validation (Req 7.3, 7.4) ------
        try:
            model = self._registry.select(config)
        except EmbeddingModelNotRegisteredError as exc:
            reason = (
                EmbedFailureReason.MODEL_NOT_CONFIGURED
                if config.embedding_model is None
                else EmbedFailureReason.MODEL_UNRECOGNIZED
            )
            return EmbedFailed(
                chunk=chunk,
                reason=reason,
                cause=str(exc),
                attempts=0,
            )

        expected_dim = config.embedding_dimension
        declared_dim = model.dimension()
        if declared_dim != expected_dim:
            # The model can never satisfy the configured dimension; reject the
            # request before spending any attempts and leave the chunk untouched
            # (Req 7.1, 7.5).
            return EmbedFailed(
                chunk=chunk,
                reason=EmbedFailureReason.DIMENSION_MISCONFIGURED,
                cause=(
                    f"configured embedding_dimension {expected_dim} does not match "
                    f"model {model.model_id()!r} dimension {declared_dim}"
                ),
                attempts=0,
            )

        # --- attempt loop with bounded retries (Req 7.2, 7.6, 7.7) ----------
        # One initial attempt plus up to embedding_max_retries retries.
        max_attempts = config.embedding_max_retries + 1
        timeout = config.embedding_timeout_seconds
        last_cause = "embedding failed"

        for attempt in range(1, max_attempts + 1):
            started = self._clock()
            deadline = started + timeout
            try:
                vector = model.embed(chunk.content, deadline=deadline)
            except EmbeddingTimeoutError as exc:
                last_cause = f"embedding timed out after {timeout}s: {exc}"
                continue
            except EmbeddingError as exc:
                last_cause = f"embedding error: {exc}"
                continue

            # Enforce the deadline even if the model ignored it (Req 7.2).
            if self._clock() > deadline:
                last_cause = (
                    f"embedding exceeded embedding_timeout_seconds ({timeout}s)"
                )
                continue

            # Enforce the configured dimension on the produced vector
            # (Req 7.1, 7.5).
            if len(vector) != expected_dim:
                last_cause = (
                    f"model returned a vector of length {len(vector)}; "
                    f"expected configured dimension {expected_dim}"
                )
                continue

            return Embedding(
                chunkId=chunk.chunkId,
                vector=vector,
                modelId=model.model_id(),
                status=EmbeddingStatus.OK,
                attempts=attempt,
            )

        # Retries exhausted: mark the chunk failed, retain its content (Req 7.7).
        return EmbedFailed(
            chunk=chunk,
            reason=EmbedFailureReason.EMBED_FAILED,
            cause=last_cause,
            attempts=max_attempts,
        )
