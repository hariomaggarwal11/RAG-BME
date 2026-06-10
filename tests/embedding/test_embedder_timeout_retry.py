"""Unit tests for Embedder timeout, misconfiguration, and retry policy (Task 10.4).

Covers the failure-handling policy the Embedder owns on top of the
``EmbeddingModel`` port:

* Timeout handling (Req 7.2) — both a model that raises
  :class:`EmbeddingTimeoutError` and a model that ignores the deadline while an
  injected clock advances past it.
* Missing / unrecognized model rejection (Req 7.3, 7.4) — the request is
  rejected before any embed attempt and the chunk is left unmodified.
* Bounded retry-then-fail (Req 7.6, 7.7) — a model that always fails is retried
  up to ``embedding_max_retries`` times (one initial attempt plus 3 retries =
  4 embed calls) and then the chunk is marked failed with its content retained.
* Recovery — a model that fails a few times then succeeds yields an OK
  embedding and reports the attempt count.

These are example-based unit tests (pytest); the dimension-consistency property
is covered separately by Task 10.3.
"""

from __future__ import annotations

import dataclasses
from typing import List, Optional

import pytest

from biomed_rag.config import PipelineConfig
from biomed_rag.embedding import (
    Embedder,
    EmbedFailed,
    EmbedFailureReason,
    EmbeddingError,
    EmbeddingModel,
    EmbeddingModelRegistry,
    EmbeddingTimeoutError,
    MockEmbeddingModel,
)
from biomed_rag.models import Chunk, Embedding, EmbeddingStatus

# The configured dimension used across these tests; must be in [64, 4096] and
# must match the model's declared dimension so we exercise the embed loop rather
# than a dimension-misconfiguration rejection.
DIM = 64


def _chunk(content: str = "some biomedical content") -> Chunk:
    return Chunk(
        documentId="doc-1",
        content=content,
        tokenCount=3,
        orderIndex=0,
    )


def _config(model: Optional[str], *, dimension: int = DIM) -> PipelineConfig:
    return PipelineConfig(embedding_model=model, embedding_dimension=dimension)


def _registry_with(model_id: str, model: EmbeddingModel) -> EmbeddingModelRegistry:
    """A registry whose factory always returns the given model instance.

    Returning a single instance (rather than a fresh one per call) lets stateful
    test models count calls across the Embedder's retry loop, which selects the
    model once per ``embed`` call.
    """
    registry = EmbeddingModelRegistry()
    registry.register(model_id, lambda: model)
    return registry


class _CallCountingModel(EmbeddingModel):
    """A port implementation that fails its first ``fail_times`` calls.

    After ``fail_times`` failures it delegates to a deterministic mock so the
    successful path returns a correctly dimensioned vector. ``calls`` records the
    total number of embed invocations so tests can assert the retry count.
    """

    def __init__(
        self,
        *,
        fail_times: int,
        model_id: str = "flaky-emb",
        dimension: int = DIM,
        error: Optional[BaseException] = None,
    ) -> None:
        self._fail_times = fail_times
        self._error = error if error is not None else EmbeddingError("transient")
        self._delegate = MockEmbeddingModel(model_id=model_id, dimension=dimension)
        self.calls = 0

    def model_id(self) -> str:
        return self._delegate.model_id()

    def dimension(self) -> int:
        return self._delegate.dimension()

    def embed(self, text: str, deadline: Optional[float] = None) -> List[float]:
        self.calls += 1
        if self.calls <= self._fail_times:
            raise self._error
        return self._delegate.embed(text, deadline=deadline)


class _SteppingClock:
    """A deterministic monotonic clock that advances by ``step`` on each read.

    With a step larger than the configured timeout, every attempt's deadline
    check fails, simulating a model that always exceeds the time budget even
    though it returns a vector.
    """

    def __init__(self, step: float) -> None:
        self._t = 0.0
        self._step = step

    def __call__(self) -> float:
        self._t += self._step
        return self._t


# --- missing / unrecognized model rejection (Req 7.3, 7.4) ------------------


def test_missing_model_rejected_without_touching_chunk() -> None:
    embedder = Embedder(EmbeddingModelRegistry())
    chunk = _chunk()
    snapshot = dataclasses.replace(chunk)

    result = embedder.embed(chunk, _config(None))

    assert isinstance(result, EmbedFailed)
    assert result.reason is EmbedFailureReason.MODEL_NOT_CONFIGURED
    assert result.reason.is_misconfiguration
    # No embed attempt was made and the chunk is returned unmodified (Req 7.4).
    assert result.attempts == 0
    assert result.chunk is chunk
    assert result.chunk == snapshot
    assert result.cause  # an error indication is present


def test_unrecognized_model_rejected_without_touching_chunk() -> None:
    # Registry has some other model, but not the one the config asks for.
    registry = _registry_with("known-emb", MockEmbeddingModel(model_id="known-emb", dimension=DIM))
    embedder = Embedder(registry)
    chunk = _chunk()
    snapshot = dataclasses.replace(chunk)

    result = embedder.embed(chunk, _config("ghost-emb"))

    assert isinstance(result, EmbedFailed)
    assert result.reason is EmbedFailureReason.MODEL_UNRECOGNIZED
    assert result.reason.is_misconfiguration
    assert result.attempts == 0
    assert result.chunk is chunk
    assert result.chunk == snapshot
    # The error indication identifies the invalid model id (Req 7.4).
    assert "ghost-emb" in result.cause


# --- timeout handling (Req 7.2) ---------------------------------------------


def test_timeout_error_from_model_exhausts_retries_then_fails() -> None:
    model = MockEmbeddingModel(
        model_id="slow-emb",
        dimension=DIM,
        raise_on_embed=EmbeddingTimeoutError("deadline exceeded"),
    )
    registry = _registry_with("slow-emb", model)
    embedder = Embedder(registry)
    chunk = _chunk()
    config = _config("slow-emb")

    result = embedder.embed(chunk, config)

    assert isinstance(result, EmbedFailed)
    assert result.reason is EmbedFailureReason.EMBED_FAILED
    # One initial attempt plus embedding_max_retries retries.
    assert result.attempts == config.embedding_max_retries + 1
    assert "timed out" in result.cause
    # Chunk content retained unmodified (Req 7.7).
    assert result.chunk.content == "some biomedical content"


def test_clock_deadline_exceeded_is_treated_as_timeout() -> None:
    # The model returns a valid vector, but the injected clock advances past the
    # deadline on every attempt, so the Embedder enforces the timeout (Req 7.2).
    model = MockEmbeddingModel(model_id="lazy-emb", dimension=DIM)
    registry = _registry_with("lazy-emb", model)
    config = _config("lazy-emb")
    # Step larger than embedding_timeout_seconds so each deadline check fails.
    clock = _SteppingClock(step=config.embedding_timeout_seconds + 5)
    embedder = Embedder(registry, clock=clock)

    result = embedder.embed(_chunk(), config)

    assert isinstance(result, EmbedFailed)
    assert result.reason is EmbedFailureReason.EMBED_FAILED
    assert result.attempts == config.embedding_max_retries + 1
    assert "embedding_timeout_seconds" in result.cause


# --- bounded retry-then-fail (Req 7.6, 7.7) ---------------------------------


def test_always_failing_model_exhausts_exactly_max_attempts() -> None:
    model = _CallCountingModel(fail_times=999, model_id="broken-emb", error=EmbeddingError("nope"))
    registry = _registry_with("broken-emb", model)
    embedder = Embedder(registry)
    chunk = _chunk()
    config = _config("broken-emb")

    result = embedder.embed(chunk, config)

    assert isinstance(result, EmbedFailed)
    assert result.reason is EmbedFailureReason.EMBED_FAILED
    # 1 initial attempt + 3 retries == 4 embed calls (Req 7.6, 7.7).
    assert config.embedding_max_retries == 3
    assert model.calls == config.embedding_max_retries + 1
    assert result.attempts == config.embedding_max_retries + 1
    assert "embedding error" in result.cause
    # Original chunk content is retained unmodified.
    assert result.chunk is chunk
    assert result.chunk.content == "some biomedical content"


def test_failed_embed_produces_failed_embedding_marker() -> None:
    model = _CallCountingModel(fail_times=999, model_id="broken-emb")
    registry = _registry_with("broken-emb", model)
    embedder = Embedder(registry)
    config = _config("broken-emb")

    result = embedder.embed(_chunk(), config)
    assert isinstance(result, EmbedFailed)

    marker = result.as_failed_embedding("broken-emb")
    assert isinstance(marker, Embedding)
    assert marker.status is EmbeddingStatus.FAILED
    assert marker.vector == []
    assert marker.attempts == config.embedding_max_retries + 1
    assert marker.chunkId == result.chunk.chunkId


def test_model_succeeds_after_transient_failures() -> None:
    # Fails twice, then succeeds on the third attempt.
    model = _CallCountingModel(fail_times=2, model_id="recovering-emb", dimension=DIM)
    registry = _registry_with("recovering-emb", model)
    embedder = Embedder(registry)
    chunk = _chunk()
    config = _config("recovering-emb")

    result = embedder.embed(chunk, config)

    assert isinstance(result, Embedding)
    assert result.status is EmbeddingStatus.OK
    assert result.dimension == DIM
    assert result.modelId == "recovering-emb"
    # Two failures + one success.
    assert model.calls == 3
    assert result.attempts == 3
    assert result.chunkId == chunk.chunkId


def test_model_succeeds_on_last_allowed_attempt() -> None:
    # Fails exactly embedding_max_retries times, succeeds on the final attempt.
    config = _config("edge-emb")
    model = _CallCountingModel(fail_times=config.embedding_max_retries, model_id="edge-emb", dimension=DIM)
    registry = _registry_with("edge-emb", model)
    embedder = Embedder(registry)

    result = embedder.embed(_chunk(), config)

    assert isinstance(result, Embedding)
    assert result.status is EmbeddingStatus.OK
    assert result.attempts == config.embedding_max_retries + 1
    assert model.calls == config.embedding_max_retries + 1
