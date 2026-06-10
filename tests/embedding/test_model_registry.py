"""Unit tests for the EmbeddingModel port, registry, and mock (Task 10.1).

These cover the contract introduced in task 10.1 only: config-driven model
selection (Req 7.3), the deterministic mock adapter, and registry semantics.
The Embedder (retry policy, dimension/timeout enforcement) is tested with its
own task (10.2 onward).
"""

from __future__ import annotations

import pytest

from biomed_rag.config import PipelineConfig
from biomed_rag.embedding import (
    EmbeddingError,
    EmbeddingModel,
    EmbeddingModelNotRegisteredError,
    EmbeddingModelRegistry,
    EmbeddingTimeoutError,
    MockEmbeddingModel,
)


def test_mock_model_implements_port() -> None:
    model = MockEmbeddingModel(model_id="mock-emb", dimension=8)
    assert isinstance(model, EmbeddingModel)
    assert model.model_id() == "mock-emb"
    assert model.dimension() == 8


def test_mock_embed_is_deterministic_and_dimensioned() -> None:
    model = MockEmbeddingModel(dimension=12)
    first = model.embed("biomedical text")
    second = model.embed("biomedical text")
    assert first == second
    assert len(first) == 12
    assert all(isinstance(v, float) for v in first)
    assert all(-1.0 <= v <= 1.0 for v in first)


def test_mock_distinct_text_yields_distinct_vector() -> None:
    model = MockEmbeddingModel(dimension=16)
    assert model.embed("alpha") != model.embed("beta")


def test_mock_distinct_model_id_yields_distinct_vector() -> None:
    a = MockEmbeddingModel(model_id="model-a", dimension=16)
    b = MockEmbeddingModel(model_id="model-b", dimension=16)
    assert a.embed("same text") != b.embed("same text")


def test_mock_raise_on_embed_simulates_failure() -> None:
    model = MockEmbeddingModel(raise_on_embed=EmbeddingError("boom"))
    with pytest.raises(EmbeddingError):
        model.embed("anything")


def test_mock_raise_on_embed_simulates_timeout() -> None:
    model = MockEmbeddingModel(raise_on_embed=EmbeddingTimeoutError("slow"))
    with pytest.raises(EmbeddingTimeoutError):
        model.embed("anything")


def test_mock_rejects_invalid_dimension() -> None:
    with pytest.raises(ValueError):
        MockEmbeddingModel(dimension=0)


def test_registry_selects_model_from_config() -> None:
    registry = EmbeddingModelRegistry()
    registry.register("mock-emb", lambda: MockEmbeddingModel(model_id="mock-emb", dimension=64))
    registry.register("other-emb", lambda: MockEmbeddingModel(model_id="other-emb", dimension=64))

    mock_cfg = PipelineConfig(embedding_model="mock-emb", embedding_dimension=64)
    other_cfg = PipelineConfig(embedding_model="other-emb", embedding_dimension=64)

    assert registry.select(mock_cfg).model_id() == "mock-emb"
    assert registry.select(other_cfg).model_id() == "other-emb"


def test_registry_create_returns_fresh_instances() -> None:
    registry = EmbeddingModelRegistry()
    registry.register("mock-emb", MockEmbeddingModel)
    first = registry.create("mock-emb")
    second = registry.create("mock-emb")
    assert first is not second


def test_registry_is_registered() -> None:
    registry = EmbeddingModelRegistry()
    assert registry.is_registered("mock-emb") is False
    registry.register("mock-emb", MockEmbeddingModel)
    assert registry.is_registered("mock-emb") is True


def test_registry_unregistered_id_raises() -> None:
    registry = EmbeddingModelRegistry()
    with pytest.raises(EmbeddingModelNotRegisteredError) as excinfo:
        registry.create("nope")
    assert excinfo.value.model_id == "nope"


def test_registry_select_none_model_raises() -> None:
    registry = EmbeddingModelRegistry()
    with pytest.raises(EmbeddingModelNotRegisteredError) as excinfo:
        registry.select(PipelineConfig(embedding_model=None))
    assert excinfo.value.model_id is None


def test_registry_duplicate_registration_requires_replace() -> None:
    registry = EmbeddingModelRegistry()
    registry.register("mock-emb", MockEmbeddingModel)
    with pytest.raises(ValueError):
        registry.register("mock-emb", MockEmbeddingModel)
    registry.register(
        "mock-emb",
        lambda: MockEmbeddingModel(model_id="mock-emb", dimension=128),
        replace=True,
    )
    assert registry.create("mock-emb").dimension() == 128


def test_registry_rejects_empty_model_id() -> None:
    registry = EmbeddingModelRegistry()
    with pytest.raises(TypeError):
        registry.register("   ", MockEmbeddingModel)
