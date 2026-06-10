"""Property-based test for embedding dimension consistency (Task 10.3).

Feature: biomedical-rag-pipeline, Property 18: Embedding dimension is consistent and configured

This test exercises Property 18 from the design's Correctness Properties:

    *For any* set of Chunks embedded with a given configured model, every
    produced Embedding has a dimension equal to the configured dimension (an
    integer in [64, 4096]).

**Validates: Requirements 7.1, 7.5**

Strategy
--------
Each example draws a configured ``embedding_dimension`` somewhere in the valid
[64, 4096] range and builds a :class:`PipelineConfig` for it. A fresh registry
is populated with a deterministic mock model whose declared dimension *matches*
the configured dimension (the Embedder rejects a mismatch as a configuration
error, so a matching model is what lets embeddings actually be produced). A set
of Chunks with arbitrary content is then embedded one at a time, and the test
asserts that every successfully produced Embedding has a vector length exactly
equal to the configured dimension — identically across the whole set.
"""

from __future__ import annotations

from typing import List

from hypothesis import given, settings
from hypothesis import strategies as st

from biomed_rag.config import PipelineConfig
from biomed_rag.embedding import (
    Embedder,
    EmbeddingModelRegistry,
    MockEmbeddingModel,
)
from biomed_rag.models import Chunk, Embedding

# The design's configured-dimension bounds (Configuration Model table, Req 7.1, 7.5).
_MIN_DIMENSION = 64
_MAX_DIMENSION = 4096

_MODEL_ID = "mock-emb"


@st.composite
def _chunks(draw) -> List[Chunk]:
    """Draw a non-empty list of Chunks with arbitrary content/metadata."""
    contents = draw(
        st.lists(st.text(min_size=0, max_size=64), min_size=1, max_size=8)
    )
    return [
        Chunk(
            documentId="doc-1",
            content=content,
            tokenCount=len(content.split()),
            orderIndex=index,
        )
        for index, content in enumerate(contents)
    ]


@settings(max_examples=150, deadline=None)
@given(
    dimension=st.integers(min_value=_MIN_DIMENSION, max_value=_MAX_DIMENSION),
    chunks=_chunks(),
)
def test_embedding_dimension_is_consistent_and_configured(
    dimension: int, chunks: List[Chunk]
) -> None:
    config = PipelineConfig(embedding_model=_MODEL_ID, embedding_dimension=dimension)

    # A registry holding a mock model whose dimension matches the config.
    registry = EmbeddingModelRegistry()
    registry.register(
        _MODEL_ID,
        lambda: MockEmbeddingModel(model_id=_MODEL_ID, dimension=dimension),
    )
    embedder = Embedder(registry)

    for chunk in chunks:
        result = embedder.embed(chunk, config)
        # The model matches the config, so every chunk must embed successfully.
        assert isinstance(result, Embedding), (
            f"expected an Embedding but got {result!r}"
        )
        # The core property: produced dimension equals the configured dimension.
        assert result.dimension == dimension
        assert len(result.vector) == dimension
