"""Unit tests for :class:`PipelineConfig` construction-time validation.

These tests exercise the bounded configuration keys defined in the design's
Configuration Model table: in-range acceptance, default values, and
out-of-range rejection, including the dependent bound that ties
``chunk_overlap_tokens`` to ``max_chunk_tokens``.

Requirements covered: 6.1, 6.2, 7.1, 9.1, 10.2 (plus the remaining bounded and
fixed keys for full validation coverage).
"""

from __future__ import annotations

import pytest

from biomed_rag.config.pipeline_config import (
    ConfigError,
    ParsingEngine,
    PipelineConfig,
    VectorStoreBackend,
)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
class TestDefaults:
    def test_default_construction_succeeds(self):
        config = PipelineConfig()
        assert isinstance(config, PipelineConfig)

    def test_default_values_match_design_table(self):
        config = PipelineConfig()
        assert config.max_file_size_bytes == 524_288_000
        assert config.max_filename_length == 255
        assert config.parsing_engine is ParsingEngine.DOCLING
        assert config.parse_timeout_seconds == 300
        assert config.ocr_confidence_threshold == pytest.approx(0.70)
        assert config.ocr_page_timeout_seconds == 60
        assert config.max_chunk_tokens == 512
        assert config.chunk_overlap_tokens == 64
        assert config.embedding_model is None
        assert config.embedding_dimension == 1536
        assert config.embedding_timeout_seconds == 10
        assert config.embedding_max_retries == 3
        assert config.vector_store_backend is VectorStoreBackend.PGVECTOR
        assert config.default_top_k == 5
        assert config.max_query_chars == 4000
        assert config.stage_retry_limit == 3

    def test_config_is_frozen(self):
        config = PipelineConfig()
        with pytest.raises(Exception):
            config.max_chunk_tokens = 256  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Requirement 6.1 - max_chunk_tokens in [128, 2048]
# ---------------------------------------------------------------------------
class TestMaxChunkTokens:
    @pytest.mark.parametrize("value", [128, 512, 1024, 2048])
    def test_in_range_accepted(self, value):
        # overlap must stay below max_chunk_tokens
        config = PipelineConfig(max_chunk_tokens=value, chunk_overlap_tokens=0)
        assert config.max_chunk_tokens == value

    @pytest.mark.parametrize("value", [127, 0, -1, 2049, 5000])
    def test_out_of_range_rejected(self, value):
        with pytest.raises(ConfigError):
            PipelineConfig(max_chunk_tokens=value, chunk_overlap_tokens=0)

    def test_non_int_rejected(self):
        with pytest.raises(ConfigError):
            PipelineConfig(max_chunk_tokens=512.0, chunk_overlap_tokens=0)

    def test_bool_rejected(self):
        with pytest.raises(ConfigError):
            PipelineConfig(max_chunk_tokens=True, chunk_overlap_tokens=0)


# ---------------------------------------------------------------------------
# Requirement 6.2 - chunk_overlap_tokens in [0, max_chunk_tokens - 1]
# ---------------------------------------------------------------------------
class TestChunkOverlapTokens:
    @pytest.mark.parametrize("value", [0, 1, 64, 511])
    def test_in_range_accepted(self, value):
        config = PipelineConfig(max_chunk_tokens=512, chunk_overlap_tokens=value)
        assert config.chunk_overlap_tokens == value

    def test_negative_rejected(self):
        with pytest.raises(ConfigError):
            PipelineConfig(max_chunk_tokens=512, chunk_overlap_tokens=-1)

    def test_equal_to_max_rejected(self):
        # overlap must be strictly less than max_chunk_tokens
        with pytest.raises(ConfigError):
            PipelineConfig(max_chunk_tokens=512, chunk_overlap_tokens=512)

    def test_greater_than_max_rejected(self):
        with pytest.raises(ConfigError):
            PipelineConfig(max_chunk_tokens=512, chunk_overlap_tokens=600)

    def test_dependency_tracks_max_chunk_tokens(self):
        # overlap of 200 is valid when max is 512 but invalid when max is 128
        ok = PipelineConfig(max_chunk_tokens=512, chunk_overlap_tokens=200)
        assert ok.chunk_overlap_tokens == 200
        with pytest.raises(ConfigError):
            PipelineConfig(max_chunk_tokens=128, chunk_overlap_tokens=200)

    def test_upper_bound_equals_max_minus_one(self):
        config = PipelineConfig(max_chunk_tokens=128, chunk_overlap_tokens=127)
        assert config.chunk_overlap_tokens == 127
        with pytest.raises(ConfigError):
            PipelineConfig(max_chunk_tokens=128, chunk_overlap_tokens=128)

    def test_error_message_mentions_dependent_bound(self):
        with pytest.raises(ConfigError) as exc:
            PipelineConfig(max_chunk_tokens=256, chunk_overlap_tokens=256)
        assert "max_chunk_tokens" in str(exc.value)


# ---------------------------------------------------------------------------
# Requirement 7.1 - embedding_dimension in [64, 4096]
# ---------------------------------------------------------------------------
class TestEmbeddingDimension:
    @pytest.mark.parametrize("value", [64, 768, 1536, 4096])
    def test_in_range_accepted(self, value):
        config = PipelineConfig(embedding_dimension=value)
        assert config.embedding_dimension == value

    @pytest.mark.parametrize("value", [63, 0, -1, 4097, 10000])
    def test_out_of_range_rejected(self, value):
        with pytest.raises(ConfigError):
            PipelineConfig(embedding_dimension=value)

    def test_non_int_rejected(self):
        with pytest.raises(ConfigError):
            PipelineConfig(embedding_dimension=1536.0)


# ---------------------------------------------------------------------------
# Requirement 9.1 - default_top_k in [1, 100]
# ---------------------------------------------------------------------------
class TestDefaultTopK:
    @pytest.mark.parametrize("value", [1, 5, 50, 100])
    def test_in_range_accepted(self, value):
        config = PipelineConfig(default_top_k=value)
        assert config.default_top_k == value

    @pytest.mark.parametrize("value", [0, -1, 101, 1000])
    def test_out_of_range_rejected(self, value):
        with pytest.raises(ConfigError):
            PipelineConfig(default_top_k=value)


# ---------------------------------------------------------------------------
# Requirement 10.2 - stage_retry_limit in [0, 10]
# ---------------------------------------------------------------------------
class TestStageRetryLimit:
    @pytest.mark.parametrize("value", [0, 1, 3, 10])
    def test_in_range_accepted(self, value):
        config = PipelineConfig(stage_retry_limit=value)
        assert config.stage_retry_limit == value

    @pytest.mark.parametrize("value", [-1, 11, 100])
    def test_out_of_range_rejected(self, value):
        with pytest.raises(ConfigError):
            PipelineConfig(stage_retry_limit=value)


# ---------------------------------------------------------------------------
# Remaining bounded keys (full validation coverage of the design table)
# ---------------------------------------------------------------------------
class TestMaxFileSizeBytes:
    @pytest.mark.parametrize("value", [1, 1024, 524_288_000])
    def test_in_range_accepted(self, value):
        assert PipelineConfig(max_file_size_bytes=value).max_file_size_bytes == value

    @pytest.mark.parametrize("value", [0, -1, 524_288_001])
    def test_out_of_range_rejected(self, value):
        with pytest.raises(ConfigError):
            PipelineConfig(max_file_size_bytes=value)


class TestParseTimeoutSeconds:
    @pytest.mark.parametrize("value", [1, 300, 10_000])
    def test_positive_accepted(self, value):
        assert PipelineConfig(parse_timeout_seconds=value).parse_timeout_seconds == value

    @pytest.mark.parametrize("value", [0, -1])
    def test_non_positive_rejected(self, value):
        with pytest.raises(ConfigError):
            PipelineConfig(parse_timeout_seconds=value)


class TestOcrConfidenceThreshold:
    @pytest.mark.parametrize("value", [0.0, 0.5, 0.70, 1.0])
    def test_in_range_accepted(self, value):
        config = PipelineConfig(ocr_confidence_threshold=value)
        assert config.ocr_confidence_threshold == pytest.approx(value)

    @pytest.mark.parametrize("value", [-0.01, 1.01, 2.0])
    def test_out_of_range_rejected(self, value):
        with pytest.raises(ConfigError):
            PipelineConfig(ocr_confidence_threshold=value)


class TestOcrPageTimeoutSeconds:
    @pytest.mark.parametrize("value", [1, 60, 600])
    def test_positive_accepted(self, value):
        config = PipelineConfig(ocr_page_timeout_seconds=value)
        assert config.ocr_page_timeout_seconds == value

    @pytest.mark.parametrize("value", [0, -1])
    def test_non_positive_rejected(self, value):
        with pytest.raises(ConfigError):
            PipelineConfig(ocr_page_timeout_seconds=value)


class TestEmbeddingTimeoutSeconds:
    @pytest.mark.parametrize("value", [1, 10, 120])
    def test_positive_accepted(self, value):
        config = PipelineConfig(embedding_timeout_seconds=value)
        assert config.embedding_timeout_seconds == value

    @pytest.mark.parametrize("value", [0, -1])
    def test_non_positive_rejected(self, value):
        with pytest.raises(ConfigError):
            PipelineConfig(embedding_timeout_seconds=value)


class TestEmbeddingModel:
    def test_none_accepted(self):
        assert PipelineConfig(embedding_model=None).embedding_model is None

    def test_non_empty_string_accepted(self):
        config = PipelineConfig(embedding_model="text-embedding-3-small")
        assert config.embedding_model == "text-embedding-3-small"

    @pytest.mark.parametrize("value", ["", "   ", "\t\n"])
    def test_empty_or_whitespace_rejected(self, value):
        with pytest.raises(ConfigError):
            PipelineConfig(embedding_model=value)


# ---------------------------------------------------------------------------
# Enum keys
# ---------------------------------------------------------------------------
class TestEnumKeys:
    def test_parsing_engine_enum_accepted(self):
        config = PipelineConfig(parsing_engine=ParsingEngine.LLAMAPARSE)
        assert config.parsing_engine is ParsingEngine.LLAMAPARSE

    def test_parsing_engine_string_coerced(self):
        config = PipelineConfig(parsing_engine="llamaparse")
        assert config.parsing_engine is ParsingEngine.LLAMAPARSE

    def test_parsing_engine_invalid_rejected(self):
        with pytest.raises(ConfigError):
            PipelineConfig(parsing_engine="unknown-engine")

    def test_vector_store_backend_enum_accepted(self):
        config = PipelineConfig(vector_store_backend=VectorStoreBackend.QDRANT)
        assert config.vector_store_backend is VectorStoreBackend.QDRANT

    def test_vector_store_backend_string_coerced(self):
        config = PipelineConfig(vector_store_backend="qdrant")
        assert config.vector_store_backend is VectorStoreBackend.QDRANT

    def test_vector_store_backend_invalid_rejected(self):
        with pytest.raises(ConfigError):
            PipelineConfig(vector_store_backend="redis")


# ---------------------------------------------------------------------------
# Fixed-value keys
# ---------------------------------------------------------------------------
class TestFixedValueKeys:
    def test_max_filename_length_fixed_accepted(self):
        assert PipelineConfig(max_filename_length=255).max_filename_length == 255

    @pytest.mark.parametrize("value", [254, 256, 0])
    def test_max_filename_length_other_rejected(self, value):
        with pytest.raises(ConfigError):
            PipelineConfig(max_filename_length=value)

    def test_embedding_max_retries_fixed_accepted(self):
        assert PipelineConfig(embedding_max_retries=3).embedding_max_retries == 3

    @pytest.mark.parametrize("value", [0, 2, 4])
    def test_embedding_max_retries_other_rejected(self, value):
        with pytest.raises(ConfigError):
            PipelineConfig(embedding_max_retries=value)

    def test_max_query_chars_fixed_accepted(self):
        assert PipelineConfig(max_query_chars=4000).max_query_chars == 4000

    @pytest.mark.parametrize("value", [3999, 4001, 0])
    def test_max_query_chars_other_rejected(self, value):
        with pytest.raises(ConfigError):
            PipelineConfig(max_query_chars=value)


# ---------------------------------------------------------------------------
# Error semantics
# ---------------------------------------------------------------------------
class TestErrorSemantics:
    def test_config_error_is_value_error(self):
        assert issubclass(ConfigError, ValueError)

    def test_error_message_identifies_offending_key(self):
        with pytest.raises(ConfigError) as exc:
            PipelineConfig(default_top_k=0)
        assert "default_top_k" in str(exc.value)
