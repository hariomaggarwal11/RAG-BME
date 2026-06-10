"""Validated configuration model for the biomedical RAG pipeline.

A single :class:`PipelineConfig` holds every bounded parameter for the
pipeline. Bounds are enforced at construction time so out-of-range values are
rejected before any job runs (design: Configuration Model table).

Requirements covered: 1.2, 1.6, 1.8, 2.2, 2.7, 4.4, 4.6, 6.1, 6.2, 7.1, 7.2,
7.6, 8.x, 9.1, 9.2, 9.3, 10.2.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from enum import Enum
from typing import Optional


class ConfigError(ValueError):
    """Raised when a :class:`PipelineConfig` is constructed with an invalid value.

    The message identifies the offending key, the supplied value, and the
    allowed range so callers can correct the configuration before any job runs.
    """


class ParsingEngine(str, Enum):
    """Supported pluggable parsing engines (design Req 2.2)."""

    DOCLING = "docling"
    LLAMAPARSE = "llamaparse"


class VectorStoreBackend(str, Enum):
    """Supported pluggable vector store backends (design Req 8.x)."""

    PGVECTOR = "pgvector"
    QDRANT = "qdrant"


# Fixed values mandated by the design table. These keys are not freely tunable;
# they must equal the fixed value, but are represented as fields so the full
# configuration surface is explicit.
_FIXED_MAX_FILENAME_LENGTH = 255
_FIXED_EMBEDDING_MAX_RETRIES = 3
_FIXED_MAX_QUERY_CHARS = 4000

_MAX_FILE_SIZE_BYTES_UPPER = 524_288_000  # 500 MB


@dataclass(frozen=True)
class PipelineConfig:
    """Immutable, fully validated pipeline configuration.

    Every field corresponds to a row in the design's Configuration Model table
    with its type, bounds, and default. Validation runs in ``__post_init__`` so
    an invalid configuration can never be observed by the rest of the system.
    """

    # filename present and <= 255 chars (Req 1.8); byte size in [1, 500 MB] (Req 1.2, 1.6)
    max_file_size_bytes: int = _MAX_FILE_SIZE_BYTES_UPPER
    max_filename_length: int = _FIXED_MAX_FILENAME_LENGTH

    # Parser (Req 2.2, 2.7)
    parsing_engine: ParsingEngine = ParsingEngine.DOCLING
    parse_timeout_seconds: int = 300

    # OCR (Req 4.4, 4.6)
    ocr_confidence_threshold: float = 0.70
    ocr_page_timeout_seconds: int = 60

    # Chunking (Req 6.1, 6.2)
    max_chunk_tokens: int = 512
    chunk_overlap_tokens: int = 64

    # Embedding (Req 7.1, 7.2, 7.3, 7.5, 7.6)
    embedding_model: Optional[str] = None
    embedding_dimension: int = 1536
    embedding_timeout_seconds: int = 10
    embedding_max_retries: int = _FIXED_EMBEDDING_MAX_RETRIES

    # Storage (Req 8.x)
    vector_store_backend: VectorStoreBackend = VectorStoreBackend.PGVECTOR

    # Retrieval (Req 9.1, 9.2, 9.3)
    default_top_k: int = 5
    max_query_chars: int = _FIXED_MAX_QUERY_CHARS

    # Orchestration (Req 10.2)
    stage_retry_limit: int = 3

    def __post_init__(self) -> None:
        self._coerce_enums()
        self._validate()

    # -- enum coercion ----------------------------------------------------
    def _coerce_enums(self) -> None:
        """Allow string values for enum fields, normalizing them to the enum.

        Frozen dataclasses require ``object.__setattr__`` to mutate fields.
        """
        if not isinstance(self.parsing_engine, ParsingEngine):
            try:
                object.__setattr__(
                    self, "parsing_engine", ParsingEngine(self.parsing_engine)
                )
            except ValueError:
                allowed = ", ".join(e.value for e in ParsingEngine)
                raise ConfigError(
                    f"parsing_engine={self.parsing_engine!r} is invalid; "
                    f"allowed values are: {allowed}"
                ) from None

        if not isinstance(self.vector_store_backend, VectorStoreBackend):
            try:
                object.__setattr__(
                    self,
                    "vector_store_backend",
                    VectorStoreBackend(self.vector_store_backend),
                )
            except ValueError:
                allowed = ", ".join(e.value for e in VectorStoreBackend)
                raise ConfigError(
                    f"vector_store_backend={self.vector_store_backend!r} is "
                    f"invalid; allowed values are: {allowed}"
                ) from None

    # -- validation -------------------------------------------------------
    def _validate(self) -> None:
        self._check_type("max_file_size_bytes", int)
        self._check_range(
            "max_file_size_bytes", self.max_file_size_bytes, 1, _MAX_FILE_SIZE_BYTES_UPPER
        )

        self._check_type("max_filename_length", int)
        self._check_fixed("max_filename_length", self.max_filename_length, _FIXED_MAX_FILENAME_LENGTH)

        self._check_type("parse_timeout_seconds", int)
        self._check_min_exclusive("parse_timeout_seconds", self.parse_timeout_seconds, 0)

        self._check_type("ocr_confidence_threshold", (int, float))
        self._check_range(
            "ocr_confidence_threshold", self.ocr_confidence_threshold, 0.0, 1.0
        )

        self._check_type("ocr_page_timeout_seconds", int)
        self._check_min_exclusive("ocr_page_timeout_seconds", self.ocr_page_timeout_seconds, 0)

        self._check_type("max_chunk_tokens", int)
        self._check_range("max_chunk_tokens", self.max_chunk_tokens, 128, 2048)

        self._check_type("chunk_overlap_tokens", int)
        # Dependent bound: chunk_overlap_tokens in [0, max_chunk_tokens - 1].
        self._check_range(
            "chunk_overlap_tokens",
            self.chunk_overlap_tokens,
            0,
            self.max_chunk_tokens - 1,
            extra_context=(
                f" (upper bound is max_chunk_tokens - 1 = {self.max_chunk_tokens - 1})"
            ),
        )

        # embedding_model: registered model id; may be None (no default model).
        # The registry validates the concrete id later; here we only reject an
        # empty/whitespace string which can never name a registered model.
        if self.embedding_model is not None:
            self._check_type("embedding_model", str)
            if not self.embedding_model.strip():
                raise ConfigError(
                    "embedding_model must be a non-empty registered model id or None; "
                    f"got {self.embedding_model!r}"
                )

        self._check_type("embedding_dimension", int)
        self._check_range("embedding_dimension", self.embedding_dimension, 64, 4096)

        self._check_type("embedding_timeout_seconds", int)
        self._check_min_exclusive(
            "embedding_timeout_seconds", self.embedding_timeout_seconds, 0
        )

        self._check_type("embedding_max_retries", int)
        self._check_fixed(
            "embedding_max_retries", self.embedding_max_retries, _FIXED_EMBEDDING_MAX_RETRIES
        )

        self._check_type("default_top_k", int)
        self._check_range("default_top_k", self.default_top_k, 1, 100)

        self._check_type("max_query_chars", int)
        self._check_fixed("max_query_chars", self.max_query_chars, _FIXED_MAX_QUERY_CHARS)

        self._check_type("stage_retry_limit", int)
        self._check_range("stage_retry_limit", self.stage_retry_limit, 0, 10)

    # -- check helpers ----------------------------------------------------
    def _check_type(self, name: str, expected) -> None:
        value = getattr(self, name)
        # bool is a subclass of int; reject it for numeric fields to avoid
        # silently accepting True/False as 1/0.
        if isinstance(value, bool) and expected is not bool:
            raise ConfigError(
                f"{name} must be of type {self._type_name(expected)}; got bool {value!r}"
            )
        if not isinstance(value, expected):
            raise ConfigError(
                f"{name} must be of type {self._type_name(expected)}; "
                f"got {type(value).__name__} {value!r}"
            )

    def _check_range(self, name, value, low, high, extra_context: str = "") -> None:
        if not (low <= value <= high):
            raise ConfigError(
                f"{name}={value!r} is out of range; must be in [{low}, {high}]"
                f"{extra_context}"
            )

    def _check_min_exclusive(self, name, value, low) -> None:
        if not value > low:
            raise ConfigError(
                f"{name}={value!r} is out of range; must be greater than {low}"
            )

    def _check_fixed(self, name, value, fixed) -> None:
        if value != fixed:
            raise ConfigError(
                f"{name}={value!r} is invalid; this is a fixed value and must equal {fixed}"
            )

    @staticmethod
    def _type_name(expected) -> str:
        if isinstance(expected, tuple):
            return " or ".join(t.__name__ for t in expected)
        return expected.__name__

    # -- convenience ------------------------------------------------------
    @classmethod
    def field_names(cls) -> tuple[str, ...]:
        """Return the ordered tuple of configuration field names."""
        return tuple(f.name for f in fields(cls))
