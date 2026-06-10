"""Parser and pluggable Parsing_Engine port (Req 2, 3).

This package defines the engine-neutral parsing port and supporting shapes:

* :class:`ParsingEngine` — the pluggable port (Req 2.2).
* :class:`RawParseResult` and friends — the shared shape adapters map into.
* :class:`ParsingEngineRegistry` / :data:`default_registry` — config-driven
  engine selection.
* :class:`MockParsingEngine` — a deterministic adapter for tests.

Concrete adapters (Docling, LlamaParse) and the ``Parser`` itself are added in
later tasks.
"""

from __future__ import annotations

from biomed_rag.config import ParsingEngine as _ParsingEngineChoice

from .docling_adapter import DoclingAdapter
from .engine import (
    EngineUnavailableError,
    ParseError,
    ParseTimeoutError,
    ParsingEngine,
)
from .llamaparse_adapter import LlamaParseAdapter
from .mock_engine import MockParsingEngine
from .parser import Parser, ParseFailure, ParseFailureKind
from .raw_result import (
    BBox,
    RawBlock,
    RawFigure,
    RawImage,
    RawPage,
    RawParseResult,
    RawTable,
    RawTableCell,
    SourceDocument,
)
from .registry import (
    EngineFactory,
    ParsingEngineNotRegisteredError,
    ParsingEngineRegistry,
    default_registry,
)

__all__ = [
    # port
    "ParsingEngine",
    "ParseError",
    "ParseTimeoutError",
    "EngineUnavailableError",
    # raw shape
    "SourceDocument",
    "BBox",
    "RawBlock",
    "RawTableCell",
    "RawTable",
    "RawFigure",
    "RawPage",
    "RawImage",
    "RawParseResult",
    # registry
    "ParsingEngineRegistry",
    "ParsingEngineNotRegisteredError",
    "EngineFactory",
    "default_registry",
    # test support
    "MockParsingEngine",
    # concrete adapters
    "DoclingAdapter",
    "LlamaParseAdapter",
    # parser
    "Parser",
    "ParseFailure",
    "ParseFailureKind",
]

# Register the concrete adapters with the process-wide default registry so a
# PipelineConfig built anywhere can select its engine by configuration (Req 2.2).
# Registration is idempotent across re-imports via replace=True.
default_registry.register(
    _ParsingEngineChoice.DOCLING, DoclingAdapter, replace=True
)
default_registry.register(
    _ParsingEngineChoice.LLAMAPARSE, LlamaParseAdapter, replace=True
)
