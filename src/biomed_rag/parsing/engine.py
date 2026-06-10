"""The pluggable Parsing_Engine port (Req 2.2).

``ParsingEngine`` is the stable interface every parsing backend implements.
Concrete adapters (DoclingAdapter, LlamaParseAdapter — implemented in a later
task) sit behind this port so the Parser and the rest of the pipeline depend
only on the contract, never on a specific engine (design: ports-and-adapters).

This module defines the port and the parse-time exceptions an engine may raise.
The Parser translates these into fail-closed Processing_Job outcomes (Req 2.5,
2.6, 2.7); the failure-handling policy itself lives in the Parser, not here.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from .raw_result import RawParseResult, SourceDocument


class ParseError(Exception):
    """Raised by an engine when it cannot parse a document (Req 2.5)."""


class ParseTimeoutError(ParseError):
    """Raised by an engine when parsing exceeds the supplied deadline (Req 2.7)."""


class EngineUnavailableError(ParseError):
    """Raised when an engine is asked to parse while unavailable (Req 2.6).

    Callers should prefer checking :meth:`ParsingEngine.is_available` first; this
    exception exists so an engine that becomes unavailable mid-call can still
    signal the condition unambiguously.
    """


class ParsingEngine(ABC):
    """Port for a pluggable document parsing backend (Req 2.2).

    Implementations are expected to be side-effect free with respect to the
    pipeline's state: they read a :class:`SourceDocument` and return a
    :class:`RawParseResult`, leaving job-state transitions to the Parser.
    """

    @abstractmethod
    def engine_id(self) -> str:
        """Return the stable identifier of this engine (e.g. ``"docling"``).

        Used for engine selection and for recording the engine in failure
        reports (Req 2.6).
        """
        raise NotImplementedError

    @abstractmethod
    def is_available(self) -> bool:
        """Return whether the engine is ready to parse.

        The Parser checks this before parsing and fails the job closed when an
        engine is unavailable (Req 2.6).
        """
        raise NotImplementedError

    @abstractmethod
    def parse(
        self,
        doc: SourceDocument,
        deadline: Optional[float] = None,
    ) -> RawParseResult:
        """Parse ``doc`` into the engine-neutral :class:`RawParseResult` shape.

        ``deadline`` is an optional monotonic-clock timestamp (seconds, as from
        :func:`time.monotonic`) past which the engine should abort and raise
        :class:`ParseTimeoutError` (Req 2.7). ``None`` means no deadline.

        Raises:
            ParseError: the document could not be parsed (Req 2.5).
            ParseTimeoutError: the deadline was exceeded (Req 2.7).
            EngineUnavailableError: the engine became unavailable (Req 2.6).
        """
        raise NotImplementedError
