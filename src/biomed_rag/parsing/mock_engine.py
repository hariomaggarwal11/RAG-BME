"""A deterministic mock :class:`ParsingEngine` for tests (Req 2.2).

The mock exercises stage logic through the port without a real backend. It is
fully deterministic: the same :class:`SourceDocument` always yields the same
:class:`RawParseResult`. Test-only knobs let callers simulate the
availability and failure conditions the Parser must handle:

* ``available=False``      → :meth:`is_available` returns False (Req 2.6).
* ``raise_on_parse=...``   → :meth:`parse` raises the given exception, used to
  simulate parse errors (Req 2.5) and timeouts (Req 2.7).
* ``preset_result=...``    → :meth:`parse` returns a caller-supplied result,
  used to drive the Parser with a precise raw shape.

With no knobs set, :meth:`parse` derives a deterministic result from the
document bytes by splitting decoded text into blank-line-separated paragraphs.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Optional

from .engine import EngineUnavailableError, ParsingEngine
from .raw_result import RawBlock, RawPage, RawParseResult, SourceDocument


class MockParsingEngine(ParsingEngine):
    """Deterministic in-memory parsing engine for property and unit tests."""

    def __init__(
        self,
        *,
        engine_id: str = "mock",
        available: bool = True,
        preset_result: Optional[RawParseResult] = None,
        raise_on_parse: Optional[BaseException] = None,
    ) -> None:
        self._engine_id = engine_id
        self._available = available
        self._preset_result = preset_result
        self._raise_on_parse = raise_on_parse

    # -- port methods -----------------------------------------------------
    def engine_id(self) -> str:
        return self._engine_id

    def is_available(self) -> bool:
        return self._available

    def parse(
        self,
        doc: SourceDocument,
        deadline: Optional[float] = None,
    ) -> RawParseResult:
        if not isinstance(doc, SourceDocument):
            raise TypeError("doc must be a SourceDocument")

        if self._raise_on_parse is not None:
            raise self._raise_on_parse

        if not self._available:
            raise EngineUnavailableError(
                f"parsing engine {self._engine_id!r} is unavailable"
            )

        if self._preset_result is not None:
            # Stamp the engine id so the result is attributable to this engine.
            return replace(self._preset_result, engine_id=self._engine_id)

        return self._derive_result(doc)

    # -- deterministic derivation ----------------------------------------
    def _derive_result(self, doc: SourceDocument) -> RawParseResult:
        """Build a stable result from the document bytes.

        Text is decoded leniently and split on blank lines into paragraphs.
        Each non-empty paragraph becomes a :class:`RawBlock` on page 0 with a
        reading order implied by its index. The derivation is pure, so identical
        bytes always produce an identical result.
        """
        text = doc.raw_bytes.decode("utf-8", errors="replace")
        paragraphs = [p.strip() for p in text.split("\n\n")]
        blocks = [
            RawBlock(text=p, page_number=0, kind="paragraph")
            for p in paragraphs
            if p
        ]
        pages = [RawPage(page_number=0, has_text_layer=bool(blocks))]
        return RawParseResult(
            engine_id=self._engine_id,
            blocks=blocks,
            pages=pages,
        )
