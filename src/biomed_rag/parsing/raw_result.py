"""Shared raw-parse data shapes for the pluggable Parsing_Engine port (Req 2.2).

A :class:`ParsingEngine` adapter (e.g. Docling, LlamaParse) maps the raw output
of its underlying engine into the engine-neutral :class:`RawParseResult` shape
defined here. The :class:`~biomed_rag.parsing.Parser` (implemented in a later
task) consumes a ``RawParseResult`` and produces the canonical
:class:`~biomed_rag.models.ParsedDocument` — assigning reading-order positions,
preserving heading hierarchy, and resolving table/figure structure.

Keeping this intermediate shape engine-neutral is what lets stage logic depend
only on the port and never on a concrete backend (design: ports-and-adapters).

This module defines the *shape* only; it deliberately contains no parsing or
ordering logic (those belong to the concrete adapters and the Parser).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from biomed_rag.models import Format


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class BBox:
    """An axis-aligned layout bounding box in page coordinates.

    Adapters supply a box per element where the engine exposes layout geometry.
    The Parser uses it to order multi-column layouts (Req 2.3). Coordinates are
    in an engine-defined unit with the origin at the top-left of the page.
    """

    x0: float
    y0: float
    x1: float
    y1: float

    def __post_init__(self) -> None:
        for name in ("x0", "y0", "x1", "y1"):
            value = getattr(self, name)
            _require(
                isinstance(value, (int, float)) and not isinstance(value, bool),
                f"BBox.{name} must be a number, got {type(value).__name__}",
            )

    def as_tuple(self) -> Tuple[float, float, float, float]:
        return (float(self.x0), float(self.y0), float(self.x1), float(self.y1))


@dataclass
class SourceDocument:
    """The raw input handed to a parsing engine.

    ``raw_bytes`` are the original document bytes; ``doc_format`` is the
    validated format when known. ``document_id`` ties the parse result back to
    the originating Processing_Job.
    """

    document_id: str
    raw_bytes: bytes
    doc_format: Optional[Format] = None

    def __post_init__(self) -> None:
        _require(
            isinstance(self.document_id, str) and len(self.document_id) > 0,
            "SourceDocument.document_id must be a non-empty str",
        )
        _require(
            isinstance(self.raw_bytes, (bytes, bytearray)),
            "SourceDocument.raw_bytes must be bytes",
        )
        if isinstance(self.raw_bytes, bytearray):
            object.__setattr__(self, "raw_bytes", bytes(self.raw_bytes))
        if self.doc_format is not None:
            _require(
                isinstance(self.doc_format, Format),
                "SourceDocument.doc_format must be a Format or None",
            )


@dataclass
class RawBlock:
    """A raw text block as produced by an engine, before reading-order assignment.

    ``kind`` is the engine's best-effort type hint (e.g. ``"paragraph"``,
    ``"heading"``, ``"caption"``). ``heading_level`` carries the nesting level
    when the block is a heading (Req 2.4). ``bbox`` and ``column`` provide the
    layout hints the Parser uses to order multi-column content (Req 2.3).
    """

    text: str
    page_number: int
    kind: str = "paragraph"
    heading_level: Optional[int] = None
    bbox: Optional[BBox] = None
    column: Optional[int] = None
    from_ocr: bool = False

    def __post_init__(self) -> None:
        _require(isinstance(self.text, str), "RawBlock.text must be a str")
        _require(
            isinstance(self.page_number, int) and not isinstance(self.page_number, bool),
            "RawBlock.page_number must be an int",
        )
        _require(self.page_number >= 0, "RawBlock.page_number must be >= 0")
        _require(isinstance(self.kind, str), "RawBlock.kind must be a str")
        if self.heading_level is not None:
            _require(
                isinstance(self.heading_level, int)
                and not isinstance(self.heading_level, bool)
                and self.heading_level >= 1,
                "RawBlock.heading_level must be an int >= 1 or None",
            )
        if self.bbox is not None:
            _require(isinstance(self.bbox, BBox), "RawBlock.bbox must be a BBox or None")
        if self.column is not None:
            _require(
                isinstance(self.column, int)
                and not isinstance(self.column, bool)
                and self.column >= 0,
                "RawBlock.column must be an int >= 0 or None",
            )
        _require(isinstance(self.from_ocr, bool), "RawBlock.from_ocr must be a bool")


@dataclass
class RawTableCell:
    """A raw table cell. Spanning cells are reported at their top-left index with
    the spanned row/column counts (Req 3.2)."""

    row_index: int
    col_index: int
    value: str
    row_span: int = 1
    col_span: int = 1

    def __post_init__(self) -> None:
        for name in ("row_index", "col_index"):
            value = getattr(self, name)
            _require(
                isinstance(value, int) and not isinstance(value, bool) and value >= 0,
                f"RawTableCell.{name} must be an int >= 0",
            )
        _require(isinstance(self.value, str), "RawTableCell.value must be a str")
        for name in ("row_span", "col_span"):
            value = getattr(self, name)
            _require(
                isinstance(value, int) and not isinstance(value, bool) and value >= 1,
                f"RawTableCell.{name} must be an int >= 1",
            )


@dataclass
class RawTable:
    """A raw table region. ``degraded`` marks a region whose structured
    extraction failed; ``raw_text`` retains its text in that case (Req 3.6)."""

    page_number: int
    cells: List[RawTableCell] = field(default_factory=list)
    bbox: Optional[BBox] = None
    degraded: bool = False
    raw_text: Optional[str] = None

    def __post_init__(self) -> None:
        _require(
            isinstance(self.page_number, int) and not isinstance(self.page_number, bool),
            "RawTable.page_number must be an int",
        )
        _require(self.page_number >= 0, "RawTable.page_number must be >= 0")
        _require(
            all(isinstance(c, RawTableCell) for c in self.cells),
            "RawTable.cells must contain only RawTableCell instances",
        )
        if self.bbox is not None:
            _require(isinstance(self.bbox, BBox), "RawTable.bbox must be a BBox or None")
        _require(isinstance(self.degraded, bool), "RawTable.degraded must be a bool")
        if self.raw_text is not None:
            _require(isinstance(self.raw_text, str), "RawTable.raw_text must be a str or None")


@dataclass
class RawFigure:
    """A raw figure/chart region with an optional caption (Req 3.3, 3.4)."""

    page_number: int
    image_ref: str
    caption: Optional[str] = None
    bbox: Optional[BBox] = None

    def __post_init__(self) -> None:
        _require(
            isinstance(self.page_number, int) and not isinstance(self.page_number, bool),
            "RawFigure.page_number must be an int",
        )
        _require(self.page_number >= 0, "RawFigure.page_number must be >= 0")
        _require(isinstance(self.image_ref, str), "RawFigure.image_ref must be a str")
        if self.caption is not None:
            _require(isinstance(self.caption, str), "RawFigure.caption must be a str or None")
        if self.bbox is not None:
            _require(isinstance(self.bbox, BBox), "RawFigure.bbox must be a BBox or None")


@dataclass
class RawImage:
    """A raw embedded image that may carry text content (Req 4.2).

    The engine flags an embedded image as carrying text via ``has_text``; the
    Parser routes such images through the OCR_Processor's
    :meth:`process_embedded_image` and stores any recovered text as an
    ``OCR_TEXT`` block, while recording a per-image error for an
    unreadable/corrupt/unsupported image without aborting the parse (Req 4.2,
    4.5). ``image_ref`` identifies the image for error reporting.
    """

    page_number: int
    image_ref: str
    has_text: bool = False
    bbox: Optional["BBox"] = None

    def __post_init__(self) -> None:
        _require(
            isinstance(self.page_number, int) and not isinstance(self.page_number, bool),
            "RawImage.page_number must be an int",
        )
        _require(self.page_number >= 0, "RawImage.page_number must be >= 0")
        _require(isinstance(self.image_ref, str), "RawImage.image_ref must be a str")
        _require(isinstance(self.has_text, bool), "RawImage.has_text must be a bool")
        if self.bbox is not None:
            _require(isinstance(self.bbox, BBox), "RawImage.bbox must be a BBox or None")


@dataclass
class RawPage:
    """Per-page metadata. ``has_text_layer`` is False for image-only pages that
    the Parser must route through OCR (Req 4.1); ``page_image_ref`` points at the
    page raster in that case."""

    page_number: int
    has_text_layer: bool = True
    page_image_ref: Optional[str] = None

    def __post_init__(self) -> None:
        _require(
            isinstance(self.page_number, int) and not isinstance(self.page_number, bool),
            "RawPage.page_number must be an int",
        )
        _require(self.page_number >= 0, "RawPage.page_number must be >= 0")
        _require(
            isinstance(self.has_text_layer, bool),
            "RawPage.has_text_layer must be a bool",
        )
        if self.page_image_ref is not None:
            _require(
                isinstance(self.page_image_ref, str),
                "RawPage.page_image_ref must be a str or None",
            )


@dataclass
class RawParseResult:
    """The engine-neutral output of a :class:`ParsingEngine` adapter.

    All concrete adapters map their backend's output into this single shape so
    that the Parser and downstream stages never depend on a specific engine
    (design: ports-and-adapters, Req 2.2). ``engine_id`` records which engine
    produced the result for traceability and failure reporting (Req 2.6).
    """

    engine_id: str
    blocks: List[RawBlock] = field(default_factory=list)
    tables: List[RawTable] = field(default_factory=list)
    figures: List[RawFigure] = field(default_factory=list)
    pages: List[RawPage] = field(default_factory=list)
    images: List["RawImage"] = field(default_factory=list)

    def __post_init__(self) -> None:
        _require(
            isinstance(self.engine_id, str) and len(self.engine_id) > 0,
            "RawParseResult.engine_id must be a non-empty str",
        )
        _require(
            all(isinstance(b, RawBlock) for b in self.blocks),
            "RawParseResult.blocks must contain only RawBlock instances",
        )
        _require(
            all(isinstance(t, RawTable) for t in self.tables),
            "RawParseResult.tables must contain only RawTable instances",
        )
        _require(
            all(isinstance(f, RawFigure) for f in self.figures),
            "RawParseResult.figures must contain only RawFigure instances",
        )
        _require(
            all(isinstance(p, RawPage) for p in self.pages),
            "RawParseResult.pages must contain only RawPage instances",
        )
        _require(
            all(isinstance(i, RawImage) for i in self.images),
            "RawParseResult.images must contain only RawImage instances",
        )

    def is_empty(self) -> bool:
        """True when the engine extracted no content of any kind.

        The Parser uses this to detect the no-extractable-content failure
        (Req 2.8); page metadata alone does not count as extracted content.
        """
        return not (self.blocks or self.tables or self.figures)
