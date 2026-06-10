"""Parsed_Document data models (Req 2, 3, 4).

The Parsed_Document is the structured intermediate representation produced by the
Parser: reading-order text blocks, tables, figures, headings, and per-image OCR
error records.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from ._validation import (
    require,
    require_float_in_range,
    require_int_in_range,
)
from .enums import BlockSource, BlockType, OCRErrorKind
from .identifiers import DocumentId


@dataclass
class TextBlock:
    """A single block of text in reading-sequence order (Req 2.1, 2.3, 4.3, 4.4)."""

    type: BlockType
    text: str
    pageNumber: int
    readingOrderPosition: int
    source: BlockSource = BlockSource.TEXT_LAYER
    ocrConfidence: Optional[float] = None
    lowConfidence: bool = False
    headingLevel: Optional[int] = None

    def __post_init__(self) -> None:
        require(isinstance(self.type, BlockType), "TextBlock.type must be a BlockType")
        require(isinstance(self.source, BlockSource), "TextBlock.source must be a BlockSource")
        require(isinstance(self.text, str), "TextBlock.text must be a str")
        require_int_in_range(self.pageNumber, "TextBlock.pageNumber", minimum=0)
        require_int_in_range(
            self.readingOrderPosition, "TextBlock.readingOrderPosition", minimum=0
        )
        if self.ocrConfidence is not None:
            self.ocrConfidence = require_float_in_range(
                self.ocrConfidence, "TextBlock.ocrConfidence", minimum=0.0, maximum=1.0
            )
        require(
            isinstance(self.lowConfidence, bool),
            "TextBlock.lowConfidence must be a bool",
        )
        if self.headingLevel is not None:
            require_int_in_range(self.headingLevel, "TextBlock.headingLevel", minimum=1)


@dataclass
class Cell:
    """A single table cell. Spanning cells live at their top-left index with the
    spanned row/column counts recorded (Req 3.1, 3.2)."""

    rowIndex: int
    colIndex: int
    value: str
    rowSpan: int = 1
    colSpan: int = 1

    def __post_init__(self) -> None:
        require_int_in_range(self.rowIndex, "Cell.rowIndex", minimum=0)
        require_int_in_range(self.colIndex, "Cell.colIndex", minimum=0)
        require(isinstance(self.value, str), "Cell.value must be a str")
        # rowSpan / colSpan are always >= 1 (a cell spans at least itself).
        require_int_in_range(self.rowSpan, "Cell.rowSpan", minimum=1)
        require_int_in_range(self.colSpan, "Cell.colSpan", minimum=1)


@dataclass
class Table:
    """An extracted table (Req 3.1, 3.2, 3.5, 3.6)."""

    pageNumber: int
    readingOrderPosition: int
    cells: List[Cell] = field(default_factory=list)
    degraded: bool = False
    rawText: Optional[str] = None

    def __post_init__(self) -> None:
        require_int_in_range(self.pageNumber, "Table.pageNumber", minimum=0)
        require_int_in_range(
            self.readingOrderPosition, "Table.readingOrderPosition", minimum=0
        )
        require(
            all(isinstance(c, Cell) for c in self.cells),
            "Table.cells must contain only Cell instances",
        )
        require(isinstance(self.degraded, bool), "Table.degraded must be a bool")
        # A degraded table retains the raw region text for downstream use (Req 3.6).
        if self.rawText is not None:
            require(isinstance(self.rawText, str), "Table.rawText must be a str or None")


@dataclass
class Figure:
    """An extracted figure/chart with an optional caption (Req 3.3, 3.4, 3.5)."""

    pageNumber: int
    readingOrderPosition: int
    imageRef: str
    caption: Optional[str] = None

    def __post_init__(self) -> None:
        require_int_in_range(self.pageNumber, "Figure.pageNumber", minimum=0)
        require_int_in_range(
            self.readingOrderPosition, "Figure.readingOrderPosition", minimum=0
        )
        require(isinstance(self.imageRef, str), "Figure.imageRef must be a str")
        # caption is None exactly when no caption was detected (Req 3.4).
        if self.caption is not None:
            require(isinstance(self.caption, str), "Figure.caption must be a str or None")


@dataclass
class Heading:
    """A section heading with its nesting level (Req 2.4)."""

    level: int
    text: str
    pageNumber: int
    readingOrderPosition: int

    def __post_init__(self) -> None:
        require_int_in_range(self.level, "Heading.level", minimum=1)
        require(isinstance(self.text, str), "Heading.text must be a str")
        require_int_in_range(self.pageNumber, "Heading.pageNumber", minimum=0)
        require_int_in_range(
            self.readingOrderPosition, "Heading.readingOrderPosition", minimum=0
        )


@dataclass
class ImageOCRError:
    """A per-image OCR failure indication that does not abort the whole parse
    (Req 4.5, 4.6)."""

    imageRef: str
    pageNumber: int
    kind: OCRErrorKind

    def __post_init__(self) -> None:
        require(isinstance(self.imageRef, str), "ImageOCRError.imageRef must be a str")
        require_int_in_range(self.pageNumber, "ImageOCRError.pageNumber", minimum=0)
        require(
            isinstance(self.kind, OCRErrorKind),
            "ImageOCRError.kind must be an OCRErrorKind",
        )


@dataclass
class ParsedDocument:
    """The structured output of the Parser (Req 2.1, 2.4, 3.x, 4.x)."""

    documentId: DocumentId
    blocks: List[TextBlock] = field(default_factory=list)
    tables: List[Table] = field(default_factory=list)
    figures: List[Figure] = field(default_factory=list)
    headings: List[Heading] = field(default_factory=list)
    ocrErrors: List[ImageOCRError] = field(default_factory=list)

    def __post_init__(self) -> None:
        require(
            isinstance(self.documentId, str) and len(self.documentId) > 0,
            "ParsedDocument.documentId must be a non-empty str",
        )
        require(
            all(isinstance(b, TextBlock) for b in self.blocks),
            "ParsedDocument.blocks must contain only TextBlock instances",
        )
        require(
            all(isinstance(t, Table) for t in self.tables),
            "ParsedDocument.tables must contain only Table instances",
        )
        require(
            all(isinstance(f, Figure) for f in self.figures),
            "ParsedDocument.figures must contain only Figure instances",
        )
        require(
            all(isinstance(h, Heading) for h in self.headings),
            "ParsedDocument.headings must contain only Heading instances",
        )
        require(
            all(isinstance(e, ImageOCRError) for e in self.ocrErrors),
            "ParsedDocument.ocrErrors must contain only ImageOCRError instances",
        )
