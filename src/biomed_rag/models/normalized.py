"""NormalizedDocument data models (Req 5).

The NormalizedDocument is the canonical, content-preserving representation
produced by the Normalizer. It is the unit that is serialized/deserialized for
the round-trip property (Req 5.6). Each content element carries its page number
and reading-order position (Req 5.5).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Union

from ._validation import require, require_int_in_range
from .enums import ElementKind
from .identifiers import DocumentId
from .parsed import Cell


@dataclass
class TextPayload:
    """Payload for a TEXT or HEADING content element."""

    text: str
    headingLevel: Optional[int] = None

    def __post_init__(self) -> None:
        require(isinstance(self.text, str), "TextPayload.text must be a str")
        if self.headingLevel is not None:
            require_int_in_range(self.headingLevel, "TextPayload.headingLevel", minimum=1)


@dataclass
class TablePayload:
    """Payload for a TABLE content element (Req 3.1, 3.2, 3.6)."""

    cells: List[Cell] = field(default_factory=list)
    degraded: bool = False
    rawText: Optional[str] = None

    def __post_init__(self) -> None:
        require(
            all(isinstance(c, Cell) for c in self.cells),
            "TablePayload.cells must contain only Cell instances",
        )
        require(isinstance(self.degraded, bool), "TablePayload.degraded must be a bool")
        if self.rawText is not None:
            require(
                isinstance(self.rawText, str), "TablePayload.rawText must be a str or None"
            )


@dataclass
class FigurePayload:
    """Payload for a FIGURE content element (Req 3.3, 3.4)."""

    imageRef: str
    caption: Optional[str] = None

    def __post_init__(self) -> None:
        require(isinstance(self.imageRef, str), "FigurePayload.imageRef must be a str")
        if self.caption is not None:
            require(
                isinstance(self.caption, str), "FigurePayload.caption must be a str or None"
            )


Payload = Union[TextPayload, TablePayload, FigurePayload]

# Which payload type each element kind requires.
_PAYLOAD_FOR_KIND = {
    ElementKind.TEXT: TextPayload,
    ElementKind.HEADING: TextPayload,
    ElementKind.TABLE: TablePayload,
    ElementKind.FIGURE: FigurePayload,
}


@dataclass
class ContentElement:
    """A canonical content element carrying page + reading-order metadata
    (Req 5.4, 5.5)."""

    kind: ElementKind
    pageNumber: int
    readingOrderPosition: int
    payload: Payload
    headingPath: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        require(
            isinstance(self.kind, ElementKind),
            "ContentElement.kind must be an ElementKind",
        )
        require_int_in_range(self.pageNumber, "ContentElement.pageNumber", minimum=0)
        require_int_in_range(
            self.readingOrderPosition, "ContentElement.readingOrderPosition", minimum=0
        )
        require(
            all(isinstance(h, str) for h in self.headingPath),
            "ContentElement.headingPath must be a list of str",
        )
        expected = _PAYLOAD_FOR_KIND[self.kind]
        require(
            isinstance(self.payload, expected),
            f"ContentElement.payload for kind {self.kind.name} must be a "
            f"{expected.__name__}, got {type(self.payload).__name__}",
        )


@dataclass
class NormalizedDocument:
    """The canonical normalized representation (Req 5.4, 5.5, 5.6)."""

    documentId: DocumentId
    elements: List[ContentElement] = field(default_factory=list)

    def __post_init__(self) -> None:
        require(
            isinstance(self.documentId, str) and len(self.documentId) > 0,
            "NormalizedDocument.documentId must be a non-empty str",
        )
        require(
            all(isinstance(e, ContentElement) for e in self.elements),
            "NormalizedDocument.elements must contain only ContentElement instances",
        )
