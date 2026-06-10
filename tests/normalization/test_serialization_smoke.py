"""Smoke tests for durable NormalizedDocument serialization (task 7.3).

Covers Req 5.6 (serialize -> deserialize round-trip producing a structurally
equivalent document) and Req 5.5 (page-number + reading-order metadata
preserved). The fixture document exercises every :class:`ElementKind`: TEXT,
HEADING, TABLE (with a spanning cell, degraded flag, and raw text) and FIGURE
(with and without a caption).
"""

from __future__ import annotations

from biomed_rag.models.enums import ElementKind
from biomed_rag.models.normalized import (
    ContentElement,
    FigurePayload,
    NormalizedDocument,
    TablePayload,
    TextPayload,
)
from biomed_rag.models.parsed import Cell
from biomed_rag.normalization import Normalizer, deserialize, serialize


def _all_kinds_document() -> NormalizedDocument:
    return NormalizedDocument(
        documentId="doc-roundtrip",
        elements=[
            ContentElement(
                kind=ElementKind.HEADING,
                pageNumber=1,
                readingOrderPosition=0,
                payload=TextPayload(text="Chapter 1", headingLevel=1),
                headingPath=[],
            ),
            ContentElement(
                kind=ElementKind.TEXT,
                pageNumber=1,
                readingOrderPosition=1,
                payload=TextPayload(text="Body paragraph with detail."),
                headingPath=["Chapter 1"],
            ),
            ContentElement(
                kind=ElementKind.TABLE,
                pageNumber=2,
                readingOrderPosition=2,
                payload=TablePayload(
                    cells=[
                        Cell(rowIndex=0, colIndex=0, value="Header", rowSpan=1, colSpan=2),
                        Cell(rowIndex=1, colIndex=0, value="A"),
                        Cell(rowIndex=1, colIndex=1, value="B"),
                    ],
                    degraded=True,
                    rawText="Header | A | B",
                ),
            ),
            ContentElement(
                kind=ElementKind.FIGURE,
                pageNumber=3,
                readingOrderPosition=3,
                payload=FigurePayload(imageRef="img-1", caption="Figure 1"),
            ),
            ContentElement(
                kind=ElementKind.FIGURE,
                pageNumber=3,
                readingOrderPosition=4,
                payload=FigurePayload(imageRef="img-2", caption=None),
            ),
        ],
    )


def test_serialize_returns_bytes():
    data = serialize(_all_kinds_document())
    assert isinstance(data, bytes)


def test_roundtrip_preserves_all_kinds_and_metadata():
    original = _all_kinds_document()

    restored = deserialize(serialize(original))

    assert isinstance(restored, NormalizedDocument)
    assert restored == original


def test_roundtrip_via_normalizer_methods():
    original = _all_kinds_document()

    restored = Normalizer.deserialize(Normalizer.serialize(original))

    assert restored == original


def test_roundtrip_empty_document():
    original = NormalizedDocument(documentId="doc-empty", elements=[])

    restored = deserialize(serialize(original))

    assert restored == original
