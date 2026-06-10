"""Smoke tests for canonical NormalizedDocument production (task 7.2).

Covers Req 5.4 (content-preserving canonical elements: text, heading hierarchy,
tables, figures, OCR text), Req 5.5 (page number + reading-order position
preserved per element), Req 5.7 (empty/no-content -> empty doc + indication),
and Req 5.8 (malformed input rejected, returning a malformed indication).
"""

from __future__ import annotations

from biomed_rag.models.enums import BlockSource, BlockType, ElementKind
from biomed_rag.models.normalized import (
    FigurePayload,
    NormalizedDocument,
    TablePayload,
    TextPayload,
)
from biomed_rag.models.parsed import (
    Cell,
    Figure,
    Heading,
    ParsedDocument,
    Table,
    TextBlock,
)
from biomed_rag.normalization import Normalizer, WordSetDictionary
from biomed_rag.normalization.result import Empty, Malformed, Normalized


def _block(text, page, pos, btype=BlockType.PARAGRAPH, level=None, source=BlockSource.TEXT_LAYER):
    return TextBlock(
        type=btype,
        text=text,
        pageNumber=page,
        readingOrderPosition=pos,
        source=source,
        headingLevel=level,
    )


# --------------------------------------------------------------------------
# Req 5.4 / 5.5 - normal document: content-preserving canonical elements
# --------------------------------------------------------------------------


def test_normalize_normal_document_preserves_all_content_and_metadata():
    parsed = ParsedDocument(
        documentId="doc-normal",
        blocks=[
            _block("Chapter 1", 1, 0, btype=BlockType.HEADING, level=1),
            _block("Section 1.1", 1, 1, btype=BlockType.HEADING, level=2),
            _block("Body paragraph under section.", 1, 2),
            _block("Scanned caption text.", 2, 4, btype=BlockType.OCR_TEXT,
                   source=BlockSource.OCR),
        ],
        tables=[
            Table(
                pageNumber=1,
                readingOrderPosition=3,
                cells=[Cell(rowIndex=0, colIndex=0, value="A")],
                degraded=False,
                rawText=None,
            )
        ],
        figures=[
            Figure(pageNumber=2, readingOrderPosition=5, imageRef="img-1",
                   caption="Figure 1")
        ],
        # Mirror index of the heading blocks (same reading-order slots).
        headings=[
            Heading(level=1, text="Chapter 1", pageNumber=1, readingOrderPosition=0),
            Heading(level=2, text="Section 1.1", pageNumber=1, readingOrderPosition=1),
        ],
    )

    result = Normalizer(WordSetDictionary()).normalize(parsed)

    assert isinstance(result, Normalized)
    doc = result.document
    assert isinstance(doc, NormalizedDocument)
    assert doc.documentId == "doc-normal"

    kinds = [(e.kind, e.pageNumber, e.readingOrderPosition) for e in doc.elements]
    # Elements are ordered by (page, reading-order) and not duplicated despite
    # the mirror headings list.
    assert kinds == [
        (ElementKind.HEADING, 1, 0),
        (ElementKind.HEADING, 1, 1),
        (ElementKind.TEXT, 1, 2),
        (ElementKind.TABLE, 1, 3),
        (ElementKind.TEXT, 2, 4),  # OCR-derived text becomes a TEXT element
        (ElementKind.FIGURE, 2, 5),
    ]

    by_pos = {e.readingOrderPosition: e for e in doc.elements}

    # Heading hierarchy is reconstructed as an ancestry path.
    assert by_pos[0].headingPath == []
    assert by_pos[1].headingPath == ["Chapter 1"]
    assert by_pos[2].headingPath == ["Chapter 1", "Section 1.1"]
    assert isinstance(by_pos[2].payload, TextPayload)

    # OCR-derived text is preserved as a TEXT element.
    assert by_pos[4].payload.text == "Scanned caption text."

    # Table structure preserved.
    table_el = by_pos[3]
    assert isinstance(table_el.payload, TablePayload)
    assert table_el.payload.cells[0].value == "A"

    # Figure structure preserved.
    fig_el = by_pos[5]
    assert isinstance(fig_el.payload, FigurePayload)
    assert fig_el.payload.imageRef == "img-1"
    assert fig_el.payload.caption == "Figure 1"


def test_normalize_applies_artifact_cleaning_before_building_elements():
    # A running header recurring on two pages must not survive into the
    # normalized representation (Req 5.1 feeding 5.4).
    parsed = ParsedDocument(
        documentId="doc-clean",
        blocks=[
            _block("Running Header", 1, 0),
            _block("Real body one.", 1, 1),
            _block("Running Header", 2, 2),
            _block("Real body two.", 2, 3),
        ],
    )

    result = Normalizer(WordSetDictionary()).normalize(parsed)

    assert isinstance(result, Normalized)
    texts = [e.payload.text for e in result.document.elements]
    assert texts == ["Real body one.", "Real body two."]


# --------------------------------------------------------------------------
# Req 5.7 - empty / no-content input
# --------------------------------------------------------------------------


def test_normalize_empty_document_returns_empty_with_indication():
    parsed = ParsedDocument(documentId="doc-empty")

    result = Normalizer(WordSetDictionary()).normalize(parsed)

    assert isinstance(result, Empty)
    assert result.reason  # a non-empty "no content" indication
    assert result.document.documentId == "doc-empty"
    assert result.document.elements == []


def test_normalize_all_artifact_document_returns_empty():
    # Every block is a recurring header artifact; after cleaning nothing remains.
    parsed = ParsedDocument(
        documentId="doc-only-artifacts",
        blocks=[
            _block("Header", 1, 0),
            _block("Header", 2, 1),
        ],
    )

    result = Normalizer(WordSetDictionary()).normalize(parsed)

    assert isinstance(result, Empty)
    assert result.document.elements == []


# --------------------------------------------------------------------------
# Req 5.8 - malformed input
# --------------------------------------------------------------------------


def test_normalize_non_parsed_document_is_malformed():
    result = Normalizer(WordSetDictionary()).normalize("not a parsed document")

    assert isinstance(result, Malformed)
    assert result.error


def test_normalize_malformed_leaves_prior_valid_output_unchanged():
    normalizer = Normalizer(WordSetDictionary())

    good = ParsedDocument(
        documentId="doc-good",
        blocks=[_block("Body.", 1, 0)],
    )
    prior = normalizer.normalize(good)
    assert isinstance(prior, Normalized)
    prior_elements = list(prior.document.elements)

    # A malformed call must not disturb the previously produced valid output.
    bad = normalizer.normalize(None)
    assert isinstance(bad, Malformed)
    assert list(prior.document.elements) == prior_elements
