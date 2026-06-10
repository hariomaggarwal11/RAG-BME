"""Property test for content/structure preservation in normalization (Req 5.4).

Feature: biomedical-rag-pipeline, Property 11: Normalization preserves all non-artifact content and structure

Statement: for any Parsed_Document, every non-artifact content element of the
source (text, heading hierarchy, tables, figures, OCR-derived text) appears in the
normalized representation with its structure intact.

Construction notes
------------------
To make *every* source element a non-artifact (so nothing is legitimately removed
by the header/footer cleaner of Req 5.1), the generator never produces a recurring
header/footer element:

* every text/heading **block** carries a globally unique, digit-free token, so no
  exact signature and no page-number-template signature can ever recur across two
  pages; the artifact remover therefore drops nothing; and
* the de-hyphenation dictionary is empty and tokens contain no line-break hyphen,
  so block text passes through unchanged (Req 5.2, 5.3).

Each source item is placed at a distinct ``(pageNumber, readingOrderPosition)``
slot. Pages are non-decreasing and the reading-order position is a single strictly
increasing global sequence, so generation order equals the canonical
``(page, reading-order)`` ordering the Normalizer sorts by. This lets the test
assert reading-order preservation by direct positional comparison.

Headings are emitted both as ``HEADING``-typed blocks in ``parsed.blocks`` *and*
mirrored into ``parsed.headings`` at the same slot, exercising the Normalizer's
``heading_block_slots`` de-duplication: each heading must yield exactly one
``HEADING`` element (no double-count).
"""

from __future__ import annotations

from typing import List, Tuple

from hypothesis import given, settings
from hypothesis import strategies as st

from biomed_rag.models.enums import BlockSource, BlockType, ElementKind
from biomed_rag.models.normalized import FigurePayload, TablePayload, TextPayload
from biomed_rag.models.parsed import (
    Cell,
    Figure,
    Heading,
    ParsedDocument,
    Table,
    TextBlock,
)
from biomed_rag.normalization import Normalizer, WordSetDictionary
from biomed_rag.normalization.result import Normalized

# Non-heading block types; all of these must normalize to a TEXT element.
_TEXT_BLOCK_TYPES = (
    BlockType.PARAGRAPH,
    BlockType.OCR_TEXT,
    BlockType.CAPTION,
    BlockType.LIST_ITEM,
    BlockType.FOOTNOTE,
)

_LETTERS = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ",
    min_size=0,
    max_size=6,
)
_LETTERS_NONEMPTY = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ",
    min_size=1,
    max_size=6,
)


def _letters_for(n: int) -> str:
    """Map a non-negative int to a unique, digit-free lowercase-letter token."""
    n += 1
    out = ""
    while n:
        n, rem = divmod(n - 1, 26)
        out = chr(ord("a") + rem) + out
    return out


@st.composite
def _cells(draw) -> List[Cell]:
    """A small list of table cells with coordinates, spans, and values (Req 3.1, 3.2)."""
    count = draw(st.integers(min_value=0, max_value=4))
    cells: List[Cell] = []
    for _ in range(count):
        cells.append(
            Cell(
                rowIndex=draw(st.integers(min_value=0, max_value=4)),
                colIndex=draw(st.integers(min_value=0, max_value=4)),
                value=draw(_LETTERS),
                rowSpan=draw(st.integers(min_value=1, max_value=3)),
                colSpan=draw(st.integers(min_value=1, max_value=3)),
            )
        )
    return cells


@st.composite
def _parsed_documents(draw):
    """Generate an artifact-free ParsedDocument and the expected element descriptors.

    Returns ``(parsed, expected)`` where ``expected`` is the canonical, ordered list
    of ``(kind, page, position, checker)`` descriptors the normalized elements must
    match one-for-one.
    """
    num_items = draw(st.integers(min_value=1, max_value=12))

    blocks: List[TextBlock] = []
    headings: List[Heading] = []
    tables: List[Table] = []
    figures: List[Figure] = []

    expected: List[dict] = []
    token_counter = 0
    page = 1

    # Heading-stack simulation mirrors Normalizer._push_heading so we can assert the
    # reconstructed heading hierarchy (headingPath) exactly (Req 5.4).
    heading_stack: List[Tuple[int, str]] = []

    for pos in range(num_items):
        # Pages are non-decreasing; positions are globally strictly increasing, so
        # (page, pos) slots are distinct and generation order == sorted order.
        if pos > 0 and draw(st.booleans()):
            page += draw(st.integers(min_value=1, max_value=2))

        kind = draw(st.sampled_from(("text", "heading", "table", "figure")))

        if kind == "text":
            btype = draw(st.sampled_from(_TEXT_BLOCK_TYPES))
            source = (
                BlockSource.OCR
                if btype is BlockType.OCR_TEXT
                else BlockSource.TEXT_LAYER
            )
            text = "t" + _letters_for(token_counter)
            token_counter += 1
            blocks.append(
                TextBlock(
                    type=btype,
                    text=text,
                    pageNumber=page,
                    readingOrderPosition=pos,
                    source=source,
                )
            )
            heading_path = [t for _, t in heading_stack]
            expected.append(
                {
                    "kind": ElementKind.TEXT,
                    "page": page,
                    "pos": pos,
                    "text": text,
                    "headingPath": heading_path,
                }
            )

        elif kind == "heading":
            level = draw(st.integers(min_value=1, max_value=4))
            text = "h" + _letters_for(token_counter)
            token_counter += 1
            # Heading appears as a HEADING block AND as a mirror Heading record at
            # the same slot (the parser's mirror index). The Normalizer must emit
            # exactly one HEADING element for the pair.
            blocks.append(
                TextBlock(
                    type=BlockType.HEADING,
                    text=text,
                    pageNumber=page,
                    readingOrderPosition=pos,
                    headingLevel=level,
                )
            )
            headings.append(
                Heading(
                    level=level,
                    text=text,
                    pageNumber=page,
                    readingOrderPosition=pos,
                )
            )
            # Reconstruct expected ancestry, then push this heading.
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            ancestors = [t for _, t in heading_stack]
            heading_stack.append((level, text))
            expected.append(
                {
                    "kind": ElementKind.HEADING,
                    "page": page,
                    "pos": pos,
                    "text": text,
                    "level": level,
                    "headingPath": ancestors,
                }
            )

        elif kind == "table":
            cells = draw(_cells())
            degraded = draw(st.booleans())
            raw_text = draw(st.one_of(st.none(), _LETTERS))
            tables.append(
                Table(
                    pageNumber=page,
                    readingOrderPosition=pos,
                    cells=cells,
                    degraded=degraded,
                    rawText=raw_text,
                )
            )
            expected.append(
                {
                    "kind": ElementKind.TABLE,
                    "page": page,
                    "pos": pos,
                    "cells": cells,
                    "degraded": degraded,
                    "rawText": raw_text,
                }
            )

        else:  # figure
            image_ref = draw(_LETTERS_NONEMPTY)
            caption = draw(st.one_of(st.none(), _LETTERS))
            figures.append(
                Figure(
                    pageNumber=page,
                    readingOrderPosition=pos,
                    imageRef=image_ref,
                    caption=caption,
                )
            )
            expected.append(
                {
                    "kind": ElementKind.FIGURE,
                    "page": page,
                    "pos": pos,
                    "imageRef": image_ref,
                    "caption": caption,
                }
            )

    parsed = ParsedDocument(
        documentId="doc-prop11",
        blocks=blocks,
        tables=tables,
        figures=figures,
        headings=headings,
    )
    return parsed, expected


# Feature: biomedical-rag-pipeline, Property 11: Normalization preserves all non-artifact content and structure
@settings(max_examples=200, deadline=None)
@given(data=_parsed_documents())
def test_normalization_preserves_all_non_artifact_content_and_structure(data) -> None:
    """Validates: Requirements 5.4"""
    parsed, expected = data

    result = Normalizer(WordSetDictionary()).normalize(parsed)

    # With at least one non-artifact item present, normalization yields content.
    assert isinstance(result, Normalized)
    elements = result.document.elements

    # No element is dropped and none is duplicated (heading block + mirror record
    # collapse to a single HEADING element).
    assert len(elements) == len(expected)

    for element, exp in zip(elements, expected):
        # Reading order preserved: page number and reading-order position intact.
        assert element.kind is exp["kind"]
        assert element.pageNumber == exp["page"]
        assert element.readingOrderPosition == exp["pos"]

        if exp["kind"] is ElementKind.TEXT:
            assert isinstance(element.payload, TextPayload)
            # Text content (including OCR-derived text) preserved.
            assert element.payload.text == exp["text"]
            assert element.headingPath == exp["headingPath"]

        elif exp["kind"] is ElementKind.HEADING:
            assert isinstance(element.payload, TextPayload)
            # Heading text, level, and reconstructed hierarchy preserved.
            assert element.payload.text == exp["text"]
            assert element.payload.headingLevel == exp["level"]
            assert element.headingPath == exp["headingPath"]

        elif exp["kind"] is ElementKind.TABLE:
            assert isinstance(element.payload, TablePayload)
            # Table cells (coords, spans, values), degraded flag, raw text preserved.
            assert element.payload.cells == exp["cells"]
            assert element.payload.degraded == exp["degraded"]
            assert element.payload.rawText == exp["rawText"]

        else:  # FIGURE
            assert isinstance(element.payload, FigurePayload)
            # Figure image reference and optional caption preserved.
            assert element.payload.imageRef == exp["imageRef"]
            assert element.payload.caption == exp["caption"]
