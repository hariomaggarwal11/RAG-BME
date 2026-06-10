"""Property test for fitting tables staying within a single chunk (Req 6.4).

Feature: biomedical-rag-pipeline, Property 16: Fitting tables stay within a single chunk

Statement: for any normalized representation containing a table whose serialized
content fits within ``max_chunk_tokens``, the table content is contained
entirely within a single Chunk.

The Chunker is driven through the deterministic ``WhitespaceTokenizer`` so token
counts are exact and reproducible. Each generated table is serialized with the
Chunker's own table serialization and the configured ``max_chunk_tokens`` is
sized to be >= that serialized token count, guaranteeing the table fits. The
test then asserts the table maps to exactly one ``isTablePart`` chunk whose
content carries every cell value, and that the chunk records no overlap.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from hypothesis import given, settings
from hypothesis import strategies as st

from biomed_rag.chunking import Chunker, WhitespaceTokenizer
from biomed_rag.chunking.chunker import Chunker as _ChunkerImpl
from biomed_rag.models import (
    Cell,
    ContentElement,
    NormalizedDocument,
    TablePayload,
    TextPayload,
)
from biomed_rag.models.enums import ElementKind


@dataclass
class _ChunkConfig:
    """Minimal stand-in for PipelineConfig carrying only the chunking bounds."""

    max_chunk_tokens: int
    chunk_overlap_tokens: int


# Visible ASCII (no whitespace) so each cell value is exactly one whitespace
# token and is unambiguously present in / absent from a chunk's content.
_VISIBLE = st.text(
    alphabet=st.characters(min_codepoint=33, max_codepoint=126), min_size=1, max_size=6
)


@st.composite
def _grid_cells(draw) -> Tuple[List[Cell], List[str]]:
    """Generate a dense ``rows x cols`` grid of non-empty cells.

    Returns the cells together with the flat list of their values (every value
    is a single visible token), so the test can assert each appears in the
    table's chunk.
    """
    rows = draw(st.integers(min_value=1, max_value=5))
    cols = draw(st.integers(min_value=1, max_value=5))
    cells: List[Cell] = []
    values: List[str] = []
    for r in range(rows):
        for c in range(cols):
            value = draw(_VISIBLE)
            values.append(value)
            cells.append(Cell(rowIndex=r, colIndex=c, value=value))
    return cells, values


@st.composite
def _fitting_table_case(draw):
    """Generate a (document, values, config) triple for a fitting table.

    The configured ``max_chunk_tokens`` is sized to be >= the serialized token
    count of the table so the table is guaranteed to fit in a single chunk
    (Req 6.4); ``chunk_overlap_tokens`` is drawn in the valid ``[0, max-1]``
    range. The document optionally surrounds the table with running text so the
    property holds even when the table coexists with other content.
    """
    cells, values = draw(_grid_cells())
    page = draw(st.integers(min_value=0, max_value=20))
    heading_path = draw(
        st.lists(_VISIBLE, min_size=0, max_size=2)
    )

    payload = TablePayload(cells=cells)
    serialized = _ChunkerImpl._serialize_table(payload)
    table_token_count = len(WhitespaceTokenizer().tokenize(serialized))

    # Size the bound so the serialized table fits within one chunk, with some
    # extra head-room drawn on top (Req 6.4: "fits within max_chunk_tokens").
    extra = draw(st.integers(min_value=0, max_value=20))
    max_tokens = table_token_count + extra
    overlap = draw(st.integers(min_value=0, max_value=max_tokens - 1))

    # Optionally wrap the table with running text on either side. Text segments
    # are windowed independently and do not affect whether the table fits.
    lead_text = draw(st.text(alphabet="abcdefg ", min_size=0, max_size=12))
    trail_text = draw(st.text(alphabet="hijklmn ", min_size=0, max_size=12))

    elements: List[ContentElement] = []
    position = 0
    if lead_text.strip():
        elements.append(
            ContentElement(
                kind=ElementKind.TEXT,
                pageNumber=page,
                readingOrderPosition=position,
                payload=TextPayload(text=lead_text),
            )
        )
        position += 1
    elements.append(
        ContentElement(
            kind=ElementKind.TABLE,
            pageNumber=page,
            readingOrderPosition=position,
            payload=payload,
            headingPath=list(heading_path),
        )
    )
    position += 1
    if trail_text.strip():
        elements.append(
            ContentElement(
                kind=ElementKind.TEXT,
                pageNumber=page,
                readingOrderPosition=position,
                payload=TextPayload(text=trail_text),
            )
        )

    doc = NormalizedDocument(documentId="doc-1", elements=elements)
    return doc, values, page, list(heading_path), _ChunkConfig(max_tokens, overlap)


# Feature: biomedical-rag-pipeline, Property 16: Fitting tables stay within a single chunk
@settings(max_examples=200)
@given(case=_fitting_table_case())
def test_fitting_table_stays_within_single_chunk(case) -> None:
    """Validates: Requirements 6.4"""
    doc, values, page, heading_path, config = case

    chunks = Chunker(WhitespaceTokenizer()).chunk(doc, config)

    # The table contributes exactly one chunk, marked as a table part: a fitting
    # table is never split (Req 6.4).
    table_chunks = [c for c in chunks if c.isTablePart]
    assert len(table_chunks) == 1

    table_chunk = table_chunks[0]
    # That single chunk stays within the configured bound and records no overlap.
    assert table_chunk.tokenCount <= config.max_chunk_tokens
    assert table_chunk.overlapTokenCount == 0
    assert table_chunk.pageNumber == page
    assert table_chunk.headingPath == heading_path

    # Every cell value is contained entirely within the one table chunk.
    chunk_tokens = WhitespaceTokenizer().tokenize(table_chunk.content)
    for value in values:
        assert value in chunk_tokens
