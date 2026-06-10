"""Property test for reading-order positions in the Parser (Req 2.1, 2.3, 3.5).

Feature: biomedical-rag-pipeline, Property 4: Reading-order positions are a contiguous zero-based sequence

Statement: for any ParsedDocument, the reading-order positions assigned to its
content elements form a contiguous sequence of integers starting at zero, and
multi-column layouts are ordered top-to-bottom within a column and
left-to-right across columns.

The Parser is driven through the ParsingEngine port using the deterministic
MockParsingEngine with a preset RawParseResult. Two properties are checked:

1. Across every content element the Parser emits — text blocks, tables, and
   figures — the ``readingOrderPosition`` values together form exactly the
   contiguous, collision-free sequence ``0 .. N-1`` (Req 2.1, 3.5).
2. For an explicitly multi-column single page, the emitted blocks are ordered
   left-to-right across columns and top-to-bottom within each column (Req 2.3).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Tuple

from hypothesis import given, settings
from hypothesis import strategies as st

from biomed_rag.config import ParsingEngine as ParsingEngineChoice
from biomed_rag.config import PipelineConfig
from biomed_rag.models import (
    DocumentMetadata,
    Format,
    ProcessingJob,
)
from biomed_rag.parsing import (
    MockParsingEngine,
    Parser,
    ParsingEngineRegistry,
)
from biomed_rag.parsing.raw_result import (
    BBox,
    RawBlock,
    RawFigure,
    RawParseResult,
    RawTable,
    RawTableCell,
    SourceDocument,
)

# Visible ASCII (no whitespace) so every generated block carries genuinely
# non-empty text and the document always has extractable content (avoids the
# Parser's no-content fail-closed path, which is out of scope here).
_VISIBLE = st.text(
    alphabet=st.characters(min_codepoint=33, max_codepoint=126), min_size=1, max_size=8
)


def _job(document_id: str = "hash-1") -> ProcessingJob:
    metadata = DocumentMetadata(
        filename="paper.pdf",
        format=Format.PDF,
        byteSize=1234,
        contentHash=document_id,
        submittedAtUtc=datetime.now(timezone.utc),
    )
    return ProcessingJob(jobId="job-1", documentId=document_id, metadata=metadata)


def _parser(engine: MockParsingEngine) -> Parser:
    registry = ParsingEngineRegistry()
    registry.register(ParsingEngineChoice.DOCLING, lambda: engine)
    config = PipelineConfig(parsing_engine=ParsingEngineChoice.DOCLING)
    return Parser(config=config, registry=registry)


@st.composite
def _bbox(draw) -> BBox:
    x0 = draw(st.floats(min_value=0.0, max_value=500.0, allow_nan=False))
    y0 = draw(st.floats(min_value=0.0, max_value=500.0, allow_nan=False))
    w = draw(st.floats(min_value=1.0, max_value=100.0, allow_nan=False))
    h = draw(st.floats(min_value=1.0, max_value=100.0, allow_nan=False))
    return BBox(x0=x0, y0=y0, x1=x0 + w, y1=y0 + h)


@st.composite
def _raw_block(draw) -> RawBlock:
    return RawBlock(
        text=draw(_VISIBLE),
        page_number=draw(st.integers(min_value=0, max_value=3)),
        kind=draw(st.sampled_from(["paragraph", "heading", "caption", "list_item"])),
        bbox=draw(st.one_of(st.none(), _bbox())),
        column=draw(st.one_of(st.none(), st.integers(min_value=0, max_value=2))),
    )


@st.composite
def _raw_table(draw) -> RawTable:
    n = draw(st.integers(min_value=0, max_value=3))
    cells = [
        RawTableCell(row_index=i, col_index=0, value=draw(_VISIBLE))
        for i in range(n)
    ]
    return RawTable(
        page_number=draw(st.integers(min_value=0, max_value=3)),
        cells=cells,
        bbox=draw(st.one_of(st.none(), _bbox())),
    )


@st.composite
def _raw_figure(draw) -> RawFigure:
    return RawFigure(
        page_number=draw(st.integers(min_value=0, max_value=3)),
        image_ref=draw(_VISIBLE),
        caption=draw(st.one_of(st.none(), _VISIBLE)),
        bbox=draw(st.one_of(st.none(), _bbox())),
    )


@st.composite
def _raw_result(draw) -> RawParseResult:
    """A mixed result with at least one content element of some kind."""
    blocks = draw(st.lists(_raw_block(), min_size=0, max_size=6))
    tables = draw(st.lists(_raw_table(), min_size=0, max_size=3))
    figures = draw(st.lists(_raw_figure(), min_size=0, max_size=3))
    # Guarantee extractable content: at least one element overall, and if there
    # are no tables/figures then ensure at least one (visible-text) block.
    if not (blocks or tables or figures):
        blocks = [RawBlock(text=draw(_VISIBLE), page_number=0)]
    return RawParseResult(
        engine_id="docling", blocks=blocks, tables=tables, figures=figures
    )


# Feature: biomedical-rag-pipeline, Property 4: Reading-order positions are a contiguous zero-based sequence
@settings(max_examples=200)
@given(raw=_raw_result())
def test_reading_order_positions_form_contiguous_zero_based_sequence(
    raw: RawParseResult,
) -> None:
    """Validates: Requirements 2.1, 2.3, 3.5"""
    parser = _parser(MockParsingEngine(engine_id="docling", preset_result=raw))

    parsed = parser.parse(
        _job(), SourceDocument(document_id="hash-1", raw_bytes=b"body")
    )

    # Every block, table, and figure carries one reading-order position. They
    # share a single document sequence (Req 3.5).
    positions = (
        [b.readingOrderPosition for b in parsed.blocks]
        + [t.readingOrderPosition for t in parsed.tables]
        + [f.readingOrderPosition for f in parsed.figures]
    )

    total = len(raw.blocks) + len(raw.tables) + len(raw.figures)
    assert len(positions) == total

    # Contiguous, zero-based, and collision-free: exactly {0, 1, ..., N-1}.
    assert sorted(positions) == list(range(total))
    assert len(set(positions)) == len(positions)


# -- multi-column ordering (Req 2.3) -------------------------------------


@st.composite
def _multi_column_blocks(draw) -> Tuple[List[RawBlock], Dict[str, Tuple[int, float]]]:
    """Blocks on a single page spread across explicit columns.

    Each block gets a unique text label and a recorded ``(column, y0)`` so the
    expected reading order can be reconstructed: left-to-right across columns,
    top-to-bottom within a column. ``x0`` follows the column so layout geometry
    agrees with the explicit column hint.
    """
    n = draw(st.integers(min_value=2, max_value=8))
    layout: Dict[str, Tuple[int, float]] = {}
    blocks: List[RawBlock] = []
    for i in range(n):
        column = draw(st.integers(min_value=0, max_value=2))
        y0 = float(draw(st.integers(min_value=0, max_value=400)))
        label = f"b{i}"
        x0 = float(column) * 1000.0
        layout[label] = (column, y0)
        blocks.append(
            RawBlock(
                text=label,
                page_number=0,
                column=column,
                bbox=BBox(x0=x0, y0=y0, x1=x0 + 100.0, y1=y0 + 10.0),
            )
        )
    # Emit out of reading order so the assertion proves the Parser sorts.
    blocks = list(draw(st.permutations(blocks)))
    return blocks, layout


# Feature: biomedical-rag-pipeline, Property 4: Reading-order positions are a contiguous zero-based sequence
@settings(max_examples=200)
@given(data=_multi_column_blocks())
def test_multi_column_ordered_left_to_right_then_top_to_bottom(
    data: Tuple[List[RawBlock], Dict[str, Tuple[int, float]]],
) -> None:
    """Validates: Requirements 2.3"""
    blocks, layout = data
    raw = RawParseResult(engine_id="docling", blocks=blocks)
    parser = _parser(MockParsingEngine(engine_id="docling", preset_result=raw))

    parsed = parser.parse(
        _job(), SourceDocument(document_id="hash-1", raw_bytes=b"body")
    )

    emitted = [b.text for b in parsed.blocks]

    # The sequence is non-decreasing in column index (left-to-right across
    # columns) and, within each column, non-decreasing in y0 (top-to-bottom).
    seen: List[Tuple[int, float]] = [layout[text] for text in emitted]
    for (prev_col, prev_y), (col, y) in zip(seen, seen[1:]):
        assert (col > prev_col) or (col == prev_col and y >= prev_y)

    # Positions remain a contiguous zero-based sequence for the multi-column case.
    assert [b.readingOrderPosition for b in parsed.blocks] == list(range(len(blocks)))
