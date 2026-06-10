"""Property test for table cell coordinate assignment in the Parser (Req 3.1, 3.2).

Feature: biomedical-rag-pipeline, Property 6: Table cell coordinates are a collision-free assignment

Statement: for any extracted (non-degraded) table, every non-empty source cell
is assigned to exactly one (rowIndex, colIndex) pair with no two non-empty cells
sharing the same coordinate, and any spanning cell's value is placed at its
top-left index with its spanned row count and column count recorded.

The Parser is driven through the ParsingEngine port using the deterministic
MockParsingEngine with a preset RawParseResult. Each generated table mixes
non-empty source cells (at distinct coordinates, with varied spans and optional
surrounding whitespace) with empty / whitespace-only cells that must be dropped.
The parsed table's cells must reproduce exactly the non-empty source cells: same
coordinate set with no collisions, and the recorded value / rowSpan / colSpan
preserved verbatim.
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
    RawParseResult,
    RawTable,
    RawTableCell,
    SourceDocument,
)

# A coordinate -> (value, row_span, col_span) record for one non-empty cell.
_Expected = Dict[Tuple[int, int], Tuple[str, int, int]]

# Visible ASCII (no whitespace) guarantees a value whose ``.strip()`` is
# non-empty, so the cell is genuinely a "non-empty source cell".
_VISIBLE = st.text(
    alphabet=st.characters(min_codepoint=33, max_codepoint=126), min_size=1
)
# Whitespace-only values the Parser must drop (Req 3.1: only non-empty cells get
# a coordinate).
_WHITESPACE = st.sampled_from(["", " ", "   ", "\t", "\n", " \t \n", "\r\n"])


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
def _value_with_whitespace(draw) -> str:
    """A non-empty cell value: a visible token, optionally padded with
    whitespace (the Parser must store the value verbatim, not stripped)."""
    token = draw(_VISIBLE)
    lead = draw(st.sampled_from(["", " ", "  ", "\t"]))
    trail = draw(st.sampled_from(["", " ", "  ", "\n"]))
    return f"{lead}{token}{trail}"


@st.composite
def _table_spec(draw, page: int) -> Tuple[int, List[RawTableCell], _Expected]:
    """Generate (page_number, raw_cells, expected) for a single non-degraded table.

    ``raw_cells`` interleaves non-empty cells (at distinct coordinates, with
    varied spans) and empty / whitespace-only cells (at their own distinct
    coordinates) in arbitrary order. ``expected`` maps each non-empty source
    coordinate to its (value, row_span, col_span) — the exact assignment the
    parsed table must reproduce.
    """
    n_nonempty = draw(st.integers(min_value=1, max_value=8))
    n_empty = draw(st.integers(min_value=0, max_value=4))
    total = n_nonempty + n_empty

    coords = draw(
        st.lists(
            st.tuples(
                st.integers(min_value=0, max_value=9),
                st.integers(min_value=0, max_value=9),
            ),
            min_size=total,
            max_size=total,
            unique=True,
        )
    )

    expected: _Expected = {}
    raw_cells: List[RawTableCell] = []

    for (row, col) in coords[:n_nonempty]:
        value = draw(_value_with_whitespace())
        row_span = draw(st.integers(min_value=1, max_value=4))
        col_span = draw(st.integers(min_value=1, max_value=4))
        expected[(row, col)] = (value, row_span, col_span)
        raw_cells.append(
            RawTableCell(
                row_index=row,
                col_index=col,
                value=value,
                row_span=row_span,
                col_span=col_span,
            )
        )

    for (row, col) in coords[n_nonempty:]:
        raw_cells.append(
            RawTableCell(
                row_index=row, col_index=col, value=draw(_WHITESPACE)
            )
        )

    raw_cells = draw(st.permutations(raw_cells))
    return page, list(raw_cells), expected


@st.composite
def _tables(draw) -> List[Tuple[int, List[RawTableCell], _Expected]]:
    """A small list of independent, non-degraded table specs on distinct pages.

    Each table is placed on a unique page so the Parser's page-ascending reading
    order matches the order in which we assert the specs (the property itself is
    per-table; distinct pages just keep the source/parsed alignment unambiguous).
    """
    n_tables = draw(st.integers(min_value=1, max_value=3))
    pages = draw(
        st.lists(
            st.integers(min_value=0, max_value=20),
            min_size=n_tables,
            max_size=n_tables,
            unique=True,
        )
    )
    pages.sort()
    return [draw(_table_spec(page=page)) for page in pages]


# Feature: biomedical-rag-pipeline, Property 6: Table cell coordinates are a collision-free assignment
@settings(max_examples=200)
@given(specs=_tables())
def test_table_cell_coordinates_are_a_collision_free_assignment(
    specs: List[Tuple[int, List[RawTableCell], _Expected]],
) -> None:
    """Validates: Requirements 3.1, 3.2"""
    raw_tables = [
        RawTable(page_number=page, cells=cells) for page, cells, _ in specs
    ]
    preset = RawParseResult(engine_id="docling", tables=raw_tables)
    parser = _parser(MockParsingEngine(engine_id="docling", preset_result=preset))

    parsed = parser.parse(_job(), SourceDocument(document_id="hash-1", raw_bytes=b"body"))

    # Tables carry no layout geometry, so the Parser preserves their source order.
    assert len(parsed.tables) == len(specs)

    for (page, _cells, expected), table in zip(specs, parsed.tables):
        assert table.degraded is False
        assert table.pageNumber == page

        coords = [(c.rowIndex, c.colIndex) for c in table.cells]

        # No two extracted cells share a coordinate: the assignment is collision-free.
        assert len(coords) == len(set(coords))

        # Every non-empty source cell is assigned to exactly one coordinate, and
        # empty / whitespace-only cells are dropped (the coordinate sets match).
        assert set(coords) == set(expected)
        assert len(table.cells) == len(expected)

        # Each spanning cell's value sits at its top-left index with its spanned
        # row/column counts recorded verbatim.
        for cell in table.cells:
            value, row_span, col_span = expected[(cell.rowIndex, cell.colIndex)]
            assert cell.value == value
            assert cell.rowSpan == row_span
            assert cell.colSpan == col_span
