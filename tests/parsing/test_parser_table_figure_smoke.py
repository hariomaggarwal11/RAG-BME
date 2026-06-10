"""Smoke tests for Parser table and figure extraction (Task 5.4).

These drive the Parser through the deterministic MockParsingEngine with preset
RawParseResults containing tables and figures, covering:

* Non-empty cell -> exactly one (row, col); spanning cells at top-left with
  recorded rowSpan/colSpan (Req 3.1, 3.2).
* Figures with and without captions; absent caption recorded without failing
  (Req 3.3, 3.4).
* Page number + zero-based reading-order position shared with text blocks
  (Req 3.5).
* Degraded tables flagged with raw region text retained (Req 3.6).
"""

from __future__ import annotations

from datetime import datetime, timezone

from biomed_rag.config import ParsingEngine as ParsingEngineChoice
from biomed_rag.config import PipelineConfig
from biomed_rag.models import DocumentMetadata, Format, ProcessingJob
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


def _job(document_id: str = "hash-1") -> ProcessingJob:
    metadata = DocumentMetadata(
        filename="paper.pdf",
        format=Format.PDF,
        byteSize=1234,
        contentHash=document_id,
        submittedAtUtc=datetime.now(timezone.utc),
    )
    return ProcessingJob(jobId="job-1", documentId=document_id, metadata=metadata)


def _source(document_id: str = "hash-1", data: bytes = b"body") -> SourceDocument:
    return SourceDocument(document_id=document_id, raw_bytes=data)


def _parser(engine: MockParsingEngine, **config_kwargs) -> Parser:
    registry = ParsingEngineRegistry()
    registry.register(ParsingEngineChoice.DOCLING, lambda: engine)
    config = PipelineConfig(parsing_engine=ParsingEngineChoice.DOCLING, **config_kwargs)
    return Parser(config=config, registry=registry)


def test_extracts_table_cells_with_spans_and_drops_empty_cells() -> None:
    preset = RawParseResult(
        engine_id="docling",
        tables=[
            RawTable(
                page_number=2,
                cells=[
                    RawTableCell(row_index=0, col_index=0, value="Gene", row_span=2, col_span=1),
                    RawTableCell(row_index=0, col_index=1, value="p"),
                    RawTableCell(row_index=1, col_index=1, value="   "),  # empty -> dropped
                    RawTableCell(row_index=2, col_index=1, value="0.01"),
                ],
            )
        ],
    )
    parser = _parser(MockParsingEngine(engine_id="docling", preset_result=preset))

    parsed = parser.parse(_job(), _source())

    assert len(parsed.tables) == 1
    table = parsed.tables[0]
    assert table.pageNumber == 2
    assert table.degraded is False

    coords = {(c.rowIndex, c.colIndex): c for c in table.cells}
    assert set(coords) == {(0, 0), (0, 1), (2, 1)}  # whitespace cell dropped
    # Every non-empty cell maps to exactly one coordinate.
    assert len(table.cells) == len(coords)
    span_cell = coords[(0, 0)]
    assert span_cell.value == "Gene"
    assert span_cell.rowSpan == 2 and span_cell.colSpan == 1


def test_extracts_figures_with_and_without_caption() -> None:
    preset = RawParseResult(
        engine_id="docling",
        figures=[
            RawFigure(page_number=1, image_ref="fig-1", caption="Figure 1: assay"),
            RawFigure(page_number=1, image_ref="fig-2"),  # caption absent
        ],
    )
    parser = _parser(MockParsingEngine(engine_id="docling", preset_result=preset))

    parsed = parser.parse(_job(), _source())

    assert [(f.imageRef, f.caption) for f in parsed.figures] == [
        ("fig-1", "Figure 1: assay"),
        ("fig-2", None),
    ]
    assert all(f.pageNumber == 1 for f in parsed.figures)


def test_degraded_table_retains_raw_text() -> None:
    preset = RawParseResult(
        engine_id="docling",
        tables=[
            RawTable(
                page_number=0,
                degraded=True,
                raw_text="unstructured table region text",
            )
        ],
    )
    parser = _parser(MockParsingEngine(engine_id="docling", preset_result=preset))

    parsed = parser.parse(_job(), _source())

    table = parsed.tables[0]
    assert table.degraded is True
    assert table.rawText == "unstructured table region text"
    assert table.cells == []


def test_tables_and_figures_share_one_contiguous_reading_order_with_blocks() -> None:
    # One page, geometry interleaves a block, a table, and a figure by y0.
    preset = RawParseResult(
        engine_id="docling",
        blocks=[
            RawBlock(text="intro", page_number=0, bbox=BBox(0, 0, 100, 10)),
            RawBlock(text="after", page_number=0, bbox=BBox(0, 200, 100, 210)),
        ],
        tables=[
            RawTable(
                page_number=0,
                bbox=BBox(0, 50, 100, 90),
                cells=[RawTableCell(row_index=0, col_index=0, value="x")],
            )
        ],
        figures=[RawFigure(page_number=0, image_ref="fig-1", bbox=BBox(0, 120, 100, 160))],
    )
    parser = _parser(MockParsingEngine(engine_id="docling", preset_result=preset))

    parsed = parser.parse(_job(), _source())

    # Collect every content element's reading-order position.
    positions = (
        [b.readingOrderPosition for b in parsed.blocks]
        + [t.readingOrderPosition for t in parsed.tables]
        + [f.readingOrderPosition for f in parsed.figures]
    )
    # Contiguous zero-based sequence with no collisions across all elements.
    assert sorted(positions) == [0, 1, 2, 3]
    assert len(set(positions)) == 4

    # Geometry order: intro(y0=0) -> table(y0=50) -> figure(y0=120) -> after(y0=200).
    assert parsed.blocks[0].readingOrderPosition == 0
    assert parsed.tables[0].readingOrderPosition == 1
    assert parsed.figures[0].readingOrderPosition == 2
    assert parsed.blocks[1].readingOrderPosition == 3
