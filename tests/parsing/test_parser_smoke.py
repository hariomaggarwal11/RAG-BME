"""Smoke tests for the Parser block/heading extraction and failure handling (Task 5.3).

These exercise the Parser through the ParsingEngine port using the deterministic
MockParsingEngine: reading-order assignment (Req 2.1), multi-column ordering
(Req 2.3), heading-hierarchy preservation (Req 2.4), and the four fail-closed
outcomes each marking the Processing_Job failed (Req 2.5, 2.6, 2.7, 2.8).

Table/figure extraction (task 5.4) and OCR wiring (task 6.2) are out of scope.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from biomed_rag.config import ParsingEngine as ParsingEngineChoice
from biomed_rag.config import PipelineConfig
from biomed_rag.models import (
    BlockSource,
    BlockType,
    DocumentMetadata,
    Format,
    OverallStatus,
    ProcessingJob,
    Stage,
    StageStatus,
)
from biomed_rag.parsing import (
    MockParsingEngine,
    ParseError,
    ParseFailure,
    ParseFailureKind,
    Parser,
    ParseTimeoutError,
    ParsingEngineRegistry,
)
from biomed_rag.parsing.raw_result import (
    BBox,
    RawBlock,
    RawFigure,
    RawParseResult,
    SourceDocument,
)


# -- fixtures / helpers ---------------------------------------------------


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


def _registry_with(engine: MockParsingEngine) -> ParsingEngineRegistry:
    registry = ParsingEngineRegistry()
    registry.register(ParsingEngineChoice.DOCLING, lambda: engine)
    return registry


def _parser(engine: MockParsingEngine, **config_kwargs) -> Parser:
    config = PipelineConfig(parsing_engine=ParsingEngineChoice.DOCLING, **config_kwargs)
    return Parser(config=config, registry=_registry_with(engine))


# -- success: reading order + structural metadata (Req 2.1) ---------------


def test_parse_assigns_contiguous_zero_based_reading_order() -> None:
    preset = RawParseResult(
        engine_id="docling",
        blocks=[
            RawBlock(text="first", page_number=0),
            RawBlock(text="second", page_number=0),
            RawBlock(text="third", page_number=1),
        ],
    )
    parser = _parser(MockParsingEngine(engine_id="docling", preset_result=preset))
    job = _job()

    parsed = parser.parse(job, _source())

    positions = [b.readingOrderPosition for b in parsed.blocks]
    assert positions == [0, 1, 2]
    assert [b.text for b in parsed.blocks] == ["first", "second", "third"]
    assert all(b.source is BlockSource.TEXT_LAYER for b in parsed.blocks)
    assert parsed.documentId == "hash-1"


def test_parse_orders_multi_column_top_to_bottom_then_left_to_right() -> None:
    # Two columns on one page. Column hints are explicit; y0 orders within a
    # column. Emitted out of order to prove the Parser sorts, not the engine.
    preset = RawParseResult(
        engine_id="docling",
        blocks=[
            RawBlock(text="R-top", page_number=0, column=1, bbox=BBox(300, 0, 400, 10)),
            RawBlock(text="L-bottom", page_number=0, column=0, bbox=BBox(0, 100, 100, 110)),
            RawBlock(text="L-top", page_number=0, column=0, bbox=BBox(0, 0, 100, 10)),
            RawBlock(text="R-bottom", page_number=0, column=1, bbox=BBox(300, 100, 400, 110)),
        ],
    )
    parser = _parser(MockParsingEngine(engine_id="docling", preset_result=preset))

    parsed = parser.parse(_job(), _source())

    assert [b.text for b in parsed.blocks] == ["L-top", "L-bottom", "R-top", "R-bottom"]
    assert [b.readingOrderPosition for b in parsed.blocks] == [0, 1, 2, 3]


def test_parse_infers_columns_from_bounding_boxes() -> None:
    # No explicit column hints; left/right columns inferred from bbox x0.
    preset = RawParseResult(
        engine_id="docling",
        blocks=[
            RawBlock(text="R-top", page_number=0, bbox=BBox(300, 0, 400, 10)),
            RawBlock(text="L-top", page_number=0, bbox=BBox(0, 0, 100, 10)),
            RawBlock(text="L-bottom", page_number=0, bbox=BBox(0, 50, 100, 60)),
            RawBlock(text="R-bottom", page_number=0, bbox=BBox(300, 50, 400, 60)),
        ],
    )
    parser = _parser(MockParsingEngine(engine_id="docling", preset_result=preset))

    parsed = parser.parse(_job(), _source())

    assert [b.text for b in parsed.blocks] == ["L-top", "L-bottom", "R-top", "R-bottom"]


# -- success: heading hierarchy (Req 2.4) ---------------------------------


def test_parse_preserves_heading_hierarchy_with_levels() -> None:
    preset = RawParseResult(
        engine_id="docling",
        blocks=[
            RawBlock(text="Chapter 1", page_number=0, kind="heading", heading_level=1),
            RawBlock(text="intro paragraph", page_number=0, kind="paragraph"),
            RawBlock(text="Section 1.1", page_number=0, kind="heading", heading_level=2),
            RawBlock(text="more text", page_number=0, kind="paragraph"),
        ],
    )
    parser = _parser(MockParsingEngine(engine_id="docling", preset_result=preset))

    parsed = parser.parse(_job(), _source())

    # Heading blocks keep their type and nesting level.
    heading_blocks = [b for b in parsed.blocks if b.type is BlockType.HEADING]
    assert [(b.text, b.headingLevel) for b in heading_blocks] == [
        ("Chapter 1", 1),
        ("Section 1.1", 2),
    ]
    # The headings list mirrors the hierarchy with matching reading-order.
    assert [(h.text, h.level) for h in parsed.headings] == [
        ("Chapter 1", 1),
        ("Section 1.1", 2),
    ]
    assert [h.readingOrderPosition for h in parsed.headings] == [0, 2]


def test_heading_without_explicit_level_defaults_to_one() -> None:
    preset = RawParseResult(
        engine_id="docling",
        blocks=[RawBlock(text="Title", page_number=0, kind="heading")],
    )
    parser = _parser(MockParsingEngine(engine_id="docling", preset_result=preset))

    parsed = parser.parse(_job(), _source())

    assert parsed.headings[0].level == 1
    assert parsed.blocks[0].headingLevel == 1


# -- failure handling: each marks the job failed (Req 2.5-2.8) ------------


def test_engine_unavailable_marks_job_failed() -> None:
    parser = _parser(MockParsingEngine(engine_id="docling", available=False))
    job = _job()

    with pytest.raises(ParseFailure) as exc:
        parser.parse(job, _source())

    assert exc.value.kind is ParseFailureKind.ENGINE_UNAVAILABLE
    assert exc.value.engine_id == "docling"
    assert job.overallStatus is OverallStatus.FAILED
    assert job.failingStage is Stage.PARSING
    assert job.stageStates[Stage.PARSING].status is StageStatus.FAILED
    assert "docling" in job.stageStates[Stage.PARSING].failureReason


def test_parse_error_marks_job_failed_and_retains_no_output() -> None:
    engine = MockParsingEngine(
        engine_id="docling", raise_on_parse=ParseError("boom")
    )
    parser = _parser(engine)
    job = _job()

    with pytest.raises(ParseFailure) as exc:
        parser.parse(job, _source())

    assert exc.value.kind is ParseFailureKind.PARSE_ERROR
    assert job.overallStatus is OverallStatus.FAILED
    assert job.stageStates[Stage.PARSING].failureReason.startswith("parse error")


def test_parse_timeout_marks_job_failed() -> None:
    engine = MockParsingEngine(
        engine_id="docling", raise_on_parse=ParseTimeoutError("too slow")
    )
    parser = _parser(engine)
    job = _job()

    with pytest.raises(ParseFailure) as exc:
        parser.parse(job, _source())

    assert exc.value.kind is ParseFailureKind.TIMEOUT
    assert job.overallStatus is OverallStatus.FAILED
    assert "limit" in job.stageStates[Stage.PARSING].failureReason


def test_slow_engine_exceeding_deadline_times_out() -> None:
    # A clock that jumps past the deadline simulates an engine that returns but
    # took longer than parse_timeout_seconds (Req 2.7).
    preset = RawParseResult(
        engine_id="docling", blocks=[RawBlock(text="content", page_number=0)]
    )
    config = PipelineConfig(
        parsing_engine=ParsingEngineChoice.DOCLING, parse_timeout_seconds=1
    )
    engine = MockParsingEngine(engine_id="docling", preset_result=preset)
    registry = _registry_with(engine)

    ticks = iter([100.0, 1000.0])
    parser = Parser(config=config, registry=registry, clock=lambda: next(ticks))

    job = _job()
    with pytest.raises(ParseFailure) as exc:
        parser.parse(job, _source())

    assert exc.value.kind is ParseFailureKind.TIMEOUT
    assert job.overallStatus is OverallStatus.FAILED


def test_no_extractable_content_marks_job_failed() -> None:
    # Engine returns only whitespace blocks -> no extractable content (Req 2.8).
    preset = RawParseResult(
        engine_id="docling",
        blocks=[RawBlock(text="   ", page_number=0)],
    )
    parser = _parser(MockParsingEngine(engine_id="docling", preset_result=preset))
    job = _job()

    with pytest.raises(ParseFailure) as exc:
        parser.parse(job, _source())

    assert exc.value.kind is ParseFailureKind.NO_CONTENT
    assert job.overallStatus is OverallStatus.FAILED
    assert job.failingStage is Stage.PARSING


def test_figure_only_document_is_extractable_content() -> None:
    # A document with no text blocks but a figure still has extractable content.
    preset = RawParseResult(
        engine_id="docling",
        blocks=[],
        figures=[RawFigure(page_number=0, image_ref="fig-1", caption="Fig 1")],
    )
    parser = _parser(MockParsingEngine(engine_id="docling", preset_result=preset))
    job = _job()

    parsed = parser.parse(job, _source())  # should not raise

    assert parsed.blocks == []
    assert job.overallStatus is not OverallStatus.FAILED


def test_unregistered_engine_fails_closed() -> None:
    # Registry has no engine registered for the configured choice.
    config = PipelineConfig(parsing_engine=ParsingEngineChoice.DOCLING)
    parser = Parser(config=config, registry=ParsingEngineRegistry())
    job = _job()

    with pytest.raises(ParseFailure) as exc:
        parser.parse(job, _source())

    assert exc.value.kind is ParseFailureKind.ENGINE_UNAVAILABLE
    assert job.overallStatus is OverallStatus.FAILED


def test_failure_without_job_still_raises() -> None:
    parser = _parser(MockParsingEngine(engine_id="docling", available=False))
    with pytest.raises(ParseFailure):
        parser.parse(None, _source())
