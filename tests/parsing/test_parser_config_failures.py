"""Unit tests for Parser configuration, fail-closed paths, captions, and degraded tables (Task 5.8).

These exercise the Parser end-to-end through the ParsingEngine port using the
deterministic MockParsingEngine, focusing on the behaviours called out in task
5.8:

* Engine selection is driven entirely by ``PipelineConfig.parsing_engine`` —
  the Parser routes through whichever engine the registry maps the configured
  choice to, with no backend hard-coding (Req 2.2).
* Each of the four fail-closed paths marks the Processing_Job failed at the
  PARSING stage and records the failure reason: engine unavailable (Req 2.6),
  parse error with no partial output retained (Req 2.5), parse timeout
  (Req 2.7), and no extractable content (Req 2.8).
* Figures carry their caption when present and record ``None`` without failing
  when absent (Req 3.3, 3.4).
* Degraded tables are flagged and retain their raw region text (Req 3.6).
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from biomed_rag.config import ParsingEngine as ParsingEngineChoice
from biomed_rag.config import PipelineConfig
from biomed_rag.models import (
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
    RawBlock,
    RawFigure,
    RawParseResult,
    RawTable,
    RawTableCell,
    SourceDocument,
)


# -- helpers --------------------------------------------------------------


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


def _content_preset(engine_id: str = "mock") -> RawParseResult:
    """A minimal result with one real text block so the parse succeeds."""
    return RawParseResult(
        engine_id=engine_id,
        blocks=[RawBlock(text="content", page_number=0)],
    )


# -- engine selection by config (Req 2.2) --------------------------------


def test_parser_selects_docling_when_config_chooses_docling() -> None:
    # Both choices are registered; the configured one must be the one used.
    registry = ParsingEngineRegistry()
    registry.register(
        ParsingEngineChoice.DOCLING,
        lambda: MockParsingEngine(
            engine_id="docling", preset_result=_content_preset("docling")
        ),
    )
    registry.register(
        ParsingEngineChoice.LLAMAPARSE,
        lambda: MockParsingEngine(
            engine_id="llamaparse", preset_result=_content_preset("llamaparse")
        ),
    )
    config = PipelineConfig(parsing_engine=ParsingEngineChoice.DOCLING)
    parser = Parser(config=config, registry=registry)

    parser.parse(_job(), _source())  # selection happens internally; must not raise

    # The Parser asks the registry for the configured engine; confirm the
    # registry resolves that choice to the docling-id engine.
    assert registry.select(config).engine_id() == "docling"


def test_parser_selects_llamaparse_when_config_chooses_llamaparse() -> None:
    selected: list[str] = []

    def docling_factory() -> MockParsingEngine:
        selected.append("docling")
        return MockParsingEngine(
            engine_id="docling", preset_result=_content_preset("docling")
        )

    def llamaparse_factory() -> MockParsingEngine:
        selected.append("llamaparse")
        return MockParsingEngine(
            engine_id="llamaparse", preset_result=_content_preset("llamaparse")
        )

    registry = ParsingEngineRegistry()
    registry.register(ParsingEngineChoice.DOCLING, docling_factory)
    registry.register(ParsingEngineChoice.LLAMAPARSE, llamaparse_factory)
    config = PipelineConfig(parsing_engine=ParsingEngineChoice.LLAMAPARSE)
    parser = Parser(config=config, registry=registry)

    parsed = parser.parse(_job(), _source())

    # Only the llamaparse factory was invoked — the docling backend was never
    # touched, proving selection is config-driven (Req 2.2).
    assert selected == ["llamaparse"]
    assert parsed.documentId == "hash-1"


# -- fail-closed: engine unavailable (Req 2.6) ---------------------------


def test_unavailable_engine_records_reason_and_engine_id() -> None:
    registry = ParsingEngineRegistry()
    registry.register(
        ParsingEngineChoice.DOCLING,
        lambda: MockParsingEngine(engine_id="docling", available=False),
    )
    parser = Parser(
        config=PipelineConfig(parsing_engine=ParsingEngineChoice.DOCLING),
        registry=registry,
    )
    job = _job()

    with pytest.raises(ParseFailure) as exc:
        parser.parse(job, _source())

    assert exc.value.kind is ParseFailureKind.ENGINE_UNAVAILABLE
    assert exc.value.engine_id == "docling"
    stage = job.stageStates[Stage.PARSING]
    assert stage.status is StageStatus.FAILED
    assert "docling" in stage.failureReason
    assert "unavailable" in stage.failureReason
    assert job.overallStatus is OverallStatus.FAILED
    assert job.failingStage is Stage.PARSING


# -- fail-closed: parse error, no partial output (Req 2.5) ---------------


def test_parse_error_records_reason_and_retains_no_output() -> None:
    registry = ParsingEngineRegistry()
    registry.register(
        ParsingEngineChoice.DOCLING,
        lambda: MockParsingEngine(
            engine_id="docling", raise_on_parse=ParseError("malformed stream")
        ),
    )
    parser = Parser(
        config=PipelineConfig(parsing_engine=ParsingEngineChoice.DOCLING),
        registry=registry,
    )
    job = _job()

    with pytest.raises(ParseFailure) as exc:
        parser.parse(job, _source())

    # No ParsedDocument is returned — the exception carries the failure and the
    # job records the reason; there is no partial output to inspect (Req 2.5).
    assert exc.value.kind is ParseFailureKind.PARSE_ERROR
    stage = job.stageStates[Stage.PARSING]
    assert stage.status is StageStatus.FAILED
    assert "parse error" in stage.failureReason
    assert "malformed stream" in stage.failureReason
    assert job.overallStatus is OverallStatus.FAILED


# -- fail-closed: timeout (Req 2.7) --------------------------------------


def test_engine_timeout_error_records_timeout_reason() -> None:
    registry = ParsingEngineRegistry()
    registry.register(
        ParsingEngineChoice.DOCLING,
        lambda: MockParsingEngine(
            engine_id="docling", raise_on_parse=ParseTimeoutError("slow")
        ),
    )
    parser = Parser(
        config=PipelineConfig(
            parsing_engine=ParsingEngineChoice.DOCLING, parse_timeout_seconds=7
        ),
        registry=registry,
    )
    job = _job()

    with pytest.raises(ParseFailure) as exc:
        parser.parse(job, _source())

    assert exc.value.kind is ParseFailureKind.TIMEOUT
    stage = job.stageStates[Stage.PARSING]
    assert stage.status is StageStatus.FAILED
    assert "7s" in stage.failureReason
    assert job.overallStatus is OverallStatus.FAILED


def test_engine_exceeding_deadline_without_self_abort_times_out() -> None:
    # Engine returns content but the clock shows it overran the deadline (Req 2.7).
    registry = ParsingEngineRegistry()
    registry.register(
        ParsingEngineChoice.DOCLING,
        lambda: MockParsingEngine(
            engine_id="docling", preset_result=_content_preset("docling")
        ),
    )
    ticks = iter([0.0, 50.0])  # start=0, end=50 with a 1s limit -> overrun
    parser = Parser(
        config=PipelineConfig(
            parsing_engine=ParsingEngineChoice.DOCLING, parse_timeout_seconds=1
        ),
        registry=registry,
        clock=lambda: next(ticks),
    )
    job = _job()

    with pytest.raises(ParseFailure) as exc:
        parser.parse(job, _source())

    assert exc.value.kind is ParseFailureKind.TIMEOUT
    assert job.overallStatus is OverallStatus.FAILED


# -- fail-closed: no extractable content (Req 2.8) -----------------------


def test_empty_result_records_no_content_reason() -> None:
    registry = ParsingEngineRegistry()
    registry.register(
        ParsingEngineChoice.DOCLING,
        lambda: MockParsingEngine(
            engine_id="docling", preset_result=RawParseResult(engine_id="docling")
        ),
    )
    parser = Parser(
        config=PipelineConfig(parsing_engine=ParsingEngineChoice.DOCLING),
        registry=registry,
    )
    job = _job()

    with pytest.raises(ParseFailure) as exc:
        parser.parse(job, _source())

    assert exc.value.kind is ParseFailureKind.NO_CONTENT
    stage = job.stageStates[Stage.PARSING]
    assert stage.status is StageStatus.FAILED
    assert "no extractable content" in stage.failureReason
    assert job.overallStatus is OverallStatus.FAILED


# -- figure caption presence / absence (Req 3.3, 3.4) --------------------


def test_figure_caption_present_is_preserved() -> None:
    registry = ParsingEngineRegistry()
    registry.register(
        ParsingEngineChoice.DOCLING,
        lambda: MockParsingEngine(
            engine_id="docling",
            preset_result=RawParseResult(
                engine_id="docling",
                figures=[
                    RawFigure(
                        page_number=3,
                        image_ref="fig-A",
                        caption="Figure 2: dose response",
                    )
                ],
            ),
        ),
    )
    parser = Parser(
        config=PipelineConfig(parsing_engine=ParsingEngineChoice.DOCLING),
        registry=registry,
    )

    parsed = parser.parse(_job(), _source())

    assert len(parsed.figures) == 1
    figure = parsed.figures[0]
    assert figure.imageRef == "fig-A"
    assert figure.caption == "Figure 2: dose response"
    assert figure.pageNumber == 3


def test_figure_caption_absent_is_recorded_as_none_without_failing() -> None:
    registry = ParsingEngineRegistry()
    registry.register(
        ParsingEngineChoice.DOCLING,
        lambda: MockParsingEngine(
            engine_id="docling",
            preset_result=RawParseResult(
                engine_id="docling",
                figures=[RawFigure(page_number=0, image_ref="fig-B")],
            ),
        ),
    )
    parser = Parser(
        config=PipelineConfig(parsing_engine=ParsingEngineChoice.DOCLING),
        registry=registry,
    )
    job = _job()

    parsed = parser.parse(job, _source())  # absent caption must not fail (Req 3.4)

    assert parsed.figures[0].caption is None
    assert job.overallStatus is not OverallStatus.FAILED


# -- degraded table handling (Req 3.6) -----------------------------------


def test_degraded_table_is_flagged_and_retains_raw_text() -> None:
    registry = ParsingEngineRegistry()
    registry.register(
        ParsingEngineChoice.DOCLING,
        lambda: MockParsingEngine(
            engine_id="docling",
            preset_result=RawParseResult(
                engine_id="docling",
                tables=[
                    RawTable(
                        page_number=4,
                        degraded=True,
                        raw_text="col1 col2\n1 2",
                    )
                ],
            ),
        ),
    )
    parser = Parser(
        config=PipelineConfig(parsing_engine=ParsingEngineChoice.DOCLING),
        registry=registry,
    )

    parsed = parser.parse(_job(), _source())

    table = parsed.tables[0]
    assert table.degraded is True
    assert table.rawText == "col1 col2\n1 2"
    assert table.cells == []
    assert table.pageNumber == 4


def test_non_degraded_table_with_cells_is_not_flagged() -> None:
    registry = ParsingEngineRegistry()
    registry.register(
        ParsingEngineChoice.DOCLING,
        lambda: MockParsingEngine(
            engine_id="docling",
            preset_result=RawParseResult(
                engine_id="docling",
                tables=[
                    RawTable(
                        page_number=0,
                        cells=[RawTableCell(row_index=0, col_index=0, value="x")],
                    )
                ],
            ),
        ),
    )
    parser = Parser(
        config=PipelineConfig(parsing_engine=ParsingEngineChoice.DOCLING),
        registry=registry,
    )

    parsed = parser.parse(_job(), _source())

    table = parsed.tables[0]
    assert table.degraded is False
    assert [c.value for c in table.cells] == ["x"]
