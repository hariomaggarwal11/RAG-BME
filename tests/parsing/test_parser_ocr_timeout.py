"""Unit test for per-page OCR timeout handling in the Parser (Task 6.5).

Requirement 4.6: WHEN OCR processing of a single page image exceeds the
configured per-page timeout (``ocr_page_timeout_seconds``), THE OCR_Processor
SHALL abort OCR for that page, record a timeout indication for the affected page
in the Parsed_Document, and continue processing the remaining pages.

These tests drive the Parser through the deterministic MockParsingEngine and the
deterministic InMemoryOCRProcessor. A page's simulated processing time is set via
``PlannedOCR.duration_seconds``; when it exceeds the configured page timeout the
mock returns an OCRTimeout, which the Parser records as an ImageOCRError with
``kind`` TIMEOUT while continuing to process the remaining readable pages.
"""

from __future__ import annotations

from datetime import datetime, timezone

from biomed_rag.config import ParsingEngine as ParsingEngineChoice
from biomed_rag.config import PipelineConfig
from biomed_rag.models import (
    BlockType,
    DocumentMetadata,
    Format,
    OCRErrorKind,
    OverallStatus,
    ProcessingJob,
)
from biomed_rag.ocr import InMemoryOCRProcessor, PlannedOCR
from biomed_rag.parsing import (
    MockParsingEngine,
    Parser,
    ParsingEngineRegistry,
)
from biomed_rag.parsing.raw_result import (
    RawPage,
    RawParseResult,
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


def _parser(
    engine: MockParsingEngine,
    ocr: InMemoryOCRProcessor,
    **config_kwargs,
) -> Parser:
    registry = ParsingEngineRegistry()
    registry.register(ParsingEngineChoice.DOCLING, lambda: engine)
    config = PipelineConfig(parsing_engine=ParsingEngineChoice.DOCLING, **config_kwargs)
    return Parser(config=config, registry=registry, ocr=ocr)


def test_page_exceeding_timeout_records_timeout_and_processing_continues() -> None:
    # A slow page (duration 90s > 60s page timeout) plus a readable page.
    preset = RawParseResult(
        engine_id="docling",
        pages=[
            RawPage(page_number=0, has_text_layer=False, page_image_ref="pg-slow"),
            RawPage(page_number=1, has_text_layer=False, page_image_ref="pg-ok"),
        ],
    )
    ocr = InMemoryOCRProcessor(
        {
            # Simulated processing time exceeds the configured page timeout.
            "pg-slow": PlannedOCR(
                text="should not surface", confidence=0.95, duration_seconds=90.0
            ),
            "pg-ok": PlannedOCR(text="recovered ok page", confidence=0.95),
        }
    )
    parser = _parser(
        MockParsingEngine(engine_id="docling", preset_result=preset),
        ocr,
        ocr_page_timeout_seconds=60,
    )
    job = _job()

    parsed = parser.parse(job, _source())

    # The timeout on one page does not fail the whole job.
    assert job.overallStatus is not OverallStatus.FAILED

    # The slow page recorded exactly one TIMEOUT indication for that page.
    timeout_errors = [e for e in parsed.ocrErrors if e.kind is OCRErrorKind.TIMEOUT]
    assert len(timeout_errors) == 1
    assert timeout_errors[0].imageRef == "pg-slow"
    assert timeout_errors[0].pageNumber == 0

    # No OCR text leaked from the timed-out page.
    ocr_blocks = [b for b in parsed.blocks if b.type is BlockType.OCR_TEXT]
    assert [b.text for b in ocr_blocks] == ["recovered ok page"]


def test_timeout_uses_configured_page_timeout_boundary() -> None:
    # duration_seconds == timeout is NOT a timeout (deadline is "exceeded by"
    # strictly greater); a duration above the timeout IS a timeout.
    preset = RawParseResult(
        engine_id="docling",
        pages=[
            RawPage(page_number=0, has_text_layer=False, page_image_ref="pg-at"),
            RawPage(page_number=1, has_text_layer=False, page_image_ref="pg-over"),
        ],
    )
    ocr = InMemoryOCRProcessor(
        {
            "pg-at": PlannedOCR(text="at limit", confidence=0.9, duration_seconds=10.0),
            "pg-over": PlannedOCR(
                text="over limit", confidence=0.9, duration_seconds=10.5
            ),
        }
    )
    parser = _parser(
        MockParsingEngine(engine_id="docling", preset_result=preset),
        ocr,
        ocr_page_timeout_seconds=10,
    )

    parsed = parser.parse(_job(), _source())

    # Only the page strictly over the timeout records a TIMEOUT indication.
    recorded = {(e.imageRef, e.kind) for e in parsed.ocrErrors}
    assert recorded == {("pg-over", OCRErrorKind.TIMEOUT)}

    # The page exactly at the limit still produced its recovered text.
    ocr_texts = [
        b.text for b in parsed.blocks if b.type is BlockType.OCR_TEXT
    ]
    assert ocr_texts == ["at limit"]
