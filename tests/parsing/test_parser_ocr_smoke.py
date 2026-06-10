"""Smoke tests for OCR invocation wired into the Parser (Task 6.2).

These drive the Parser through the deterministic MockParsingEngine (for the
engine-produced content) and the deterministic InMemoryOCRProcessor (for the OCR
sub-activity), covering:

* OCR for pages lacking an extractable text layer -> OCR_TEXT blocks (Req 4.1).
* OCR for embedded images carrying text -> OCR_TEXT blocks (Req 4.2).
* OCR confidence recorded and low-confidence flag set below threshold
  (Req 4.3, 4.4).
* Per-image error for unreadable/corrupt/unsupported images and per-page
  timeouts recorded on ParsedDocument.ocrErrors while remaining pages/images
  still process (Req 4.5, 4.6).
* OCR disabled (no processor injected) leaves the Parser behaviour unchanged.
"""

from __future__ import annotations

from datetime import datetime, timezone

from biomed_rag.config import ParsingEngine as ParsingEngineChoice
from biomed_rag.config import PipelineConfig
from biomed_rag.models import (
    BlockSource,
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
    RawBlock,
    RawImage,
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
    ocr: InMemoryOCRProcessor | None = None,
    **config_kwargs,
) -> Parser:
    registry = ParsingEngineRegistry()
    registry.register(ParsingEngineChoice.DOCLING, lambda: engine)
    config = PipelineConfig(parsing_engine=ParsingEngineChoice.DOCLING, **config_kwargs)
    return Parser(config=config, registry=registry, ocr=ocr)


# -- page OCR (Req 4.1) ---------------------------------------------------


def test_image_only_page_is_recovered_as_ocr_text_block() -> None:
    preset = RawParseResult(
        engine_id="docling",
        pages=[RawPage(page_number=0, has_text_layer=False, page_image_ref="pg-0")],
    )
    ocr = InMemoryOCRProcessor(
        {"pg-0": PlannedOCR(text="recovered page text", confidence=0.95)}
    )
    parser = _parser(MockParsingEngine(engine_id="docling", preset_result=preset), ocr)
    job = _job()

    parsed = parser.parse(job, _source())  # OCR text is extractable content

    assert job.overallStatus is not OverallStatus.FAILED
    ocr_blocks = [b for b in parsed.blocks if b.type is BlockType.OCR_TEXT]
    assert len(ocr_blocks) == 1
    block = ocr_blocks[0]
    assert block.text == "recovered page text"
    assert block.source is BlockSource.OCR
    assert block.ocrConfidence == 0.95
    assert block.lowConfidence is False
    assert block.readingOrderPosition == 0
    assert parsed.ocrErrors == []


def test_page_with_text_layer_is_not_ocred() -> None:
    preset = RawParseResult(
        engine_id="docling",
        blocks=[RawBlock(text="native text", page_number=0)],
        pages=[RawPage(page_number=0, has_text_layer=True)],
    )
    ocr = InMemoryOCRProcessor(default=PlannedOCR(text="should not appear"))
    parser = _parser(MockParsingEngine(engine_id="docling", preset_result=preset), ocr)

    parsed = parser.parse(_job(), _source())

    assert [b.type for b in parsed.blocks] == [BlockType.PARAGRAPH]
    assert all(b.type is not BlockType.OCR_TEXT for b in parsed.blocks)


# -- embedded image OCR (Req 4.2) -----------------------------------------


def test_embedded_image_with_text_is_recovered() -> None:
    preset = RawParseResult(
        engine_id="docling",
        blocks=[RawBlock(text="surrounding paragraph", page_number=0)],
        images=[
            RawImage(page_number=0, image_ref="img-text", has_text=True),
            RawImage(page_number=0, image_ref="img-plain", has_text=False),
        ],
    )
    ocr = InMemoryOCRProcessor(
        {
            "img-text": PlannedOCR(text="text inside image", confidence=0.5),
            "img-plain": PlannedOCR(text="never invoked", confidence=1.0),
        }
    )
    parser = _parser(
        MockParsingEngine(engine_id="docling", preset_result=preset),
        ocr,
        ocr_confidence_threshold=0.70,
    )

    parsed = parser.parse(_job(), _source())

    ocr_blocks = [b for b in parsed.blocks if b.type is BlockType.OCR_TEXT]
    assert [b.text for b in ocr_blocks] == ["text inside image"]
    # confidence 0.5 < threshold 0.70 -> flagged low-confidence, text retained
    assert ocr_blocks[0].lowConfidence is True
    assert ocr_blocks[0].ocrConfidence == 0.5


# -- resilience: errors/timeouts recorded, others continue (Req 4.5, 4.6) -


def test_bad_images_record_errors_without_aborting_remaining() -> None:
    preset = RawParseResult(
        engine_id="docling",
        pages=[
            RawPage(page_number=0, has_text_layer=False, page_image_ref="pg-good"),
            RawPage(page_number=1, has_text_layer=False, page_image_ref="pg-corrupt"),
            RawPage(page_number=2, has_text_layer=False, page_image_ref="pg-slow"),
        ],
        images=[
            RawImage(page_number=0, image_ref="img-unsupported", has_text=True),
        ],
    )
    ocr = InMemoryOCRProcessor(
        {
            "pg-good": PlannedOCR(text="good page", confidence=0.9),
            "pg-corrupt": PlannedOCR(error_kind=OCRErrorKind.CORRUPT),
            # duration exceeds the 1s page timeout -> OCRTimeout
            "pg-slow": PlannedOCR(text="late", confidence=0.9, duration_seconds=5.0),
            "img-unsupported": PlannedOCR(error_kind=OCRErrorKind.UNSUPPORTED),
        }
    )
    parser = _parser(
        MockParsingEngine(engine_id="docling", preset_result=preset),
        ocr,
        ocr_page_timeout_seconds=1,
    )

    parsed = parser.parse(_job(), _source())

    # The readable page still produced recovered text.
    ocr_blocks = [b for b in parsed.blocks if b.type is BlockType.OCR_TEXT]
    assert [b.text for b in ocr_blocks] == ["good page"]

    # Every bad item recorded a per-image error indication.
    recorded = {(e.imageRef, e.kind) for e in parsed.ocrErrors}
    assert recorded == {
        ("pg-corrupt", OCRErrorKind.CORRUPT),
        ("pg-slow", OCRErrorKind.TIMEOUT),
        ("img-unsupported", OCRErrorKind.UNSUPPORTED),
    }


def test_ocr_text_joins_contiguous_reading_order() -> None:
    preset = RawParseResult(
        engine_id="docling",
        blocks=[RawBlock(text="page-0 native", page_number=0)],
        pages=[RawPage(page_number=1, has_text_layer=False, page_image_ref="pg-1")],
    )
    ocr = InMemoryOCRProcessor({"pg-1": PlannedOCR(text="page-1 ocr", confidence=0.9)})
    parser = _parser(MockParsingEngine(engine_id="docling", preset_result=preset), ocr)

    parsed = parser.parse(_job(), _source())

    positions = sorted(b.readingOrderPosition for b in parsed.blocks)
    assert positions == [0, 1]
    assert len(set(positions)) == 2


# -- OCR disabled when no processor injected ------------------------------


def test_no_processor_leaves_parser_unchanged() -> None:
    preset = RawParseResult(
        engine_id="docling",
        blocks=[RawBlock(text="native text", page_number=0)],
        pages=[RawPage(page_number=1, has_text_layer=False, page_image_ref="pg-1")],
        images=[RawImage(page_number=0, image_ref="img", has_text=True)],
    )
    parser = _parser(MockParsingEngine(engine_id="docling", preset_result=preset))

    parsed = parser.parse(_job(), _source())

    assert all(b.type is not BlockType.OCR_TEXT for b in parsed.blocks)
    assert parsed.ocrErrors == []
