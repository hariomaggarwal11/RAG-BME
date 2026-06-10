"""Smoke / unit tests for the OCR_Processor (task 6.1, Req 4.3, 4.4).

These exercise the deterministic in-memory adapter and the confidence contract.
The property-based tests for confidence bounds (Property 7), resilience
(Property 8), and the per-page timeout unit test are separate tasks (6.3-6.5).
"""

from __future__ import annotations

import pytest

from biomed_rag.config import PipelineConfig
from biomed_rag.models.enums import BlockSource, BlockType, OCRErrorKind
from biomed_rag.ocr import (
    Deadline,
    EmbeddedImage,
    InMemoryOCRProcessor,
    OCRError,
    OCRText,
    OCRTimeout,
    PageImage,
    PlannedOCR,
    build_ocr_text_block,
    flag_low_confidence,
)


def _deadline() -> Deadline:
    return Deadline(seconds=PipelineConfig().ocr_page_timeout_seconds)


def test_process_page_returns_text_and_confidence() -> None:
    proc = InMemoryOCRProcessor(
        {"p1": PlannedOCR(text="recovered text", confidence=0.92)}
    )
    result = proc.process_page(PageImage(imageRef="p1", pageNumber=0), _deadline())
    assert isinstance(result, OCRText)
    assert result.text == "recovered text"
    assert result.confidence == pytest.approx(0.92)


def test_process_embedded_image_returns_text() -> None:
    proc = InMemoryOCRProcessor(
        {"img1": PlannedOCR(text="figure label", confidence=0.5)}
    )
    result = proc.process_embedded_image(
        EmbeddedImage(imageRef="img1", pageNumber=3), _deadline()
    )
    assert isinstance(result, OCRText)
    assert result.text == "figure label"


def test_unreadable_image_returns_error_not_raises() -> None:
    proc = InMemoryOCRProcessor(
        {"bad": PlannedOCR(error_kind=OCRErrorKind.CORRUPT)}
    )
    result = proc.process_page(PageImage(imageRef="bad", pageNumber=1), _deadline())
    assert isinstance(result, OCRError)
    assert result.kind is OCRErrorKind.CORRUPT
    assert result.imageRef == "bad"
    assert result.pageNumber == 1


def test_missing_plan_defaults_to_unreadable_error() -> None:
    proc = InMemoryOCRProcessor()
    result = proc.process_page(PageImage(imageRef="unknown", pageNumber=0), _deadline())
    assert isinstance(result, OCRError)
    assert result.kind is OCRErrorKind.UNREADABLE


def test_slow_page_times_out() -> None:
    proc = InMemoryOCRProcessor(
        {"slow": PlannedOCR(text="x", confidence=1.0, duration_seconds=120.0)}
    )
    result = proc.process_page(
        PageImage(imageRef="slow", pageNumber=2), Deadline(seconds=60)
    )
    assert isinstance(result, OCRTimeout)
    assert result.kind is OCRErrorKind.TIMEOUT


def test_confidence_must_be_within_unit_interval() -> None:
    with pytest.raises(Exception):
        OCRText(text="x", confidence=1.5)
    with pytest.raises(Exception):
        OCRText(text="x", confidence=-0.1)


def test_low_confidence_flag_is_strict_below_threshold() -> None:
    # Below threshold -> flagged.
    assert flag_low_confidence(0.69, 0.70) is True
    # Exactly at threshold -> NOT flagged.
    assert flag_low_confidence(0.70, 0.70) is False
    # Above threshold -> not flagged.
    assert flag_low_confidence(0.95, 0.70) is False


def test_build_text_block_retains_text_and_sets_flag() -> None:
    threshold = PipelineConfig().ocr_confidence_threshold  # 0.70
    low = OCRText(text="blurry scan", confidence=0.4)
    block = build_ocr_text_block(
        low, page_number=5, reading_order_position=0, threshold=threshold
    )
    assert block.type is BlockType.OCR_TEXT
    assert block.source is BlockSource.OCR
    assert block.text == "blurry scan"  # text retained regardless of flag
    assert block.ocrConfidence == pytest.approx(0.4)
    assert block.lowConfidence is True

    high = OCRText(text="clean scan", confidence=0.99)
    block2 = build_ocr_text_block(
        high, page_number=5, reading_order_position=1, threshold=threshold
    )
    assert block2.text == "clean scan"
    assert block2.lowConfidence is False
