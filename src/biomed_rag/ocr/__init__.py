"""OCR_Processor component for the biomedical RAG pipeline (Req 4).

Exposes the pluggable :class:`OCRProcessor` port, its result types
(``OCRText`` / ``OCRError`` / ``OCRTimeout``), the confidence/low-confidence
contract, and a deterministic in-memory adapter for tests. Wiring OCR into the
Parser is handled separately (task 6.2).
"""

from __future__ import annotations

from biomed_rag.ocr.mock import InMemoryOCRProcessor, PlannedOCR
from biomed_rag.ocr.processor import (
    Deadline,
    EmbeddedImage,
    OCRError,
    OCRProcessor,
    OCRResult,
    OCRText,
    OCRTimeout,
    PageImage,
    build_ocr_text_block,
    flag_low_confidence,
)

__all__ = [
    # port
    "OCRProcessor",
    # inputs
    "Deadline",
    "PageImage",
    "EmbeddedImage",
    # results
    "OCRResult",
    "OCRText",
    "OCRError",
    "OCRTimeout",
    # confidence handling
    "flag_low_confidence",
    "build_ocr_text_block",
    # deterministic test adapter
    "InMemoryOCRProcessor",
    "PlannedOCR",
]
