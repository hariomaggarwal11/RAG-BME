"""Property test for OCR confidence bounds and the low-confidence flag (Req 4.3, 4.4).

Feature: biomedical-rag-pipeline, Property 7: OCR confidence is bounded and the low-confidence flag is correct

Statement: for any OCR-extracted block with confidence ``c`` and threshold ``t``
both in ``[0.0, 1.0]``, the resulting block records a confidence in ``[0.0, 1.0]``,
is flagged low-confidence exactly when ``c < t``, and retains the extracted text
regardless of the flag.

The confidence contract is owned by the OCR module: ``OCRText`` carries the raw
``{text, confidence}`` extraction and ``build_ocr_text_block`` turns it into an
``OCR_TEXT`` :class:`TextBlock`, computing ``lowConfidence`` against the configured
threshold while always retaining the text (the Parser wiring is a separate task).
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from biomed_rag.models.enums import BlockSource, BlockType
from biomed_rag.ocr import OCRText, build_ocr_text_block

# Arbitrary text, including the empty string: OCR may recover an empty result and
# the text must still be retained verbatim.
_TEXT = st.text(max_size=64)

# Confidence and threshold range over the full closed unit interval, including the
# boundaries 0.0 and 1.0 where the strict ``c < t`` rule is most easily violated.
_UNIT = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)

# Reading-order positions are non-negative; the exact value is irrelevant to this
# property but must be valid for TextBlock construction.
_POSITION = st.integers(min_value=0, max_value=10_000)


# Feature: biomedical-rag-pipeline, Property 7: OCR confidence is bounded and the low-confidence flag is correct
@settings(max_examples=200)
@given(text=_TEXT, c=_UNIT, t=_UNIT, page=_POSITION, position=_POSITION)
def test_ocr_confidence_bounded_and_low_confidence_flag_correct(
    text: str, c: float, t: float, page: int, position: int
) -> None:
    """Validates: Requirements 4.3, 4.4"""
    result = OCRText(text=text, confidence=c)
    block = build_ocr_text_block(
        result,
        page_number=page,
        reading_order_position=position,
        threshold=t,
    )

    # Req 4.3: confidence is always recorded within the closed unit interval.
    assert block.ocrConfidence is not None
    assert 0.0 <= block.ocrConfidence <= 1.0
    assert block.ocrConfidence == c

    # Req 4.4: a block is flagged low-confidence *exactly when* c < t (strict;
    # equal-to-threshold is NOT flagged).
    assert block.lowConfidence == (c < t)

    # Req 4.4: the extracted text is retained verbatim regardless of the flag,
    # and the block is tagged as OCR-sourced recovered text.
    assert block.text == text
    assert block.type is BlockType.OCR_TEXT
    assert block.source is BlockSource.OCR
