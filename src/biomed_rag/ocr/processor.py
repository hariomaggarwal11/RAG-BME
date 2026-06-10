"""OCR_Processor port, result types, and confidence handling (Req 4.3, 4.4).

The :class:`OCRProcessor` is a pluggable port: the concrete OCR engine (e.g. a
Tesseract / cloud OCR backend) lives behind this interface so the rest of the
pipeline depends only on the contract. Per the design (OCR_Processor section):

    interface OCRProcessor:
        processPage(pageImage, deadline) -> OCRResult
        processEmbeddedImage(image, deadline) -> OCRResult
        # OCRResult = { text, confidence } | OCRError | OCRTimeout

This module also owns the confidence contract:

* every recovered text block carries a confidence score in ``[0.0, 1.0]`` (Req 4.3);
* a block is flagged low-confidence exactly when its confidence is strictly below
  the configured ``ocr_confidence_threshold`` while the extracted text is always
  retained (Req 4.4).

Wiring OCR invocation into the Parser is a separate concern (task 6.2); this
module deliberately does not import or modify the Parser.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Union

from biomed_rag.models._validation import (
    require,
    require_float_in_range,
    require_int_in_range,
    require_non_empty_str,
)
from biomed_rag.models.enums import BlockSource, BlockType, OCRErrorKind
from biomed_rag.models.parsed import TextBlock

# Error kinds that represent a failed OCR extraction (as opposed to a timeout).
# A timeout is modelled separately by :class:`OCRTimeout` but maps to the
# ``TIMEOUT`` kind when recorded as an ``ImageOCRError`` by the Parser.
_FAILURE_KINDS = (
    OCRErrorKind.UNREADABLE,
    OCRErrorKind.CORRUPT,
    OCRErrorKind.UNSUPPORTED,
)


@dataclass(frozen=True)
class Deadline:
    """A wall-clock budget for a single OCR operation.

    The Parser derives a page deadline from ``ocr_page_timeout_seconds`` (Req
    4.6). The port receives the deadline so an adapter can abort work that would
    exceed it; tests use :meth:`exceeded_by` to decide timeouts deterministically
    without real clocks.
    """

    seconds: float

    def __post_init__(self) -> None:
        require(
            isinstance(self.seconds, (int, float)) and not isinstance(self.seconds, bool),
            "Deadline.seconds must be a number",
        )
        require(self.seconds > 0, f"Deadline.seconds must be > 0, got {self.seconds!r}")

    def exceeded_by(self, duration_seconds: float) -> bool:
        """Return ``True`` when an operation taking ``duration_seconds`` would
        exceed this deadline."""
        return float(duration_seconds) > float(self.seconds)


# -- OCR input descriptors ------------------------------------------------


@dataclass(frozen=True)
class PageImage:
    """A full-page image lacking an extractable text layer (Req 4.1)."""

    imageRef: str
    pageNumber: int

    def __post_init__(self) -> None:
        require_non_empty_str(self.imageRef, "PageImage.imageRef")
        require_int_in_range(self.pageNumber, "PageImage.pageNumber", minimum=0)


@dataclass(frozen=True)
class EmbeddedImage:
    """An image embedded in a page that may contain text (Req 4.2)."""

    imageRef: str
    pageNumber: int

    def __post_init__(self) -> None:
        require_non_empty_str(self.imageRef, "EmbeddedImage.imageRef")
        require_int_in_range(self.pageNumber, "EmbeddedImage.pageNumber", minimum=0)


# -- OCR results ----------------------------------------------------------


@dataclass(frozen=True)
class OCRText:
    """A successful OCR extraction: recovered text plus a confidence in [0, 1].

    This is the ``{ text, confidence }`` variant of ``OCRResult``. The
    low-confidence decision is intentionally not stored here; it is a function of
    the configured threshold and is computed by :meth:`is_low_confidence` /
    :func:`flag_low_confidence`, so the same extraction can be evaluated against
    any threshold (Req 4.3, 4.4).
    """

    text: str
    confidence: float

    def __post_init__(self) -> None:
        require(isinstance(self.text, str), "OCRText.text must be a str")
        # Confidence is always within the closed unit interval (Req 4.3).
        object.__setattr__(
            self,
            "confidence",
            require_float_in_range(
                self.confidence, "OCRText.confidence", minimum=0.0, maximum=1.0
            ),
        )

    def is_low_confidence(self, threshold: float) -> bool:
        """Return ``True`` exactly when confidence is strictly below ``threshold``
        (Req 4.4)."""
        return flag_low_confidence(self.confidence, threshold)


@dataclass(frozen=True)
class OCRError:
    """A per-image OCR failure for an unreadable / corrupt / unsupported image
    (Req 4.5). Processing of remaining items continues; the Parser records this
    as an ``ImageOCRError``."""

    imageRef: str
    pageNumber: int
    kind: OCRErrorKind

    def __post_init__(self) -> None:
        require_non_empty_str(self.imageRef, "OCRError.imageRef")
        require_int_in_range(self.pageNumber, "OCRError.pageNumber", minimum=0)
        require(
            self.kind in _FAILURE_KINDS,
            f"OCRError.kind must be one of {[k.value for k in _FAILURE_KINDS]}, "
            f"got {self.kind!r}",
        )


@dataclass(frozen=True)
class OCRTimeout:
    """A per-page OCR timeout (Req 4.6). Modelled distinctly from
    :class:`OCRError` but maps to ``OCRErrorKind.TIMEOUT`` when recorded."""

    imageRef: str
    pageNumber: int
    kind: OCRErrorKind = OCRErrorKind.TIMEOUT

    def __post_init__(self) -> None:
        require_non_empty_str(self.imageRef, "OCRTimeout.imageRef")
        require_int_in_range(self.pageNumber, "OCRTimeout.pageNumber", minimum=0)
        require(
            self.kind is OCRErrorKind.TIMEOUT,
            "OCRTimeout.kind must be OCRErrorKind.TIMEOUT",
        )


# Discriminated union returned by the OCR port.
OCRResult = Union[OCRText, OCRError, OCRTimeout]


# -- Confidence handling --------------------------------------------------


def flag_low_confidence(confidence: float, threshold: float) -> bool:
    """Return ``True`` when ``confidence`` is strictly below ``threshold``.

    This is the single source of truth for the low-confidence rule (Req 4.4): a
    block is low-confidence *exactly when* ``confidence < threshold``. A block
    whose confidence equals the threshold is **not** flagged.
    """
    c = require_float_in_range(confidence, "confidence", minimum=0.0, maximum=1.0)
    t = require_float_in_range(threshold, "threshold", minimum=0.0, maximum=1.0)
    return c < t


def build_ocr_text_block(
    result: OCRText,
    *,
    page_number: int,
    reading_order_position: int,
    threshold: float,
) -> TextBlock:
    """Build an ``OCR_TEXT`` :class:`TextBlock` from a successful OCR result.

    The block always retains the extracted text and records the confidence; the
    ``lowConfidence`` flag is set when the confidence is below ``threshold`` (Req
    4.3, 4.4). The Parser uses this when storing recovered text (task 6.2); it is
    provided here so the confidence contract is owned by the OCR module.
    """
    require(isinstance(result, OCRText), "build_ocr_text_block requires an OCRText result")
    return TextBlock(
        type=BlockType.OCR_TEXT,
        text=result.text,
        pageNumber=page_number,
        readingOrderPosition=reading_order_position,
        source=BlockSource.OCR,
        ocrConfidence=result.confidence,
        lowConfidence=result.is_low_confidence(threshold),
    )


# -- The port -------------------------------------------------------------


class OCRProcessor(abc.ABC):
    """Pluggable OCR engine port (Req 4.1-4.6).

    Concrete adapters implement :meth:`process_page` and
    :meth:`process_embedded_image`. Both return an :data:`OCRResult` and never
    raise for a single bad image: failures are reported as :class:`OCRError` /
    :class:`OCRTimeout` so the caller can continue with remaining items (Req 4.5,
    4.6).
    """

    @abc.abstractmethod
    def process_page(self, page: PageImage, deadline: Deadline) -> OCRResult:
        """OCR a full page image lacking a text layer (Req 4.1)."""
        raise NotImplementedError

    @abc.abstractmethod
    def process_embedded_image(
        self, image: EmbeddedImage, deadline: Deadline
    ) -> OCRResult:
        """OCR an embedded image that may contain text (Req 4.2)."""
        raise NotImplementedError
