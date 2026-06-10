"""Deterministic in-memory OCR adapter for tests (design Testing Strategy).

The concrete OCR engine is out of scope for this task; property and unit tests
drive the :class:`OCRProcessor` port through this fully deterministic adapter.
Each image's outcome is described up front by a :class:`PlannedOCR`, so the same
inputs always yield the same results - including simulated timeouts, which are
decided by comparing a planned ``duration_seconds`` against the supplied
:class:`Deadline` rather than using a real clock.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from biomed_rag.models._validation import require, require_float_in_range
from biomed_rag.models.enums import OCRErrorKind
from biomed_rag.ocr.processor import (
    Deadline,
    EmbeddedImage,
    OCRError,
    OCRProcessor,
    OCRResult,
    OCRText,
    OCRTimeout,
    PageImage,
)

_FAILURE_KINDS = (
    OCRErrorKind.UNREADABLE,
    OCRErrorKind.CORRUPT,
    OCRErrorKind.UNSUPPORTED,
)


@dataclass(frozen=True)
class PlannedOCR:
    """A deterministic instruction describing what the mock engine produces for
    one image.

    Exactly one of two outcomes is described:

    * a successful extraction - set ``text`` and ``confidence`` (``error_kind``
      left ``None``);
    * a failure - set ``error_kind`` to an unreadable/corrupt/unsupported kind.

    ``duration_seconds`` is the simulated processing time. When it exceeds the
    operation's :class:`Deadline`, the mock returns :class:`OCRTimeout`
    regardless of the other fields (Req 4.6), which takes precedence so timeout
    behaviour can be exercised independently of the underlying outcome.
    """

    text: Optional[str] = None
    confidence: float = 1.0
    error_kind: Optional[OCRErrorKind] = None
    duration_seconds: float = 0.0

    def __post_init__(self) -> None:
        if self.error_kind is None:
            require(
                isinstance(self.text, str),
                "PlannedOCR.text must be a str for a successful outcome",
            )
            require_float_in_range(
                self.confidence, "PlannedOCR.confidence", minimum=0.0, maximum=1.0
            )
        else:
            require(
                self.error_kind in _FAILURE_KINDS,
                f"PlannedOCR.error_kind must be one of "
                f"{[k.value for k in _FAILURE_KINDS]}, got {self.error_kind!r}",
            )
        require(
            isinstance(self.duration_seconds, (int, float))
            and not isinstance(self.duration_seconds, bool),
            "PlannedOCR.duration_seconds must be a number",
        )
        require(
            self.duration_seconds >= 0,
            f"PlannedOCR.duration_seconds must be >= 0, got {self.duration_seconds!r}",
        )


class InMemoryOCRProcessor(OCRProcessor):
    """A deterministic :class:`OCRProcessor` driven by a per-image plan.

    Outcomes are looked up by ``imageRef``. A ``default`` plan covers any image
    not explicitly listed; when no plan and no default apply, the adapter returns
    an ``UNREADABLE`` :class:`OCRError` (the safe, resilient choice - a missing
    plan never raises and never aborts a batch).
    """

    def __init__(
        self,
        plans: Optional[Dict[str, PlannedOCR]] = None,
        *,
        default: Optional[PlannedOCR] = None,
    ) -> None:
        self._plans: Dict[str, PlannedOCR] = dict(plans or {})
        self._default = default

    def set_plan(self, image_ref: str, plan: PlannedOCR) -> None:
        """Register or replace the planned outcome for ``image_ref``."""
        self._plans[image_ref] = plan

    def process_page(self, page: PageImage, deadline: Deadline) -> OCRResult:
        return self._run(page.imageRef, page.pageNumber, deadline)

    def process_embedded_image(
        self, image: EmbeddedImage, deadline: Deadline
    ) -> OCRResult:
        return self._run(image.imageRef, image.pageNumber, deadline)

    # -- internals --------------------------------------------------------
    def _run(self, image_ref: str, page_number: int, deadline: Deadline) -> OCRResult:
        plan = self._plans.get(image_ref, self._default)
        if plan is None:
            # No instruction for this image: treat as unreadable so the caller
            # records an error and continues (Req 4.5).
            return OCRError(
                imageRef=image_ref,
                pageNumber=page_number,
                kind=OCRErrorKind.UNREADABLE,
            )

        # A timeout takes precedence over the planned outcome (Req 4.6).
        if deadline.exceeded_by(plan.duration_seconds):
            return OCRTimeout(imageRef=image_ref, pageNumber=page_number)

        if plan.error_kind is not None:
            return OCRError(
                imageRef=image_ref, pageNumber=page_number, kind=plan.error_kind
            )

        return OCRText(text=plan.text, confidence=plan.confidence)
