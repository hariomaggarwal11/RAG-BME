"""The Parser: block/heading extraction, reading-order, and fail-closed handling.

The :class:`Parser` selects the configured :class:`ParsingEngine` from a
registry, runs it against a :class:`SourceDocument`, and turns the engine-neutral
:class:`RawParseResult` into the canonical
:class:`~biomed_rag.models.ParsedDocument`.

Scope of this module (task 5.3):

* Produce :class:`~biomed_rag.models.TextBlock`\\ s in reading-sequence order with
  structural metadata (block type, page number, zero-based reading-order
  position) (Req 2.1).
* Order multi-column layouts top-to-bottom within a column and left-to-right
  across columns (Req 2.3).
* Preserve the heading hierarchy with nesting levels, emitting both HEADING
  blocks and :class:`~biomed_rag.models.Heading` entries (Req 2.4).
* Fail closed — retaining no partial output — for the four failure modes, each
  marking the Processing_Job failed with the recorded reason:
  engine unavailable (Req 2.6), parse error (Req 2.5), parse timeout
  (``parse_timeout_seconds``, Req 2.7), and no extractable content (Req 2.8).

Table and figure extraction (task 5.4) is handled here as well:

* Map every non-empty table cell to exactly one ``(rowIndex, colIndex)`` and
  carry spanning cells at their top-left index with the recorded ``rowSpan`` /
  ``colSpan`` (Req 3.1, 3.2).
* Extract figures with their optional caption, recording an absent caption as
  ``None`` without failing the extraction (Req 3.3, 3.4).
* Associate every table and figure with its source page number and its
  zero-based position in the single document reading-order sequence shared with
  the text blocks (Req 3.5).
* Flag degraded tables and retain their raw region text (Req 3.6).

OCR wiring (task 6.2) is handled here when an :class:`OCRProcessor` is injected:

* Invoke OCR for pages lacking an extractable text layer (``has_text_layer`` is
  ``False`` with a ``page_image_ref``) via :meth:`OCRProcessor.process_page`
  (Req 4.1) and for embedded images carrying text (``RawImage.has_text``) via
  :meth:`OCRProcessor.process_embedded_image` (Req 4.2).
* Store recovered text as ``OCR_TEXT`` :class:`~biomed_rag.models.TextBlock`\\ s
  participating in the single document reading-order sequence, carrying the OCR
  confidence and the low-confidence flag (Req 4.3, 4.4).
* Record a per-image :class:`~biomed_rag.models.ImageOCRError` for
  unreadable/corrupt/unsupported images and a per-page timeout
  (``ocr_page_timeout_seconds``) while continuing the remaining pages/images —
  a single bad image never aborts the whole parse (Req 4.5, 4.6).

When no :class:`OCRProcessor` is injected, OCR is disabled and the Parser behaves
exactly as before.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple

from biomed_rag.config import PipelineConfig
from biomed_rag.models import (
    BlockSource,
    BlockType,
    Cell,
    Figure,
    Heading,
    ImageOCRError,
    OverallStatus,
    ParsedDocument,
    ProcessingJob,
    Stage,
    StageState,
    StageStatus,
    Table,
    TextBlock,
)
from biomed_rag.ocr import (
    Deadline,
    EmbeddedImage,
    OCRError,
    OCRProcessor,
    OCRText,
    OCRTimeout,
    PageImage,
    build_ocr_text_block,
)

from .engine import (
    EngineUnavailableError,
    ParseError,
    ParseTimeoutError,
    ParsingEngine,
)
from .raw_result import (
    BBox,
    RawBlock,
    RawFigure,
    RawParseResult,
    RawTable,
    SourceDocument,
)
from .registry import (
    ParsingEngineRegistry,
    ParsingEngineNotRegisteredError,
    default_registry,
)


class ParseFailureKind(Enum):
    """The four fail-closed parsing outcomes (Req 2.5, 2.6, 2.7, 2.8)."""

    ENGINE_UNAVAILABLE = "engine_unavailable"  # Req 2.6
    PARSE_ERROR = "parse_error"  # Req 2.5
    TIMEOUT = "timeout"  # Req 2.7
    NO_CONTENT = "no_extractable_content"  # Req 2.8


class ParseFailure(Exception):
    """Raised when parsing fails closed; carries the recorded failure reason.

    The Parser has already marked the Processing_Job failed (recording the same
    reason on the PARSING stage state) by the time this propagates. No partial
    :class:`ParsedDocument` is produced (Req 2.5).
    """

    def __init__(
        self,
        kind: ParseFailureKind,
        reason: str,
        *,
        engine_id: Optional[str] = None,
    ) -> None:
        self.kind = kind
        self.reason = reason
        # Set for ENGINE_UNAVAILABLE so callers can report the engine (Req 2.6).
        self.engine_id = engine_id
        super().__init__(reason)


# Engine ``kind`` hints map to the canonical block types; anything unknown falls
# back to OTHER. A block is treated as a heading when this map says HEADING *or*
# when the engine supplied a heading level (see :func:`_is_heading`).
_KIND_TO_BLOCK_TYPE: Dict[str, BlockType] = {
    "paragraph": BlockType.PARAGRAPH,
    "heading": BlockType.HEADING,
    "title": BlockType.HEADING,
    "caption": BlockType.CAPTION,
    "ocr_text": BlockType.OCR_TEXT,
    "list_item": BlockType.LIST_ITEM,
    "footnote": BlockType.FOOTNOTE,
}


def _is_heading(raw: RawBlock) -> bool:
    """A block is a heading when it carries a nesting level or is typed heading."""
    if raw.heading_level is not None:
        return True
    return raw.kind.strip().lower() in ("heading", "title")


def _block_type(raw: RawBlock) -> BlockType:
    if _is_heading(raw):
        return BlockType.HEADING
    return _KIND_TO_BLOCK_TYPE.get(raw.kind.strip().lower(), BlockType.OTHER)


def _heading_level(raw: RawBlock) -> int:
    """The nesting level for a heading block; defaults to 1 when unspecified."""
    return raw.heading_level if raw.heading_level is not None else 1


def _block_source(raw: RawBlock) -> BlockSource:
    return BlockSource.OCR if raw.from_ocr else BlockSource.TEXT_LAYER


@dataclass
class _Orderable:
    """A content item participating in the single document reading-order sequence.

    Blocks, tables, and figures are ordered together so that their assigned
    reading-order positions form one contiguous zero-based sequence (Req 3.5).
    ``origin`` is a stable global index (blocks first, then tables, then
    figures) used as the final, deterministic tie-break; it also makes
    block-only documents order exactly as before.
    """

    kind: str  # "block" | "table" | "figure"
    page_number: int
    origin: int
    bbox: Optional[BBox] = None
    column: Optional[int] = None
    ref: object = None


def _collect_orderables(raw: RawParseResult) -> List[_Orderable]:
    """Wrap every block, table, and figure into a single orderable list.

    Origins are assigned blocks-first, then tables, then figures so that, absent
    layout geometry, blocks keep their emitted order and tables/figures fall
    after the blocks on the same page (a deterministic tie-break).
    """
    items: List[_Orderable] = []
    origin = 0
    for rb in raw.blocks:
        items.append(
            _Orderable(
                kind="block",
                page_number=rb.page_number,
                origin=origin,
                bbox=rb.bbox,
                column=rb.column,
                ref=rb,
            )
        )
        origin += 1
    for rt in raw.tables:
        items.append(
            _Orderable(
                kind="table",
                page_number=rt.page_number,
                origin=origin,
                bbox=rt.bbox,
                ref=rt,
            )
        )
        origin += 1
    for rf in raw.figures:
        items.append(
            _Orderable(
                kind="figure",
                page_number=rf.page_number,
                origin=origin,
                bbox=rf.bbox,
                ref=rf,
            )
        )
        origin += 1
    return items


def _infer_bbox_columns(items: List[_Orderable]) -> Dict[int, int]:
    """Infer left-to-right column indices from bounding boxes on a single page.

    Items are clustered by their left edge (``x0``): a gap wider than half the
    average item width opens a new column. Columns are numbered left-to-right
    starting at 0. Only items that carry a ``bbox`` participate; the rest are
    handled by the caller. The returned map is keyed by ``origin``.
    """
    boxed = [it for it in items if it.bbox is not None]
    if not boxed:
        return {}

    xs = sorted({it.bbox.x0 for it in boxed})  # type: ignore[union-attr]
    widths = [
        it.bbox.x1 - it.bbox.x0  # type: ignore[union-attr]
        for it in boxed
        if it.bbox is not None and it.bbox.x1 > it.bbox.x0
    ]
    avg_width = (sum(widths) / len(widths)) if widths else 0.0
    threshold = max(avg_width * 0.5, 1.0)

    band_of_x: Dict[float, int] = {}
    band = 0
    prev: Optional[float] = None
    for x in xs:
        if prev is not None and (x - prev) > threshold:
            band += 1
        band_of_x[x] = band
        prev = x

    return {it.origin: band_of_x[it.bbox.x0] for it in boxed}  # type: ignore[union-attr]


def _assign_columns(items: List[_Orderable]) -> Dict[int, int]:
    """Resolve a column index for every item, keyed by ``origin``.

    An explicit ``column`` hint (only blocks carry one) is authoritative.
    Otherwise the column is inferred from the bounding box per page; items with
    neither hint default to column 0 (single-column reading order).
    """
    columns: Dict[int, int] = {}
    per_page: Dict[int, List[_Orderable]] = {}
    for it in items:
        per_page.setdefault(it.page_number, []).append(it)

    for page_items in per_page.values():
        to_infer = [it for it in page_items if it.column is None]
        inferred = _infer_bbox_columns(to_infer)
        for it in page_items:
            if it.column is not None:
                columns[it.origin] = it.column
            else:
                columns[it.origin] = inferred.get(it.origin, 0)
    return columns


def _reading_order(items: List[_Orderable]) -> List[_Orderable]:
    """Return content items in document reading order (Req 2.1, 2.3, 3.5).

    Ordering key, applied in priority order:

    1. page number (ascending),
    2. column (ascending) — left-to-right across columns,
    3. ``bbox.y0`` (ascending) — top-to-bottom within a column,
    4. ``bbox.x0`` (ascending) — left-to-right within a row,
    5. ``origin`` — a stable tie-break so the result is deterministic and,
       absent any layout hints, preserves the engine's emitted order.
    """
    columns = _assign_columns(items)

    def key(it: _Orderable) -> Tuple[int, int, float, float, int]:
        y0 = it.bbox.y0 if it.bbox is not None else 0.0
        x0 = it.bbox.x0 if it.bbox is not None else 0.0
        return (it.page_number, columns[it.origin], y0, x0, it.origin)

    return sorted(items, key=key)


def _build_cells(raw_table: RawTable) -> List[Cell]:
    """Map every non-empty source cell to exactly one (row, col) (Req 3.1, 3.2).

    Cells whose value is empty or whitespace-only are omitted — only non-empty
    source cells receive a coordinate. Spanning cells keep their top-left index
    with the recorded ``rowSpan`` / ``colSpan``.
    """
    cells: List[Cell] = []
    for rc in raw_table.cells:
        if not rc.value.strip():
            continue
        cells.append(
            Cell(
                rowIndex=rc.row_index,
                colIndex=rc.col_index,
                value=rc.value,
                rowSpan=rc.row_span,
                colSpan=rc.col_span,
            )
        )
    return cells


@dataclass
class Parser:
    """Converts a Source_Document into a canonical Parsed_Document (Req 2).

    The Parser depends only on the :class:`ParsingEngine` port; the concrete
    engine is selected from ``config.parsing_engine`` via the registry, so no
    business logic is tied to a specific backend (design: ports-and-adapters).
    """

    config: PipelineConfig
    registry: ParsingEngineRegistry = default_registry
    clock: Callable[[], float] = field(default=time.monotonic)
    ocr: Optional[OCRProcessor] = None

    def parse(
        self,
        job: Optional[ProcessingJob],
        source: SourceDocument,
    ) -> ParsedDocument:
        """Parse ``source`` into a :class:`ParsedDocument`.

        On success returns the parsed document (the Orchestrator owns the
        success stage transition). On any failure, marks ``job`` failed with the
        recorded reason and raises :class:`ParseFailure`, retaining no partial
        output (Req 2.5, 2.6, 2.7, 2.8).
        """
        if not isinstance(source, SourceDocument):
            raise TypeError("source must be a SourceDocument")

        engine = self._select_engine(job)

        if not engine.is_available():
            self._fail(
                job,
                ParseFailureKind.ENGINE_UNAVAILABLE,
                f"parsing engine {engine.engine_id()!r} is unavailable",
                engine_id=engine.engine_id(),
            )

        raw = self._run_engine(job, engine, source)
        document = self._build_document(source, raw)

        if not self._has_extractable_content(document, raw):
            self._fail(
                job,
                ParseFailureKind.NO_CONTENT,
                "no extractable content was found in the document",
            )

        return document

    # -- engine selection / execution ------------------------------------

    def _select_engine(self, job: Optional[ProcessingJob]) -> ParsingEngine:
        try:
            return self.registry.select(self.config)
        except ParsingEngineNotRegisteredError:
            # An unregistered engine is treated as unavailable so the job fails
            # closed rather than crashing (Req 2.6).
            engine_id = self.config.parsing_engine.value
            self._fail(
                job,
                ParseFailureKind.ENGINE_UNAVAILABLE,
                f"parsing engine {engine_id!r} is unavailable",
                engine_id=engine_id,
            )
            raise AssertionError("unreachable")  # pragma: no cover

    def _run_engine(
        self,
        job: Optional[ProcessingJob],
        engine: ParsingEngine,
        source: SourceDocument,
    ) -> RawParseResult:
        timeout = self.config.parse_timeout_seconds
        start = self.clock()
        deadline = start + timeout
        try:
            raw = engine.parse(source, deadline)
        except ParseTimeoutError:
            self._fail(
                job,
                ParseFailureKind.TIMEOUT,
                f"parsing exceeded the {timeout}s limit",
            )
            raise AssertionError("unreachable")  # pragma: no cover
        except EngineUnavailableError:
            self._fail(
                job,
                ParseFailureKind.ENGINE_UNAVAILABLE,
                f"parsing engine {engine.engine_id()!r} is unavailable",
                engine_id=engine.engine_id(),
            )
            raise AssertionError("unreachable")  # pragma: no cover
        except ParseError as exc:
            self._fail(
                job,
                ParseFailureKind.PARSE_ERROR,
                f"parse error: {exc}",
            )
            raise AssertionError("unreachable")  # pragma: no cover

        # Enforce the deadline even when the engine did not self-abort (Req 2.7).
        if (self.clock() - start) > timeout:
            self._fail(
                job,
                ParseFailureKind.TIMEOUT,
                f"parsing exceeded the {timeout}s limit",
            )
        return raw

    # -- document construction -------------------------------------------

    def _build_document(
        self,
        source: SourceDocument,
        raw: RawParseResult,
    ) -> ParsedDocument:
        items = _collect_orderables(raw)
        # OCR-recovered text joins the same reading-order sequence; its origins
        # start after every engine-produced item so it falls after engine
        # content on the same page (a deterministic tie-break).
        ocr_items, ocr_errors = self._run_ocr(raw, origin_start=len(items))
        ordered = _reading_order(items + ocr_items)

        threshold = self.config.ocr_confidence_threshold

        blocks: List[TextBlock] = []
        headings: List[Heading] = []
        tables: List[Table] = []
        figures: List[Figure] = []

        for position, item in enumerate(ordered):
            if item.kind == "block":
                rb: RawBlock = item.ref  # type: ignore[assignment]
                is_heading = _is_heading(rb)
                level = _heading_level(rb) if is_heading else None
                blocks.append(
                    TextBlock(
                        type=_block_type(rb),
                        text=rb.text,
                        pageNumber=rb.page_number,
                        readingOrderPosition=position,
                        source=_block_source(rb),
                        headingLevel=level,
                    )
                )
                if is_heading:
                    headings.append(
                        Heading(
                            level=level,  # type: ignore[arg-type]
                            text=rb.text,
                            pageNumber=rb.page_number,
                            readingOrderPosition=position,
                        )
                    )
            elif item.kind == "table":
                rt: RawTable = item.ref  # type: ignore[assignment]
                tables.append(
                    Table(
                        pageNumber=rt.page_number,
                        readingOrderPosition=position,
                        cells=_build_cells(rt),
                        degraded=rt.degraded,
                        rawText=rt.raw_text,
                    )
                )
            elif item.kind == "ocr":
                ocr_text: OCRText = item.ref  # type: ignore[assignment]
                blocks.append(
                    build_ocr_text_block(
                        ocr_text,
                        page_number=item.page_number,
                        reading_order_position=position,
                        threshold=threshold,
                    )
                )
            else:  # figure
                rf: RawFigure = item.ref  # type: ignore[assignment]
                figures.append(
                    Figure(
                        pageNumber=rf.page_number,
                        readingOrderPosition=position,
                        imageRef=rf.image_ref,
                        caption=rf.caption,
                    )
                )

        return ParsedDocument(
            documentId=source.document_id,
            blocks=blocks,
            tables=tables,
            figures=figures,
            headings=headings,
            ocrErrors=ocr_errors,
        )

    # -- OCR invocation (Req 4.1, 4.2, 4.5, 4.6) -------------------------

    def _run_ocr(
        self,
        raw: RawParseResult,
        *,
        origin_start: int,
    ) -> Tuple[List[_Orderable], List[ImageOCRError]]:
        """Invoke OCR for image-only pages and text-bearing embedded images.

        Returns the OCR-recovered text as orderables (to be folded into the
        document reading order) plus the per-image error indications. OCR is a
        best-effort, resilient sub-activity: a single unreadable/corrupt/
        unsupported image or a per-page timeout records an
        :class:`ImageOCRError` and processing continues for the remaining
        pages/images (Req 4.5, 4.6). When no processor is injected, OCR is
        disabled and nothing is produced.
        """
        if self.ocr is None:
            return [], []

        deadline = Deadline(seconds=float(self.config.ocr_page_timeout_seconds))
        items: List[_Orderable] = []
        errors: List[ImageOCRError] = []
        origin = origin_start

        # Pages with no extractable text layer are OCR'd from the page image
        # (Req 4.1).
        for page in raw.pages:
            if page.has_text_layer or page.page_image_ref is None:
                continue
            result = self.ocr.process_page(
                PageImage(imageRef=page.page_image_ref, pageNumber=page.page_number),
                deadline,
            )
            origin = self._handle_ocr_result(
                result, page.page_number, items, errors, origin
            )

        # Embedded images that carry text content are OCR'd individually
        # (Req 4.2).
        for image in raw.images:
            if not image.has_text:
                continue
            result = self.ocr.process_embedded_image(
                EmbeddedImage(imageRef=image.image_ref, pageNumber=image.page_number),
                deadline,
            )
            origin = self._handle_ocr_result(
                result, image.page_number, items, errors, origin
            )

        return items, errors

    @staticmethod
    def _handle_ocr_result(
        result: object,
        page_number: int,
        items: List[_Orderable],
        errors: List[ImageOCRError],
        origin: int,
    ) -> int:
        """Fold a single OCR result into ``items``/``errors``; return next origin.

        A successful :class:`OCRText` becomes an orderable carrying the recovered
        text (Req 4.1, 4.2). An :class:`OCRError` or :class:`OCRTimeout` records a
        per-image :class:`ImageOCRError` without aborting the parse (Req 4.5,
        4.6).
        """
        if isinstance(result, OCRText):
            items.append(
                _Orderable(
                    kind="ocr",
                    page_number=page_number,
                    origin=origin,
                    ref=result,
                )
            )
            return origin + 1
        if isinstance(result, (OCRError, OCRTimeout)):
            errors.append(
                ImageOCRError(
                    imageRef=result.imageRef,
                    pageNumber=result.pageNumber,
                    kind=result.kind,
                )
            )
        return origin

    @staticmethod
    def _has_extractable_content(
        document: ParsedDocument,
        raw: RawParseResult,
    ) -> bool:
        """True when the document yielded any extractable content (Req 2.8).

        Text blocks count only when they carry non-whitespace text. Tables and
        figures reported by the engine also count as extractable content even
        though their structured extraction is handled by a later task.
        """
        if any(block.text.strip() for block in document.blocks):
            return True
        return bool(raw.tables or raw.figures)

    # -- fail-closed job marking -----------------------------------------

    def _fail(
        self,
        job: Optional[ProcessingJob],
        kind: ParseFailureKind,
        reason: str,
        *,
        engine_id: Optional[str] = None,
    ) -> None:
        """Mark ``job`` failed at the PARSING stage and raise :class:`ParseFailure`.

        No partial Parsed_Document is produced: the caller raises before any
        document is returned (Req 2.5). The failure reason is recorded on the
        PARSING stage state and the job is marked failed (Req 2.5-2.8).
        """
        if job is not None:
            now = datetime.now(timezone.utc)
            previous = job.stageStates.get(Stage.PARSING)
            attempts = (previous.attempts if previous is not None else 0) + 1
            job.stageStates[Stage.PARSING] = StageState(
                stage=Stage.PARSING,
                status=StageStatus.FAILED,
                attempts=attempts,
                lastTransitionAt=now,
                failureReason=reason,
            )
            job.currentStage = Stage.PARSING
            job.failingStage = Stage.PARSING
            job.overallStatus = OverallStatus.FAILED

        raise ParseFailure(kind, reason, engine_id=engine_id)
