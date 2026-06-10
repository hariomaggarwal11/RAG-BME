"""Property test for OCR resilience to bad images (Req 4.1, 4.2, 4.5).

Feature: biomedical-rag-pipeline, Property 8: OCR is resilient to bad images

Statement: for any document whose pages and embedded images include an
arbitrary mix of readable and unreadable/corrupt/unsupported images, every
readable item produces stored recovered text and every unreadable item produces
a recorded error indication, with no item causing the remaining items to be
skipped.

The Parser is driven through the ParsingEngine port using the deterministic
MockParsingEngine with a preset RawParseResult, and OCR is performed by the
deterministic InMemoryOCRProcessor. Each item (image-only page or text-bearing
embedded image) is given a unique ``imageRef`` and a matching PlannedOCR:

* a readable item plans a unique recovered text + an in-range confidence;
* an unreadable item plans an UNREADABLE / CORRUPT / UNSUPPORTED error.

A single guaranteed native text block keeps the document non-empty so the parse
never takes the no-extractable-content fail-closed path (out of scope here).

We assert that:

1. every readable item's unique text is stored in exactly one OCR_TEXT block;
2. every unreadable item is recorded as exactly one ImageOCRError carrying its
   ``imageRef`` and planned error kind;
3. the counts add up — #OCR_TEXT blocks == #readable and #ocrErrors ==
   #unreadable — so no item was skipped because of another bad item.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List

from hypothesis import given, settings
from hypothesis import strategies as st

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
    RawBlock,
    RawImage,
    RawPage,
    RawParseResult,
    SourceDocument,
)

_DOCUMENT_ID = "hash-1"

# The three "bad image" error kinds covered by Req 4.5 (timeouts are Req 4.6 and
# exercised separately). All planned durations are 0, so the page deadline is
# never exceeded and no item turns into a timeout.
_BAD_KINDS = st.sampled_from(
    [OCRErrorKind.UNREADABLE, OCRErrorKind.CORRUPT, OCRErrorKind.UNSUPPORTED]
)

_UNIT = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
_PAGE = st.integers(min_value=0, max_value=4)


@st.composite
def _item(draw) -> dict:
    """One OCR item: an image-only page or a text-bearing embedded image.

    ``readable`` decides whether it plans recovered text or a bad-image error.
    ``imageRef`` and ``text`` are filled in by :func:`_items` so they are unique
    across the whole document.
    """
    return {
        "is_page": draw(st.booleans()),
        "readable": draw(st.booleans()),
        "page_number": draw(_PAGE),
        "confidence": draw(_UNIT),
        "error_kind": draw(_BAD_KINDS),
    }


@st.composite
def _items(draw) -> List[dict]:
    """An arbitrary mix of items, each given a unique imageRef and (if readable)
    a unique recovered text so it can be matched back unambiguously."""
    items = draw(st.lists(_item(), min_size=1, max_size=8))
    for i, item in enumerate(items):
        item["image_ref"] = f"img-{i}"
        item["text"] = f"ocr-text-{i}"
    return items


def _job() -> ProcessingJob:
    metadata = DocumentMetadata(
        filename="paper.pdf",
        format=Format.PDF,
        byteSize=1234,
        contentHash=_DOCUMENT_ID,
        submittedAtUtc=datetime.now(timezone.utc),
    )
    return ProcessingJob(jobId="job-1", documentId=_DOCUMENT_ID, metadata=metadata)


def _parser(engine: MockParsingEngine, ocr: InMemoryOCRProcessor) -> Parser:
    registry = ParsingEngineRegistry()
    registry.register(ParsingEngineChoice.DOCLING, lambda: engine)
    config = PipelineConfig(parsing_engine=ParsingEngineChoice.DOCLING)
    return Parser(config=config, registry=registry, ocr=ocr)


def _build(items: List[dict]) -> tuple[RawParseResult, InMemoryOCRProcessor]:
    """Assemble the preset RawParseResult and the matching OCR plan."""
    # A guaranteed native text block (page has a text layer, so it is not OCR'd)
    # keeps the document non-empty regardless of the OCR outcomes.
    blocks = [RawBlock(text="native-anchor", page_number=0, kind="paragraph")]
    pages: List[RawPage] = [RawPage(page_number=0, has_text_layer=True)]
    images: List[RawImage] = []
    plans: Dict[str, PlannedOCR] = {}

    for item in items:
        ref = item["image_ref"]
        if item["is_page"]:
            pages.append(
                RawPage(
                    page_number=item["page_number"],
                    has_text_layer=False,
                    page_image_ref=ref,
                )
            )
        else:
            images.append(
                RawImage(
                    page_number=item["page_number"],
                    image_ref=ref,
                    has_text=True,
                )
            )

        if item["readable"]:
            plans[ref] = PlannedOCR(text=item["text"], confidence=item["confidence"])
        else:
            plans[ref] = PlannedOCR(error_kind=item["error_kind"])

    raw = RawParseResult(
        engine_id="docling", blocks=blocks, pages=pages, images=images
    )
    return raw, InMemoryOCRProcessor(plans)


# Feature: biomedical-rag-pipeline, Property 8: OCR is resilient to bad images
@settings(max_examples=200, deadline=None)
@given(items=_items())
def test_ocr_is_resilient_to_bad_images(items: List[dict]) -> None:
    """Validates: Requirements 4.1, 4.2, 4.5"""
    raw, ocr = _build(items)
    parser = _parser(
        MockParsingEngine(engine_id="docling", preset_result=raw), ocr
    )

    parsed = parser.parse(
        _job(), SourceDocument(document_id=_DOCUMENT_ID, raw_bytes=b"body")
    )

    # A bad image never fails the whole parse (Req 4.5).
    assert parsed.documentId == _DOCUMENT_ID

    readable = [it for it in items if it["readable"]]
    unreadable = [it for it in items if not it["readable"]]

    # 1. Every readable item's unique text is stored as recovered OCR text
    #    (Req 4.1 for pages, Req 4.2 for embedded images).
    ocr_texts = [b.text for b in parsed.blocks if b.type is BlockType.OCR_TEXT]
    for it in readable:
        assert ocr_texts.count(it["text"]) == 1, (
            f"readable item {it['image_ref']} did not produce exactly one "
            f"OCR_TEXT block"
        )

    # 2. Every unreadable item produces exactly one recorded error indication
    #    carrying its imageRef and planned kind (Req 4.5).
    errors_by_ref: Dict[str, List[OCRErrorKind]] = {}
    for err in parsed.ocrErrors:
        errors_by_ref.setdefault(err.imageRef, []).append(err.kind)
    for it in unreadable:
        kinds = errors_by_ref.get(it["image_ref"], [])
        assert kinds == [it["error_kind"]], (
            f"unreadable item {it['image_ref']} expected one "
            f"{it['error_kind']} error, got {kinds}"
        )

    # 3. The counts add up: nothing was skipped because of another bad item.
    assert len(ocr_texts) == len(readable)
    assert len(parsed.ocrErrors) == len(unreadable)
    assert len(readable) + len(unreadable) == len(items)
