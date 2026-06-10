"""Property test for heading-hierarchy preservation in the Parser (Req 2.4).

Feature: biomedical-rag-pipeline, Property 5: Heading hierarchy is preserved

Statement: for any Source_Document containing section headings, the heading
nesting levels in the Parsed_Document structural metadata are identical to those
of the source heading structure.

The Parser is driven through the ParsingEngine port using the deterministic
MockParsingEngine with a preset RawParseResult: a generated sequence of heading
RawBlocks (with varied nesting levels) interleaved with paragraph blocks. The
parsed document's heading entries must reproduce the source headings' nesting
levels (and order) exactly, ignoring the interleaved paragraphs.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Tuple

from hypothesis import given, settings
from hypothesis import strategies as st

from biomed_rag.config import ParsingEngine as ParsingEngineChoice
from biomed_rag.config import PipelineConfig
from biomed_rag.models import (
    BlockType,
    DocumentMetadata,
    Format,
    ProcessingJob,
)
from biomed_rag.parsing import (
    MockParsingEngine,
    Parser,
    ParsingEngineRegistry,
)
from biomed_rag.parsing.raw_result import (
    RawBlock,
    RawParseResult,
    SourceDocument,
)

# Heading nesting levels must be >= 1 (RawBlock invariant); cap at a realistic
# document depth so generators stay in a sensible space.
_MAX_HEADING_LEVEL = 6


def _job(document_id: str = "hash-1") -> ProcessingJob:
    metadata = DocumentMetadata(
        filename="paper.pdf",
        format=Format.PDF,
        byteSize=1234,
        contentHash=document_id,
        submittedAtUtc=datetime.now(timezone.utc),
    )
    return ProcessingJob(jobId="job-1", documentId=document_id, metadata=metadata)


def _parser(engine: MockParsingEngine) -> Parser:
    registry = ParsingEngineRegistry()
    registry.register(ParsingEngineChoice.DOCLING, lambda: engine)
    config = PipelineConfig(parsing_engine=ParsingEngineChoice.DOCLING)
    return Parser(config=config, registry=registry)


@st.composite
def _documents(draw) -> Tuple[List[RawBlock], List[int]]:
    """Generate (blocks, expected_levels) where blocks interleave headings and
    paragraphs and at least one heading is present.

    Each item is a heading (carrying a varied nesting level) or a paragraph.
    ``expected_levels`` is the in-source-order sequence of heading nesting
    levels — the structure the parsed document must reproduce.

    Blocks are emitted on a single page with no layout geometry, so the Parser's
    reading order follows the emitted order (its deterministic origin tie-break),
    meaning parsed headings preserve the generated order.
    """
    n = draw(st.integers(min_value=1, max_value=12))
    blocks: List[RawBlock] = []
    expected_levels: List[int] = []
    for i in range(n):
        is_heading = draw(st.booleans())
        if is_heading:
            level = draw(st.integers(min_value=1, max_value=_MAX_HEADING_LEVEL))
            blocks.append(
                RawBlock(
                    text=f"Heading {i} L{level}",
                    page_number=0,
                    kind="heading",
                    heading_level=level,
                )
            )
            expected_levels.append(level)
        else:
            blocks.append(
                RawBlock(text=f"paragraph {i}", page_number=0, kind="paragraph")
            )

    # Ensure the document actually contains at least one section heading.
    if not expected_levels:
        level = draw(st.integers(min_value=1, max_value=_MAX_HEADING_LEVEL))
        blocks.append(
            RawBlock(
                text=f"Heading {n} L{level}",
                page_number=0,
                kind="heading",
                heading_level=level,
            )
        )
        expected_levels.append(level)

    return blocks, expected_levels


# Feature: biomedical-rag-pipeline, Property 5: Heading hierarchy is preserved
@settings(max_examples=200)
@given(document=_documents())
def test_heading_hierarchy_is_preserved(
    document: Tuple[List[RawBlock], List[int]],
) -> None:
    """Validates: Requirements 2.4"""
    blocks, expected_levels = document
    preset = RawParseResult(engine_id="docling", blocks=blocks)
    parser = _parser(MockParsingEngine(engine_id="docling", preset_result=preset))

    source = SourceDocument(document_id="hash-1", raw_bytes=b"body")
    parsed = parser.parse(_job(), source)

    # The Heading entries reproduce the source heading nesting levels in order.
    assert [h.level for h in parsed.headings] == expected_levels

    # The HEADING text blocks carry the same nesting levels, in the same order.
    heading_block_levels = [
        b.headingLevel for b in parsed.blocks if b.type is BlockType.HEADING
    ]
    assert heading_block_levels == expected_levels

    # Heading entries appear in ascending reading-order position (no reordering).
    positions = [h.readingOrderPosition for h in parsed.headings]
    assert positions == sorted(positions)
