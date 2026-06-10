"""Property test for the Chunker's configured overlap (task 8.4, Req 6.2).

Feature: biomedical-rag-pipeline, Property 14: Configured overlap is honored

Statement: for any valid configuration with overlap o in [0, maxChunkTokens - 1],
each pair of consecutive Chunks shares exactly the configured overlap (bounded by
the smaller chunk), recorded in each chunk's overlapTokenCount.

This test focuses on running-text documents (a single TEXT element of contiguous
text) where the overlap windowing semantics apply. It varies the token count,
``max_chunk_tokens``, and the configured ``chunk_overlap_tokens`` and asserts, for
each consecutive pair of chunks, that:

* the later chunk's recorded ``overlapTokenCount`` equals the configured overlap,
  bounded by the smaller of the two chunks (so a final short chunk cannot record
  an overlap larger than itself), and
* the recorded overlap tokens (the leading ``overlapTokenCount`` tokens of the
  later chunk) are exactly the trailing tokens of the previous chunk.

The deterministic WhitespaceTokenizer with distinct tokens (``w0 w1 ...``) makes
the shared-token comparison unambiguous.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from hypothesis import given, settings
from hypothesis import strategies as st

from biomed_rag.chunking import Chunker, WhitespaceTokenizer
from biomed_rag.models import ContentElement, NormalizedDocument, TextPayload
from biomed_rag.models.enums import ElementKind


@dataclass
class _ChunkConfig:
    """Minimal config carrying only the two chunking bounds the Chunker reads.

    Using a small-bounds stub (instead of a full PipelineConfig, which clamps
    ``max_chunk_tokens`` to [128, 2048]) lets the generator exercise the
    windowing/overlap logic across many small, fast configurations while still
    satisfying the Chunker's invariant ``0 <= overlap < max_chunk_tokens``.
    """

    max_chunk_tokens: int
    chunk_overlap_tokens: int


def _text_element(text: str) -> ContentElement:
    return ContentElement(
        kind=ElementKind.TEXT,
        pageNumber=1,
        readingOrderPosition=0,
        payload=TextPayload(text=text),
        headingPath=[],
    )


def _doc(text: str, document_id: str = "doc-overlap") -> NormalizedDocument:
    return NormalizedDocument(documentId=document_id, elements=[_text_element(text)])


@st.composite
def _cases(draw) -> Tuple[str, _ChunkConfig]:
    """Generate (running_text, config) with a valid overlap in [0, max - 1].

    ``max_chunk_tokens`` is kept small so multi-chunk windows (and therefore
    consecutive pairs) arise frequently; the token count ranges from a single
    token up to several windows' worth so both the single-chunk (vacuous) and
    multi-chunk cases are covered.
    """
    max_tokens = draw(st.integers(min_value=2, max_value=40))
    overlap = draw(st.integers(min_value=0, max_value=max_tokens - 1))
    token_count = draw(st.integers(min_value=1, max_value=200))
    text = " ".join(f"w{i}" for i in range(token_count))
    return text, _ChunkConfig(max_chunk_tokens=max_tokens, chunk_overlap_tokens=overlap)


# Feature: biomedical-rag-pipeline, Property 14: Configured overlap is honored
@settings(max_examples=200)
@given(case=_cases())
def test_configured_overlap_is_honored(case: Tuple[str, _ChunkConfig]) -> None:
    """Validates: Requirements 6.2"""
    text, config = case
    tokenizer = WhitespaceTokenizer()
    chunks = Chunker(tokenizer).chunk(_doc(text), config)

    ordered = sorted(chunks, key=lambda c: c.orderIndex)
    overlap = config.chunk_overlap_tokens

    # The first chunk has no predecessor, so it carries no overlap.
    if ordered:
        assert ordered[0].overlapTokenCount == 0

    for prev, curr in zip(ordered, ordered[1:]):
        prev_tokens = tokenizer.tokenize(prev.content)
        curr_tokens = tokenizer.tokenize(curr.content)

        # The shared overlap is the configured value, but cannot exceed either
        # of the two chunks it is shared between (bounded by the smaller chunk).
        expected = min(overlap, len(prev_tokens), len(curr_tokens))
        assert curr.overlapTokenCount == expected

        # The recorded overlap tokens are exactly the trailing tokens of the
        # previous chunk == the leading tokens of this chunk.
        shared = curr.overlapTokenCount
        if shared:
            assert curr_tokens[:shared] == prev_tokens[-shared:]
