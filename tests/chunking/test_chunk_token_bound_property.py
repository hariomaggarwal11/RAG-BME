"""Property test for the chunk token bound (Task 8.3, Req 6.1, 6.6).

Feature: biomedical-rag-pipeline, Property 13: Chunk token bound is never exceeded

Statement: for any NormalizedDocument and any valid chunking config, every Chunk
produced by the Chunker -- including chunks produced by splitting an oversized
table -- has ``tokenCount <= max_chunk_tokens``, where the token count is
measured with the same injected tokenizer the Chunker uses.

The Chunker is driven with the deterministic WhitespaceTokenizer so token counts
are exact and reproducible. Generated documents interleave text/heading elements
and tables (including oversized tables whose serialized content exceeds the
window), and the config varies ``max_chunk_tokens`` over deliberately small
windows (via a ``_ChunkConfig`` stub, since PipelineConfig clamps
``max_chunk_tokens`` to [128, 2048]) with ``chunk_overlap_tokens`` constrained to
the valid range ``[0, max_chunk_tokens - 1]``.
"""

from __future__ import annotations

import string
from dataclasses import dataclass
from typing import List

from hypothesis import given, settings
from hypothesis import strategies as st

from biomed_rag.chunking import Chunker, WhitespaceTokenizer
from biomed_rag.models import (
    Cell,
    ContentElement,
    NormalizedDocument,
    TablePayload,
    TextPayload,
)
from biomed_rag.models.enums import ElementKind


@dataclass
class _ChunkConfig:
    """Minimal config carrying only the chunking bounds the Chunker reads.

    PipelineConfig clamps ``max_chunk_tokens`` to [128, 2048]; this stub lets the
    property exercise small windows so oversized text runs and tables actually
    split. The Chunker depends only on these two attributes.
    """

    max_chunk_tokens: int
    chunk_overlap_tokens: int


# Tokens are non-whitespace words so the WhitespaceTokenizer counts them exactly.
_WORD = st.text(alphabet=string.ascii_lowercase + string.digits, min_size=1, max_size=5)


@st.composite
def _text_element(draw, position):
    """A TEXT or HEADING element whose payload text is whitespace-joined words.

    Word lists may be empty (an artifact-only / no-token element), exercising the
    Chunker's "no chunkable text" path alongside multi-window runs.
    """
    words = draw(st.lists(_WORD, min_size=0, max_size=12))
    kind = draw(st.sampled_from([ElementKind.TEXT, ElementKind.HEADING]))
    page = draw(st.integers(min_value=0, max_value=25))
    heading_path = draw(st.lists(_WORD, max_size=3))
    return ContentElement(
        kind=kind,
        pageNumber=page,
        readingOrderPosition=position,
        payload=TextPayload(text=" ".join(words)),
        headingPath=heading_path,
    )


@st.composite
def _table_element(draw, position):
    """A TABLE element. Grids range from empty up to oversized relative to the
    small windows under test, so both the single-chunk and split paths run."""
    n_rows = draw(st.integers(min_value=0, max_value=6))
    n_cols = draw(st.integers(min_value=0, max_value=6))
    cells: List[Cell] = []
    for r in range(n_rows):
        for c in range(n_cols):
            value = draw(st.text(alphabet=string.ascii_lowercase, min_size=0, max_size=4))
            cells.append(Cell(rowIndex=r, colIndex=c, value=value))
    page = draw(st.integers(min_value=0, max_value=25))
    heading_path = draw(st.lists(_WORD, max_size=3))
    # Occasionally a degraded table that falls back to retained raw text.
    use_raw = draw(st.booleans())
    raw_text = draw(st.text(alphabet=string.ascii_lowercase + " ", max_size=40)) if use_raw else None
    return ContentElement(
        kind=ElementKind.TABLE,
        pageNumber=page,
        readingOrderPosition=position,
        payload=TablePayload(cells=cells, degraded=use_raw and not cells, rawText=raw_text),
        headingPath=heading_path,
    )


@st.composite
def _documents(draw):
    """A NormalizedDocument mixing text/heading and table elements in order."""
    n = draw(st.integers(min_value=0, max_value=8))
    elements: List[ContentElement] = []
    for position in range(n):
        is_table = draw(st.booleans())
        if is_table:
            elements.append(draw(_table_element(position)))
        else:
            elements.append(draw(_text_element(position)))
    return NormalizedDocument(documentId="doc-prop-13", elements=elements)


@st.composite
def _configs(draw):
    """A valid chunking config over small windows: ``0 <= overlap < max``."""
    max_tokens = draw(st.integers(min_value=1, max_value=20))
    overlap = draw(st.integers(min_value=0, max_value=max_tokens - 1))
    return _ChunkConfig(max_chunk_tokens=max_tokens, chunk_overlap_tokens=overlap)


# Feature: biomedical-rag-pipeline, Property 13: Chunk token bound is never exceeded
# Validates: Requirements 6.1, 6.6
@settings(max_examples=200)
@given(doc=_documents(), config=_configs())
def test_chunk_token_bound_is_never_exceeded(doc, config):
    tokenizer = WhitespaceTokenizer()
    chunks = Chunker(tokenizer).chunk(doc, config)

    for chunk in chunks:
        # The recorded count never exceeds the configured maximum (Req 6.1, 6.6),
        assert chunk.tokenCount <= config.max_chunk_tokens
        # and it agrees with re-measuring the content via the same tokenizer.
        assert tokenizer.count(chunk.content) <= config.max_chunk_tokens
        assert chunk.tokenCount == tokenizer.count(chunk.content)
