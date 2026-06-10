"""Property test for chunk completeness (task 8.7, Req 6.5, 6.8).

Feature: biomedical-rag-pipeline, Property 17: Chunk completeness

Statement (design Property 17): for any ``NormalizedDocument``, concatenating the
contents of the produced ``Chunk`` objects in ``orderIndex`` order, dropping the
leading ``overlapTokenCount`` tokens of each chunk after the first, yields a token
stream that covers all non-artifact text of the normalized representation; and a
document containing no non-artifact text yields zero chunks.

The Chunker is driven through the deterministic ``WhitespaceTokenizer`` so token
counts are exact and the ``tokenize`` / ``detokenize`` round-trip is lossless,
making "overlap removed" an unambiguous, reproducible operation.

Reconstruction semantics (verified against ``Chunker`` source):

* Consecutive TEXT/HEADING elements accumulate into one text segment that is
  windowed with overlap; the first chunk of every segment records
  ``overlapTokenCount == 0`` and each later chunk records the tokens it shares
  with its predecessor, so dropping those shared tokens reproduces the segment's
  token stream exactly.
* Each TABLE with non-empty serialized content becomes its own segment, split
  into disjoint parts that all record ``overlapTokenCount == 0`` and therefore
  partition the table's serialized tokens exactly.
* FIGURE elements and content-free text/tables contribute no tokens.

Because overlap is only non-zero between consecutive chunks of a single text
segment, the global overlap-removed concatenation equals the in-order
concatenation of every segment's tokens, i.e. the document's non-artifact token
stream built in reading order. The expected stream is built independently here
(text/heading element tokens in reading order, plus the Chunker's own table
serialization) and asserted equal to the reconstruction.

The generators include oversized tables (small ``max_chunk_tokens`` against
multi-cell tables forces table splitting), running text longer than one window
(exercising overlap windowing), and elements with empty/missing heading metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from hypothesis import given, settings
from hypothesis import strategies as st

from biomed_rag.chunking import Chunker, WhitespaceTokenizer
from biomed_rag.chunking.chunker import Chunker as _ChunkerImpl
from biomed_rag.models import (
    Cell,
    ContentElement,
    FigurePayload,
    NormalizedDocument,
    TablePayload,
    TextPayload,
)
from biomed_rag.models.enums import ElementKind


@dataclass
class _ChunkConfig:
    """Minimal stand-in for PipelineConfig carrying only the chunking bounds.

    Small bounds are used so windowing/overlap and oversized-table splitting are
    exercised directly; the Chunker only reads these two attributes.
    """

    max_chunk_tokens: int
    chunk_overlap_tokens: int


# Visible ASCII (no whitespace) so every token round-trips exactly through the
# WhitespaceTokenizer and is unambiguously identifiable in chunk content.
_VISIBLE = st.text(
    alphabet=st.characters(min_codepoint=33, max_codepoint=126),
    min_size=1,
    max_size=6,
)

# A heading path that is frequently empty (missing heading metadata) and at most
# two levels deep otherwise.
_HEADING_PATH = st.lists(_VISIBLE, min_size=0, max_size=2)


def _words(draw, min_words: int, max_words: int) -> str:
    """Draw a space-joined run of visible-token words (possibly empty)."""
    n = draw(st.integers(min_value=min_words, max_value=max_words))
    return " ".join(draw(_VISIBLE) for _ in range(n))


@st.composite
def _text_element(draw, position: int) -> ContentElement:
    kind = draw(st.sampled_from([ElementKind.TEXT, ElementKind.HEADING]))
    # 0 words yields empty (artifact) text; longer runs exercise multi-window
    # overlap reconstruction under small bounds.
    text = _words(draw, 0, 8)
    return ContentElement(
        kind=kind,
        pageNumber=draw(st.integers(min_value=0, max_value=20)),
        readingOrderPosition=position,
        payload=TextPayload(text=text),
        headingPath=draw(_HEADING_PATH),
    )


@st.composite
def _table_element(draw, position: int) -> ContentElement:
    # Dense grid of non-empty visible cells. Up to 5x5 cells against the small
    # configured bound forces oversized tables to be split (Req 6.6) while still
    # exercising the completeness reconstruction across table parts.
    rows = draw(st.integers(min_value=1, max_value=5))
    cols = draw(st.integers(min_value=1, max_value=5))
    cells: List[Cell] = []
    for r in range(rows):
        for c in range(cols):
            cells.append(Cell(rowIndex=r, colIndex=c, value=draw(_VISIBLE)))
    return ContentElement(
        kind=ElementKind.TABLE,
        pageNumber=draw(st.integers(min_value=0, max_value=20)),
        readingOrderPosition=position,
        payload=TablePayload(cells=cells),
        headingPath=draw(_HEADING_PATH),
    )


@st.composite
def _figure_element(draw, position: int) -> ContentElement:
    # Figures carry no chunkable text -> pure artifact w.r.t. completeness.
    return ContentElement(
        kind=ElementKind.FIGURE,
        pageNumber=draw(st.integers(min_value=0, max_value=20)),
        readingOrderPosition=position,
        payload=FigurePayload(
            imageRef=draw(_VISIBLE),
            caption=draw(st.one_of(st.none(), _VISIBLE)),
        ),
        headingPath=draw(_HEADING_PATH),
    )


@st.composite
def _content_case(draw):
    """Generate a (document, config) pair mixing text, headings, tables, figures.

    ``max_chunk_tokens`` is intentionally small so text runs span multiple
    windows and multi-cell tables become oversized and are split.
    """
    max_tokens = draw(st.integers(min_value=2, max_value=16))
    overlap = draw(st.integers(min_value=0, max_value=max_tokens - 1))

    n = draw(st.integers(min_value=0, max_value=6))
    elements: List[ContentElement] = []
    for position in range(n):
        kind = draw(st.sampled_from(["text", "table", "figure"]))
        if kind == "text":
            elements.append(draw(_text_element(position)))
        elif kind == "table":
            elements.append(draw(_table_element(position)))
        else:
            elements.append(draw(_figure_element(position)))

    doc = NormalizedDocument(documentId="doc-1", elements=elements)
    return doc, _ChunkConfig(max_tokens, overlap)


@st.composite
def _artifact_only_case(draw):
    """Generate a document containing no non-artifact text (Req 6.8).

    Elements are limited to figures, empty/whitespace-only text and headings, and
    content-free tables (no cells; absent or whitespace-only raw text). None of
    these contribute chunkable tokens, so the Chunker must produce zero chunks.
    """
    n = draw(st.integers(min_value=0, max_value=5))
    elements: List[ContentElement] = []
    for position in range(n):
        kind = draw(st.sampled_from(["empty_text", "figure", "empty_table"]))
        page = draw(st.integers(min_value=0, max_value=20))
        if kind == "empty_text":
            blank = draw(st.sampled_from(["", " ", "   ", "\t", "\n"]))
            elements.append(
                ContentElement(
                    kind=draw(st.sampled_from([ElementKind.TEXT, ElementKind.HEADING])),
                    pageNumber=page,
                    readingOrderPosition=position,
                    payload=TextPayload(text=blank),
                    headingPath=draw(_HEADING_PATH),
                )
            )
        elif kind == "figure":
            elements.append(
                ContentElement(
                    kind=ElementKind.FIGURE,
                    pageNumber=page,
                    readingOrderPosition=position,
                    payload=FigurePayload(imageRef=draw(_VISIBLE)),
                    headingPath=draw(_HEADING_PATH),
                )
            )
        else:  # empty_table: no cells, raw text absent or whitespace-only
            raw = draw(st.sampled_from([None, "", "   ", "\n"]))
            elements.append(
                ContentElement(
                    kind=ElementKind.TABLE,
                    pageNumber=page,
                    readingOrderPosition=position,
                    payload=TablePayload(cells=[], rawText=raw),
                    headingPath=draw(_HEADING_PATH),
                )
            )

    return NormalizedDocument(documentId="doc-1", elements=elements)


def _reconstruct(chunks, tokenizer) -> List[str]:
    """Concatenate chunk contents in ``orderIndex`` order, dropping the leading
    ``overlapTokenCount`` tokens of every chunk after the first (Req 6.5)."""
    out: List[str] = []
    for i, chunk in enumerate(sorted(chunks, key=lambda c: c.orderIndex)):
        toks = tokenizer.tokenize(chunk.content)
        if i > 0:
            toks = toks[chunk.overlapTokenCount:]
        out.extend(toks)
    return out


def _expected_tokens(doc, tokenizer) -> List[str]:
    """Build the non-artifact token stream the Chunker should cover, in reading
    order: TEXT/HEADING element tokens plus the Chunker's own table serialization.
    """
    tokens: List[str] = []
    for element in sorted(doc.elements, key=lambda e: e.readingOrderPosition):
        if element.kind == ElementKind.TABLE:
            serialized = _ChunkerImpl._serialize_table(element.payload)
            if serialized and serialized.strip():
                tokens.extend(tokenizer.tokenize(serialized))
        elif element.kind in (ElementKind.TEXT, ElementKind.HEADING):
            text = element.payload.text
            if text:
                tokens.extend(tokenizer.tokenize(text))
        # FIGURE contributes no chunkable text.
    return tokens


# Feature: biomedical-rag-pipeline, Property 17: Chunk completeness
@settings(max_examples=200)
@given(case=_content_case())
def test_chunk_reconstruction_covers_all_non_artifact_text(case) -> None:
    """Validates: Requirements 6.5"""
    doc, config = case
    tokenizer = WhitespaceTokenizer()

    chunks = Chunker(tokenizer).chunk(doc, config)

    expected = _expected_tokens(doc, tokenizer)
    reconstruction = _reconstruct(chunks, tokenizer)

    # The overlap-removed, in-order concatenation reproduces the document's
    # non-artifact token stream exactly: every non-artifact token is covered,
    # in reading order, with no loss and no duplication (Req 6.5).
    assert reconstruction == expected

    # Coverage restated explicitly: every expected non-artifact token is present
    # in the reconstruction.
    for token in expected:
        assert token in reconstruction

    # When (and only when) there is non-artifact text, chunks are produced.
    assert (len(chunks) == 0) == (len(expected) == 0)


# Feature: biomedical-rag-pipeline, Property 17: Chunk completeness
@settings(max_examples=200)
@given(doc=_artifact_only_case())
def test_artifact_only_document_yields_zero_chunks(doc) -> None:
    """Validates: Requirements 6.8"""
    config = _ChunkConfig(max_chunk_tokens=128, chunk_overlap_tokens=16)

    chunks = Chunker(WhitespaceTokenizer()).chunk(doc, config)

    # No non-artifact text -> zero chunks (Req 6.8).
    assert chunks == []
