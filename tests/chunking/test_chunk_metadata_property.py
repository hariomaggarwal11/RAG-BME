"""Property test for chunk metadata attachment in the Chunker (Req 6.3, 6.7).

Feature: biomedical-rag-pipeline, Property 15: Chunk metadata is always attached

Statement: for any produced Chunk, the documentId is present; page number and
heading path are attached when available and set to an empty value (None / [])
when unavailable.

The Chunker is driven with the deterministic ``WhitespaceTokenizer``. To map
every produced chunk back to the source element it came from, the document is
chunked with ``max_chunk_tokens=1`` and ``chunk_overlap_tokens=0`` so that each
content token becomes exactly one chunk that inherits the metadata of its
single source element. Generated elements vary their ``pageNumber`` and their
``headingPath`` (present vs. empty) and mix TEXT/HEADING running text with
single-cell tables, so the per-chunk metadata assertion exercises both the
text-windowing and table paths.

Note on "unavailable": within the canonical NormalizedDocument model every
``ContentElement`` carries an integer ``pageNumber`` (so the page is always
available and is therefore always attached to the chunk), while ``headingPath``
may be empty -> the chunk's ``headingPath`` is the empty list, which is exactly
the "empty value when unavailable" branch of Req 6.7. ``documentId`` is always
present and non-empty (Req 6.3).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

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
    """Minimal config carrying only the two bounds the Chunker reads.

    ``max_chunk_tokens=1`` / ``chunk_overlap_tokens=0`` makes every content
    token its own chunk, giving a deterministic chunk -> source-element mapping.
    """

    max_chunk_tokens: int
    chunk_overlap_tokens: int


# Visible, whitespace-free labels so heading-path entries and the document id
# are genuinely non-empty single tokens.
_LABEL = st.text(
    alphabet=st.characters(min_codepoint=33, max_codepoint=126),
    min_size=1,
    max_size=6,
)


@st.composite
def _doc_and_expected(
    draw,
) -> Tuple[NormalizedDocument, List[Tuple[int, Tuple[str, ...]]]]:
    """Build a document plus the expected (page, headingPath) per produced chunk.

    Elements are emitted in reading order (``readingOrderPosition == index``) so
    the generation order equals the chunk ``orderIndex`` order. Each text/heading
    element contributes one chunk per whitespace token; each single-cell table
    contributes exactly one (table-part) chunk. Page numbers and heading paths
    are varied independently, including empty heading paths.
    """
    n = draw(st.integers(min_value=1, max_value=6))
    elements: List[ContentElement] = []
    expected: List[Tuple[int, Tuple[str, ...]]] = []
    counter = 0

    for index in range(n):
        page = draw(st.integers(min_value=0, max_value=12))
        path = tuple(draw(st.lists(_LABEL, min_size=0, max_size=3)))
        kind_choice = draw(st.sampled_from(["text", "heading", "table"]))

        if kind_choice == "table":
            token = f"t{counter}"
            counter += 1
            elements.append(
                ContentElement(
                    kind=ElementKind.TABLE,
                    pageNumber=page,
                    readingOrderPosition=index,
                    payload=TablePayload(
                        cells=[Cell(rowIndex=0, colIndex=0, value=token)]
                    ),
                    headingPath=list(path),
                )
            )
            expected.append((page, path))
        else:
            num_tokens = draw(st.integers(min_value=1, max_value=4))
            tokens = []
            for _ in range(num_tokens):
                tokens.append(f"t{counter}")
                counter += 1
            kind = (
                ElementKind.TEXT
                if kind_choice == "text"
                else ElementKind.HEADING
            )
            elements.append(
                ContentElement(
                    kind=kind,
                    pageNumber=page,
                    readingOrderPosition=index,
                    payload=TextPayload(text=" ".join(tokens)),
                    headingPath=list(path),
                )
            )
            # One chunk per token (max_chunk_tokens=1), all sharing the element.
            expected.extend((page, path) for _ in tokens)

    doc = NormalizedDocument(documentId=draw(_LABEL), elements=elements)
    return doc, expected


# Feature: biomedical-rag-pipeline, Property 15: Chunk metadata is always attached
@settings(max_examples=200)
@given(data=_doc_and_expected())
def test_chunk_metadata_is_always_attached(
    data: Tuple[NormalizedDocument, List[Tuple[int, Tuple[str, ...]]]],
) -> None:
    """Validates: Requirements 6.3, 6.7"""
    doc, expected = data
    chunks = Chunker(WhitespaceTokenizer()).chunk(
        doc, _ChunkConfig(max_chunk_tokens=1, chunk_overlap_tokens=0)
    )

    # Each content token produced exactly one chunk.
    assert len(chunks) == len(expected)

    ordered = sorted(chunks, key=lambda c: c.orderIndex)
    for chunk, (page, path) in zip(ordered, expected):
        # documentId is always present, non-empty, and equal to the doc's id
        # (Req 6.3).
        assert chunk.documentId == doc.documentId
        assert chunk.documentId  # non-empty

        # Page number is available for every ContentElement, so it is attached
        # and matches the source element (Req 6.3).
        assert chunk.pageNumber == page

        # Heading path matches the source element: the present path when the
        # element had one, and the empty list when it did not (Req 6.7).
        assert chunk.headingPath == list(path)
