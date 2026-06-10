"""Smoke / unit tests for table-aware chunking (task 8.2, Req 6.4, 6.6).

These exercise the Chunker against the deterministic WhitespaceTokenizer for the
two table behaviors: a table whose serialized content fits within the configured
maximum stays in a single chunk (Req 6.4), and an oversized table is split into
consecutive ``isTablePart`` chunks each within the bound (Req 6.6). They also
confirm tables do not disturb the running-text overlap/completeness semantics.
"""

from __future__ import annotations

from dataclasses import dataclass

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
    max_chunk_tokens: int
    chunk_overlap_tokens: int


def _config(max_tokens, overlap):
    return _ChunkConfig(max_chunk_tokens=max_tokens, chunk_overlap_tokens=overlap)


def _text_element(text, *, page=1, position=0, heading_path=None):
    return ContentElement(
        kind=ElementKind.TEXT,
        pageNumber=page,
        readingOrderPosition=position,
        payload=TextPayload(text=text),
        headingPath=list(heading_path or []),
    )


def _table_element(cells, *, page=1, position=0, heading_path=None, raw_text=None,
                   degraded=False):
    return ContentElement(
        kind=ElementKind.TABLE,
        pageNumber=page,
        readingOrderPosition=position,
        payload=TablePayload(cells=cells, degraded=degraded, rawText=raw_text),
        headingPath=list(heading_path or []),
    )


def _grid(rows):
    """Build a list of Cells from a 2D list of string values."""
    cells = []
    for r, row in enumerate(rows):
        for c, value in enumerate(row):
            cells.append(Cell(rowIndex=r, colIndex=c, value=value))
    return cells


def _doc(*elements, document_id="doc-1"):
    return NormalizedDocument(documentId=document_id, elements=list(elements))


def test_fitting_table_stays_in_single_chunk():
    # A small 2x2 table serializes to 4 value tokens (plus " | " separators,
    # which are dropped by the whitespace tokenizer): well within max=128.
    table = _table_element(_grid([["alpha", "beta"], ["gamma", "delta"]]),
                           page=3, heading_path=["Results"])
    chunks = Chunker().chunk(_doc(table), _config(128, 16))

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.isTablePart is True
    assert chunk.pageNumber == 3
    assert chunk.headingPath == ["Results"]
    assert chunk.orderIndex == 0
    assert chunk.overlapTokenCount == 0
    # Every cell value is present in the single chunk (Req 6.4).
    for value in ("alpha", "beta", "gamma", "delta"):
        assert value in chunk.content


def test_oversized_table_split_into_table_parts_within_bound():
    tokenizer = WhitespaceTokenizer()
    # 30 single-token cells in one row -> 30 value tokens, bound = 8.
    values = [f"v{i}" for i in range(30)]
    table = _table_element(_grid([values]), page=5)
    chunks = Chunker(tokenizer).chunk(_doc(table), _config(8, 0))

    assert len(chunks) > 1
    assert all(c.isTablePart for c in chunks)
    assert all(c.tokenCount <= 8 for c in chunks)
    assert all(c.overlapTokenCount == 0 for c in chunks)
    assert all(c.pageNumber == 5 for c in chunks)

    # The split is exact and lossless: every value token appears exactly once,
    # in order (ignoring the `|` cell separators introduced by serialization).
    recovered = []
    for c in sorted(chunks, key=lambda c: c.orderIndex):
        recovered.extend(tokenizer.tokenize(c.content))
    assert [t for t in recovered if t != "|"] == values


def test_degraded_table_uses_raw_text():
    table = _table_element([], raw_text="raw region text here", degraded=True,
                           page=2)
    chunks = Chunker().chunk(_doc(table), _config(128, 16))

    assert len(chunks) == 1
    assert chunks[0].isTablePart is True
    assert chunks[0].content == "raw region text here"


def test_table_between_text_keeps_text_overlap_and_contiguous_order():
    tokenizer = WhitespaceTokenizer()
    before = _text_element("aaa bbb ccc", page=1, position=0)
    table = _table_element(_grid([["t1", "t2"]]), page=1, position=1)
    after = _text_element("ddd eee fff", page=2, position=2)

    chunks = Chunker(tokenizer).chunk(_doc(before, table, after), _config(2, 0))
    ordered = sorted(chunks, key=lambda c: c.orderIndex)

    # orderIndex is contiguous and zero-based across text + table chunks.
    assert [c.orderIndex for c in ordered] == list(range(len(ordered)))

    # The table chunk is marked; text chunks are not.
    assert any(c.isTablePart for c in ordered)
    text_chunks = [c for c in ordered if not c.isTablePart]
    assert text_chunks and all(not c.isTablePart for c in text_chunks)

    # All non-artifact content (text + table cell values) is present.
    all_tokens = []
    for c in ordered:
        all_tokens.extend(tokenizer.tokenize(c.content))
    for token in ("aaa", "bbb", "ccc", "t1", "t2", "ddd", "eee", "fff"):
        assert token in all_tokens


def test_only_empty_table_yields_zero_chunks():
    empty = _table_element([], raw_text=None)
    assert Chunker().chunk(_doc(empty), _config(128, 16)) == []
