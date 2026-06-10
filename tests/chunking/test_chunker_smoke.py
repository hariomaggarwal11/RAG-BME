"""Smoke / unit tests for the Chunker token-bounded chunking (task 8.1, Req 6).

These exercise the Chunker against the deterministic WhitespaceTokenizer and
verify the task-8.1 behaviors: the token bound (Req 6.1), recorded overlap and
overlap-based reconstruction (Req 6.2, 6.5), metadata attachment with empty
values (Req 6.3, 6.7), contiguous orderIndex, and zero chunks for artifact-only
input (Req 6.8). The formal property tests live in tasks 8.3-8.7.
"""

from __future__ import annotations

from dataclasses import dataclass

from biomed_rag.chunking import Chunker, WhitespaceTokenizer
from biomed_rag.config import PipelineConfig
from biomed_rag.models import ContentElement, NormalizedDocument, TextPayload
from biomed_rag.models.enums import ElementKind


@dataclass
class _ChunkConfig:
    """Minimal config carrying only the chunking bounds the Chunker reads.

    PipelineConfig clamps ``max_chunk_tokens`` to [128, 2048]; these unit tests
    use small windows to exercise the windowing logic directly. The Chunker only
    depends on the two attributes below, and ``test_with_real_pipeline_config``
    covers a genuine PipelineConfig.
    """

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


def _doc(*elements, document_id="doc-1"):
    return NormalizedDocument(documentId=document_id, elements=list(elements))


def _reconstruct(chunks, tokenizer):
    """Concatenate chunk contents in orderIndex order, dropping the leading
    overlapTokenCount tokens of every chunk after the first (Req 6.5)."""
    out = []
    for i, chunk in enumerate(sorted(chunks, key=lambda c: c.orderIndex)):
        toks = tokenizer.tokenize(chunk.content)
        if i > 0:
            toks = toks[chunk.overlapTokenCount:]
        out.extend(toks)
    return out


def test_token_bound_never_exceeded():
    text = " ".join(f"w{i}" for i in range(50))
    chunks = Chunker().chunk(_doc(_text_element(text)), _config(10, 3))

    assert len(chunks) > 1
    assert all(c.tokenCount <= 10 for c in chunks)


def test_overlap_recorded_and_reconstruction_is_complete():
    tokenizer = WhitespaceTokenizer()
    words = [f"w{i}" for i in range(50)]
    text = " ".join(words)
    chunks = Chunker(tokenizer).chunk(_doc(_text_element(text)), _config(10, 3))

    # First chunk has no carryover; every later chunk records exactly 3 tokens.
    ordered = sorted(chunks, key=lambda c: c.orderIndex)
    assert ordered[0].overlapTokenCount == 0
    assert all(c.overlapTokenCount == 3 for c in ordered[1:])

    # Overlap-removed concatenation reproduces the original token stream (Req 6.5).
    assert _reconstruct(chunks, tokenizer) == words


def test_order_index_is_contiguous_zero_based():
    text = " ".join(f"w{i}" for i in range(30))
    chunks = Chunker().chunk(_doc(_text_element(text)), _config(8, 2))

    indices = sorted(c.orderIndex for c in chunks)
    assert indices == list(range(len(chunks)))


def test_metadata_always_includes_document_id_and_empty_when_unavailable():
    chunks = Chunker().chunk(
        _doc(_text_element("alpha beta gamma", page=4, heading_path=["A", "B"])),
        _config(128, 16),
    )
    chunk = chunks[0]
    assert chunk.documentId == "doc-1"
    assert chunk.pageNumber == 4
    assert chunk.headingPath == ["A", "B"]

    # Heading path unavailable -> empty list, documentId still present (Req 6.7).
    chunks = Chunker().chunk(
        _doc(_text_element("alpha beta gamma", page=2)), _config(128, 16)
    )
    assert chunks[0].headingPath == []
    assert chunks[0].documentId == "doc-1"
    assert chunks[0].pageNumber == 2


def test_zero_overlap_partitions_tokens_exactly():
    tokenizer = WhitespaceTokenizer()
    words = [f"w{i}" for i in range(25)]
    chunks = Chunker(tokenizer).chunk(_doc(_text_element(" ".join(words))), _config(10, 0))

    assert all(c.overlapTokenCount == 0 for c in chunks)
    assert _reconstruct(chunks, tokenizer) == words


def test_artifact_only_input_yields_zero_chunks():
    # No elements at all.
    assert Chunker().chunk(_doc(), _config(128, 16)) == []
    # Elements with only whitespace / empty text contribute no tokens (Req 6.8).
    blank = _text_element("   ", page=1)
    assert Chunker().chunk(_doc(blank), _config(128, 16)) == []


def test_metadata_follows_source_element_in_reading_order():
    e0 = _text_element("aaa bbb", page=1, position=0, heading_path=["Intro"])
    e1 = _text_element("ccc ddd", page=2, position=1, heading_path=["Methods"])
    # Small chunks so each element maps to its own chunk.
    chunks = Chunker().chunk(_doc(e0, e1), _config(2, 0))
    ordered = sorted(chunks, key=lambda c: c.orderIndex)
    assert ordered[0].pageNumber == 1 and ordered[0].headingPath == ["Intro"]
    assert ordered[-1].pageNumber == 2 and ordered[-1].headingPath == ["Methods"]


def test_with_real_pipeline_config():
    # A genuine PipelineConfig (max=128, overlap=64) over text longer than one
    # window exercises the real bounds path end-to-end.
    tokenizer = WhitespaceTokenizer()
    words = [f"w{i}" for i in range(400)]
    config = PipelineConfig(max_chunk_tokens=128, chunk_overlap_tokens=64)
    chunks = Chunker(tokenizer).chunk(_doc(_text_element(" ".join(words))), config)

    assert len(chunks) > 1
    assert all(c.tokenCount <= 128 for c in chunks)
    ordered = sorted(chunks, key=lambda c: c.orderIndex)
    assert all(c.overlapTokenCount == 64 for c in ordered[1:])
    assert _reconstruct(chunks, tokenizer) == words
