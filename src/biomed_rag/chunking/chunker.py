"""Chunker component (Req 6).

The :class:`Chunker` turns a canonical :class:`NormalizedDocument` into an
ordered list of :class:`Chunk`. This module implements task 8.1:

* token-bounded chunks ``<= maxChunkTokens`` using the injected tokenizer
  (Req 6.1),
* a configured token overlap between consecutive chunks, recorded in each
  chunk's ``overlapTokenCount`` (Req 6.2),
* ``{documentId, pageNumber, headingPath}`` metadata on every chunk, with empty
  values when page/heading are unavailable but ``documentId`` always present
  (Req 6.3, 6.7),
* a contiguous zero-based ``orderIndex`` over the produced chunks, and
* zero chunks for artifact-only (no non-artifact text) input (Req 6.8).

Token counting uses a pluggable :class:`~biomed_rag.chunking.tokenizer.Tokenizer`
that is injected for deterministic testing (design: Chunker section).

Table-aware chunking (task 8.2) keeps a table whose serialized content fits
within ``maxChunkTokens`` inside a single chunk (Req 6.4) and splits an
oversized table across consecutive chunks, each within the bound and marked
``isTablePart`` (Req 6.6). Tables are treated as their own segments so a fitting
table is never split by the surrounding text window; the running-text windowing
(and its overlap/completeness semantics) is preserved for the text segments
between tables.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from biomed_rag.models.enums import ElementKind
from biomed_rag.models.normalized import (
    ContentElement,
    NormalizedDocument,
    TablePayload,
)

from ..models.chunk import Chunk
from .tokenizer import Tokenizer, WhitespaceTokenizer


class ChunkingError(ValueError):
    """Raised when chunking cannot proceed (e.g. inconsistent bounds)."""


# A single token paired with the source metadata of the element it came from.
# pageNumber is Optional to honor Req 6.7 (empty when unavailable); headingPath
# is the (possibly empty) heading hierarchy of the source element.
_TokenRecord = Tuple[str, Optional[int], Tuple[str, ...]]

# A segment is a maximal run of content that is chunked as a unit. Running text
# (and headings) accumulate into TEXT segments that are windowed with overlap;
# each table becomes its own TABLE segment so it is kept whole when it fits and
# split independently when it does not (Req 6.4, 6.6).
_TextSegment = Tuple[str, List[_TokenRecord]]
_TableSegment = Tuple[str, ContentElement, str]
_Segment = object


class Chunker:
    """Produces token-bounded, overlapping chunks with source metadata (Req 6).

    Parameters
    ----------
    tokenizer:
        Injected tokenizer used for all token counting and content
        reconstruction (design: Chunker). Defaults to a deterministic
        :class:`WhitespaceTokenizer`.
    """

    def __init__(self, tokenizer: Optional[Tokenizer] = None) -> None:
        self.tokenizer: Tokenizer = tokenizer or WhitespaceTokenizer()

    # -- public API -------------------------------------------------------
    def chunk(self, doc: NormalizedDocument, config) -> List[Chunk]:
        """Split ``doc`` into an ordered list of :class:`Chunk` (Req 6.1-6.3, 6.7, 6.8).

        ``config`` supplies ``max_chunk_tokens`` and ``chunk_overlap_tokens``
        (a :class:`~biomed_rag.config.PipelineConfig`). The configuration model
        already enforces ``0 <= chunk_overlap_tokens < max_chunk_tokens``; this
        method re-checks defensively so a hand-built config cannot violate the
        chunk invariants.
        """
        if not isinstance(doc, NormalizedDocument):
            raise ChunkingError("doc must be a NormalizedDocument")

        max_tokens = int(config.max_chunk_tokens)
        overlap = int(config.chunk_overlap_tokens)
        if max_tokens < 1:
            raise ChunkingError(
                f"max_chunk_tokens must be >= 1, got {max_tokens}"
            )
        if not (0 <= overlap < max_tokens):
            raise ChunkingError(
                "chunk_overlap_tokens must be in [0, max_chunk_tokens - 1]; "
                f"got overlap={overlap}, max_chunk_tokens={max_tokens}"
            )

        segments = self._build_segments(doc)

        # Artifact-only input (no non-artifact text/table content) produces zero
        # chunks (Req 6.8).
        if not segments:
            return []

        chunks: List[Chunk] = []
        for segment in segments:
            if segment[0] == "text":
                _, records = segment
                chunks.extend(
                    self._window(doc.documentId, records, max_tokens, overlap)
                )
            else:  # "table"
                _, element, table_text = segment
                chunks.extend(
                    self._table_chunks(
                        doc.documentId, element, table_text, max_tokens
                    )
                )

        # orderIndex is contiguous and zero-based across the whole document,
        # spanning text and table segments alike. Each segment built its chunks
        # with local indices; renumber them globally here.
        for index, chunk in enumerate(chunks):
            chunk.orderIndex = index

        return chunks

    # -- segmentation -----------------------------------------------------
    def _build_segments(self, doc: NormalizedDocument) -> List[_Segment]:
        """Split the document into chunkable segments in reading order.

        Consecutive text/heading elements accumulate into a single TEXT segment
        (so the running-text window and its overlap semantics are unchanged from
        task 8.1). A TABLE element with serialized content flushes the current
        text run and becomes its own TABLE segment, ensuring a fitting table is
        never split by the surrounding text window (Req 6.4, 6.6). FIGURE
        elements and content-free tables contribute no tokens and do not break a
        text run.
        """
        segments: List[_Segment] = []
        current_text: List[_TokenRecord] = []

        def flush_text() -> None:
            if current_text:
                segments.append(("text", list(current_text)))
                current_text.clear()

        ordered = sorted(doc.elements, key=lambda e: e.readingOrderPosition)
        for element in ordered:
            if element.kind == ElementKind.TABLE:
                table_text = self._serialize_table(element.payload)
                if table_text and table_text.strip():
                    flush_text()
                    segments.append(("table", element, table_text))
                continue

            text = self._text_of(element)
            if not text:
                continue
            page = self._page_of(element)
            path = tuple(element.headingPath)
            for token in self.tokenizer.tokenize(text):
                current_text.append((token, page, path))

        flush_text()
        return segments

    def _text_of(self, element: ContentElement) -> Optional[str]:
        """Return the running-text of ``element``, or ``None`` if it has none.

        Only TEXT and HEADING elements are flattened into the running-text
        stream. TABLE elements are handled as their own segments (see
        :meth:`_build_segments`); FIGURE elements carry no chunkable text.
        """
        if element.kind in (ElementKind.TEXT, ElementKind.HEADING):
            return element.payload.text
        return None

    @staticmethod
    def _serialize_table(payload: TablePayload) -> str:
        """Serialize a table payload to a deterministic text representation.

        A structured table is rendered row-by-row in (row, column) order with
        cells separated by `` | `` and rows by newlines, so every non-empty cell
        value contributes its tokens to the chunked content. A degraded table
        with no structured cells falls back to its retained raw region text
        (Req 3.6). Returns an empty string when the table carries no content.
        """
        if payload.cells:
            rows: dict = {}
            for cell in sorted(
                payload.cells, key=lambda c: (c.rowIndex, c.colIndex)
            ):
                rows.setdefault(cell.rowIndex, []).append(cell.value)
            lines = [" | ".join(rows[r]) for r in sorted(rows)]
            return "\n".join(lines)
        if payload.rawText is not None:
            return payload.rawText
        return ""

    @staticmethod
    def _page_of(element: ContentElement) -> Optional[int]:
        """Return the element's page number, or ``None`` when unavailable (Req 6.7)."""
        page = getattr(element, "pageNumber", None)
        return page if isinstance(page, int) else None

    # -- windowing --------------------------------------------------------
    def _window(
        self,
        document_id: str,
        records: Sequence[_TokenRecord],
        max_tokens: int,
        overlap: int,
    ) -> List[Chunk]:
        """Slide a ``max_tokens`` window with ``overlap`` carryover over ``records``.

        Each window becomes one :class:`Chunk`. ``overlapTokenCount`` is the
        number of tokens shared with the previous chunk (Req 6.2, 6.5), and
        ``orderIndex`` is a contiguous zero-based sequence. The window inherits
        the metadata of its first token (Req 6.3, 6.7).
        """
        tokens = [r[0] for r in records]
        total = len(tokens)
        step = max_tokens - overlap  # >= 1 because overlap < max_tokens

        chunks: List[Chunk] = []
        start = 0
        order_index = 0
        prev_end: Optional[int] = None

        while True:
            end = min(start + max_tokens, total)
            window = tokens[start:end]

            # Tokens shared with the previous chunk (0 for the first chunk).
            overlap_count = 0 if prev_end is None else (prev_end - start)

            _, page, path = records[start]
            chunks.append(
                Chunk(
                    documentId=document_id,
                    content=self.tokenizer.detokenize(window),
                    tokenCount=len(window),
                    orderIndex=order_index,
                    overlapTokenCount=overlap_count,
                    pageNumber=page,
                    headingPath=list(path),
                    isTablePart=False,
                )
            )

            order_index += 1
            if end >= total:
                break
            prev_end = end
            start += step

        return chunks

    # -- table chunking ---------------------------------------------------
    def _table_chunks(
        self,
        document_id: str,
        element: ContentElement,
        table_text: str,
        max_tokens: int,
    ) -> List[Chunk]:
        """Chunk a single table's serialized text (Req 6.4, 6.6).

        When the table fits within ``max_tokens`` it is emitted as a single
        chunk (Req 6.4). When it is oversized it is split into consecutive,
        disjoint parts each within the bound (Req 6.6). Every chunk derived from
        a table is marked ``isTablePart`` and carries no overlap, so a table's
        content is partitioned exactly once across its chunks (preserving the
        completeness semantics). The ``orderIndex`` here is local and is
        renumbered globally by :meth:`chunk`.
        """
        tokens = self.tokenizer.tokenize(table_text)
        page = self._page_of(element)
        path = tuple(element.headingPath)

        if len(tokens) <= max_tokens:
            windows = [tokens]
        else:
            windows = [
                tokens[i : i + max_tokens]
                for i in range(0, len(tokens), max_tokens)
            ]

        chunks: List[Chunk] = []
        for order_index, window in enumerate(windows):
            chunks.append(
                Chunk(
                    documentId=document_id,
                    content=self.tokenizer.detokenize(window),
                    tokenCount=len(window),
                    orderIndex=order_index,
                    overlapTokenCount=0,
                    pageNumber=page,
                    headingPath=list(path),
                    isTablePart=True,
                )
            )
        return chunks
