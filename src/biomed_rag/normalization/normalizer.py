"""Normalizer component (Req 5).

The :class:`Normalizer` cleans formatting artifacts and (in later tasks) converts
a :class:`ParsedDocument` into a canonical :class:`NormalizedDocument`. Per the
design (Normalizer section) the de-hyphenation dictionary is *injected* so its
behavior is deterministic and testable.

This module implements the two artifact-cleaning transformations of task 7.1:

* recurring header/footer (and page-number) removal (Req 5.1), and
* line-break de-hyphenation that respects the injected dictionary while leaving
  intrinsic hyphens unchanged (Req 5.2, 5.3).

Canonical ``NormalizedDocument`` production (task 7.2) is implemented here;
``serialize`` / ``deserialize`` (task 7.3) build on this and provide the durable
persisted form used by the Orchestrator for resume.
"""

from __future__ import annotations

from typing import List, Tuple

from biomed_rag.models.enums import BlockType, ElementKind
from biomed_rag.models.normalized import (
    ContentElement,
    FigurePayload,
    NormalizedDocument,
    TablePayload,
    TextPayload,
)
from biomed_rag.models.parsed import (
    Figure,
    Heading,
    ParsedDocument,
    Table,
    TextBlock,
)

from .artifacts import HeaderFooterArtifactRemover
from .dehyphenation import dehyphenate_blocks
from .dictionary import Dictionary, WordSetDictionary
from .result import Empty, Malformed, NormalizationResult, Normalized
from .serialization import deserialize as _deserialize
from .serialization import serialize as _serialize


class Normalizer:
    """Cleans formatting artifacts from a :class:`ParsedDocument` (Req 5.1-5.3).

    Parameters
    ----------
    dictionary:
        Injected word-membership oracle backing the de-hyphenation decision (Req
        5.2). Defaults to an empty :class:`WordSetDictionary`, which rejoins
        nothing -- callers/tests are expected to supply a deterministic word list.
    min_recurring_pages:
        Page-recurrence threshold for header/footer artifacts (Req 5.1); default 2.
    region_size:
        Header/footer region size, in blocks per page, used to locate candidate
        artifact blocks.
    """

    def __init__(
        self,
        dictionary: Dictionary | None = None,
        *,
        min_recurring_pages: int = 2,
        region_size: int = 1,
    ) -> None:
        self.dictionary: Dictionary = (
            dictionary if dictionary is not None else WordSetDictionary()
        )
        self._artifact_remover = HeaderFooterArtifactRemover(
            min_recurring_pages=min_recurring_pages,
            region_size=region_size,
        )

    # -- task 7.1: artifact cleaning -------------------------------------
    def remove_recurring_artifacts(self, blocks: List[TextBlock]) -> List[TextBlock]:
        """Remove text blocks recurring in the header/footer region (Req 5.1)."""
        return self._artifact_remover.remove(blocks)

    def dehyphenate(self, blocks: List[TextBlock]) -> List[TextBlock]:
        """Rejoin line-break-hyphenated dictionary words; keep intrinsic hyphens
        unchanged (Req 5.2, 5.3)."""
        return dehyphenate_blocks(blocks, self.dictionary)

    def clean_blocks(self, blocks: List[TextBlock]) -> List[TextBlock]:
        """Apply artifact removal then de-hyphenation to a block list.

        This is the text-cleaning core that :meth:`normalize` (task 7.2) builds on:
        recurring header/footer artifacts are removed first (Req 5.1), then the
        surviving blocks are de-hyphenated (Req 5.2, 5.3).
        """
        without_artifacts = self.remove_recurring_artifacts(blocks)
        return self.dehyphenate(without_artifacts)

    def clean_parsed_document(self, parsed: ParsedDocument) -> ParsedDocument:
        """Return a copy of ``parsed`` with its text blocks artifact-cleaned.

        Tables, figures, headings, and OCR-error records are carried through
        unchanged; only the reading-order text blocks are cleaned. Full canonical
        normalization (heading paths, content-element production) is task 7.2.
        """
        return ParsedDocument(
            documentId=parsed.documentId,
            blocks=self.clean_blocks(parsed.blocks),
            tables=list(parsed.tables),
            figures=list(parsed.figures),
            headings=list(parsed.headings),
            ocrErrors=list(parsed.ocrErrors),
        )

    # -- task 7.2: canonical NormalizedDocument production ----------------
    def normalize(self, parsed: ParsedDocument) -> NormalizationResult:
        """Produce the canonical :class:`NormalizedDocument` (Req 5.4, 5.5, 5.7, 5.8).

        Returns a three-arm :data:`NormalizationResult`:

        * :class:`Malformed` when ``parsed`` is not an interpretable
          :class:`ParsedDocument` (Req 5.8). Rejection is total: no document is
          produced and, because this method is pure and mutates nothing, any
          previously produced valid output is left unchanged.
        * :class:`Empty` when the document carries no recognizable content
          elements -- either because it was empty or because every block was a
          recurring header/footer artifact. An empty :class:`NormalizedDocument`
          is produced alongside a "no content" reason (Req 5.7).
        * :class:`Normalized` wrapping a content-preserving
          :class:`NormalizedDocument` whose :class:`ContentElement`s carry the
          source page number and reading-order position (Req 5.4, 5.5).

        Artifact removal and de-hyphenation (Req 5.1-5.3) are applied to the text
        blocks via :meth:`clean_blocks` before the canonical elements are built.
        """
        # Req 5.8: anything that is not a ParsedDocument cannot be interpreted.
        if not isinstance(parsed, ParsedDocument):
            return Malformed(
                error=(
                    "input is not a ParsedDocument: "
                    f"{type(parsed).__name__}"
                )
            )

        document_id = getattr(parsed, "documentId", None)
        if not (isinstance(document_id, str) and document_id):
            return Malformed(error="ParsedDocument.documentId is missing or empty")

        try:
            elements = self._build_elements(parsed)
        except (TypeError, AttributeError, ValueError, KeyError) as exc:
            # The structure could not be interpreted (e.g. a collection held an
            # unexpected element type). Reject as malformed (Req 5.8).
            return Malformed(
                error=f"ParsedDocument structure could not be interpreted: {exc}"
            )

        if not elements:
            # Req 5.7: empty / no-content input -> empty representation + reason.
            empty_doc = NormalizedDocument(documentId=document_id, elements=[])
            return Empty(
                reason="no content available for normalization",
                document=empty_doc,
            )

        document = NormalizedDocument(documentId=document_id, elements=elements)
        return Normalized(document=document)

    # -- element production helpers --------------------------------------
    def _build_elements(self, parsed: ParsedDocument) -> List[ContentElement]:
        """Build the canonical, reading-order list of :class:`ContentElement`.

        Text blocks are artifact-cleaned first (Req 5.1-5.3). Headings, body text,
        OCR-derived text, tables, and figures are then merged into a single stream
        ordered by ``(pageNumber, readingOrderPosition)`` so the heading hierarchy
        can be reconstructed and attached as a ``headingPath`` ancestry to TEXT and
        HEADING elements (Req 5.4). Every element preserves its source page number
        and reading-order position (Req 5.5).

        ``parsed.blocks`` is the authoritative reading-order content stream and
        already contains HEADING-typed blocks; ``parsed.headings`` is the parser's
        mirror index of those headings. To preserve all source content without
        duplication, heading records are emitted only when no heading block already
        occupies the same ``(pageNumber, readingOrderPosition)``.
        """
        cleaned_blocks = self.clean_blocks(list(parsed.blocks))

        # Reading-order slots already filled by HEADING-typed blocks; heading
        # records at the same slot are the parser's mirror and must not be
        # duplicated.
        heading_block_slots = {
            (b.pageNumber, b.readingOrderPosition)
            for b in cleaned_blocks
            if b.type is BlockType.HEADING
        }

        # Tag each source item so the merged stream can be processed uniformly.
        items: List[Tuple[int, int, str, object]] = []
        for block in cleaned_blocks:
            items.append((block.pageNumber, block.readingOrderPosition, "block", block))
        for heading in parsed.headings:
            if (heading.pageNumber, heading.readingOrderPosition) in heading_block_slots:
                continue
            items.append(
                (heading.pageNumber, heading.readingOrderPosition, "heading", heading)
            )
        for table in parsed.tables:
            items.append((table.pageNumber, table.readingOrderPosition, "table", table))
        for figure in parsed.figures:
            items.append(
                (figure.pageNumber, figure.readingOrderPosition, "figure", figure)
            )

        # Stable sort by (page, reading-order); ties keep insertion order, which
        # favors blocks/headings over tables/figures at the same slot.
        items.sort(key=lambda it: (it[0], it[1]))

        elements: List[ContentElement] = []
        heading_stack: List[Tuple[int, str]] = []
        for _, _, kind, src in items:
            if kind == "block":
                elements.append(self._block_element(src, heading_stack))
            elif kind == "heading":
                elements.append(self._heading_record_element(src, heading_stack))
            elif kind == "table":
                elements.append(self._table_element(src))
            elif kind == "figure":
                elements.append(self._figure_element(src))
        return elements

    @staticmethod
    def _push_heading(heading_stack: List[Tuple[int, str]], level: int, text: str) -> List[str]:
        """Update ``heading_stack`` for a heading at ``level`` and return its ancestry.

        Sibling and deeper headings are popped so the returned ancestry contains
        only strictly-shallower ancestors; the heading is then pushed for the
        elements that follow it.
        """
        while heading_stack and heading_stack[-1][0] >= level:
            heading_stack.pop()
        ancestors = [t for _, t in heading_stack]
        heading_stack.append((level, text))
        return ancestors

    def _block_element(
        self, block: TextBlock, heading_stack: List[Tuple[int, str]]
    ) -> ContentElement:
        """Convert a cleaned :class:`TextBlock` into a TEXT or HEADING element.

        HEADING-typed blocks become HEADING elements and update the heading stack;
        every other block type (paragraph, caption, OCR text, list item, footnote)
        becomes a TEXT element carrying the active heading ancestry (Req 5.4, 5.5).
        """
        if block.type is BlockType.HEADING:
            level = block.headingLevel if block.headingLevel is not None else 1
            ancestors = self._push_heading(heading_stack, level, block.text)
            return ContentElement(
                kind=ElementKind.HEADING,
                pageNumber=block.pageNumber,
                readingOrderPosition=block.readingOrderPosition,
                payload=TextPayload(text=block.text, headingLevel=block.headingLevel),
                headingPath=ancestors,
            )
        return ContentElement(
            kind=ElementKind.TEXT,
            pageNumber=block.pageNumber,
            readingOrderPosition=block.readingOrderPosition,
            payload=TextPayload(text=block.text),
            headingPath=[t for _, t in heading_stack],
        )

    def _heading_record_element(
        self, heading: Heading, heading_stack: List[Tuple[int, str]]
    ) -> ContentElement:
        """Convert a :class:`Heading` record into a HEADING element (Req 5.4).

        Only reached for heading records not already represented by a heading
        block at the same reading-order slot.
        """
        ancestors = self._push_heading(heading_stack, heading.level, heading.text)
        return ContentElement(
            kind=ElementKind.HEADING,
            pageNumber=heading.pageNumber,
            readingOrderPosition=heading.readingOrderPosition,
            payload=TextPayload(text=heading.text, headingLevel=heading.level),
            headingPath=ancestors,
        )

    @staticmethod
    def _table_element(table: Table) -> ContentElement:
        """Convert a :class:`Table` into a TABLE element, preserving its cells,
        degraded flag, and raw text (Req 3.1, 3.2, 3.6, 5.4, 5.5)."""
        return ContentElement(
            kind=ElementKind.TABLE,
            pageNumber=table.pageNumber,
            readingOrderPosition=table.readingOrderPosition,
            payload=TablePayload(
                cells=list(table.cells),
                degraded=table.degraded,
                rawText=table.rawText,
            ),
        )

    @staticmethod
    def _figure_element(figure: Figure) -> ContentElement:
        """Convert a :class:`Figure` into a FIGURE element, preserving its image
        reference and optional caption (Req 3.3, 3.4, 5.4, 5.5)."""
        return ContentElement(
            kind=ElementKind.FIGURE,
            pageNumber=figure.pageNumber,
            readingOrderPosition=figure.readingOrderPosition,
            payload=FigurePayload(imageRef=figure.imageRef, caption=figure.caption),
        )

    # -- task 7.3: durable serialization ---------------------------------
    @staticmethod
    def serialize(doc: NormalizedDocument) -> bytes:
        """Serialize a :class:`NormalizedDocument` to its durable byte form (Req 5.6).

        This is the persisted representation the Orchestrator stores between
        stages so a failed job can resume without re-normalizing. Every field
        contributing to structural equivalence is captured; see
        :func:`biomed_rag.normalization.serialization.serialize`.
        """
        return _serialize(doc)

    @staticmethod
    def deserialize(data: bytes) -> NormalizedDocument:
        """Reconstruct a :class:`NormalizedDocument` from its durable byte form.

        Inverse of :meth:`serialize`; together they satisfy the round-trip
        property (Req 5.6). See
        :func:`biomed_rag.normalization.serialization.deserialize`.
        """
        return _deserialize(data)
