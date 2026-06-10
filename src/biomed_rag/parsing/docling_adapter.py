"""Docling adapter for the pluggable Parsing_Engine port (Req 2.2, 2.6).

:class:`DoclingAdapter` implements the :class:`ParsingEngine` port on top of the
`Docling <https://github.com/docling-project/docling>`_ document parser. Like
every adapter it maps its backend's native output into the engine-neutral
:class:`RawParseResult` shape so the Parser and downstream stages never depend
on a concrete engine (design: ports-and-adapters).

Docling is a heavy, optional third-party dependency that is frequently absent
from a given environment. This adapter therefore degrades gracefully:

* It never imports Docling at module load time. Availability is probed lazily
  via :func:`importlib.util.find_spec`, so importing this module is always safe
  (Req 2.6 — the Parser checks ``is_available`` and fails the job closed when an
  engine is unavailable, rather than crashing on a missing import).
* :meth:`parse` raises :class:`EngineUnavailableError` when Docling is not
  installed, which the Parser translates into a fail-closed job outcome.

Testability seam
----------------
The mapping from Docling's output to :class:`RawParseResult` is the part worth
testing without the real library installed. ``parse`` obtains the engine's
*native* result (a dict shaped like Docling's ``DoclingDocument.export_to_dict``)
from an injectable ``backend`` callable and then maps it with
:meth:`map_native`. Tests inject a fake ``backend`` returning a hand-built dict
to exercise the mapping deterministically; production uses the default backend
that lazily drives the real Docling converter.
"""

from __future__ import annotations

import importlib.util
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from .engine import EngineUnavailableError, ParseError, ParsingEngine
from .raw_result import (
    BBox,
    RawBlock,
    RawFigure,
    RawPage,
    RawParseResult,
    RawTable,
    RawTableCell,
    SourceDocument,
)

#: Engine identifier, matching ``PipelineConfig.parsing_engine`` value "docling".
ENGINE_ID = "docling"

#: The import name probed to decide availability.
_MODULE_NAME = "docling"

# A backend turns a SourceDocument into Docling's native export dict. It is the
# single seam that touches the real library; everything else is pure mapping.
NativeResult = Mapping[str, Any]
Backend = Callable[[SourceDocument, Optional[float]], NativeResult]

# Docling text labels that denote a heading/title (carry a nesting level).
_HEADING_LABELS = frozenset({"section_header", "title", "heading", "subtitle"})

# Map Docling text labels to the engine-neutral RawBlock ``kind`` vocabulary the
# Parser understands. Unknown labels fall through to "paragraph".
_LABEL_TO_KIND: Dict[str, str] = {
    "paragraph": "paragraph",
    "text": "paragraph",
    "section_header": "heading",
    "title": "heading",
    "subtitle": "heading",
    "heading": "heading",
    "caption": "caption",
    "list_item": "list_item",
    "footnote": "footnote",
}


class DoclingAdapter(ParsingEngine):
    """Parsing engine backed by Docling, behind the :class:`ParsingEngine` port."""

    def __init__(
        self,
        *,
        backend: Optional[Backend] = None,
        available: Optional[bool] = None,
    ) -> None:
        """Create the adapter.

        Args:
            backend: optional injectable seam producing Docling's native export
                dict for a document. When omitted, a default backend that lazily
                drives the real Docling converter is used. Injecting a fake here
                lets tests exercise the mapping without installing Docling.
            available: optional override for :meth:`is_available`. When ``None``
                (the default) availability is probed from the environment.
        """
        self._backend = backend
        self._available_override = available

    # -- port methods -----------------------------------------------------
    def engine_id(self) -> str:
        return ENGINE_ID

    def is_available(self) -> bool:
        """Whether Docling can be used in this environment (Req 2.6).

        Resolution order: an explicit ``available`` override, then the presence
        of an injected backend (always usable), then a lazy probe for the
        ``docling`` import without actually importing it.
        """
        if self._available_override is not None:
            return self._available_override
        if self._backend is not None:
            return True
        return _module_installed(_MODULE_NAME)

    def parse(
        self,
        doc: SourceDocument,
        deadline: Optional[float] = None,
    ) -> RawParseResult:
        """Parse ``doc`` and map Docling's output into :class:`RawParseResult`.

        Raises:
            EngineUnavailableError: Docling is not available (Req 2.6).
            ParseError: the backend produced output that could not be mapped
                (Req 2.5).
        """
        if not isinstance(doc, SourceDocument):
            raise TypeError("doc must be a SourceDocument")
        if not self.is_available():
            raise EngineUnavailableError(
                f"parsing engine {ENGINE_ID!r} is unavailable: "
                "the 'docling' package is not installed"
            )

        backend = self._backend or _default_backend
        native = backend(doc, deadline)
        try:
            return self.map_native(native)
        except EngineUnavailableError:
            raise
        except ParseError:
            raise
        except Exception as exc:  # pragma: no cover - defensive
            raise ParseError(f"failed to map Docling output: {exc}") from exc

    # -- native -> RawParseResult mapping --------------------------------
    @classmethod
    def map_native(cls, native: NativeResult) -> RawParseResult:
        """Map Docling's ``export_to_dict``-shaped output to :class:`RawParseResult`.

        The expected (best-effort, defensively read) shape is::

            {
              "texts":    [{"text", "label", "level"?, "prov": [{"page_no", "bbox"?}]}],
              "tables":   [{"prov": [...], "data": {"table_cells": [...]}}],
              "pictures": [{"prov": [...], "captions": [{"text"}] | "caption"}],
              "pages":    {"1": {"text_layer"?}} | [{"page_no", "text_layer"?}],
            }

        Every field is read defensively so partial or slightly divergent Docling
        versions still map without raising.
        """
        if not isinstance(native, Mapping):
            raise ParseError(
                f"Docling result must be a mapping, got {type(native).__name__}"
            )

        blocks = [cls._map_text(t) for t in _as_sequence(native.get("texts"))]
        blocks = [b for b in blocks if b is not None]

        tables = [cls._map_table(t) for t in _as_sequence(native.get("tables"))]
        figures = [cls._map_figure(f) for f in _as_sequence(native.get("pictures"))]
        pages = cls._map_pages(native.get("pages"))

        return RawParseResult(
            engine_id=ENGINE_ID,
            blocks=blocks,
            tables=tables,
            figures=figures,
            pages=pages,
        )

    @staticmethod
    def _map_text(item: Any) -> Optional[RawBlock]:
        if not isinstance(item, Mapping):
            return None
        text = item.get("text")
        if not isinstance(text, str):
            return None
        label = str(item.get("label", "paragraph")).strip().lower()
        kind = _LABEL_TO_KIND.get(label, "paragraph")
        page_no, bbox = _first_prov(item.get("prov"))
        heading_level: Optional[int] = None
        if label in _HEADING_LABELS or kind == "heading":
            kind = "heading"
            level = item.get("level")
            heading_level = level if isinstance(level, int) and level >= 1 else 1
        return RawBlock(
            text=text,
            page_number=page_no,
            kind=kind,
            heading_level=heading_level,
            bbox=bbox,
        )

    @staticmethod
    def _map_table(item: Any) -> RawTable:
        if not isinstance(item, Mapping):
            return RawTable(page_number=0, degraded=True, raw_text=None)
        page_no, bbox = _first_prov(item.get("prov"))
        data = item.get("data")
        cells: List[RawTableCell] = []
        if isinstance(data, Mapping):
            for raw_cell in _as_sequence(data.get("table_cells")):
                cell = _map_table_cell(raw_cell)
                if cell is not None:
                    cells.append(cell)
        raw_text = item.get("raw_text") if isinstance(item.get("raw_text"), str) else None
        degraded = bool(item.get("degraded", False)) or (not cells and raw_text is not None)
        return RawTable(
            page_number=page_no,
            cells=cells,
            bbox=bbox,
            degraded=degraded,
            raw_text=raw_text,
        )

    @staticmethod
    def _map_figure(item: Any) -> RawFigure:
        if not isinstance(item, Mapping):
            return RawFigure(page_number=0, image_ref="", caption=None)
        page_no, bbox = _first_prov(item.get("prov"))
        caption = _extract_caption(item)
        image_ref = item.get("image_ref")
        if not isinstance(image_ref, str):
            self_ref = item.get("self_ref")
            image_ref = self_ref if isinstance(self_ref, str) else ""
        return RawFigure(
            page_number=page_no,
            image_ref=image_ref,
            caption=caption,
            bbox=bbox,
        )

    @staticmethod
    def _map_pages(pages: Any) -> List[RawPage]:
        result: List[RawPage] = []
        if isinstance(pages, Mapping):
            items: Sequence[Any] = list(pages.values())
            keys: Sequence[Any] = list(pages.keys())
        elif isinstance(pages, (list, tuple)):
            items = list(pages)
            keys = [None] * len(items)
        else:
            return result
        for key, page in zip(keys, items):
            if not isinstance(page, Mapping):
                continue
            page_no = page.get("page_no")
            if not isinstance(page_no, int) or isinstance(page_no, bool):
                page_no = _coerce_page_key(key)
            has_text_layer = bool(page.get("text_layer", page.get("has_text_layer", True)))
            image_ref = page.get("page_image_ref")
            result.append(
                RawPage(
                    page_number=max(0, page_no),
                    has_text_layer=has_text_layer,
                    page_image_ref=image_ref if isinstance(image_ref, str) else None,
                )
            )
        return result


# -- module-level helpers -------------------------------------------------
def _module_installed(name: str) -> bool:
    """Return whether ``name`` is importable without importing it."""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):  # pragma: no cover - defensive
        return False


def _as_sequence(value: Any) -> Sequence[Any]:
    if isinstance(value, (list, tuple)):
        return value
    return ()


def _coerce_page_key(key: Any) -> int:
    try:
        return max(0, int(key))
    except (TypeError, ValueError):
        return 0


def _first_prov(prov: Any) -> "tuple[int, Optional[BBox]]":
    """Read the page number and bbox from the first provenance entry."""
    items = _as_sequence(prov)
    if not items:
        return 0, None
    first = items[0]
    if not isinstance(first, Mapping):
        return 0, None
    page_no = first.get("page_no", first.get("page", 0))
    if not isinstance(page_no, int) or isinstance(page_no, bool):
        page_no = 0
    # Docling page numbers are 1-based; the neutral shape only requires >= 0.
    return max(0, page_no), _map_bbox(first.get("bbox"))


def _map_bbox(bbox: Any) -> Optional[BBox]:
    if not isinstance(bbox, Mapping):
        return None
    try:
        return BBox(
            x0=float(bbox["l"]),
            y0=float(bbox["t"]),
            x1=float(bbox["r"]),
            y1=float(bbox["b"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _map_table_cell(raw_cell: Any) -> Optional[RawTableCell]:
    if not isinstance(raw_cell, Mapping):
        return None
    text = raw_cell.get("text", "")
    if not isinstance(text, str):
        text = str(text)
    # Skip genuinely empty cells: only non-empty source cells are assigned a
    # coordinate (Req 3.1).
    if text.strip() == "":
        return None
    row = raw_cell.get("start_row_offset_idx", raw_cell.get("row_index", 0))
    col = raw_cell.get("start_col_offset_idx", raw_cell.get("col_index", 0))
    row_span = raw_cell.get("row_span", 1)
    col_span = raw_cell.get("col_span", 1)
    return RawTableCell(
        row_index=max(0, int(row) if isinstance(row, int) and not isinstance(row, bool) else 0),
        col_index=max(0, int(col) if isinstance(col, int) and not isinstance(col, bool) else 0),
        value=text,
        row_span=max(1, int(row_span) if isinstance(row_span, int) and not isinstance(row_span, bool) else 1),
        col_span=max(1, int(col_span) if isinstance(col_span, int) and not isinstance(col_span, bool) else 1),
    )


def _extract_caption(item: Mapping[str, Any]) -> Optional[str]:
    captions = item.get("captions")
    if isinstance(captions, (list, tuple)) and captions:
        first = captions[0]
        if isinstance(first, Mapping) and isinstance(first.get("text"), str):
            return first["text"]
        if isinstance(first, str):
            return first
    caption = item.get("caption")
    if isinstance(caption, str):
        return caption
    return None


def _default_backend(
    doc: SourceDocument,
    deadline: Optional[float] = None,
) -> NativeResult:
    """Drive the real Docling converter, lazily importing it (Req 2.6).

    Imported inside the function so importing this module never requires Docling.
    Raises :class:`EngineUnavailableError` if the import fails despite the
    availability probe, and :class:`ParseError` on conversion failure (Req 2.5).
    """
    try:
        from docling.document_converter import DocumentConverter  # type: ignore
        from docling.datamodel.base_models import DocumentStream  # type: ignore
    except Exception as exc:  # pragma: no cover - requires docling absent
        raise EngineUnavailableError(
            f"parsing engine {ENGINE_ID!r} is unavailable: {exc}"
        ) from exc

    import io

    try:  # pragma: no cover - exercised only with docling installed
        stream = DocumentStream(name=doc.document_id, stream=io.BytesIO(doc.raw_bytes))
        converter = DocumentConverter()
        result = converter.convert(stream)
        return result.document.export_to_dict()
    except Exception as exc:  # pragma: no cover - requires docling installed
        raise ParseError(f"Docling failed to parse the document: {exc}") from exc
