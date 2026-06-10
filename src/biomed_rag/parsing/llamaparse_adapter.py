"""LlamaParse adapter for the pluggable Parsing_Engine port (Req 2.2, 2.6).

:class:`LlamaParseAdapter` implements the :class:`ParsingEngine` port on top of
`LlamaParse <https://github.com/run-llama/llama_cloud_services>`_, a hosted
document parsing service. As with every adapter, it maps the backend's native
output into the engine-neutral :class:`RawParseResult` shape so stage logic
depends only on the port (design: ports-and-adapters).

LlamaParse is an optional dependency *and* a network service that needs an API
key, so two things can make it unavailable: the client library being absent, or
no API key being configured. This adapter reports availability for both and
never imports the client at module load time:

* Availability is probed lazily — the client library import is checked via
  :func:`importlib.util.find_spec` and an API key is looked for in the
  environment (``LLAMA_CLOUD_API_KEY`` / ``LLAMA_PARSE_API_KEY``). The Parser
  checks :meth:`is_available` and fails the job closed when unavailable rather
  than crashing (Req 2.6).
* :meth:`parse` raises :class:`EngineUnavailableError` when the client cannot be
  used, which the Parser turns into a fail-closed job outcome.

Testability seam
----------------
``parse`` obtains the engine's *native* result (the per-page JSON LlamaParse
returns from ``get_json_result``) from an injectable ``backend`` callable and
maps it with :meth:`map_native`. Tests inject a fake ``backend`` returning a
hand-built structure to exercise the mapping without the real service.
"""

from __future__ import annotations

import importlib.util
import os
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from .engine import EngineUnavailableError, ParseError, ParsingEngine
from .raw_result import (
    RawBlock,
    RawFigure,
    RawPage,
    RawParseResult,
    RawTable,
    RawTableCell,
    SourceDocument,
)

#: Engine identifier, matching ``PipelineConfig.parsing_engine`` value "llamaparse".
ENGINE_ID = "llamaparse"

#: Client import names to probe (newer SDK first, then the legacy package).
_MODULE_CANDIDATES = ("llama_cloud_services", "llama_parse")

#: Environment variables that may hold the LlamaParse API key.
_API_KEY_ENV_VARS = ("LLAMA_CLOUD_API_KEY", "LLAMA_PARSE_API_KEY")

# A backend turns a SourceDocument into LlamaParse's native JSON result.
NativeResult = Any
Backend = Callable[[SourceDocument, Optional[float]], NativeResult]

# LlamaParse item types that denote a heading.
_HEADING_TYPES = frozenset({"heading", "title"})

# Map LlamaParse item types to the engine-neutral RawBlock ``kind`` vocabulary.
_TYPE_TO_KIND: Dict[str, str] = {
    "text": "paragraph",
    "paragraph": "paragraph",
    "heading": "heading",
    "title": "heading",
    "caption": "caption",
}


class LlamaParseAdapter(ParsingEngine):
    """Parsing engine backed by LlamaParse, behind the :class:`ParsingEngine` port."""

    def __init__(
        self,
        *,
        backend: Optional[Backend] = None,
        available: Optional[bool] = None,
    ) -> None:
        """Create the adapter.

        Args:
            backend: optional injectable seam producing LlamaParse's native JSON
                result for a document. When omitted, a default backend that
                lazily drives the real client is used. Injecting a fake lets
                tests exercise the mapping without the client or a network call.
            available: optional override for :meth:`is_available`. When ``None``
                availability is probed from the environment.
        """
        self._backend = backend
        self._available_override = available

    # -- port methods -----------------------------------------------------
    def engine_id(self) -> str:
        return ENGINE_ID

    def is_available(self) -> bool:
        """Whether LlamaParse can be used in this environment (Req 2.6).

        Resolution order: an explicit ``available`` override, then the presence
        of an injected backend (always usable), then a lazy probe requiring both
        the client library and a configured API key.
        """
        if self._available_override is not None:
            return self._available_override
        if self._backend is not None:
            return True
        return _client_installed() and _api_key_configured()

    def parse(
        self,
        doc: SourceDocument,
        deadline: Optional[float] = None,
    ) -> RawParseResult:
        """Parse ``doc`` and map LlamaParse's output into :class:`RawParseResult`.

        Raises:
            EngineUnavailableError: LlamaParse is not available (Req 2.6).
            ParseError: the backend produced output that could not be mapped
                (Req 2.5).
        """
        if not isinstance(doc, SourceDocument):
            raise TypeError("doc must be a SourceDocument")
        if not self.is_available():
            raise EngineUnavailableError(
                f"parsing engine {ENGINE_ID!r} is unavailable: "
                "the LlamaParse client or API key is not configured"
            )

        backend = self._backend or _default_backend
        native = backend(doc, deadline)
        try:
            return self.map_native(native)
        except (EngineUnavailableError, ParseError):
            raise
        except Exception as exc:  # pragma: no cover - defensive
            raise ParseError(f"failed to map LlamaParse output: {exc}") from exc

    # -- native -> RawParseResult mapping --------------------------------
    @classmethod
    def map_native(cls, native: NativeResult) -> RawParseResult:
        """Map LlamaParse's JSON result to :class:`RawParseResult`.

        LlamaParse returns a list of job results, each with a ``pages`` list.
        The expected (defensively read) per-page shape is::

            {
              "page": 1,
              "items": [
                {"type": "heading", "lvl": 2, "value": "..."},
                {"type": "text", "value": "..."},
                {"type": "table", "rows": [[...]], "csv"?, "md"?},
              ],
              "images": [{"name": "img_1", "caption"?}],
              "text": "raw page text",
            }

        A bare ``pages`` mapping, a single job-result mapping, or a list of job
        results are all accepted.
        """
        pages = _extract_pages(native)

        blocks: List[RawBlock] = []
        tables: List[RawTable] = []
        figures: List[RawFigure] = []
        raw_pages: List[RawPage] = []

        for page in pages:
            if not isinstance(page, Mapping):
                continue
            page_no = _page_number(page)
            items = _as_sequence(page.get("items"))
            had_text = False
            for item in items:
                mapped = cls._map_item(item, page_no)
                if isinstance(mapped, RawBlock):
                    blocks.append(mapped)
                    had_text = True
                elif isinstance(mapped, RawTable):
                    tables.append(mapped)
            for image in _as_sequence(page.get("images")):
                fig = cls._map_image(image, page_no)
                if fig is not None:
                    figures.append(fig)
            # When no structured items were produced, fall back to the page's
            # raw text so genuine content is not lost.
            if not items and isinstance(page.get("text"), str) and page["text"].strip():
                blocks.append(
                    RawBlock(text=page["text"], page_number=page_no, kind="paragraph")
                )
                had_text = True
            raw_pages.append(RawPage(page_number=page_no, has_text_layer=had_text))

        return RawParseResult(
            engine_id=ENGINE_ID,
            blocks=blocks,
            tables=tables,
            figures=figures,
            pages=raw_pages,
        )

    @staticmethod
    def _map_item(item: Any, page_no: int):
        if not isinstance(item, Mapping):
            return None
        item_type = str(item.get("type", "text")).strip().lower()
        if item_type == "table":
            return _map_table_item(item, page_no)
        value = item.get("value", item.get("text", item.get("md")))
        if not isinstance(value, str):
            return None
        kind = _TYPE_TO_KIND.get(item_type, "paragraph")
        heading_level: Optional[int] = None
        if item_type in _HEADING_TYPES or kind == "heading":
            kind = "heading"
            level = item.get("lvl", item.get("level"))
            heading_level = level if isinstance(level, int) and level >= 1 else 1
        return RawBlock(
            text=value,
            page_number=page_no,
            kind=kind,
            heading_level=heading_level,
        )

    @staticmethod
    def _map_image(image: Any, page_no: int) -> Optional[RawFigure]:
        if not isinstance(image, Mapping):
            return None
        image_ref = image.get("name", image.get("image_ref", ""))
        if not isinstance(image_ref, str):
            image_ref = ""
        caption = image.get("caption")
        return RawFigure(
            page_number=page_no,
            image_ref=image_ref,
            caption=caption if isinstance(caption, str) else None,
        )


# -- module-level helpers -------------------------------------------------
def _client_installed() -> bool:
    """Return whether any supported LlamaParse client is importable."""
    for name in _MODULE_CANDIDATES:
        try:
            if importlib.util.find_spec(name) is not None:
                return True
        except (ImportError, ValueError):  # pragma: no cover - defensive
            continue
    return False


def _api_key_configured() -> bool:
    return any(os.environ.get(var) for var in _API_KEY_ENV_VARS)


def _as_sequence(value: Any) -> Sequence[Any]:
    if isinstance(value, (list, tuple)):
        return value
    return ()


def _extract_pages(native: NativeResult) -> Sequence[Any]:
    """Pull the per-page list out of LlamaParse's several result shapes."""
    if isinstance(native, Mapping):
        if isinstance(native.get("pages"), (list, tuple)):
            return native["pages"]
        return ()
    if isinstance(native, (list, tuple)):
        pages: List[Any] = []
        for entry in native:
            if isinstance(entry, Mapping) and isinstance(entry.get("pages"), (list, tuple)):
                pages.extend(entry["pages"])
            elif isinstance(entry, Mapping) and "items" in entry:
                # A bare list of page dicts.
                pages.append(entry)
        return pages
    raise ParseError(
        f"LlamaParse result must be a list or mapping, got {type(native).__name__}"
    )


def _page_number(page: Mapping[str, Any]) -> int:
    page_no = page.get("page", page.get("page_no", 0))
    if not isinstance(page_no, int) or isinstance(page_no, bool):
        return 0
    return max(0, page_no)


def _map_table_item(item: Mapping[str, Any], page_no: int) -> RawTable:
    """Map a LlamaParse ``table`` item into a :class:`RawTable`.

    LlamaParse tables come as a ``rows`` matrix (list of row lists). Each
    non-empty cell is assigned to its (row, col); LlamaParse does not report
    spans, so every cell is single-span. A ``degraded`` table with no rows
    retains its raw markdown/text (Req 3.6).
    """
    rows = item.get("rows")
    cells: List[RawTableCell] = []
    if isinstance(rows, (list, tuple)):
        for r, row in enumerate(rows):
            if not isinstance(row, (list, tuple)):
                continue
            for c, value in enumerate(row):
                text = value if isinstance(value, str) else ("" if value is None else str(value))
                if text.strip() == "":
                    continue
                cells.append(RawTableCell(row_index=r, col_index=c, value=text))
    raw_text = None
    for key in ("md", "csv", "text"):
        candidate = item.get(key)
        if isinstance(candidate, str):
            raw_text = candidate
            break
    degraded = bool(item.get("degraded", False)) or (not cells and raw_text is not None)
    return RawTable(
        page_number=page_no,
        cells=cells,
        degraded=degraded,
        raw_text=raw_text if (degraded or not cells) else None,
    )


def _default_backend(
    doc: SourceDocument,
    deadline: Optional[float] = None,
) -> NativeResult:
    """Drive the real LlamaParse client, lazily importing it (Req 2.6).

    Imported inside the function so importing this module never requires the
    client. Raises :class:`EngineUnavailableError` if the import or key lookup
    fails, and :class:`ParseError` on a parse failure (Req 2.5).
    """
    api_key = next((os.environ.get(v) for v in _API_KEY_ENV_VARS if os.environ.get(v)), None)
    if not api_key:
        raise EngineUnavailableError(
            f"parsing engine {ENGINE_ID!r} is unavailable: no API key configured"
        )

    parser = None
    try:  # pragma: no cover - requires the client installed
        try:
            from llama_cloud_services import LlamaParse  # type: ignore
        except Exception:
            from llama_parse import LlamaParse  # type: ignore
        parser = LlamaParse(api_key=api_key, result_type="json")
    except Exception as exc:  # pragma: no cover - requires client absent
        raise EngineUnavailableError(
            f"parsing engine {ENGINE_ID!r} is unavailable: {exc}"
        ) from exc

    import io
    import os.path

    try:  # pragma: no cover - requires the client + network
        extra_info = {"file_name": os.path.basename(doc.document_id) or "document"}
        return parser.get_json_result(io.BytesIO(doc.raw_bytes), extra_info=extra_info)
    except Exception as exc:  # pragma: no cover - requires the client + network
        raise ParseError(f"LlamaParse failed to parse the document: {exc}") from exc
