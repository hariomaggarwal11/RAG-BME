"""Smoke tests for the Docling and LlamaParse adapters (Task 5.2).

These cover the two responsibilities of the concrete adapters:

* Availability reporting (Req 2.6) — the ``available`` override, an injected
  backend implying availability, and graceful degradation when the optional
  third-party library is absent.
* Mapping the backend's native output into the shared ``RawParseResult`` shape
  (Req 2.2) — exercised with a deterministic fake backend so the mapping is
  tested without installing Docling or calling the LlamaParse service.

The adapters are also checked to be registered with the default registry so a
``PipelineConfig`` selects them by configuration (Req 2.2).
"""

from __future__ import annotations

import pytest

from biomed_rag.config import ParsingEngine as ParsingEngineChoice
from biomed_rag.config import PipelineConfig
from biomed_rag.parsing import (
    DoclingAdapter,
    EngineUnavailableError,
    LlamaParseAdapter,
    ParsingEngine,
    RawParseResult,
    SourceDocument,
    default_registry,
)


def _doc(data: bytes = b"%PDF-1.7 fake") -> SourceDocument:
    return SourceDocument(document_id="doc-1", raw_bytes=data)


# -- Docling --------------------------------------------------------------
def test_docling_implements_port_and_id() -> None:
    adapter = DoclingAdapter(available=False)
    assert isinstance(adapter, ParsingEngine)
    assert adapter.engine_id() == "docling"


def test_docling_availability_override() -> None:
    assert DoclingAdapter(available=True).is_available() is True
    assert DoclingAdapter(available=False).is_available() is False


def test_docling_injected_backend_is_available() -> None:
    adapter = DoclingAdapter(backend=lambda doc, deadline: {})
    assert adapter.is_available() is True


def test_docling_parse_unavailable_raises() -> None:
    adapter = DoclingAdapter(available=False)
    with pytest.raises(EngineUnavailableError):
        adapter.parse(_doc())


def test_docling_maps_native_output() -> None:
    native = {
        "texts": [
            {
                "text": "Methods",
                "label": "section_header",
                "level": 2,
                "prov": [{"page_no": 1, "bbox": {"l": 0, "t": 10, "r": 100, "b": 20}}],
            },
            {
                "text": "The assay was performed.",
                "label": "paragraph",
                "prov": [{"page_no": 1}],
            },
        ],
        "tables": [
            {
                "prov": [{"page_no": 2}],
                "data": {
                    "table_cells": [
                        {
                            "start_row_offset_idx": 0,
                            "start_col_offset_idx": 0,
                            "row_span": 2,
                            "col_span": 1,
                            "text": "Gene",
                        },
                        {"start_row_offset_idx": 0, "start_col_offset_idx": 1, "text": "p"},
                        # An empty cell is not assigned a coordinate (Req 3.1).
                        {"start_row_offset_idx": 1, "start_col_offset_idx": 1, "text": "  "},
                    ]
                },
            }
        ],
        "pictures": [
            {"prov": [{"page_no": 3}], "captions": [{"text": "Figure 1"}]},
            {"prov": [{"page_no": 3}]},  # caption absent (Req 3.4)
        ],
        "pages": {"1": {"page_no": 1, "text_layer": True}},
    }
    adapter = DoclingAdapter(backend=lambda doc, deadline: native)
    result = adapter.parse(_doc())

    assert isinstance(result, RawParseResult)
    assert result.engine_id == "docling"

    # Heading carries its nesting level; paragraph does not.
    heading, paragraph = result.blocks
    assert heading.kind == "heading"
    assert heading.heading_level == 2
    assert heading.page_number == 1
    assert heading.bbox is not None and heading.bbox.as_tuple() == (0.0, 10.0, 100.0, 20.0)
    assert paragraph.kind == "paragraph"
    assert paragraph.heading_level is None

    # Table: empty cell dropped; spanning cell recorded at top-left.
    assert len(result.tables) == 1
    cells = result.tables[0].cells
    assert len(cells) == 2
    span_cell = next(c for c in cells if c.value == "Gene")
    assert (span_cell.row_index, span_cell.col_index) == (0, 0)
    assert span_cell.row_span == 2 and span_cell.col_span == 1

    # Figures: one with caption, one without.
    assert [f.caption for f in result.figures] == ["Figure 1", None]
    assert [f.page_number for f in result.figures] == [3, 3]


def test_docling_degraded_table_retains_raw_text() -> None:
    native = {"tables": [{"prov": [{"page_no": 1}], "raw_text": "unstructured table text"}]}
    result = DoclingAdapter(backend=lambda doc, deadline: native).parse(_doc())
    table = result.tables[0]
    assert table.degraded is True
    assert table.raw_text == "unstructured table text"
    assert table.cells == []


# -- LlamaParse -----------------------------------------------------------
def test_llamaparse_implements_port_and_id() -> None:
    adapter = LlamaParseAdapter(available=False)
    assert isinstance(adapter, ParsingEngine)
    assert adapter.engine_id() == "llamaparse"


def test_llamaparse_availability_override() -> None:
    assert LlamaParseAdapter(available=True).is_available() is True
    assert LlamaParseAdapter(available=False).is_available() is False


def test_llamaparse_injected_backend_is_available() -> None:
    adapter = LlamaParseAdapter(backend=lambda doc, deadline: [])
    assert adapter.is_available() is True


def test_llamaparse_parse_unavailable_raises() -> None:
    adapter = LlamaParseAdapter(available=False)
    with pytest.raises(EngineUnavailableError):
        adapter.parse(_doc())


def test_llamaparse_maps_native_output() -> None:
    native = [
        {
            "pages": [
                {
                    "page": 1,
                    "items": [
                        {"type": "heading", "lvl": 1, "value": "Results"},
                        {"type": "text", "value": "Expression increased."},
                        {
                            "type": "table",
                            "rows": [["Gene", "p"], ["TP53", "0.01"]],
                        },
                    ],
                    "images": [{"name": "img_1", "caption": "Figure 1"}, {"name": "img_2"}],
                }
            ]
        }
    ]
    adapter = LlamaParseAdapter(backend=lambda doc, deadline: native)
    result = adapter.parse(_doc())

    assert result.engine_id == "llamaparse"
    heading, paragraph = result.blocks
    assert heading.kind == "heading" and heading.heading_level == 1
    assert paragraph.kind == "paragraph"
    assert all(b.page_number == 1 for b in result.blocks)

    # Table rows -> (row, col) cell assignment, single-span.
    assert len(result.tables) == 1
    values = {(c.row_index, c.col_index): c.value for c in result.tables[0].cells}
    assert values == {
        (0, 0): "Gene",
        (0, 1): "p",
        (1, 0): "TP53",
        (1, 1): "0.01",
    }

    # Images -> figures, caption optional.
    assert [f.caption for f in result.figures] == ["Figure 1", None]


def test_llamaparse_falls_back_to_page_text() -> None:
    native = {"pages": [{"page": 2, "text": "raw page text only"}]}
    result = LlamaParseAdapter(backend=lambda doc, deadline: native).parse(_doc())
    assert len(result.blocks) == 1
    assert result.blocks[0].text == "raw page text only"
    assert result.blocks[0].page_number == 2


def test_llamaparse_rejects_bad_native_shape() -> None:
    from biomed_rag.parsing import ParseError

    adapter = LlamaParseAdapter(backend=lambda doc, deadline: 42)
    with pytest.raises(ParseError):
        adapter.parse(_doc())


# -- registry wiring ------------------------------------------------------
def test_adapters_registered_and_selectable_by_config() -> None:
    docling_cfg = PipelineConfig(parsing_engine=ParsingEngineChoice.DOCLING)
    llama_cfg = PipelineConfig(parsing_engine=ParsingEngineChoice.LLAMAPARSE)

    assert default_registry.is_registered(ParsingEngineChoice.DOCLING)
    assert default_registry.is_registered(ParsingEngineChoice.LLAMAPARSE)
    assert isinstance(default_registry.select(docling_cfg), DoclingAdapter)
    assert isinstance(default_registry.select(llama_cfg), LlamaParseAdapter)
