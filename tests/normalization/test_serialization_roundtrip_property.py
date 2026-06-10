"""Property test for the NormalizedDocument serialization round-trip (Req 5.5, 5.6).

Feature: biomedical-rag-pipeline, Property 12: Normalized representation serialization round-trip

Statement: for any NormalizedDocument ``doc``, ``deserialize(serialize(doc))`` is
structurally equivalent to ``doc``. The durable serialized form is what the
Orchestrator persists between stages for resume, so it must losslessly preserve
the document identifier and every content element -- its kind, page number,
reading-order position, heading path, and kind-specific payload (text +
heading level; table cells with spans, the degraded flag, and raw text; figure
image reference with/without a caption).

The dataclasses (:class:`NormalizedDocument`, :class:`ContentElement`,
:class:`TextPayload`, :class:`TablePayload`, :class:`FigurePayload`,
:class:`Cell`) all compare structurally via generated ``__eq__``, so equality of
the restored document and the original is exact structural equivalence.
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from biomed_rag.models.enums import ElementKind
from biomed_rag.models.normalized import (
    ContentElement,
    FigurePayload,
    NormalizedDocument,
    TablePayload,
    TextPayload,
)
from biomed_rag.models.parsed import Cell
from biomed_rag.normalization import deserialize, serialize

# Restrict text to UTF-8 encodable code points (excludes lone surrogates) so the
# JSON-then-UTF-8 serialized form is always well defined; this constrains the
# generator to the actual input space (strings that can be persisted) rather
# than weakening the property.
_text = st.text(alphabet=st.characters(codec="utf-8"), max_size=24)
_page_number = st.integers(min_value=0, max_value=5000)
_reading_order = st.integers(min_value=0, max_value=5000)
_heading_path = st.lists(_text, max_size=4)


def _text_payloads() -> st.SearchStrategy[TextPayload]:
    # headingLevel is either absent (None) or an int >= 1 (model invariant).
    return st.builds(
        TextPayload,
        text=_text,
        headingLevel=st.one_of(st.none(), st.integers(min_value=1, max_value=10)),
    )


def _cells() -> st.SearchStrategy[Cell]:
    # Cell spans are always >= 1 (a cell spans at least itself); indices >= 0.
    return st.builds(
        Cell,
        rowIndex=st.integers(min_value=0, max_value=100),
        colIndex=st.integers(min_value=0, max_value=100),
        value=_text,
        rowSpan=st.integers(min_value=1, max_value=8),
        colSpan=st.integers(min_value=1, max_value=8),
    )


def _table_payloads() -> st.SearchStrategy[TablePayload]:
    return st.builds(
        TablePayload,
        cells=st.lists(_cells(), max_size=6),
        degraded=st.booleans(),
        rawText=st.one_of(st.none(), _text),
    )


def _figure_payloads() -> st.SearchStrategy[FigurePayload]:
    # caption is present or absent (None) -- both must round-trip (Req 3.4).
    return st.builds(
        FigurePayload,
        imageRef=_text,
        caption=st.one_of(st.none(), _text),
    )


def _elements() -> st.SearchStrategy[ContentElement]:
    """An element of every kind, with a matching payload and varied metadata."""
    text_like = st.tuples(
        st.sampled_from([ElementKind.TEXT, ElementKind.HEADING]), _text_payloads()
    )
    table_like = st.tuples(st.just(ElementKind.TABLE), _table_payloads())
    figure_like = st.tuples(st.just(ElementKind.FIGURE), _figure_payloads())

    return st.one_of(text_like, table_like, figure_like).flatmap(
        lambda kp: st.builds(
            ContentElement,
            kind=st.just(kp[0]),
            pageNumber=_page_number,
            readingOrderPosition=_reading_order,
            payload=st.just(kp[1]),
            headingPath=_heading_path,
        )
    )


def _normalized_documents() -> st.SearchStrategy[NormalizedDocument]:
    return st.builds(
        NormalizedDocument,
        documentId=st.text(
            alphabet=st.characters(codec="utf-8"), min_size=1, max_size=24
        ),
        elements=st.lists(_elements(), max_size=10),
    )


# Feature: biomedical-rag-pipeline, Property 12: Normalized representation serialization round-trip
@settings(max_examples=200)
@given(doc=_normalized_documents())
def test_serialization_roundtrip_is_structurally_equivalent(
    doc: NormalizedDocument,
) -> None:
    """Validates: Requirements 5.5, 5.6"""
    restored = deserialize(serialize(doc))

    # serialize must produce bytes (the durable persisted form).
    assert isinstance(serialize(doc), bytes)

    # Round-trip yields a structurally equivalent NormalizedDocument: the
    # dataclass __eq__ compares the documentId and every element (kind, page
    # number, reading-order position, heading path, and payload) recursively.
    assert isinstance(restored, NormalizedDocument)
    assert restored == doc
