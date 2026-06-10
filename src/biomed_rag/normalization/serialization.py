"""Durable serialization of :class:`NormalizedDocument` (task 7.3, Req 5.5, 5.6).

The :class:`Normalizer` produces a canonical :class:`NormalizedDocument`; the
Orchestrator persists that representation between stages so a failed job can be
resumed without re-normalizing (design: "durable persisted form used by the
Orchestrator for resume"). This module defines that durable form.

The format is structured JSON encoded to UTF-8 ``bytes``. Every field that
contributes to structural equivalence is captured explicitly so that
``deserialize(serialize(doc))`` reconstructs a structurally-equivalent document
(the round-trip property, Req 5.6):

* ``documentId`` (Req 5.4);
* per element ``kind``, ``pageNumber``, ``readingOrderPosition`` and
  ``headingPath`` (Req 5.4, 5.5); and
* the kind-specific payload --
  ``text`` + ``headingLevel`` for TEXT/HEADING,
  ``cells`` (each cell's row/col index, span, value) + ``degraded`` + ``rawText``
  for TABLE, and ``imageRef`` + ``caption`` for FIGURE (Req 3.1-3.4, 3.6, 5.4).

A ``schemaVersion`` is embedded so the persisted form can evolve compatibly.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from biomed_rag.models.enums import ElementKind
from biomed_rag.models.normalized import (
    ContentElement,
    FigurePayload,
    NormalizedDocument,
    Payload,
    TablePayload,
    TextPayload,
)
from biomed_rag.models.parsed import Cell

# Schema version of the durable serialized form. Bump when the on-disk shape
# changes in a way that requires migration on read.
SCHEMA_VERSION = 1


class DeserializationError(ValueError):
    """Raised when serialized bytes cannot be interpreted as a NormalizedDocument."""


# -- encoding ----------------------------------------------------------------
def _encode_payload(kind: ElementKind, payload: Payload) -> Dict[str, Any]:
    """Encode a kind-specific payload into a JSON-serializable mapping."""
    if kind in (ElementKind.TEXT, ElementKind.HEADING):
        assert isinstance(payload, TextPayload)
        return {"text": payload.text, "headingLevel": payload.headingLevel}
    if kind is ElementKind.TABLE:
        assert isinstance(payload, TablePayload)
        return {
            "cells": [
                {
                    "rowIndex": cell.rowIndex,
                    "colIndex": cell.colIndex,
                    "value": cell.value,
                    "rowSpan": cell.rowSpan,
                    "colSpan": cell.colSpan,
                }
                for cell in payload.cells
            ],
            "degraded": payload.degraded,
            "rawText": payload.rawText,
        }
    if kind is ElementKind.FIGURE:
        assert isinstance(payload, FigurePayload)
        return {"imageRef": payload.imageRef, "caption": payload.caption}
    raise ValueError(f"unknown element kind: {kind!r}")  # pragma: no cover


def _encode_element(element: ContentElement) -> Dict[str, Any]:
    return {
        "kind": element.kind.value,
        "pageNumber": element.pageNumber,
        "readingOrderPosition": element.readingOrderPosition,
        "headingPath": list(element.headingPath),
        "payload": _encode_payload(element.kind, element.payload),
    }


def serialize(doc: NormalizedDocument) -> bytes:
    """Serialize ``doc`` into the durable byte form (Req 5.6).

    The output is deterministic UTF-8 JSON capturing every field required for
    structural equivalence.
    """
    if not isinstance(doc, NormalizedDocument):
        raise TypeError(
            f"serialize expects a NormalizedDocument, got {type(doc).__name__}"
        )
    envelope = {
        "schemaVersion": SCHEMA_VERSION,
        "documentId": doc.documentId,
        "elements": [_encode_element(e) for e in doc.elements],
    }
    return json.dumps(envelope, ensure_ascii=False, sort_keys=True).encode("utf-8")


# -- decoding ----------------------------------------------------------------
def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DeserializationError(message)


def _decode_payload(kind: ElementKind, raw: Any) -> Payload:
    _require(isinstance(raw, dict), "element payload must be an object")
    if kind in (ElementKind.TEXT, ElementKind.HEADING):
        return TextPayload(text=raw["text"], headingLevel=raw.get("headingLevel"))
    if kind is ElementKind.TABLE:
        raw_cells = raw.get("cells", [])
        _require(isinstance(raw_cells, list), "TABLE payload cells must be a list")
        cells: List[Cell] = []
        for rc in raw_cells:
            _require(isinstance(rc, dict), "each table cell must be an object")
            cells.append(
                Cell(
                    rowIndex=rc["rowIndex"],
                    colIndex=rc["colIndex"],
                    value=rc["value"],
                    rowSpan=rc.get("rowSpan", 1),
                    colSpan=rc.get("colSpan", 1),
                )
            )
        return TablePayload(
            cells=cells,
            degraded=raw.get("degraded", False),
            rawText=raw.get("rawText"),
        )
    if kind is ElementKind.FIGURE:
        return FigurePayload(imageRef=raw["imageRef"], caption=raw.get("caption"))
    raise DeserializationError(f"unknown element kind: {kind!r}")  # pragma: no cover


def _decode_kind(raw: Any) -> ElementKind:
    try:
        return ElementKind(raw)
    except ValueError as exc:
        raise DeserializationError(f"unknown element kind value: {raw!r}") from exc


def _decode_element(raw: Any) -> ContentElement:
    _require(isinstance(raw, dict), "each element must be an object")
    kind = _decode_kind(raw["kind"])
    return ContentElement(
        kind=kind,
        pageNumber=raw["pageNumber"],
        readingOrderPosition=raw["readingOrderPosition"],
        payload=_decode_payload(kind, raw["payload"]),
        headingPath=list(raw.get("headingPath", [])),
    )


def deserialize(data: bytes) -> NormalizedDocument:
    """Reconstruct a :class:`NormalizedDocument` from its durable byte form.

    Inverse of :func:`serialize`; together they satisfy the round-trip property
    (Req 5.6). Raises :class:`DeserializationError` when ``data`` is not a valid
    serialized NormalizedDocument.
    """
    _require(isinstance(data, (bytes, bytearray)), "serialized data must be bytes")
    try:
        envelope = json.loads(bytes(data).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DeserializationError(f"data is not valid serialized JSON: {exc}") from exc

    _require(isinstance(envelope, dict), "serialized root must be an object")
    raw_elements = envelope.get("elements", [])
    _require(isinstance(raw_elements, list), "serialized elements must be a list")

    try:
        elements = [_decode_element(e) for e in raw_elements]
        return NormalizedDocument(
            documentId=envelope["documentId"],
            elements=elements,
        )
    except KeyError as exc:
        raise DeserializationError(f"missing required field: {exc}") from exc
