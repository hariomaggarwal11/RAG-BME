"""Discriminated result of :meth:`Normalizer.normalize` (Req 5.4, 5.7, 5.8).

The design (Normalizer section) specifies::

    normalize(parsed: ParsedDocument) -> NormalizationResult
    # NormalizationResult = Normalized(NormalizedDocument) | Empty(reason) | Malformed(error)

This module defines that three-arm discriminated union following the same
frozen-dataclass convention used elsewhere in the pipeline (``IngestionResult``,
``OCRResult``, ``EmbedResult``):

* :class:`Normalized` -- the document was interpreted and produced a non-empty
  canonical representation (Req 5.4, 5.5).
* :class:`Empty` -- the input was empty or carried no recognizable content; an
  *empty* :class:`NormalizedDocument` is still produced alongside a "no content"
  reason so callers receive both the empty representation and the indication
  (Req 5.7).
* :class:`Malformed` -- the input could not be interpreted as a Parsed_Document;
  it is rejected with an error indication and no document is produced (Req 5.8).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

from biomed_rag.models.normalized import NormalizedDocument


@dataclass(frozen=True)
class Normalized:
    """A successfully normalized, content-preserving document (Req 5.4, 5.5)."""

    document: NormalizedDocument


@dataclass(frozen=True)
class Empty:
    """Empty / no-content input (Req 5.7).

    ``document`` is the produced *empty* normalized representation (its
    ``elements`` list is empty) and ``reason`` is the "no content" indication.
    Both are surfaced so a caller has the empty representation to persist and the
    human-readable reason it was empty.
    """

    reason: str
    document: NormalizedDocument


@dataclass(frozen=True)
class Malformed:
    """Malformed input that could not be interpreted (Req 5.8).

    The document is rejected and no normalized representation is produced; any
    previously produced valid output is left untouched because :meth:`normalize`
    is a pure function that never mutates shared state.
    """

    error: str


# NormalizationResult = Normalized(NormalizedDocument) | Empty(reason) | Malformed(error)
NormalizationResult = Union[Normalized, Empty, Malformed]


__all__ = [
    "Normalized",
    "Empty",
    "Malformed",
    "NormalizationResult",
]
