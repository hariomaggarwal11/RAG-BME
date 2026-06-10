"""Normalizer component for the biomedical RAG pipeline (Req 5).

Task 7.1 implements the two artifact-cleaning transformations:

* recurring header/footer (and page-number) removal (Req 5.1), via
  :class:`HeaderFooterArtifactRemover`; and
* line-break de-hyphenation against an injected :class:`Dictionary` while
  retaining intrinsic mid-line hyphens (Req 5.2, 5.3), via
  :func:`dehyphenate_text` / :func:`dehyphenate_blocks`.

The :class:`Normalizer` ties these together. Task 7.2 adds canonical
``NormalizedDocument`` production with empty/malformed handling, returning a
:data:`NormalizationResult` (``Normalized`` | ``Empty`` | ``Malformed``); task
7.3 adds the durable :func:`serialize` / :func:`deserialize` pair underpinning
the round-trip property (Req 5.6).
"""

from __future__ import annotations

from .artifacts import HeaderFooterArtifactRemover
from .dehyphenation import dehyphenate_blocks, dehyphenate_text
from .dictionary import Dictionary, WordSetDictionary
from .normalizer import Normalizer
from .result import Empty, Malformed, NormalizationResult, Normalized
from .serialization import DeserializationError, deserialize, serialize

__all__ = [
    "Normalizer",
    "Dictionary",
    "WordSetDictionary",
    "HeaderFooterArtifactRemover",
    "dehyphenate_text",
    "dehyphenate_blocks",
    "NormalizationResult",
    "Normalized",
    "Empty",
    "Malformed",
    "serialize",
    "deserialize",
    "DeserializationError",
]
