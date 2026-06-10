"""A deterministic mock :class:`EmbeddingModel` for tests (Req 7.3).

The mock exercises the Embedder and downstream stages through the port without
a real embedding backend. It is fully deterministic: the same ``text`` always
yields the same vector, and the vector always has the configured
:meth:`dimension`. Test-only knobs let callers simulate the failure conditions
the Embedder must handle:

* ``raise_on_embed=...`` → :meth:`embed` raises the given exception, used to
  simulate embed errors (Req 7.6) and timeouts (Req 7.2).

With no knobs set, :meth:`embed` derives a deterministic vector from a SHA-256
hash of the model id and the text, so different models and different texts
produce different (but reproducible across processes) vectors.
"""

from __future__ import annotations

import hashlib
import random
from typing import List, Optional

from .model import EmbeddingModel


class MockEmbeddingModel(EmbeddingModel):
    """Deterministic in-memory embedding model for property and unit tests."""

    def __init__(
        self,
        *,
        model_id: str = "mock-emb",
        dimension: int = 8,
        raise_on_embed: Optional[BaseException] = None,
    ) -> None:
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError("model_id must be a non-empty string")
        if not isinstance(dimension, int) or isinstance(dimension, bool):
            raise TypeError("dimension must be an int")
        if dimension < 1:
            raise ValueError("dimension must be >= 1")
        self._model_id = model_id
        self._dimension = dimension
        self._raise_on_embed = raise_on_embed

    # -- port methods -----------------------------------------------------
    def model_id(self) -> str:
        return self._model_id

    def dimension(self) -> int:
        return self._dimension

    def embed(
        self,
        text: str,
        deadline: Optional[float] = None,
    ) -> List[float]:
        if not isinstance(text, str):
            raise TypeError("text must be a str")

        if self._raise_on_embed is not None:
            raise self._raise_on_embed

        return self._derive_vector(text)

    # -- deterministic derivation ----------------------------------------
    def _derive_vector(self, text: str) -> List[float]:
        """Build a stable unit-range vector from the model id and text.

        Seeds a private PRNG with a SHA-256 digest of ``model_id`` + ``text``
        so the derivation is pure and reproducible across processes (unlike the
        salted built-in :func:`hash`). Identical inputs always produce an
        identical vector of length :meth:`dimension`.
        """
        digest = hashlib.sha256(
            f"{self._model_id}\x00{text}".encode("utf-8")
        ).digest()
        seed = int.from_bytes(digest[:8], "big")
        rng = random.Random(seed)
        return [rng.uniform(-1.0, 1.0) for _ in range(self._dimension)]
