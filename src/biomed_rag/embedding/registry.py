"""Embedding model registry — selects an :class:`EmbeddingModel` from config
(Req 7.3).

The registry maps each registered model id (a string, matching
:attr:`~biomed_rag.config.PipelineConfig.embedding_model`) to a factory that
builds the corresponding adapter. The Embedder asks the registry to ``select``
a model from a :class:`~biomed_rag.config.PipelineConfig`, keeping model choice
fully configuration-driven and free of hard-coded backends.

Concrete adapters register themselves with the ``default_registry`` in a later
task; tests register the deterministic mock.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

from biomed_rag.config import PipelineConfig

from .model import EmbeddingModel

# A factory builds a fresh model instance on demand.
ModelFactory = Callable[[], EmbeddingModel]


class EmbeddingModelNotRegisteredError(KeyError):
    """Raised when no model is registered for a requested model id (Req 7.3, 7.4).

    Also raised when selection is attempted but no model id is configured
    (``embedding_model`` is ``None``), since that can never name a registered
    model.
    """

    def __init__(self, model_id: Optional[str]) -> None:
        self.model_id = model_id
        if model_id is None:
            message = (
                "no embedding model configured (embedding_model is None); "
                "set a registered model id before selecting a model"
            )
        else:
            message = (
                f"no embedding model registered for {model_id!r}; "
                "register an adapter before selecting it"
            )
        super().__init__(message)


class EmbeddingModelRegistry:
    """A model-id-keyed registry of embedding model factories (Req 7.3)."""

    def __init__(self) -> None:
        self._factories: Dict[str, ModelFactory] = {}

    def register(
        self,
        model_id: str,
        factory: ModelFactory,
        *,
        replace: bool = False,
    ) -> None:
        """Register ``factory`` as the builder for ``model_id``.

        Raises ``ValueError`` when ``model_id`` is already registered unless
        ``replace`` is True (useful for tests swapping in the mock).
        """
        if not isinstance(model_id, str) or not model_id.strip():
            raise TypeError("model_id must be a non-empty string")
        if not callable(factory):
            raise TypeError("factory must be callable")
        if model_id in self._factories and not replace:
            raise ValueError(
                f"an embedding model is already registered for {model_id!r}; "
                "pass replace=True to override"
            )
        self._factories[model_id] = factory

    def is_registered(self, model_id: Optional[str]) -> bool:
        """Return whether a model is registered under ``model_id``."""
        return model_id in self._factories

    def create(self, model_id: Optional[str]) -> EmbeddingModel:
        """Build a fresh model for ``model_id``.

        Raises:
            EmbeddingModelNotRegisteredError: nothing is registered for
                ``model_id`` (including when ``model_id`` is ``None``)
                (Req 7.3, 7.4).
        """
        try:
            factory = self._factories[model_id]  # type: ignore[index]
        except KeyError:
            raise EmbeddingModelNotRegisteredError(model_id) from None
        return factory()

    def select(self, config: PipelineConfig) -> EmbeddingModel:
        """Build the model named by ``config.embedding_model`` (Req 7.3)."""
        return self.create(config.embedding_model)


# Process-wide default registry. Adapters register against this so a
# PipelineConfig built anywhere can select its model.
default_registry = EmbeddingModelRegistry()
