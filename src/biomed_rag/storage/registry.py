"""Vector store registry — selects a :class:`VectorStore` from config (Req 8.x).

The registry maps each configurable
:class:`~biomed_rag.config.VectorStoreBackend` choice to a factory that builds
the corresponding adapter, mirroring the parsing-engine registry. Consumers ask
the registry to ``select`` a backend from a
:class:`~biomed_rag.config.PipelineConfig`, keeping the backend choice fully
configuration-driven and free of hard-coded dependencies.

Concrete backend adapters (e.g. pgvector) register themselves with the
``default_registry`` in a later task. The in-memory adapter is a test fixture
and is registered explicitly by tests rather than by default.
"""

from __future__ import annotations

from typing import Callable, Dict

from biomed_rag.config import PipelineConfig
from biomed_rag.config import VectorStoreBackend as VectorStoreBackendChoice

from .port import VectorStore

# A factory builds a fresh store instance on demand.
VectorStoreFactory = Callable[[], VectorStore]


class VectorStoreNotRegisteredError(KeyError):
    """Raised when no store is registered for a requested configuration choice."""

    def __init__(self, choice: VectorStoreBackendChoice) -> None:
        self.choice = choice
        super().__init__(
            f"no vector store registered for {choice.value!r}; "
            "register an adapter before selecting it"
        )


class VectorStoreRegistry:
    """A configuration-keyed registry of vector store factories."""

    def __init__(self) -> None:
        self._factories: Dict[VectorStoreBackendChoice, VectorStoreFactory] = {}

    def register(
        self,
        choice: VectorStoreBackendChoice,
        factory: VectorStoreFactory,
        *,
        replace: bool = False,
    ) -> None:
        """Register ``factory`` as the builder for ``choice``.

        Raises ``ValueError`` when ``choice`` is already registered unless
        ``replace`` is True (useful for tests swapping in the in-memory adapter).
        """
        if not isinstance(choice, VectorStoreBackendChoice):
            raise TypeError(
                "choice must be a config VectorStoreBackend, got "
                f"{type(choice).__name__}"
            )
        if not callable(factory):
            raise TypeError("factory must be callable")
        if choice in self._factories and not replace:
            raise ValueError(
                f"a vector store is already registered for {choice.value!r}; "
                "pass replace=True to override"
            )
        self._factories[choice] = factory

    def is_registered(self, choice: VectorStoreBackendChoice) -> bool:
        """Return whether a store is registered for ``choice``."""
        return choice in self._factories

    def create(self, choice: VectorStoreBackendChoice) -> VectorStore:
        """Build a fresh store for ``choice``.

        Raises:
            VectorStoreNotRegisteredError: nothing is registered for ``choice``.
        """
        try:
            factory = self._factories[choice]
        except KeyError:
            raise VectorStoreNotRegisteredError(choice) from None
        return factory()

    def select(self, config: PipelineConfig) -> VectorStore:
        """Build the store named by ``config.vector_store_backend`` (Req 8.x)."""
        return self.create(config.vector_store_backend)


# Process-wide default registry. Adapters register against this so a
# PipelineConfig built anywhere can select its backend.
default_registry = VectorStoreRegistry()
