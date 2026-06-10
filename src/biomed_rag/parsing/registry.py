"""Parsing engine registry — selects a :class:`ParsingEngine` from config (Req 2.2).

The registry maps each configurable :class:`~biomed_rag.config.ParsingEngine`
choice to a factory that builds the corresponding adapter. The Parser asks the
registry to ``select`` an engine from a :class:`~biomed_rag.config.PipelineConfig`,
keeping engine choice fully configuration-driven and free of hard-coded backends.

Concrete adapters (Docling, LlamaParse) register themselves with the
``default_registry`` in a later task; tests register the deterministic mock.
"""

from __future__ import annotations

from typing import Callable, Dict

from biomed_rag.config import ParsingEngine as ParsingEngineChoice
from biomed_rag.config import PipelineConfig

from .engine import ParsingEngine

# A factory builds a fresh engine instance on demand.
EngineFactory = Callable[[], ParsingEngine]


class ParsingEngineNotRegisteredError(KeyError):
    """Raised when no engine is registered for a requested configuration choice."""

    def __init__(self, choice: ParsingEngineChoice) -> None:
        self.choice = choice
        super().__init__(
            f"no parsing engine registered for {choice.value!r}; "
            "register an adapter before selecting it"
        )


class ParsingEngineRegistry:
    """A configuration-keyed registry of parsing engine factories."""

    def __init__(self) -> None:
        self._factories: Dict[ParsingEngineChoice, EngineFactory] = {}

    def register(
        self,
        choice: ParsingEngineChoice,
        factory: EngineFactory,
        *,
        replace: bool = False,
    ) -> None:
        """Register ``factory`` as the builder for ``choice``.

        Raises ``ValueError`` when ``choice`` is already registered unless
        ``replace`` is True (useful for tests swapping in the mock).
        """
        if not isinstance(choice, ParsingEngineChoice):
            raise TypeError(
                f"choice must be a config ParsingEngine, got {type(choice).__name__}"
            )
        if not callable(factory):
            raise TypeError("factory must be callable")
        if choice in self._factories and not replace:
            raise ValueError(
                f"a parsing engine is already registered for {choice.value!r}; "
                "pass replace=True to override"
            )
        self._factories[choice] = factory

    def is_registered(self, choice: ParsingEngineChoice) -> bool:
        """Return whether an engine is registered for ``choice``."""
        return choice in self._factories

    def create(self, choice: ParsingEngineChoice) -> ParsingEngine:
        """Build a fresh engine for ``choice``.

        Raises:
            ParsingEngineNotRegisteredError: nothing is registered for ``choice``.
        """
        try:
            factory = self._factories[choice]
        except KeyError:
            raise ParsingEngineNotRegisteredError(choice) from None
        return factory()

    def select(self, config: PipelineConfig) -> ParsingEngine:
        """Build the engine named by ``config.parsing_engine`` (Req 2.2)."""
        return self.create(config.parsing_engine)


# Process-wide default registry. Adapters register against this so a
# PipelineConfig built anywhere can select its engine.
default_registry = ParsingEngineRegistry()
