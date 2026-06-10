"""Injectable dictionary used by the Normalizer's de-hyphenation decision (Req 5.2).

The de-hyphenation rule (rejoin a line-break-hyphenated word only when the joined
token is a known word) depends on a word list. Per the design (Normalizer
section):

    A configurable dictionary (biomedical-aware word list) backs the
    de-hyphenation decision. It is injected so tests can supply a deterministic
    dictionary.

This module defines the :class:`Dictionary` port and a simple in-memory
implementation, :class:`WordSetDictionary`, that tests can construct from a
deterministic word list. Membership is case-insensitive: real-world line breaks
capitalize the leading fragment at sentence starts, so ``"Inflamma"`` + ``"tion"``
must match the dictionary entry ``"inflammation"``.
"""

from __future__ import annotations

from typing import Iterable, Protocol, runtime_checkable


@runtime_checkable
class Dictionary(Protocol):
    """A word-membership oracle for the de-hyphenation decision (Req 5.2).

    Implementations decide whether a candidate joined token is a known word.
    Membership must be case-insensitive.
    """

    def contains(self, word: str) -> bool:
        """Return ``True`` when ``word`` is a known dictionary word."""
        ...


class WordSetDictionary:
    """A deterministic, case-insensitive :class:`Dictionary` backed by a set.

    Words are normalized to lowercase on construction so lookups ignore case.
    This is the implementation injected by tests to supply a known word list,
    and is also a perfectly serviceable production dictionary when seeded from a
    biomedical word list.
    """

    __slots__ = ("_words",)

    def __init__(self, words: Iterable[str] = ()) -> None:
        # Normalize once at construction; ignore empty/whitespace-only entries.
        self._words = frozenset(
            w.strip().lower() for w in words if isinstance(w, str) and w.strip()
        )

    def contains(self, word: str) -> bool:
        """Return ``True`` when ``word`` (compared case-insensitively) is known."""
        if not isinstance(word, str):
            return False
        return word.strip().lower() in self._words

    def __contains__(self, word: object) -> bool:
        return isinstance(word, str) and self.contains(word)

    def __len__(self) -> int:
        return len(self._words)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"WordSetDictionary({sorted(self._words)!r})"
