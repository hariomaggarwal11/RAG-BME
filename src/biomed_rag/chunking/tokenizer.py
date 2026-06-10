"""The pluggable Tokenizer port for the Chunker (Req 6.1, 6.2).

Token counting drives every chunking bound: chunk sizes are measured in tokens
(Req 6.1) and overlap is measured in tokens (Req 6.2). The design (Chunker
section) specifies that "token counting uses a pluggable tokenizer consistent
with the embedding model family; it is injected for deterministic testing."

This module defines the :class:`Tokenizer` port and a deterministic
:class:`WhitespaceTokenizer` default. The whitespace tokenizer is intentionally
simple and reproducible so property and unit tests can reason exactly about
token counts and overlap without depending on a heavyweight subword tokenizer.

A tokenizer must satisfy one structural contract the Chunker relies on for the
completeness property (Req 6.5): for any list of ``tokens`` it produced,
``tokenize(detokenize(tokens))`` yields the same token list. The whitespace
tokenizer satisfies this because its tokens never contain whitespace.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Sequence


class Tokenizer(ABC):
    """Port for a pluggable, deterministic tokenizer (design: Chunker).

    Implementations split text into tokens and join tokens back into text. The
    Chunker measures all bounds in tokens, so the same injected tokenizer is
    used for counting, slicing, and reconstructing chunk content.
    """

    @abstractmethod
    def tokenize(self, text: str) -> List[str]:
        """Split ``text`` into an ordered list of tokens."""
        raise NotImplementedError

    @abstractmethod
    def detokenize(self, tokens: Sequence[str]) -> str:
        """Join ``tokens`` back into a text string.

        Must be a left-inverse of :meth:`tokenize` on token lists this
        tokenizer produced: ``tokenize(detokenize(toks)) == list(toks)``.
        """
        raise NotImplementedError

    def count(self, text: str) -> int:
        """Return the number of tokens in ``text`` (Req 6.1)."""
        return len(self.tokenize(text))


class WhitespaceTokenizer(Tokenizer):
    """A deterministic whitespace tokenizer used as the default and in tests.

    Tokens are maximal runs of non-whitespace characters; joining uses a single
    space. Because tokens never contain whitespace, ``tokenize`` and
    ``detokenize`` round-trip exactly on token lists this tokenizer produced.
    """

    def tokenize(self, text: str) -> List[str]:
        if not isinstance(text, str):
            raise TypeError("text must be a str")
        return text.split()

    def detokenize(self, tokens: Sequence[str]) -> str:
        return " ".join(tokens)
