"""Line-break de-hyphenation (Req 5.2, 5.3).

When a word is split across a line break by a trailing hyphen (``"inflamma-\\n
tion"``) and the joined form (``"inflammation"``) is a known dictionary word, the
Normalizer rejoins it into a single token without the hyphen (Req 5.2). Any
hyphen that is *not* at a line break -- an intrinsic, mid-line hyphen such as in
``"well-being"`` -- is retained unchanged, as is a line-break hyphen whose joined
form is not a dictionary word (Req 5.3).

The decision is delegated to an injected :class:`~biomed_rag.normalization.dictionary.Dictionary`
so the behavior is deterministic and testable.
"""

from __future__ import annotations

import re
from typing import List

from biomed_rag.models.parsed import TextBlock

from .dictionary import Dictionary

# A line-break hyphenation is: word characters, a hyphen, then a line break
# (CRLF or LF) optionally followed by horizontal whitespace, then the
# continuation word. Only ``-`` immediately preceding the newline is a
# line-break hyphen; a hyphen anywhere else on the line is intrinsic and is left
# untouched because this pattern never matches it.
_LINE_BREAK_HYPHEN = re.compile(r"(\w+)-\r?\n[ \t]*(\w+)")


def dehyphenate_text(text: str, dictionary: Dictionary) -> str:
    """Rejoin line-break-hyphenated words whose joined form is in ``dictionary``.

    For every occurrence of ``left-<line break>right``:

    * if ``left + right`` is a known dictionary word, the span is replaced by the
      single joined token ``left + right`` (hyphen and line break removed) (Req 5.2);
    * otherwise the span is left exactly as it was, preserving the hyphen (Req 5.3).

    Intrinsic mid-line hyphens never match the line-break pattern and are
    therefore returned unchanged (Req 5.3).
    """
    if not isinstance(text, str) or "-" not in text:
        return text

    def _join(match: "re.Match[str]") -> str:
        left, right = match.group(1), match.group(2)
        candidate = left + right
        if dictionary.contains(candidate):
            return candidate
        # Not a known word: retain the original token unchanged (Req 5.3).
        return match.group(0)

    return _LINE_BREAK_HYPHEN.sub(_join, text)


def dehyphenate_blocks(
    blocks: List[TextBlock], dictionary: Dictionary
) -> List[TextBlock]:
    """Return copies of ``blocks`` with each block's text de-hyphenated.

    Every other field (type, page number, reading-order position, OCR metadata)
    is preserved exactly; only the text is rewritten according to
    :func:`dehyphenate_text`.
    """
    result: List[TextBlock] = []
    for block in blocks:
        new_text = dehyphenate_text(block.text, dictionary)
        if new_text == block.text:
            result.append(block)
        else:
            result.append(
                TextBlock(
                    type=block.type,
                    text=new_text,
                    pageNumber=block.pageNumber,
                    readingOrderPosition=block.readingOrderPosition,
                    source=block.source,
                    ocrConfidence=block.ocrConfidence,
                    lowConfidence=block.lowConfidence,
                    headingLevel=block.headingLevel,
                )
            )
    return result
