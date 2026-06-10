"""Property test for line-break de-hyphenation (Task 7.5, Req 5.2, 5.3).

Feature: biomedical-rag-pipeline, Property 10: Line-break de-hyphenation respects the dictionary

Statement: for any token split across a line break by a trailing hyphen whose
joined form is a known dictionary word, the Normalizer outputs the single joined
token without the hyphen (Req 5.2); for any line-break-hyphenated token whose
joined form is *not* in the dictionary the original span (hyphen + line break) is
retained unchanged; and for any intrinsic (mid-line) hyphen, the original token
is retained unchanged regardless of the dictionary (Req 5.3).

Two word fragments ``left``/``right`` are generated from word characters so the
line-break pattern (``\\w+-\\r?\\n\\w+``) matches. A deterministic
``WordSetDictionary`` is built to either contain or exclude the joined form, and
the de-hyphenation outcome is asserted for the line-break and intrinsic cases.
"""

from __future__ import annotations

import string

from hypothesis import given, settings
from hypothesis import strategies as st

from biomed_rag.normalization import WordSetDictionary, dehyphenate_text

# Fragments are word characters (mixed case to also exercise case-insensitive
# membership). They never contain a newline, so the only line break in the text
# is the one we insert explicitly.
_FRAGMENT = st.text(alphabet=string.ascii_letters, min_size=1, max_size=8)
# Line breaks may be LF or CRLF; both are recognized by the de-hyphenation rule.
_LINE_BREAK = st.sampled_from(["\n", "\r\n"])


# Feature: biomedical-rag-pipeline, Property 10: Line-break de-hyphenation respects the dictionary
# Validates: Requirements 5.2, 5.3
@settings(max_examples=200)
@given(left=_FRAGMENT, right=_FRAGMENT, line_break=_LINE_BREAK)
def test_line_break_hyphen_joined_when_joined_form_in_dictionary(left, right, line_break):
    joined = left + right
    dictionary = WordSetDictionary([joined])
    text = f"{left}-{line_break}{right}"
    # The joined form is a known word -> collapse to the single token (Req 5.2).
    assert dehyphenate_text(text, dictionary) == joined


# Feature: biomedical-rag-pipeline, Property 10: Line-break de-hyphenation respects the dictionary
# Validates: Requirements 5.2, 5.3
@settings(max_examples=200)
@given(
    left=_FRAGMENT,
    right=_FRAGMENT,
    line_break=_LINE_BREAK,
    other_words=st.lists(_FRAGMENT, max_size=5),
)
def test_line_break_hyphen_retained_when_joined_form_not_in_dictionary(
    left, right, line_break, other_words
):
    joined = left + right
    # Seed the dictionary with unrelated words, guaranteeing the joined form
    # (compared case-insensitively, as WordSetDictionary does) is absent.
    words = [w for w in other_words if w.lower() != joined.lower()]
    dictionary = WordSetDictionary(words)
    text = f"{left}-{line_break}{right}"
    # Not a known word -> the original span is retained unchanged (Req 5.3).
    assert dehyphenate_text(text, dictionary) == text


# Feature: biomedical-rag-pipeline, Property 10: Line-break de-hyphenation respects the dictionary
# Validates: Requirements 5.2, 5.3
@settings(max_examples=200)
@given(left=_FRAGMENT, right=_FRAGMENT)
def test_intrinsic_mid_line_hyphen_is_always_retained(left, right):
    joined = left + right
    # Even when the joined form is a known word, an intrinsic mid-line hyphen
    # (no line break) is never a line-break hyphenation and must be retained
    # exactly as written (Req 5.3).
    dictionary = WordSetDictionary([joined])
    text = f"{left}-{right}"
    assert dehyphenate_text(text, dictionary) == text
