"""Property test for recurring header/footer artifact removal (Req 5.1).

Feature: biomedical-rag-pipeline, Property 9: Normalization removes recurring header/footer artifacts

Statement: for any Parsed_Document in which a text element recurs in the header or
footer region on two or more pages (including page numbers), the normalized
representation contains no occurrence of that recurring element, while distinct
body text is retained.

Construction notes
------------------
``HeaderFooterArtifactRemover`` (default ``region_size=1``) treats the first block
of a page (by reading order) as its header region and the last block as its footer
region. To guarantee the injected running element lands in the region, each page is
built as ``[header?, body+, footer?]`` with reading-order positions assigned in a
single increasing global sequence, so the header is always first and the footer is
always last on its page.

Two kinds of running element are exercised, matching the two signatures the remover
computes (Req 5.1):

* ``constant`` - identical text on every page (caught by the exact signature);
* ``page_number`` - a fixed prefix followed by the page's number, so only the digits
  vary (caught by the page-number template signature).

Body blocks are generated as digit-free, globally unique tokens, so they can never
form a recurring signature themselves (even when a body lands in the region for a
header-only or footer-only page) and can never collide with the injected element.
"""

from __future__ import annotations

from typing import List, Tuple

from hypothesis import given, settings
from hypothesis import strategies as st

from biomed_rag.models.enums import BlockType
from biomed_rag.models.parsed import TextBlock
from biomed_rag.normalization import HeaderFooterArtifactRemover

# Letters only: lets us build digit-free, collision-free body tokens and constant
# running-head text whose whitespace-collapsed signature is always non-empty.
_LETTERS = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ",
    min_size=1,
    max_size=8,
)


def _letters_for(n: int) -> str:
    """Map a non-negative int to a unique, digit-free lowercase-letter token."""
    n += 1
    out = ""
    while n:
        n, rem = divmod(n - 1, 26)
        out = chr(ord("a") + rem) + out
    return out


@st.composite
def _documents(draw) -> Tuple[List[TextBlock], List[str], List[str]]:
    """Generate a multi-page block sequence with an injected running element.

    Returns ``(blocks, body_texts, recurring_texts)`` where ``body_texts`` is the
    in-order list of body block texts (which must survive removal) and
    ``recurring_texts`` is the set of injected header/footer strings (which must be
    removed).
    """
    num_pages = draw(st.integers(min_value=2, max_value=5))
    placement = draw(st.sampled_from(("header", "footer", "both")))
    kind = draw(st.sampled_from(("constant", "page_number")))

    inject_header = placement in ("header", "both")
    inject_footer = placement in ("footer", "both")

    # The running element. For ``constant`` every page shares one string; for
    # ``page_number`` a fixed (possibly empty) prefix is followed by the page number.
    const_text = draw(_LETTERS)
    prefix = draw(st.sampled_from(("", "Page ", "p.")))

    def running_text(page: int) -> str:
        if kind == "constant":
            return const_text
        return f"{prefix}{page}"

    blocks: List[TextBlock] = []
    body_texts: List[str] = []
    recurring_texts: List[str] = []
    pos = 0  # single increasing reading-order sequence -> header first, footer last
    body_counter = 0

    for page in range(1, num_pages + 1):
        if inject_header:
            text = running_text(page)
            recurring_texts.append(text)
            blocks.append(
                TextBlock(
                    type=BlockType.PARAGRAPH,
                    text=text,
                    pageNumber=page,
                    readingOrderPosition=pos,
                )
            )
            pos += 1

        for _ in range(draw(st.integers(min_value=1, max_value=3))):
            body = f"body-{_letters_for(body_counter)}"
            body_counter += 1
            body_texts.append(body)
            blocks.append(
                TextBlock(
                    type=BlockType.PARAGRAPH,
                    text=body,
                    pageNumber=page,
                    readingOrderPosition=pos,
                )
            )
            pos += 1

        if inject_footer:
            text = running_text(page)
            recurring_texts.append(text)
            blocks.append(
                TextBlock(
                    type=BlockType.PARAGRAPH,
                    text=text,
                    pageNumber=page,
                    readingOrderPosition=pos,
                )
            )
            pos += 1

    return blocks, body_texts, recurring_texts


# Feature: biomedical-rag-pipeline, Property 9: Normalization removes recurring header/footer artifacts
@settings(max_examples=200, deadline=None)
@given(document=_documents())
def test_recurring_header_footer_artifacts_are_removed(
    document: Tuple[List[TextBlock], List[str], List[str]],
) -> None:
    """Validates: Requirements 5.1"""
    blocks, body_texts, recurring_texts = document

    kept = HeaderFooterArtifactRemover().remove(blocks)
    kept_texts = [b.text for b in kept]

    # Req 5.1: the recurring element (running head/foot or page number) that recurs
    # in the header/footer region on >= 2 pages has no occurrence in the result.
    for recurring in set(recurring_texts):
        assert recurring not in kept_texts

    # Distinct body text is retained in its original order, and nothing else is kept:
    # every removed block was an injected recurring artifact.
    assert kept_texts == body_texts
