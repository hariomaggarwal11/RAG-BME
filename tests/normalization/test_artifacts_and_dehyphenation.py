"""Unit tests for Normalizer artifact removal and de-hyphenation (task 7.1).

Covers Req 5.1 (recurring header/footer + page-number removal) and Req 5.2/5.3
(dictionary-respecting line-break de-hyphenation with intrinsic hyphens retained).
"""

from __future__ import annotations

from biomed_rag.models.enums import BlockSource, BlockType
from biomed_rag.models.parsed import ParsedDocument, TextBlock
from biomed_rag.normalization import (
    HeaderFooterArtifactRemover,
    Normalizer,
    WordSetDictionary,
    dehyphenate_text,
)


def _block(text, page, pos, btype=BlockType.PARAGRAPH):
    return TextBlock(type=btype, text=text, pageNumber=page, readingOrderPosition=pos)


# --------------------------------------------------------------------------
# Req 5.1 - recurring header/footer artifact removal
# --------------------------------------------------------------------------


def test_identical_running_header_recurring_on_two_pages_is_removed():
    blocks = [
        _block("Journal of Biomedical Engineering", 1, 0),
        _block("Body text page one.", 1, 1),
        _block("Journal of Biomedical Engineering", 2, 2),
        _block("Body text page two.", 2, 3),
    ]
    kept = HeaderFooterArtifactRemover().remove(blocks)
    texts = [b.text for b in kept]
    assert "Journal of Biomedical Engineering" not in texts
    assert texts == ["Body text page one.", "Body text page two."]


def test_page_numbers_with_varying_digits_are_removed_as_artifacts():
    blocks = [
        _block("Intro.", 1, 0),
        _block("Page 1", 1, 1),
        _block("More.", 2, 2),
        _block("Page 2", 2, 3),
        _block("End.", 3, 4),
        _block("Page 3", 3, 5),
    ]
    kept = HeaderFooterArtifactRemover().remove(blocks)
    assert [b.text for b in kept] == ["Intro.", "More.", "End."]


def test_bare_numeric_page_numbers_are_removed():
    blocks = [
        _block("alpha", 1, 0),
        _block("1", 1, 1),
        _block("beta", 2, 2),
        _block("2", 2, 3),
    ]
    kept = HeaderFooterArtifactRemover().remove(blocks)
    assert [b.text for b in kept] == ["alpha", "beta"]


def test_element_recurring_on_only_one_page_is_retained():
    blocks = [
        _block("Header A", 1, 0),
        _block("Body one.", 1, 1),
        _block("Body two.", 2, 2),
        _block("Footer B", 2, 3),
    ]
    kept = HeaderFooterArtifactRemover().remove(blocks)
    # Neither header nor footer recurs across >= 2 pages, so nothing is removed.
    assert len(kept) == 4


def test_body_text_matching_header_is_not_removed_outside_region():
    # "Repeat" recurs in the header region on pages 1 and 2 (artifact), but also
    # appears as a body block on page 1 which must be preserved.
    blocks = [
        _block("Repeat", 1, 0),  # header region -> removed
        _block("Repeat", 1, 1),  # body -> kept
        _block("middle", 1, 2),
        _block("Repeat", 2, 3),  # header region -> removed
        _block("tail", 2, 4),
    ]
    kept = HeaderFooterArtifactRemover().remove(blocks)
    kept_texts = [b.text for b in kept]
    assert kept_texts.count("Repeat") == 1
    assert kept_texts == ["Repeat", "middle", "tail"]


def test_whitespace_differences_do_not_prevent_artifact_detection():
    blocks = [
        _block("Running   Head", 1, 0),
        _block("x", 1, 1),
        _block("Running Head", 2, 2),
        _block("y", 2, 3),
    ]
    kept = HeaderFooterArtifactRemover().remove(blocks)
    assert [b.text for b in kept] == ["x", "y"]


def test_remove_preserves_block_order_and_metadata():
    blocks = [
        _block("Title", 1, 0),
        _block("keep me", 1, 1, btype=BlockType.HEADING),
        _block("Title", 2, 2),
        _block("also keep", 2, 3),
    ]
    kept = HeaderFooterArtifactRemover().remove(blocks)
    assert kept[0].type is BlockType.HEADING
    assert [b.readingOrderPosition for b in kept] == [1, 3]


# --------------------------------------------------------------------------
# Req 5.2 / 5.3 - line-break de-hyphenation
# --------------------------------------------------------------------------


def test_line_break_hyphen_joined_when_in_dictionary():
    d = WordSetDictionary(["inflammation"])
    assert dehyphenate_text("chronic inflamma-\ntion persists", d) == (
        "chronic inflammation persists"
    )


def test_line_break_hyphen_retained_when_not_in_dictionary():
    d = WordSetDictionary(["something-else"])
    text = "ortho-\npedic care"
    # "orthopedic" is not in the dictionary -> retain original unchanged (Req 5.3).
    assert dehyphenate_text(text, d) == text


def test_intrinsic_mid_line_hyphen_is_unchanged():
    d = WordSetDictionary(["wellbeing", "well-being"])
    text = "patient well-being improved"
    assert dehyphenate_text(text, d) == text


def test_crlf_line_break_hyphen_is_joined():
    d = WordSetDictionary(["biomedical"])
    assert dehyphenate_text("bio-\r\nmedical", d) == "biomedical"


def test_dehyphenation_is_case_insensitive_and_preserves_case():
    d = WordSetDictionary(["inflammation"])
    assert dehyphenate_text("Inflamma-\ntion", d) == "Inflammation"


def test_text_without_hyphen_is_returned_unchanged():
    d = WordSetDictionary(["anything"])
    assert dehyphenate_text("no hyphens here", d) == "no hyphens here"


# --------------------------------------------------------------------------
# Normalizer integration
# --------------------------------------------------------------------------


def test_normalizer_clean_blocks_removes_artifacts_then_dehyphenates():
    d = WordSetDictionary(["inflammation"])
    normalizer = Normalizer(d)
    blocks = [
        _block("Running Head", 1, 0),
        _block("chronic inflamma-\ntion", 1, 1),
        _block("Running Head", 2, 2),
        _block("more body", 2, 3),
    ]
    cleaned = normalizer.clean_blocks(blocks)
    texts = [b.text for b in cleaned]
    assert texts == ["chronic inflammation", "more body"]


def test_normalizer_clean_parsed_document_carries_through_non_text():
    d = WordSetDictionary([])
    normalizer = Normalizer(d)
    parsed = ParsedDocument(
        documentId="doc-1",
        blocks=[
            _block("Foot", 1, 0),
            _block("body one", 1, 1),
            _block("Foot", 2, 2),
            _block("body two", 2, 3),
        ],
    )
    cleaned = normalizer.clean_parsed_document(parsed)
    assert cleaned.documentId == "doc-1"
    assert [b.text for b in cleaned.blocks] == ["body one", "body two"]
    assert cleaned.tables == [] and cleaned.figures == []


def test_smoke_end_to_end_minimal():
    """Quick smoke test exercising the public Normalizer surface."""
    normalizer = Normalizer(WordSetDictionary(["proteins"]))
    blocks = [
        _block("Page 1", 1, 0),
        _block("Membrane pro-\nteins fold.", 1, 1),
        _block("Page 2", 2, 2),
        _block("Final.", 2, 3),
    ]
    cleaned = normalizer.clean_blocks(blocks)
    assert [b.text for b in cleaned] == ["Membrane proteins fold.", "Final."]
