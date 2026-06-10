"""Unit tests for empty and malformed normalization (task 7.8).

Focused coverage of the two error arms of :meth:`Normalizer.normalize`:

* Req 5.7 -- an empty or no-content :class:`ParsedDocument` yields an *empty*
  :class:`NormalizedDocument` together with a non-empty "no content" indication.
* Req 5.8 -- a malformed input (something that is not an interpretable
  :class:`ParsedDocument`) is rejected with an error indication, no document is
  produced, and any previously produced valid output is left unchanged (purity).

These complement ``test_normalize_smoke.py``: that file establishes the happy
path and a couple of baseline empty/malformed checks; here we exercise the
distinct empty/malformed *inputs* (no-blocks vs all-artifact, the
missing/empty ``documentId`` rejection, several malformed input shapes) and
verify purity holds across repeated malformed calls.
"""

from __future__ import annotations

from biomed_rag.models.enums import BlockType
from biomed_rag.models.normalized import NormalizedDocument
from biomed_rag.models.parsed import ParsedDocument, TextBlock
from biomed_rag.normalization import Normalizer, WordSetDictionary
from biomed_rag.normalization.result import Empty, Malformed, Normalized


def _normalizer() -> Normalizer:
    # Deterministic, empty de-hyphenation dictionary: rejoins nothing.
    return Normalizer(WordSetDictionary())


def _block(text, page, pos, btype=BlockType.PARAGRAPH):
    return TextBlock(
        type=btype,
        text=text,
        pageNumber=page,
        readingOrderPosition=pos,
    )


# --------------------------------------------------------------------------
# Req 5.7 - empty / no-content input -> empty representation + indication
# --------------------------------------------------------------------------


def test_empty_parsed_document_returns_empty_with_nonempty_reason():
    """A ParsedDocument with no blocks (and no other content) is no-content."""
    parsed = ParsedDocument(documentId="doc-empty")

    result = _normalizer().normalize(parsed)

    assert isinstance(result, Empty)
    # The indication that no content was available must be present and non-empty.
    assert isinstance(result.reason, str) and result.reason.strip() != ""
    # An empty normalized representation is still produced...
    assert isinstance(result.document, NormalizedDocument)
    assert result.document.elements == []
    # ...and it carries the source document identifier through.
    assert result.document.documentId == "doc-empty"


def test_all_artifact_document_returns_empty_after_cleaning():
    """A document whose every block is a recurring header artifact normalizes to
    no content: cleaning strips all blocks, leaving an empty representation."""
    parsed = ParsedDocument(
        documentId="doc-all-artifacts",
        blocks=[
            _block("Recurring Header", 1, 0),
            _block("Recurring Header", 2, 0),
            _block("Recurring Header", 3, 0),
        ],
    )

    result = _normalizer().normalize(parsed)

    assert isinstance(result, Empty)
    assert isinstance(result.reason, str) and result.reason.strip() != ""
    assert result.document.elements == []
    assert result.document.documentId == "doc-all-artifacts"


# --------------------------------------------------------------------------
# Req 5.8 - malformed input -> rejected with an error, no document produced
# --------------------------------------------------------------------------


def test_none_input_is_malformed_with_error():
    result = _normalizer().normalize(None)

    assert isinstance(result, Malformed)
    assert isinstance(result.error, str) and result.error.strip() != ""


def test_string_input_is_malformed_with_error():
    result = _normalizer().normalize("not a parsed document")

    assert isinstance(result, Malformed)
    assert isinstance(result.error, str) and result.error.strip() != ""


def test_dict_input_is_malformed_with_error():
    # A plausible-looking but wrong-typed payload is still uninterpretable.
    result = _normalizer().normalize({"documentId": "x", "blocks": []})

    assert isinstance(result, Malformed)
    assert isinstance(result.error, str) and result.error.strip() != ""


def test_parsed_document_with_empty_document_id_is_malformed():
    """A ParsedDocument whose documentId has been corrupted to empty cannot be
    interpreted (no durable identity to attach to its normalized output).

    ``ParsedDocument`` forbids an empty ``documentId`` at construction, so we
    build a valid document and then corrupt the field to simulate a malformed
    structure reaching the Normalizer."""
    parsed = ParsedDocument(documentId="doc-x", blocks=[_block("Body.", 1, 0)])
    parsed.documentId = ""  # corrupt after construction

    result = _normalizer().normalize(parsed)

    assert isinstance(result, Malformed)
    assert isinstance(result.error, str) and result.error.strip() != ""


def test_parsed_document_with_none_document_id_is_malformed():
    parsed = ParsedDocument(documentId="doc-x", blocks=[_block("Body.", 1, 0)])
    parsed.documentId = None  # corrupt after construction

    result = _normalizer().normalize(parsed)

    assert isinstance(result, Malformed)
    assert isinstance(result.error, str) and result.error.strip() != ""


# --------------------------------------------------------------------------
# Req 5.8 - purity: a malformed call leaves prior valid output unchanged
# --------------------------------------------------------------------------


def test_malformed_call_leaves_prior_valid_output_unchanged():
    normalizer = _normalizer()

    good = ParsedDocument(
        documentId="doc-good",
        blocks=[_block("First body.", 1, 0), _block("Second body.", 1, 1)],
    )
    prior = normalizer.normalize(good)
    assert isinstance(prior, Normalized)

    # Snapshot the prior output's observable structure.
    prior_doc = prior.document
    prior_id = prior_doc.documentId
    prior_kinds = [(e.kind, e.pageNumber, e.readingOrderPosition) for e in prior_doc.elements]
    prior_texts = [e.payload.text for e in prior_doc.elements]

    # A sequence of malformed inputs must not disturb the prior valid output.
    for bad in (None, "garbage", 42, {"blocks": []}):
        rejected = normalizer.normalize(bad)
        assert isinstance(rejected, Malformed)

    assert prior_doc.documentId == prior_id
    assert [(e.kind, e.pageNumber, e.readingOrderPosition) for e in prior_doc.elements] == prior_kinds
    assert [e.payload.text for e in prior_doc.elements] == prior_texts


def test_malformed_then_valid_still_normalizes():
    """A malformed call does not poison the normalizer; a subsequent valid call
    succeeds normally (normalize is pure and stateless)."""
    normalizer = _normalizer()

    assert isinstance(normalizer.normalize(None), Malformed)

    good = ParsedDocument(documentId="doc-after", blocks=[_block("Body.", 1, 0)])
    result = normalizer.normalize(good)

    assert isinstance(result, Normalized)
    assert result.document.documentId == "doc-after"
    assert [e.payload.text for e in result.document.elements] == ["Body."]
