"""Recurring header/footer artifact removal (Req 5.1).

A running header, a running footer, or a page number recurs in the top or bottom
region of many pages. The Normalizer removes any text element that recurs in the
header or footer region on two or more pages, including page numbers (Req 5.1).

The :class:`ParsedDocument` model carries no geometric coordinates, so the
"header or footer region" is approximated structurally: the first ``region_size``
blocks of a page (in reading order) form its header region and the last
``region_size`` blocks form its footer region. A block in one of those regions is
an artifact when its signature recurs on at least ``min_recurring_pages`` distinct
pages.

Two signatures are computed per candidate block:

* an **exact** signature (whitespace-collapsed text) catches running heads/feet
  whose text is identical on every page (e.g. a journal title);
* a **page-number template** signature replaces each run of digits with a
  placeholder so that ``"Page 1"``, ``"Page 2"`` ... collapse to one signature,
  catching page numbers whose only variation is the number itself (Req 5.1).
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Dict, List, Set

from biomed_rag.models.parsed import TextBlock

# Maximal runs of digits are replaced by this placeholder when building a
# page-number template signature.
_DIGIT_RUN = re.compile(r"\d+")
_NUM_PLACEHOLDER = "\x00N\x00"
_WHITESPACE = re.compile(r"\s+")


def _exact_signature(text: str) -> str:
    """Whitespace-collapsed, stripped form used to compare two texts for equality."""
    return _WHITESPACE.sub(" ", text).strip()


def _template_signature(exact: str) -> str:
    """Replace digit runs in ``exact`` with a placeholder (page-number template)."""
    return _DIGIT_RUN.sub(_NUM_PLACEHOLDER, exact)


def _has_number(exact: str) -> bool:
    return bool(_DIGIT_RUN.search(exact))


class HeaderFooterArtifactRemover:
    """Removes text blocks recurring in the header/footer region (Req 5.1).

    Parameters
    ----------
    min_recurring_pages:
        A signature must appear on at least this many *distinct* pages within the
        header/footer region to be treated as an artifact. The requirement is
        "2 or more pages", so the default is 2.
    region_size:
        Number of leading blocks (header) and trailing blocks (footer) per page
        considered part of the header/footer region.
    """

    def __init__(self, *, min_recurring_pages: int = 2, region_size: int = 1) -> None:
        if min_recurring_pages < 2:
            raise ValueError("min_recurring_pages must be >= 2 (Req 5.1)")
        if region_size < 1:
            raise ValueError("region_size must be >= 1")
        self.min_recurring_pages = min_recurring_pages
        self.region_size = region_size

    # -- region identification -------------------------------------------
    def _candidate_indices(self, blocks: List[TextBlock]) -> Set[int]:
        """Indices of blocks lying in any page's header or footer region.

        Blocks are grouped by page and ordered by reading-order position; the
        first and last ``region_size`` of each page are candidates.
        """
        by_page: Dict[int, List[int]] = defaultdict(list)
        for idx, block in enumerate(blocks):
            by_page[block.pageNumber].append(idx)

        candidates: Set[int] = set()
        for indices in by_page.values():
            ordered = sorted(indices, key=lambda i: blocks[i].readingOrderPosition)
            candidates.update(ordered[: self.region_size])
            candidates.update(ordered[-self.region_size :])
        return candidates

    # -- artifact detection ----------------------------------------------
    def _artifact_signatures(
        self, blocks: List[TextBlock], candidates: Set[int]
    ) -> tuple[Set[str], Set[str]]:
        """Return ``(exact_artifacts, template_artifacts)``.

        A signature is an artifact when it occurs on ``>= min_recurring_pages``
        distinct pages among the candidate (header/footer) blocks. Template
        signatures are only considered when they actually contain a number, so a
        genuinely identical running head is handled by the exact set and not
        double-counted.
        """
        exact_pages: Dict[str, Set[int]] = defaultdict(set)
        template_pages: Dict[str, Set[int]] = defaultdict(set)

        for idx in candidates:
            block = blocks[idx]
            exact = _exact_signature(block.text)
            if not exact:
                continue
            exact_pages[exact].add(block.pageNumber)
            if _has_number(exact):
                template_pages[_template_signature(exact)].add(block.pageNumber)

        exact_artifacts = {
            sig
            for sig, pages in exact_pages.items()
            if len(pages) >= self.min_recurring_pages
        }
        template_artifacts = {
            sig
            for sig, pages in template_pages.items()
            if len(pages) >= self.min_recurring_pages
        }
        return exact_artifacts, template_artifacts

    def _is_artifact(
        self,
        block: TextBlock,
        exact_artifacts: Set[str],
        template_artifacts: Set[str],
    ) -> bool:
        exact = _exact_signature(block.text)
        if not exact:
            return False
        if exact in exact_artifacts:
            return True
        if _has_number(exact) and _template_signature(exact) in template_artifacts:
            return True
        return False

    # -- public API -------------------------------------------------------
    def remove(self, blocks: List[TextBlock]) -> List[TextBlock]:
        """Return ``blocks`` with recurring header/footer artifacts removed (Req 5.1).

        Only blocks in a page's header/footer region are eligible for removal, so
        body text that coincidentally matches an artifact signature is retained.
        """
        if not blocks:
            return list(blocks)

        candidates = self._candidate_indices(blocks)
        exact_artifacts, template_artifacts = self._artifact_signatures(
            blocks, candidates
        )
        if not exact_artifacts and not template_artifacts:
            return list(blocks)

        kept: List[TextBlock] = []
        for idx, block in enumerate(blocks):
            if idx in candidates and self._is_artifact(
                block, exact_artifacts, template_artifacts
            ):
                continue
            kept.append(block)
        return kept
