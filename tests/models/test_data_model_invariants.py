"""Unit tests for shared data model invariants (Task 3.2).

Covers construction guards and default/empty-value handling for:
* ``Chunk`` metadata        - Requirements 6.3, 6.7
* ``Cell`` spans            - Requirement 3.2
* ``Embedding`` status/attempts - Requirement 7.7

These tests exercise the field-level invariants enforced in
``__post_init__`` so that an invalid model can never exist in memory.
"""

from __future__ import annotations

import pytest

from biomed_rag.models._validation import ModelValidationError
from biomed_rag.models.chunk import Chunk, Embedding
from biomed_rag.models.enums import EmbeddingStatus
from biomed_rag.models.parsed import Cell


# ---------------------------------------------------------------------------
# Chunk metadata (Req 6.3, 6.7)
# ---------------------------------------------------------------------------
class TestChunkMetadataDefaults:
    """Default/empty-value handling for Chunk source metadata."""

    def test_minimal_chunk_uses_empty_defaults(self) -> None:
        chunk = Chunk(
            documentId="doc-1",
            content="hello world",
            tokenCount=2,
            orderIndex=0,
        )
        # documentId is always present.
        assert chunk.documentId == "doc-1"
        # page number defaults to None (empty/unavailable) (Req 6.7).
        assert chunk.pageNumber is None
        # heading path defaults to an empty list (Req 6.7).
        assert chunk.headingPath == []
        # other defaults.
        assert chunk.overlapTokenCount == 0
        assert chunk.isTablePart is False
        # a chunkId is auto-generated and non-empty.
        assert isinstance(chunk.chunkId, str) and chunk.chunkId

    def test_default_heading_path_is_independent_per_instance(self) -> None:
        a = Chunk(documentId="doc-a", content="a", tokenCount=1, orderIndex=0)
        b = Chunk(documentId="doc-b", content="b", tokenCount=1, orderIndex=0)
        a.headingPath.append("Section 1")
        # The default factory must not share a single list across instances.
        assert b.headingPath == []

    def test_auto_chunk_ids_are_unique(self) -> None:
        a = Chunk(documentId="doc", content="a", tokenCount=1, orderIndex=0)
        b = Chunk(documentId="doc", content="b", tokenCount=1, orderIndex=0)
        assert a.chunkId != b.chunkId

    def test_explicit_metadata_is_retained(self) -> None:
        chunk = Chunk(
            documentId="doc-9",
            content="cell text",
            tokenCount=5,
            orderIndex=3,
            overlapTokenCount=2,
            pageNumber=7,
            headingPath=["Intro", "Background"],
            isTablePart=True,
        )
        assert chunk.pageNumber == 7
        assert chunk.headingPath == ["Intro", "Background"]
        assert chunk.overlapTokenCount == 2
        assert chunk.isTablePart is True


class TestChunkConstructionGuards:
    """Construction-time guards for Chunk (Req 6.3, 6.7)."""

    def test_empty_document_id_rejected(self) -> None:
        # documentId is mandatory for every chunk (Req 6.3, 6.7).
        with pytest.raises(ModelValidationError):
            Chunk(documentId="", content="x", tokenCount=1, orderIndex=0)

    def test_non_str_document_id_rejected(self) -> None:
        with pytest.raises(ModelValidationError):
            Chunk(documentId=123, content="x", tokenCount=1, orderIndex=0)  # type: ignore[arg-type]

    def test_non_str_content_rejected(self) -> None:
        with pytest.raises(ModelValidationError):
            Chunk(documentId="doc", content=None, tokenCount=0, orderIndex=0)  # type: ignore[arg-type]

    def test_negative_token_count_rejected(self) -> None:
        with pytest.raises(ModelValidationError):
            Chunk(documentId="doc", content="x", tokenCount=-1, orderIndex=0)

    def test_negative_order_index_rejected(self) -> None:
        with pytest.raises(ModelValidationError):
            Chunk(documentId="doc", content="x", tokenCount=1, orderIndex=-1)

    def test_overlap_exceeding_token_count_rejected(self) -> None:
        # overlap cannot exceed this chunk's own token count (Req 6.2, 6.5).
        with pytest.raises(ModelValidationError):
            Chunk(
                documentId="doc",
                content="x",
                tokenCount=3,
                orderIndex=0,
                overlapTokenCount=4,
            )

    def test_overlap_equal_to_token_count_allowed(self) -> None:
        chunk = Chunk(
            documentId="doc",
            content="x",
            tokenCount=3,
            orderIndex=0,
            overlapTokenCount=3,
        )
        assert chunk.overlapTokenCount == 3

    def test_negative_page_number_rejected(self) -> None:
        with pytest.raises(ModelValidationError):
            Chunk(
                documentId="doc",
                content="x",
                tokenCount=1,
                orderIndex=0,
                pageNumber=-1,
            )

    def test_non_str_heading_path_entry_rejected(self) -> None:
        with pytest.raises(ModelValidationError):
            Chunk(
                documentId="doc",
                content="x",
                tokenCount=1,
                orderIndex=0,
                headingPath=["ok", 5],  # type: ignore[list-item]
            )

    def test_empty_chunk_id_rejected(self) -> None:
        with pytest.raises(ModelValidationError):
            Chunk(
                documentId="doc",
                content="x",
                tokenCount=1,
                orderIndex=0,
                chunkId="",
            )


# ---------------------------------------------------------------------------
# Cell spans (Req 3.2)
# ---------------------------------------------------------------------------
class TestCellSpanDefaults:
    """Default span handling for table cells."""

    def test_default_spans_are_one(self) -> None:
        # A cell spans at least itself: rowSpan/colSpan default to 1 (Req 3.2).
        cell = Cell(rowIndex=0, colIndex=0, value="v")
        assert cell.rowSpan == 1
        assert cell.colSpan == 1

    def test_explicit_spans_retained(self) -> None:
        cell = Cell(rowIndex=1, colIndex=2, value="merged", rowSpan=2, colSpan=3)
        assert cell.rowSpan == 2
        assert cell.colSpan == 3

    def test_empty_value_allowed(self) -> None:
        # Cell value may be an empty string (an empty but present cell).
        cell = Cell(rowIndex=0, colIndex=0, value="")
        assert cell.value == ""


class TestCellConstructionGuards:
    """Construction-time guards for Cell spans and indices (Req 3.2)."""

    def test_row_span_below_one_rejected(self) -> None:
        with pytest.raises(ModelValidationError):
            Cell(rowIndex=0, colIndex=0, value="v", rowSpan=0)

    def test_col_span_below_one_rejected(self) -> None:
        with pytest.raises(ModelValidationError):
            Cell(rowIndex=0, colIndex=0, value="v", colSpan=0)

    def test_negative_row_index_rejected(self) -> None:
        with pytest.raises(ModelValidationError):
            Cell(rowIndex=-1, colIndex=0, value="v")

    def test_negative_col_index_rejected(self) -> None:
        with pytest.raises(ModelValidationError):
            Cell(rowIndex=0, colIndex=-1, value="v")

    def test_non_str_value_rejected(self) -> None:
        with pytest.raises(ModelValidationError):
            Cell(rowIndex=0, colIndex=0, value=42)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Embedding status / attempts (Req 7.7)
# ---------------------------------------------------------------------------
class TestEmbeddingStatusAttemptsDefaults:
    """Default status/attempts handling for embeddings."""

    def test_defaults_are_ok_and_zero_attempts(self) -> None:
        emb = Embedding(chunkId="c1", vector=[0.1, 0.2], modelId="m1")
        assert emb.status is EmbeddingStatus.OK
        assert emb.attempts == 0
        assert emb.dimension == 2

    def test_failed_status_with_attempts_retained(self) -> None:
        # After exhausting retries the chunk is marked failed (Req 7.7).
        emb = Embedding(
            chunkId="c1",
            vector=[],
            modelId="m1",
            status=EmbeddingStatus.FAILED,
            attempts=3,
        )
        assert emb.status is EmbeddingStatus.FAILED
        assert emb.attempts == 3
        assert emb.dimension == 0

    def test_int_vector_values_coerced_to_float(self) -> None:
        emb = Embedding(chunkId="c1", vector=[1, 2, 3], modelId="m1")
        assert emb.vector == [1.0, 2.0, 3.0]
        assert all(isinstance(v, float) for v in emb.vector)


class TestEmbeddingConstructionGuards:
    """Construction-time guards for Embedding (Req 7.7)."""

    def test_empty_chunk_id_rejected(self) -> None:
        with pytest.raises(ModelValidationError):
            Embedding(chunkId="", vector=[0.1], modelId="m1")

    def test_empty_model_id_rejected(self) -> None:
        with pytest.raises(ModelValidationError):
            Embedding(chunkId="c1", vector=[0.1], modelId="")

    def test_non_list_vector_rejected(self) -> None:
        with pytest.raises(ModelValidationError):
            Embedding(chunkId="c1", vector="0.1", modelId="m1")  # type: ignore[arg-type]

    def test_non_numeric_vector_entry_rejected(self) -> None:
        with pytest.raises(ModelValidationError):
            Embedding(chunkId="c1", vector=[0.1, "x"], modelId="m1")  # type: ignore[list-item]

    def test_bool_vector_entry_rejected(self) -> None:
        # bools are not accepted as numeric vector values.
        with pytest.raises(ModelValidationError):
            Embedding(chunkId="c1", vector=[True, 0.2], modelId="m1")  # type: ignore[list-item]

    def test_non_status_enum_rejected(self) -> None:
        with pytest.raises(ModelValidationError):
            Embedding(chunkId="c1", vector=[0.1], modelId="m1", status="ok")  # type: ignore[arg-type]

    def test_negative_attempts_rejected(self) -> None:
        with pytest.raises(ModelValidationError):
            Embedding(chunkId="c1", vector=[0.1], modelId="m1", attempts=-1)

    def test_bool_attempts_rejected(self) -> None:
        # bool is rejected as an int attempts count.
        with pytest.raises(ModelValidationError):
            Embedding(chunkId="c1", vector=[0.1], modelId="m1", attempts=True)  # type: ignore[arg-type]
