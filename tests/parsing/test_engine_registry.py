"""Unit tests for the ParsingEngine port, registry, RawParseResult, and mock (Task 5.1).

These cover the contract introduced in task 5.1 only: config-driven engine
selection (Req 2.2), the deterministic mock adapter, and the shared
RawParseResult shape. Parser logic and concrete adapters are tested with their
own tasks.
"""

from __future__ import annotations

import pytest

from biomed_rag.config import ParsingEngine as ParsingEngineChoice
from biomed_rag.config import PipelineConfig
from biomed_rag.parsing import (
    EngineUnavailableError,
    MockParsingEngine,
    ParseError,
    ParsingEngine,
    ParsingEngineNotRegisteredError,
    ParsingEngineRegistry,
    RawBlock,
    RawParseResult,
    SourceDocument,
)


def _doc(data: bytes = b"hello world\n\nsecond paragraph") -> SourceDocument:
    return SourceDocument(document_id="doc-1", raw_bytes=data)


def test_mock_engine_implements_port() -> None:
    engine = MockParsingEngine()
    assert isinstance(engine, ParsingEngine)
    assert engine.engine_id() == "mock"
    assert engine.is_available() is True


def test_mock_parse_is_deterministic() -> None:
    engine = MockParsingEngine()
    result_a = engine.parse(_doc())
    result_b = engine.parse(_doc())
    assert isinstance(result_a, RawParseResult)
    assert result_a == result_b
    # Two non-empty paragraphs -> two blocks.
    assert [b.text for b in result_a.blocks] == ["hello world", "second paragraph"]


def test_mock_unavailable_raises_on_parse() -> None:
    engine = MockParsingEngine(available=False)
    assert engine.is_available() is False
    with pytest.raises(EngineUnavailableError):
        engine.parse(_doc())


def test_mock_raise_on_parse_simulates_failure() -> None:
    engine = MockParsingEngine(raise_on_parse=ParseError("boom"))
    with pytest.raises(ParseError):
        engine.parse(_doc())


def test_mock_preset_result_is_stamped_with_engine_id() -> None:
    preset = RawParseResult(
        engine_id="ignored",
        blocks=[RawBlock(text="x", page_number=0)],
    )
    engine = MockParsingEngine(engine_id="custom", preset_result=preset)
    result = engine.parse(_doc())
    assert result.engine_id == "custom"
    assert [b.text for b in result.blocks] == ["x"]


def test_registry_selects_engine_from_config() -> None:
    registry = ParsingEngineRegistry()
    registry.register(
        ParsingEngineChoice.DOCLING,
        lambda: MockParsingEngine(engine_id="docling"),
    )
    registry.register(
        ParsingEngineChoice.LLAMAPARSE,
        lambda: MockParsingEngine(engine_id="llamaparse"),
    )

    docling_cfg = PipelineConfig(parsing_engine=ParsingEngineChoice.DOCLING)
    llama_cfg = PipelineConfig(parsing_engine=ParsingEngineChoice.LLAMAPARSE)

    assert registry.select(docling_cfg).engine_id() == "docling"
    assert registry.select(llama_cfg).engine_id() == "llamaparse"


def test_registry_create_returns_fresh_instances() -> None:
    registry = ParsingEngineRegistry()
    registry.register(ParsingEngineChoice.DOCLING, MockParsingEngine)
    first = registry.create(ParsingEngineChoice.DOCLING)
    second = registry.create(ParsingEngineChoice.DOCLING)
    assert first is not second


def test_registry_unregistered_choice_raises() -> None:
    registry = ParsingEngineRegistry()
    assert registry.is_registered(ParsingEngineChoice.DOCLING) is False
    with pytest.raises(ParsingEngineNotRegisteredError):
        registry.create(ParsingEngineChoice.DOCLING)


def test_registry_duplicate_registration_requires_replace() -> None:
    registry = ParsingEngineRegistry()
    registry.register(ParsingEngineChoice.DOCLING, MockParsingEngine)
    with pytest.raises(ValueError):
        registry.register(ParsingEngineChoice.DOCLING, MockParsingEngine)
    # replace=True overrides without error.
    registry.register(
        ParsingEngineChoice.DOCLING,
        lambda: MockParsingEngine(engine_id="replaced"),
        replace=True,
    )
    assert registry.create(ParsingEngineChoice.DOCLING).engine_id() == "replaced"


def test_raw_parse_result_is_empty() -> None:
    assert RawParseResult(engine_id="mock").is_empty() is True
    populated = RawParseResult(
        engine_id="mock", blocks=[RawBlock(text="x", page_number=0)]
    )
    assert populated.is_empty() is False


def test_raw_parse_result_rejects_wrong_block_type() -> None:
    with pytest.raises(ValueError):
        RawParseResult(engine_id="mock", blocks=["not a block"])  # type: ignore[list-item]
