"""Biomedical RAG pipeline package.

The :class:`~biomed_rag.pipeline.Pipeline` facade assembles every component from
a single :class:`~biomed_rag.config.PipelineConfig` and exposes the submit →
process → retrieve entry points.
"""

from __future__ import annotations

from .pipeline import (
    DEFAULT_EMBEDDING_DIMENSION,
    DEFAULT_EMBEDDING_MODEL_ID,
    Pipeline,
    default_config,
)

__all__ = [
    "Pipeline",
    "default_config",
    "DEFAULT_EMBEDDING_MODEL_ID",
    "DEFAULT_EMBEDDING_DIMENSION",
]
