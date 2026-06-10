"""Retrieval for LLM ingestion (Req 9).

This package serves the most relevant stored chunks for a query as grounded
context. It exposes:

* :class:`Retriever` — ranks stored chunks against the Vector_Store using an
  injected embedding model (Req 9.1, 9.7, 9.9).
* :class:`QueryRequest` — a query's text with optional count and filter.
* :class:`RetrievalResult` / :class:`RetrievedChunk` — the result shape with a
  status and per-chunk source metadata (Req 9.4, 9.5).
* :class:`RetrievalStatus` — the outcome status enum (Req 9.2, 9.3, 9.6, 9.8).
* :data:`PLACEHOLDER_DOCUMENT_ID` / :data:`PLACEHOLDER_PAGE_NUMBER` — placeholder
  metadata values (Req 9.5).
"""

from __future__ import annotations

from .retriever import (
    PLACEHOLDER_DOCUMENT_ID,
    PLACEHOLDER_PAGE_NUMBER,
    QueryRequest,
    RetrievalResult,
    RetrievalStatus,
    RetrievedChunk,
    Retriever,
)

__all__ = [
    "Retriever",
    "QueryRequest",
    "RetrievalResult",
    "RetrievedChunk",
    "RetrievalStatus",
    "PLACEHOLDER_DOCUMENT_ID",
    "PLACEHOLDER_PAGE_NUMBER",
]
