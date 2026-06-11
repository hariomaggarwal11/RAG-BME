"""Book-level metadata: auto-extraction from PDFs + a persisted registry.

Each document in the library is identified by its content hash (documentId).
This module captures the *book* metadata (title, author, edition, year) for a
documentId so that every generated answer can cite the source book and author.

Metadata is captured in two ways (per the chosen design):
1. **Auto** from the PDF's embedded metadata (pypdf ``reader.metadata``).
2. **Manual override** supplied by the user in the web UI.

The registry persists to a small JSON file so citations survive across runs,
alongside the persisted vector store.
"""

from __future__ import annotations

import io
import json
import os
from dataclasses import asdict, dataclass
from typing import Dict, Optional

DEFAULT_BOOKS_PATH = os.environ.get("RAG_BOOKS_PATH", ".rag_books.json")


@dataclass
class BookMeta:
    """Bibliographic metadata for one ingested document."""

    title: str = ""
    author: str = ""
    edition: str = ""
    year: str = ""
    source_filename: str = ""

    def display(self) -> str:
        """A short human-readable citation label, e.g. 'Title - Author'."""
        title = self.title or self.source_filename or "Unknown title"
        if self.author:
            return f"{title} - {self.author}"
        return title


def extract_pdf_metadata(data: bytes, filename: str = "") -> BookMeta:
    """Best-effort extraction of title/author from a PDF's embedded metadata.

    Falls back to the filename (without extension) for the title when the PDF
    carries no title. Never raises - returns whatever could be read.
    """
    title = ""
    author = ""
    year = ""
    try:
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        meta = reader.metadata
        if meta is not None:
            title = (getattr(meta, "title", None) or "").strip()
            author = (getattr(meta, "author", None) or "").strip()
            # Creation date often looks like "D:20180102..."; keep the year only.
            raw_date = (getattr(meta, "creation_date_raw", None) or "")
            if isinstance(raw_date, str) and raw_date.startswith("D:") and len(raw_date) >= 6:
                year = raw_date[2:6]
    except Exception:
        pass

    if not title:
        base = os.path.basename(filename)
        title = os.path.splitext(base)[0] if base else ""

    return BookMeta(title=title, author=author, year=year, source_filename=filename)


class BookRegistry:
    """A persisted documentId -> BookMeta mapping."""

    def __init__(self, path: str = DEFAULT_BOOKS_PATH) -> None:
        self._path = path
        self._books: Dict[str, BookMeta] = {}
        self._load()

    def _load(self) -> None:
        if os.path.exists(self._path):
            try:
                with open(self._path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                self._books = {k: BookMeta(**v) for k, v in raw.items()}
            except Exception:
                self._books = {}

    def _save(self) -> None:
        try:
            with open(self._path, "w", encoding="utf-8") as f:
                json.dump({k: asdict(v) for k, v in self._books.items()}, f, indent=2)
        except Exception:
            pass

    def get(self, document_id: str) -> Optional[BookMeta]:
        return self._books.get(str(document_id))

    def set(self, document_id: str, meta: BookMeta) -> None:
        self._books[str(document_id)] = meta
        self._save()

    def all(self) -> Dict[str, BookMeta]:
        return dict(self._books)
