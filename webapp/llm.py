"""Pluggable LLM answer-generation layer for the RAG web app.

Turns retrieved chunks into a natural-language answer. Three providers are
supported, auto-selected at runtime:

1. OpenAI       - used when OPENAI_API_KEY is set (needs `pip install openai`).
2. Ollama       - used when a local Ollama server is reachable (free, local).
3. Extractive   - zero-setup fallback that returns the most relevant passages
                  with a note that no generative LLM is configured.

This keeps the app usable with no API key at all, while letting you plug in a
real LLM for true RAG generation simply by setting an environment variable.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import List, Protocol, Sequence


# ---------------------------------------------------------------------------
# Retrieved-context shape (kept tiny so this module has no internal deps)
# ---------------------------------------------------------------------------

@dataclass
class ContextChunk:
    """A retrieved passage used to ground the answer, with source attribution."""

    content: str
    similarity: float
    page: object
    document_id: object
    book: str = "Unknown source"
    author: str = ""
    section: str = ""


def _source_label(c: ContextChunk) -> str:
    """A compact source label for one passage, e.g. 'Book, Author, p.12'."""
    parts = [c.book]
    if c.author:
        parts.append(c.author)
    if c.section:
        parts.append(f"section: {c.section}")
    parts.append(f"p.{c.page}")
    return ", ".join(parts)


def _build_prompt(question: str, chunks: Sequence[ContextChunk]) -> str:
    """Build a grounded RAG prompt from the question and retrieved chunks."""
    context_blocks = []
    for i, c in enumerate(chunks, 1):
        snippet = c.content.strip().replace("\n", " ")
        context_blocks.append(f"[{i}] (Source: {_source_label(c)})\n{snippet}")
    context = "\n\n".join(context_blocks)
    return (
        "You are a Biomedical Engineering reference librarian. Answer the "
        "question using ONLY the context passages below, which come from "
        "verified BME textbooks. Be accurate and do not invent facts. After "
        "the answer, add a 'Sources:' line listing the book title and author "
        "for each passage you used (with page numbers). Cite passages inline "
        "like [1], [2]. If the answer is not in the context, say you could not "
        "find it in the available books.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )


# ---------------------------------------------------------------------------
# Provider protocol
# ---------------------------------------------------------------------------

class LLM(Protocol):
    name: str

    def generate(self, question: str, chunks: Sequence[ContextChunk]) -> str:
        ...


# ---------------------------------------------------------------------------
# OpenAI provider
# ---------------------------------------------------------------------------

class OpenAILLM:
    """OpenAI-compatible chat client.

    Works with the official OpenAI API and any OpenAI-compatible endpoint
    (e.g. freemodel.dev) by passing a custom ``base_url`` / ``api_key``.
    """

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        label: str = "OpenAI",
    ) -> None:
        self.model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        # base_url falls back to the SDK's OPENAI_BASE_URL handling when None.
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.name = f"{label} ({self.model})"

    def generate(self, question: str, chunks: Sequence[ContextChunk]) -> str:
        from openai import OpenAI

        kwargs = {}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        if self.api_key:
            kwargs["api_key"] = self.api_key
        client = OpenAI(**kwargs)
        prompt = _build_prompt(question, chunks)
        resp = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You answer strictly from provided context."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
        )
        return resp.choices[0].message.content or ""


class FreeModelLLM(OpenAILLM):
    """freemodel.dev provider (OpenAI-compatible endpoint).

    Configure via env vars:
        FREEMODEL_API_KEY   - your freemodel.dev API key (required)
        FREEMODEL_BASE_URL  - default https://api.freemodel.dev/v1
        FREEMODEL_MODEL     - model id, e.g. claude-t0
    """

    def __init__(self, model: str | None = None) -> None:
        base_url = os.environ.get("FREEMODEL_BASE_URL", "https://api.freemodel.dev/v1")
        api_key = os.environ.get("FREEMODEL_API_KEY") or os.environ.get("OPENAI_API_KEY")
        model = model or os.environ.get("FREEMODEL_MODEL", "claude-t0")
        super().__init__(model=model, base_url=base_url, api_key=api_key, label="freemodel.dev")


# ---------------------------------------------------------------------------
# Ollama provider (local, free)
# ---------------------------------------------------------------------------

class OllamaLLM:
    def __init__(self, host: str | None = None, model: str | None = None) -> None:
        self.host = (host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")
        self.model = model or os.environ.get("OLLAMA_MODEL", "llama3.2")
        self.name = f"Ollama ({self.model})"

    @staticmethod
    def is_reachable(host: str) -> bool:
        try:
            with urllib.request.urlopen(f"{host.rstrip('/')}/api/tags", timeout=1.5) as r:
                return r.status == 200
        except Exception:
            return False

    def generate(self, question: str, chunks: Sequence[ContextChunk]) -> str:
        prompt = _build_prompt(question, chunks)
        payload = json.dumps(
            {"model": self.model, "prompt": prompt, "stream": False}
        ).encode("utf-8")
        req = urllib.request.Request(
            f"{self.host}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                data = json.loads(r.read().decode("utf-8"))
            return data.get("response", "").strip()
        except urllib.error.URLError as exc:
            return f"(Ollama request failed: {exc})"


# ---------------------------------------------------------------------------
# Extractive fallback (no LLM, zero setup)
# ---------------------------------------------------------------------------

class ExtractiveLLM:
    name = "Extractive (no LLM configured)"

    def generate(self, question: str, chunks: Sequence[ContextChunk]) -> str:
        if not chunks:
            return "No relevant passages were found in the BME library."
        lines = [
            "No generative LLM is configured, so here are the most relevant "
            "passages from the BME library (set FREEMODEL_API_KEY, "
            "OPENAI_API_KEY, or run Ollama for natural-language answers):\n"
        ]
        for i, c in enumerate(chunks, 1):
            snippet = c.content.strip().replace("\n", " ")
            if len(snippet) > 500:
                snippet = snippet[:500] + "..."
            lines.append(f"[{i}] ({_source_label(c)}, score {c.similarity:.2f})\n{snippet}")
        return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# Auto-selection
# ---------------------------------------------------------------------------

def get_llm(prefer: str | None = None) -> LLM:
    """Return the best available LLM provider.

    Selection order (unless ``prefer`` forces one): OpenAI -> Ollama ->
    Extractive. ``prefer`` may be "openai", "ollama", or "extractive".
    """
    prefer = (prefer or os.environ.get("RAG_LLM_PROVIDER", "")).lower()

    if prefer == "freemodel":
        return FreeModelLLM()
    if prefer == "openai":
        return OpenAILLM()
    if prefer == "ollama":
        return OllamaLLM()
    if prefer == "extractive":
        return ExtractiveLLM()

    # Auto: freemodel.dev if its key is set, else OpenAI, else Ollama, else extractive.
    if os.environ.get("FREEMODEL_API_KEY"):
        return FreeModelLLM()
    if os.environ.get("OPENAI_API_KEY"):
        return OpenAILLM()
    ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    if OllamaLLM.is_reachable(ollama_host):
        return OllamaLLM()
    return ExtractiveLLM()
