"""Real production RAG pipeline for PDF e-books, papers, and articles.

Usage:
    python run_real.py "my-paper.pdf" "What causes heart disease?"
    python run_real.py "my-paper.pdf"          # interactive Q&A

Requirements:
    pip install pypdf sentence-transformers

Notes:
    This uses pypdf for text extraction (no OCR, no Docling). It works for any
    PDF that has a real text layer -- which is almost every e-book, research
    paper, and article. For SCANNED/image-only PDFs you would need OCR instead.
"""

from __future__ import annotations

import io
import sys
import time
from typing import List, Optional

from sentence_transformers import SentenceTransformer

from biomed_rag.config import PipelineConfig
from biomed_rag.embedding.model import EmbeddingModel, EmbeddingTimeoutError
from biomed_rag.embedding.registry import EmbeddingModelRegistry
from biomed_rag.ingestion.service import Accepted, FileInput, Rejected
from biomed_rag.parsing.engine import ParseError, ParsingEngine
from biomed_rag.parsing.raw_result import RawBlock, RawPage, RawParseResult, SourceDocument
from biomed_rag.parsing.registry import ParsingEngineRegistry
from biomed_rag.pipeline import Pipeline
from biomed_rag.retrieval.retriever import QueryRequest


# ---------------------------------------------------------------------------
# Real PDF parsing engine using pypdf (no OCR, no Docling)
# ---------------------------------------------------------------------------

class PyPDFEngine(ParsingEngine):
    """A simple, reliable text-layer PDF parser built on pypdf.

    Extracts text page by page and emits one RawBlock per paragraph. This avoids
    Docling and its OCR dependencies entirely, which is ideal for text PDFs.
    """

    def engine_id(self) -> str:
        return "pypdf"

    def is_available(self) -> bool:
        try:
            import pypdf  # noqa: F401
            return True
        except ImportError:
            return False

    def parse(self, doc: SourceDocument, deadline: Optional[float] = None) -> RawParseResult:
        from pypdf import PdfReader

        try:
            reader = PdfReader(io.BytesIO(doc.raw_bytes))
        except Exception as exc:
            raise ParseError(f"pypdf could not open the PDF: {exc}") from exc

        blocks: List[RawBlock] = []
        pages: List[RawPage] = []

        for page_index, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception:
                text = ""

            has_text = bool(text.strip())
            pages.append(RawPage(page_number=page_index, has_text_layer=has_text))

            # Split the page text into paragraphs on blank lines; fall back to
            # the whole page as one block when there are no blank-line breaks.
            paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
            if not paragraphs and has_text:
                paragraphs = [text.strip()]

            for para in paragraphs:
                blocks.append(RawBlock(text=para, page_number=page_index, kind="paragraph"))

        return RawParseResult(engine_id="pypdf", blocks=blocks, pages=pages)


# ---------------------------------------------------------------------------
# Real embedding model using sentence-transformers
# ---------------------------------------------------------------------------

class SentenceTransformerModel(EmbeddingModel):
    """Real embedding model backed by sentence-transformers."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        model_id: str = "st-minilm",
    ) -> None:
        self._model_name = model_name
        self._model_id = model_id
        self._model = SentenceTransformer(model_name)
        self._dimension = self._model.get_sentence_embedding_dimension()

    def model_id(self) -> str:
        return self._model_id

    def dimension(self) -> int:
        return self._dimension

    def embed(self, text: str, deadline: Optional[float] = None) -> List[float]:
        if deadline is not None and time.monotonic() > deadline:
            raise EmbeddingTimeoutError("deadline already passed")
        vector = self._model.encode(text, normalize_embeddings=True)
        if deadline is not None and time.monotonic() > deadline:
            raise EmbeddingTimeoutError("embedding exceeded deadline")
        return vector.tolist()


# ---------------------------------------------------------------------------
# Pipeline assembly
# ---------------------------------------------------------------------------

MODEL_ID = "st-minilm"
MODEL_NAME = "all-MiniLM-L6-v2"  # 384 dimensions, fast, good quality
DIMENSION = 384


def build_pipeline() -> Pipeline:
    """Build the pipeline with real PDF parsing + real embeddings."""

    config = PipelineConfig(
        parsing_engine="docling",  # the config key; we register pypdf under it
        embedding_model=MODEL_ID,
        embedding_dimension=DIMENSION,
        embedding_timeout_seconds=60,
    )

    # Register our pypdf engine for the configured parsing choice.
    parsing_registry = ParsingEngineRegistry()
    parsing_registry.register(config.parsing_engine, lambda: PyPDFEngine())

    # Real embedding model.
    embedding_registry = EmbeddingModelRegistry()
    embedding_registry.register(MODEL_ID, lambda: SentenceTransformerModel(MODEL_NAME, MODEL_ID))

    return Pipeline(
        config=config,
        parsing_registry=parsing_registry,
        embedding_registry=embedding_registry,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python run_real.py <file.pdf> [query]")
        print('Example: python run_real.py paper.pdf "What is PCSK9?"')
        sys.exit(1)

    pdf_path = sys.argv[1]
    query = sys.argv[2] if len(sys.argv) > 2 else None

    print(f"Loading {pdf_path}...")
    with open(pdf_path, "rb") as f:
        content = f.read()
    print(f"  Size: {len(content) / 1024 / 1024:.1f} MB")

    print("Building pipeline (loading embedding model on first run)...")
    pipeline = build_pipeline()

    # Submit
    result = pipeline.submit(FileInput(filename=pdf_path, content=content))
    if isinstance(result, Rejected):
        print(f"REJECTED: [{result.code.value}] {result.message}")
        sys.exit(1)
    job_id = result.jobId if isinstance(result, Accepted) else result.existingJobId
    print(f"  Accepted. Job: {job_id}")

    # Process
    print("Processing (parse -> normalize -> chunk -> embed -> store)...")
    t0 = time.time()
    outcome = pipeline.process(job_id)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")
    print(f"  Result: {type(outcome).__name__}")

    # If the job failed, print WHICH stage and WHY, then show stage statuses.
    if type(outcome).__name__ == "JobFailed":
        print(f"  FAILED at stage : {outcome.failingStage.name}")
        print(f"  Reason          : {outcome.reason}")
        if getattr(outcome, "unstoredChunkIds", None):
            print(f"  Unstored chunks : {len(outcome.unstoredChunkIds)}")
        print("\n  Stage statuses:")
        status = pipeline.status(job_id)
        for stage, st in status.stageStatuses.items():
            print(f"    {stage.name:<14} {st.name}")
        sys.exit(1)

    stored = getattr(outcome, "storedChunkIds", [])
    print(f"  Chunks stored: {len(stored)}")

    if not stored:
        print("No chunks stored (document may have produced no extractable text).")
        sys.exit(1)

    # Query
    if query:
        _run_query(pipeline, query)
    else:
        print("\nReady. Type your questions (Ctrl+C to exit):\n")
        try:
            while True:
                q = input("Query> ").strip()
                if not q:
                    continue
                _run_query(pipeline, q)
                print()
        except (KeyboardInterrupt, EOFError):
            print("\nBye.")


def _run_query(pipeline: Pipeline, query: str) -> None:
    res = pipeline.retrieve(QueryRequest(text=query, topK=5))
    print(f"\n  Results ({len(res.chunks)} chunks):")
    for i, c in enumerate(res.chunks, 1):
        print(f"\n  [{i}] similarity={c.similarity:.4f} | page={c.pageNumber}")
        print(f"      {c.content[:200]}...")


if __name__ == "__main__":
    main()
