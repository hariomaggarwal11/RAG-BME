"""Top-level pipeline assembly (Task 14.1, Req 8.3, 10.1).

The :class:`Pipeline` is the single facade that wires every component of the
biomedical RAG pipeline together from one :class:`PipelineConfig` and exposes
the three end-to-end entry points:

* :meth:`Pipeline.submit` — accept a document through the
  :class:`~biomed_rag.ingestion.IngestionService` (Req 1).
* :meth:`Pipeline.process` (and :meth:`Pipeline.resume`) — drive a submitted job
  through parsing → normalization → chunking → embedding → storage via the
  :class:`~biomed_rag.orchestration.Orchestrator` to completion (Req 8.3, 10.1).
* :meth:`Pipeline.retrieve` — serve the most relevant stored chunks for a query
  through the :class:`~biomed_rag.retrieval.Retriever` (Req 9).

Every component the design names is assembled here and reachable from the
facade, so none is orphaned: the ingestion service and its job state store, the
parser (with its engine registry and optional OCR processor), the normalizer,
the chunker, the embedder (with its model registry), the vector store, the
retriever, and the orchestrator.

Bytes are not retained by the Ingestion_Service, so the Pipeline keeps the
submitted :class:`SourceDocument` for each job and hands the Orchestrator a
``source_resolver`` that maps a job back to those retained bytes — this is what
lets the parsing stage re-read the original document.

Defaults make the facade runnable in tests with no external services: a
deterministic mock parsing engine, a deterministic mock embedding model, and the
in-memory vector store. Each piece is injectable, so a real deployment supplies
its own registries/adapters (and a config selecting concrete backends) without
changing this module.
"""

from __future__ import annotations

from typing import Dict, Optional, Union

from .chunking.chunker import Chunker
from .chunking.tokenizer import Tokenizer
from .config import PipelineConfig
from .embedding.embedder import Embedder
from .embedding.mock import MockEmbeddingModel
from .embedding.model import EmbeddingModel
from .embedding.registry import EmbeddingModelRegistry
from .ingestion.job_state_store import JobStateStore
from .ingestion.service import (
    Accepted,
    Duplicate,
    FileInput,
    IngestionResult,
    IngestionService,
)
from .models import DocumentId, ProcessingJob
from .normalization.normalizer import Normalizer
from .ocr.processor import OCRProcessor
from .orchestration.orchestrator import Orchestrator
from .orchestration.results import JobOutcome, JobStatus, ResumeOutcome
from .parsing.mock_engine import MockParsingEngine
from .parsing.parser import Parser
from .parsing.raw_result import SourceDocument
from .parsing.registry import ParsingEngineRegistry
from .retrieval.retriever import QueryRequest, RetrievalResult, Retriever
from .storage.in_memory import InMemoryVectorStore
from .storage.port import VectorStore

# Defaults that make the facade runnable end-to-end in tests without any
# external service. They are used only when the caller supplies neither a
# custom config nor the corresponding injected adapter/registry.
DEFAULT_EMBEDDING_MODEL_ID = "mock-embedding"
DEFAULT_EMBEDDING_DIMENSION = 64


def default_config() -> PipelineConfig:
    """A :class:`PipelineConfig` wired for the deterministic in-memory defaults.

    It names the mock embedding model and a small matching dimension so the
    default embedding registry and retriever resolve a usable model out of the
    box. All other values are the design defaults.
    """
    return PipelineConfig(
        embedding_model=DEFAULT_EMBEDDING_MODEL_ID,
        embedding_dimension=DEFAULT_EMBEDDING_DIMENSION,
    )


def _default_parsing_registry(config: PipelineConfig) -> ParsingEngineRegistry:
    """Build a registry whose configured engine is a deterministic mock."""
    registry = ParsingEngineRegistry()
    engine_id = config.parsing_engine.value
    registry.register(
        config.parsing_engine,
        lambda: MockParsingEngine(engine_id=engine_id),
    )
    return registry


def _default_embedding_registry(config: PipelineConfig) -> EmbeddingModelRegistry:
    """Build a registry whose configured model is a deterministic mock.

    Requires ``config.embedding_model`` to be set; the default config sets it.
    """
    model_id = config.embedding_model
    if not model_id:
        raise ValueError(
            "cannot build a default embedding registry: config.embedding_model "
            "is not set. Pass a config naming a registered model, or inject an "
            "embedding_registry."
        )
    dimension = config.embedding_dimension
    registry = EmbeddingModelRegistry()
    registry.register(
        model_id,
        lambda: MockEmbeddingModel(model_id=model_id, dimension=dimension),
    )
    return registry


class Pipeline:
    """End-to-end facade assembling the full pipeline from one config (Task 14.1).

    Construct with a :class:`PipelineConfig`; everything else has a runnable
    default but is injectable for production wiring or finer-grained tests.

    Parameters
    ----------
    config:
        The single source of truth for every bounded parameter. Defaults to
        :func:`default_config` (mock embedding model + in-memory-friendly
        dimension) when omitted.
    job_store:
        The :class:`JobStateStore` shared by ingestion and orchestration.
    parsing_registry:
        Config-driven :class:`ParsingEngineRegistry`. Defaults to a registry
        whose configured engine is a deterministic :class:`MockParsingEngine`.
    ocr_processor:
        Optional :class:`OCRProcessor` wired into the parser; ``None`` disables
        OCR (the parser behaves exactly as without it).
    embedding_registry:
        Config-driven :class:`EmbeddingModelRegistry`. Defaults to a registry
        whose configured model is a deterministic :class:`MockEmbeddingModel`.
    vector_store:
        The :class:`VectorStore` used for storage and retrieval. Defaults to an
        :class:`InMemoryVectorStore`.
    tokenizer:
        Optional tokenizer for the :class:`Chunker`; defaults to the chunker's
        deterministic whitespace tokenizer.
    retrieval_model:
        Optional explicit :class:`EmbeddingModel` for the retriever. Defaults to
        the model selected from ``embedding_registry`` for ``config`` so the
        query is embedded with the same model family used for the chunks.
    """

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        *,
        job_store: Optional[JobStateStore] = None,
        parsing_registry: Optional[ParsingEngineRegistry] = None,
        ocr_processor: Optional[OCRProcessor] = None,
        embedding_registry: Optional[EmbeddingModelRegistry] = None,
        vector_store: Optional[VectorStore] = None,
        tokenizer: Optional[Tokenizer] = None,
        retrieval_model: Optional[EmbeddingModel] = None,
    ) -> None:
        self.config = config if config is not None else default_config()

        # Shared state store: ingestion writes jobs, orchestration reads/updates.
        self.job_store = job_store if job_store is not None else JobStateStore()

        # Retained submitted bytes per document so the orchestrator can re-read
        # the source during parsing (the IngestionService does not keep bytes).
        self._sources: Dict[DocumentId, SourceDocument] = {}

        # Registries / adapters (all default to deterministic, in-memory pieces).
        self._parsing_registry = (
            parsing_registry
            if parsing_registry is not None
            else _default_parsing_registry(self.config)
        )
        self._embedding_registry = (
            embedding_registry
            if embedding_registry is not None
            else _default_embedding_registry(self.config)
        )
        self.vector_store: VectorStore = (
            vector_store if vector_store is not None else InMemoryVectorStore()
        )

        # -- component assembly ------------------------------------------
        # Ingestion (Req 1).
        self.ingestion = IngestionService(self.job_store, self.config)

        # Parser + engine registry + optional OCR processor (Req 2, 3, 4).
        self.parser = Parser(
            config=self.config,
            registry=self._parsing_registry,
            ocr=ocr_processor,
        )

        # Normalizer (Req 5).
        self.normalizer = Normalizer()

        # Chunker (Req 6).
        self.chunker = Chunker(tokenizer=tokenizer)

        # Embedder + model registry (Req 7).
        self.embedder = Embedder(self._embedding_registry)

        # Orchestrator wiring all stages with the source resolver (Req 8.3, 10.1).
        self.orchestrator = Orchestrator(
            self.config,
            self.job_store,
            parser=self.parser,
            normalizer=self.normalizer,
            chunker=self.chunker,
            embedder=self.embedder,
            vector_store=self.vector_store,
            source_resolver=self._resolve_source,
        )

        # Retriever is resolved lazily so submit/process work even when no
        # embedding model is configured (it is only needed for retrieve).
        self._retrieval_model = retrieval_model
        self._retriever: Optional[Retriever] = None

    # -- entry point: submit ---------------------------------------------
    def submit(self, file: FileInput) -> IngestionResult:
        """Submit ``file`` for ingestion and retain its bytes for processing.

        Runs the Ingestion_Service validation gate (Req 1). On an accepted or
        duplicate submission the original :class:`SourceDocument` is retained,
        keyed by the job's document id, so the parsing stage can re-read it
        during :meth:`process` / :meth:`resume`. Rejections retain nothing.
        """
        result = self.ingestion.submit(file)

        job_id = None
        if isinstance(result, Accepted):
            job_id = result.jobId
        elif isinstance(result, Duplicate):
            job_id = result.existingJobId

        if job_id is not None:
            job = self.job_store.get(job_id)
            self._sources[job.documentId] = SourceDocument(
                document_id=job.documentId,
                raw_bytes=file.content,
                doc_format=job.metadata.format,
            )
        return result

    # -- entry point: process / resume -----------------------------------
    def process(self, job_id: str) -> JobOutcome:
        """Run the full stage pipeline for ``job_id`` to a terminal outcome.

        Returns :class:`~biomed_rag.orchestration.JobCompleted` once every chunk
        is stored (Req 8.3) or :class:`~biomed_rag.orchestration.JobFailed`
        identifying the failing stage (Req 8.7, 10.3).
        """
        return self.orchestrator.run(job_id)

    def resume(self, job_id: str) -> ResumeOutcome:
        """Resume a previously failed ``job_id`` from its recorded failing stage.

        Reuses the artifacts of completed stages (Req 10.4) and rejects jobs with
        no recorded failing stage (Req 10.5).
        """
        return self.orchestrator.resume(job_id)

    # -- entry point: retrieve -------------------------------------------
    def retrieve(self, query: Union[QueryRequest, str]) -> RetrievalResult:
        """Return the most relevant stored chunks for ``query`` (Req 9).

        ``query`` may be a :class:`QueryRequest` or a plain string (wrapped into
        a default request). The query is embedded with the configured model and
        matched against the shared vector store.
        """
        request = query if isinstance(query, QueryRequest) else QueryRequest(text=query)
        return self.retriever.retrieve(request)

    # -- observability passthroughs --------------------------------------
    def status(self, job_id: str) -> JobStatus:
        """Return the live ``{currentStage, stageStatuses, progressPercent}`` view."""
        return self.orchestrator.status(job_id)

    def transitions(self, job_id: str):
        """Return the ordered stage-transition history recorded for ``job_id``."""
        return self.orchestrator.transitions(job_id)

    # -- lazily-built retriever ------------------------------------------
    @property
    def retriever(self) -> Retriever:
        """The :class:`Retriever`, built on first use from the shared store/model."""
        if self._retriever is None:
            model = (
                self._retrieval_model
                if self._retrieval_model is not None
                else self._embedding_registry.select(self.config)
            )
            self._retriever = Retriever(self.vector_store, model, self.config)
        return self._retriever

    # -- source resolution (bytes retained by the facade) ----------------
    def _resolve_source(self, job: ProcessingJob) -> SourceDocument:
        """Return the retained :class:`SourceDocument` for ``job``.

        Raises a descriptive error when no bytes were retained for the job's
        document; the Orchestrator turns this into a recorded parsing-stage
        failure (Req 2.5).
        """
        try:
            return self._sources[job.documentId]
        except KeyError:
            raise KeyError(
                f"no submitted source bytes retained for document "
                f"{job.documentId!r}; submit the document via Pipeline.submit "
                "before processing"
            ) from None


__all__ = [
    "Pipeline",
    "default_config",
    "DEFAULT_EMBEDDING_MODEL_ID",
    "DEFAULT_EMBEDDING_DIMENSION",
]
