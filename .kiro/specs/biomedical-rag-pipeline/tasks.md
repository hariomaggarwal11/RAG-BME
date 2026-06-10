# Implementation Plan: Biomedical RAG Pipeline

## Overview

This plan converts the biomedical RAG pipeline design into incremental, test-driven Python coding tasks. Foundational pieces (validated `PipelineConfig`, shared data models, and the in-memory `JobStateStore`) come first, followed by per-component implementation paired with its tests, then the `Orchestrator`, and finally end-to-end wiring.

The implementation language is **Python**. Property-based tests use **Hypothesis** (per the design's Testing Strategy); each correctness property from the design is implemented by exactly one property-based test running a minimum of 100 iterations and tagged with the comment format `Feature: biomedical-rag-pipeline, Property {number}: {property_text}`. External backends (parsing engines, OCR engines, embedding models, vector stores) are exercised through their port interfaces with deterministic mock/in-memory adapters in property and unit tests; concrete adapters get a small number of integration tests.

Tasks marked with `*` are optional test tasks and can be skipped for a faster MVP.

## Tasks

- [x] 1. Set up project structure and tooling
  - Create the package layout (`src/biomed_rag/` with sub-packages for `config`, `models`, `ingestion`, `parsing`, `ocr`, `normalization`, `chunking`, `embedding`, `storage`, `retrieval`, `orchestration`) and a `tests/` tree mirroring it
  - Configure `pyproject.toml` with runtime deps and dev deps (`pytest`, `hypothesis`)
  - Add a shared test support module for deterministic generators, the injected dictionary, and the injected tokenizer used by later property tests
  - _Requirements: foundational (supports all)_

- [x] 2. Implement the validated configuration model
  - [x] 2.1 Implement `PipelineConfig` with construction-time bound enforcement
    - Define every config key from the design table with its type, bounds, and default (`maxFileSizeBytes`, `maxFilenameLength`, `parsingEngine`, `parseTimeoutSeconds`, `ocrConfidenceThreshold`, `ocrPageTimeoutSeconds`, `maxChunkTokens`, `chunkOverlapTokens`, `embeddingModel`, `embeddingDimension`, `embeddingTimeoutSeconds`, `embeddingMaxRetries`, `vectorStoreBackend`, `defaultTopK`, `maxQueryChars`, `stageRetryLimit`)
    - Reject out-of-range values at construction with a descriptive error; enforce the dependent bound `chunkOverlapTokens ∈ [0, maxChunkTokens-1]`
    - _Requirements: 1.2, 1.6, 1.8, 2.2, 2.7, 4.4, 4.6, 6.1, 6.2, 7.1, 7.2, 7.6, 8.x, 9.1, 9.2, 9.3, 10.2_

  - [x] 2.2 Write unit tests for `PipelineConfig` validation
    - Test in-range acceptance, defaults, and out-of-range rejection for each bounded key, including the `chunkOverlapTokens` dependency on `maxChunkTokens`
    - _Requirements: 6.1, 6.2, 7.1, 9.1, 10.2_

- [x] 3. Implement core identifiers, enums, and shared data models
  - [x] 3.1 Implement shared types and data models
    - Define `JobId`, `DocumentId`, and enums `Stage`, `StageStatus`, `Format`
    - Define `ProcessingJob`, `DocumentMetadata`, `StageState`
    - Define `ParsedDocument`, `TextBlock`, `Table`, `Cell`, `Figure`, `Heading`, `ImageOCRError`
    - Define `NormalizedDocument`, `ContentElement` and its `TextPayload`/`TablePayload`/`FigurePayload`
    - Define `Chunk`, `Embedding`, `VectorRecord`, `ScoredRecord` with field-level invariants (e.g. `rowSpan`/`colSpan` >= 1, `tokenCount` <= max, `progressPercent` in [0,100])
    - _Requirements: 1.4, 2.1, 2.4, 3.1, 3.2, 4.3, 5.4, 5.5, 6.1, 6.3, 7.1, 8.1, 9.1, 10.6_

  - [x] 3.2 Write unit tests for data model invariants
    - Test construction guards and default/empty-value handling for `Chunk` metadata, `Cell` spans, and `Embedding` status/attempts
    - _Requirements: 3.2, 6.3, 6.7, 7.7_

- [x] 4. Implement Ingestion_Service and Job State Store
  - [x] 4.1 Implement an in-memory `JobStateStore` with a content-hash dedup index
    - Persist `ProcessingJob` records, generate UUID job identifiers guaranteed unique across all jobs, and expose lookup by content hash and by job id
    - _Requirements: 1.1, 1.5_

  - [x] 4.2 Implement `IngestionService.submit` with the ordered validation gate
    - Apply the gate in order: filename present and <= 255 chars; byte size in [1, 500 MB]; format in {PDF, EPUB, DOCX, HTML} via content sniffing; well-formed-file check; SHA-256 dedup
    - On success create a `ProcessingJob`, record metadata `{filename, format, byteSize, contentHash, submittedAtUtc}` in UTC, and return the job id; emit distinct rejection codes/messages per failure case and the existing job id on duplicate
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8_

  - [x] 4.3 Write property test for ingestion validation totality and bounds
    - **Property 2: Ingestion validation is total and bound-correct**
    - **Validates: Requirements 1.2, 1.3, 1.6, 1.7, 1.8**

  - [x] 4.4 Write property test for job identifier uniqueness
    - **Property 1: Job identifier uniqueness**
    - **Validates: Requirements 1.1**

  - [x] 4.5 Write property test for content-hash deduplication idempotency
    - **Property 3: Content-hash deduplication is idempotent**
    - **Validates: Requirements 1.4, 1.5**

- [x] 5. Implement Parser and pluggable Parsing_Engine adapters
  - [x] 5.1 Define the `ParsingEngine` port and engine registry
    - Define the port (`isAvailable`, `parse`, `engineId`) and a registry that selects the engine from `PipelineConfig.parsingEngine`; provide a deterministic mock adapter for tests
    - _Requirements: 2.2_

  - [x] 5.2 Implement the Docling and LlamaParse adapters
    - Implement `DoclingAdapter` and `LlamaParseAdapter` against the `ParsingEngine` port, mapping raw engine output into the shared `RawParseResult` shape and reporting availability
    - _Requirements: 2.2, 2.6_

  - [x] 5.3 Implement the `Parser` block/heading extraction and failure handling
    - Produce `TextBlock`s in reading-sequence order with structural metadata; order multi-column layouts top-to-bottom within a column and left-to-right across columns; preserve heading hierarchy with nesting levels
    - Implement fail-closed handling: engine unavailable, parse error (discard partial output), parse timeout (`parseTimeoutSeconds`), and no-extractable-content, each marking the job failed with the recorded reason
    - _Requirements: 2.1, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8_

  - [x] 5.4 Implement table and figure extraction in the `Parser`
    - Map every non-empty cell to exactly one (row, col); assign spanning cells to top-left with recorded `rowSpan`/`colSpan`; extract figures with optional captions; record absent captions without failing; associate page number and zero-based reading-order position; flag degraded tables and retain raw region text
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6_

  - [x] 5.5 Write property test for reading-order positions
    - **Property 4: Reading-order positions are a contiguous zero-based sequence**
    - **Validates: Requirements 2.1, 2.3, 3.5**

  - [x] 5.6 Write property test for heading hierarchy preservation
    - **Property 5: Heading hierarchy is preserved**
    - **Validates: Requirements 2.4**

  - [x] 5.7 Write property test for table cell coordinate assignment
    - **Property 6: Table cell coordinates are a collision-free assignment**
    - **Validates: Requirements 3.1, 3.2**

  - [x] 5.8 Write unit tests for parser configuration, failures, and captions
    - Test engine selection by config, engine-unavailable/parse-error/timeout/no-content failure paths, figure caption presence/absence, and degraded-table handling
    - _Requirements: 2.2, 2.5, 2.6, 2.7, 2.8, 3.3, 3.4, 3.6_

- [x] 6. Implement OCR_Processor and wire it into parsing
  - [x] 6.1 Implement the `OCRProcessor`
    - Implement `processPage` and `processEmbeddedImage` producing `{text, confidence}`, `OCRError`, or `OCRTimeout`; record confidence in [0.0, 1.0]; flag blocks below `ocrConfidenceThreshold` as low-confidence while retaining text
    - _Requirements: 4.3, 4.4_

  - [x] 6.2 Wire OCR invocation into the `Parser`
    - Invoke OCR for pages lacking an extractable text layer and for embedded images with text content; store recovered text as `OCR_TEXT` blocks; record per-image `ImageOCRError` for unreadable/corrupt/unsupported images and per-page timeouts (`ocrPageTimeoutSeconds`) while continuing remaining pages/images
    - _Requirements: 4.1, 4.2, 4.5, 4.6_

  - [x] 6.3 Write property test for OCR confidence bounds and low-confidence flag
    - **Property 7: OCR confidence is bounded and the low-confidence flag is correct**
    - **Validates: Requirements 4.3, 4.4**

  - [x] 6.4 Write property test for OCR resilience to bad images
    - **Property 8: OCR is resilient to bad images**
    - **Validates: Requirements 4.1, 4.2, 4.5**

  - [x] 6.5 Write unit test for per-page OCR timeout
    - Test that a page exceeding `ocrPageTimeoutSeconds` records a timeout indication and processing continues
    - _Requirements: 4.6_

- [x] 7. Implement Normalizer with serialize/deserialize
  - [x] 7.1 Implement artifact removal and de-hyphenation
    - Remove header/footer text elements recurring on >= 2 pages (including page numbers); rejoin line-break-hyphenated words when the joined token is in the injected dictionary; retain intrinsic mid-line hyphens unchanged
    - _Requirements: 5.1, 5.2, 5.3_

  - [x] 7.2 Implement canonical `NormalizedDocument` production and empty/malformed handling
    - Produce content-preserving `ContentElement`s (text, heading hierarchy, tables, figures, OCR text) carrying page number and reading-order position; return an empty normalized doc + indication for empty/no-content input; reject malformed input leaving prior valid output unchanged
    - _Requirements: 5.4, 5.5, 5.7, 5.8_

  - [x] 7.3 Implement `serialize` and `deserialize` for `NormalizedDocument`
    - Implement the durable serialized form used by the Orchestrator for resume, preserving all content elements and metadata
    - _Requirements: 5.5, 5.6_

  - [x] 7.4 Write property test for recurring header/footer removal
    - **Property 9: Normalization removes recurring header/footer artifacts**
    - **Validates: Requirements 5.1**

  - [x] 7.5 Write property test for line-break de-hyphenation
    - **Property 10: Line-break de-hyphenation respects the dictionary**
    - **Validates: Requirements 5.2, 5.3**

  - [x] 7.6 Write property test for content/structure preservation
    - **Property 11: Normalization preserves all non-artifact content and structure**
    - **Validates: Requirements 5.4**

  - [x] 7.7 Write property test for the serialization round-trip (REQUIRED)
    - **Property 12: Normalized representation serialization round-trip**
    - Generate arbitrary `NormalizedDocument` values (varied element kinds, heading paths, tables with spans, figures with/without captions, page numbers, reading-order positions) and assert `deserialize(serialize(doc))` is structurally equivalent to `doc`
    - **Validates: Requirements 5.5, 5.6**

  - [x] 7.8 Write unit tests for empty and malformed normalization
    - Test empty/no-content indication and malformed-input rejection preserving prior output
    - _Requirements: 5.7, 5.8_

- [x] 8. Implement Chunker
  - [x] 8.1 Implement token-bounded chunking with overlap and metadata
    - Use the injected tokenizer to produce chunks <= `maxChunkTokens`; apply the configured overlap recorded in `overlapTokenCount`; attach `{documentId, pageNumber, headingPath}` with empty values when page/heading unavailable but always include `documentId`; assign `orderIndex`; produce zero chunks for artifact-only input
    - _Requirements: 6.1, 6.2, 6.3, 6.7, 6.8_

  - [x] 8.2 Implement table-aware chunking
    - Keep a fitting table within a single chunk; split an oversized table across chunks each within `maxChunkTokens`, marking parts with `isTablePart`
    - _Requirements: 6.4, 6.6_

  - [x] 8.3 Write property test for the chunk token bound
    - **Property 13: Chunk token bound is never exceeded**
    - **Validates: Requirements 6.1, 6.6**

  - [x] 8.4 Write property test for configured overlap
    - **Property 14: Configured overlap is honored**
    - **Validates: Requirements 6.2**

  - [x] 8.5 Write property test for chunk metadata attachment
    - **Property 15: Chunk metadata is always attached**
    - **Validates: Requirements 6.3, 6.7**

  - [x] 8.6 Write property test for fitting tables staying in one chunk
    - **Property 16: Fitting tables stay within a single chunk**
    - **Validates: Requirements 6.4**

  - [x] 8.7 Write property test for chunk completeness (REQUIRED)
    - **Property 17: Chunk completeness**
    - Reconstruct text by concatenating chunk contents in `orderIndex` order, dropping the leading `overlapTokenCount` tokens of each chunk after the first; assert it covers all non-artifact text and that artifact-only input yields zero chunks; include oversized tables and missing page/heading metadata in generators
    - **Validates: Requirements 6.5, 6.8**

- [x] 9. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 10. Implement Embedder and pluggable embedding model
  - [x] 10.1 Define the `EmbeddingModel` port, registry, and mock
    - Define the port (`modelId`, `dimension`, `embed`) and a registry selecting the model from config; provide a deterministic mock model for tests
    - _Requirements: 7.3_

  - [x] 10.2 Implement the `Embedder` with retry policy and dimension/timeout enforcement
    - Produce embeddings of the configured dimension (64..4096) within `embeddingTimeoutSeconds`; reject missing/unrecognized models leaving the chunk unmodified; on failure retain the chunk and retry up to 3 attempts, then mark the chunk failed retaining original content
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7_

  - [x] 10.3 Write property test for embedding dimension consistency
    - **Property 18: Embedding dimension is consistent and configured**
    - **Validates: Requirements 7.1, 7.5**

  - [x] 10.4 Write unit tests for embedding timeout, misconfiguration, and retries
    - Test timeout handling, missing/unrecognized model rejection, and the 3-attempt retry-then-fail behavior
    - _Requirements: 7.2, 7.3, 7.4, 7.6, 7.7_

- [x] 11. Implement pluggable Vector_Store
  - [x] 11.1 Define the `VectorStore` port and an in-memory adapter
    - Define `upsertBatch`, `replaceDocument` (atomic swap), `deleteDocument` (not-found error when absent), and `query`; implement an in-memory adapter for fast property/unit tests
    - _Requirements: 8.1, 8.2, 8.4, 8.5, 8.6, 8.8_

  - [x] 11.2 Implement the pgvector adapter
    - Implement the same port against pgvector with a transactional/versioned swap for atomic reprocess replacement and persistence within 5 seconds of generation
    - _Requirements: 8.1, 8.4, 8.6_

  - [x] 11.3 Write property test for retrievability by document identifier
    - **Property 19: Stored embeddings are retrievable by document identifier**
    - **Validates: Requirements 8.2**

  - [x] 11.4 Write property test for atomic reprocess replacement
    - **Property 20: Reprocess replacement is atomic**
    - **Validates: Requirements 8.4**

  - [x] 11.5 Write property test for complete document removal
    - **Property 21: Document removal is complete**
    - **Validates: Requirements 8.5**

  - [x] 11.6 Write unit/integration tests for persistence failure, not-found, and latency
    - Test persistence-failure retains prior chunks and returns an error, removal of an unknown id returns not-found, and pgvector persistence latency (integration)
    - _Requirements: 8.1, 8.6, 8.8_

- [x] 12. Implement Retriever
  - [x] 12.1 Implement `Retriever.retrieve`
    - Validate query (reject empty / > 4000 chars; reject topK < 1 or > 100); return top-K (default 5) scored chunks with similarity in [0.0, 1.0]; include `{documentId, pageNumber}` metadata with placeholders when missing; apply metadata filter; return empty result + status for empty library / no filter match; order by descending similarity with ascending-documentId tie-break
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 9.8, 9.9_

  - [x] 12.2 Write property test for retrieval cardinality and score range
    - **Property 22: Retrieval cardinality and score range**
    - **Validates: Requirements 9.1**

  - [x] 12.3 Write property test for metadata filtering
    - **Property 23: Returned chunks satisfy the metadata filter**
    - **Validates: Requirements 9.7, 9.8**

  - [x] 12.4 Write property test for source metadata presence
    - **Property 24: Returned chunks always carry source metadata**
    - **Validates: Requirements 9.4, 9.5**

  - [x] 12.5 Write property test for result ordering and tie-break
    - **Property 25: Result ordering is by descending similarity with deterministic tie-break**
    - **Validates: Requirements 9.9**

  - [x] 12.6 Write unit tests for query validation and empty library
    - Test empty/oversized query rejection, out-of-range topK rejection, and empty-library status
    - _Requirements: 9.2, 9.3, 9.6_

- [x] 13. Implement Orchestrator with retry, resume, and observability
  - [x] 13.1 Implement sequential stage execution, progress, and transition recording
    - Execute parsing → normalization → chunking → embedding → storage, starting each stage only after the prior succeeds; persist each stage artifact; record `{stage, status, timestamp}` on every transition; expose current stage and integer progress percent (0..100); mark the job completed when all chunks are stored and failed (reporting unstored chunk ids) on storage failure
    - _Requirements: 8.3, 8.7, 10.1, 10.6, 10.7_

  - [x] 13.2 Implement retry policy, failure preservation, and resume
    - Retry a failed stage up to `stageRetryLimit` (0..10); on exhaustion mark the job failed, record the failing stage, and preserve completed-stage outputs; `resume` restarts from the recorded failing stage reusing preserved upstream artifacts; reject resume when there is no recorded failing stage
    - _Requirements: 10.2, 10.3, 10.4, 10.5_

  - [x] 13.3 Write property test for strictly sequential stage execution
    - **Property 26: Stage execution is strictly sequential**
    - **Validates: Requirements 10.1**

  - [x] 13.4 Write property test for bounded retry attempts
    - **Property 27: Retry attempts are bounded by the configured limit**
    - **Validates: Requirements 10.2**

  - [x] 13.5 Write property test for failure preservation and correct resume
    - **Property 28: Failure preserves completed-stage outputs and enables correct resume**
    - **Validates: Requirements 10.3, 10.4**

  - [x] 13.6 Write property test for stage transition recording
    - **Property 29: Every stage transition is fully recorded**
    - **Validates: Requirements 10.6**

  - [x] 13.7 Write property test for bounded, monotonic progress
    - **Property 30: Progress is bounded and monotonic**
    - **Validates: Requirements 10.7**

  - [x] 13.8 Write unit tests for non-resumable resume and job completion/failure
    - Test rejection of resume with no recorded failing stage, job completion on full storage, and job failure reporting unstored chunk ids
    - _Requirements: 8.3, 8.7, 10.5_

- [x] 14. Integration and end-to-end wiring
  - [x] 14.1 Wire the full pipeline together
    - Assemble `IngestionService`, `Parser` (+ engine registry and `OCRProcessor`), `Normalizer`, `Chunker`, `Embedder` (+ model registry), `VectorStore`, `Retriever`, and `Orchestrator` from a single `PipelineConfig`, exposing submit → process → retrieve entry points with no orphaned components
    - _Requirements: 8.3, 10.1_

  - [x] 14.2 Write end-to-end integration tests
    - Drive a submitted document through ingestion → storage to job completion with mock adapters, then retrieve; cover a reprocess/replace flow
    - _Requirements: 8.3, 8.4, 10.1_

- [x] 15. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional test tasks and can be skipped for a faster MVP.
- Each task references specific requirements clauses for traceability; property test tasks additionally cite the exact design property they validate.
- The two design-mandated anchor properties are Task 7.7 (Property 12, normalization round-trip) and Task 8.7 (Property 17, chunk completeness).
- Property-based tests use Hypothesis, run >= 100 iterations, and are tagged `Feature: biomedical-rag-pipeline, Property {number}: {property_text}`; each correctness property is covered by a single property test.
- Pluggable backends (parsing engines, embedding models, vector stores) are tested through their ports with deterministic mock/in-memory adapters; concrete adapters get a small number of integration tests.
- Checkpoints (Tasks 9 and 15) ensure incremental validation.

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1"] },
    { "id": 1, "tasks": ["2.1", "3.1"] },
    { "id": 2, "tasks": ["2.2", "3.2", "4.1", "5.1", "6.1", "10.1", "11.1"] },
    { "id": 3, "tasks": ["4.2", "5.2", "5.3", "7.1", "8.1", "10.2", "11.2", "12.1"] },
    { "id": 4, "tasks": ["4.3", "4.4", "4.5", "5.4", "7.2", "8.2", "10.3", "10.4", "11.3", "11.4", "11.5", "11.6", "12.2", "12.3", "12.4", "12.5", "12.6"] },
    { "id": 5, "tasks": ["5.5", "5.6", "5.7", "5.8", "6.2", "7.3", "8.3", "8.4", "8.5", "8.6", "8.7"] },
    { "id": 6, "tasks": ["6.3", "6.4", "6.5", "7.4", "7.5", "7.6", "7.7", "7.8", "13.1"] },
    { "id": 7, "tasks": ["13.2"] },
    { "id": 8, "tasks": ["13.3", "13.4", "13.5", "13.6", "13.7", "13.8", "14.1"] },
    { "id": 9, "tasks": ["14.2"] }
  ]
}
```
