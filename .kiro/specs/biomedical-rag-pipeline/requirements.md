# Requirements Document

## Introduction

This document specifies the requirements for an end-to-end Retrieval-Augmented Generation (RAG) pipeline focused on biomedical engineering content. The pipeline ingests biomedical source documents (EBooks, research papers, articles, and journals), parses them with advanced document parsers capable of extracting tables, charts/figures, and scanned-image text (OCR) while handling formatting artifacts, chunks the extracted content, generates vector embeddings, and stores them in a vector store to form a reusable knowledge library. This library then serves as the retrieval source that supplies relevant context to Large Language Models (LLMs) at query time.

The pipeline is organized into discrete, composable stages — Ingestion, Parsing, Normalization, Chunking, Embedding, Storage, and Retrieval — coordinated by an orchestration layer that supports incremental, resumable, and observable processing.

## Glossary

- **Pipeline**: The complete end-to-end system that transforms biomedical source documents into a queryable vector knowledge library and serves retrieved context to LLMs.
- **Ingestion_Service**: The component responsible for accepting, validating, and registering source documents for processing.
- **Source_Document**: A single biomedical input file (EBook, research paper, article, or journal) in a supported format (PDF, EPUB, DOCX, HTML).
- **Parser**: The component that converts a Source_Document into a structured Parsed_Document using an advanced parsing engine (for example, Docling or LlamaParse).
- **Parsing_Engine**: A configurable, pluggable third-party or library backend (for example, Docling or LlamaParse) used by the Parser to perform document extraction.
- **Parsed_Document**: The structured intermediate representation produced by the Parser, containing text blocks, tables, figures, image-derived text, and structural metadata.
- **OCR_Processor**: The component that extracts text from scanned images and image-only pages within a Source_Document.
- **Normalizer**: The component that cleans formatting artifacts and converts a Parsed_Document into a canonical normalized representation.
- **Formatting_Artifact**: Unwanted content introduced by document layout, such as repeated headers, footers, page numbers, line-break hyphenation, and column-flow disorder.
- **Chunker**: The component that splits a normalized document into retrievable Chunks.
- **Chunk**: A bounded segment of normalized content paired with its source metadata, used as the unit of embedding and retrieval.
- **Embedder**: The component that converts a Chunk into a numeric vector Embedding using an embedding model.
- **Embedding**: A fixed-dimension numeric vector representation of a Chunk.
- **Vector_Store**: The persistent store that holds Embeddings and associated Chunk metadata and supports similarity search.
- **Knowledge_Library**: The accumulated collection of stored Chunks and Embeddings across all processed Source_Documents.
- **Retriever**: The component that, given a query, returns the most relevant Chunks from the Vector_Store.
- **Orchestrator**: The component that coordinates execution of pipeline stages, tracks processing state, and manages retries.
- **Query**: A user- or system-supplied request used by the Retriever to fetch relevant Chunks.
- **Processing_Job**: A tracked unit of work representing the processing of one Source_Document through the pipeline.

## Requirements

### Requirement 1: Document Ingestion

**User Story:** As a knowledge engineer, I want to submit biomedical EBooks, papers, articles, and journals to the pipeline, so that they are registered and queued for processing.

#### Acceptance Criteria

1. WHEN a Source_Document in a supported format (PDF, EPUB, DOCX, or HTML) is submitted, THE Ingestion_Service SHALL create a Processing_Job, assign a job identifier that is unique across all Processing_Jobs, and return the job identifier within 5 seconds of submission.
2. THE Ingestion_Service SHALL accept Source_Documents in PDF, EPUB, DOCX, and HTML formats, where the document byte size is between 1 byte and 524,288,000 bytes (500 MB) inclusive.
3. IF a submitted Source_Document is in a format other than PDF, EPUB, DOCX, or HTML, THEN THE Ingestion_Service SHALL reject the submission without creating a Processing_Job and return an error message identifying the detected unsupported format.
4. WHEN a Source_Document is submitted, THE Ingestion_Service SHALL record document metadata including filename, format, byte size, content hash, and submission timestamp recorded in UTC.
5. IF a submitted Source_Document is identical to a previously ingested Source_Document, as determined by content hash, THEN THE Ingestion_Service SHALL reject the duplicate without creating a new Processing_Job and return the existing job identifier.
6. IF a submitted Source_Document exceeds 524,288,000 bytes (500 MB), THEN THE Ingestion_Service SHALL reject the submission without creating a Processing_Job and return an error message stating the 500 MB size limit.
7. IF a submitted Source_Document is empty (0 bytes) or cannot be read as a complete, well-formed file of its declared format, THEN THE Ingestion_Service SHALL reject the submission without creating a Processing_Job and return an error message indicating the document is empty or corrupted.
8. IF the submitted filename is empty or exceeds 255 characters, THEN THE Ingestion_Service SHALL reject the submission without creating a Processing_Job and return an error message indicating the filename is missing or exceeds the 255-character limit.

### Requirement 2: Advanced Document Parsing

**User Story:** As a knowledge engineer, I want documents parsed by an advanced parsing engine, so that text and document structure are accurately extracted from complex biomedical layouts.

#### Acceptance Criteria

1. WHEN a Processing_Job begins parsing, THE Parser SHALL convert the Source_Document into a Parsed_Document containing text blocks ordered in reading sequence and structural metadata identifying each block's type and position.
2. THE Parser SHALL support selection of the Parsing_Engine through configuration, including Docling and LlamaParse.
3. WHEN a Source_Document contains multiple columns, THE Parser SHALL order the produced text blocks top-to-bottom within each column and left-to-right across columns.
4. WHEN a Source_Document contains section headings, THE Parser SHALL preserve the heading hierarchy, including each heading's nesting level, in the Parsed_Document structural metadata.
5. IF the Parser fails to parse a Source_Document, THEN THE Parser SHALL mark the Processing_Job as failed, record the failure reason, and retain no partial Parsed_Document output.
6. WHERE a Parsing_Engine is unavailable, THE Parser SHALL mark the Processing_Job as failed and record the unavailable engine identifier.
7. IF parsing of a Source_Document exceeds a configurable maximum duration of 300 seconds, THEN THE Parser SHALL mark the Processing_Job as failed and record a timeout failure reason.
8. IF a Source_Document contains no extractable text, THEN THE Parser SHALL mark the Processing_Job as failed and record a no-extractable-content failure reason.

### Requirement 3: Table and Figure Extraction

**User Story:** As a knowledge engineer, I want tables and charts/figures extracted with their structure preserved, so that quantitative biomedical data remains usable for retrieval.

#### Acceptance Criteria

1. WHEN a Source_Document contains a table, THE Parser SHALL extract the table as a structured representation in which every non-empty source cell is assigned to exactly one row index and one column index.
2. IF a table contains a cell that spans multiple rows or columns, THEN THE Parser SHALL assign the cell's value to its top-left row and column index and record the spanned row count and column count for that cell.
3. WHEN a Source_Document contains a chart or figure, THE Parser SHALL extract the figure and, where a caption is present, its associated caption text.
4. IF a chart or figure has no detectable caption, THEN THE Parser SHALL extract the figure and record the caption as absent without failing the extraction.
5. WHEN the Parser extracts a table or figure, THE Parser SHALL associate the extracted item with its source page number and its zero-based position in document reading order.
6. IF a table cannot be extracted as a structured representation, THEN THE Parser SHALL record the extraction as degraded, retain the raw text content of the table region, and produce an indication that the item failed structured extraction.

### Requirement 4: Scanned Image and OCR Processing

**User Story:** As a knowledge engineer, I want text recovered from scanned pages and embedded images, so that older or image-based biomedical sources are searchable.

#### Acceptance Criteria

1. WHEN a Source_Document page contains no extractable text layer, THE OCR_Processor SHALL extract text from the page image and store the recovered text in the Parsed_Document.
2. WHEN a Source_Document contains an embedded image with text content, THE OCR_Processor SHALL extract the text from the image and store the recovered text in the Parsed_Document.
3. THE OCR_Processor SHALL record a confidence score, expressed as a value from 0.0 to 1.0, for each OCR-extracted text block.
4. IF an OCR-extracted text block has a confidence score below the configured threshold (default 0.70, configurable within the range 0.0 to 1.0), THEN THE OCR_Processor SHALL flag the text block as low-confidence in the Parsed_Document while retaining the extracted text.
5. IF the OCR_Processor cannot extract text from a page image or embedded image because the image is unreadable, corrupt, or in an unsupported format, THEN THE OCR_Processor SHALL record an error indication identifying the affected image in the Parsed_Document and SHALL continue processing the remaining pages and images.
6. WHEN OCR processing of a single page image exceeds 60 seconds, THE OCR_Processor SHALL abort OCR for that page, record a timeout indication for the affected page in the Parsed_Document, and continue processing the remaining pages.

### Requirement 5: Formatting Artifact Normalization

**User Story:** As a knowledge engineer, I want formatting artifacts removed and content canonicalized, so that chunks contain clean, coherent biomedical text.

#### Acceptance Criteria

1. WHEN the Normalizer processes a Parsed_Document, THE Normalizer SHALL remove text elements that recur in the header or footer region on 2 or more pages, including page numbers.
2. WHEN a word is split across a line break by a trailing hyphen and the resulting joined token matches a known dictionary word, THE Normalizer SHALL rejoin the word into a single token without the hyphen.
3. IF a hyphen occurs within a single line and is not at a line break (an intrinsic hyphen), THEN THE Normalizer SHALL retain the hyphen and the original token unchanged.
4. THE Normalizer SHALL produce a canonical normalized representation that preserves all source content without loss, including heading hierarchy, tables, figures, and OCR-derived text.
5. WHILE normalizing, THE Normalizer SHALL preserve the source page number and reading-order position for each content element.
6. FOR ALL Parsed_Documents, serializing a normalized representation and then deserializing the serialized form SHALL produce a normalized representation equivalent to the original, where equivalence means identical content elements, identical heading hierarchy, identical table and figure structures, and identical page-number and reading-order metadata for every element.
7. IF a Parsed_Document is empty or contains no recognizable content elements, THEN THE Normalizer SHALL produce an empty normalized representation and return an indication that no content was available for normalization.
8. IF a Parsed_Document is malformed such that its structure cannot be interpreted, THEN THE Normalizer SHALL reject the document, leave any prior valid output unchanged, and return an error indication identifying the document as malformed.

### Requirement 6: Content Chunking

**User Story:** As a knowledge engineer, I want normalized content split into retrievable chunks, so that retrieval returns focused, relevant context.

#### Acceptance Criteria

1. WHEN the Chunker processes a normalized representation, THE Chunker SHALL produce Chunks whose token counts do not exceed the configured maximum chunk size, where the configured maximum chunk size is a value between 128 and 2048 tokens inclusive.
2. THE Chunker SHALL produce Chunks with a configured token overlap between consecutive Chunks, where the configured token overlap is a value between 0 and the configured maximum chunk size minus 1 tokens inclusive.
3. THE Chunker SHALL attach source metadata to each Chunk, including document identifier, page number, and heading path.
4. WHEN a table is present in the normalized representation and the table content fits within the configured maximum chunk size, THE Chunker SHALL keep the table content within a single Chunk.
5. FOR ALL normalized representations, the concatenation of Chunk contents in order, with overlap removed, SHALL contain all non-artifact text from the normalized representation (completeness property).
6. IF a table is present in the normalized representation and the table content exceeds the configured maximum chunk size, THEN THE Chunker SHALL split the table content across multiple Chunks such that each resulting Chunk does not exceed the configured maximum chunk size.
7. IF the page number or heading path is unavailable for a Chunk, THEN THE Chunker SHALL attach an empty value for the unavailable metadata field while still attaching the document identifier.
8. IF the normalized representation contains no non-artifact text, THEN THE Chunker SHALL produce zero Chunks.

### Requirement 7: Embedding Generation

**User Story:** As a knowledge engineer, I want chunks converted into vector embeddings, so that semantic similarity search is possible.

#### Acceptance Criteria

1. WHEN a Chunk is provided to the Embedder, THE Embedder SHALL produce an Embedding whose dimension equals the configured dimension, where the configured dimension is an integer between 64 and 4096 inclusive.
2. WHEN a Chunk is provided to the Embedder, THE Embedder SHALL produce the corresponding Embedding within 10 seconds.
3. THE Embedder SHALL support selection of the embedding model through configuration.
4. IF the configured embedding model is missing or not recognized, THEN THE Embedder SHALL reject the embedding request, produce an error indication identifying the invalid model configuration, and leave the affected Chunk unmodified.
5. FOR ALL Chunks embedded with the same configured embedding model, THE Embedder SHALL produce Embeddings of identical dimension.
6. IF the Embedder fails to generate an Embedding for a Chunk, THEN THE Embedder SHALL record the failure with an error indication describing the cause and SHALL retain the Chunk for retry for up to 3 retry attempts.
7. IF the Embedder fails to generate an Embedding for a Chunk after 3 retry attempts, THEN THE Embedder SHALL mark the Chunk as failed and SHALL retain the original Chunk content unmodified.

### Requirement 8: Vector Storage and Knowledge Library

**User Story:** As a knowledge engineer, I want embeddings and chunks persisted, so that a reusable biomedical knowledge library is built and maintained.

#### Acceptance Criteria

1. WHEN an Embedding is generated for a Chunk, THE Vector_Store SHALL persist the Embedding together with the Chunk content and the complete Chunk metadata within 5 seconds of generation.
2. THE Vector_Store SHALL associate each stored Embedding with its source document identifier such that every stored Embedding is retrievable by that identifier.
3. WHEN all Chunks of a Processing_Job are successfully stored, THE Orchestrator SHALL mark the Processing_Job as completed.
4. WHEN a Source_Document is reprocessed, THE Vector_Store SHALL replace the previously stored Chunks for that source document identifier only after the newly generated Chunks are successfully stored, retaining the previous Chunks if replacement does not complete.
5. WHERE a document is removed from the Knowledge_Library, THE Vector_Store SHALL delete all Chunks and Embeddings associated with that source document identifier.
6. IF persisting an Embedding or Chunk fails, THEN THE Vector_Store SHALL retain any previously stored Chunks for that source document identifier and return an error response indicating the persistence failure.
7. IF one or more Chunks of a Processing_Job fail to be stored, THEN THE Orchestrator SHALL mark the Processing_Job as failed and return an error response indicating which Chunks were not stored.
8. IF a document removal is requested for a source document identifier that has no associated Chunks or Embeddings, THEN THE Vector_Store SHALL return an error response indicating the document was not found.

### Requirement 9: Retrieval for LLM Ingestion

**User Story:** As an application developer, I want to retrieve the most relevant chunks for a query, so that I can supply grounded context to an LLM.

#### Acceptance Criteria

1. WHEN a Query is submitted to the Retriever, THE Retriever SHALL return the configured number of most similar Chunks (default 5, configurable from 1 to 100) from the Vector_Store, each with a similarity score in the range 0.0 to 1.0, within 2 seconds.
2. IF a Query is submitted with empty text or text exceeding 4000 characters, THEN THE Retriever SHALL reject the Query, return no Chunks, and return a status indicating the Query is invalid.
3. IF the configured number of Chunks to return is less than 1 or greater than 100, THEN THE Retriever SHALL reject the Query and return a status indicating the requested count is out of range.
4. THE Retriever SHALL include source metadata with each returned Chunk, comprising the document identifier and the page number.
5. WHERE source metadata is missing for a returned Chunk, THE Retriever SHALL return a placeholder value indicating the document identifier and page number are unavailable.
6. IF the Knowledge_Library contains no Chunks, THEN THE Retriever SHALL return an empty result set and a status indicating the library is empty.
7. WHERE a metadata filter is supplied with a Query, THE Retriever SHALL restrict returned Chunks to those matching the filter.
8. IF a supplied metadata filter matches no Chunks, THEN THE Retriever SHALL return an empty result set and a status indicating no Chunks matched the filter.
9. WHEN the Retriever returns results, THE Retriever SHALL order the results by descending similarity score, and for Chunks with equal similarity scores SHALL order them by ascending document identifier.

### Requirement 10: Pipeline Orchestration and Observability

**User Story:** As a pipeline operator, I want the stages coordinated, resumable, and observable, so that large biomedical corpora are processed reliably.

#### Acceptance Criteria

1. WHEN a Processing_Job is created, THE Orchestrator SHALL execute the parsing, normalization, chunking, embedding, and storage stages in sequential order, starting each stage only after the immediately preceding stage has completed successfully.
2. IF a stage fails for a Processing_Job, THEN THE Orchestrator SHALL retry the failed stage up to the configured retry limit, where the retry limit is an integer between 0 and 10 with a default of 3.
3. IF a stage fails after reaching the configured retry limit, THEN THE Orchestrator SHALL mark the Processing_Job as failed, record the identifier of the failing stage, and preserve the output of all previously completed stages.
4. WHEN a previously failed Processing_Job is resumed, THE Orchestrator SHALL restart processing from the recorded failing stage rather than from the first stage, reusing the preserved output of all stages completed before the failing stage.
5. IF a resume is requested for a Processing_Job that has no recorded failing stage, THEN THE Orchestrator SHALL reject the request and return an error response indicating that the Processing_Job is not in a resumable state.
6. WHEN a Processing_Job transitions between stages, THE Orchestrator SHALL record the stage identifier, the resulting stage status (one of: pending, running, succeeded, failed), and a timestamp of the transition.
7. WHILE a Processing_Job is running, THE Orchestrator SHALL expose the current stage identifier and a completion progress value expressed as an integer percentage from 0 to 100.
