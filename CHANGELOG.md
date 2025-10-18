# Changelog

All notable changes to ChunkFlow will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2025-10-18

### Added

**Core Framework (Phases 1-4)**
- Base architecture with `ChunkingStrategy`, `EmbeddingProvider`, and `EvaluationMetric` base classes
- Configuration management with pydantic-settings and YAML support
- Structured logging with structlog (JSON and pretty formats)
- Async utilities: `gather_with_concurrency`, `retry_async`, `AsyncBatchProcessor`

**Chunking Strategies (6 strategies)**
- `FixedSizeChunker` - Character/token-based splitting (10K+ chunks/sec)
- `RecursiveCharacterChunker` - Hierarchical splitting with natural boundaries (recommended default)
- `MarkdownChunker` - Header-aware markdown processing
- `HTMLChunker` - Tag-based HTML chunking
- `SemanticChunker` - Embedding-based topic detection
- `LateChunker` - Revolutionary context-preserving approach (6-9% accuracy improvement, Jina AI 2024)

**Embedding Providers (2 providers)**
- `OpenAIEmbeddingProvider` - text-embedding-3-small/large with automatic cost tracking
- `HuggingFaceEmbeddingProvider` - Sentence Transformers (local, free, GPU/CPU support)

**Evaluation Metrics (12 metrics across 3 categories)**
- Retrieval metrics: NDCG@k, Recall@k, Precision@k, MRR
- Semantic metrics: Coherence, Boundary Quality, Chunk Stickiness (MoC), Topic Diversity
- RAG quality metrics: Context Relevance, Answer Faithfulness, Context Precision, Context Recall (RAGAS-inspired)
- `EvaluationPipeline` for orchestrating multi-metric evaluation
- `StrategyComparator` for comprehensive strategy analysis

**Analysis & Visualization (Phase 5)**
- `ResultsDataFrame` - Pandas-based analysis with 15+ methods
  - Ranking, filtering, aggregation, statistical analysis
  - Export to CSV, JSON, Parquet, Excel
  - Correlation analysis and summary statistics
- `StrategyVisualizer` - 7 visualization types
  - Heatmaps, bar charts, radar charts, box plots
  - Correlation matrices, scatter plots, automated dashboards
  - Publication-quality plots (matplotlib/seaborn)

**REST API & Deployment (Phase 6)**
- FastAPI application with 10+ endpoints
  - `/chunk` - Chunk text with any strategy
  - `/embed` - Generate embeddings
  - `/evaluate` - Evaluate chunks with metrics
  - `/compare` - Compare multiple strategies
  - `/health`, `/strategies`, `/metrics`, `/providers` - Discovery endpoints
- OpenAPI documentation (Swagger UI at `/docs`, ReDoc at `/redoc`)
- CORS middleware and global exception handling
- Multi-stage Docker setup for production deployment
- docker-compose.yml for local development
- Kubernetes deployment examples

**Testing & Quality (Phase 7)**
- Comprehensive test suite with pytest
  - Unit tests for core components
  - Integration tests for pipelines
  - API endpoint tests
- Benchmark suite with standard test documents
- Performance profiling and throughput analysis

**CI/CD & Infrastructure (Phase 8)**
- GitHub Actions workflows
  - CI pipeline: code quality, tests across Python 3.9-3.12, security scans
  - Release pipeline: PyPI publishing, Docker image publishing
- Pre-commit hooks (Black, isort, Ruff, mypy)
- Code coverage reporting (Codecov integration)
- Security scanning (Bandit, Safety)

**Documentation (Phase 8)**
- Comprehensive README with quick start
- GETTING_STARTED.md - Installation and basic usage
- API_REFERENCE.md - Complete API documentation
- DOCKER.md - Docker deployment guide
- 5+ working examples
  - Basic usage, chunking with embeddings
  - Strategy comparison, analysis & visualization
  - API client example

**Development Tools**
- Makefile with common commands
- .env.example for configuration
- Production-grade .gitignore and .gitattributes
- CONTRIBUTING.md with contribution guidelines

### Dependencies

**Core**
- pydantic >= 2.0.0
- pandas >= 2.0.0
- numpy >= 1.24.0
- scikit-learn >= 1.3.0
- scipy >= 1.11.0
- httpx >= 0.24.0
- structlog >= 23.1.0

**Optional**
- FastAPI >= 0.104.0 (API server)
- sentence-transformers >= 2.2.0 (HuggingFace provider)
- openai >= 1.0.0 (OpenAI provider)
- matplotlib >= 3.7.0, seaborn >= 0.12.0 (Visualization)

### Known Limitations

- Late chunking requires 8K+ context embedding models
- Semantic chunking requires sentence-transformers
- Some metrics require ground truth data for evaluation
- Large document processing may require memory optimization

### Breaking Changes

None (initial release)

---

## [Unreleased]

### Planned Features

- Additional embedding providers (Google Vertex, Cohere, Voyage AI)
- LLM-based chunking strategies
- Agentic chunking with dynamic boundary detection
- Redis caching for API responses
- PostgreSQL backend for result storage
- Streamlit dashboard for interactive exploration
- Fine-tuning pipeline for custom strategies
- Model-based chunking (trainable)
- Expanded benchmark suite with public datasets
- Video tutorials and Jupyter notebooks

---

[0.1.0]: https://github.com/chunkflow/chunk-flow/releases/tag/v0.1.0
