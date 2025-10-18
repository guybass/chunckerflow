# ChunkFlow

**Production-grade async text chunking framework for RAG systems**

[![PyPI version](https://badge.fury.io/py/chunk-flow.svg)](https://badge.fury.io/py/chunk-flow)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/chunk-flow)](https://pypi.org/project/chunk-flow/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/chunk-flow)](https://pypi.org/project/chunk-flow/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![GitHub stars](https://img.shields.io/github/stars/chunkflow/chunk-flow?style=social)](https://github.com/chunkflow/chunk-flow)

ChunkFlow is a comprehensive, extensible framework for text chunking in Retrieval-Augmented Generation (RAG) systems. Built with production-grade practices, it provides multiple chunking strategies, pluggable embedding providers, and comprehensive evaluation metrics to help you make data-driven decisions.

## Why ChunkFlow?

RAG systems process billions of documents daily, and **chunking quality directly impacts retrieval accuracy, computational costs, and user experience**. Poor chunking causes hallucinations, missed context, and wasted API calls.

ChunkFlow addresses this with:
- **6+ chunking strategies** - From simple fixed-size to revolutionary late chunking
- **Pluggable architecture** - Easy integration with any embedding provider
- **Comprehensive evaluation** - 12+ metrics including RAGAS-inspired, NDCG, semantic coherence
- **Data-driven comparison** - Built-in strategy comparison and ranking framework
- **Production-ready** - Async-first, type-safe, structured logging, extensible design

## Key Features

### Chunking Strategies

- **Fixed-Size** - Simple character/token-based splitting (10K+ chunks/sec)
- **Recursive** - Hierarchical splitting with natural boundaries (recommended default)
- **Document-Based** - Format-aware (Markdown, HTML)
- **Semantic** - Embedding-based topic detection with similarity thresholds
- **Late Chunking** - Revolutionary context-preserving approach (6-9% accuracy improvement, Jina AI 2024)

### Embedding Providers

- **OpenAI** - text-embedding-3-small/large with automatic cost tracking
- **HuggingFace** - Sentence Transformers (local, free, GPU/CPU support)
- **Extensible** - Easy to add custom providers via EmbeddingProvider base class

### Evaluation Metrics

- **Retrieval** (4 metrics): NDCG@k, Recall@k, Precision@k, MRR
- **Semantic** (4 metrics): Coherence, Boundary Quality, Chunk Stickiness (MoC), Topic Diversity
- **RAG Quality** (4 metrics): Context Relevance, Answer Faithfulness, Context Precision, Context Recall (RAGAS-inspired)
- **Framework**: Unified EvaluationPipeline + StrategyComparator for comprehensive analysis

## Quick Start

### Installation

```bash
# Basic installation
pip install chunk-flow

# With specific providers
pip install chunk-flow[openai]
pip install chunk-flow[huggingface]

# With API server
pip install chunk-flow[api]

# Everything
pip install chunk-flow[all]
```

### Basic Usage

```python
from chunk_flow.chunking import StrategyRegistry
from chunk_flow.embeddings import EmbeddingProviderFactory
from chunk_flow.evaluation import EvaluationPipeline

# 1. Chunk your document
chunker = StrategyRegistry.create("recursive", {"chunk_size": 512, "overlap": 100})
result = await chunker.chunk(document)

# 2. Embed chunks
embedder = EmbeddingProviderFactory.create("openai", {"model": "text-embedding-3-small"})
emb_result = await embedder.embed_texts(result.chunks)

# 3. Evaluate quality (semantic metrics - no ground truth needed)
pipeline = EvaluationPipeline(metrics=["semantic_coherence", "boundary_quality", "chunk_stickiness"])
metrics = await pipeline.evaluate(
    chunks=result.chunks,
    embeddings=emb_result.embeddings,
)

print(f"Semantic Coherence: {metrics['semantic_coherence'].score:.4f}")
print(f"Boundary Quality: {metrics['boundary_quality'].score:.4f}")
```

### Strategy Comparison

Compare multiple strategies to find the best for your use case:

```python
from chunk_flow.chunking import StrategyRegistry
from chunk_flow.embeddings import EmbeddingProviderFactory
from chunk_flow.evaluation import EvaluationPipeline, StrategyComparator

# Create strategies to compare
strategies = [
    StrategyRegistry.create("fixed_size", {"chunk_size": 500, "overlap": 50}),
    StrategyRegistry.create("recursive", {"chunk_size": 400, "overlap": 80}),
    StrategyRegistry.create("semantic", {"threshold_percentile": 80}),
]

# Get embedder
embedder = EmbeddingProviderFactory.create("huggingface")

# Set up evaluation pipeline
pipeline = EvaluationPipeline(
    metrics=["ndcg_at_k", "semantic_coherence", "chunk_stickiness"],
)

# Compare strategies
comparison = await pipeline.compare_strategies(
    strategies=strategies,
    text=document,
    ground_truth={"query_embedding": query_emb, "relevant_indices": [0, 2, 5]},
)

# Generate comparison report
report = StrategyComparator.generate_comparison_report(
    {name: comparison["strategies"][name]["metric_results"]
     for name in comparison["strategies"].keys()}
)
print(report)

# See examples/strategy_comparison.py for complete working example
```

### API Server

```bash
# Start FastAPI server
uvicorn chunk_flow.api.app:app --reload

# Use the API
curl -X POST "http://localhost:8000/chunk" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Your document here...",
    "strategy": "recursive",
    "strategy_config": {"chunk_size": 512}
  }'
```

## Architecture

ChunkFlow follows a clean, extensible architecture:

```
┌─────────────────────────────────────────────────────────────┐
│                     API Layer (FastAPI)                     │
│  /chunk, /evaluate, /compare, /benchmark, /export          │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    Orchestration Layer                      │
│  ChunkingPipeline | EvaluationEngine | ResultsAggregator   │
└─────────────────────────────────────────────────────────────┘
                              ↓
        ┌─────────────────────┴─────────────────────┐
        ↓                                           ↓
┌──────────────────┐                    ┌──────────────────────┐
│ Chunking Module  │                    │  Embedding Module    │
│ ----------------│                    │ -------------------- │
│ • Fixed-Size    │                    │ • OpenAI             │
│ • Recursive     │                    │ • HuggingFace        │
│ • Semantic      │                    │ • Google Vertex      │
│ • Late          │                    │ • Cohere             │
└──────────────────┘                    └──────────────────────┘
```

## Research-Backed

ChunkFlow implements cutting-edge research findings:

- **Late Chunking** (Jina AI, 2025): 6-9% improvement in retrieval accuracy
- **Optimal Chunk Sizes** (Bhat et al., 2025): 64-128 tokens for facts, 512-1024 for context
- **Semantic Independence** (HOPE, 2025): 56% gain in factual correctness
- **MoC Metrics** (Zhao et al., 2025): Boundary clarity and chunk stickiness
- **RAGAS** (ExplodingGradients, 2023): Reference-free RAG evaluation

See [rag-summery-framework.md](rag-summery-framework.md) for comprehensive research review.

## Documentation

- 📚 **[Documentation Hub](docs/README.md)** - Complete documentation index
- 🚀 **[Getting Started](docs/GETTING_STARTED.md)** - Installation and quick start
- 📖 **[API Reference](docs/API_REFERENCE.md)** - Complete API documentation
- 🐳 **[Docker Guide](docs/DOCKER.md)** - Docker deployment
- 📓 **[Examples](examples/)** - Code examples and Jupyter notebooks

## Development

```bash
# Clone repository
git clone https://github.com/chunkflow/chunk-flow.git
cd chunk-flow

# Install with dev dependencies
make install-dev

# Run tests
make test

# Format and lint
make format
make lint

# Run full CI locally
make ci
```

## Contributing

ChunkFlow is currently a solo project. While contributions are not being accepted at this time, you can:

- **Report Bugs**: [GitHub Issues](https://github.com/chunkflow/chunk-flow/issues)
- **Request Features**: [GitHub Issues](https://github.com/chunkflow/chunk-flow/issues)
- **Ask Questions**: [GitHub Discussions](https://github.com/chunkflow/chunk-flow/discussions)
- **Star the Repo**: Help spread the word!

See [CONTRIBUTING.md](CONTRIBUTING.md) for more details.

## Roadmap

**Phase 1-4: Core Framework** ✅ COMPLETED
- [x] Core chunking strategies (Fixed, Recursive, Document-based)
- [x] Embedding providers (OpenAI, HuggingFace)
- [x] Semantic chunking
- [x] Late chunking implementation
- [x] Comprehensive evaluation metrics (12 metrics across 3 categories)
- [x] Evaluation pipeline and comparison framework

**Phase 5-6: Analysis & API** ✅ COMPLETED
- [x] ResultsDataFrame with analysis methods
- [x] Visualization utilities (heatmaps, comparison charts)
- [x] FastAPI server with all endpoints
- [x] Docker setup (multi-stage, production-ready)

**Phase 7-9: Testing & Release** ✅ COMPLETED
- [x] Comprehensive testing (unit, integration, E2E)
- [x] Benchmark suite with standard datasets
- [x] CI/CD pipeline (GitHub Actions)
- [x] Complete documentation
- [x] PyPI package release workflow
- [x] Production deployment guides

**v0.1.0 READY FOR RELEASE!** 🚀

**Future Roadmap (v0.2.0+)**
- [ ] Additional providers (Google Vertex, Cohere, Voyage AI)
- [ ] LLM-based chunking (GPT/Claude)
- [ ] Streamlit dashboard
- [ ] Redis caching and PostgreSQL storage
- [ ] Agentic chunking with dynamic boundaries
- [ ] Fine-tuning pipeline for custom strategies
- [ ] Public benchmark datasets (BeIR, MS MARCO)

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Citation

If you use ChunkFlow in your research, please cite:

```bibtex
@software{chunkflow2024,
  title = {ChunkFlow: Production-Grade Text Chunking Framework for RAG Systems},
  author = {ChunkFlow Development},
  year = {2024},
  url = {https://github.com/chunkflow/chunk-flow}
}
```

## Acknowledgments

ChunkFlow builds on research from Jina AI, ExplodingGradients, and the broader RAG community. Built with passion for the neglected field of text chunking.

---

**Built with passion for the neglected field of text chunking** 🚀

[Documentation](https://chunk-flow.readthedocs.io) | [GitHub](https://github.com/chunkflow/chunk-flow) | [PyPI](https://pypi.org/project/chunk-flow/)
