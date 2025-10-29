# Getting Started with ChunkFlow

## Installation

### Basic Installation

```bash
pip install chunk-flow
```

### With Optional Dependencies

```bash
# For API server
pip install chunk-flow[api]

# For HuggingFace embeddings (local, free)
pip install chunk-flow[huggingface]

# For OpenAI embeddings
pip install chunk-flow[openai]

# For visualization
pip install chunk-flow[viz]

# Everything
pip install chunk-flow[all]
```

### Development Installation

```bash
git clone https://github.com/chunkflow/chunk-flow.git
cd chunk-flow
pip install -e .[dev]

# Set up pre-commit hooks
pre-commit install
```

## Quick Start

### 1. Basic Chunking

```python
import asyncio
from chunk_flow.chunking import StrategyRegistry

async def main():
    # Create a chunker
    chunker = StrategyRegistry.create(
        "recursive",
        {"chunk_size": 500, "overlap": 100}
    )

    # Chunk your text
    text = "Your document here..."
    result = await chunker.chunk(text)

    print(f"Created {len(result.chunks)} chunks")
    for i, chunk in enumerate(result.chunks, 1):
        print(f"Chunk {i}: {chunk[:100]}...")

asyncio.run(main())
```

### 2. Generate Embeddings

```python
from chunk_flow.embeddings import EmbeddingProviderFactory

# Use HuggingFace (local, free)
embedder = EmbeddingProviderFactory.create(
    "huggingface",
    {"model": "sentence-transformers/all-MiniLM-L6-v2"}
)

# Or use OpenAI
embedder = EmbeddingProviderFactory.create(
    "openai",
    {"model": "text-embedding-3-small"}
)

# Generate embeddings
emb_result = await embedder.embed_texts(result.chunks)
print(f"Generated {len(emb_result.embeddings)} embeddings ({emb_result.dimensions}D)")
```

### 3. Evaluate Quality

```python
from chunk_flow.evaluation import EvaluationPipeline

# Create evaluation pipeline
pipeline = EvaluationPipeline(
    metrics=["semantic_coherence", "boundary_quality", "chunk_stickiness"]
)

# Evaluate chunks
eval_results = await pipeline.evaluate(
    chunks=result.chunks,
    embeddings=emb_result.embeddings,
)

for metric_name, metric_result in eval_results.items():
    print(f"{metric_name}: {metric_result.score:.4f}")
```

### 4. Compare Strategies

```python
from chunk_flow.evaluation import StrategyComparator
from chunk_flow.analysis import ResultsDataFrame

# Compare multiple strategies
strategies = [
    StrategyRegistry.create("fixed_size", {"chunk_size": 500}),
    StrategyRegistry.create("recursive", {"chunk_size": 500}),
    StrategyRegistry.create("semantic", {"threshold_percentile": 80}),
]

comparison = await pipeline.compare_strategies(
    strategies=strategies,
    text=document,
)

# Generate report
report = StrategyComparator.generate_comparison_report(
    {name: comparison["strategies"][name]["metric_results"]
     for name in comparison["strategies"].keys()}
)
print(report)
```

## Common Use Cases

### Use Case 1: RAG System Chunking

```python
# Recommended: Recursive chunking with overlap
chunker = StrategyRegistry.create(
    "recursive",
    {
        "chunk_size": 512,
        "overlap": 100,
        "separators": ["\n\n", "\n", ". ", " "]
    }
)
```

### Use Case 2: Research Paper Processing

```python
# Use markdown chunking for structured documents
chunker = StrategyRegistry.create(
    "markdown",
    {
        "respect_headers": True,
        "min_chunk_size": 200,
        "max_chunk_size": 1000,
    }
)
```

### Use Case 3: Long-Form Context Preservation

```python
# Use late chunking for 8K+ context models
chunker = StrategyRegistry.create(
    "late",
    {
        "model_name": "jinaai/jina-embeddings-v2-base-en",
        "chunk_size": 256,
    }
)
```

## Configuration

### Environment Variables

Create a `.env` file:

```env
# API Keys
CHUNK_FLOW_OPENAI_API_KEY=sk-...
CHUNK_FLOW_GOOGLE_API_KEY=...

# Logging
CHUNK_FLOW_LOG_LEVEL=INFO
CHUNK_FLOW_LOG_FORMAT=pretty

# Defaults
CHUNK_FLOW_DEFAULT_STRATEGY=recursive
CHUNK_FLOW_DEFAULT_EMBEDDING_PROVIDER=huggingface
```

### Configuration File

Create `config/default.yaml`:

```yaml
strategies:
  recursive:
    chunk_size: 512
    overlap: 100

  semantic:
    threshold_percentile: 80
    min_chunk_size: 100

embeddings:
  huggingface:
    model: "sentence-transformers/all-MiniLM-L6-v2"
    device: "cpu"
```

## Next Steps

- **[API Documentation](API.md)** - Complete API reference
- **[Strategies Guide](strategies.md)** - Detailed strategy documentation
- **[Metrics Guide](metrics.md)** - Understanding evaluation metrics
- **[Examples](../examples/)** - Working code examples
- **[Tutorials](tutorials/)** - Step-by-step guides

## Troubleshooting

### Import Errors

```bash
# Make sure you have the right optional dependencies
pip install chunk-flow[huggingface]  # For sentence-transformers
pip install chunk-flow[openai]        # For OpenAI
```

### Memory Issues

```python
# Use batch processing for large datasets
from chunk_flow.utils.async_helpers import AsyncBatchProcessor

processor = AsyncBatchProcessor(batch_size=100, max_concurrency=5)
```

### Performance Optimization

```python
# Use fixed-size for maximum speed
chunker = StrategyRegistry.create("fixed_size", {"chunk_size": 500})

# Or use multiprocessing for CPU-bound tasks
import multiprocessing
multiprocessing.set_start_method('spawn')
```

## Getting Help

- **GitHub Issues**: [Report bugs or request features](https://github.com/chunkflow/chunk-flow/issues)
- **Discussions**: [Ask questions](https://github.com/chunkflow/chunk-flow/discussions)
- **Documentation**: [Full docs](https://chunk-flow.readthedocs.io)
