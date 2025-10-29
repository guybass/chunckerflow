# API Reference

## REST API Endpoints

Base URL: `http://localhost:8000`

### System Endpoints

#### GET /health

Health check endpoint with system information.

**Response:**
```json
{
  "status": "healthy",
  "version": "0.1.0",
  "available_strategies": ["fixed_size", "recursive", "markdown", "semantic", "late"],
  "available_metrics": ["ndcg_at_k", "semantic_coherence", ...],
  "available_providers": ["huggingface", "openai"]
}
```

#### GET /strategies

List all available chunking strategies with metadata.

**Response:**
```json
{
  "strategies": [
    {
      "name": "recursive",
      "version": "1.0.0",
      "description": "Hierarchical splitting with natural boundaries"
    }
  ]
}
```

#### GET /metrics

List all available metrics organized by category.

**Response:**
```json
{
  "metrics": [...],
  "by_category": {
    "retrieval": ["ndcg_at_k", "recall_at_k", "precision_at_k", "mrr"],
    "semantic": ["semantic_coherence", "boundary_quality", ...],
    "rag_quality": ["context_relevance", ...]
  }
}
```

### Chunking Endpoints

#### POST /chunk

Chunk text using specified strategy.

**Request:**
```json
{
  "text": "Your document here...",
  "strategy": "recursive",
  "config": {
    "chunk_size": 512,
    "overlap": 100
  },
  "doc_id": "optional-doc-id"
}
```

**Response:**
```json
{
  "chunks": ["chunk1", "chunk2", ...],
  "num_chunks": 5,
  "processing_time_ms": 45.2,
  "strategy": "recursive",
  "strategy_version": "1.0.0",
  "metadata": [...]
}
```

### Embedding Endpoints

#### POST /embed

Generate embeddings for texts.

**Request:**
```json
{
  "texts": ["text1", "text2", ...],
  "provider": "huggingface",
  "config": {
    "model": "sentence-transformers/all-MiniLM-L6-v2"
  }
}
```

**Response:**
```json
{
  "embeddings": [[0.1, 0.2, ...], ...],
  "dimensions": 384,
  "token_count": 150,
  "processing_time_ms": 123.4,
  "provider": "huggingface",
  "model": "sentence-transformers/all-MiniLM-L6-v2",
  "cost_usd": null
}
```

### Evaluation Endpoints

#### POST /evaluate

Evaluate chunks with specified metrics.

**Request:**
```json
{
  "chunks": ["chunk1", "chunk2", ...],
  "embeddings": [[0.1, 0.2, ...], ...],
  "metrics": ["semantic_coherence", "boundary_quality"],
  "ground_truth": null,
  "context": null
}
```

**Response:**
```json
{
  "results": {
    "semantic_coherence": {
      "score": 0.85,
      "version": "1.0.0",
      "details": {...}
    }
  },
  "num_metrics": 2,
  "processing_time_ms": 89.3
}
```

#### POST /compare

Compare multiple chunking strategies.

**Request:**
```json
{
  "text": "Document to chunk...",
  "strategies": [
    {"name": "fixed_size", "config": {"chunk_size": 500}},
    {"name": "recursive", "config": {"chunk_size": 500}}
  ],
  "embedding_provider": "huggingface",
  "metrics": ["semantic_coherence"]
}
```

**Response:**
```json
{
  "strategies": {
    "fixed_size": {
      "num_chunks": 10,
      "metric_results": {...}
    },
    "recursive": {
      "num_chunks": 8,
      "metric_results": {...}
    }
  },
  "rankings": {...},
  "best_strategy": "recursive",
  "processing_time_ms": 234.5
}
```

## Python API

### Chunking Strategies

#### StrategyRegistry

```python
from chunk_flow.chunking import StrategyRegistry

# List available strategies
strategies = StrategyRegistry.get_strategy_names()

# Get strategy class
FixedSizeChunker = StrategyRegistry.get("fixed_size")

# Create strategy instance
chunker = StrategyRegistry.create("recursive", {"chunk_size": 512})

# Check if registered
is_available = StrategyRegistry.is_registered("semantic")
```

#### ChunkingStrategy Base Class

All strategies inherit from `ChunkingStrategy`:

```python
async def chunk(self, text: str, doc_id: Optional[str] = None) -> ChunkResult
```

**Returns:** `ChunkResult` with:
- `chunks: List[str]` - The text chunks
- `metadata: List[ChunkMetadata]` - Metadata for each chunk
- `processing_time_ms: float` - Processing time
- `strategy_version: str` - Strategy version
- `config: Dict[str, Any]` - Configuration used

### Embedding Providers

#### EmbeddingProviderFactory

```python
from chunk_flow.embeddings import EmbeddingProviderFactory

# List providers
providers = EmbeddingProviderFactory.list_providers()

# Create provider
embedder = EmbeddingProviderFactory.create(
    "huggingface",
    {"model": "sentence-transformers/all-MiniLM-L6-v2"}
)

# Generate embeddings
result = await embedder.embed_texts(["text1", "text2"])
```

#### EmbeddingProvider Base Class

```python
async def embed_texts(self, texts: List[str]) -> EmbeddingResult
async def embed_query(self, query: str) -> List[float]
```

**Returns:** `EmbeddingResult` with:
- `embeddings: List[List[float]]`
- `dimensions: int`
- `token_count: Optional[int]`
- `processing_time_ms: float`
- `cost_usd: Optional[float]`

### Evaluation Metrics

#### MetricRegistry

```python
from chunk_flow.evaluation import MetricRegistry

# List metrics
metrics = MetricRegistry.get_metric_names()

# Get by category
categories = MetricRegistry.get_metrics_by_category()

# Create metric
metric = MetricRegistry.create("semantic_coherence", {"config_key": "value"})
```

#### EvaluationPipeline

```python
from chunk_flow.evaluation import EvaluationPipeline

# Create pipeline
pipeline = EvaluationPipeline(
    metrics=["semantic_coherence", "ndcg_at_k"],
    metric_configs={"ndcg_at_k": {"k": 5}},
    max_concurrency=4,
)

# Evaluate chunks
results = await pipeline.evaluate(
    chunks=chunks,
    embeddings=embeddings,
    ground_truth=ground_truth,
)

# Compare strategies
comparison = await pipeline.compare_strategies(
    strategies=[strategy1, strategy2],
    text=document,
)
```

### Analysis & Visualization

#### ResultsDataFrame

```python
from chunk_flow.analysis import ResultsDataFrame

# Create from comparison results
df = ResultsDataFrame.from_comparison_results(comparison)

# Rank strategies
ranked = df.rank_strategies(weights={"ndcg_at_k": 2.0})

# Get best
best = df.get_best(metric="semantic_coherence", n=3)

# Filter
filtered = df.filter_strategies(num_chunks__lt=10)

# Export
df.export("results.csv", format="csv")
df.export("results.parquet", format="parquet")
```

#### StrategyVisualizer

```python
from chunk_flow.analysis import StrategyVisualizer

# Create heatmap
StrategyVisualizer.plot_heatmap(
    data=df.df,
    title="Performance Heatmap",
    output_path="heatmap.png",
)

# Create comparison chart
StrategyVisualizer.plot_strategy_comparison(
    results_df=df.df,
    metric="semantic_coherence",
    output_path="comparison.png",
)

# Create full dashboard
plots = StrategyVisualizer.create_comparison_dashboard(
    results_df=df.df,
    output_dir="dashboard/",
)
```

## Error Handling

All exceptions inherit from `ChunkFlowError`:

- `ConfigurationError` - Invalid configuration
- `ValidationError` - Invalid input data
- `ChunkingError` - Chunking operation failed
- `EmbeddingError` - Embedding generation failed
- `EvaluationError` - Metric computation failed
- `RegistryError` - Registry operation failed

```python
from chunk_flow.core.exceptions import ChunkingError, ValidationError

try:
    result = await chunker.chunk(text)
except ValidationError as e:
    print(f"Invalid input: {e}")
except ChunkingError as e:
    print(f"Chunking failed: {e}")
```

## Configuration

### ChunkFlowSettings

Pydantic settings loaded from environment variables:

```python
from chunk_flow.core.config import ChunkFlowSettings

settings = ChunkFlowSettings()
print(settings.log_level)  # From CHUNK_FLOW_LOG_LEVEL
print(settings.openai_api_key)  # From CHUNK_FLOW_OPENAI_API_KEY
```

### ConfigLoader

Load configuration from YAML files:

```python
from chunk_flow.core.config import ConfigLoader

config = ConfigLoader.load("config/default.yaml")
strategy_config = config.get("strategies", {}).get("recursive", {})
```

## Async Utilities

### gather_with_concurrency

```python
from chunk_flow.utils.async_helpers import gather_with_concurrency

results = await gather_with_concurrency(
    max_concurrency=5,
    task1,
    task2,
    task3,
    ...
)
```

### retry_async

```python
from chunk_flow.utils.async_helpers import retry_async

@retry_async(max_attempts=3, backoff_factor=2.0)
async def unstable_operation():
    # May fail, will retry
    pass
```

### AsyncBatchProcessor

```python
from chunk_flow.utils.async_helpers import AsyncBatchProcessor

processor = AsyncBatchProcessor(batch_size=100, max_concurrency=5)
results = await processor.process(items, process_function)
```

## Logging

ChunkFlow uses structured logging with `structlog`:

```python
from chunk_flow.utils.logging import get_logger

logger = get_logger(__name__)

logger.info("operation_started", doc_id="123", num_chunks=10)
logger.error("operation_failed", error=str(e), exc_info=True)
```

Configure via environment:
- `CHUNK_FLOW_LOG_LEVEL`: DEBUG, INFO, WARNING, ERROR
- `CHUNK_FLOW_LOG_FORMAT`: json, pretty

## Type Hints

ChunkFlow is fully typed. Import types for type checking:

```python
from chunk_flow.core.models import ChunkResult, EmbeddingResult, MetricResult
from chunk_flow.core.base import ChunkingStrategy, EmbeddingProvider, EvaluationMetric
from typing import List, Dict, Optional, Any
```
