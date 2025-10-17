# Text Chunking Framework: Complete Specification & Implementation Roadmap

## Executive Summary

**Framework Name:** `chunk-flow` (Async Text Chunking Framework)

**Philosophy:** "Keep place for air condition" - every component designed for v2, v3, v∞ with backward compatibility, versioned interfaces, and pluggable architecture.

**Core Output:** Unified DataFrame with techniques × embeddings × metrics = comprehensive evaluation matrix for data-driven decision making.

---

## 1. Framework Specification

### 1.1 Architecture Overview

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
│ • Strategy v1   │                    │ • Provider v1        │
│ • Strategy v2   │                    │ • Provider v2        │
│ • Model-based   │                    │ • Custom Models      │
│ • Plugin System │                    │ • Fine-tuned         │
└──────────────────┘                    └──────────────────────┘
        ↓                                           ↓
┌─────────────────────────────────────────────────────────────┐
│                    Evaluation Module                         │
│  Metrics v1 | Metrics v2 | Custom | A/B Testing | Benchmark│
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                  Results Storage & Export                    │
│  DataFrame Builder | Parquet | CSV | JSON | DB | Dashboard  │
└─────────────────────────────────────────────────────────────┘
```

---

### 1.2 Core Components

#### 1.2.1 Chunking Strategy Interface (Versioned)

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import asyncio

@dataclass
class ChunkMetadata:
    """Metadata for each chunk"""
    chunk_id: str
    start_idx: int
    end_idx: int
    token_count: int
    char_count: int
    semantic_score: Optional[float] = None
    version: str = "1.0.0"
    strategy_name: str = ""
    custom_fields: Dict[str, Any] = None

@dataclass
class ChunkResult:
    """Result from chunking operation"""
    chunks: List[str]
    metadata: List[ChunkMetadata]
    processing_time_ms: float
    strategy_version: str
    config: Dict[str, Any]

class ChunkingStrategy(ABC):
    """Base interface for all chunking strategies"""
    
    VERSION = "1.0.0"  # Strategy version
    NAME = "base"
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._validate_config()
    
    @abstractmethod
    async def chunk(self, text: str, doc_id: Optional[str] = None) -> ChunkResult:
        """Chunk text into segments"""
        pass
    
    @abstractmethod
    def _validate_config(self) -> None:
        """Validate configuration parameters"""
        pass
    
    @abstractmethod
    def get_default_config(self) -> Dict[str, Any]:
        """Return default configuration"""
        pass
    
    def get_info(self) -> Dict[str, Any]:
        """Strategy metadata"""
        return {
            "name": self.NAME,
            "version": self.VERSION,
            "config": self.config,
            "description": self.__doc__
        }

# Example: Leave room for v2, v3
class ChunkingStrategyV2(ChunkingStrategy):
    """Future: Support streaming, partial results, adaptive chunking"""
    VERSION = "2.0.0"
    
    async def chunk_stream(self, text_stream: AsyncIterator[str]) -> AsyncIterator[ChunkResult]:
        """V2: Streaming support"""
        pass
    
    async def adaptive_chunk(self, text: str, context: Dict[str, Any]) -> ChunkResult:
        """V2: Context-aware adaptive chunking"""
        pass
```

#### 1.2.2 Model-Based Chunking Interface

```python
from typing import Protocol
import torch

class ChunkingModel(Protocol):
    """Protocol for ML-based chunking models"""
    
    def predict_boundaries(self, text: str) -> List[int]:
        """Predict chunk boundary positions"""
        ...
    
    def predict_probabilities(self, text: str) -> List[float]:
        """Predict boundary probabilities at each position"""
        ...

class TrainableChunkingStrategy(ChunkingStrategy):
    """Base for ML-based chunking strategies"""
    
    VERSION = "1.0.0-ml"
    
    def __init__(self, model_path: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.model = self._load_model(model_path)
    
    @abstractmethod
    def _load_model(self, path: Optional[str]) -> ChunkingModel:
        """Load pre-trained model"""
        pass
    
    @abstractmethod
    async def train(self, training_data: List[Dict], **kwargs):
        """Fine-tune model on custom data"""
        pass
    
    async def chunk(self, text: str, doc_id: Optional[str] = None) -> ChunkResult:
        """Use model for chunking"""
        boundaries = self.model.predict_boundaries(text)
        chunks = self._boundaries_to_chunks(text, boundaries)
        # ... create ChunkResult
        return result

# Placeholder for future deep learning chunking models
class BERTChunkingStrategy(TrainableChunkingStrategy):
    """BERT-based boundary detection (v1.0-ml)"""
    NAME = "bert-chunking"
    VERSION = "1.0.0-ml"
    
    def _load_model(self, path: Optional[str]) -> ChunkingModel:
        # Load BERT-based chunking model
        # Future: Pre-trained models on HuggingFace
        pass

class TransformerChunkingStrategy(TrainableChunkingStrategy):
    """Transformer encoder-decoder for chunking (v2.0-ml)"""
    NAME = "transformer-chunking"
    VERSION = "2.0.0-ml"
    # Future implementation space
```

#### 1.2.3 Embedding Provider Interface (Versioned)

```python
@dataclass
class EmbeddingResult:
    embeddings: List[List[float]]
    model_name: str
    dimensions: int
    processing_time_ms: float
    token_count: int
    cost_usd: Optional[float] = None
    provider_version: str = "1.0.0"

class EmbeddingProvider(ABC):
    """Base interface for embedding providers"""
    
    VERSION = "1.0.0"
    PROVIDER_NAME = "base"
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._initialize()
    
    @abstractmethod
    async def embed_texts(self, texts: List[str]) -> EmbeddingResult:
        """Embed multiple texts"""
        pass
    
    @abstractmethod
    async def embed_query(self, query: str) -> EmbeddingResult:
        """Embed single query (may use different model/params)"""
        pass
    
    @abstractmethod
    def _initialize(self) -> None:
        """Initialize provider (API keys, model loading, etc.)"""
        pass
    
    def get_info(self) -> Dict[str, Any]:
        return {
            "provider": self.PROVIDER_NAME,
            "version": self.VERSION,
            "config": self.config
        }

# V2: Support for late chunking, multimodal, fine-tuning
class EmbeddingProviderV2(EmbeddingProvider):
    """Future enhancements"""
    VERSION = "2.0.0"
    
    async def embed_with_late_chunking(
        self, 
        full_text: str, 
        chunk_boundaries: List[int]
    ) -> EmbeddingResult:
        """V2: Late chunking support"""
        pass
    
    async def embed_multimodal(
        self, 
        texts: List[str], 
        images: Optional[List] = None
    ) -> EmbeddingResult:
        """V2: Multimodal embeddings"""
        pass
```

#### 1.2.4 Evaluation Metrics Interface (Versioned)

```python
@dataclass
class MetricResult:
    metric_name: str
    score: float
    version: str
    details: Optional[Dict[str, Any]] = None
    confidence: Optional[float] = None

class EvaluationMetric(ABC):
    """Base interface for evaluation metrics"""
    
    VERSION = "1.0.0"
    METRIC_NAME = "base"
    REQUIRES_GROUND_TRUTH = False
    
    @abstractmethod
    async def compute(
        self,
        chunks: List[str],
        embeddings: Optional[List[List[float]]] = None,
        ground_truth: Optional[Any] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> MetricResult:
        """Compute metric score"""
        pass
    
    def get_info(self) -> Dict[str, Any]:
        return {
            "name": self.METRIC_NAME,
            "version": self.VERSION,
            "requires_ground_truth": self.REQUIRES_GROUND_TRUTH
        }

# Leave room for new metrics
class EvaluationMetricV2(EvaluationMetric):
    """V2: Online metrics, user feedback integration"""
    VERSION = "2.0.0"
    
    async def compute_online(
        self, 
        chunks: List[str], 
        user_feedback: Dict[str, Any]
    ) -> MetricResult:
        """V2: Incorporate real-time user feedback"""
        pass
```

#### 1.2.5 Results DataFrame Schema

```python
from dataclasses import dataclass
from typing import List, Dict, Optional
import pandas as pd

@dataclass
class ExperimentRun:
    """Single experiment run result"""
    run_id: str
    timestamp: str
    
    # Strategy info
    strategy_name: str
    strategy_version: str
    strategy_config: Dict[str, Any]
    
    # Embedding info
    embedding_provider: str
    embedding_model: str
    embedding_dimensions: int
    embedding_version: str
    
    # Document info
    doc_id: str
    doc_length: int
    num_chunks: int
    avg_chunk_size: float
    
    # Performance metrics
    chunking_time_ms: float
    embedding_time_ms: float
    total_time_ms: float
    
    # Quality metrics (expandable)
    metric_scores: Dict[str, float]  # {metric_name: score}
    
    # Cost metrics
    embedding_cost_usd: Optional[float]
    total_cost_usd: Optional[float]
    
    # Metadata (room for future fields)
    metadata: Dict[str, Any]
    framework_version: str

class ResultsDataFrame:
    """Unified results storage and analysis"""
    
    def __init__(self):
        self.df = pd.DataFrame()
        self._schema_version = "1.0.0"
    
    def add_run(self, run: ExperimentRun) -> None:
        """Add single experiment run"""
        row = self._run_to_dict(run)
        self.df = pd.concat([self.df, pd.DataFrame([row])], ignore_index=True)
    
    def add_runs(self, runs: List[ExperimentRun]) -> None:
        """Add multiple runs"""
        rows = [self._run_to_dict(run) for run in runs]
        self.df = pd.concat([self.df, pd.DataFrame(rows)], ignore_index=True)
    
    def get_comparison_matrix(
        self, 
        strategies: Optional[List[str]] = None,
        embeddings: Optional[List[str]] = None,
        metric: str = "primary_score"
    ) -> pd.DataFrame:
        """
        Generate comparison matrix:
        Rows: Strategies
        Cols: Embeddings
        Values: Metric scores
        """
        df_filtered = self.df
        if strategies:
            df_filtered = df_filtered[df_filtered['strategy_name'].isin(strategies)]
        if embeddings:
            df_filtered = df_filtered[df_filtered['embedding_model'].isin(embeddings)]
        
        pivot = df_filtered.pivot_table(
            index='strategy_name',
            columns='embedding_model',
            values=f'metric_{metric}',
            aggfunc='mean'
        )
        return pivot
    
    def get_best_combination(self, metric: str = "primary_score") -> Dict[str, Any]:
        """Find best strategy × embedding combination"""
        best_idx = self.df[f'metric_{metric}'].idxmax()
        best_run = self.df.iloc[best_idx]
        return {
            "strategy": best_run['strategy_name'],
            "embedding": best_run['embedding_model'],
            "score": best_run[f'metric_{metric}'],
            "config": best_run['strategy_config']
        }
    
    def export(self, format: str = "parquet", path: str = "results.parquet") -> None:
        """Export results"""
        if format == "parquet":
            self.df.to_parquet(path)
        elif format == "csv":
            self.df.to_csv(path, index=False)
        elif format == "json":
            self.df.to_json(path, orient='records')
    
    def _run_to_dict(self, run: ExperimentRun) -> Dict:
        """Convert run to flat dictionary for DataFrame"""
        base_dict = {
            "run_id": run.run_id,
            "timestamp": run.timestamp,
            "strategy_name": run.strategy_name,
            "strategy_version": run.strategy_version,
            "embedding_provider": run.embedding_provider,
            "embedding_model": run.embedding_model,
            "embedding_dimensions": run.embedding_dimensions,
            "doc_id": run.doc_id,
            "doc_length": run.doc_length,
            "num_chunks": run.num_chunks,
            "avg_chunk_size": run.avg_chunk_size,
            "chunking_time_ms": run.chunking_time_ms,
            "embedding_time_ms": run.embedding_time_ms,
            "total_time_ms": run.total_time_ms,
            "embedding_cost_usd": run.embedding_cost_usd,
            "total_cost_usd": run.total_cost_usd,
            "framework_version": run.framework_version,
        }
        
        # Flatten metric scores
        for metric_name, score in run.metric_scores.items():
            base_dict[f"metric_{metric_name}"] = score
        
        # Flatten metadata (with prefix to avoid collisions)
        for key, value in run.metadata.items():
            base_dict[f"meta_{key}"] = value
        
        return base_dict
```

---

## 2. Ten-Step Implementation Roadmap

### Phase 1: Foundation (Steps 1-3)

#### **Step 1: Core Infrastructure Setup** ⏱️ Week 1-2

**Deliverables:**
- Project structure with modular architecture
- Base abstract classes (ChunkingStrategy, EmbeddingProvider, EvaluationMetric)
- Configuration management system (YAML/TOML based)
- Logging and monitoring infrastructure
- Async utilities (semaphores, queues, error handling)

**Code Organization:**
```
chunk-flow/
├── chunk_flow/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── base.py              # Abstract base classes
│   │   ├── config.py            # Configuration management
│   │   ├── exceptions.py        # Custom exceptions
│   │   └── version.py           # Version tracking
│   ├── chunking/
│   │   ├── __init__.py
│   │   ├── strategies/          # Empty, ready for step 2
│   │   └── registry.py          # Strategy registry
│   ├── embeddings/
│   │   ├── __init__.py
│   │   └── providers/           # Empty, ready for step 3
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── metrics/             # Empty, ready for step 4
│   ├── results/
│   │   ├── __init__.py
│   │   ├── dataframe.py         # ResultsDataFrame class
│   │   └── export.py            # Export utilities
│   └── utils/
│       ├── async_helpers.py
│       └── logging.py
├── tests/
├── config/
│   ├── default.yaml
│   └── strategies.yaml
├── setup.py
├── requirements.txt
└── README.md
```

**Key Implementation:**
```python
# chunk_flow/core/base.py
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import asyncio
from datetime import datetime

class Versioned:
    """Mixin for version tracking"""
    VERSION = "1.0.0"
    
    @classmethod
    def get_version(cls) -> str:
        return cls.VERSION
    
    @classmethod
    def is_compatible(cls, version: str) -> bool:
        """Check version compatibility"""
        major_current = int(cls.VERSION.split('.')[0])
        major_other = int(version.split('.')[0])
        return major_current == major_other

# chunk_flow/core/config.py
import yaml
from typing import Dict, Any

class Config:
    """Centralized configuration management"""
    
    def __init__(self, config_path: str = "config/default.yaml"):
        with open(config_path, 'r') as f:
            self._config = yaml.safe_load(f)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with dot notation"""
        keys = key.split('.')
        value = self._config
        for k in keys:
            value = value.get(k, default)
            if value is None:
                return default
        return value
    
    def get_strategy_config(self, strategy_name: str) -> Dict[str, Any]:
        return self.get(f"strategies.{strategy_name}", {})
```

**Testing:**
- Unit tests for base classes
- Configuration loading tests
- Version compatibility tests

---

#### **Step 2: Rule-Based Chunking Strategies** ⏱️ Week 3-4

**Deliverables:**
- FixedSizeChunker v1.0
- RecursiveCharacterChunker v1.0
- DocumentBasedChunker v1.0 (Markdown, HTML)
- Strategy factory and registry
- Comprehensive tests for each strategy

**Implementation:**
```python
# chunk_flow/chunking/strategies/recursive.py
from chunk_flow.core.base import ChunkingStrategy, ChunkResult, ChunkMetadata
from typing import List, Dict, Any
import time

class RecursiveCharacterChunker(ChunkingStrategy):
    """Recursive chunking with hierarchical separators"""
    
    VERSION = "1.0.0"
    NAME = "recursive"
    
    def get_default_config(self) -> Dict[str, Any]:
        return {
            "chunk_size": 512,
            "chunk_overlap": 100,
            "separators": ["\n\n", "\n", ". ", " ", ""],
            "length_function": "char"  # or "token"
        }
    
    def _validate_config(self) -> None:
        required = ["chunk_size", "separators"]
        for key in required:
            if key not in self.config:
                raise ValueError(f"Missing required config: {key}")
    
    async def chunk(self, text: str, doc_id: Optional[str] = None) -> ChunkResult:
        start_time = time.time()
        
        chunks = []
        metadata = []
        
        # Recursive splitting logic
        chunks_raw = await self._recursive_split(
            text, 
            self.config["separators"]
        )
        
        # Create metadata
        current_pos = 0
        for i, chunk in enumerate(chunks_raw):
            chunks.append(chunk)
            metadata.append(ChunkMetadata(
                chunk_id=f"{doc_id}_{i}" if doc_id else f"chunk_{i}",
                start_idx=current_pos,
                end_idx=current_pos + len(chunk),
                token_count=len(chunk.split()),  # Rough estimate
                char_count=len(chunk),
                version=self.VERSION,
                strategy_name=self.NAME
            ))
            current_pos += len(chunk)
        
        processing_time = (time.time() - start_time) * 1000
        
        return ChunkResult(
            chunks=chunks,
            metadata=metadata,
            processing_time_ms=processing_time,
            strategy_version=self.VERSION,
            config=self.config
        )
    
    async def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        """Recursive splitting logic"""
        # Implementation here
        pass

# chunk_flow/chunking/registry.py
from typing import Dict, Type
from chunk_flow.core.base import ChunkingStrategy

class StrategyRegistry:
    """Registry for all chunking strategies"""
    
    _strategies: Dict[str, Type[ChunkingStrategy]] = {}
    
    @classmethod
    def register(cls, name: str, strategy_class: Type[ChunkingStrategy]):
        """Register new strategy"""
        cls._strategies[name] = strategy_class
    
    @classmethod
    def get(cls, name: str) -> Type[ChunkingStrategy]:
        """Get strategy by name"""
        if name not in cls._strategies:
            raise ValueError(f"Unknown strategy: {name}")
        return cls._strategies[name]
    
    @classmethod
    def list_strategies(cls) -> List[Dict[str, Any]]:
        """List all registered strategies with metadata"""
        return [
            {
                "name": name,
                "version": strategy.VERSION,
                "description": strategy.__doc__
            }
            for name, strategy in cls._strategies.items()
        ]

# Auto-register on import
from chunk_flow.chunking.strategies.recursive import RecursiveCharacterChunker
StrategyRegistry.register("recursive", RecursiveCharacterChunker)
```

**Testing:**
- Test each strategy with various input sizes
- Test edge cases (empty text, very long text)
- Benchmark performance

---

#### **Step 3: Embedding Providers Integration** ⏱️ Week 5-6

**Deliverables:**
- OpenAI provider v1.0
- HuggingFace provider v1.0
- Google Vertex AI provider v1.0
- Cohere provider v1.0 (if budget allows)
- Provider factory and async batch processing
- Cost tracking utilities

**Implementation:**
```python
# chunk_flow/embeddings/providers/openai.py
from chunk_flow.core.base import EmbeddingProvider, EmbeddingResult
import openai
import time
from typing import List

class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI embedding provider"""
    
    VERSION = "1.0.0"
    PROVIDER_NAME = "openai"
    
    def _initialize(self) -> None:
        self.client = openai.AsyncOpenAI(
            api_key=self.config.get("api_key")
        )
        self.model = self.config.get("model", "text-embedding-3-small")
        self.dimensions = self.config.get("dimensions", None)
    
    async def embed_texts(self, texts: List[str]) -> EmbeddingResult:
        start_time = time.time()
        
        response = await self.client.embeddings.create(
            input=texts,
            model=self.model,
            dimensions=self.dimensions
        )
        
        embeddings = [item.embedding for item in response.data]
        
        # Cost calculation
        tokens = response.usage.total_tokens
        cost = self._calculate_cost(tokens)
        
        processing_time = (time.time() - start_time) * 1000
        
        return EmbeddingResult(
            embeddings=embeddings,
            model_name=self.model,
            dimensions=len(embeddings[0]),
            processing_time_ms=processing_time,
            token_count=tokens,
            cost_usd=cost,
            provider_version=self.VERSION
        )
    
    async def embed_query(self, query: str) -> EmbeddingResult:
        return await self.embed_texts([query])
    
    def _calculate_cost(self, tokens: int) -> float:
        """Calculate cost based on token count"""
        # text-embedding-3-small: $0.00002 per 1K tokens
        if "3-small" in self.model:
            return (tokens / 1000) * 0.00002
        elif "3-large" in self.model:
            return (tokens / 1000) * 0.00013
        return 0.0

# chunk_flow/embeddings/factory.py
class EmbeddingFactory:
    """Factory for creating embedding providers"""
    
    _providers = {}
    
    @classmethod
    def register(cls, name: str, provider_class):
        cls._providers[name] = provider_class
    
    @classmethod
    def create(cls, provider_name: str, config: Dict[str, Any]) -> EmbeddingProvider:
        if provider_name not in cls._providers:
            raise ValueError(f"Unknown provider: {provider_name}")
        return cls._providers[provider_name](config)
```

**Testing:**
- Integration tests with actual API calls (mocked for CI/CD)
- Cost calculation accuracy tests
- Batch processing tests
- Rate limiting tests

---

### Phase 2: Advanced Features (Steps 4-6)

#### **Step 4: Semantic & ML-Based Chunking** ⏱️ Week 7-9

**Deliverables:**
- SemanticChunker v1.0 (embedding-based)
- LLMChunker v1.0 (GPT-based propositional)
- **Model-based chunking framework** (trainable)
- Pre-trained model integration points
- Fine-tuning pipeline (placeholder)

**Implementation:**
```python
# chunk_flow/chunking/strategies/semantic.py
class SemanticChunker(ChunkingStrategy):
    """Semantic chunking using embeddings"""
    
    VERSION = "1.0.0"
    NAME = "semantic"
    
    def __init__(self, config: Dict[str, Any], embedding_provider: EmbeddingProvider):
        super().__init__(config)
        self.embedding_provider = embedding_provider
    
    async def chunk(self, text: str, doc_id: Optional[str] = None) -> ChunkResult:
        # Split into sentences
        sentences = self._split_sentences(text)
        
        # Embed each sentence
        embedding_result = await self.embedding_provider.embed_texts(sentences)
        embeddings = embedding_result.embeddings
        
        # Calculate similarities
        distances = self._calculate_distances(embeddings)
        
        # Determine breakpoints
        threshold = self.config.get("threshold_percentile", 80)
        breakpoints = self._find_breakpoints(distances, threshold)
        
        # Create chunks
        chunks = self._group_sentences(sentences, breakpoints)
        
        # ... create metadata and return ChunkResult
        pass

# chunk_flow/chunking/strategies/model_based.py
from chunk_flow.core.base import ChunkingStrategy, TrainableChunkingStrategy
import torch
from transformers import AutoTokenizer, AutoModel

class BERTBoundaryChunker(TrainableChunkingStrategy):
    """BERT-based boundary detection (ML v1.0)"""
    
    VERSION = "1.0.0-ml"
    NAME = "bert-boundary"
    
    def _load_model(self, path: Optional[str]) -> ChunkingModel:
        """Load BERT boundary detection model"""
        if path:
            # Load custom trained model
            self.model = torch.load(path)
        else:
            # Load pre-trained from HuggingFace (placeholder)
            self.tokenizer = AutoTokenizer.from_pretrained(
                "bert-base-uncased"
            )
            # TODO: Load actual boundary detection head
            # For v1.0: Use BERT + simple classification layer
        return self.model
    
    async def train(self, training_data: List[Dict], **kwargs):
        """
        Fine-tune model on custom data
        
        training_data format:
        [
            {
                "text": "full document",
                "boundaries": [100, 250, 400]  # character positions
            }
        ]
        """
        # TODO: Implement fine-tuning pipeline
        # - Create boundary classification dataset
        # - Train BERT with boundary detection head
        # - Save fine-tuned model
        pass
    
    async def chunk(self, text: str, doc_id: Optional[str] = None) -> ChunkResult:
        # Tokenize
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True)
        
        # Get model predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            boundary_probs = torch.sigmoid(outputs.logits)
        
        # Extract boundaries above threshold
        boundaries = self._extract_boundaries(boundary_probs, threshold=0.5)
        
        # Create chunks from boundaries
        chunks = self._boundaries_to_chunks(text, boundaries)
        
        # ... create metadata and return ChunkResult
        pass

# Placeholder for future models
class TransformerSeq2SeqChunker(TrainableChunkingStrategy):
    """
    V2.0-ml: Sequence-to-sequence transformer for optimal chunking
    Input: Document
    Output: Chunk boundaries with confidence scores
    """
    VERSION = "2.0.0-ml"
    NAME = "transformer-seq2seq"
    # Implementation space for future deep learning models
```

**Model Integration Points:**
```python
# chunk_flow/chunking/models/
# ├── __init__.py
# ├── base.py                    # Base model interfaces
# ├── boundary_detector.py       # Boundary classification models
# ├── seq2seq_chunker.py        # Seq2seq chunking models
# └── pretrained/               # Pre-trained model registry
#     ├── bert_chunker_v1.pt
#     └── config.json
```

**Testing:**
- Test semantic chunking with various threshold values
- Benchmark model inference speed
- Test fine-tuning pipeline (when implemented)

---

#### **Step 5: Comprehensive Evaluation System** ⏱️ Week 10-11

**Deliverables:**
- 10+ evaluation metrics implemented
- RAGAS integration (faithfulness, answer relevancy)
- Custom RAG-specific metrics
- Batch evaluation pipeline
- Statistical comparison tools

**Implementation:**
```python
# chunk_flow/evaluation/metrics/base.py
from chunk_flow.core.base import EvaluationMetric, MetricResult

# chunk_flow/evaluation/metrics/retrieval.py
class NDCGMetric(EvaluationMetric):
    """NDCG@k for retrieval quality"""
    
    VERSION = "1.0.0"
    METRIC_NAME = "ndcg_at_k"
    REQUIRES_GROUND_TRUTH = True
    
    async def compute(
        self,
        chunks: List[str],
        embeddings: Optional[List[List[float]]] = None,
        ground_truth: Optional[Dict] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> MetricResult:
        # Extract query and relevant docs from ground_truth
        query_embedding = ground_truth["query_embedding"]
        relevant_doc_ids = ground_truth["relevant_ids"]
        
        # Calculate similarities
        similarities = self._compute_similarities(query_embedding, embeddings)
        
        # Compute NDCG
        k = self.config.get("k", 5)
        ndcg_score = self._calculate_ndcg(similarities, relevant_doc_ids, k)
        
        return MetricResult(
            metric_name=self.METRIC_NAME,
            score=ndcg_score,
            version=self.VERSION,
            details={"k": k}
        )

# chunk_flow/evaluation/metrics/coherence.py
class SemanticCoherenceMetric(EvaluationMetric):
    """Measures semantic coherence within chunks"""
    
    VERSION = "1.0.0"
    METRIC_NAME = "semantic_coherence"
    REQUIRES_GROUND_TRUTH = False
    
    async def compute(self, chunks: List[str], embeddings: List[List[float]], **kwargs) -> MetricResult:
        # Calculate average intra-chunk similarity
        coherence_scores = []
        for chunk_embedding in embeddings:
            # Split chunk into sentences, embed, calculate internal similarity
            # Higher = more coherent
            pass
        
        avg_coherence = np.mean(coherence_scores)
        return MetricResult(
            metric_name=self.METRIC_NAME,
            score=avg_coherence,
            version=self.VERSION
        )

# chunk_flow/evaluation/pipeline.py
class EvaluationPipeline:
    """Orchestrates evaluation of chunking strategies"""
    
    def __init__(self):
        self.metrics: List[EvaluationMetric] = []
    
    def add_metric(self, metric: EvaluationMetric) -> 'EvaluationPipeline':
        """Fluent interface for adding metrics"""
        self.metrics.append(metric)
        return self
    
    async def evaluate_single(
        self,
        chunk_result: ChunkResult,
        embedding_result: EmbeddingResult,
        ground_truth: Optional[Any] = None
    ) -> Dict[str, MetricResult]:
        """Evaluate single chunking result"""
        results = {}
        for metric in self.metrics:
            result = await metric.compute(
                chunks=chunk_result.chunks,
                embeddings=embedding_result.embeddings,
                ground_truth=ground_truth
            )
            results[metric.METRIC_NAME] = result
        return results
    
    async def evaluate_comparison(
        self,
        strategies: List[ChunkingStrategy],
        embedding_providers: List[EmbeddingProvider],
        test_documents: List[str],
        ground_truth: Optional[List[Any]] = None
    ) -> ResultsDataFrame:
        """
        Full comparison matrix:
        - For each strategy × embedding combination
        - For each test document
        - Compute all metrics
        - Return comprehensive DataFrame
        """
        results_df = ResultsDataFrame()
        
        for strategy in strategies:
            for embedding_provider in embedding_providers:
                for doc_id, doc in enumerate(test_documents):
                    # Chunk
                    chunk_result = await strategy.chunk(doc, doc_id=str(doc_id))
                    
                    # Embed
                    embedding_result = await embedding_provider.embed_texts(
                        chunk_result.chunks
                    )
                    
                    # Evaluate
                    gt = ground_truth[doc_id] if ground_truth else None
                    metric_results = await self.evaluate_single(
                        chunk_result, embedding_result, gt
                    )
                    
                    # Create ExperimentRun
                    run = ExperimentRun(
                        run_id=f"{strategy.NAME}_{embedding_provider.PROVIDER_NAME}_{doc_id}",
                        timestamp=datetime.now().isoformat(),
                        strategy_name=strategy.NAME,
                        strategy_version=strategy.VERSION,
                        strategy_config=strategy.config,
                        embedding_provider=embedding_provider.PROVIDER_NAME,
                        embedding_model=embedding_result.model_name,
                        embedding_dimensions=embedding_result.dimensions,
                        embedding_version=embedding_provider.VERSION,
                        doc_id=str(doc_id),
                        doc_length=len(doc),
                        num_chunks=len(chunk_result.chunks),
                        avg_chunk_size=np.mean([len(c) for c in chunk_result.chunks]),
                        chunking_time_ms=chunk_result.processing_time_ms,
                        embedding_time_ms=embedding_result.processing_time_ms,
                        total_time_ms=chunk_result.processing_time_ms + embedding_result.processing_time_ms,
                        metric_scores={name: result.score for name, result in metric_results.items()},
                        embedding_cost_usd=embedding_result.cost_usd,
                        total_cost_usd=embedding_result.cost_usd,
                        metadata={},
                        framework_version="1.0.0"
                    )
                    
                    results_df.add_run(run)
        
        return results_df
```

**Testing:**
- Validate metric calculations with known ground truth
- Test evaluation pipeline with multiple strategies
- Benchmark evaluation performance

---

#### **Step 6: Late Chunking Implementation** ⏱️ Week 12-13

**Deliverables:**
- LateChunker v1.0
- Integration with long-context embedding models (Jina AI, custom)
- Benchmark comparison: traditional vs late chunking
- Documentation on when to use late chunking

**Implementation:**
```python
# chunk_flow/chunking/strategies/late.py
class LateChunker(ChunkingStrategy):
    """
    Late chunking: Embed full document first, then chunk at token level
    Requires long-context embedding model (8K+ tokens)
    """
    
    VERSION = "1.0.0"
    NAME = "late"
    
    def __init__(self, config: Dict[str, Any], embedding_provider: EmbeddingProvider):
        super().__init__(config)
        self.embedding_provider = embedding_provider
        
        # Validate embedding provider supports long context
        if not hasattr(embedding_provider, 'max_context_length'):
            raise ValueError("Late chunking requires embedding provider with max_context_length")
    
    async def chunk(self, text: str, doc_id: Optional[str] = None) -> ChunkResult:
        """
        1. Tokenize full document
        2. Get token-level embeddings for entire document
        3. Define chunk boundaries
        4. Mean pool tokens within each chunk
        """
        start_time = time.time()
        
        # Tokenize (using embedding provider's tokenizer)
        tokens = await self.embedding_provider.tokenize(text)
        
        # Get token-level embeddings
        token_embeddings = await self.embedding_provider.embed_tokens(tokens)
        
        # Define chunk boundaries (e.g., 256-token chunks)
        chunk_size = self.config.get("chunk_size", 256)
        boundaries = list(range(0, len(tokens), chunk_size))
        
        # Mean pool within chunks
        chunk_embeddings = []
        chunks_text = []
        metadata = []
        
        for i in range(len(boundaries) - 1):
            start_idx = boundaries[i]
            end_idx = boundaries[i + 1]
            
            # Mean pool token embeddings
            chunk_emb = np.mean(token_embeddings[start_idx:end_idx], axis=0)
            chunk_embeddings.append(chunk_emb.tolist())
            
            # Reconstruct text from tokens
            chunk_text = self.embedding_provider.decode_tokens(
                tokens[start_idx:end_idx]
            )
            chunks_text.append(chunk_text)
            
            # Metadata
            metadata.append(ChunkMetadata(
                chunk_id=f"{doc_id}_late_{i}",
                start_idx=start_idx,
                end_idx=end_idx,
                token_count=end_idx - start_idx,
                char_count=len(chunk_text),
                version=self.VERSION,
                strategy_name=self.NAME
            ))
        
        processing_time = (time.time() - start_time) * 1000
        
        return ChunkResult(
            chunks=chunks_text,
            metadata=metadata,
            processing_time_ms=processing_time,
            strategy_version=self.VERSION,
            config=self.config
        )
```

**Testing:**
- Compare late vs traditional chunking on long documents
- Validate embedding quality preservation
- Benchmark performance impact

---

### Phase 3: Production & Advanced Features (Steps 7-10)

#### **Step 7: API Layer & Orchestration** ⏱️ Week 14-15

**Deliverables:**
- FastAPI application with all endpoints
- Request/response validation with Pydantic
- Async request handling
- API documentation (Swagger/OpenAPI)
- Docker containerization

**Implementation:**
```python
# chunk_flow/api/app.py
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from typing import List, Dict, Optional
import asyncio

app = FastAPI(title="ChunkFlow API", version="1.0.0")

class ChunkRequest(BaseModel):
    text: str
    strategy: str = "recursive"
    strategy_config: Optional[Dict] = None
    doc_id: Optional[str] = None

class ChunkResponse(BaseModel):
    chunks: List[str]
    metadata: List[Dict]
    processing_time_ms: float
    strategy_version: str

class EvaluateRequest(BaseModel):
    documents: List[str]
    strategies: List[str]
    embeddings: List[str]
    metrics: List[str]
    ground_truth: Optional[List[Dict]] = None

class ComparisonMatrixResponse(BaseModel):
    matrix: Dict[str, Dict[str, float]]
    best_combination: Dict[str, Any]
    full_results_url: str

@app.post("/chunk", response_model=ChunkResponse)
async def chunk_text(request: ChunkRequest):
    """Chunk single document"""
    strategy = StrategyRegistry.get(request.strategy)(
        config=request.strategy_config
    )
    result = await strategy.chunk(request.text, request.doc_id)
    
    return ChunkResponse(
        chunks=result.chunks,
        metadata=[meta.__dict__ for meta in result.metadata],
        processing_time_ms=result.processing_time_ms,
        strategy_version=result.strategy_version
    )

@app.post("/evaluate", response_model=ComparisonMatrixResponse)
async def evaluate_strategies(request: EvaluateRequest, background_tasks: BackgroundTasks):
    """
    Full evaluation: strategies × embeddings × documents
    Returns comparison matrix and saves detailed results
    """
    # Create pipeline
    pipeline = EvaluationPipeline()
    for metric_name in request.metrics:
        metric = MetricRegistry.get(metric_name)()
        pipeline.add_metric(metric)
    
    # Load strategies and embeddings
    strategies = [StrategyRegistry.get(s)() for s in request.strategies]
    embeddings = [EmbeddingFactory.create(e, {}) for e in request.embeddings]
    
    # Run evaluation
    results_df = await pipeline.evaluate_comparison(
        strategies=strategies,
        embedding_providers=embeddings,
        test_documents=request.documents,
        ground_truth=request.ground_truth
    )
    
    # Generate comparison matrix (primary metric: first in list)
    primary_metric = request.metrics[0]
    matrix = results_df.get_comparison_matrix(metric=primary_metric)
    
    # Get best combination
    best = results_df.get_best_combination(metric=primary_metric)
    
    # Save full results
    results_id = generate_id()
    results_path = f"results/{results_id}.parquet"
    background_tasks.add_task(results_df.export, "parquet", results_path)
    
    return ComparisonMatrixResponse(
        matrix=matrix.to_dict(),
        best_combination=best,
        full_results_url=f"/results/{results_id}"
    )

@app.get("/strategies")
async def list_strategies():
    """List all available strategies"""
    return StrategyRegistry.list_strategies()

@app.get("/embeddings")
async def list_embeddings():
    """List all available embedding providers"""
    return EmbeddingFactory.list_providers()

@app.get("/metrics")
async def list_metrics():
    """List all available metrics"""
    return MetricRegistry.list_metrics()
```

**Testing:**
- API endpoint tests
- Load testing with concurrent requests
- Integration tests with full pipeline

---

#### **Step 8: Results Analysis & Visualization** ⏱️ Week 16

**Deliverables:**
- Interactive dashboard (Streamlit/Dash)
- Automated report generation
- Statistical significance testing
- Export to multiple formats (Parquet, CSV, JSON, Excel)
- Visualization utilities (heatmaps, comparison charts)

**Implementation:**
```python
# chunk_flow/results/visualization.py
import plotly.express as px
import plotly.graph_objects as go

class ResultsVisualizer:
    """Generate visualizations from ResultsDataFrame"""
    
    def __init__(self, results_df: ResultsDataFrame):
        self.df = results_df.df
    
    def plot_comparison_heatmap(self, metric: str = "primary_score") -> go.Figure:
        """
        Heatmap: Strategies (rows) × Embeddings (cols)
        """
        matrix = self.df.pivot_table(
            index='strategy_name',
            columns='embedding_model',
            values=f'metric_{metric}',
            aggfunc='mean'
        )
        
        fig = px.imshow(
            matrix,
            labels=dict(x="Embedding Model", y="Strategy", color="Score"),
            title=f"Strategy × Embedding Performance ({metric})",
            color_continuous_scale="RdYlGn"
        )
        return fig
    
    def plot_cost_vs_performance(self, metric: str = "primary_score") -> go.Figure:
        """Scatter plot: Cost vs Performance"""
        fig = px.scatter(
            self.df,
            x='total_cost_usd',
            y=f'metric_{metric}',
            color='strategy_name',
            size='num_chunks',
            hover_data=['embedding_model', 'total_time_ms'],
            title="Cost vs Performance Trade-off"
        )
        return fig
    
    def plot_latency_breakdown(self) -> go.Figure:
        """Stacked bar: Chunking vs Embedding time"""
        fig = go.Figure()
        
        for strategy in self.df['strategy_name'].unique():
            df_strategy = self.df[self.df['strategy_name'] == strategy]
            
            fig.add_trace(go.Bar(
                name=f"{strategy} - Chunking",
                x=df_strategy['embedding_model'],
                y=df_strategy['chunking_time_ms']
            ))
            fig.add_trace(go.Bar(
                name=f"{strategy} - Embedding",
                x=df_strategy['embedding_model'],
                y=df_strategy['embedding_time_ms']
            ))
        
        fig.update_layout(
            barmode='stack',
            title="Latency Breakdown by Strategy",
            xaxis_title="Embedding Model",
            yaxis_title="Time (ms)"
        )
        return fig

# chunk_flow/api/dashboard.py (Streamlit app)
import streamlit as st

st.title("ChunkFlow Results Dashboard")

# Load results
results_file = st.file_uploader("Upload results (Parquet)", type=['parquet'])
if results_file:
    results_df = ResultsDataFrame()
    results_df.df = pd.read_parquet(results_file)
    
    viz = ResultsVisualizer(results_df)
    
    # Metric selection
    metric = st.selectbox("Select Metric", 
                          [col for col in results_df.df.columns if col.startswith('metric_')])
    
    # Heatmap
    st.plotly_chart(viz.plot_comparison_heatmap(metric))
    
    # Cost vs Performance
    st.plotly_chart(viz.plot_cost_vs_performance(metric))
    
    # Best combination
    st.subheader("Best Configuration")
    best = results_df.get_best_combination(metric)
    st.json(best)
```

**Testing:**
- Visualization rendering tests
- Dashboard interactivity tests

---

#### **Step 9: Benchmark Suite & Testing** ⏱️ Week 17

**Deliverables:**
- Standard benchmark datasets (WikiQA, FinanceBench subset, custom)
- Automated benchmark runner
- Performance regression tests
- Continuous benchmarking in CI/CD
- Leaderboard generation

**Implementation:**
```python
# chunk_flow/benchmark/datasets.py
from typing import List, Dict
import json

class BenchmarkDataset:
    """Base class for benchmark datasets"""
    
    def __init__(self, name: str, path: str):
        self.name = name
        self.path = path
        self.documents = []
        self.ground_truth = []
        self._load()
    
    def _load(self):
        """Load dataset from disk"""
        with open(self.path, 'r') as f:
            data = json.load(f)
            self.documents = data['documents']
            self.ground_truth = data.get('ground_truth', [])
    
    def get_documents(self) -> List[str]:
        return self.documents
    
    def get_ground_truth(self) -> List[Dict]:
        return self.ground_truth

# chunk_flow/benchmark/runner.py
class BenchmarkRunner:
    """Automated benchmark execution"""
    
    def __init__(self):
        self.datasets = {}
        self.strategies = []
        self.embeddings = []
        self.metrics = []
    
    def add_dataset(self, dataset: BenchmarkDataset):
        self.datasets[dataset.name] = dataset
        return self
    
    def add_strategy(self, strategy: ChunkingStrategy):
        self.strategies.append(strategy)
        return self
    
    def add_embedding(self, embedding: EmbeddingProvider):
        self.embeddings.append(embedding)
        return self
    
    def add_metric(self, metric: EvaluationMetric):
        self.metrics.append(metric)
        return self
    
    async def run_all(self) -> Dict[str, ResultsDataFrame]:
        """Run benchmarks on all datasets"""
        results = {}
        
        for dataset_name, dataset in self.datasets.items():
            pipeline = EvaluationPipeline()
            for metric in self.metrics:
                pipeline.add_metric(metric)
            
            results_df = await pipeline.evaluate_comparison(
                strategies=self.strategies,
                embedding_providers=self.embeddings,
                test_documents=dataset.get_documents(),
                ground_truth=dataset.get_ground_truth()
            )
            
            results[dataset_name] = results_df
        
        return results
    
    def generate_leaderboard(self, results: Dict[str, ResultsDataFrame]) -> pd.DataFrame:
        """Generate leaderboard across all benchmarks"""
        leaderboard_data = []
        
        for dataset_name, results_df in results.items():
            # Aggregate by strategy × embedding
            for strategy in self.strategies:
                for embedding in self.embeddings:
                    # Filter results
                    mask = (
                        (results_df.df['strategy_name'] == strategy.NAME) &
                        (results_df.df['embedding_provider'] == embedding.PROVIDER_NAME)
                    )
                    subset = results_df.df[mask]
                    
                    # Compute averages
                    row = {
                        'dataset': dataset_name,
                        'strategy': strategy.NAME,
                        'embedding': embedding.PROVIDER_NAME,
                    }
                    
                    # Add metric averages
                    for col in subset.columns:
                        if col.startswith('metric_'):
                            row[col] = subset[col].mean()
                    
                    leaderboard_data.append(row)
        
        return pd.DataFrame(leaderboard_data)

# tests/benchmark_test.py
@pytest.mark.benchmark
async def test_performance_regression():
    """Ensure performance doesn't regress"""
    runner = BenchmarkRunner()
    runner.add_dataset(BenchmarkDataset("wikiqa", "data/wikiqa_subset.json"))
    runner.add_strategy(RecursiveCharacterChunker())
    runner.add_embedding(OpenAIEmbeddingProvider({"model": "text-embedding-3-small"}))
    runner.add_metric(NDCGMetric())
    
    results = await runner.run_all()
    
    # Load baseline
    with open("benchmarks/baseline.json", 'r') as f:
        baseline = json.load(f)
    
    # Compare
    current_score = results['wikiqa'].df['metric_ndcg_at_k'].mean()
    baseline_score = baseline['wikiqa']['ndcg_at_k']
    
    # Assert no regression (within 2% tolerance)
    assert current_score >= baseline_score * 0.98, "Performance regression detected!"
```

**Testing:**
- Run full benchmark suite
- Validate leaderboard generation
- Performance regression tests in CI/CD

---

#### **Step 10: Documentation, Packaging & Release** ⏱️ Week 18

**Deliverables:**
- Comprehensive documentation (ReadTheDocs/MkDocs)
- API reference (auto-generated from docstrings)
- Tutorial notebooks (Jupyter)
- PyPI package release
- Docker images (Docker Hub / GHCR)
- GitHub Actions CI/CD pipeline
- Example projects

**Structure:**
```
docs/
├── index.md                  # Overview
├── getting-started.md        # Quick start guide
├── concepts/
│   ├── chunking-strategies.md
│   ├── embeddings.md
│   ├── evaluation.md
│   └── dataframe-output.md
├── guides/
│   ├── choosing-strategy.md
│   ├── custom-strategies.md
│   ├── model-based-chunking.md
│   └── production-deployment.md
├── api-reference/
│   ├── chunking.md
│   ├── embeddings.md
│   ├── evaluation.md
│   └── api-endpoints.md
├── tutorials/
│   ├── basic-usage.ipynb
│   ├── strategy-comparison.ipynb
│   ├── custom-metric.ipynb
│   └── benchmark-your-data.ipynb
└── changelog.md

examples/
├── quickstart.py
├── full_evaluation.py
├── custom_strategy.py
├── model_based_chunking.py
└── dashboard_app.py

.github/
└── workflows/
    ├── test.yml             # Run tests on PR
    ├── benchmark.yml        # Weekly benchmarks
    ├── publish-pypi.yml     # Publish on release
    └── docker-build.yml     # Build Docker images
```

**PyPI Package:**
```python
# setup.py
from setuptools import setup, find_packages

setup(
    name="chunk-flow",
    version="1.0.0",
    description="Async text chunking framework for RAG systems",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="you@example.com",
    url="https://github.com/yourusername/chunk-flow",
    packages=find_packages(),
    install_requires=[
        "fastapi>=0.104.0",
        "pydantic>=2.0.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "openai>=1.0.0",
        "transformers>=4.30.0",
        "torch>=2.0.0",
        "scikit-learn>=1.3.0",
        "plotly>=5.14.0",
        "pyarrow>=12.0.0",  # For Parquet
        "httpx>=0.24.0",     # Async HTTP
        "pyyaml>=6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "mypy>=1.4.0",
            "ruff>=0.0.280",
        ],
        "viz": [
            "streamlit>=1.25.0",
            "plotly>=5.14.0",
        ],
        "all": ["chunk-flow[dev,viz]"],
    },
    python_requires=">=3.9",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
```

**Testing:**
- Documentation build tests
- Package installation tests
- Example code execution tests

---

## 3. "Air Condition" Philosophy: Extensibility Design

### 3.1 Version Compatibility System

```python
# chunk_flow/core/compatibility.py
from typing import List, Tuple
from dataclasses import dataclass

@dataclass
class VersionRange:
    """Semantic version range"""
    min_version: str
    max_version: str
    
    def is_compatible(self, version: str) -> bool:
        """Check if version falls within range"""
        return (
            self._compare_versions(version, self.min_version) >= 0 and
            self._compare_versions(version, self.max_version) <= 0
        )
    
    @staticmethod
    def _compare_versions(v1: str, v2: str) -> int:
        """Compare two semantic versions"""
        parts1 = [int(x) for x in v1.split('.')]
        parts2 = [int(x) for x in v2.split('.')]
        
        for p1, p2 in zip(parts1, parts2):
            if p1 < p2:
                return -1
            elif p1 > p2:
                return 1
        return 0

class CompatibilityChecker:
    """Check compatibility between components"""
    
    @staticmethod
    def check_strategy_embedding(
        strategy: ChunkingStrategy,
        embedding: EmbeddingProvider
    ) -> Tuple[bool, str]:
        """Check if strategy and embedding are compatible"""
        
        # Special requirements (e.g., late chunking needs long-context)
        if strategy.NAME == "late":
            if not hasattr(embedding, 'max_context_length'):
                return False, "Late chunking requires embedding with max_context_length"
            if embedding.max_context_length < 8192:
                return False, f"Late chunking requires 8K+ context, got {embedding.max_context_length}"
        
        # Version compatibility (major version must match)
        if not strategy.is_compatible(embedding.VERSION):
            return False, f"Version mismatch: strategy={strategy.VERSION}, embedding={embedding.VERSION}"
        
        return True, "Compatible"
```

### 3.2 Plugin System for Extensions

```python
# chunk_flow/plugins/loader.py
import importlib
import pkgutil
from typing import Dict, Type

class PluginLoader:
    """Dynamic plugin loading system"""
    
    @staticmethod
    def discover_plugins(package_name: str = "chunk_flow_plugins"):
        """Discover and load all plugins"""
        try:
            package = importlib.import_module(package_name)
        except ImportError:
            return
        
        for _, name, _ in pkgutil.iter_modules(package.__path__):
            module = importlib.import_module(f"{package_name}.{name}")
            
            # Auto-register strategies
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if isinstance(attr, type) and issubclass(attr, ChunkingStrategy):
                    if attr != ChunkingStrategy:
                        StrategyRegistry.register(attr.NAME, attr)
            
            # Auto-register embeddings
            # Auto-register metrics
            # etc.

# Usage: Third-party plugins
# pip install chunk-flow-plugin-custom
# chunk_flow_plugins/
#   custom/
#     __init__.py
#     strategy.py  # Contains CustomChunkingStrategy

# Auto-discovered and registered on import
```

### 3.3 Configuration Versioning

```yaml
# config/strategies.yaml
version: "1.0.0"  # Config schema version

strategies:
  recursive:
    v1.0:
      chunk_size: 512
      chunk_overlap: 100
      separators: ["\n\n", "\n", ". ", " ", ""]
    v2.0:  # Room for future improvements
      chunk_size: 512
      chunk_overlap: 100
      separators: ["\n\n", "\n", ". ", " ", ""]
      adaptive_sizing: true  # New feature in v2
      context_aware: true

  semantic:
    v1.0:
      threshold_percentile: 80
      embedding_model: "all-MiniLM-L6-v2"
    v2.0:
      threshold_percentile: 80
      embedding_model: "all-mpnet-base-v2"
      multi_level: true  # Hierarchical semantic chunking

# Migrations
migrations:
  "1.0.0->2.0.0":
    recursive:
      - add_field: "adaptive_sizing"
        default: false
      - add_field: "context_aware"
        default: false
```

### 3.4 Future-Proof Data Schema

```python
# chunk_flow/results/schema.py
from typing import Any, Dict
from pydantic import BaseModel, Field

class ExperimentRunV1(BaseModel):
    """V1 schema (current)"""
    schema_version: str = "1.0.0"
    # ... all current fields
    
    # Reserved for future
    extensions: Dict[str, Any] = Field(default_factory=dict)
    custom_metrics: Dict[str, float] = Field(default_factory=dict)

class ExperimentRunV2(BaseModel):
    """V2 schema (future) - maintains backward compatibility"""
    schema_version: str = "2.0.0"
    # ... all V1 fields
    
    # New V2 fields
    retrieval_metrics: Dict[str, float] = Field(default_factory=dict)
    generation_metrics: Dict[str, float] = Field(default_factory=dict)
    user_feedback: Optional[Dict[str, Any]] = None
    a_b_test_group: Optional[str] = None

class SchemaUpgrader:
    """Upgrade older schemas to current version"""
    
    @staticmethod
    def upgrade(data: Dict[str, Any]) -> ExperimentRun:
        """Upgrade data to current schema version"""
        version = data.get("schema_version", "1.0.0")
        
        if version == "1.0.0":
            # Already current
            return ExperimentRunV1(**data)
        elif version == "0.9.0":
            # Upgrade from beta
            return SchemaUpgrader._upgrade_0_9_to_1_0(data)
        else:
            raise ValueError(f"Unknown schema version: {version}")
```

---

## 4. Major Output: DataFrame Specification

### 4.1 Complete DataFrame Schema

```python
# Column naming convention: category_name_detail
DATAFRAME_SCHEMA = {
    # Run identification
    "run_id": "string",
    "run_timestamp": "datetime64[ns]",
    "framework_version": "string",
    "schema_version": "string",
    
    # Strategy information
    "strategy_name": "string",
    "strategy_version": "string",
    "strategy_config_chunk_size": "int64",
    "strategy_config_overlap": "int64",
    "strategy_config_*": "variant",  # Other config params
    
    # Embedding information
    "embedding_provider": "string",
    "embedding_model": "string",
    "embedding_version": "string",
    "embedding_dimensions": "int64",
    "embedding_max_context": "int64",
    
    # Document information
    "doc_id": "string",
    "doc_length_chars": "int64",
    "doc_length_tokens": "int64",
    "doc_type": "string",  # e.g., "technical", "narrative", "code"
    
    # Chunking results
    "chunk_num_chunks": "int64",
    "chunk_avg_size_chars": "float64",
    "chunk_avg_size_tokens": "float64",
    "chunk_min_size": "int64",
    "chunk_max_size": "int64",
    "chunk_std_size": "float64",
    
    # Performance metrics
    "perf_chunking_time_ms": "float64",
    "perf_embedding_time_ms": "float64",
    "perf_total_time_ms": "float64",
    "perf_throughput_chars_per_sec": "float64",
    
    # Cost metrics
    "cost_embedding_usd": "float64",
    "cost_total_usd": "float64",
    "cost_per_chunk_usd": "float64",
    
    # Quality metrics - Retrieval
    "metric_ndcg_at_5": "float64",
    "metric_ndcg_at_10": "float64",
    "metric_recall_at_5": "float64",
    "metric_recall_at_10": "float64",
    "metric_mrr": "float64",
    "metric_precision_at_5": "float64",
    
    # Quality metrics - Semantic
    "metric_semantic_coherence": "float64",
    "metric_boundary_clarity": "float64",
    "metric_chunk_stickiness": "float64",
    "metric_inter_chunk_similarity": "float64",
    
    # Quality metrics - RAG-specific
    "metric_faithfulness": "float64",
    "metric_answer_relevancy": "float64",
    "metric_context_precision": "float64",
    "metric_context_recall": "float64",
    "metric_ragas_score": "float64",
    
    # Metadata (extensible)
    "meta_dataset_name": "string",
    "meta_experiment_group": "string",
    "meta_user_id": "string",
    "meta_notes": "string",
    "meta_*": "variant",  # Custom metadata
}
```

### 4.2 DataFrame Analysis Methods

```python
# chunk_flow/results/analysis.py
class ResultsAnalyzer:
    """Advanced analysis of results DataFrame"""
    
    def __init__(self, results_df: ResultsDataFrame):
        self.df = results_df.df
    
    def pareto_frontier(
        self, 
        cost_metric: str = "cost_total_usd",
        quality_metric: str = "metric_ragas_score"
    ) -> pd.DataFrame:
        """
        Identify Pareto-optimal configurations
        (maximize quality, minimize cost)
        """
        pareto_points = []
        
        for idx, row in self.df.iterrows():
            is_pareto = True
            for idx2, row2 in self.df.iterrows():
                if idx == idx2:
                    continue
                # Dominated if worse on both dimensions
                if (row2[quality_metric] >= row[quality_metric] and 
                    row2[cost_metric] <= row[cost_metric]):
                    if (row2[quality_metric] > row[quality_metric] or 
                        row2[cost_metric] < row[cost_metric]):
                        is_pareto = False
                        break
            
            if is_pareto:
                pareto_points.append(row)
        
        return pd.DataFrame(pareto_points)
    
    def strategy_ranking(
        self,
        metrics: List[str],
        weights: Optional[List[float]] = None
    ) -> pd.DataFrame:
        """
        Rank strategies by weighted combination of metrics
        """
        if weights is None:
            weights = [1.0] * len(metrics)
        
        # Normalize metrics to 0-1
        normalized = self.df.copy()
        for metric in metrics:
            min_val = normalized[f"metric_{metric}"].min()
            max_val = normalized[f"metric_{metric}"].max()
            normalized[f"norm_{metric}"] = (
                (normalized[f"metric_{metric}"] - min_val) / (max_val - min_val)
            )
        
        # Compute weighted score
        normalized['weighted_score'] = sum(
            normalized[f"norm_{metric}"] * weight 
            for metric, weight in zip(metrics, weights)
        )
        
        # Rank by strategy
        ranking = normalized.groupby('strategy_name').agg({
            'weighted_score': 'mean',
            **{f"metric_{m}": 'mean' for m in metrics}
        }).sort_values('weighted_score', ascending=False)
        
        return ranking
    
    def statistical_comparison(
        self,
        strategy1: str,
        strategy2: str,
        metric: str,
        alpha: float = 0.05
    ) -> Dict[str, Any]:
        """
        Statistical significance test between two strategies
        """
        from scipy import stats
        
        data1 = self.df[self.df['strategy_name'] == strategy1][f"metric_{metric}"]
        data2 = self.df[self.df['strategy_name'] == strategy2][f"metric_{metric}"]
        
        # Perform t-test
        t_stat, p_value = stats.ttest_ind(data1, data2)
        
        is_significant = p_value < alpha
        effect_size = (data1.mean() - data2.mean()) / np.sqrt(
            (data1.std()**2 + data2.std()**2) / 2
        )
        
        return {
            "strategy1": strategy1,
            "strategy2": strategy2,
            "metric": metric,
            "mean1": data1.mean(),
            "mean2": data2.mean(),
            "diff": data1.mean() - data2.mean(),
            "t_statistic": t_stat,
            "p_value": p_value,
            "is_significant": is_significant,
            "effect_size": effect_size,
            "confidence_level": 1 - alpha
        }
```

---

## 5. Implementation Priorities & Timeline

### Timeline Overview (18 weeks total)

**Phase 1 - Foundation (Weeks 1-6):**
- ✅ Step 1: Core Infrastructure
- ✅ Step 2: Rule-Based Chunking
- ✅ Step 3: Embedding Providers

**Phase 2 - Advanced (Weeks 7-13):**
- ✅ Step 4: Semantic & ML-Based Chunking
- ✅ Step 5: Evaluation System
- ✅ Step 6: Late Chunking

**Phase 3 - Production (Weeks 14-18):**
- ✅ Step 7: API & Orchestration
- ✅ Step 8: Visualization
- ✅ Step 9: Benchmarking
- ✅ Step 10: Documentation & Release

### Critical Path
1. Steps 1-3 must be completed sequentially (foundation)
2. Steps 4-6 can be parallelized with 2-3 developers
3. Steps 7-10 can partially overlap

### Resource Requirements
- **1 Senior Engineer:** Architecture, core infrastructure, ML integration
- **1-2 Mid-Level Engineers:** Strategy implementations, API development
- **1 ML Engineer (Part-time):** Model-based chunking, fine-tuning pipeline
- **1 DevOps Engineer (Part-time):** CI/CD, containerization, deployment

---

## 6. Success Metrics

### Technical Metrics
- **Performance:** Process 10K documents in <5 minutes (all strategies)
- **Accuracy:** Achieve >0.80 RAGAS score on standard benchmarks
- **Cost Efficiency:** 90% cost reduction vs naive approach
- **Reliability:** 99.9% uptime for API, <0.1% error rate

### Adoption Metrics
- **Documentation:** 100% API coverage, 10+ tutorial notebooks
- **Community:** 1K+ GitHub stars, 100+ PyPI downloads/week
- **Extensions:** 5+ community-contributed plugins within 6 months

### Extensibility Metrics
- **Plugin Development Time:** New strategy implemented in <4 hours
- **Version Compatibility:** Zero breaking changes in minor versions
- **Custom Metrics:** New metric added in <2 hours

---

## 7. Risk Mitigation

### Technical Risks
1. **Model-based chunking complexity**
   - Mitigation: Start with simple boundary detection, iterate
   - Fallback: Focus on rule-based strategies for v1.0

2. **Late chunking performance**
   - Mitigation: Thorough benchmarking, optimize token processing
   - Fallback: Offer both traditional and late chunking

3. **API scalability**
   - Mitigation: Load testing, async implementation, caching
   - Fallback: Rate limiting, queue-based processing

### Resource Risks
1. **ML engineer availability**
   - Mitigation: Pre-train models, use transfer learning
   - Fallback: Defer model-based chunking to v1.1

2. **API costs (embeddings)**
   - Mitigation: Use free tiers, caching, HuggingFace models
   - Fallback: Limit benchmark size, use smaller models

---

This comprehensive specification provides a complete roadmap for building a production-grade, extensible text chunking framework with strong "air condition" philosophy baked in from day one.