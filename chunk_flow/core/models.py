"""Pydantic models for data structures."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ChunkMetadata(BaseModel):
    """Metadata for a single chunk."""

    chunk_id: str = Field(..., description="Unique identifier for this chunk")
    start_idx: int = Field(..., ge=0, description="Start index in original document")
    end_idx: int = Field(..., ge=0, description="End index in original document")
    token_count: int = Field(..., ge=0, description="Number of tokens in chunk")
    char_count: int = Field(..., ge=0, description="Number of characters in chunk")
    semantic_score: Optional[float] = Field(None, description="Semantic coherence score")
    version: str = Field(default="1.0.0", description="Chunking strategy version")
    strategy_name: str = Field(..., description="Name of chunking strategy used")
    custom_fields: Dict[str, Any] = Field(
        default_factory=dict, description="Additional custom metadata"
    )

    @field_validator("end_idx")
    @classmethod
    def validate_end_idx(cls, v: int, info: Any) -> int:
        """Validate end_idx is greater than start_idx."""
        if "start_idx" in info.data and v <= info.data["start_idx"]:
            raise ValueError("end_idx must be greater than start_idx")
        return v


class ChunkResult(BaseModel):
    """Result from a chunking operation."""

    chunks: List[str] = Field(..., description="List of text chunks")
    metadata: List[ChunkMetadata] = Field(..., description="Metadata for each chunk")
    processing_time_ms: float = Field(..., ge=0, description="Processing time in milliseconds")
    strategy_version: str = Field(..., description="Version of chunking strategy")
    config: Dict[str, Any] = Field(..., description="Configuration used for chunking")
    doc_id: Optional[str] = Field(None, description="Original document ID")

    @field_validator("metadata")
    @classmethod
    def validate_metadata_length(cls, v: List[ChunkMetadata], info: Any) -> List[ChunkMetadata]:
        """Validate metadata list matches chunks list length."""
        if "chunks" in info.data and len(v) != len(info.data["chunks"]):
            raise ValueError("Number of metadata items must match number of chunks")
        return v


class EmbeddingResult(BaseModel):
    """Result from an embedding operation."""

    embeddings: List[List[float]] = Field(..., description="List of embedding vectors")
    model_name: str = Field(..., description="Name of embedding model used")
    dimensions: int = Field(..., gt=0, description="Dimensionality of embeddings")
    processing_time_ms: float = Field(..., ge=0, description="Processing time in milliseconds")
    token_count: int = Field(..., ge=0, description="Total tokens processed")
    cost_usd: Optional[float] = Field(None, ge=0, description="Cost in USD")
    provider_version: str = Field(default="1.0.0", description="Provider version")
    provider_name: str = Field(..., description="Name of embedding provider")

    @field_validator("embeddings")
    @classmethod
    def validate_embeddings(cls, v: List[List[float]]) -> List[List[float]]:
        """Validate all embeddings have same dimensions."""
        if not v:
            return v
        first_dim = len(v[0])
        if not all(len(emb) == first_dim for emb in v):
            raise ValueError("All embeddings must have the same dimensionality")
        return v


class MetricResult(BaseModel):
    """Result from a metric computation."""

    metric_name: str = Field(..., description="Name of the metric")
    score: float = Field(..., description="Metric score")
    version: str = Field(default="1.0.0", description="Metric version")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional metric details")
    confidence: Optional[float] = Field(None, ge=0, le=1, description="Confidence score")


class ExperimentRun(BaseModel):
    """Complete experiment run result."""

    # Identifiers
    run_id: str = Field(..., description="Unique run identifier")
    timestamp: datetime = Field(default_factory=datetime.now, description="Run timestamp")

    # Strategy information
    strategy_name: str = Field(..., description="Chunking strategy name")
    strategy_version: str = Field(..., description="Strategy version")
    strategy_config: Dict[str, Any] = Field(..., description="Strategy configuration")

    # Embedding information
    embedding_provider: str = Field(..., description="Embedding provider name")
    embedding_model: str = Field(..., description="Embedding model name")
    embedding_dimensions: int = Field(..., gt=0, description="Embedding dimensions")
    embedding_version: str = Field(..., description="Embedding provider version")

    # Document information
    doc_id: str = Field(..., description="Document identifier")
    doc_length: int = Field(..., ge=0, description="Document length in characters")
    num_chunks: int = Field(..., gt=0, description="Number of chunks created")
    avg_chunk_size: float = Field(..., gt=0, description="Average chunk size")

    # Performance metrics
    chunking_time_ms: float = Field(..., ge=0, description="Chunking time in ms")
    embedding_time_ms: float = Field(..., ge=0, description="Embedding time in ms")
    total_time_ms: float = Field(..., ge=0, description="Total processing time in ms")

    # Quality metrics
    metric_scores: Dict[str, float] = Field(..., description="Metric scores")

    # Cost metrics
    embedding_cost_usd: Optional[float] = Field(None, ge=0, description="Embedding cost")
    total_cost_usd: Optional[float] = Field(None, ge=0, description="Total cost")

    # Metadata (extensible)
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    framework_version: str = Field(default="0.1.0", description="ChunkFlow version")
    schema_version: str = Field(default="1.0.0", description="Schema version")


class StrategyInfo(BaseModel):
    """Information about a chunking strategy."""

    name: str = Field(..., description="Strategy name")
    version: str = Field(..., description="Strategy version")
    description: Optional[str] = Field(None, description="Strategy description")
    default_config: Dict[str, Any] = Field(..., description="Default configuration")
    required_config: List[str] = Field(
        default_factory=list, description="Required config parameters"
    )


class ProviderInfo(BaseModel):
    """Information about an embedding provider."""

    name: str = Field(..., description="Provider name")
    version: str = Field(..., description="Provider version")
    models: List[str] = Field(..., description="Available models")
    dimensions_range: tuple[int, int] = Field(..., description="Min and max dimensions")
    max_context_length: Optional[int] = Field(None, description="Maximum context length")


class MetricInfo(BaseModel):
    """Information about an evaluation metric."""

    name: str = Field(..., description="Metric name")
    version: str = Field(..., description="Metric version")
    description: Optional[str] = Field(None, description="Metric description")
    requires_ground_truth: bool = Field(..., description="Whether ground truth is required")
    range: tuple[float, float] = Field(..., description="Metric value range")


class Config(BaseModel):
    """Configuration model with validation."""

    # Logging
    log_level: str = Field(default="INFO", description="Logging level")
    log_format: str = Field(default="json", description="Log format (json or console)")

    # API settings
    api_host: str = Field(default="0.0.0.0", description="API host")
    api_port: int = Field(default=8000, gt=0, lt=65536, description="API port")
    api_workers: int = Field(default=4, gt=0, description="Number of API workers")

    # Cache settings
    cache_enabled: bool = Field(default=True, description="Enable caching")
    cache_ttl: int = Field(default=3600, gt=0, description="Cache TTL in seconds")

    # Processing settings
    max_concurrent_tasks: int = Field(default=10, gt=0, description="Max concurrent tasks")
    request_timeout: int = Field(default=300, gt=0, description="Request timeout in seconds")

    # API keys (loaded from environment)
    openai_api_key: Optional[str] = Field(None, description="OpenAI API key")
    cohere_api_key: Optional[str] = Field(None, description="Cohere API key")
    voyage_api_key: Optional[str] = Field(None, description="Voyage API key")
    jina_api_key: Optional[str] = Field(None, description="Jina API key")

    model_config = ConfigDict(
        extra="allow",  # Allow additional fields for extensibility
    )
