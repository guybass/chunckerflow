"""Pydantic models for API requests and responses."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ChunkRequest(BaseModel):
    """Request model for chunking endpoint."""

    text: str = Field(..., description="Text to chunk", min_length=1)
    strategy: str = Field("recursive", description="Chunking strategy name")
    config: Optional[Dict[str, Any]] = Field(None, description="Strategy configuration")
    doc_id: Optional[str] = Field(None, description="Document identifier")


class ChunkResponse(BaseModel):
    """Response model for chunking endpoint."""

    chunks: List[str]
    num_chunks: int
    processing_time_ms: float
    strategy: str
    strategy_version: str
    metadata: Optional[List[Dict[str, Any]]] = None


class EmbedRequest(BaseModel):
    """Request model for embedding endpoint."""

    texts: List[str] = Field(..., description="Texts to embed", min_length=1)
    provider: str = Field("huggingface", description="Embedding provider name")
    config: Optional[Dict[str, Any]] = Field(None, description="Provider configuration")


class EmbedResponse(BaseModel):
    """Response model for embedding endpoint."""

    embeddings: List[List[float]]
    dimensions: int
    token_count: Optional[int] = None
    processing_time_ms: float
    provider: str
    model: str
    cost_usd: Optional[float] = None


class EvaluateRequest(BaseModel):
    """Request model for evaluation endpoint."""

    chunks: List[str] = Field(..., description="Chunks to evaluate")
    embeddings: Optional[List[List[float]]] = Field(None, description="Chunk embeddings")
    metrics: Optional[List[str]] = Field(None, description="Metrics to compute (None = all)")
    ground_truth: Optional[Dict[str, Any]] = Field(None, description="Ground truth data")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context")


class EvaluateResponse(BaseModel):
    """Response model for evaluation endpoint."""

    results: Dict[str, Dict[str, Any]]
    num_metrics: int
    processing_time_ms: float


class CompareRequest(BaseModel):
    """Request model for strategy comparison endpoint."""

    text: str = Field(..., description="Text to chunk and compare")
    strategies: List[Dict[str, Any]] = Field(
        ...,
        description="List of strategy configs [{name: 'recursive', config: {...}}]",
        min_length=2,
    )
    embedding_provider: Optional[str] = Field("huggingface", description="Embedding provider")
    embedding_config: Optional[Dict[str, Any]] = Field(None, description="Embedding config")
    metrics: Optional[List[str]] = Field(None, description="Metrics to use")
    ground_truth: Optional[Dict[str, Any]] = Field(None, description="Ground truth")


class CompareResponse(BaseModel):
    """Response model for strategy comparison endpoint."""

    strategies: Dict[str, Any]
    rankings: Dict[str, List[Dict[str, Any]]]
    best_strategy: str
    processing_time_ms: float


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""

    status: str
    version: str
    available_strategies: List[str]
    available_metrics: List[str]
    available_providers: List[str]


class ErrorResponse(BaseModel):
    """Response model for errors."""

    error: str
    detail: Optional[str] = None
    error_type: Optional[str] = None
