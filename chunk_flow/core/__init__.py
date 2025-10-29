"""Core abstractions and base classes for ChunkFlow."""

from chunk_flow.core import exceptions
from chunk_flow.core.base import (
    ChunkingStrategy,
    EmbeddingProvider,
    EvaluationMetric,
    TrainableChunkingStrategy,
)
from chunk_flow.core.models import (
    ChunkMetadata,
    ChunkResult,
    Config,
    EmbeddingResult,
    ExperimentRun,
    MetricInfo,
    MetricResult,
    ProviderInfo,
    StrategyInfo,
)
from chunk_flow.core.version import Version, Versioned

__all__ = [
    # Exceptions
    "exceptions",
    # Base classes
    "ChunkingStrategy",
    "EmbeddingProvider",
    "EvaluationMetric",
    "TrainableChunkingStrategy",
    # Models
    "ChunkMetadata",
    "ChunkResult",
    "EmbeddingResult",
    "MetricResult",
    "ExperimentRun",
    "StrategyInfo",
    "ProviderInfo",
    "MetricInfo",
    "Config",
    # Version tracking
    "Version",
    "Versioned",
]
