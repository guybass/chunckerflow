"""Evaluation framework for chunking strategies."""

from chunk_flow.evaluation.registry import MetricRegistry
from chunk_flow.evaluation.pipeline import EvaluationPipeline
from chunk_flow.evaluation.comparison import StrategyComparator

# Import metrics for easier access
try:
    from chunk_flow.evaluation.metrics.retrieval import (
        NDCGMetric,
        RecallAtKMetric,
        PrecisionAtKMetric,
        MRRMetric,
    )
except ImportError:
    pass

try:
    from chunk_flow.evaluation.metrics.semantic import (
        SemanticCoherenceMetric,
        ChunkBoundaryQualityMetric,
        ChunkStickinessMetric,
        TopicDiversityMetric,
    )
except ImportError:
    pass

try:
    from chunk_flow.evaluation.metrics.rag_quality import (
        ContextRelevanceMetric,
        AnswerFaithfulnessMetric,
        ContextPrecisionMetric,
        ContextRecallMetric,
    )
except ImportError:
    pass

__all__ = [
    "MetricRegistry",
    "EvaluationPipeline",
    "StrategyComparator",
    # Retrieval metrics
    "NDCGMetric",
    "RecallAtKMetric",
    "PrecisionAtKMetric",
    "MRRMetric",
    # Semantic metrics
    "SemanticCoherenceMetric",
    "ChunkBoundaryQualityMetric",
    "ChunkStickinessMetric",
    "TopicDiversityMetric",
    # RAG quality metrics
    "ContextRelevanceMetric",
    "AnswerFaithfulnessMetric",
    "ContextPrecisionMetric",
    "ContextRecallMetric",
]
