"""Registry for evaluation metrics."""

from typing import Any, Dict, List, Type

from chunk_flow.core.base import EvaluationMetric
from chunk_flow.core.exceptions import RegistryError
from chunk_flow.utils.logging import get_logger

logger = get_logger(__name__)


class MetricRegistry:
    """
    Registry for all evaluation metrics.

    Provides metric discovery, registration, and instantiation.
    """

    _metrics: Dict[str, Type[EvaluationMetric]] = {}

    @classmethod
    def register(cls, name: str, metric_class: Type[EvaluationMetric]) -> None:
        """
        Register an evaluation metric.

        Args:
            name: Metric name (must match metric_class.METRIC_NAME)
            metric_class: Metric class to register

        Raises:
            RegistryError: If metric already registered or name mismatch
        """
        if name in cls._metrics:
            logger.warning("metric_already_registered", name=name)
            return  # Allow re-registration for hot reload

        if hasattr(metric_class, "METRIC_NAME") and metric_class.METRIC_NAME != name:
            raise RegistryError(
                f"Metric name mismatch: {name} != {metric_class.METRIC_NAME}"
            )

        cls._metrics[name] = metric_class
        logger.info("metric_registered", name=name, version=metric_class.VERSION)

    @classmethod
    def unregister(cls, name: str) -> None:
        """
        Unregister a metric.

        Args:
            name: Metric name to unregister
        """
        if name in cls._metrics:
            del cls._metrics[name]
            logger.info("metric_unregistered", name=name)

    @classmethod
    def get(cls, name: str) -> Type[EvaluationMetric]:
        """
        Get metric class by name.

        Args:
            name: Metric name

        Returns:
            Metric class

        Raises:
            RegistryError: If metric not found
        """
        if name not in cls._metrics:
            available = ", ".join(cls._metrics.keys())
            raise RegistryError(
                f"Unknown metric: {name}. Available metrics: {available}"
            )
        return cls._metrics[name]

    @classmethod
    def create(cls, name: str, config: Dict[str, Any] | None = None) -> EvaluationMetric:
        """
        Create metric instance.

        Args:
            name: Metric name
            config: Metric configuration

        Returns:
            Metric instance

        Raises:
            RegistryError: If metric not found
        """
        metric_class = cls.get(name)
        return metric_class(config=config)

    @classmethod
    def list_metrics(cls) -> List[Dict[str, Any]]:
        """
        List all registered metrics with metadata.

        Returns:
            List of metric info dicts
        """
        infos: List[Dict[str, Any]] = []

        for name, metric_class in cls._metrics.items():
            infos.append({
                "name": name,
                "version": metric_class.VERSION,
                "requires_ground_truth": metric_class.REQUIRES_GROUND_TRUTH,
                "description": metric_class.__doc__.strip() if metric_class.__doc__ else "",
            })

        return infos

    @classmethod
    def get_metric_names(cls) -> List[str]:
        """
        Get list of registered metric names.

        Returns:
            List of metric names
        """
        return list(cls._metrics.keys())

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """
        Check if metric is registered.

        Args:
            name: Metric name

        Returns:
            True if registered, False otherwise
        """
        return name in cls._metrics

    @classmethod
    def get_metrics_by_category(cls) -> Dict[str, List[str]]:
        """
        Get metrics organized by category.

        Returns:
            Dict mapping category to list of metric names
        """
        categories: Dict[str, List[str]] = {
            "retrieval": [],
            "semantic": [],
            "rag_quality": [],
            "other": [],
        }

        for name, metric_class in cls._metrics.items():
            # Categorize based on module or metric name
            module = metric_class.__module__

            if "retrieval" in module:
                categories["retrieval"].append(name)
            elif "semantic" in module:
                categories["semantic"].append(name)
            elif "rag_quality" in module:
                categories["rag_quality"].append(name)
            else:
                categories["other"].append(name)

        return categories

    @classmethod
    def clear(cls) -> None:
        """Clear all registered metrics (useful for testing)."""
        cls._metrics.clear()
        logger.info("metric_registry_cleared")


# Auto-register built-in metrics
def _register_builtin_metrics() -> None:
    """Register all built-in metrics."""
    count = 0

    # Retrieval metrics
    try:
        from chunk_flow.evaluation.metrics.retrieval import (
            NDCGMetric,
            RecallAtKMetric,
            PrecisionAtKMetric,
            MRRMetric,
        )

        MetricRegistry.register("ndcg_at_k", NDCGMetric)
        MetricRegistry.register("recall_at_k", RecallAtKMetric)
        MetricRegistry.register("precision_at_k", PrecisionAtKMetric)
        MetricRegistry.register("mrr", MRRMetric)
        count += 4

    except ImportError as e:
        logger.warning("retrieval_metrics_registration_failed", error=str(e))

    # Semantic metrics
    try:
        from chunk_flow.evaluation.metrics.semantic import (
            SemanticCoherenceMetric,
            ChunkBoundaryQualityMetric,
            ChunkStickinessMetric,
            TopicDiversityMetric,
        )

        MetricRegistry.register("semantic_coherence", SemanticCoherenceMetric)
        MetricRegistry.register("boundary_quality", ChunkBoundaryQualityMetric)
        MetricRegistry.register("chunk_stickiness", ChunkStickinessMetric)
        MetricRegistry.register("topic_diversity", TopicDiversityMetric)
        count += 4

    except ImportError as e:
        logger.warning("semantic_metrics_registration_failed", error=str(e))

    # RAG quality metrics
    try:
        from chunk_flow.evaluation.metrics.rag_quality import (
            ContextRelevanceMetric,
            AnswerFaithfulnessMetric,
            ContextPrecisionMetric,
            ContextRecallMetric,
        )

        MetricRegistry.register("context_relevance", ContextRelevanceMetric)
        MetricRegistry.register("answer_faithfulness", AnswerFaithfulnessMetric)
        MetricRegistry.register("context_precision", ContextPrecisionMetric)
        MetricRegistry.register("context_recall", ContextRecallMetric)
        count += 4

    except ImportError as e:
        logger.warning("rag_metrics_registration_failed", error=str(e))

    logger.info("builtin_metrics_registered", count=count)


# Register on import
_register_builtin_metrics()
