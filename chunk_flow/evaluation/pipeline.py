"""Evaluation pipeline for comprehensive chunking strategy assessment."""

import time
from typing import Any, Dict, List, Optional

from chunk_flow.core.base import ChunkingStrategy, EvaluationMetric
from chunk_flow.core.exceptions import EvaluationError
from chunk_flow.core.models import ChunkResult, MetricResult
from chunk_flow.evaluation.registry import MetricRegistry
from chunk_flow.utils.async_helpers import gather_with_concurrency
from chunk_flow.utils.logging import get_logger

logger = get_logger(__name__)


class EvaluationPipeline:
    """
    Evaluation pipeline for chunking strategies.

    Orchestrates multi-metric evaluation with configurable metrics,
    ground truth, and comparison capabilities.
    """

    def __init__(
        self,
        metrics: Optional[List[str]] = None,
        metric_configs: Optional[Dict[str, Dict[str, Any]]] = None,
        max_concurrency: int = 5,
    ) -> None:
        """
        Initialize evaluation pipeline.

        Args:
            metrics: List of metric names to use (default: all available)
            metric_configs: Optional per-metric configurations
            max_concurrency: Maximum concurrent metric evaluations
        """
        self.metric_names = metrics or MetricRegistry.get_metric_names()
        self.metric_configs = metric_configs or {}
        self.max_concurrency = max_concurrency

        # Create metric instances
        self.metrics: List[EvaluationMetric] = []
        for name in self.metric_names:
            config = self.metric_configs.get(name, {})
            try:
                metric = MetricRegistry.create(name, config)
                self.metrics.append(metric)
            except Exception as e:
                logger.warning("metric_initialization_failed", name=name, error=str(e))

        logger.info("evaluation_pipeline_initialized", num_metrics=len(self.metrics))

    async def evaluate(
        self,
        chunks: List[str],
        embeddings: Optional[List[List[float]]] = None,
        ground_truth: Optional[Any] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, MetricResult]:
        """
        Evaluate chunks with all configured metrics.

        Args:
            chunks: Text chunks to evaluate
            embeddings: Chunk embeddings (required for most metrics)
            ground_truth: Ground truth data (required for some metrics)
            context: Additional context for metrics

        Returns:
            Dict mapping metric names to results
        """
        start_time = time.time()

        logger.info(
            "evaluation_started",
            num_chunks=len(chunks),
            num_metrics=len(self.metrics),
            has_embeddings=embeddings is not None,
            has_ground_truth=ground_truth is not None,
        )

        # Create evaluation tasks
        tasks = []
        for metric in self.metrics:
            # Skip metrics that require ground truth if not provided
            if metric.REQUIRES_GROUND_TRUTH and ground_truth is None:
                logger.debug(
                    "skipping_metric_no_ground_truth",
                    metric=metric.METRIC_NAME,
                )
                continue

            tasks.append(self._evaluate_metric(metric, chunks, embeddings, ground_truth, context))

        # Execute metrics concurrently
        try:
            results = await gather_with_concurrency(self.max_concurrency, *tasks)

            # Build results dict
            results_dict = {result.metric_name: result for result in results if result is not None}

            elapsed = (time.time() - start_time) * 1000
            logger.info(
                "evaluation_completed",
                num_results=len(results_dict),
                elapsed_ms=elapsed,
            )

            return results_dict

        except Exception as e:
            logger.error("evaluation_failed", error=str(e), exc_info=True)
            raise EvaluationError(f"Evaluation pipeline failed: {str(e)}") from e

    async def _evaluate_metric(
        self,
        metric: EvaluationMetric,
        chunks: List[str],
        embeddings: Optional[List[List[float]]],
        ground_truth: Optional[Any],
        context: Optional[Dict[str, Any]],
    ) -> Optional[MetricResult]:
        """
        Evaluate single metric with error handling.

        Args:
            metric: Metric instance
            chunks: Text chunks
            embeddings: Chunk embeddings
            ground_truth: Ground truth data
            context: Additional context

        Returns:
            MetricResult or None if evaluation failed
        """
        try:
            result = await metric.compute(chunks, embeddings, ground_truth, context)
            logger.debug("metric_computed", metric=metric.METRIC_NAME, score=result.score)
            return result

        except Exception as e:
            logger.error(
                "metric_computation_failed",
                metric=metric.METRIC_NAME,
                error=str(e),
                exc_info=True,
            )
            return None

    async def evaluate_strategy(
        self,
        strategy: ChunkingStrategy,
        text: str,
        embeddings: Optional[List[List[float]]] = None,
        ground_truth: Optional[Any] = None,
        context: Optional[Dict[str, Any]] = None,
        doc_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate a chunking strategy end-to-end.

        Args:
            strategy: Chunking strategy to evaluate
            text: Input text to chunk
            embeddings: Optional pre-computed embeddings
            ground_truth: Ground truth for metrics
            context: Additional context
            doc_id: Document identifier

        Returns:
            Dict with chunk_result and metric_results
        """
        logger.info("evaluating_strategy", strategy=strategy.NAME, doc_id=doc_id)

        # Step 1: Chunk the text
        chunk_result = await strategy.chunk(text, doc_id=doc_id)

        # Step 2: Evaluate chunks
        metric_results = await self.evaluate(
            chunks=chunk_result.chunks,
            embeddings=embeddings,
            ground_truth=ground_truth,
            context=context,
        )

        return {
            "chunk_result": chunk_result,
            "metric_results": metric_results,
            "strategy_name": strategy.NAME,
            "strategy_version": strategy.VERSION,
        }

    async def compare_strategies(
        self,
        strategies: List[ChunkingStrategy],
        text: str,
        embeddings_per_strategy: Optional[Dict[str, List[List[float]]]] = None,
        ground_truth: Optional[Any] = None,
        context: Optional[Dict[str, Any]] = None,
        doc_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Compare multiple chunking strategies.

        Args:
            strategies: List of strategies to compare
            text: Input text
            embeddings_per_strategy: Optional embeddings per strategy name
            ground_truth: Ground truth for evaluation
            context: Additional context
            doc_id: Document identifier

        Returns:
            Dict with comparison results and rankings
        """
        start_time = time.time()

        logger.info(
            "comparing_strategies",
            num_strategies=len(strategies),
            doc_id=doc_id,
        )

        # Evaluate each strategy
        tasks = []
        for strategy in strategies:
            embeddings = None
            if embeddings_per_strategy:
                embeddings = embeddings_per_strategy.get(strategy.NAME)

            tasks.append(
                self.evaluate_strategy(
                    strategy=strategy,
                    text=text,
                    embeddings=embeddings,
                    ground_truth=ground_truth,
                    context=context,
                    doc_id=doc_id,
                )
            )

        results = await gather_with_concurrency(self.max_concurrency, *tasks)

        # Build comparison dict
        comparison = {
            "strategies": {},
            "rankings": {},
            "metadata": {
                "num_strategies": len(strategies),
                "doc_id": doc_id,
                "elapsed_ms": (time.time() - start_time) * 1000,
            },
        }

        # Organize results by strategy
        for result in results:
            strategy_name = result["strategy_name"]
            comparison["strategies"][strategy_name] = result

        # Compute rankings per metric
        comparison["rankings"] = self._compute_rankings(results)

        logger.info(
            "strategy_comparison_completed",
            num_strategies=len(strategies),
            elapsed_ms=comparison["metadata"]["elapsed_ms"],
        )

        return comparison

    def _compute_rankings(self, results: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Compute strategy rankings per metric.

        Args:
            results: List of strategy evaluation results

        Returns:
            Dict mapping metric names to ranked strategies
        """
        rankings: Dict[str, List[Dict[str, Any]]] = {}

        # Get all metric names
        all_metrics = set()
        for result in results:
            all_metrics.update(result["metric_results"].keys())

        # Rank for each metric
        for metric_name in all_metrics:
            metric_scores = []

            for result in results:
                metric_result = result["metric_results"].get(metric_name)
                if metric_result:
                    metric_scores.append({
                        "strategy": result["strategy_name"],
                        "score": metric_result.score,
                        "details": metric_result.details,
                    })

            # Sort by score (descending for most metrics, ascending for stickiness)
            reverse = metric_name != "chunk_stickiness"  # Lower is better for stickiness
            metric_scores.sort(key=lambda x: x["score"], reverse=reverse)

            # Add ranks
            for rank, item in enumerate(metric_scores, 1):
                item["rank"] = rank

            rankings[metric_name] = metric_scores

        return rankings

    def get_summary(self, evaluation_result: Dict[str, MetricResult]) -> Dict[str, Any]:
        """
        Generate summary statistics from evaluation results.

        Args:
            evaluation_result: Dict of metric results

        Returns:
            Summary dict with aggregated stats
        """
        if not evaluation_result:
            return {"message": "No evaluation results"}

        summary = {
            "num_metrics": len(evaluation_result),
            "scores": {},
            "average_score": 0.0,
            "metrics": {},
        }

        total_score = 0.0
        for metric_name, result in evaluation_result.items():
            summary["scores"][metric_name] = result.score
            summary["metrics"][metric_name] = {
                "score": result.score,
                "version": result.version,
                "details": result.details,
            }
            total_score += result.score

        summary["average_score"] = total_score / len(evaluation_result)

        return summary

    def filter_metrics_by_category(self, category: str) -> "EvaluationPipeline":
        """
        Create new pipeline with metrics filtered by category.

        Args:
            category: Category name (retrieval, semantic, rag_quality)

        Returns:
            New EvaluationPipeline instance
        """
        categories = MetricRegistry.get_metrics_by_category()
        filtered_metrics = categories.get(category, [])

        return EvaluationPipeline(
            metrics=filtered_metrics,
            metric_configs=self.metric_configs,
            max_concurrency=self.max_concurrency,
        )
