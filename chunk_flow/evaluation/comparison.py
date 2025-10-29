"""Strategy comparison framework with analysis utilities."""

from typing import Any, Dict, List, Optional

import numpy as np

from chunk_flow.core.models import MetricResult
from chunk_flow.utils.logging import get_logger

logger = get_logger(__name__)


class StrategyComparator:
    """
    Utility for comparing and analyzing chunking strategy performance.

    Provides methods for ranking, statistical analysis, and comparison reports.
    """

    @staticmethod
    def rank_strategies(
        results: Dict[str, Dict[str, MetricResult]],
        metric_weights: Optional[Dict[str, float]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Rank strategies by weighted average of metric scores.

        Args:
            results: Dict mapping strategy names to their metric results
            metric_weights: Optional weights per metric (default: equal weights)

        Returns:
            List of strategies ranked by weighted score
        """
        if not results:
            return []

        # Get all metrics
        all_metrics = set()
        for strategy_results in results.values():
            all_metrics.update(strategy_results.keys())

        # Default equal weights
        if metric_weights is None:
            metric_weights = {metric: 1.0 for metric in all_metrics}

        # Compute weighted scores
        strategy_scores = []
        for strategy_name, strategy_results in results.items():
            weighted_score = 0.0
            total_weight = 0.0

            for metric_name, metric_result in strategy_results.items():
                weight = metric_weights.get(metric_name, 1.0)

                # Invert stickiness (lower is better)
                score = metric_result.score
                if metric_name == "chunk_stickiness":
                    score = 1.0 - score

                weighted_score += score * weight
                total_weight += weight

            avg_score = weighted_score / total_weight if total_weight > 0 else 0.0

            strategy_scores.append({
                "strategy": strategy_name,
                "weighted_score": avg_score,
                "raw_scores": {
                    name: result.score for name, result in strategy_results.items()
                },
            })

        # Sort by weighted score
        strategy_scores.sort(key=lambda x: x["weighted_score"], reverse=True)

        # Add ranks
        for rank, item in enumerate(strategy_scores, 1):
            item["rank"] = rank

        return strategy_scores

    @staticmethod
    def compute_pairwise_differences(
        results: Dict[str, Dict[str, MetricResult]],
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute pairwise score differences between strategies.

        Args:
            results: Dict mapping strategy names to metric results

        Returns:
            Dict of pairwise differences
        """
        strategies = list(results.keys())
        differences = {}

        for i, strategy_a in enumerate(strategies):
            for strategy_b in strategies[i + 1:]:
                pair_key = f"{strategy_a}_vs_{strategy_b}"
                differences[pair_key] = {}

                # Get common metrics
                metrics_a = set(results[strategy_a].keys())
                metrics_b = set(results[strategy_b].keys())
                common_metrics = metrics_a & metrics_b

                for metric in common_metrics:
                    score_a = results[strategy_a][metric].score
                    score_b = results[strategy_b][metric].score
                    differences[pair_key][metric] = score_a - score_b

        return differences

    @staticmethod
    def identify_best_strategy_per_metric(
        results: Dict[str, Dict[str, MetricResult]],
    ) -> Dict[str, Dict[str, Any]]:
        """
        Identify best performing strategy for each metric.

        Args:
            results: Dict mapping strategy names to metric results

        Returns:
            Dict mapping metric names to best strategy
        """
        best_per_metric = {}

        # Get all metrics
        all_metrics = set()
        for strategy_results in results.values():
            all_metrics.update(strategy_results.keys())

        for metric in all_metrics:
            scores = []
            for strategy_name, strategy_results in results.items():
                if metric in strategy_results:
                    score = strategy_results[metric].score

                    # Invert stickiness (lower is better)
                    if metric == "chunk_stickiness":
                        score = 1.0 - score

                    scores.append({
                        "strategy": strategy_name,
                        "score": strategy_results[metric].score,
                        "adjusted_score": score,
                    })

            if scores:
                # Find best
                best = max(scores, key=lambda x: x["adjusted_score"])
                best_per_metric[metric] = best

        return best_per_metric

    @staticmethod
    def compute_statistical_summary(
        results: Dict[str, Dict[str, MetricResult]],
    ) -> Dict[str, Any]:
        """
        Compute statistical summary across all strategies and metrics.

        Args:
            results: Dict mapping strategy names to metric results

        Returns:
            Dict with statistical summaries
        """
        summary = {
            "num_strategies": len(results),
            "metrics": {},
            "strategies": {},
        }

        # Get all metrics
        all_metrics = set()
        for strategy_results in results.values():
            all_metrics.update(strategy_results.keys())

        # Per-metric statistics
        for metric in all_metrics:
            scores = []
            for strategy_results in results.values():
                if metric in strategy_results:
                    scores.append(strategy_results[metric].score)

            if scores:
                summary["metrics"][metric] = {
                    "mean": float(np.mean(scores)),
                    "std": float(np.std(scores)),
                    "min": float(np.min(scores)),
                    "max": float(np.max(scores)),
                    "range": float(np.max(scores) - np.min(scores)),
                }

        # Per-strategy statistics
        for strategy_name, strategy_results in results.items():
            scores = [result.score for result in strategy_results.values()]

            summary["strategies"][strategy_name] = {
                "num_metrics": len(strategy_results),
                "mean_score": float(np.mean(scores)),
                "std_score": float(np.std(scores)),
                "min_score": float(np.min(scores)),
                "max_score": float(np.max(scores)),
            }

        return summary

    @staticmethod
    def generate_comparison_report(
        results: Dict[str, Dict[str, MetricResult]],
        metric_weights: Optional[Dict[str, float]] = None,
    ) -> str:
        """
        Generate human-readable comparison report.

        Args:
            results: Dict mapping strategy names to metric results
            metric_weights: Optional metric weights

        Returns:
            Formatted comparison report string
        """
        if not results:
            return "No results to compare."

        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("CHUNKING STRATEGY COMPARISON REPORT")
        report_lines.append("=" * 80)

        # Overall rankings
        rankings = StrategyComparator.rank_strategies(results, metric_weights)
        report_lines.append("\nOVERALL RANKINGS:")
        report_lines.append("-" * 80)

        for item in rankings:
            report_lines.append(
                f"  {item['rank']}. {item['strategy']:<30} "
                f"(weighted score: {item['weighted_score']:.4f})"
            )

        # Best per metric
        best_per_metric = StrategyComparator.identify_best_strategy_per_metric(results)
        report_lines.append("\nBEST STRATEGY PER METRIC:")
        report_lines.append("-" * 80)

        for metric, info in best_per_metric.items():
            report_lines.append(
                f"  {metric:<30} → {info['strategy']:<20} "
                f"(score: {info['score']:.4f})"
            )

        # Statistical summary
        stats = StrategyComparator.compute_statistical_summary(results)
        report_lines.append("\nMETRIC STATISTICS:")
        report_lines.append("-" * 80)

        for metric, metric_stats in stats["metrics"].items():
            report_lines.append(
                f"  {metric:<30} "
                f"mean={metric_stats['mean']:.4f} "
                f"std={metric_stats['std']:.4f} "
                f"range=[{metric_stats['min']:.4f}, {metric_stats['max']:.4f}]"
            )

        # Per-strategy breakdown
        report_lines.append("\nPER-STRATEGY BREAKDOWN:")
        report_lines.append("-" * 80)

        for strategy_name, strategy_results in results.items():
            report_lines.append(f"\n  {strategy_name}:")
            for metric_name, metric_result in sorted(strategy_results.items()):
                report_lines.append(f"    {metric_name:<28} {metric_result.score:.4f}")

        report_lines.append("\n" + "=" * 80)

        return "\n".join(report_lines)

    @staticmethod
    def get_score_matrix(
        results: Dict[str, Dict[str, MetricResult]],
    ) -> Dict[str, Any]:
        """
        Get score matrix for visualization.

        Args:
            results: Dict mapping strategy names to metric results

        Returns:
            Dict with matrix data, strategy names, and metric names
        """
        strategies = sorted(results.keys())

        # Get all metrics
        all_metrics = set()
        for strategy_results in results.values():
            all_metrics.update(strategy_results.keys())
        metrics = sorted(all_metrics)

        # Build matrix
        matrix = []
        for strategy in strategies:
            row = []
            for metric in metrics:
                if metric in results[strategy]:
                    row.append(results[strategy][metric].score)
                else:
                    row.append(None)  # Missing value
            matrix.append(row)

        return {
            "matrix": matrix,
            "strategies": strategies,
            "metrics": metrics,
        }

    @staticmethod
    def compute_win_matrix(
        results: Dict[str, Dict[str, MetricResult]],
    ) -> Dict[str, Any]:
        """
        Compute win matrix: how often each strategy beats others.

        Args:
            results: Dict mapping strategy names to metric results

        Returns:
            Win matrix and statistics
        """
        strategies = sorted(results.keys())
        n = len(strategies)

        # Initialize win matrix
        win_matrix = [[0 for _ in range(n)] for _ in range(n)]

        # Get all metrics
        all_metrics = set()
        for strategy_results in results.values():
            all_metrics.update(strategy_results.keys())

        # Count wins
        for metric in all_metrics:
            # Get scores for this metric
            metric_scores = {}
            for strategy in strategies:
                if metric in results[strategy]:
                    score = results[strategy][metric].score
                    # Invert stickiness
                    if metric == "chunk_stickiness":
                        score = 1.0 - score
                    metric_scores[strategy] = score

            # Count wins
            for i, strategy_a in enumerate(strategies):
                for j, strategy_b in enumerate(strategies):
                    if i != j:
                        if strategy_a in metric_scores and strategy_b in metric_scores:
                            if metric_scores[strategy_a] > metric_scores[strategy_b]:
                                win_matrix[i][j] += 1

        # Compute win percentages
        total_comparisons = len(all_metrics) * (n - 1)
        win_percentages = []
        for i, strategy in enumerate(strategies):
            total_wins = sum(win_matrix[i])
            win_pct = (total_wins / total_comparisons * 100) if total_comparisons > 0 else 0.0
            win_percentages.append({
                "strategy": strategy,
                "wins": total_wins,
                "win_percentage": win_pct,
            })

        return {
            "win_matrix": win_matrix,
            "strategies": strategies,
            "win_percentages": sorted(win_percentages, key=lambda x: x["win_percentage"], reverse=True),
        }
