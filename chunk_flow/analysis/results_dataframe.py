"""ResultsDataFrame for comprehensive strategy analysis."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from chunk_flow.core.models import MetricResult
from chunk_flow.utils.logging import get_logger

logger = get_logger(__name__)


class ResultsDataFrame:
    """
    DataFrame wrapper for comprehensive chunking strategy analysis.

    Provides methods for filtering, ranking, aggregation, statistical analysis,
    and export capabilities. Built on pandas for powerful data manipulation.
    """

    def __init__(self, data: Optional[pd.DataFrame] = None) -> None:
        """
        Initialize ResultsDataFrame.

        Args:
            data: Optional pandas DataFrame with results
        """
        self.df = data if data is not None else pd.DataFrame()
        logger.info("results_dataframe_initialized", rows=len(self.df))

    @classmethod
    def from_comparison_results(
        cls, comparison_results: Dict[str, Any]
    ) -> "ResultsDataFrame":
        """
        Create ResultsDataFrame from EvaluationPipeline comparison results.

        Args:
            comparison_results: Output from EvaluationPipeline.compare_strategies()

        Returns:
            ResultsDataFrame instance
        """
        rows = []

        strategies_data = comparison_results.get("strategies", {})

        for strategy_name, strategy_data in strategies_data.items():
            metric_results = strategy_data.get("metric_results", {})
            chunk_result = strategy_data.get("chunk_result")

            base_row = {
                "strategy": strategy_name,
                "strategy_version": strategy_data.get("strategy_version"),
                "num_chunks": len(chunk_result.chunks) if chunk_result else 0,
                "processing_time_ms": (
                    chunk_result.processing_time_ms if chunk_result else 0.0
                ),
            }

            # Add metrics as columns
            for metric_name, metric_result in metric_results.items():
                base_row[f"metric_{metric_name}"] = metric_result.score

                # Add key details as separate columns
                if metric_result.details:
                    for detail_key, detail_value in metric_result.details.items():
                        # Only add scalar values
                        if isinstance(detail_value, (int, float, str, bool)):
                            base_row[f"{metric_name}_{detail_key}"] = detail_value

            rows.append(base_row)

        df = pd.DataFrame(rows)
        logger.info("dataframe_created_from_comparison", rows=len(df))

        return cls(df)

    @classmethod
    def from_evaluation_results(
        cls,
        results: Dict[str, Dict[str, MetricResult]],
        metadata: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> "ResultsDataFrame":
        """
        Create ResultsDataFrame from evaluation results.

        Args:
            results: Dict mapping strategy names to metric results
            metadata: Optional metadata per strategy (num_chunks, processing_time, etc.)

        Returns:
            ResultsDataFrame instance
        """
        rows = []

        for strategy_name, metric_results in results.items():
            row = {"strategy": strategy_name}

            # Add metadata if provided
            if metadata and strategy_name in metadata:
                row.update(metadata[strategy_name])

            # Add metric scores
            for metric_name, metric_result in metric_results.items():
                row[f"metric_{metric_name}"] = metric_result.score

            rows.append(row)

        df = pd.DataFrame(rows)
        return cls(df)

    def get_metric_columns(self) -> List[str]:
        """
        Get list of metric column names.

        Returns:
            List of column names starting with 'metric_'
        """
        return [col for col in self.df.columns if col.startswith("metric_")]

    def rank_strategies(
        self,
        by: Optional[Union[str, List[str]]] = None,
        ascending: bool = False,
        weights: Optional[Dict[str, float]] = None,
    ) -> "ResultsDataFrame":
        """
        Rank strategies by metric(s).

        Args:
            by: Metric name(s) to rank by. If None, uses weighted average of all metrics.
            ascending: Sort order (False = best first)
            weights: Metric weights for weighted average (only used if by=None)

        Returns:
            New ResultsDataFrame with ranked results
        """
        df = self.df.copy()

        if by is None:
            # Compute weighted average
            metric_cols = self.get_metric_columns()

            if not metric_cols:
                logger.warning("no_metrics_found", msg="Cannot rank without metrics")
                return ResultsDataFrame(df)

            # Default equal weights
            if weights is None:
                weights = {col.replace("metric_", ""): 1.0 for col in metric_cols}

            # Compute weighted score
            weighted_sum = pd.Series(0.0, index=df.index)
            total_weight = 0.0

            for col in metric_cols:
                metric_name = col.replace("metric_", "")
                weight = weights.get(metric_name, 1.0)

                # Handle stickiness (lower is better)
                if "stickiness" in metric_name:
                    weighted_sum += (1.0 - df[col]) * weight
                else:
                    weighted_sum += df[col] * weight

                total_weight += weight

            df["weighted_score"] = weighted_sum / total_weight if total_weight > 0 else 0.0
            df = df.sort_values("weighted_score", ascending=ascending)
            df["rank"] = range(1, len(df) + 1)

        else:
            # Rank by specific metric(s)
            if isinstance(by, str):
                by = [by]

            # Add 'metric_' prefix if needed
            sort_cols = [col if col.startswith("metric_") else f"metric_{col}" for col in by]

            # Check columns exist
            missing = [col for col in sort_cols if col not in df.columns]
            if missing:
                raise ValueError(f"Metrics not found: {missing}")

            df = df.sort_values(sort_cols, ascending=ascending)
            df["rank"] = range(1, len(df) + 1)

        return ResultsDataFrame(df)

    def get_best(
        self,
        metric: Optional[str] = None,
        n: int = 1,
    ) -> Union[pd.Series, pd.DataFrame]:
        """
        Get best performing strategy/strategies.

        Args:
            metric: Metric to optimize for (None = weighted average)
            n: Number of top strategies to return

        Returns:
            Series (if n=1) or DataFrame (if n>1) with best strategies
        """
        ranked = self.rank_strategies(by=metric)

        if n == 1:
            return ranked.df.iloc[0]
        else:
            return ranked.df.head(n)

    def filter_strategies(self, **conditions: Any) -> "ResultsDataFrame":
        """
        Filter strategies by conditions.

        Args:
            **conditions: Column conditions (e.g., num_chunks__gt=10)

        Returns:
            Filtered ResultsDataFrame

        Examples:
            df.filter_strategies(num_chunks__gt=5, processing_time_ms__lt=1000)
        """
        df = self.df.copy()

        for key, value in conditions.items():
            if "__" in key:
                col, op = key.rsplit("__", 1)
            else:
                col, op = key, "eq"

            if col not in df.columns:
                logger.warning("column_not_found", column=col)
                continue

            if op == "gt":
                df = df[df[col] > value]
            elif op == "gte":
                df = df[df[col] >= value]
            elif op == "lt":
                df = df[df[col] < value]
            elif op == "lte":
                df = df[df[col] <= value]
            elif op == "eq":
                df = df[df[col] == value]
            elif op == "ne":
                df = df[df[col] != value]
            elif op == "in":
                df = df[df[col].isin(value)]
            else:
                logger.warning("unknown_operator", operator=op)

        return ResultsDataFrame(df)

    def aggregate_by(
        self,
        group_by: Union[str, List[str]],
        agg_func: str = "mean",
    ) -> pd.DataFrame:
        """
        Aggregate results by column(s).

        Args:
            group_by: Column(s) to group by
            agg_func: Aggregation function ('mean', 'median', 'min', 'max', 'std')

        Returns:
            Aggregated DataFrame
        """
        metric_cols = self.get_metric_columns()

        if not metric_cols:
            logger.warning("no_metrics_to_aggregate")
            return pd.DataFrame()

        numeric_cols = metric_cols + ["num_chunks", "processing_time_ms"]
        numeric_cols = [col for col in numeric_cols if col in self.df.columns]

        return self.df.groupby(group_by)[numeric_cols].agg(agg_func)

    def describe(self) -> pd.DataFrame:
        """
        Get statistical summary of all metrics.

        Returns:
            DataFrame with descriptive statistics
        """
        metric_cols = self.get_metric_columns()

        if not metric_cols:
            return pd.DataFrame()

        return self.df[metric_cols].describe()

    def correlation_matrix(self) -> pd.DataFrame:
        """
        Compute correlation matrix between metrics.

        Returns:
            Correlation matrix DataFrame
        """
        metric_cols = self.get_metric_columns()

        if len(metric_cols) < 2:
            logger.warning("insufficient_metrics_for_correlation")
            return pd.DataFrame()

        return self.df[metric_cols].corr()

    def compare_metrics(self, metric1: str, metric2: str) -> pd.DataFrame:
        """
        Compare two metrics across strategies.

        Args:
            metric1: First metric name
            metric2: Second metric name

        Returns:
            DataFrame with comparison
        """
        col1 = f"metric_{metric1}" if not metric1.startswith("metric_") else metric1
        col2 = f"metric_{metric2}" if not metric2.startswith("metric_") else metric2

        if col1 not in self.df.columns or col2 not in self.df.columns:
            raise ValueError(f"Metrics not found: {metric1}, {metric2}")

        comparison = self.df[["strategy", col1, col2]].copy()
        comparison["difference"] = comparison[col1] - comparison[col2]
        comparison["ratio"] = comparison[col1] / comparison[col2]

        return comparison

    def pivot_table(
        self,
        values: str,
        index: str = "strategy",
        aggfunc: str = "mean",
    ) -> pd.DataFrame:
        """
        Create pivot table.

        Args:
            values: Column to aggregate
            index: Row index
            aggfunc: Aggregation function

        Returns:
            Pivot table DataFrame
        """
        return pd.pivot_table(
            self.df,
            values=values,
            index=index,
            aggfunc=aggfunc,
        )

    def export(
        self,
        path: Union[str, Path],
        format: str = "csv",
        **kwargs: Any,
    ) -> None:
        """
        Export results to file.

        Args:
            path: Output file path
            format: Export format ('csv', 'json', 'parquet', 'excel')
            **kwargs: Additional arguments for export function
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if format == "csv":
            self.df.to_csv(path, index=False, **kwargs)
        elif format == "json":
            self.df.to_json(path, orient="records", indent=2, **kwargs)
        elif format == "parquet":
            self.df.to_parquet(path, index=False, **kwargs)
        elif format == "excel":
            self.df.to_excel(path, index=False, **kwargs)
        else:
            raise ValueError(f"Unsupported format: {format}")

        logger.info("results_exported", path=str(path), format=format, rows=len(self.df))

    @classmethod
    def load(cls, path: Union[str, Path], format: Optional[str] = None) -> "ResultsDataFrame":
        """
        Load results from file.

        Args:
            path: Input file path
            format: File format (auto-detected from extension if None)

        Returns:
            ResultsDataFrame instance
        """
        path = Path(path)

        if format is None:
            format = path.suffix.lstrip(".")

        if format == "csv":
            df = pd.read_csv(path)
        elif format == "json":
            df = pd.read_json(path)
        elif format == "parquet":
            df = pd.read_parquet(path)
        elif format in ("xlsx", "excel"):
            df = pd.read_excel(path)
        else:
            raise ValueError(f"Unsupported format: {format}")

        logger.info("results_loaded", path=str(path), rows=len(df))
        return cls(df)

    def to_dict(self, orient: str = "records") -> Union[Dict, List[Dict]]:
        """
        Convert to dictionary.

        Args:
            orient: Orientation ('records', 'dict', 'list', 'index')

        Returns:
            Dict or List of dicts
        """
        return self.df.to_dict(orient=orient)

    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics.

        Returns:
            Summary dict with key statistics
        """
        metric_cols = self.get_metric_columns()

        summary = {
            "num_strategies": len(self.df),
            "num_metrics": len(metric_cols),
            "metrics": {},
        }

        for col in metric_cols:
            metric_name = col.replace("metric_", "")
            summary["metrics"][metric_name] = {
                "mean": float(self.df[col].mean()),
                "std": float(self.df[col].std()),
                "min": float(self.df[col].min()),
                "max": float(self.df[col].max()),
                "median": float(self.df[col].median()),
            }

        return summary

    def __repr__(self) -> str:
        """String representation."""
        return f"ResultsDataFrame(strategies={len(self.df)}, metrics={len(self.get_metric_columns())})"

    def __len__(self) -> int:
        """Number of strategies."""
        return len(self.df)
