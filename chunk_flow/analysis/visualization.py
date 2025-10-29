"""Visualization utilities for chunking strategy analysis."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from chunk_flow.utils.logging import get_logger

logger = get_logger(__name__)

# Set style
sns.set_style("whitegrid")
sns.set_palette("husl")


class StrategyVisualizer:
    """
    Visualization utilities for chunking strategy analysis.

    Provides methods for creating heatmaps, bar charts, radar charts,
    and other visualizations for strategy comparison.
    """

    @staticmethod
    def plot_heatmap(
        data: Union[pd.DataFrame, np.ndarray],
        strategies: Optional[List[str]] = None,
        metrics: Optional[List[str]] = None,
        title: str = "Strategy Performance Heatmap",
        figsize: tuple = (12, 8),
        cmap: str = "RdYlGn",
        annot: bool = True,
        fmt: str = ".3f",
        output_path: Optional[Union[str, Path]] = None,
    ) -> plt.Figure:
        """
        Create heatmap of strategy performance across metrics.

        Args:
            data: DataFrame or matrix with scores
            strategies: Strategy names (row labels)
            metrics: Metric names (column labels)
            title: Plot title
            figsize: Figure size
            cmap: Colormap name
            annot: Show values in cells
            fmt: Value format string
            output_path: Optional path to save figure

        Returns:
            Matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=figsize)

        if isinstance(data, pd.DataFrame):
            sns.heatmap(
                data,
                annot=annot,
                fmt=fmt,
                cmap=cmap,
                ax=ax,
                cbar_kws={"label": "Score"},
                vmin=0,
                vmax=1,
            )
        else:
            # Convert numpy array
            sns.heatmap(
                data,
                annot=annot,
                fmt=fmt,
                cmap=cmap,
                ax=ax,
                xticklabels=metrics or [],
                yticklabels=strategies or [],
                cbar_kws={"label": "Score"},
                vmin=0,
                vmax=1,
            )

        ax.set_title(title, fontsize=16, fontweight="bold")
        ax.set_xlabel("Metrics", fontsize=12)
        ax.set_ylabel("Strategies", fontsize=12)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            logger.info("heatmap_saved", path=str(output_path))

        return fig

    @staticmethod
    def plot_strategy_comparison(
        results_df: pd.DataFrame,
        metric: Optional[str] = None,
        title: Optional[str] = None,
        figsize: tuple = (12, 6),
        output_path: Optional[Union[str, Path]] = None,
    ) -> plt.Figure:
        """
        Create bar chart comparing strategies.

        Args:
            results_df: DataFrame with results
            metric: Specific metric to plot (None = weighted average)
            title: Plot title
            figsize: Figure size
            output_path: Optional save path

        Returns:
            Matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Prepare data
        if metric:
            col = f"metric_{metric}" if not metric.startswith("metric_") else metric
            if col not in results_df.columns:
                raise ValueError(f"Metric not found: {metric}")

            y_values = results_df[col]
            y_label = metric.replace("metric_", "").replace("_", " ").title()
            plot_title = title or f"Strategy Comparison: {y_label}"
        else:
            # Use weighted average if available
            if "weighted_score" in results_df.columns:
                y_values = results_df["weighted_score"]
                y_label = "Weighted Score"
            else:
                # Compute average of all metrics
                metric_cols = [col for col in results_df.columns if col.startswith("metric_")]
                y_values = results_df[metric_cols].mean(axis=1)
                y_label = "Average Score"

            plot_title = title or "Strategy Comparison: Overall Performance"

        # Create bar chart
        bars = ax.bar(
            range(len(results_df)),
            y_values,
            color=sns.color_palette("husl", len(results_df)),
            edgecolor="black",
            linewidth=1.5,
        )

        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, y_values)):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

        # Customize
        ax.set_xlabel("Strategy", fontsize=12, fontweight="bold")
        ax.set_ylabel(y_label, fontsize=12, fontweight="bold")
        ax.set_title(plot_title, fontsize=16, fontweight="bold")
        ax.set_xticks(range(len(results_df)))
        ax.set_xticklabels(results_df["strategy"], rotation=45, ha="right")
        ax.set_ylim(0, min(1.1, y_values.max() * 1.15))
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            logger.info("comparison_chart_saved", path=str(output_path))

        return fig

    @staticmethod
    def plot_radar_chart(
        results_df: pd.DataFrame,
        strategies: Optional[List[str]] = None,
        metrics: Optional[List[str]] = None,
        title: str = "Multi-Metric Strategy Comparison",
        figsize: tuple = (10, 10),
        output_path: Optional[Union[str, Path]] = None,
    ) -> plt.Figure:
        """
        Create radar/spider chart for multi-metric comparison.

        Args:
            results_df: DataFrame with results
            strategies: Strategy names to include (None = all)
            metrics: Metrics to include (None = all)
            title: Plot title
            figsize: Figure size
            output_path: Optional save path

        Returns:
            Matplotlib Figure
        """
        # Filter strategies
        if strategies:
            df = results_df[results_df["strategy"].isin(strategies)]
        else:
            df = results_df

        # Get metric columns
        if metrics:
            metric_cols = [f"metric_{m}" if not m.startswith("metric_") else m for m in metrics]
        else:
            metric_cols = [col for col in df.columns if col.startswith("metric_")]

        if not metric_cols:
            raise ValueError("No metrics found for radar chart")

        # Number of metrics
        num_metrics = len(metric_cols)
        angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()
        angles += angles[:1]  # Close the circle

        # Create figure
        fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection="polar"))

        # Plot each strategy
        colors = sns.color_palette("husl", len(df))

        for idx, (_, row) in enumerate(df.iterrows()):
            values = row[metric_cols].tolist()
            values += values[:1]  # Close the polygon

            ax.plot(angles, values, "o-", linewidth=2, label=row["strategy"], color=colors[idx])
            ax.fill(angles, values, alpha=0.15, color=colors[idx])

        # Customize
        ax.set_xticks(angles[:-1])
        metric_labels = [col.replace("metric_", "").replace("_", " ").title() for col in metric_cols]
        ax.set_xticklabels(metric_labels, fontsize=10)
        ax.set_ylim(0, 1)
        ax.set_title(title, fontsize=16, fontweight="bold", pad=20)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            logger.info("radar_chart_saved", path=str(output_path))

        return fig

    @staticmethod
    def plot_metric_distribution(
        results_df: pd.DataFrame,
        metrics: Optional[List[str]] = None,
        title: str = "Metric Distribution Across Strategies",
        figsize: tuple = (14, 6),
        output_path: Optional[Union[str, Path]] = None,
    ) -> plt.Figure:
        """
        Create box plots showing metric distribution.

        Args:
            results_df: DataFrame with results
            metrics: Metrics to plot (None = all)
            title: Plot title
            figsize: Figure size
            output_path: Optional save path

        Returns:
            Matplotlib Figure
        """
        # Get metric columns
        if metrics:
            metric_cols = [f"metric_{m}" if not m.startswith("metric_") else m for m in metrics]
        else:
            metric_cols = [col for col in results_df.columns if col.startswith("metric_")]

        if not metric_cols:
            raise ValueError("No metrics found")

        # Prepare data for plotting
        data_to_plot = []
        labels = []

        for col in metric_cols:
            data_to_plot.append(results_df[col].values)
            labels.append(col.replace("metric_", "").replace("_", " ").title())

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        bp = ax.boxplot(
            data_to_plot,
            labels=labels,
            patch_artist=True,
            notch=True,
            showmeans=True,
        )

        # Color boxes
        colors = sns.color_palette("husl", len(metric_cols))
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        # Customize
        ax.set_xlabel("Metrics", fontsize=12, fontweight="bold")
        ax.set_ylabel("Score", fontsize=12, fontweight="bold")
        ax.set_title(title, fontsize=16, fontweight="bold")
        ax.set_ylim(0, 1.1)
        ax.grid(axis="y", alpha=0.3)
        plt.xticks(rotation=45, ha="right")

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            logger.info("distribution_plot_saved", path=str(output_path))

        return fig

    @staticmethod
    def plot_correlation_matrix(
        results_df: pd.DataFrame,
        title: str = "Metric Correlation Matrix",
        figsize: tuple = (10, 8),
        annot: bool = True,
        output_path: Optional[Union[str, Path]] = None,
    ) -> plt.Figure:
        """
        Create correlation heatmap between metrics.

        Args:
            results_df: DataFrame with results
            title: Plot title
            figsize: Figure size
            annot: Show correlation values
            output_path: Optional save path

        Returns:
            Matplotlib Figure
        """
        metric_cols = [col for col in results_df.columns if col.startswith("metric_")]

        if len(metric_cols) < 2:
            raise ValueError("Need at least 2 metrics for correlation matrix")

        # Compute correlation
        corr = results_df[metric_cols].corr()

        # Rename columns for readability
        corr.columns = [col.replace("metric_", "").replace("_", " ").title() for col in corr.columns]
        corr.index = corr.columns

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        sns.heatmap(
            corr,
            annot=annot,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
            square=True,
            linewidths=1,
            cbar_kws={"label": "Correlation"},
            ax=ax,
        )

        ax.set_title(title, fontsize=16, fontweight="bold")

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            logger.info("correlation_matrix_saved", path=str(output_path))

        return fig

    @staticmethod
    def plot_performance_vs_cost(
        results_df: pd.DataFrame,
        performance_metric: str = "ndcg_at_k",
        cost_metric: str = "processing_time_ms",
        title: Optional[str] = None,
        figsize: tuple = (10, 6),
        output_path: Optional[Union[str, Path]] = None,
    ) -> plt.Figure:
        """
        Create scatter plot of performance vs cost trade-off.

        Args:
            results_df: DataFrame with results
            performance_metric: Metric for y-axis
            cost_metric: Metric for x-axis (processing time, tokens, etc.)
            title: Plot title
            figsize: Figure size
            output_path: Optional save path

        Returns:
            Matplotlib Figure
        """
        perf_col = (
            f"metric_{performance_metric}"
            if not performance_metric.startswith("metric_")
            else performance_metric
        )
        cost_col = cost_metric

        if perf_col not in results_df.columns:
            raise ValueError(f"Performance metric not found: {performance_metric}")
        if cost_col not in results_df.columns:
            raise ValueError(f"Cost metric not found: {cost_metric}")

        fig, ax = plt.subplots(figsize=figsize)

        # Create scatter plot
        scatter = ax.scatter(
            results_df[cost_col],
            results_df[perf_col],
            s=200,
            c=range(len(results_df)),
            cmap="viridis",
            alpha=0.7,
            edgecolors="black",
            linewidth=1.5,
        )

        # Add labels for each point
        for _, row in results_df.iterrows():
            ax.annotate(
                row["strategy"],
                (row[cost_col], row[perf_col]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=10,
                fontweight="bold",
            )

        # Customize
        plot_title = title or f"{performance_metric.replace('_', ' ').title()} vs {cost_metric.replace('_', ' ').title()}"
        ax.set_xlabel(cost_metric.replace("_", " ").title(), fontsize=12, fontweight="bold")
        ax.set_ylabel(performance_metric.replace("_", " ").title(), fontsize=12, fontweight="bold")
        ax.set_title(plot_title, fontsize=16, fontweight="bold")
        ax.grid(alpha=0.3)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            logger.info("scatter_plot_saved", path=str(output_path))

        return fig

    @staticmethod
    def create_comparison_dashboard(
        results_df: pd.DataFrame,
        output_dir: Union[str, Path],
        prefix: str = "strategy_comparison",
    ) -> Dict[str, Path]:
        """
        Create comprehensive dashboard with multiple visualizations.

        Args:
            results_df: DataFrame with results
            output_dir: Output directory for plots
            prefix: Filename prefix

        Returns:
            Dict mapping plot names to file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        saved_plots = {}

        # 1. Heatmap
        metric_cols = [col for col in results_df.columns if col.startswith("metric_")]
        if metric_cols:
            heatmap_data = results_df.set_index("strategy")[metric_cols]
            heatmap_data.columns = [col.replace("metric_", "") for col in heatmap_data.columns]

            path = output_dir / f"{prefix}_heatmap.png"
            StrategyVisualizer.plot_heatmap(heatmap_data, title="Strategy Performance Heatmap", output_path=path)
            saved_plots["heatmap"] = path
            plt.close()

        # 2. Bar chart
        path = output_dir / f"{prefix}_comparison.png"
        StrategyVisualizer.plot_strategy_comparison(results_df, output_path=path)
        saved_plots["comparison"] = path
        plt.close()

        # 3. Radar chart (if <=5 strategies)
        if len(results_df) <= 5 and len(metric_cols) >= 3:
            path = output_dir / f"{prefix}_radar.png"
            StrategyVisualizer.plot_radar_chart(results_df, output_path=path)
            saved_plots["radar"] = path
            plt.close()

        # 4. Distribution
        if len(metric_cols) >= 2:
            path = output_dir / f"{prefix}_distribution.png"
            StrategyVisualizer.plot_metric_distribution(results_df, output_path=path)
            saved_plots["distribution"] = path
            plt.close()

        # 5. Correlation matrix
        if len(metric_cols) >= 2:
            path = output_dir / f"{prefix}_correlation.png"
            StrategyVisualizer.plot_correlation_matrix(results_df, output_path=path)
            saved_plots["correlation"] = path
            plt.close()

        logger.info("dashboard_created", num_plots=len(saved_plots), output_dir=str(output_dir))

        return saved_plots
