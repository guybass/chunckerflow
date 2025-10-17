"""Tests for ResultsDataFrame."""

import pytest
import pandas as pd
from pathlib import Path

from chunk_flow.analysis import ResultsDataFrame


@pytest.fixture
def sample_results_data():
    """Sample results data for testing."""
    return pd.DataFrame([
        {
            "strategy": "fixed_size",
            "metric_semantic_coherence": 0.75,
            "metric_boundary_quality": 0.80,
            "num_chunks": 10,
            "processing_time_ms": 50.0,
        },
        {
            "strategy": "recursive",
            "metric_semantic_coherence": 0.85,
            "metric_boundary_quality": 0.75,
            "num_chunks": 8,
            "processing_time_ms": 45.0,
        },
        {
            "strategy": "markdown",
            "metric_semantic_coherence": 0.80,
            "metric_boundary_quality": 0.90,
            "num_chunks": 6,
            "processing_time_ms": 40.0,
        },
    ])


class TestResultsDataFrame:
    """Tests for ResultsDataFrame."""

    def test_initialization(self, sample_results_data):
        """Test DataFrame initialization."""
        df = ResultsDataFrame(sample_results_data)
        assert len(df) == 3
        assert df.df is not None

    def test_get_metric_columns(self, sample_results_data):
        """Test getting metric columns."""
        df = ResultsDataFrame(sample_results_data)
        metrics = df.get_metric_columns()

        assert "metric_semantic_coherence" in metrics
        assert "metric_boundary_quality" in metrics
        assert "num_chunks" not in metrics  # Not a metric

    def test_rank_strategies(self, sample_results_data):
        """Test ranking strategies."""
        df = ResultsDataFrame(sample_results_data)
        ranked = df.rank_strategies()

        assert len(ranked.df) == 3
        assert "rank" in ranked.df.columns
        assert ranked.df.iloc[0]["rank"] == 1

    def test_get_best(self, sample_results_data):
        """Test getting best strategy."""
        df = ResultsDataFrame(sample_results_data)
        best = df.get_best(n=1)

        assert isinstance(best, pd.Series)
        assert best["rank"] == 1

    def test_get_best_multiple(self, sample_results_data):
        """Test getting top N strategies."""
        df = ResultsDataFrame(sample_results_data)
        top2 = df.get_best(n=2)

        assert isinstance(top2, pd.DataFrame)
        assert len(top2) == 2

    def test_filter_strategies(self, sample_results_data):
        """Test filtering strategies."""
        df = ResultsDataFrame(sample_results_data)

        # Filter by metric
        filtered = df.filter_strategies(metric_semantic_coherence__gt=0.80)
        assert len(filtered.df) <= 3

        # Filter by processing time
        fast = df.filter_strategies(processing_time_ms__lt=50)
        assert len(fast.df) >= 1

    def test_describe(self, sample_results_data):
        """Test statistical description."""
        df = ResultsDataFrame(sample_results_data)
        desc = df.describe()

        assert isinstance(desc, pd.DataFrame)
        assert "mean" in desc.index
        assert "std" in desc.index

    def test_correlation_matrix(self, sample_results_data):
        """Test correlation matrix."""
        df = ResultsDataFrame(sample_results_data)
        corr = df.correlation_matrix()

        assert isinstance(corr, pd.DataFrame)
        assert corr.shape[0] == corr.shape[1]  # Square matrix

    def test_export_csv(self, sample_results_data, tmp_path):
        """Test exporting to CSV."""
        df = ResultsDataFrame(sample_results_data)
        output_path = tmp_path / "test.csv"

        df.export(output_path, format="csv")
        assert output_path.exists()

        # Load and verify
        loaded_df = pd.read_csv(output_path)
        assert len(loaded_df) == 3

    def test_export_json(self, sample_results_data, tmp_path):
        """Test exporting to JSON."""
        df = ResultsDataFrame(sample_results_data)
        output_path = tmp_path / "test.json"

        df.export(output_path, format="json")
        assert output_path.exists()

    def test_load_csv(self, sample_results_data, tmp_path):
        """Test loading from CSV."""
        df = ResultsDataFrame(sample_results_data)
        output_path = tmp_path / "test.csv"

        df.export(output_path, format="csv")
        loaded = ResultsDataFrame.load(output_path, format="csv")

        assert len(loaded) == len(df)

    def test_get_summary(self, sample_results_data):
        """Test getting summary statistics."""
        df = ResultsDataFrame(sample_results_data)
        summary = df.get_summary()

        assert "num_strategies" in summary
        assert "num_metrics" in summary
        assert "metrics" in summary
        assert summary["num_strategies"] == 3
