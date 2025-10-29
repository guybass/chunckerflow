"""Tests for evaluation metrics."""

import pytest

from chunk_flow.evaluation import MetricRegistry
from chunk_flow.core.exceptions import ValidationError


class TestSemanticCoherenceMetric:
    """Tests for SemanticCoherenceMetric."""

    @pytest.mark.asyncio
    async def test_basic_computation(self, sample_chunks, sample_embeddings):
        """Test basic coherence computation."""
        metric = MetricRegistry.create("semantic_coherence")
        result = await metric.compute(
            chunks=sample_chunks,
            embeddings=sample_embeddings,
        )

        assert result.score >= 0.0
        assert result.score <= 1.0
        assert result.metric_name == "semantic_coherence"
        assert "min_coherence" in result.details
        assert "max_coherence" in result.details

    @pytest.mark.asyncio
    async def test_requires_embeddings(self, sample_chunks):
        """Test that metric requires embeddings."""
        metric = MetricRegistry.create("semantic_coherence")

        with pytest.raises(ValidationError):
            await metric.compute(chunks=sample_chunks, embeddings=None)


class TestBoundaryQualityMetric:
    """Tests for ChunkBoundaryQualityMetric."""

    @pytest.mark.asyncio
    async def test_basic_computation(self, sample_chunks, sample_embeddings):
        """Test basic boundary quality computation."""
        metric = MetricRegistry.create("boundary_quality")
        result = await metric.compute(
            chunks=sample_chunks,
            embeddings=sample_embeddings,
        )

        assert result.score >= 0.0
        assert result.score <= 1.0
        assert "num_boundaries" in result.details

    @pytest.mark.asyncio
    async def test_single_chunk(self, sample_embeddings):
        """Test with single chunk (no boundaries)."""
        metric = MetricRegistry.create("boundary_quality")
        result = await metric.compute(
            chunks=["single chunk"],
            embeddings=[sample_embeddings[0]],
        )

        assert result.score == 1.0
        assert result.details["num_boundaries"] == 0


class TestNDCGMetric:
    """Tests for NDCGMetric."""

    @pytest.mark.asyncio
    async def test_basic_computation(self, sample_chunks, sample_embeddings, sample_ground_truth):
        """Test basic NDCG computation."""
        metric = MetricRegistry.create("ndcg_at_k", {"k": 3})
        result = await metric.compute(
            chunks=sample_chunks,
            embeddings=sample_embeddings,
            ground_truth=sample_ground_truth,
        )

        assert result.score >= 0.0
        assert result.score <= 1.0
        assert result.metric_name == "ndcg_at_k"
        assert "k" in result.details
        assert result.details["k"] == 3

    @pytest.mark.asyncio
    async def test_requires_ground_truth(self, sample_chunks, sample_embeddings):
        """Test that NDCG requires ground truth."""
        from chunk_flow.core.exceptions import ValidationError

        metric = MetricRegistry.create("ndcg_at_k")

        # Should raise ValidationError when ground_truth is None
        with pytest.raises(ValidationError, match="requires ground truth"):
            await metric.compute(
                chunks=sample_chunks,
                embeddings=sample_embeddings,
                ground_truth=None,
            )


class TestMetricRegistry:
    """Tests for MetricRegistry."""

    def test_list_metrics(self):
        """Test listing available metrics."""
        metrics = MetricRegistry.get_metric_names()
        assert "semantic_coherence" in metrics
        assert "ndcg_at_k" in metrics
        assert "boundary_quality" in metrics

    def test_get_metrics_by_category(self):
        """Test getting metrics by category."""
        categories = MetricRegistry.get_metrics_by_category()
        assert "retrieval" in categories
        assert "semantic" in categories
        assert "rag_quality" in categories

        # Check specific metrics are in correct categories
        assert "ndcg_at_k" in categories["retrieval"]
        assert "semantic_coherence" in categories["semantic"]

    def test_create_metric(self):
        """Test creating metric instance."""
        metric = MetricRegistry.create("semantic_coherence")
        assert metric is not None
        assert metric.METRIC_NAME == "semantic_coherence"

    def test_is_registered(self):
        """Test checking if metric is registered."""
        assert MetricRegistry.is_registered("semantic_coherence")
        assert not MetricRegistry.is_registered("nonexistent_metric")
