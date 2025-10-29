"""Tests for chunking strategies."""

import pytest

from chunk_flow.chunking import StrategyRegistry
from chunk_flow.core.exceptions import RegistryError


class TestFixedSizeChunker:
    """Tests for FixedSizeChunker."""

    @pytest.mark.asyncio
    async def test_basic_chunking(self, sample_text):
        """Test basic fixed-size chunking."""
        chunker = StrategyRegistry.create("fixed_size", {"chunk_size": 100, "overlap": 20})
        result = await chunker.chunk(sample_text)

        assert len(result.chunks) > 0
        assert all(isinstance(chunk, str) for chunk in result.chunks)
        assert result.processing_time_ms >= 0
        assert len(result.metadata) == len(result.chunks)

    @pytest.mark.asyncio
    async def test_chunk_size_respected(self, sample_text):
        """Test that chunk size is approximately respected."""
        chunker = StrategyRegistry.create("fixed_size", {"chunk_size": 50})
        result = await chunker.chunk(sample_text)

        # Most chunks should be around the specified size (±10%)
        for chunk in result.chunks:
            assert len(chunk) <= 60  # Allow some flexibility

    @pytest.mark.asyncio
    async def test_overlap(self, sample_text):
        """Test overlap functionality."""
        chunker = StrategyRegistry.create("fixed_size", {"chunk_size": 100, "overlap": 20})
        result = await chunker.chunk(sample_text)

        # With overlap, should have more chunks than without
        chunker_no_overlap = StrategyRegistry.create("fixed_size", {"chunk_size": 100, "overlap": 0})
        result_no_overlap = await chunker_no_overlap.chunk(sample_text)

        assert len(result.chunks) >= len(result_no_overlap.chunks)


class TestRecursiveCharacterChunker:
    """Tests for RecursiveCharacterChunker."""

    @pytest.mark.asyncio
    async def test_basic_chunking(self, sample_text):
        """Test basic recursive chunking."""
        chunker = StrategyRegistry.create("recursive", {"chunk_size": 200, "overlap": 50})
        result = await chunker.chunk(sample_text)

        assert len(result.chunks) > 0
        assert all(isinstance(chunk, str) for chunk in result.chunks)

    @pytest.mark.asyncio
    async def test_separator_hierarchy(self):
        """Test that separators are respected in order."""
        text = "Paragraph 1.\n\nParagraph 2.\n\nParagraph 3."
        chunker = StrategyRegistry.create("recursive", {"chunk_size": 50, "separators": ["\n\n", "\n", ". "]})
        result = await chunker.chunk(text)

        # Should split on double newlines first
        assert len(result.chunks) > 0


class TestMarkdownChunker:
    """Tests for MarkdownChunker."""

    @pytest.mark.asyncio
    async def test_header_detection(self, sample_text):
        """Test that markdown headers are detected."""
        chunker = StrategyRegistry.create("markdown", {"respect_headers": True})
        result = await chunker.chunk(sample_text)

        assert len(result.chunks) > 0
        # Should have chunks starting with headers
        header_chunks = [c for c in result.chunks if c.strip().startswith("#")]
        assert len(header_chunks) > 0


class TestStrategyRegistry:
    """Tests for StrategyRegistry."""

    def test_list_strategies(self):
        """Test listing available strategies."""
        strategies = StrategyRegistry.get_strategy_names()
        assert "fixed_size" in strategies
        assert "recursive" in strategies
        assert "markdown" in strategies

    def test_get_strategy(self):
        """Test getting strategy by name."""
        strategy_class = StrategyRegistry.get("fixed_size")
        assert strategy_class is not None

    def test_unknown_strategy(self):
        """Test that unknown strategy raises error."""
        with pytest.raises(RegistryError):
            StrategyRegistry.get("nonexistent_strategy")

    def test_create_strategy(self):
        """Test creating strategy instance."""
        strategy = StrategyRegistry.create("fixed_size", {"chunk_size": 100})
        assert strategy is not None
        assert strategy.NAME == "fixed_size"

    def test_is_registered(self):
        """Test checking if strategy is registered."""
        assert StrategyRegistry.is_registered("fixed_size")
        assert not StrategyRegistry.is_registered("nonexistent")
