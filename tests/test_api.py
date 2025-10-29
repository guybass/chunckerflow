"""Tests for FastAPI application."""

import pytest
from httpx import AsyncClient, ASGITransport

from chunk_flow.api.app import app


@pytest.mark.asyncio
class TestAPIEndpoints:
    """Tests for API endpoints."""

    async def test_health_check(self):
        """Test health check endpoint."""
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "available_strategies" in data

    async def test_chunk_endpoint(self, sample_text):
        """Test chunking endpoint."""
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/chunk",
                json={
                    "text": sample_text,
                    "strategy": "fixed_size",
                    "config": {"chunk_size": 100},
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert "chunks" in data
        assert "num_chunks" in data
        assert len(data["chunks"]) > 0

    async def test_chunk_invalid_strategy(self, sample_text):
        """Test chunking with invalid strategy."""
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/chunk",
                json={
                    "text": sample_text,
                    "strategy": "nonexistent",
                },
            )

        assert response.status_code == 400

    async def test_evaluate_endpoint(self, sample_chunks, sample_embeddings):
        """Test evaluation endpoint."""
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/evaluate",
                json={
                    "chunks": sample_chunks,
                    "embeddings": sample_embeddings,
                    "metrics": ["semantic_coherence"],
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert "num_metrics" in data

    async def test_list_strategies(self):
        """Test listing strategies endpoint."""
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/strategies")

        assert response.status_code == 200
        data = response.json()
        assert "strategies" in data
        assert len(data["strategies"]) > 0

    async def test_list_metrics(self):
        """Test listing metrics endpoint."""
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/metrics")

        assert response.status_code == 200
        data = response.json()
        assert "metrics" in data
        assert "by_category" in data

    async def test_list_providers(self):
        """Test listing providers endpoint."""
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/providers")

        assert response.status_code == 200
        data = response.json()
        assert "providers" in data
