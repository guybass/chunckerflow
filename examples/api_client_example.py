"""Example: Using the ChunkFlow API."""

import asyncio
from typing import Any, Dict

import httpx


BASE_URL = "http://localhost:8000"


async def test_api() -> None:
    """Test all API endpoints."""

    print("=" * 80)
    print("ChunkFlow API Client Example")
    print("=" * 80)
    print("\nMake sure the API server is running:")
    print("  uvicorn chunk_flow.api.app:app --reload")
    print("  or")
    print("  docker-compose up")
    print("=" * 80)

    async with httpx.AsyncClient(timeout=30.0) as client:
        # 1. Health check
        print("\n1. Health Check")
        print("-" * 80)

        response = await client.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            health = response.json()
            print(f"✓ Status: {health['status']}")
            print(f"✓ Version: {health['version']}")
            print(f"✓ Available strategies: {', '.join(health['available_strategies'])}")
            print(f"✓ Available metrics: {len(health['available_metrics'])} metrics")
            print(f"✓ Available providers: {', '.join(health['available_providers'])}")
        else:
            print(f"✗ Health check failed: {response.status_code}")
            return

        # 2. Chunk text
        print("\n2. Chunking Text")
        print("-" * 80)

        chunk_request = {
            "text": """
                Machine learning is a subset of artificial intelligence. It enables
                computers to learn from data without explicit programming.

                Deep learning uses neural networks with many layers. It has achieved
                breakthrough results in image recognition and natural language processing.

                Applications include autonomous vehicles, medical diagnosis, and
                recommendation systems.
            """,
            "strategy": "recursive",
            "config": {"chunk_size": 100, "overlap": 20},
        }

        response = await client.post(f"{BASE_URL}/chunk", json=chunk_request)
        if response.status_code == 200:
            chunk_result = response.json()
            print(f"✓ Strategy: {chunk_result['strategy']}")
            print(f"✓ Number of chunks: {chunk_result['num_chunks']}")
            print(f"✓ Processing time: {chunk_result['processing_time_ms']:.2f}ms")
            print(f"\nChunks:")
            for i, chunk in enumerate(chunk_result["chunks"][:3], 1):  # Show first 3
                print(f"  {i}. {chunk[:80]}...")

            # Save chunks for later use
            chunks = chunk_result["chunks"]
        else:
            print(f"✗ Chunking failed: {response.status_code} - {response.text}")
            return

        # 3. Generate embeddings
        print("\n3. Generating Embeddings")
        print("-" * 80)

        embed_request = {
            "texts": chunks,
            "provider": "huggingface",
            "config": {"model": "sentence-transformers/all-MiniLM-L6-v2"},
        }

        response = await client.post(f"{BASE_URL}/embed", json=embed_request)
        if response.status_code == 200:
            embed_result = response.json()
            print(f"✓ Provider: {embed_result['provider']}")
            print(f"✓ Model: {embed_result['model']}")
            print(f"✓ Dimensions: {embed_result['dimensions']}")
            print(f"✓ Processing time: {embed_result['processing_time_ms']:.2f}ms")

            if embed_result.get("cost_usd"):
                print(f"✓ Cost: ${embed_result['cost_usd']:.6f}")
            else:
                print(f"✓ Cost: Free (local)")

            # Save embeddings for evaluation
            embeddings = embed_result["embeddings"]
        else:
            print(f"✗ Embedding failed: {response.status_code} - {response.text}")
            return

        # 4. Evaluate chunks
        print("\n4. Evaluating Chunks")
        print("-" * 80)

        eval_request = {
            "chunks": chunks,
            "embeddings": embeddings,
            "metrics": ["semantic_coherence", "boundary_quality", "topic_diversity"],
        }

        response = await client.post(f"{BASE_URL}/evaluate", json=eval_request)
        if response.status_code == 200:
            eval_result = response.json()
            print(f"✓ Number of metrics: {eval_result['num_metrics']}")
            print(f"✓ Processing time: {eval_result['processing_time_ms']:.2f}ms")
            print(f"\nMetric Results:")
            for metric_name, metric_data in eval_result["results"].items():
                print(f"  {metric_name:<25} {metric_data['score']:.4f}")
        else:
            print(f"✗ Evaluation failed: {response.status_code} - {response.text}")

        # 5. Compare strategies
        print("\n5. Comparing Multiple Strategies")
        print("-" * 80)

        compare_request = {
            "text": chunk_request["text"],
            "strategies": [
                {"name": "fixed_size", "config": {"chunk_size": 100}},
                {"name": "recursive", "config": {"chunk_size": 100, "overlap": 20}},
                {"name": "markdown", "config": {}},
            ],
            "embedding_provider": "huggingface",
            "metrics": ["semantic_coherence", "boundary_quality"],
        }

        response = await client.post(f"{BASE_URL}/compare", json=compare_request)
        if response.status_code == 200:
            compare_result = response.json()
            print(f"✓ Best strategy: {compare_result['best_strategy']}")
            print(f"✓ Processing time: {compare_result['processing_time_ms']:.2f}ms")
            print(f"\nStrategy Results:")
            for strategy_name, strategy_data in compare_result["strategies"].items():
                print(f"\n  {strategy_name}:")
                print(f"    Num chunks: {strategy_data['num_chunks']}")
                print(f"    Metrics:")
                for metric_name, metric_data in strategy_data["metric_results"].items():
                    print(f"      {metric_name:<25} {metric_data['score']:.4f}")
        else:
            print(f"✗ Comparison failed: {response.status_code} - {response.text}")

        # 6. Discovery endpoints
        print("\n6. Discovery Endpoints")
        print("-" * 80)

        # List strategies
        response = await client.get(f"{BASE_URL}/strategies")
        if response.status_code == 200:
            strategies = response.json()["strategies"]
            print(f"✓ Available strategies ({len(strategies)}):")
            for strategy in strategies[:5]:  # Show first 5
                print(f"  - {strategy['name']} (v{strategy['version']})")

        # List metrics
        response = await client.get(f"{BASE_URL}/metrics")
        if response.status_code == 200:
            metrics = response.json()
            print(f"\n✓ Available metrics: {len(metrics['metrics'])}")
            for category, metric_list in metrics["by_category"].items():
                if metric_list:
                    print(f"  {category}: {', '.join(metric_list)}")

    print("\n" + "=" * 80)
    print("✓ API client example complete!")
    print("\nYou can also explore the API docs at:")
    print(f"  - Swagger UI: {BASE_URL}/docs")
    print(f"  - ReDoc: {BASE_URL}/redoc")
    print("=" * 80)


if __name__ == "__main__":
    try:
        asyncio.run(test_api())
    except httpx.ConnectError:
        print("\n✗ ERROR: Could not connect to API server!")
        print("Please start the server first:")
        print("  uvicorn chunk_flow.api.app:app --reload")
        print("  or")
        print("  docker-compose up")
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
