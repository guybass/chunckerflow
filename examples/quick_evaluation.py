"""Quick evaluation example - Simple workflow for testing."""

import asyncio

from chunk_flow.chunking import StrategyRegistry
from chunk_flow.embeddings import EmbeddingProviderFactory
from chunk_flow.evaluation import EvaluationPipeline


async def main() -> None:
    """Quick evaluation of a single chunking strategy."""

    # Sample text
    text = """
    Machine learning has revolutionized artificial intelligence. Deep learning
    models can now process images, text, and audio with unprecedented accuracy.

    Neural networks form the backbone of modern AI systems. They learn patterns
    from data through iterative training processes.

    Applications range from computer vision to natural language processing.
    Self-driving cars, voice assistants, and recommendation systems all rely
    on machine learning algorithms.
    """

    print("ChunkFlow: Quick Evaluation Example")
    print("=" * 60)

    # Step 1: Create strategy and chunk
    print("\n1. Chunking text...")
    chunker = StrategyRegistry.create("recursive", {"chunk_size": 150, "overlap": 30})
    chunk_result = await chunker.chunk(text)
    print(f"✓ Created {len(chunk_result.chunks)} chunks")

    # Display chunks
    for i, chunk in enumerate(chunk_result.chunks, 1):
        print(f"\n  Chunk {i}: {chunk[:80]}...")

    # Step 2: Generate embeddings
    print("\n2. Generating embeddings...")
    try:
        embedder = EmbeddingProviderFactory.create(
            "huggingface",
            {"model": "sentence-transformers/all-MiniLM-L6-v2"},
        )
    except Exception:
        print("✗ HuggingFace not available, trying OpenAI...")
        try:
            embedder = EmbeddingProviderFactory.create("openai")
        except Exception:
            print("✗ No embedding provider available!")
            print("Install: pip install 'chunk-flow[huggingface]'")
            return

    emb_result = await embedder.embed_texts(chunk_result.chunks)
    print(f"✓ Generated {len(emb_result.embeddings)} embeddings ({emb_result.dimensions}D)")

    # Step 3: Evaluate with semantic metrics (no ground truth needed)
    print("\n3. Evaluating chunks...")
    pipeline = EvaluationPipeline(
        metrics=["semantic_coherence", "boundary_quality", "chunk_stickiness"],
    )

    results = await pipeline.evaluate(
        chunks=chunk_result.chunks,
        embeddings=emb_result.embeddings,
    )

    print("\nResults:")
    for metric_name, metric_result in results.items():
        print(f"  {metric_name:<25} {metric_result.score:.4f}")
        if "details" in metric_result.details:
            print(f"    Details: {metric_result.details}")

    # Summary
    summary = pipeline.get_summary(results)
    print(f"\nAverage Score: {summary['average_score']:.4f}")

    print("\n" + "=" * 60)
    print("✓ Evaluation complete!")


if __name__ == "__main__":
    asyncio.run(main())
