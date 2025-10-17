"""Complete example: Chunking + Embeddings."""

import asyncio
from typing import List

from chunk_flow.chunking import RecursiveCharacterChunker, StrategyRegistry
from chunk_flow.embeddings import EmbeddingProviderFactory


async def main() -> None:
    """Demonstrate chunking with embeddings."""

    # Sample document
    document = """
    # Understanding Retrieval-Augmented Generation (RAG)

    RAG systems combine retrieval with generation to produce accurate,
    contextually relevant responses.

    ## How RAG Works

    1. **Document Processing**: Text is split into chunks
    2. **Embedding**: Chunks are converted to vector representations
    3. **Storage**: Vectors are stored in a vector database
    4. **Retrieval**: Relevant chunks are fetched based on query similarity
    5. **Generation**: LLM generates response using retrieved context

    ## Why Chunking Matters

    Proper chunking directly impacts:
    - Retrieval accuracy
    - Computational costs
    - Response quality
    - User experience

    Research shows that late chunking achieves 6-9% improvement in retrieval accuracy.
    """

    print("=" * 80)
    print("ChunkFlow: Chunking + Embeddings Example")
    print("=" * 80)

    # Step 1: Chunk the document
    print("\n1. Chunking document...")
    print("-" * 80)

    chunker = StrategyRegistry.create(
        "recursive",
        config={
            "chunk_size": 200,
            "overlap": 50,
            "separators": ["\n\n", "\n", ". "],
        },
    )

    chunk_result = await chunker.chunk(document, doc_id="rag_intro")

    print(f"✓ Created {len(chunk_result.chunks)} chunks")
    print(f"✓ Processing time: {chunk_result.processing_time_ms:.2f}ms")

    # Display chunks
    print("\nChunks:")
    for i, chunk in enumerate(chunk_result.chunks, 1):
        print(f"\nChunk {i} ({chunk_result.metadata[i-1].char_count} chars):")
        print(f"  {chunk[:100]}..." if len(chunk) > 100 else f"  {chunk}")

    # Step 2: Generate embeddings (choose provider)
    print("\n\n2. Generating embeddings...")
    print("-" * 80)

    # List available providers
    available_providers = EmbeddingProviderFactory.list_providers()
    print(f"Available providers: {', '.join(available_providers)}")

    # Try HuggingFace first (local, free), fallback to OpenAI if available
    provider_name = None
    if "huggingface" in available_providers:
        provider_name = "huggingface"
        provider_config = {
            "model": "sentence-transformers/all-MiniLM-L6-v2",
            "device": "cpu",
            "batch_size": 32,
            "normalize": True,
        }
        print("\n→ Using HuggingFace (local, free)")
    elif "openai" in available_providers:
        provider_name = "openai"
        provider_config = {
            "model": "text-embedding-3-small",
            # "api_key": "your-key-here"  # Or set CHUNK_FLOW_OPENAI_API_KEY
        }
        print("\n→ Using OpenAI (requires API key)")
    else:
        print("\n✗ No embedding providers available!")
        print("Install: pip install 'chunk-flow[huggingface]' or 'chunk-flow[openai]'")
        return

    if provider_name:
        try:
            embedder = EmbeddingProviderFactory.create(provider_name, provider_config)

            embedding_result = await embedder.embed_texts(chunk_result.chunks)

            print(f"\n✓ Generated embeddings for {len(embedding_result.embeddings)} chunks")
            print(f"✓ Dimensions: {embedding_result.dimensions}")
            print(f"✓ Processing time: {embedding_result.processing_time_ms:.2f}ms")
            print(f"✓ Total tokens: {embedding_result.token_count}")
            print(f"✓ Cost: ${embedding_result.cost_usd:.6f}" if embedding_result.cost_usd else "✓ Cost: Free (local)")

            # Step 3: Compute similarity between chunks
            print("\n\n3. Computing chunk similarities...")
            print("-" * 80)

            if "huggingface" in provider_name:
                # HuggingFace provider has similarity matrix method
                similarity_matrix = embedder.get_similarity_matrix(
                    embedding_result.embeddings
                )
                print("\nSimilarity matrix (showing top-left 3x3):")
                print(similarity_matrix[:3, :3])
            else:
                # Compute pairwise similarities manually
                from sklearn.metrics.pairwise import cosine_similarity
                import numpy as np

                emb_array = np.array(embedding_result.embeddings)
                similarities = cosine_similarity(emb_array)
                print("\nSimilarity matrix (showing top-left 3x3):")
                print(similarities[:3, :3])

            # Find most similar chunk pairs
            print("\nMost similar chunk pairs:")
            import numpy as np

            if "huggingface" in provider_name:
                sim_matrix = similarity_matrix
            else:
                sim_matrix = similarities

            # Get upper triangle indices (avoid diagonal and duplicates)
            triu_indices = np.triu_indices_from(sim_matrix, k=1)
            similarities_flat = sim_matrix[triu_indices]

            # Get top 3 similar pairs
            top_indices = np.argsort(similarities_flat)[-3:][::-1]

            for idx in top_indices:
                i, j = triu_indices[0][idx], triu_indices[1][idx]
                similarity = sim_matrix[i, j]
                print(f"\n  Chunk {i+1} ↔ Chunk {j+1}: {similarity:.3f}")
                print(f"    → Chunk {i+1}: {chunk_result.chunks[i][:60]}...")
                print(f"    → Chunk {j+1}: {chunk_result.chunks[j][:60]}...")

            # Step 4: Summary stats
            print("\n\n4. Summary Statistics:")
            print("-" * 80)

            print(f"\nChunking:")
            print(f"  Strategy: {chunk_result.strategy_version}")
            print(f"  Num chunks: {len(chunk_result.chunks)}")
            print(f"  Avg chunk size: {sum(m.char_count for m in chunk_result.metadata) / len(chunk_result.metadata):.1f} chars")
            print(f"  Processing time: {chunk_result.processing_time_ms:.2f}ms")

            print(f"\nEmbedding:")
            print(f"  Provider: {embedding_result.provider_name}")
            print(f"  Model: {embedding_result.model_name}")
            print(f"  Dimensions: {embedding_result.dimensions}")
            print(f"  Total tokens: {embedding_result.token_count}")
            print(f"  Processing time: {embedding_result.processing_time_ms:.2f}ms")
            print(f"  Cost: ${embedding_result.cost_usd:.6f}" if embedding_result.cost_usd else "  Cost: Free")

            print(f"\nTotal pipeline:")
            print(f"  End-to-end time: {chunk_result.processing_time_ms + embedding_result.processing_time_ms:.2f}ms")

        except Exception as e:
            print(f"\n✗ Error: {str(e)}")
            import traceback

            traceback.print_exc()

    print("\n" + "=" * 80)
    print("Example complete!")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
