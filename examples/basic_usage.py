"""Basic usage example for ChunkFlow."""

import asyncio

from chunk_flow.chunking import RecursiveCharacterChunker, StrategyRegistry


async def main() -> None:
    """Demonstrate basic chunking."""

    # Sample text
    text = """
    # Introduction to ChunkFlow

    ChunkFlow is a production-grade text chunking framework for RAG systems.

    ## Why Chunking Matters

    Text chunking directly impacts retrieval accuracy, computational costs,
    and user experience in RAG systems.

    ### Key Features

    - Multiple chunking strategies
    - Pluggable embedding providers
    - Comprehensive evaluation metrics
    - Production-ready design

    ## Getting Started

    Installation is simple with pip:
    ```
    pip install chunk-flow
    ```

    Then you can start chunking your documents immediately.
    """

    print("=" * 80)
    print("ChunkFlow Basic Usage Example")
    print("=" * 80)

    # Example 1: Using StrategyRegistry
    print("\n1. Using StrategyRegistry to create chunker:")
    print("-" * 80)

    chunker = StrategyRegistry.create(
        "recursive", config={"chunk_size": 200, "overlap": 50}
    )

    result = await chunker.chunk(text, doc_id="demo_doc")

    print(f"Number of chunks: {result.chunks.__len__()}")
    print(f"Processing time: {result.processing_time_ms:.2f}ms")
    print(f"Strategy: {result.strategy_version}")

    print("\nChunks:")
    for i, chunk in enumerate(result.chunks, 1):
        print(f"\n--- Chunk {i} ({result.metadata[i-1].char_count} chars) ---")
        print(chunk[:100] + "..." if len(chunk) > 100 else chunk)

    # Example 2: Direct instantiation
    print("\n\n2. Direct strategy instantiation:")
    print("-" * 80)

    chunker2 = RecursiveCharacterChunker(
        config={"chunk_size": 150, "overlap": 30, "separators": ["\n\n", "\n", ". "]}
    )

    result2 = await chunker2.chunk(text)
    print(f"Number of chunks: {len(result2.chunks)}")

    # Example 3: List available strategies
    print("\n\n3. Available strategies:")
    print("-" * 80)

    strategies = StrategyRegistry.list_strategies()
    for strategy in strategies:
        print(f"\n{strategy.name} (v{strategy.version})")
        print(f"  Description: {strategy.description[:80]}...")
        print(f"  Default config: {strategy.default_config}")

    # Example 4: Chunk metadata
    print("\n\n4. Chunk metadata:")
    print("-" * 80)

    meta = result.metadata[0]
    print(f"Chunk ID: {meta.chunk_id}")
    print(f"Position: {meta.start_idx} - {meta.end_idx}")
    print(f"Token count: {meta.token_count}")
    print(f"Character count: {meta.char_count}")
    print(f"Strategy: {meta.strategy_name} v{meta.version}")

    print("\n" + "=" * 80)
    print("Example complete!")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
