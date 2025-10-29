"""Comprehensive example: Strategy comparison with evaluation pipeline."""

import asyncio
from typing import Dict, List

import numpy as np

from chunk_flow.chunking import StrategyRegistry
from chunk_flow.embeddings import EmbeddingProviderFactory
from chunk_flow.evaluation import EvaluationPipeline, StrategyComparator, MetricRegistry


async def main() -> None:
    """Demonstrate comprehensive strategy evaluation and comparison."""

    # Sample research document about RAG chunking
    document = """
    # The Critical Role of Chunking in Retrieval-Augmented Generation

    Retrieval-Augmented Generation (RAG) systems have emerged as a powerful paradigm
    for combining the knowledge retrieval capabilities of vector databases with the
    reasoning abilities of large language models. However, the effectiveness of RAG
    systems heavily depends on how documents are chunked before embedding and retrieval.

    ## Understanding Chunking Strategies

    Traditional chunking approaches include fixed-size chunking, which splits text
    into uniform character or token counts, and recursive character splitting, which
    respects natural document boundaries like paragraphs and sentences. While these
    methods are fast and straightforward, they often fail to preserve semantic coherence.

    ## The Problem with Semantic Chunking

    Semantic chunking attempts to group related content by analyzing embedding similarity
    between consecutive sentences. Research from the "Mismatch of Chunks" paper revealed
    a critical issue: chunk stickiness. When topics gradually transition, semantic
    chunking creates boundaries that misalign with actual topic changes, leading to
    reduced retrieval accuracy.

    ## Late Chunking: A Revolutionary Approach

    Jina AI researchers introduced late chunking in 2024, achieving 6-9% improvement
    in retrieval accuracy on the BeIR benchmark. The key insight: embed the entire
    document first (leveraging 8K+ context windows), then derive chunk embeddings
    while preserving full context. This approach achieves the speed of a single model
    pass with superior contextual awareness.

    ## Evaluation Best Practices

    Proper chunking evaluation requires multiple metrics: NDCG@k for ranking quality,
    recall@k for completeness, semantic coherence for topic consistency, and chunk
    stickiness to detect boundary problems. No single metric tells the whole story.

    ## Practical Recommendations

    For most applications, start with recursive character chunking (200-500 tokens,
    10-20% overlap). For long-form technical documents, consider late chunking if
    your embedding model supports 8K+ tokens. Always benchmark on your specific
    domain and query patterns before production deployment.
    """

    print("=" * 80)
    print("ChunkFlow: Comprehensive Strategy Comparison")
    print("=" * 80)

    # Step 1: List available strategies and metrics
    print("\n1. Discovering available components...")
    print("-" * 80)

    available_strategies = StrategyRegistry.get_strategy_names()
    print(f"✓ Available strategies: {', '.join(available_strategies)}")

    metrics_by_category = MetricRegistry.get_metrics_by_category()
    print(f"✓ Retrieval metrics: {', '.join(metrics_by_category['retrieval'])}")
    print(f"✓ Semantic metrics: {', '.join(metrics_by_category['semantic'])}")
    print(f"✓ RAG quality metrics: {', '.join(metrics_by_category['rag_quality'])}")

    # Step 2: Create multiple chunking strategies to compare
    print("\n2. Creating chunking strategies...")
    print("-" * 80)

    strategies = [
        StrategyRegistry.create("fixed_size", {"chunk_size": 500, "overlap": 50}),
        StrategyRegistry.create("recursive", {"chunk_size": 400, "overlap": 80}),
        StrategyRegistry.create("markdown", {"respect_headers": True}),
    ]

    # Try to add semantic if available
    try:
        strategies.append(
            StrategyRegistry.create(
                "semantic",
                {"threshold_percentile": 80, "min_chunk_size": 200, "max_chunk_size": 800},
            )
        )
        print("✓ Created 4 strategies (including semantic)")
    except Exception:
        print("✓ Created 3 strategies (semantic not available)")

    # Step 3: Chunk document with each strategy
    print("\n3. Chunking document with each strategy...")
    print("-" * 80)

    chunk_results = {}
    for strategy in strategies:
        result = await strategy.chunk(document, doc_id="rag_chunking_paper")
        chunk_results[strategy.NAME] = result
        print(
            f"  {strategy.NAME:<15} → {len(result.chunks)} chunks "
            f"({result.processing_time_ms:.2f}ms)"
        )

    # Step 4: Generate embeddings for all chunks
    print("\n4. Generating embeddings...")
    print("-" * 80)

    # Check for available embedding providers
    available_providers = EmbeddingProviderFactory.list_providers()

    if "huggingface" in available_providers:
        embedder = EmbeddingProviderFactory.create(
            "huggingface",
            {"model": "sentence-transformers/all-MiniLM-L6-v2", "normalize": True},
        )
        print("✓ Using HuggingFace embeddings (local)")
    elif "openai" in available_providers:
        embedder = EmbeddingProviderFactory.create(
            "openai",
            {"model": "text-embedding-3-small"},
        )
        print("✓ Using OpenAI embeddings")
    else:
        print("✗ No embedding provider available!")
        print("Install: pip install 'chunk-flow[huggingface]' or 'chunk-flow[openai]'")
        return

    # Generate embeddings for each strategy's chunks
    embeddings_per_strategy: Dict[str, List[List[float]]] = {}

    for strategy_name, chunk_result in chunk_results.items():
        emb_result = await embedder.embed_texts(chunk_result.chunks)
        embeddings_per_strategy[strategy_name] = emb_result.embeddings
        print(
            f"  {strategy_name:<15} → {len(emb_result.embeddings)} embeddings "
            f"({emb_result.dimensions}D, {emb_result.processing_time_ms:.2f}ms)"
        )

    # Step 5: Create ground truth for retrieval metrics
    print("\n5. Creating ground truth for evaluation...")
    print("-" * 80)

    # Simulate a query about "late chunking"
    query_text = "What is late chunking and how does it improve retrieval accuracy?"
    query_emb_result = await embedder.embed_texts([query_text])
    query_embedding = query_emb_result.embeddings[0]

    print(f"✓ Query: '{query_text}'")
    print(f"✓ Generated query embedding ({len(query_embedding)}D)")

    # For demonstration, identify relevant chunks (containing "late chunking" or "Jina AI")
    ground_truth_per_strategy = {}
    for strategy_name, chunk_result in chunk_results.items():
        relevant_indices = []
        for i, chunk in enumerate(chunk_result.chunks):
            if any(term in chunk.lower() for term in ["late chunking", "jina ai", "8k+ context"]):
                relevant_indices.append(i)

        ground_truth_per_strategy[strategy_name] = {
            "query_embedding": query_embedding,
            "relevant_indices": relevant_indices,
        }

        print(
            f"  {strategy_name:<15} → {len(relevant_indices)} relevant chunks "
            f"(out of {len(chunk_result.chunks)})"
        )

    # Step 6: Set up evaluation pipeline
    print("\n6. Setting up evaluation pipeline...")
    print("-" * 80)

    # Use semantic metrics only (no ground truth required)
    semantic_pipeline = EvaluationPipeline(
        metrics=["semantic_coherence", "boundary_quality", "chunk_stickiness", "topic_diversity"],
        max_concurrency=4,
    )

    print(f"✓ Created semantic evaluation pipeline with 4 metrics")

    # Use retrieval metrics (requires ground truth)
    retrieval_pipeline = EvaluationPipeline(
        metrics=["ndcg_at_k", "recall_at_k", "precision_at_k", "mrr"],
        metric_configs={
            "ndcg_at_k": {"k": 3},
            "recall_at_k": {"k": 5},
            "precision_at_k": {"k": 3},
        },
        max_concurrency=4,
    )

    print(f"✓ Created retrieval evaluation pipeline with 4 metrics")

    # Step 7: Evaluate all strategies with semantic metrics
    print("\n7. Evaluating strategies (semantic metrics)...")
    print("-" * 80)

    semantic_results = {}
    for strategy_name, chunk_result in chunk_results.items():
        embeddings = embeddings_per_strategy[strategy_name]
        results = await semantic_pipeline.evaluate(
            chunks=chunk_result.chunks,
            embeddings=embeddings,
        )
        semantic_results[strategy_name] = results

        print(f"\n  {strategy_name}:")
        for metric_name, metric_result in results.items():
            print(f"    {metric_name:<25} {metric_result.score:.4f}")

    # Step 8: Evaluate with retrieval metrics
    print("\n8. Evaluating strategies (retrieval metrics)...")
    print("-" * 80)

    retrieval_results = {}
    for strategy_name, chunk_result in chunk_results.items():
        embeddings = embeddings_per_strategy[strategy_name]
        ground_truth = ground_truth_per_strategy[strategy_name]

        results = await retrieval_pipeline.evaluate(
            chunks=chunk_result.chunks,
            embeddings=embeddings,
            ground_truth=ground_truth,
        )
        retrieval_results[strategy_name] = results

        print(f"\n  {strategy_name}:")
        for metric_name, metric_result in results.items():
            print(f"    {metric_name:<25} {metric_result.score:.4f}")

    # Step 9: Combine results and generate comparison report
    print("\n9. Generating comparison report...")
    print("-" * 80)

    # Combine semantic and retrieval results
    combined_results = {}
    for strategy_name in chunk_results.keys():
        combined_results[strategy_name] = {
            **semantic_results.get(strategy_name, {}),
            **retrieval_results.get(strategy_name, {}),
        }

    # Generate report
    report = StrategyComparator.generate_comparison_report(
        combined_results,
        metric_weights={
            # Retrieval metrics (higher weight)
            "ndcg_at_k": 2.0,
            "recall_at_k": 1.5,
            "mrr": 1.5,
            # Semantic metrics
            "semantic_coherence": 1.0,
            "boundary_quality": 1.0,
            "chunk_stickiness": 1.0,  # Lower is better
            "topic_diversity": 0.8,
        },
    )

    print(report)

    # Step 10: Additional analysis
    print("\n10. Additional Analysis:")
    print("-" * 80)

    # Best strategy per metric
    best_per_metric = StrategyComparator.identify_best_strategy_per_metric(combined_results)
    print("\nBest performers by metric:")
    for metric, info in best_per_metric.items():
        print(f"  {metric:<25} {info['strategy']:<15} (score: {info['score']:.4f})")

    # Win matrix
    win_data = StrategyComparator.compute_win_matrix(combined_results)
    print("\nWin percentages (how often each strategy beats others):")
    for item in win_data["win_percentages"]:
        print(
            f"  {item['strategy']:<15} "
            f"{item['wins']:>3} wins ({item['win_percentage']:>5.1f}%)"
        )

    # Statistical summary
    stats = StrategyComparator.compute_statistical_summary(combined_results)
    print("\nStrategy performance summary:")
    for strategy_name, strategy_stats in stats["strategies"].items():
        print(
            f"  {strategy_name:<15} "
            f"mean={strategy_stats['mean_score']:.4f} "
            f"std={strategy_stats['std_score']:.4f}"
        )

    print("\n" + "=" * 80)
    print("Comparison complete!")
    print("=" * 80)

    # Step 11: Show score matrix for visualization
    print("\n11. Score Matrix (for visualization):")
    print("-" * 80)

    matrix_data = StrategyComparator.get_score_matrix(combined_results)
    print(f"\nStrategies: {matrix_data['strategies']}")
    print(f"Metrics: {matrix_data['metrics']}")
    print("\nMatrix (strategies × metrics):")

    for i, strategy in enumerate(matrix_data["strategies"]):
        scores = [
            f"{score:.3f}" if score is not None else "  N/A"
            for score in matrix_data["matrix"][i]
        ]
        print(f"  {strategy:<15} {scores}")

    print("\n" + "=" * 80)
    print("✓ Example complete!")
    print("\nKey Insights:")
    print("  1. Different strategies excel at different metrics")
    print("  2. Recursive chunking often provides best overall balance")
    print("  3. Semantic chunking may have higher stickiness (topic bleeding)")
    print("  4. Markdown chunking leverages document structure")
    print("  5. Always evaluate on YOUR domain before production!")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
