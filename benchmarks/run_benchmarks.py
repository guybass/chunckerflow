"""Comprehensive benchmark suite for ChunkFlow."""

import asyncio
import time
from pathlib import Path
from typing import Dict, List

from chunk_flow.analysis import ResultsDataFrame, StrategyVisualizer
from chunk_flow.chunking import StrategyRegistry
from chunk_flow.evaluation import EvaluationPipeline


# Standard test documents
BENCHMARK_DOCUMENTS = {
    "short_technical": """
    # Neural Network Architecture
    
    A neural network consists of layers of interconnected nodes. Each layer
    transforms input data through weighted connections and activation functions.
    """,
    "medium_tutorial": """
    # Introduction to Machine Learning
    
    Machine learning enables computers to learn from data without explicit
    programming. This tutorial covers the fundamentals.
    
    ## Supervised Learning
    
    Supervised learning uses labeled data to train models. Common algorithms
    include linear regression, decision trees, and neural networks.
    
    ## Unsupervised Learning
    
    Unsupervised learning finds patterns in unlabeled data. Clustering and
    dimensionality reduction are key techniques.
    
    ## Applications
    
    Machine learning powers recommendation systems, fraud detection, image
    recognition, and natural language processing.
    """,
    "long_research": """
    # Deep Learning: A Comprehensive Overview
    
    Deep learning has revolutionized artificial intelligence, achieving
    breakthrough results across multiple domains.
    
    ## Neural Network Architectures
    
    Modern deep learning relies on sophisticated architectures. Convolutional
    neural networks excel at image processing, while recurrent networks handle
    sequential data.
    
    ### Convolutional Neural Networks
    
    CNNs use convolutional layers to detect features in images. Pooling layers
    reduce dimensionality while preserving important patterns.
    
    ### Recurrent Neural Networks
    
    RNNs process sequences by maintaining hidden states. LSTMs and GRUs address
    the vanishing gradient problem.
    
    ## Training Techniques
    
    Training deep networks requires careful optimization. Batch normalization,
    dropout, and learning rate scheduling improve convergence.
    
    ## Applications in Industry
    
    Deep learning powers autonomous vehicles, medical diagnosis, speech recognition,
    and natural language understanding. Transfer learning enables adapting
    pre-trained models to new tasks.
    
    ## Challenges and Future Directions
    
    Despite successes, deep learning faces challenges: interpretability,
    data efficiency, and computational requirements. Research continues on
    few-shot learning, neural architecture search, and efficient training.
    """,
}


async def run_strategy_benchmark(document: str, strategy_name: str, config: Dict) -> Dict:
    """Benchmark a single strategy."""
    strategy = StrategyRegistry.create(strategy_name, config)

    # Warm-up
    await strategy.chunk(document)

    # Timed run
    start = time.time()
    result = await strategy.chunk(document)
    elapsed = (time.time() - start) * 1000

    return {
        "strategy": strategy_name,
        "num_chunks": len(result.chunks),
        "processing_time_ms": elapsed,
        "avg_chunk_size": sum(len(c) for c in result.chunks) / len(result.chunks),
        "config": str(config),
    }


async def main():
    """Run comprehensive benchmarks."""
    print("=" * 80)
    print("ChunkFlow Benchmark Suite")
    print("=" * 80)

    output_dir = Path("benchmark_results")
    output_dir.mkdir(exist_ok=True)

    # Define strategy configurations to test
    strategies_to_test = [
        ("fixed_size", {"chunk_size": 200, "overlap": 0}),
        ("fixed_size", {"chunk_size": 200, "overlap": 50}),
        ("fixed_size", {"chunk_size": 500, "overlap": 100}),
        ("recursive", {"chunk_size": 200, "overlap": 50}),
        ("recursive", {"chunk_size": 500, "overlap": 100}),
        ("markdown", {"respect_headers": True}),
    ]

    # Try semantic if available
    try:
        StrategyRegistry.get("semantic")
        strategies_to_test.append(("semantic", {"threshold_percentile": 80}))
    except Exception:
        pass

    # Run benchmarks for each document
    all_results = []

    for doc_name, document in BENCHMARK_DOCUMENTS.items():
        print(f"\n{doc_name.upper()}")
        print("-" * 80)
        print(f"Document length: {len(document)} characters\n")

        for strategy_name, config in strategies_to_test:
            try:
                result = await run_strategy_benchmark(document, strategy_name, config)
                result["document"] = doc_name
                result["doc_length"] = len(document)
                all_results.append(result)

                print(
                    f"  {strategy_name:<15} | "
                    f"chunks: {result['num_chunks']:>3} | "
                    f"time: {result['processing_time_ms']:>6.2f}ms | "
                    f"avg size: {result['avg_chunk_size']:>5.1f}"
                )

            except Exception as e:
                print(f"  {strategy_name:<15} | FAILED: {str(e)[:50]}")

    # Create results dataframe
    print("\n" + "=" * 80)
    print("Generating Analysis...")
    print("=" * 80)

    import pandas as pd

    df = pd.DataFrame(all_results)

    # Save detailed results
    df.to_csv(output_dir / "benchmark_results.csv", index=False)
    print(f"\n✓ Saved detailed results: {output_dir / 'benchmark_results.csv'}")

    # Performance summary by strategy
    print("\nPerformance Summary by Strategy:")
    print("-" * 80)

    summary = df.groupby("strategy").agg({
        "processing_time_ms": ["mean", "std", "min", "max"],
        "num_chunks": "mean",
        "avg_chunk_size": "mean",
    }).round(2)

    print(summary.to_string())

    # Performance by document type
    print("\nPerformance by Document Type:")
    print("-" * 80)

    doc_summary = df.groupby("document").agg({
        "processing_time_ms": "mean",
        "num_chunks": "mean",
    }).round(2)

    print(doc_summary.to_string())

    # Find fastest strategies
    print("\nFastest Strategies:")
    print("-" * 80)

    fastest = df.nsmallest(5, "processing_time_ms")[
        ["strategy", "document", "processing_time_ms", "num_chunks"]
    ]
    print(fastest.to_string(index=False))

    # Throughput analysis (chars/sec)
    df["throughput"] = df["doc_length"] / (df["processing_time_ms"] / 1000)

    print("\nThroughput (chars/sec):")
    print("-" * 80)

    throughput = df.groupby("strategy")["throughput"].agg(["mean", "std"]).round(0)
    throughput = throughput.sort_values("mean", ascending=False)
    print(throughput.to_string())

    print("\n" + "=" * 80)
    print("✓ Benchmark Complete!")
    print(f"\nResults saved to: {output_dir}/")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
