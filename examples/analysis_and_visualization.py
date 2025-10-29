"""Example: Analysis and visualization of chunking strategy comparison."""

import asyncio
from pathlib import Path

from chunk_flow.analysis import ResultsDataFrame, StrategyVisualizer
from chunk_flow.chunking import StrategyRegistry
from chunk_flow.embeddings import EmbeddingProviderFactory
from chunk_flow.evaluation import EvaluationPipeline


async def main() -> None:
    """Demonstrate ResultsDataFrame and visualization capabilities."""

    # Sample document
    document = """
    # Understanding Neural Networks

    Neural networks are computational models inspired by biological neurons.
    They consist of interconnected nodes (neurons) organized in layers.

    ## Architecture

    A typical neural network has three types of layers: input, hidden, and output.
    Information flows from input through hidden layers to the output layer.
    Each connection has a weight that adjusts during training.

    ## Training Process

    Training involves forward propagation and backpropagation. Forward propagation
    computes predictions, while backpropagation adjusts weights to minimize error.

    ## Applications

    Neural networks power many modern applications: image recognition, natural
    language processing, speech synthesis, and autonomous systems. Deep learning,
    which uses many hidden layers, has achieved breakthrough results in these domains.

    ## Challenges

    Despite their power, neural networks face challenges: they require large datasets,
    significant computational resources, and can be difficult to interpret. Overfitting
    is a common problem when models memorize training data instead of learning patterns.
    """

    print("=" * 80)
    print("ChunkFlow: Analysis & Visualization Example")
    print("=" * 80)

    # Step 1: Chunk document with multiple strategies
    print("\n1. Chunking with multiple strategies...")
    print("-" * 80)

    strategies = [
        StrategyRegistry.create("fixed_size", {"chunk_size": 200, "overlap": 40}),
        StrategyRegistry.create("recursive", {"chunk_size": 250, "overlap": 50}),
        StrategyRegistry.create("markdown", {"respect_headers": True}),
    ]

    # Try semantic if available
    try:
        strategies.append(
            StrategyRegistry.create("semantic", {"threshold_percentile": 75})
        )
        print("✓ Created 4 strategies")
    except Exception:
        print("✓ Created 3 strategies (semantic not available)")

    chunk_results = {}
    for strategy in strategies:
        result = await strategy.chunk(document)
        chunk_results[strategy.NAME] = result
        print(f"  {strategy.NAME:<15} → {len(result.chunks)} chunks")

    # Step 2: Generate embeddings
    print("\n2. Generating embeddings...")
    print("-" * 80)

    try:
        embedder = EmbeddingProviderFactory.create("huggingface")
        print("✓ Using HuggingFace embeddings")
    except Exception:
        try:
            embedder = EmbeddingProviderFactory.create("openai")
            print("✓ Using OpenAI embeddings")
        except Exception:
            print("✗ No embedding provider available!")
            return

    embeddings_per_strategy = {}
    for strategy_name, chunk_result in chunk_results.items():
        emb_result = await embedder.embed_texts(chunk_result.chunks)
        embeddings_per_strategy[strategy_name] = emb_result.embeddings

    # Step 3: Evaluate with multiple metrics
    print("\n3. Evaluating strategies...")
    print("-" * 80)

    pipeline = EvaluationPipeline(
        metrics=[
            "semantic_coherence",
            "boundary_quality",
            "chunk_stickiness",
            "topic_diversity",
        ]
    )

    all_results = {}
    for strategy_name, chunk_result in chunk_results.items():
        embeddings = embeddings_per_strategy[strategy_name]

        results = await pipeline.evaluate(
            chunks=chunk_result.chunks,
            embeddings=embeddings,
        )

        all_results[strategy_name] = results
        print(f"\n  {strategy_name}:")
        for metric_name, metric_result in results.items():
            print(f"    {metric_name:<25} {metric_result.score:.4f}")

    # Step 4: Create ResultsDataFrame
    print("\n4. Creating ResultsDataFrame...")
    print("-" * 80)

    # Add metadata
    metadata = {
        strategy_name: {
            "num_chunks": len(chunk_results[strategy_name].chunks),
            "processing_time_ms": chunk_results[strategy_name].processing_time_ms,
        }
        for strategy_name in chunk_results.keys()
    }

    # Create dataframe
    results_df = ResultsDataFrame.from_evaluation_results(all_results, metadata)

    print(f"✓ Created DataFrame: {results_df}")
    print(f"\nDataFrame preview:")
    print(results_df.df.to_string())

    # Step 5: Analyze results
    print("\n5. Analyzing results...")
    print("-" * 80)

    # Get best strategy
    best = results_df.get_best(metric=None, n=1)  # Weighted average
    print(f"\nBest strategy (weighted avg): {best['strategy']}")
    print(f"  Weighted score: {best.get('weighted_score', 'N/A'):.4f}")

    # Get top 3
    top3 = results_df.get_best(n=3)
    print(f"\nTop 3 strategies:")
    for i, row in top3.iterrows():
        print(f"  {row['rank']}. {row['strategy']} (score: {row.get('weighted_score', 0):.4f})")

    # Summary statistics
    summary = results_df.get_summary()
    print(f"\nSummary statistics:")
    print(f"  Number of strategies: {summary['num_strategies']}")
    print(f"  Number of metrics: {summary['num_metrics']}")
    print(f"\n  Metric averages:")
    for metric, stats in summary["metrics"].items():
        print(f"    {metric:<25} mean={stats['mean']:.4f}, std={stats['std']:.4f}")

    # Correlation analysis
    print(f"\nMetric correlations:")
    corr_matrix = results_df.correlation_matrix()
    print(corr_matrix.to_string())

    # Step 6: Export results
    print("\n6. Exporting results...")
    print("-" * 80)

    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    # Export to CSV
    results_df.export(output_dir / "strategy_comparison.csv", format="csv")
    print(f"✓ Exported to CSV: {output_dir / 'strategy_comparison.csv'}")

    # Export to JSON
    results_df.export(output_dir / "strategy_comparison.json", format="json")
    print(f"✓ Exported to JSON: {output_dir / 'strategy_comparison.json'}")

    # Export to Parquet (efficient for large datasets)
    results_df.export(output_dir / "strategy_comparison.parquet", format="parquet")
    print(f"✓ Exported to Parquet: {output_dir / 'strategy_comparison.parquet'}")

    # Step 7: Create visualizations
    print("\n7. Creating visualizations...")
    print("-" * 80)

    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(exist_ok=True)

    # Heatmap
    print("  Creating heatmap...")
    metric_cols = results_df.get_metric_columns()
    heatmap_data = results_df.df.set_index("strategy")[metric_cols]
    heatmap_data.columns = [col.replace("metric_", "") for col in heatmap_data.columns]

    StrategyVisualizer.plot_heatmap(
        heatmap_data,
        title="Strategy Performance Heatmap",
        output_path=viz_dir / "heatmap.png",
    )
    print(f"  ✓ Saved: {viz_dir / 'heatmap.png'}")

    # Bar chart
    print("  Creating comparison chart...")
    StrategyVisualizer.plot_strategy_comparison(
        results_df.df,
        metric=None,  # Overall performance
        output_path=viz_dir / "comparison.png",
    )
    print(f"  ✓ Saved: {viz_dir / 'comparison.png'}")

    # Radar chart (if <= 5 strategies)
    if len(results_df) <= 5:
        print("  Creating radar chart...")
        StrategyVisualizer.plot_radar_chart(
            results_df.df,
            title="Multi-Metric Performance",
            output_path=viz_dir / "radar.png",
        )
        print(f"  ✓ Saved: {viz_dir / 'radar.png'}")

    # Distribution plot
    print("  Creating distribution plot...")
    StrategyVisualizer.plot_metric_distribution(
        results_df.df,
        title="Metric Distribution",
        output_path=viz_dir / "distribution.png",
    )
    print(f"  ✓ Saved: {viz_dir / 'distribution.png'}")

    # Correlation matrix
    print("  Creating correlation matrix...")
    StrategyVisualizer.plot_correlation_matrix(
        results_df.df,
        title="Metric Correlations",
        output_path=viz_dir / "correlation.png",
    )
    print(f"  ✓ Saved: {viz_dir / 'correlation.png'}")

    # Performance vs cost
    if "processing_time_ms" in results_df.df.columns:
        print("  Creating performance vs cost plot...")
        StrategyVisualizer.plot_performance_vs_cost(
            results_df.df,
            performance_metric="semantic_coherence",
            cost_metric="processing_time_ms",
            output_path=viz_dir / "performance_vs_cost.png",
        )
        print(f"  ✓ Saved: {viz_dir / 'performance_vs_cost.png'}")

    # Step 8: Create complete dashboard
    print("\n8. Creating comprehensive dashboard...")
    print("-" * 80)

    dashboard_dir = output_dir / "dashboard"
    saved_plots = StrategyVisualizer.create_comparison_dashboard(
        results_df.df,
        output_dir=dashboard_dir,
        prefix="neural_networks",
    )

    print(f"✓ Created dashboard with {len(saved_plots)} plots:")
    for plot_name, plot_path in saved_plots.items():
        print(f"  - {plot_name}: {plot_path}")

    # Step 9: Advanced filtering and analysis
    print("\n9. Advanced filtering...")
    print("-" * 80)

    # Filter strategies with good coherence
    high_coherence = results_df.filter_strategies(metric_semantic_coherence__gt=0.7)
    print(f"\nStrategies with semantic coherence > 0.7:")
    if len(high_coherence) > 0:
        for _, row in high_coherence.df.iterrows():
            print(f"  - {row['strategy']}: {row['metric_semantic_coherence']:.4f}")
    else:
        print("  None found")

    # Filter by processing time
    fast_strategies = results_df.filter_strategies(processing_time_ms__lt=100)
    print(f"\nFast strategies (< 100ms):")
    if len(fast_strategies) > 0:
        for _, row in fast_strategies.df.iterrows():
            print(f"  - {row['strategy']}: {row['processing_time_ms']:.2f}ms")
    else:
        print("  None found (all strategies were fast!)")

    print("\n" + "=" * 80)
    print("✓ Example complete!")
    print("\nKey Outputs:")
    print(f"  - CSV export: {output_dir / 'strategy_comparison.csv'}")
    print(f"  - Visualizations: {viz_dir}/")
    print(f"  - Dashboard: {dashboard_dir}/")
    print("\nNext Steps:")
    print("  1. Open the visualizations to compare strategies")
    print("  2. Load the CSV in Excel/pandas for deeper analysis")
    print("  3. Use ResultsDataFrame for custom queries")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
