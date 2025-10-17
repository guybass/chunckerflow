"""FastAPI application for ChunkFlow."""

import time
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from chunk_flow import __version__
from chunk_flow.api.models import (
    ChunkRequest,
    ChunkResponse,
    CompareRequest,
    CompareResponse,
    EmbedRequest,
    EmbedResponse,
    ErrorResponse,
    EvaluateRequest,
    EvaluateResponse,
    HealthResponse,
)
from chunk_flow.chunking import StrategyRegistry
from chunk_flow.embeddings import EmbeddingProviderFactory
from chunk_flow.evaluation import EvaluationPipeline, MetricRegistry
from chunk_flow.utils.logging import get_logger

logger = get_logger(__name__)

# Create FastAPI app
app = FastAPI(
    title="ChunkFlow API",
    description="Production-grade text chunking API for RAG systems",
    version=__version__,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request, exc: Exception):
    """Handle all exceptions globally."""
    logger.error("api_error", error=str(exc), exc_info=True)

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error=str(exc),
            error_type=type(exc).__name__,
        ).model_dump(),
    )


@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint redirect."""
    return {"message": "ChunkFlow API - see /docs for documentation"}


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    Health check endpoint.

    Returns system status and available components.
    """
    return HealthResponse(
        status="healthy",
        version=__version__,
        available_strategies=StrategyRegistry.get_strategy_names(),
        available_metrics=MetricRegistry.get_metric_names(),
        available_providers=EmbeddingProviderFactory.list_providers(),
    )


@app.post("/chunk", response_model=ChunkResponse, tags=["Chunking"])
async def chunk_text(request: ChunkRequest):
    """
    Chunk text using specified strategy.

    Args:
        request: ChunkRequest with text, strategy, and config

    Returns:
        ChunkResponse with chunks and metadata

    Raises:
        HTTPException: If strategy not found or chunking fails
    """
    start_time = time.time()

    try:
        # Create strategy
        strategy = StrategyRegistry.create(request.strategy, request.config)

        # Chunk text
        result = await strategy.chunk(request.text, doc_id=request.doc_id)

        logger.info(
            "chunking_completed",
            strategy=request.strategy,
            num_chunks=len(result.chunks),
            processing_time_ms=result.processing_time_ms,
        )

        return ChunkResponse(
            chunks=result.chunks,
            num_chunks=len(result.chunks),
            processing_time_ms=result.processing_time_ms,
            strategy=strategy.NAME,
            strategy_version=strategy.VERSION,
            metadata=[m.model_dump() for m in result.metadata] if result.metadata else None,
        )

    except Exception as e:
        logger.error("chunking_failed", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@app.post("/embed", response_model=EmbedResponse, tags=["Embedding"])
async def embed_texts(request: EmbedRequest):
    """
    Generate embeddings for texts.

    Args:
        request: EmbedRequest with texts and provider config

    Returns:
        EmbedResponse with embeddings and metadata

    Raises:
        HTTPException: If provider not found or embedding fails
    """
    try:
        # Create provider
        provider = EmbeddingProviderFactory.create(request.provider, request.config)

        # Generate embeddings
        result = await provider.embed_texts(request.texts)

        logger.info(
            "embedding_completed",
            provider=request.provider,
            num_texts=len(request.texts),
            dimensions=result.dimensions,
            processing_time_ms=result.processing_time_ms,
        )

        return EmbedResponse(
            embeddings=result.embeddings,
            dimensions=result.dimensions,
            token_count=result.token_count,
            processing_time_ms=result.processing_time_ms,
            provider=result.provider_name,
            model=result.model_name,
            cost_usd=result.cost_usd,
        )

    except Exception as e:
        logger.error("embedding_failed", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@app.post("/evaluate", response_model=EvaluateResponse, tags=["Evaluation"])
async def evaluate_chunks(request: EvaluateRequest):
    """
    Evaluate chunks with specified metrics.

    Args:
        request: EvaluateRequest with chunks, embeddings, and config

    Returns:
        EvaluateResponse with metric results

    Raises:
        HTTPException: If evaluation fails
    """
    start_time = time.time()

    try:
        # Create evaluation pipeline
        pipeline = EvaluationPipeline(metrics=request.metrics)

        # Evaluate
        results = await pipeline.evaluate(
            chunks=request.chunks,
            embeddings=request.embeddings,
            ground_truth=request.ground_truth,
            context=request.context,
        )

        processing_time = (time.time() - start_time) * 1000

        logger.info(
            "evaluation_completed",
            num_metrics=len(results),
            processing_time_ms=processing_time,
        )

        # Convert to dict
        results_dict = {
            name: {
                "score": result.score,
                "version": result.version,
                "details": result.details,
            }
            for name, result in results.items()
        }

        return EvaluateResponse(
            results=results_dict,
            num_metrics=len(results),
            processing_time_ms=processing_time,
        )

    except Exception as e:
        logger.error("evaluation_failed", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@app.post("/compare", response_model=CompareResponse, tags=["Evaluation"])
async def compare_strategies(request: CompareRequest):
    """
    Compare multiple chunking strategies.

    Args:
        request: CompareRequest with text, strategies, and config

    Returns:
        CompareResponse with comparison results and rankings

    Raises:
        HTTPException: If comparison fails
    """
    start_time = time.time()

    try:
        # Create strategies
        strategies = []
        for strategy_config in request.strategies:
            strategy = StrategyRegistry.create(
                strategy_config.get("name", "recursive"),
                strategy_config.get("config"),
            )
            strategies.append(strategy)

        # Create embedding provider
        embedder = EmbeddingProviderFactory.create(
            request.embedding_provider,
            request.embedding_config,
        )

        # Create evaluation pipeline
        pipeline = EvaluationPipeline(metrics=request.metrics)

        # Generate embeddings per strategy
        embeddings_per_strategy = {}
        for strategy in strategies:
            # Chunk first
            chunk_result = await strategy.chunk(request.text)

            # Embed
            emb_result = await embedder.embed_texts(chunk_result.chunks)
            embeddings_per_strategy[strategy.NAME] = emb_result.embeddings

        # Compare
        comparison = await pipeline.compare_strategies(
            strategies=strategies,
            text=request.text,
            embeddings_per_strategy=embeddings_per_strategy,
            ground_truth=request.ground_truth,
        )

        processing_time = (time.time() - start_time) * 1000

        # Determine best strategy (highest weighted score)
        best_strategy = None
        best_score = -1.0

        for strategy_name in comparison["strategies"].keys():
            metric_results = comparison["strategies"][strategy_name]["metric_results"]

            # Compute average score
            if metric_results:
                avg_score = sum(m.score for m in metric_results.values()) / len(metric_results)
                if avg_score > best_score:
                    best_score = avg_score
                    best_strategy = strategy_name

        logger.info(
            "comparison_completed",
            num_strategies=len(strategies),
            best_strategy=best_strategy,
            processing_time_ms=processing_time,
        )

        # Convert to serializable format
        strategies_dict = {}
        for name, data in comparison["strategies"].items():
            strategies_dict[name] = {
                "strategy_version": data["strategy_version"],
                "num_chunks": len(data["chunk_result"].chunks),
                "metric_results": {
                    metric_name: {
                        "score": result.score,
                        "details": result.details,
                    }
                    for metric_name, result in data["metric_results"].items()
                },
            }

        return CompareResponse(
            strategies=strategies_dict,
            rankings=comparison["rankings"],
            best_strategy=best_strategy or "unknown",
            processing_time_ms=processing_time,
        )

    except Exception as e:
        logger.error("comparison_failed", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@app.get("/strategies", tags=["Discovery"])
async def list_strategies():
    """
    List all available chunking strategies.

    Returns:
        List of strategy info dicts
    """
    return {"strategies": StrategyRegistry.list_strategies()}


@app.get("/metrics", tags=["Discovery"])
async def list_metrics():
    """
    List all available evaluation metrics.

    Returns:
        Dict with metrics organized by category
    """
    return {
        "metrics": MetricRegistry.list_metrics(),
        "by_category": MetricRegistry.get_metrics_by_category(),
    }


@app.get("/providers", tags=["Discovery"])
async def list_providers():
    """
    List all available embedding providers.

    Returns:
        List of provider names
    """
    return {"providers": EmbeddingProviderFactory.list_providers()}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
