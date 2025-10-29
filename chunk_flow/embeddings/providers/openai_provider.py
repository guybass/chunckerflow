"""OpenAI embedding provider."""

import time
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from chunk_flow.core.base import EmbeddingProvider
from chunk_flow.core.config import get_settings
from chunk_flow.core.exceptions import APIError, AuthenticationError, EmbeddingError, RateLimitError
from chunk_flow.core.models import EmbeddingResult
from chunk_flow.utils.logging import get_logger

logger = get_logger(__name__)


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """
    OpenAI embedding provider.

    Supports:
    - text-embedding-3-small (1536 dims, $0.00002/1K tokens) - Cost-effective
    - text-embedding-3-large (3072 dims, $0.00013/1K tokens) - Best quality
    - text-embedding-ada-002 (1536 dims, legacy)

    Features:
    - Configurable dimensions (Matryoshka embeddings)
    - Async batch processing
    - Automatic retries with exponential backoff
    - Cost tracking
    """

    VERSION = "1.0.0"
    PROVIDER_NAME = "openai"

    # Pricing per 1K tokens (as of 2024)
    PRICING = {
        "text-embedding-3-small": 0.00002,
        "text-embedding-3-large": 0.00013,
        "text-embedding-ada-002": 0.0001,
    }

    def _initialize(self) -> None:
        """Initialize OpenAI client."""
        settings = get_settings()
        api_key = self.config.get("api_key") or settings.openai_api_key

        if not api_key:
            raise AuthenticationError(
                "OpenAI API key not found. Set CHUNK_FLOW_OPENAI_API_KEY or pass api_key in config."
            )

        self.client = AsyncOpenAI(api_key=api_key)
        self.model = self.config.get("model", "text-embedding-3-small")
        self.dimensions = self.config.get("dimensions")  # Optional
        self.batch_size = self.config.get("batch_size", 100)
        self.timeout = self.config.get("timeout", 30)

        logger.info(
            "openai_provider_initialized",
            model=self.model,
            dimensions=self.dimensions,
            batch_size=self.batch_size,
        )

    def _validate_config(self) -> None:
        """Validate configuration."""
        model = self.config.get("model", "text-embedding-3-small")
        if model not in self.PRICING:
            logger.warning("unknown_model", model=model)

    @retry(
        retry=retry_if_exception_type((APIError, RateLimitError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    async def embed_texts(self, texts: List[str]) -> EmbeddingResult:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            EmbeddingResult with embeddings and metadata

        Raises:
            EmbeddingError: If embedding generation fails
            RateLimitError: If rate limit exceeded
        """
        self.validate_inputs(texts)
        start_time = time.time()

        try:
            # Prepare request
            request_params: Dict[str, Any] = {
                "input": texts,
                "model": self.model,
            }

            if self.dimensions:
                request_params["dimensions"] = self.dimensions

            # Call OpenAI API
            response = await self.client.embeddings.create(**request_params)

            # Extract embeddings
            embeddings = [item.embedding for item in response.data]

            # Calculate metrics
            token_count = response.usage.total_tokens
            cost = self._calculate_cost(token_count)
            processing_time = (time.time() - start_time) * 1000

            logger.info(
                "embeddings_generated",
                provider=self.PROVIDER_NAME,
                model=self.model,
                num_texts=len(texts),
                token_count=token_count,
                cost_usd=cost,
                processing_time_ms=processing_time,
            )

            return EmbeddingResult(
                embeddings=embeddings,
                model_name=self.model,
                dimensions=len(embeddings[0]) if embeddings else 0,
                processing_time_ms=processing_time,
                token_count=token_count,
                cost_usd=cost,
                provider_version=self.VERSION,
                provider_name=self.PROVIDER_NAME,
            )

        except Exception as e:
            error_msg = str(e)
            logger.error(
                "embedding_failed",
                provider=self.PROVIDER_NAME,
                error=error_msg,
                exc_info=True,
            )

            # Parse OpenAI errors
            if "rate_limit" in error_msg.lower():
                raise RateLimitError(f"OpenAI rate limit exceeded: {error_msg}")
            elif "authentication" in error_msg.lower() or "api_key" in error_msg.lower():
                raise AuthenticationError(f"OpenAI authentication failed: {error_msg}")
            elif "invalid_request" in error_msg.lower():
                raise EmbeddingError(f"Invalid request to OpenAI: {error_msg}")
            else:
                raise EmbeddingError(f"OpenAI embedding failed: {error_msg}") from e

    async def embed_query(self, query: str) -> EmbeddingResult:
        """
        Generate embedding for a single query.

        Args:
            query: Query text to embed

        Returns:
            EmbeddingResult with single embedding
        """
        return await self.embed_texts([query])

    def _calculate_cost(self, tokens: int) -> float:
        """
        Calculate cost based on token count.

        Args:
            tokens: Number of tokens

        Returns:
            Cost in USD
        """
        price_per_1k = self.PRICING.get(self.model, 0.0)
        return (tokens / 1000) * price_per_1k

    def get_info(self) -> Dict[str, Any]:
        """Get provider information."""
        info = super().get_info()
        info.update(
            {
                "model": self.model,
                "dimensions": self.dimensions or "default",
                "supported_models": list(self.PRICING.keys()),
                "pricing": self.PRICING,
            }
        )
        return info
