"""HuggingFace Sentence Transformers embedding provider."""

import time
from typing import Any, Dict, List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from chunk_flow.core.base import EmbeddingProvider
from chunk_flow.core.exceptions import EmbeddingError
from chunk_flow.core.models import EmbeddingResult
from chunk_flow.utils.logging import get_logger

logger = get_logger(__name__)


class HuggingFaceEmbeddingProvider(EmbeddingProvider):
    """
    HuggingFace Sentence Transformers embedding provider.

    Popular models:
    - all-MiniLM-L6-v2 (384 dims, 22.7M params) - Fast, most popular
    - all-mpnet-base-v2 (768 dims) - Better quality
    - BAAI/bge-small-en-v1.5 (384 dims) - High quality for size
    - BAAI/bge-base-en-v1.5 (768 dims) - Strong performance
    - nomic-ai/nomic-embed-text-v1 (768 dims) - Long context (8K)

    Features:
    - Local inference (no API costs!)
    - GPU/CPU support
    - Batch processing
    - Normalization options
    - ONNX/OpenVINO optimization support
    """

    VERSION = "1.0.0"
    PROVIDER_NAME = "huggingface"

    def _initialize(self) -> None:
        """Initialize Sentence Transformer model."""
        try:
            model_name = self.config.get("model", "sentence-transformers/all-MiniLM-L6-v2")
            device = self.config.get("device", "cpu")
            cache_folder = self.config.get("cache_folder")

            logger.info(
                "loading_huggingface_model",
                model=model_name,
                device=device,
            )

            self.model = SentenceTransformer(
                model_name,
                device=device,
                cache_folder=cache_folder,
            )

            self.batch_size = self.config.get("batch_size", 32)
            self.normalize = self.config.get("normalize", True)
            self.show_progress = self.config.get("show_progress", False)

            # Get model dimensions
            self.dimensions = self.model.get_sentence_embedding_dimension()

            logger.info(
                "huggingface_model_loaded",
                model=model_name,
                dimensions=self.dimensions,
                device=device,
            )

        except Exception as e:
            logger.error("model_loading_failed", error=str(e), exc_info=True)
            raise EmbeddingError(f"Failed to load HuggingFace model: {str(e)}") from e

    def _validate_config(self) -> None:
        """Validate configuration."""
        device = self.config.get("device", "cpu")
        if device not in ["cpu", "cuda", "mps"]:
            logger.warning("unknown_device", device=device)

    async def embed_texts(self, texts: List[str]) -> EmbeddingResult:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            EmbeddingResult with embeddings and metadata
        """
        self.validate_inputs(texts)
        start_time = time.time()

        try:
            # Encode texts (synchronous, runs in thread pool implicitly)
            embeddings_np = self.model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=self.show_progress,
                normalize_embeddings=self.normalize,
                convert_to_numpy=True,
            )

            # Convert to list of lists
            embeddings: List[List[float]] = embeddings_np.tolist()

            # Estimate token count (rough approximation)
            token_count = sum(len(text.split()) for text in texts)

            processing_time = (time.time() - start_time) * 1000

            logger.info(
                "embeddings_generated",
                provider=self.PROVIDER_NAME,
                model=self.config.get("model"),
                num_texts=len(texts),
                dimensions=self.dimensions,
                processing_time_ms=processing_time,
            )

            return EmbeddingResult(
                embeddings=embeddings,
                model_name=self.config.get("model", "unknown"),
                dimensions=self.dimensions,
                processing_time_ms=processing_time,
                token_count=token_count,
                cost_usd=0.0,  # Local inference is free!
                provider_version=self.VERSION,
                provider_name=self.PROVIDER_NAME,
            )

        except Exception as e:
            logger.error(
                "embedding_failed",
                provider=self.PROVIDER_NAME,
                error=str(e),
                exc_info=True,
            )
            raise EmbeddingError(f"HuggingFace embedding failed: {str(e)}") from e

    async def embed_query(self, query: str) -> EmbeddingResult:
        """
        Generate embedding for a single query.

        Args:
            query: Query text to embed

        Returns:
            EmbeddingResult with single embedding
        """
        return await self.embed_texts([query])

    def get_similarity_matrix(self, embeddings: List[List[float]]) -> np.ndarray:
        """
        Compute pairwise cosine similarities between embeddings.

        Args:
            embeddings: List of embedding vectors

        Returns:
            Similarity matrix (n x n)
        """
        from sklearn.metrics.pairwise import cosine_similarity

        embeddings_array = np.array(embeddings)
        return cosine_similarity(embeddings_array)

    def get_info(self) -> Dict[str, Any]:
        """Get provider information."""
        info = super().get_info()
        info.update(
            {
                "model": self.config.get("model"),
                "dimensions": self.dimensions,
                "device": self.config.get("device", "cpu"),
                "batch_size": self.batch_size,
                "normalize": self.normalize,
                "cost": "Free (local inference)",
            }
        )
        return info
