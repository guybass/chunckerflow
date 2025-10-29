"""Embedding providers and factory."""

from chunk_flow.embeddings.factory import EmbeddingProviderFactory

# Import providers if dependencies available
try:
    from chunk_flow.embeddings.providers.openai_provider import OpenAIEmbeddingProvider
except ImportError:
    OpenAIEmbeddingProvider = None  # type: ignore

try:
    from chunk_flow.embeddings.providers.huggingface_provider import (
        HuggingFaceEmbeddingProvider,
    )
except ImportError:
    HuggingFaceEmbeddingProvider = None  # type: ignore

__all__ = [
    "EmbeddingProviderFactory",
    "OpenAIEmbeddingProvider",
    "HuggingFaceEmbeddingProvider",
]
