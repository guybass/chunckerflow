"""Fixed-size chunking strategy."""

import time
from typing import Any, Dict, List, Optional

import tiktoken

from chunk_flow.core.base import ChunkingStrategy
from chunk_flow.core.exceptions import ChunkingError, ConfigurationError
from chunk_flow.core.models import ChunkMetadata, ChunkResult
from chunk_flow.utils.logging import get_logger

logger = get_logger(__name__)


class FixedSizeChunker(ChunkingStrategy):
    """
    Fixed-size chunking strategy.

    Splits text into uniform segments based on predetermined character or token counts.
    Simple and fast, but may break semantic coherence.

    Performance: 10,000+ chunks/second on single CPU.
    Best for: Prototyping, simple documents, uniform content structures.
    """

    VERSION = "1.0.0"
    NAME = "fixed_size"

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize fixed-size chunker.

        Args:
            config: Configuration with chunk_size, overlap, length_function
        """
        super().__init__(config)
        self.tokenizer = None
        if self.config.get("length_function") == "token":
            try:
                encoding_name = self.config.get("encoding_name", "cl100k_base")
                self.tokenizer = tiktoken.get_encoding(encoding_name)
            except Exception as e:
                logger.warning("tokenizer_init_failed", error=str(e))
                self.config["length_function"] = "char"

    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "chunk_size": 512,
            "overlap": 50,
            "length_function": "char",  # char or token
            "encoding_name": "cl100k_base",  # for tiktoken
        }

    def _validate_config(self) -> None:
        """Validate configuration."""
        required = ["chunk_size", "overlap"]
        for key in required:
            if key not in self.config:
                raise ConfigurationError(f"Missing required config: {key}")

        if self.config["chunk_size"] <= 0:
            raise ConfigurationError("chunk_size must be positive")

        if self.config["overlap"] < 0:
            raise ConfigurationError("overlap must be non-negative")

        if self.config["overlap"] >= self.config["chunk_size"]:
            raise ConfigurationError("overlap must be less than chunk_size")

        if self.config.get("length_function") not in ["char", "token"]:
            raise ConfigurationError("length_function must be 'char' or 'token'")

    async def chunk(self, text: str, doc_id: Optional[str] = None) -> ChunkResult:
        """
        Chunk text into fixed-size segments.

        Args:
            text: Input text to chunk
            doc_id: Optional document identifier

        Returns:
            ChunkResult with chunks and metadata

        Raises:
            ChunkingError: If chunking fails
        """
        self.validate_input(text)
        start_time = time.time()

        try:
            chunk_size = self.config["chunk_size"]
            overlap = self.config["overlap"]
            length_func = self.config.get("length_function", "char")

            if length_func == "token" and self.tokenizer:
                chunks, metadata = await self._chunk_by_tokens(text, chunk_size, overlap, doc_id)
            else:
                chunks, metadata = await self._chunk_by_chars(text, chunk_size, overlap, doc_id)

            processing_time = (time.time() - start_time) * 1000

            logger.info(
                "chunking_completed",
                strategy=self.NAME,
                num_chunks=len(chunks),
                doc_id=doc_id,
                processing_time_ms=processing_time,
            )

            return ChunkResult(
                chunks=chunks,
                metadata=metadata,
                processing_time_ms=processing_time,
                strategy_version=self.VERSION,
                config=self.config,
                doc_id=doc_id,
            )

        except Exception as e:
            logger.error("chunking_failed", strategy=self.NAME, error=str(e), exc_info=True)
            raise ChunkingError(f"Fixed-size chunking failed: {str(e)}") from e

    async def _chunk_by_chars(
        self, text: str, chunk_size: int, overlap: int, doc_id: Optional[str]
    ) -> tuple[List[str], List[ChunkMetadata]]:
        """Chunk by character count."""
        chunks: List[str] = []
        metadata: List[ChunkMetadata] = []

        start = 0
        chunk_idx = 0

        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_text = text[start:end]

            if chunk_text.strip():  # Skip empty chunks
                chunks.append(chunk_text)
                metadata.append(
                    ChunkMetadata(
                        chunk_id=f"{doc_id or 'doc'}_{self.NAME}_{chunk_idx}",
                        start_idx=start,
                        end_idx=end,
                        token_count=len(chunk_text.split()),  # Rough estimate
                        char_count=len(chunk_text),
                        version=self.VERSION,
                        strategy_name=self.NAME,
                    )
                )
                chunk_idx += 1

            # Move start position (with overlap)
            start = end - overlap if end < len(text) else end

        return chunks, metadata

    async def _chunk_by_tokens(
        self, text: str, chunk_size: int, overlap: int, doc_id: Optional[str]
    ) -> tuple[List[str], List[ChunkMetadata]]:
        """Chunk by token count."""
        if not self.tokenizer:
            raise ChunkingError("Tokenizer not initialized")

        # Tokenize entire text
        tokens = self.tokenizer.encode(text)
        chunks: List[str] = []
        metadata: List[ChunkMetadata] = []

        start_idx = 0
        chunk_idx = 0
        char_offset = 0

        while start_idx < len(tokens):
            end_idx = min(start_idx + chunk_size, len(tokens))
            chunk_tokens = tokens[start_idx:end_idx]

            # Decode tokens back to text
            chunk_text = self.tokenizer.decode(chunk_tokens)

            if chunk_text.strip():
                chunks.append(chunk_text)
                metadata.append(
                    ChunkMetadata(
                        chunk_id=f"{doc_id or 'doc'}_{self.NAME}_{chunk_idx}",
                        start_idx=char_offset,
                        end_idx=char_offset + len(chunk_text),
                        token_count=len(chunk_tokens),
                        char_count=len(chunk_text),
                        version=self.VERSION,
                        strategy_name=self.NAME,
                    )
                )
                chunk_idx += 1
                char_offset += len(chunk_text)

            # Move start position (with overlap)
            start_idx = end_idx - overlap if end_idx < len(tokens) else end_idx

        return chunks, metadata
