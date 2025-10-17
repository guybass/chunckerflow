"""Recursive character chunking strategy."""

import re
import time
from typing import Any, Dict, List, Optional

import tiktoken

from chunk_flow.core.base import ChunkingStrategy
from chunk_flow.core.exceptions import ChunkingError, ConfigurationError
from chunk_flow.core.models import ChunkMetadata, ChunkResult
from chunk_flow.utils.logging import get_logger

logger = get_logger(__name__)


class RecursiveCharacterChunker(ChunkingStrategy):
    """
    Recursive character chunking strategy.

    Hierarchically divides text using prioritized separators (paragraphs → sentences → words).
    Recursively applies next separator when chunks exceed target size.

    Performance: 5,000+ chunks/second.
    Best for: General-purpose text, articles, books (recommended default).

    Research: R100-0 (recursive 100 tokens, 0 overlap) consistently outperforms
    across 48 embedding models (Chunk Twice, Embed Once study).
    """

    VERSION = "1.0.0"
    NAME = "recursive"

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize recursive chunker.

        Args:
            config: Configuration with chunk_size, overlap, separators
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
            "overlap": 100,
            "separators": ["\n\n", "\n", ". ", " ", ""],
            "length_function": "char",  # char or token
            "encoding_name": "cl100k_base",
            "keep_separator": False,
        }

    def _validate_config(self) -> None:
        """Validate configuration."""
        required = ["chunk_size", "separators"]
        for key in required:
            if key not in self.config:
                raise ConfigurationError(f"Missing required config: {key}")

        if self.config["chunk_size"] <= 0:
            raise ConfigurationError("chunk_size must be positive")

        if not isinstance(self.config["separators"], list):
            raise ConfigurationError("separators must be a list")

    async def chunk(self, text: str, doc_id: Optional[str] = None) -> ChunkResult:
        """
        Chunk text recursively.

        Args:
            text: Input text to chunk
            doc_id: Optional document identifier

        Returns:
            ChunkResult with chunks and metadata
        """
        self.validate_input(text)
        start_time = time.time()

        try:
            chunk_size = self.config["chunk_size"]
            overlap = self.config.get("overlap", 0)
            separators = self.config["separators"]

            # Perform recursive splitting
            chunks = await self._recursive_split(text, separators, chunk_size)

            # Add overlap if specified
            if overlap > 0:
                chunks = await self._add_overlap(chunks, overlap)

            # Create metadata
            metadata = await self._create_metadata(chunks, text, doc_id)

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
            raise ChunkingError(f"Recursive chunking failed: {str(e)}") from e

    async def _recursive_split(
        self, text: str, separators: List[str], chunk_size: int
    ) -> List[str]:
        """
        Recursively split text using hierarchical separators.

        Args:
            text: Text to split
            separators: List of separators in priority order
            chunk_size: Maximum chunk size

        Returns:
            List of text chunks
        """
        if not separators:
            # Base case: no more separators, return text as-is
            return [text] if text else []

        separator = separators[0]
        remaining_separators = separators[1:]

        # Split by current separator
        if separator:
            splits = text.split(separator)
            if self.config.get("keep_separator"):
                # Re-add separator to splits (except last)
                splits = [
                    s + separator if i < len(splits) - 1 else s for i, s in enumerate(splits)
                ]
        else:
            # Empty separator means split by character
            splits = list(text)

        chunks: List[str] = []
        current_chunk = ""

        for split in splits:
            # Check if adding this split would exceed chunk size
            if self._get_length(current_chunk + split) <= chunk_size:
                current_chunk += split
            else:
                # Current chunk is full
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = ""

                # If split itself is too large, recursively split it
                if self._get_length(split) > chunk_size:
                    if remaining_separators:
                        sub_chunks = await self._recursive_split(
                            split, remaining_separators, chunk_size
                        )
                        chunks.extend(sub_chunks)
                    else:
                        # Force split by characters
                        for i in range(0, len(split), chunk_size):
                            chunks.append(split[i : i + chunk_size])
                else:
                    current_chunk = split

        # Don't forget the last chunk
        if current_chunk:
            chunks.append(current_chunk)

        return [c for c in chunks if c.strip()]  # Filter empty chunks

    async def _add_overlap(self, chunks: List[str], overlap: int) -> List[str]:
        """Add overlap between chunks."""
        if len(chunks) <= 1:
            return chunks

        overlapped_chunks: List[str] = [chunks[0]]

        for i in range(1, len(chunks)):
            prev_chunk = chunks[i - 1]
            curr_chunk = chunks[i]

            # Get overlap from previous chunk
            if self._get_length(prev_chunk) >= overlap:
                overlap_text = self._get_last_n(prev_chunk, overlap)
                overlapped_chunks.append(overlap_text + curr_chunk)
            else:
                overlapped_chunks.append(curr_chunk)

        return overlapped_chunks

    def _get_length(self, text: str) -> int:
        """Get length of text (chars or tokens)."""
        if self.config.get("length_function") == "token" and self.tokenizer:
            return len(self.tokenizer.encode(text))
        return len(text)

    def _get_last_n(self, text: str, n: int) -> str:
        """Get last n characters/tokens of text."""
        if self.config.get("length_function") == "token" and self.tokenizer:
            tokens = self.tokenizer.encode(text)
            if len(tokens) <= n:
                return text
            overlap_tokens = tokens[-n:]
            return self.tokenizer.decode(overlap_tokens)
        return text[-n:] if len(text) > n else text

    async def _create_metadata(
        self, chunks: List[str], original_text: str, doc_id: Optional[str]
    ) -> List[ChunkMetadata]:
        """Create metadata for chunks."""
        metadata: List[ChunkMetadata] = []
        current_pos = 0

        for i, chunk in enumerate(chunks):
            # Find chunk position in original text (approximate for overlapped chunks)
            start_idx = original_text.find(chunk, current_pos)
            if start_idx == -1:
                # Chunk might be modified by overlap, use current position
                start_idx = current_pos

            end_idx = start_idx + len(chunk)
            current_pos = start_idx + 1  # For next search

            metadata.append(
                ChunkMetadata(
                    chunk_id=f"{doc_id or 'doc'}_{self.NAME}_{i}",
                    start_idx=start_idx,
                    end_idx=end_idx,
                    token_count=self._get_length(chunk),
                    char_count=len(chunk),
                    version=self.VERSION,
                    strategy_name=self.NAME,
                )
            )

        return metadata
