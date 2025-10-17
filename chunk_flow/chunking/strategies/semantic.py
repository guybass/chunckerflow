"""Semantic chunking using embeddings for topic detection."""

import time
from typing import Any, Dict, List, Optional

import numpy as np
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer

from chunk_flow.core.base import ChunkingStrategy
from chunk_flow.core.exceptions import ChunkingError, ConfigurationError
from chunk_flow.core.models import ChunkMetadata, ChunkResult
from chunk_flow.utils.logging import get_logger

logger = get_logger(__name__)


class SemanticChunker(ChunkingStrategy):
    """
    Semantic chunking using embeddings.

    Analyzes sentence embeddings to detect topic changes, grouping semantically
    similar sentences together. Creates boundaries when similarity drops below threshold.

    Performance: Slower (embedding overhead), high semantic coherence.
    Best for: Multi-topic documents, complex technical content, accuracy-critical use cases.

    Research insight (MoC paper): Semantic chunking can underperform due to high
    chunk stickiness when logical transitions don't correlate with embedding similarity.
    """

    VERSION = "1.0.0"
    NAME = "semantic"

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize semantic chunker."""
        super().__init__(config)
        self.model: Optional[SentenceTransformer] = None

    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "threshold_percentile": 80,  # 80th percentile for breakpoints
            "buffer_size": 1,  # Sentences around each sentence for context
            "min_chunk_size": 100,  # Minimum characters per chunk
            "max_chunk_size": 2000,  # Maximum characters per chunk
            "device": "cpu",
        }

    def _validate_config(self) -> None:
        """Validate configuration."""
        if self.config.get("threshold_percentile", 80) < 0 or self.config.get("threshold_percentile", 80) > 100:
            raise ConfigurationError("threshold_percentile must be between 0 and 100")

    def _load_model(self) -> None:
        """Load embedding model lazily."""
        if self.model is None:
            model_name = self.config.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
            device = self.config.get("device", "cpu")

            logger.info("loading_semantic_model", model=model_name, device=device)

            try:
                self.model = SentenceTransformer(model_name, device=device)
                logger.info("semantic_model_loaded", model=model_name)
            except Exception as e:
                raise ChunkingError(f"Failed to load embedding model: {str(e)}") from e

    async def chunk(self, text: str, doc_id: Optional[str] = None) -> ChunkResult:
        """
        Chunk text semantically using embeddings.

        Args:
            text: Input text to chunk
            doc_id: Optional document identifier

        Returns:
            ChunkResult with semantically coherent chunks
        """
        self.validate_input(text)
        start_time = time.time()

        try:
            # Load model if needed
            self._load_model()

            # Step 1: Split into sentences
            sentences = self._split_into_sentences(text)

            if len(sentences) <= 1:
                # Not enough sentences for semantic chunking
                return ChunkResult(
                    chunks=[text],
                    metadata=[
                        ChunkMetadata(
                            chunk_id=f"{doc_id or 'doc'}_{self.NAME}_0",
                            start_idx=0,
                            end_idx=len(text),
                            token_count=len(text.split()),
                            char_count=len(text),
                            version=self.VERSION,
                            strategy_name=self.NAME,
                        )
                    ],
                    processing_time_ms=(time.time() - start_time) * 1000,
                    strategy_version=self.VERSION,
                    config=self.config,
                    doc_id=doc_id,
                )

            # Step 2: Generate embeddings for each sentence
            logger.info("generating_sentence_embeddings", num_sentences=len(sentences))
            embeddings = self.model.encode(sentences, convert_to_numpy=True)

            # Step 3: Calculate cosine distances between consecutive sentences
            distances = self._calculate_distances(embeddings)

            # Step 4: Determine threshold (percentile-based)
            threshold_percentile = self.config.get("threshold_percentile", 80)
            threshold = np.percentile(distances, threshold_percentile)

            logger.info(
                "semantic_threshold_calculated",
                threshold=threshold,
                percentile=threshold_percentile,
            )

            # Step 5: Find breakpoints where distance > threshold
            breakpoints = [0]  # Start with first sentence
            for i, distance in enumerate(distances):
                if distance > threshold:
                    breakpoints.append(i + 1)
            breakpoints.append(len(sentences))  # End with last sentence

            # Step 6: Group sentences into chunks
            chunks = []
            metadata = []

            for i in range(len(breakpoints) - 1):
                start_idx = breakpoints[i]
                end_idx = breakpoints[i + 1]

                chunk_sentences = sentences[start_idx:end_idx]
                chunk_text = " ".join(chunk_sentences)

                # Apply size constraints
                if len(chunk_text) < self.config.get("min_chunk_size", 100) and i > 0:
                    # Merge with previous chunk if too small
                    chunks[-1] += " " + chunk_text
                    metadata[-1].end_idx += len(chunk_text)
                    metadata[-1].char_count += len(chunk_text)
                    metadata[-1].token_count += len(chunk_text.split())
                    continue

                if len(chunk_text) > self.config.get("max_chunk_size", 2000):
                    # Split large chunk by sentences
                    sub_chunks = self._split_large_chunk(chunk_sentences, self.config["max_chunk_size"])
                    for j, sub_chunk in enumerate(sub_chunks):
                        chunks.append(sub_chunk)
                        metadata.append(
                            self._create_metadata(sub_chunk, len(chunks) - 1, doc_id, i)
                        )
                else:
                    chunks.append(chunk_text)
                    metadata.append(self._create_metadata(chunk_text, len(chunks) - 1, doc_id, i))

            processing_time = (time.time() - start_time) * 1000

            logger.info(
                "semantic_chunking_completed",
                num_sentences=len(sentences),
                num_chunks=len(chunks),
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
            logger.error("semantic_chunking_failed", error=str(e), exc_info=True)
            raise ChunkingError(f"Semantic chunking failed: {str(e)}") from e

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        import re

        # Simple sentence splitting (for production, consider using nltk or spacy)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _calculate_distances(self, embeddings: np.ndarray) -> List[float]:
        """
        Calculate cosine distances between consecutive sentence embeddings.

        Args:
            embeddings: Sentence embeddings array

        Returns:
            List of distances between consecutive sentences
        """
        distances = []
        for i in range(len(embeddings) - 1):
            distance = cosine(embeddings[i], embeddings[i + 1])
            distances.append(distance)
        return distances

    def _split_large_chunk(self, sentences: List[str], max_size: int) -> List[str]:
        """Split large chunk into smaller chunks."""
        chunks = []
        current_chunk = []
        current_size = 0

        for sentence in sentences:
            sentence_size = len(sentence)

            if current_size + sentence_size <= max_size:
                current_chunk.append(sentence)
                current_size += sentence_size + 1  # +1 for space
            else:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_size = sentence_size

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def _create_metadata(
        self, chunk_text: str, chunk_idx: int, doc_id: Optional[str], section_idx: int
    ) -> ChunkMetadata:
        """Create metadata for a chunk."""
        return ChunkMetadata(
            chunk_id=f"{doc_id or 'doc'}_{self.NAME}_{chunk_idx}",
            start_idx=0,  # Will be calculated if needed
            end_idx=len(chunk_text),
            token_count=len(chunk_text.split()),
            char_count=len(chunk_text),
            version=self.VERSION,
            strategy_name=self.NAME,
            custom_fields={"section_index": section_idx},
        )
