"""Late chunking - Revolutionary context-preserving approach."""

import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from chunk_flow.core.base import ChunkingStrategy
from chunk_flow.core.exceptions import ChunkingError, ConfigurationError
from chunk_flow.core.models import ChunkMetadata, ChunkResult
from chunk_flow.utils.logging import get_logger

logger = get_logger(__name__)


class LateChunker(ChunkingStrategy):
    """
    Late chunking strategy - Contextual chunk embeddings using long-context models.

    Revolutionary approach: Reverses traditional pipeline by embedding entire document
    first (8K+ tokens), THEN deriving chunk embeddings while preserving full context.
    Chunking occurs after transformer layer but before mean pooling.

    Performance: Fast (single model pass), superior retrieval accuracy.
    Breakthrough Results (BeIR Benchmark):
    - SciFact: 64.20% → 66.10% (+1.9% nDCG)
    - NFCorpus: 23.46% → 29.98% (+6.5% nDCG)

    Best for: Long documents (1000+ words), cross-references, anaphoric references,
    research papers, technical reports.

    Research: Jina AI (arXiv:2409.04701, July 2025) - generic method applicable
    to any 8K+ context model without additional training.
    """

    VERSION = "1.0.0"
    NAME = "late"

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize late chunker."""
        super().__init__(config)
        self.model: Optional[Any] = None
        self.tokenizer: Optional[Any] = None

    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "model_name": "jinaai/jina-embeddings-v2-base-en",  # 8K context
            "chunk_size": 256,  # tokens per chunk
            "max_context_length": 8192,
            "device": "cpu",
            "pooling_strategy": "mean",  # mean or cls
        }

    def _validate_config(self) -> None:
        """Validate configuration."""
        chunk_size = self.config.get("chunk_size", 256)
        max_context = self.config.get("max_context_length", 8192)

        if chunk_size <= 0:
            raise ConfigurationError("chunk_size must be positive")

        if chunk_size > max_context:
            raise ConfigurationError("chunk_size cannot exceed max_context_length")

    def _load_model(self) -> None:
        """Load model and tokenizer lazily."""
        if self.model is None:
            model_name = self.config.get("model_name", "jinaai/jina-embeddings-v2-base-en")
            device = self.config.get("device", "cpu")

            logger.info("loading_late_chunking_model", model=model_name, device=device)

            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
                self.model.to(device)
                self.model.eval()  # Set to evaluation mode

                logger.info("late_chunking_model_loaded", model=model_name, device=device)

            except Exception as e:
                raise ChunkingError(f"Failed to load late chunking model: {str(e)}") from e

    async def chunk(self, text: str, doc_id: Optional[str] = None) -> ChunkResult:
        """
        Chunk using late chunking approach.

        Process:
        1. Tokenize full document
        2. Get token-level embeddings for entire document (preserves context!)
        3. Define chunk boundaries at token level
        4. Mean pool tokens within each chunk

        Args:
            text: Input text to chunk
            doc_id: Optional document identifier

        Returns:
            ChunkResult with context-aware chunks
        """
        self.validate_input(text)
        start_time = time.time()

        try:
            # Load model if needed
            self._load_model()

            chunk_size = self.config.get("chunk_size", 256)
            max_length = self.config.get("max_context_length", 8192)

            # Step 1: Tokenize full document
            inputs = self.tokenizer(
                text,
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
                padding=False,
            )

            device = self.config.get("device", "cpu")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Step 2: Get token-level embeddings for entire document
            with torch.no_grad():
                outputs = self.model(**inputs)

                # Get last hidden state (token-level embeddings)
                # Shape: [batch_size, sequence_length, hidden_dim]
                token_embeddings = outputs.last_hidden_state[0]  # Remove batch dimension

            # Convert to numpy for easier manipulation
            token_embeddings_np = token_embeddings.cpu().numpy()

            # Step 3: Define chunk boundaries (256-token chunks)
            num_tokens = token_embeddings_np.shape[0]
            boundaries = list(range(0, num_tokens, chunk_size))
            if boundaries[-1] != num_tokens:
                boundaries.append(num_tokens)

            # Step 4: Mean pool tokens within each chunk & reconstruct text
            chunks = []
            metadata = []
            chunk_embeddings = []

            input_ids = inputs["input_ids"][0].cpu().numpy()

            for i in range(len(boundaries) - 1):
                start_idx = boundaries[i]
                end_idx = boundaries[i + 1]

                # Mean pool token embeddings for this chunk
                chunk_emb = np.mean(token_embeddings_np[start_idx:end_idx], axis=0)
                chunk_embeddings.append(chunk_emb.tolist())

                # Decode tokens back to text
                chunk_token_ids = input_ids[start_idx:end_idx]
                chunk_text = self.tokenizer.decode(chunk_token_ids, skip_special_tokens=True)

                if chunk_text.strip():  # Skip empty chunks
                    chunks.append(chunk_text)
                    metadata.append(
                        ChunkMetadata(
                            chunk_id=f"{doc_id or 'doc'}_{self.NAME}_{i}",
                            start_idx=start_idx,
                            end_idx=end_idx,
                            token_count=end_idx - start_idx,
                            char_count=len(chunk_text),
                            version=self.VERSION,
                            strategy_name=self.NAME,
                            custom_fields={
                                "has_context": True,
                                "context_window": num_tokens,
                            },
                        )
                    )

            processing_time = (time.time() - start_time) * 1000

            logger.info(
                "late_chunking_completed",
                num_chunks=len(chunks),
                total_tokens=num_tokens,
                chunk_size=chunk_size,
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
            logger.error("late_chunking_failed", error=str(e), exc_info=True)
            raise ChunkingError(f"Late chunking failed: {str(e)}") from e

    def get_chunk_embeddings(self, chunk_result: ChunkResult) -> Optional[List[List[float]]]:
        """
        Get pre-computed chunk embeddings if available.

        Late chunking produces embeddings as a byproduct. This method would return
        them if we store them. For now, this is a placeholder for future enhancement.

        Args:
            chunk_result: Result from chunk() method

        Returns:
            List of chunk embeddings if available, None otherwise
        """
        # TODO: Store embeddings in custom_fields during chunking
        # and retrieve them here
        return None
