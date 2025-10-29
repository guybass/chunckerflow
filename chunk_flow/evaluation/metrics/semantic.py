"""Semantic evaluation metrics for chunking quality."""

from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from chunk_flow.core.base import EvaluationMetric
from chunk_flow.core.exceptions import EvaluationError, ValidationError
from chunk_flow.core.models import MetricResult
from chunk_flow.utils.logging import get_logger

logger = get_logger(__name__)


class SemanticCoherenceMetric(EvaluationMetric):
    """
    Semantic coherence within chunks.

    Measures how semantically similar sentences within a chunk are.
    Higher scores indicate chunks contain closely related content.
    Range: 0-1, with 1 = perfect coherence.

    Implementation: Average pairwise cosine similarity of sentence embeddings
    within each chunk.
    """

    VERSION = "1.0.0"
    METRIC_NAME = "semantic_coherence"
    REQUIRES_GROUND_TRUTH = False

    async def compute(
        self,
        chunks: List[str],
        embeddings: Optional[List[List[float]]] = None,
        ground_truth: Optional[Any] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> MetricResult:
        """
        Compute semantic coherence.

        Args:
            chunks: List of text chunks
            embeddings: Sentence embeddings for each chunk (required)
            ground_truth: Not required
            context: Optional context with sentence-level embeddings

        Returns:
            MetricResult with coherence score
        """
        self.validate_inputs(chunks, embeddings, ground_truth)

        if embeddings is None:
            raise ValidationError("Semantic coherence requires chunk embeddings")

        try:
            # Get sentence-level embeddings if provided in context
            sentence_embeddings_per_chunk = context.get("sentence_embeddings", []) if context else []

            if not sentence_embeddings_per_chunk:
                # Fallback: Use chunk embeddings as proxy
                logger.warning("no_sentence_embeddings", msg="Using chunk embeddings as proxy")
                coherence_scores = self._compute_chunk_coherence(embeddings)
            else:
                # Compute true intra-chunk coherence
                coherence_scores = self._compute_sentence_coherence(sentence_embeddings_per_chunk)

            # Average across all chunks
            avg_coherence = float(np.mean(coherence_scores))
            min_coherence = float(np.min(coherence_scores))
            max_coherence = float(np.max(coherence_scores))

            logger.info(
                "semantic_coherence_computed",
                avg=avg_coherence,
                min=min_coherence,
                max=max_coherence,
                num_chunks=len(chunks),
            )

            return MetricResult(
                metric_name=self.METRIC_NAME,
                score=avg_coherence,
                version=self.VERSION,
                details={
                    "min_coherence": min_coherence,
                    "max_coherence": max_coherence,
                    "per_chunk_scores": coherence_scores,
                    "num_chunks": len(chunks),
                },
            )

        except Exception as e:
            logger.error("semantic_coherence_error", error=str(e), exc_info=True)
            raise EvaluationError(f"Semantic coherence computation failed: {str(e)}") from e

    def _compute_chunk_coherence(self, embeddings: List[List[float]]) -> List[float]:
        """
        Compute coherence using consecutive chunk similarity as proxy.

        Args:
            embeddings: Chunk embeddings

        Returns:
            Coherence scores (using inter-chunk similarity as proxy)
        """
        if len(embeddings) <= 1:
            return [1.0]

        emb_array = np.array(embeddings)
        similarities = cosine_similarity(emb_array)

        # Use average similarity to neighboring chunks as proxy for coherence
        scores = []
        for i in range(len(embeddings)):
            neighbors = []
            if i > 0:
                neighbors.append(similarities[i, i - 1])
            if i < len(embeddings) - 1:
                neighbors.append(similarities[i, i + 1])

            scores.append(float(np.mean(neighbors)) if neighbors else 1.0)

        return scores

    def _compute_sentence_coherence(
        self, sentence_embeddings_per_chunk: List[List[List[float]]]
    ) -> List[float]:
        """
        Compute true intra-chunk coherence using sentence embeddings.

        Args:
            sentence_embeddings_per_chunk: List of sentence embeddings for each chunk

        Returns:
            Coherence scores per chunk
        """
        scores = []

        for sentence_embs in sentence_embeddings_per_chunk:
            if len(sentence_embs) <= 1:
                # Single sentence chunk = perfect coherence
                scores.append(1.0)
                continue

            # Compute pairwise similarity between all sentences in chunk
            emb_array = np.array(sentence_embs)
            sim_matrix = cosine_similarity(emb_array)

            # Average all pairwise similarities (excluding diagonal)
            n = len(sentence_embs)
            mask = np.ones_like(sim_matrix, dtype=bool)
            np.fill_diagonal(mask, False)

            avg_similarity = float(np.mean(sim_matrix[mask]))
            scores.append(avg_similarity)

        return scores


class ChunkBoundaryQualityMetric(EvaluationMetric):
    """
    Chunk boundary quality metric.

    Measures how well chunks separate distinct topics.
    Good boundaries = low similarity between consecutive chunks.
    Range: 0-1, with 1 = perfect boundaries (very distinct topics).

    Research-backed: From "Mismatch of Chunks" paper - measures boundary clarity.
    """

    VERSION = "1.0.0"
    METRIC_NAME = "boundary_quality"
    REQUIRES_GROUND_TRUTH = False

    async def compute(
        self,
        chunks: List[str],
        embeddings: Optional[List[List[float]]] = None,
        ground_truth: Optional[Any] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> MetricResult:
        """
        Compute boundary quality.

        Args:
            chunks: List of text chunks
            embeddings: Chunk embeddings
            ground_truth: Not required
            context: Optional context

        Returns:
            MetricResult with boundary quality score
        """
        self.validate_inputs(chunks, embeddings, ground_truth)

        if embeddings is None:
            raise ValidationError("Boundary quality requires chunk embeddings")

        if len(chunks) <= 1:
            # Single chunk = no boundaries to evaluate
            return MetricResult(
                metric_name=self.METRIC_NAME,
                score=1.0,
                version=self.VERSION,
                details={"num_boundaries": 0, "message": "Single chunk, no boundaries"},
            )

        try:
            emb_array = np.array(embeddings)
            similarities = cosine_similarity(emb_array)

            # Extract consecutive chunk similarities (boundary strength)
            boundary_similarities = []
            for i in range(len(chunks) - 1):
                boundary_similarities.append(similarities[i, i + 1])

            # Lower similarity = better boundary (more distinct topics)
            # Invert: boundary_quality = 1 - similarity
            boundary_qualities = [1.0 - sim for sim in boundary_similarities]

            avg_quality = float(np.mean(boundary_qualities))
            min_quality = float(np.min(boundary_qualities))
            max_quality = float(np.max(boundary_qualities))

            logger.info(
                "boundary_quality_computed",
                avg=avg_quality,
                min=min_quality,
                max=max_quality,
                num_boundaries=len(boundary_similarities),
            )

            return MetricResult(
                metric_name=self.METRIC_NAME,
                score=avg_quality,
                version=self.VERSION,
                details={
                    "num_boundaries": len(boundary_similarities),
                    "min_quality": min_quality,
                    "max_quality": max_quality,
                    "per_boundary_quality": boundary_qualities,
                    "boundary_similarities": boundary_similarities,
                },
            )

        except Exception as e:
            logger.error("boundary_quality_error", error=str(e), exc_info=True)
            raise EvaluationError(f"Boundary quality computation failed: {str(e)}") from e


class ChunkStickinessMetric(EvaluationMetric):
    """
    Chunk stickiness metric (from MoC paper).

    Measures how "sticky" chunks are to adjacent chunks.
    High stickiness = poor chunking (topics bleeding across boundaries).
    Low stickiness = good chunking (clean topic separation).

    Range: 0-1, with 0 = no stickiness (perfect), 1 = maximum stickiness (poor).

    Research: "Mismatch of Chunks and Mismatched Retrieval" paper identified
    chunk stickiness as a key failure mode in semantic chunking.
    """

    VERSION = "1.0.0"
    METRIC_NAME = "chunk_stickiness"
    REQUIRES_GROUND_TRUTH = False

    async def compute(
        self,
        chunks: List[str],
        embeddings: Optional[List[List[float]]] = None,
        ground_truth: Optional[Any] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> MetricResult:
        """
        Compute chunk stickiness.

        Args:
            chunks: List of text chunks
            embeddings: Chunk embeddings
            ground_truth: Not required
            context: Optional context

        Returns:
            MetricResult with stickiness score (lower = better)
        """
        self.validate_inputs(chunks, embeddings, ground_truth)

        if embeddings is None:
            raise ValidationError("Chunk stickiness requires chunk embeddings")

        if len(chunks) <= 1:
            # Single chunk = no stickiness
            return MetricResult(
                metric_name=self.METRIC_NAME,
                score=0.0,
                version=self.VERSION,
                details={"num_chunks": 1, "message": "Single chunk, no stickiness"},
            )

        try:
            emb_array = np.array(embeddings)
            similarities = cosine_similarity(emb_array)

            # Stickiness = average similarity between consecutive chunks
            # (Higher similarity = more stickiness = worse separation)
            consecutive_similarities = []
            for i in range(len(chunks) - 1):
                consecutive_similarities.append(similarities[i, i + 1])

            avg_stickiness = float(np.mean(consecutive_similarities))
            max_stickiness = float(np.max(consecutive_similarities))
            min_stickiness = float(np.min(consecutive_similarities))

            # Identify highly sticky boundaries (> 75th percentile)
            threshold = float(np.percentile(consecutive_similarities, 75))
            sticky_boundaries = [
                i for i, sim in enumerate(consecutive_similarities) if sim > threshold
            ]

            logger.info(
                "chunk_stickiness_computed",
                avg_stickiness=avg_stickiness,
                max_stickiness=max_stickiness,
                num_sticky_boundaries=len(sticky_boundaries),
            )

            return MetricResult(
                metric_name=self.METRIC_NAME,
                score=avg_stickiness,
                version=self.VERSION,
                details={
                    "max_stickiness": max_stickiness,
                    "min_stickiness": min_stickiness,
                    "sticky_boundaries": sticky_boundaries,
                    "threshold": threshold,
                    "per_boundary_stickiness": consecutive_similarities,
                },
            )

        except Exception as e:
            logger.error("chunk_stickiness_error", error=str(e), exc_info=True)
            raise EvaluationError(f"Chunk stickiness computation failed: {str(e)}") from e


class TopicDiversityMetric(EvaluationMetric):
    """
    Topic diversity across chunks.

    Measures how diverse the topics are across all chunks.
    Higher diversity = chunks cover different topics well.
    Range: 0-1, with 1 = maximum diversity.

    Implementation: Average pairwise distance across all chunks.
    """

    VERSION = "1.0.0"
    METRIC_NAME = "topic_diversity"
    REQUIRES_GROUND_TRUTH = False

    async def compute(
        self,
        chunks: List[str],
        embeddings: Optional[List[List[float]]] = None,
        ground_truth: Optional[Any] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> MetricResult:
        """
        Compute topic diversity.

        Args:
            chunks: List of text chunks
            embeddings: Chunk embeddings
            ground_truth: Not required
            context: Optional context

        Returns:
            MetricResult with diversity score
        """
        self.validate_inputs(chunks, embeddings, ground_truth)

        if embeddings is None:
            raise ValidationError("Topic diversity requires chunk embeddings")

        if len(chunks) <= 1:
            return MetricResult(
                metric_name=self.METRIC_NAME,
                score=0.0,
                version=self.VERSION,
                details={"num_chunks": len(chunks), "message": "Insufficient chunks for diversity"},
            )

        try:
            emb_array = np.array(embeddings)
            similarities = cosine_similarity(emb_array)

            # Diversity = average pairwise distance (1 - similarity)
            # Exclude diagonal (self-similarity)
            n = len(chunks)
            mask = np.ones_like(similarities, dtype=bool)
            np.fill_diagonal(mask, False)

            avg_similarity = float(np.mean(similarities[mask]))
            diversity = 1.0 - avg_similarity

            # Additional stats
            distances = 1.0 - similarities[mask]
            max_diversity = float(np.max(distances))
            min_diversity = float(np.min(distances))

            logger.info(
                "topic_diversity_computed",
                diversity=diversity,
                avg_similarity=avg_similarity,
                num_chunks=n,
            )

            return MetricResult(
                metric_name=self.METRIC_NAME,
                score=diversity,
                version=self.VERSION,
                details={
                    "num_chunks": n,
                    "avg_similarity": avg_similarity,
                    "max_diversity": max_diversity,
                    "min_diversity": min_diversity,
                },
            )

        except Exception as e:
            logger.error("topic_diversity_error", error=str(e), exc_info=True)
            raise EvaluationError(f"Topic diversity computation failed: {str(e)}") from e
