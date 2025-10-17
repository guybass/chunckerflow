"""Retrieval evaluation metrics (NDCG, Recall, MRR, Precision)."""

from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.metrics import ndcg_score
from sklearn.metrics.pairwise import cosine_similarity

from chunk_flow.core.base import EvaluationMetric
from chunk_flow.core.exceptions import EvaluationError, ValidationError
from chunk_flow.core.models import MetricResult
from chunk_flow.utils.logging import get_logger

logger = get_logger(__name__)


class NDCGMetric(EvaluationMetric):
    """
    Normalized Discounted Cumulative Gain @ k.

    Measures ranking quality with position-based discounting.
    Range: 0-1, with 1 = perfect ranking.

    Standard metric for MTEB leaderboard retrieval tasks.
    """

    VERSION = "1.0.0"
    METRIC_NAME = "ndcg_at_k"
    REQUIRES_GROUND_TRUTH = True

    async def compute(
        self,
        chunks: List[str],
        embeddings: Optional[List[List[float]]] = None,
        ground_truth: Optional[Any] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> MetricResult:
        """
        Compute NDCG@k score.

        Args:
            chunks: List of text chunks
            embeddings: Chunk embeddings
            ground_truth: Dict with 'query_embedding' and 'relevant_indices'
            context: Optional context (query, etc.)

        Returns:
            MetricResult with NDCG score
        """
        self.validate_inputs(chunks, embeddings, ground_truth)

        if embeddings is None:
            raise ValidationError("NDCG requires chunk embeddings")

        if not isinstance(ground_truth, dict):
            raise ValidationError("ground_truth must be dict with query_embedding and relevant_indices")

        k = self.config.get("k", 5)

        try:
            # Get query embedding and relevant indices
            query_emb = ground_truth.get("query_embedding")
            relevant_indices = ground_truth.get("relevant_indices", [])

            if query_emb is None:
                raise ValidationError("ground_truth must contain 'query_embedding'")

            # Compute similarities between query and all chunks
            query_emb_arr = np.array(query_emb).reshape(1, -1)
            chunk_embs_arr = np.array(embeddings)

            similarities = cosine_similarity(query_emb_arr, chunk_embs_arr)[0]

            # Create relevance scores (1 for relevant, 0 for irrelevant)
            true_relevance = np.zeros(len(chunks))
            for idx in relevant_indices:
                if idx < len(chunks):
                    true_relevance[idx] = 1

            # Compute NDCG@k
            try:
                score = ndcg_score(
                    y_true=[true_relevance],
                    y_score=[similarities],
                    k=k,
                )
            except Exception as e:
                logger.warning("ndcg_computation_failed", error=str(e))
                score = 0.0

            # Get top-k retrieved indices for details
            top_k_indices = np.argsort(similarities)[::-1][:k]
            retrieved_relevant = sum(1 for idx in top_k_indices if idx in relevant_indices)

            logger.info(
                "ndcg_computed",
                k=k,
                score=score,
                retrieved_relevant=retrieved_relevant,
                total_relevant=len(relevant_indices),
            )

            return MetricResult(
                metric_name=self.METRIC_NAME,
                score=float(score),
                version=self.VERSION,
                details={
                    "k": k,
                    "retrieved_relevant": retrieved_relevant,
                    "total_relevant": len(relevant_indices),
                    "top_k_indices": top_k_indices.tolist(),
                },
            )

        except Exception as e:
            logger.error("ndcg_computation_error", error=str(e), exc_info=True)
            raise EvaluationError(f"NDCG computation failed: {str(e)}") from e


class RecallAtKMetric(EvaluationMetric):
    """
    Recall @ k metric.

    Measures completeness of retrieval: what fraction of relevant items are in top-k?
    Range: 0-1, higher = better.
    """

    VERSION = "1.0.0"
    METRIC_NAME = "recall_at_k"
    REQUIRES_GROUND_TRUTH = True

    async def compute(
        self,
        chunks: List[str],
        embeddings: Optional[List[List[float]]] = None,
        ground_truth: Optional[Any] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> MetricResult:
        """Compute Recall@k."""
        self.validate_inputs(chunks, embeddings, ground_truth)

        if embeddings is None:
            raise ValidationError("Recall@k requires chunk embeddings")

        k = self.config.get("k", 10)

        try:
            query_emb = ground_truth.get("query_embedding")
            relevant_indices = ground_truth.get("relevant_indices", [])

            # Compute similarities
            query_emb_arr = np.array(query_emb).reshape(1, -1)
            chunk_embs_arr = np.array(embeddings)
            similarities = cosine_similarity(query_emb_arr, chunk_embs_arr)[0]

            # Get top-k indices
            top_k_indices = np.argsort(similarities)[::-1][:k]

            # Calculate recall
            if len(relevant_indices) == 0:
                score = 0.0
            else:
                retrieved_relevant = sum(1 for idx in top_k_indices if idx in relevant_indices)
                score = retrieved_relevant / len(relevant_indices)

            return MetricResult(
                metric_name=self.METRIC_NAME,
                score=float(score),
                version=self.VERSION,
                details={
                    "k": k,
                    "retrieved_relevant": retrieved_relevant,
                    "total_relevant": len(relevant_indices),
                },
            )

        except Exception as e:
            raise EvaluationError(f"Recall@k computation failed: {str(e)}") from e


class PrecisionAtKMetric(EvaluationMetric):
    """
    Precision @ k metric.

    Measures accuracy of retrieval: what fraction of top-k are relevant?
    Range: 0-1, higher = better.
    """

    VERSION = "1.0.0"
    METRIC_NAME = "precision_at_k"
    REQUIRES_GROUND_TRUTH = True

    async def compute(
        self,
        chunks: List[str],
        embeddings: Optional[List[List[float]]] = None,
        ground_truth: Optional[Any] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> MetricResult:
        """Compute Precision@k."""
        self.validate_inputs(chunks, embeddings, ground_truth)

        if embeddings is None:
            raise ValidationError("Precision@k requires chunk embeddings")

        k = self.config.get("k", 5)

        try:
            query_emb = ground_truth.get("query_embedding")
            relevant_indices = ground_truth.get("relevant_indices", [])

            # Compute similarities
            query_emb_arr = np.array(query_emb).reshape(1, -1)
            chunk_embs_arr = np.array(embeddings)
            similarities = cosine_similarity(query_emb_arr, chunk_embs_arr)[0]

            # Get top-k indices
            top_k_indices = np.argsort(similarities)[::-1][:k]

            # Calculate precision
            retrieved_relevant = sum(1 for idx in top_k_indices if idx in relevant_indices)
            score = retrieved_relevant / k if k > 0 else 0.0

            return MetricResult(
                metric_name=self.METRIC_NAME,
                score=float(score),
                version=self.VERSION,
                details={
                    "k": k,
                    "retrieved_relevant": retrieved_relevant,
                },
            )

        except Exception as e:
            raise EvaluationError(f"Precision@k computation failed: {str(e)}") from e


class MRRMetric(EvaluationMetric):
    """
    Mean Reciprocal Rank.

    Measures: at what position is the first relevant item?
    Range: 0-1, with 1 = first result always correct.
    """

    VERSION = "1.0.0"
    METRIC_NAME = "mrr"
    REQUIRES_GROUND_TRUTH = True

    async def compute(
        self,
        chunks: List[str],
        embeddings: Optional[List[List[float]]] = None,
        ground_truth: Optional[Any] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> MetricResult:
        """Compute MRR."""
        self.validate_inputs(chunks, embeddings, ground_truth)

        if embeddings is None:
            raise ValidationError("MRR requires chunk embeddings")

        try:
            query_emb = ground_truth.get("query_embedding")
            relevant_indices = ground_truth.get("relevant_indices", [])

            # Compute similarities
            query_emb_arr = np.array(query_emb).reshape(1, -1)
            chunk_embs_arr = np.array(embeddings)
            similarities = cosine_similarity(query_emb_arr, chunk_embs_arr)[0]

            # Get ranked indices
            ranked_indices = np.argsort(similarities)[::-1]

            # Find rank of first relevant item
            first_relevant_rank = None
            for rank, idx in enumerate(ranked_indices, 1):
                if idx in relevant_indices:
                    first_relevant_rank = rank
                    break

            # Calculate MRR
            score = 1.0 / first_relevant_rank if first_relevant_rank else 0.0

            return MetricResult(
                metric_name=self.METRIC_NAME,
                score=float(score),
                version=self.VERSION,
                details={
                    "first_relevant_rank": first_relevant_rank,
                },
            )

        except Exception as e:
            raise EvaluationError(f"MRR computation failed: {str(e)}") from e
