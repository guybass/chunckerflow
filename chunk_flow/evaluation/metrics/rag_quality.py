"""RAG-specific quality metrics (RAGAS-inspired)."""

from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from chunk_flow.core.base import EvaluationMetric
from chunk_flow.core.exceptions import EvaluationError, ValidationError
from chunk_flow.core.models import MetricResult
from chunk_flow.utils.logging import get_logger

logger = get_logger(__name__)


class ContextRelevanceMetric(EvaluationMetric):
    """
    Context relevance for RAG systems.

    Measures how relevant retrieved chunks are to the query.
    RAGAS-inspired metric: Evaluates if retrieved context is pertinent.

    Range: 0-1, with 1 = all retrieved chunks highly relevant to query.

    Ground truth format:
    {
        "query_embedding": [...],
        "retrieved_indices": [0, 2, 5]  # Indices of retrieved chunks
    }
    """

    VERSION = "1.0.0"
    METRIC_NAME = "context_relevance"
    REQUIRES_GROUND_TRUTH = True

    async def compute(
        self,
        chunks: List[str],
        embeddings: Optional[List[List[float]]] = None,
        ground_truth: Optional[Any] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> MetricResult:
        """
        Compute context relevance.

        Args:
            chunks: List of text chunks
            embeddings: Chunk embeddings
            ground_truth: Dict with query_embedding and retrieved_indices
            context: Optional context

        Returns:
            MetricResult with relevance score
        """
        self.validate_inputs(chunks, embeddings, ground_truth)

        if embeddings is None:
            raise ValidationError("Context relevance requires chunk embeddings")

        if not isinstance(ground_truth, dict):
            raise ValidationError("ground_truth must be dict with query_embedding and retrieved_indices")

        try:
            query_emb = ground_truth.get("query_embedding")
            retrieved_indices = ground_truth.get("retrieved_indices", [])

            if query_emb is None:
                raise ValidationError("ground_truth must contain 'query_embedding'")

            if not retrieved_indices:
                # No chunks retrieved = zero relevance
                return MetricResult(
                    metric_name=self.METRIC_NAME,
                    score=0.0,
                    version=self.VERSION,
                    details={"num_retrieved": 0, "message": "No chunks retrieved"},
                )

            # Compute similarity between query and retrieved chunks
            query_emb_arr = np.array(query_emb).reshape(1, -1)
            chunk_embs_arr = np.array(embeddings)

            similarities = cosine_similarity(query_emb_arr, chunk_embs_arr)[0]

            # Get similarities for retrieved chunks only
            retrieved_similarities = [similarities[i] for i in retrieved_indices if i < len(chunks)]

            if not retrieved_similarities:
                return MetricResult(
                    metric_name=self.METRIC_NAME,
                    score=0.0,
                    version=self.VERSION,
                    details={"num_retrieved": 0, "message": "No valid retrieved indices"},
                )

            # Average similarity = context relevance
            relevance_score = float(np.mean(retrieved_similarities))
            min_relevance = float(np.min(retrieved_similarities))
            max_relevance = float(np.max(retrieved_similarities))

            logger.info(
                "context_relevance_computed",
                score=relevance_score,
                num_retrieved=len(retrieved_similarities),
                min=min_relevance,
                max=max_relevance,
            )

            return MetricResult(
                metric_name=self.METRIC_NAME,
                score=relevance_score,
                version=self.VERSION,
                details={
                    "num_retrieved": len(retrieved_similarities),
                    "min_relevance": min_relevance,
                    "max_relevance": max_relevance,
                    "retrieved_similarities": retrieved_similarities,
                },
            )

        except Exception as e:
            logger.error("context_relevance_error", error=str(e), exc_info=True)
            raise EvaluationError(f"Context relevance computation failed: {str(e)}") from e


class AnswerFaithfulnessMetric(EvaluationMetric):
    """
    Answer faithfulness metric (simplified RAGAS).

    Measures if the generated answer is faithful to the retrieved context.
    In chunking evaluation context: measures how well a chunk supports its neighbors.

    Range: 0-1, with 1 = perfect faithfulness.

    This is a simplified proxy: true faithfulness requires LLM-based evaluation.
    Here we measure chunk self-containment vs context dependence.

    Ground truth format:
    {
        "chunk_index": 3,  # Index of chunk to evaluate
        "context_indices": [2, 4]  # Surrounding chunks for context
    }
    """

    VERSION = "1.0.0"
    METRIC_NAME = "answer_faithfulness"
    REQUIRES_GROUND_TRUTH = False  # Can work without ground truth (simplified mode)

    async def compute(
        self,
        chunks: List[str],
        embeddings: Optional[List[List[float]]] = None,
        ground_truth: Optional[Any] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> MetricResult:
        """
        Compute answer faithfulness.

        Args:
            chunks: List of text chunks
            embeddings: Chunk embeddings
            ground_truth: Optional dict with specific chunk to evaluate
            context: Optional context

        Returns:
            MetricResult with faithfulness score
        """
        self.validate_inputs(chunks, embeddings, ground_truth)

        if embeddings is None:
            raise ValidationError("Answer faithfulness requires chunk embeddings")

        try:
            # Simplified faithfulness: measure chunk self-similarity vs neighbor similarity
            # Self-contained chunks = high faithfulness
            # Context-dependent chunks = lower faithfulness

            if len(chunks) <= 1:
                return MetricResult(
                    metric_name=self.METRIC_NAME,
                    score=1.0,
                    version=self.VERSION,
                    details={"num_chunks": len(chunks), "message": "Single chunk, perfect faithfulness"},
                )

            emb_array = np.array(embeddings)
            similarities = cosine_similarity(emb_array)

            # For each chunk, compare self-embedding strength vs neighbor dependency
            faithfulness_scores = []

            for i in range(len(chunks)):
                # Get similarity to neighbors
                neighbor_sims = []
                if i > 0:
                    neighbor_sims.append(similarities[i, i - 1])
                if i < len(chunks) - 1:
                    neighbor_sims.append(similarities[i, i + 1])

                if neighbor_sims:
                    avg_neighbor_sim = np.mean(neighbor_sims)
                    # Faithfulness = how independent the chunk is (1 - neighbor dependency)
                    faithfulness = 1.0 - avg_neighbor_sim
                else:
                    faithfulness = 1.0  # Isolated chunk = perfect faithfulness

                faithfulness_scores.append(float(faithfulness))

            avg_faithfulness = float(np.mean(faithfulness_scores))
            min_faithfulness = float(np.min(faithfulness_scores))
            max_faithfulness = float(np.max(faithfulness_scores))

            logger.info(
                "answer_faithfulness_computed",
                avg=avg_faithfulness,
                min=min_faithfulness,
                max=max_faithfulness,
            )

            return MetricResult(
                metric_name=self.METRIC_NAME,
                score=avg_faithfulness,
                version=self.VERSION,
                details={
                    "min_faithfulness": min_faithfulness,
                    "max_faithfulness": max_faithfulness,
                    "per_chunk_scores": faithfulness_scores,
                },
            )

        except Exception as e:
            logger.error("answer_faithfulness_error", error=str(e), exc_info=True)
            raise EvaluationError(f"Answer faithfulness computation failed: {str(e)}") from e


class ContextPrecisionMetric(EvaluationMetric):
    """
    Context precision for RAG (RAGAS-inspired).

    Measures: Are the top-ranked chunks relevant?
    Important for RAG: You want relevant chunks at the top of retrieval results.

    Range: 0-1, with 1 = all top chunks are relevant.

    Ground truth format:
    {
        "query_embedding": [...],
        "relevant_indices": [1, 3, 5],  # Ground truth relevant chunks
        "k": 5  # Top-k to evaluate
    }
    """

    VERSION = "1.0.0"
    METRIC_NAME = "context_precision"
    REQUIRES_GROUND_TRUTH = True

    async def compute(
        self,
        chunks: List[str],
        embeddings: Optional[List[List[float]]] = None,
        ground_truth: Optional[Any] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> MetricResult:
        """
        Compute context precision.

        Args:
            chunks: List of text chunks
            embeddings: Chunk embeddings
            ground_truth: Dict with query_embedding and relevant_indices
            context: Optional context

        Returns:
            MetricResult with precision score
        """
        self.validate_inputs(chunks, embeddings, ground_truth)

        if embeddings is None:
            raise ValidationError("Context precision requires chunk embeddings")

        if not isinstance(ground_truth, dict):
            raise ValidationError("ground_truth must be dict")

        try:
            query_emb = ground_truth.get("query_embedding")
            relevant_indices = set(ground_truth.get("relevant_indices", []))
            k = ground_truth.get("k", 5)

            if query_emb is None:
                raise ValidationError("ground_truth must contain 'query_embedding'")

            # Rank chunks by similarity to query
            query_emb_arr = np.array(query_emb).reshape(1, -1)
            chunk_embs_arr = np.array(embeddings)
            similarities = cosine_similarity(query_emb_arr, chunk_embs_arr)[0]

            # Get top-k
            top_k_indices = np.argsort(similarities)[::-1][:k]

            # Compute precision at each position
            precisions = []
            for i, idx in enumerate(top_k_indices, 1):
                if idx in relevant_indices:
                    # How many relevant in top-i?
                    relevant_in_top_i = sum(1 for j in top_k_indices[:i] if j in relevant_indices)
                    precisions.append(relevant_in_top_i / i)

            # Context precision = average of precisions at relevant positions
            if precisions:
                score = float(np.mean(precisions))
            else:
                score = 0.0

            logger.info(
                "context_precision_computed",
                score=score,
                k=k,
                num_relevant_in_topk=sum(1 for idx in top_k_indices if idx in relevant_indices),
            )

            return MetricResult(
                metric_name=self.METRIC_NAME,
                score=score,
                version=self.VERSION,
                details={
                    "k": k,
                    "top_k_indices": top_k_indices.tolist(),
                    "relevant_in_topk": sum(1 for idx in top_k_indices if idx in relevant_indices),
                    "precisions_at_relevant": precisions,
                },
            )

        except Exception as e:
            logger.error("context_precision_error", error=str(e), exc_info=True)
            raise EvaluationError(f"Context precision computation failed: {str(e)}") from e


class ContextRecallMetric(EvaluationMetric):
    """
    Context recall for RAG (RAGAS-inspired).

    Measures: What fraction of relevant chunks are retrieved in top-k?
    Complementary to precision.

    Range: 0-1, with 1 = all relevant chunks retrieved.

    Ground truth format:
    {
        "query_embedding": [...],
        "relevant_indices": [1, 3, 5],  # Ground truth relevant chunks
        "k": 10  # Top-k to retrieve
    }
    """

    VERSION = "1.0.0"
    METRIC_NAME = "context_recall"
    REQUIRES_GROUND_TRUTH = True

    async def compute(
        self,
        chunks: List[str],
        embeddings: Optional[List[List[float]]] = None,
        ground_truth: Optional[Any] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> MetricResult:
        """
        Compute context recall.

        Args:
            chunks: List of text chunks
            embeddings: Chunk embeddings
            ground_truth: Dict with query_embedding and relevant_indices
            context: Optional context

        Returns:
            MetricResult with recall score
        """
        self.validate_inputs(chunks, embeddings, ground_truth)

        if embeddings is None:
            raise ValidationError("Context recall requires chunk embeddings")

        if not isinstance(ground_truth, dict):
            raise ValidationError("ground_truth must be dict")

        try:
            query_emb = ground_truth.get("query_embedding")
            relevant_indices = set(ground_truth.get("relevant_indices", []))
            k = ground_truth.get("k", 10)

            if query_emb is None:
                raise ValidationError("ground_truth must contain 'query_embedding'")

            if not relevant_indices:
                return MetricResult(
                    metric_name=self.METRIC_NAME,
                    score=0.0,
                    version=self.VERSION,
                    details={"message": "No relevant chunks specified"},
                )

            # Rank chunks by similarity
            query_emb_arr = np.array(query_emb).reshape(1, -1)
            chunk_embs_arr = np.array(embeddings)
            similarities = cosine_similarity(query_emb_arr, chunk_embs_arr)[0]

            # Get top-k
            top_k_indices = set(np.argsort(similarities)[::-1][:k].tolist())

            # Recall = relevant in top-k / total relevant
            relevant_retrieved = len(top_k_indices & relevant_indices)
            score = relevant_retrieved / len(relevant_indices)

            logger.info(
                "context_recall_computed",
                score=score,
                k=k,
                relevant_retrieved=relevant_retrieved,
                total_relevant=len(relevant_indices),
            )

            return MetricResult(
                metric_name=self.METRIC_NAME,
                score=float(score),
                version=self.VERSION,
                details={
                    "k": k,
                    "relevant_retrieved": relevant_retrieved,
                    "total_relevant": len(relevant_indices),
                },
            )

        except Exception as e:
            logger.error("context_recall_error", error=str(e), exc_info=True)
            raise EvaluationError(f"Context recall computation failed: {str(e)}") from e
