"""Abstract base classes for ChunkFlow components."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from chunk_flow.core.exceptions import ConfigurationError, ValidationError
from chunk_flow.core.models import ChunkResult, EmbeddingResult, MetricResult, StrategyInfo
from chunk_flow.core.version import Versioned


class ChunkingStrategy(Versioned, ABC):
    """
    Abstract base class for all chunking strategies.

    Subclasses must implement:
    - chunk(): The main chunking logic
    - _validate_config(): Configuration validation
    - get_default_config(): Default configuration

    Version compatibility is handled automatically via Versioned mixin.
    """

    VERSION = "1.0.0"
    NAME = "base"

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize chunking strategy.

        Args:
            config: Strategy configuration

        Raises:
            ConfigurationError: If configuration is invalid
        """
        self.config = config or self.get_default_config()
        self._validate_config()

    @abstractmethod
    async def chunk(self, text: str, doc_id: Optional[str] = None) -> ChunkResult:
        """
        Chunk text into segments.

        Args:
            text: Input text to chunk
            doc_id: Optional document identifier for tracking

        Returns:
            ChunkResult containing chunks and metadata

        Raises:
            ChunkingError: If chunking fails
            ValidationError: If input is invalid
        """
        pass

    @abstractmethod
    def _validate_config(self) -> None:
        """
        Validate configuration parameters.

        Raises:
            ConfigurationError: If configuration is invalid
        """
        pass

    @abstractmethod
    def get_default_config(self) -> Dict[str, Any]:
        """
        Get default configuration for this strategy.

        Returns:
            Dictionary of default configuration parameters
        """
        pass

    def get_info(self) -> StrategyInfo:
        """
        Get strategy information.

        Returns:
            StrategyInfo containing metadata about this strategy
        """
        return StrategyInfo(
            name=self.NAME,
            version=self.VERSION,
            description=self.__doc__ or "",
            default_config=self.get_default_config(),
            required_config=self._get_required_config(),
        )

    def _get_required_config(self) -> List[str]:
        """
        Get list of required configuration parameters.

        Override in subclasses to specify required parameters.

        Returns:
            List of required parameter names
        """
        return []

    def validate_input(self, text: str) -> None:
        """
        Validate input text.

        Args:
            text: Input text to validate

        Raises:
            ValidationError: If input is invalid
        """
        if not isinstance(text, str):
            raise ValidationError(f"Input must be string, got {type(text)}")
        if not text or not text.strip():
            raise ValidationError("Input text cannot be empty")


class EmbeddingProvider(Versioned, ABC):
    """
    Abstract base class for embedding providers.

    Subclasses must implement:
    - embed_texts(): Batch embedding generation
    - embed_query(): Single query embedding
    - _initialize(): Provider initialization (API keys, model loading, etc.)
    """

    VERSION = "1.0.0"
    PROVIDER_NAME = "base"

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize embedding provider.

        Args:
            config: Provider configuration

        Raises:
            ConfigurationError: If configuration is invalid
        """
        self.config = config or {}
        self._validate_config()
        self._initialize()

    @abstractmethod
    async def embed_texts(self, texts: List[str]) -> EmbeddingResult:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            EmbeddingResult containing embeddings and metadata

        Raises:
            EmbeddingError: If embedding generation fails
            ValidationError: If input is invalid
        """
        pass

    @abstractmethod
    async def embed_query(self, query: str) -> EmbeddingResult:
        """
        Generate embedding for a single query.

        May use different parameters than embed_texts() for retrieval optimization.

        Args:
            query: Query text to embed

        Returns:
            EmbeddingResult containing single embedding

        Raises:
            EmbeddingError: If embedding generation fails
            ValidationError: If input is invalid
        """
        pass

    @abstractmethod
    def _initialize(self) -> None:
        """
        Initialize provider (load models, set up API clients, etc.).

        Called automatically during __init__.

        Raises:
            ConfigurationError: If initialization fails
        """
        pass

    def _validate_config(self) -> None:
        """
        Validate provider configuration.

        Override in subclasses to add validation.

        Raises:
            ConfigurationError: If configuration is invalid
        """
        pass

    def get_info(self) -> Dict[str, Any]:
        """
        Get provider information.

        Returns:
            Dictionary containing provider metadata
        """
        return {
            "provider": self.PROVIDER_NAME,
            "version": self.VERSION,
            "config": self._get_public_config(),
        }

    def _get_public_config(self) -> Dict[str, Any]:
        """
        Get public configuration (excluding secrets).

        Returns:
            Dictionary of non-sensitive configuration
        """
        sensitive_keys = {"api_key", "secret_key", "token", "password"}
        return {
            k: v
            for k, v in self.config.items()
            if not any(sensitive in k.lower() for sensitive in sensitive_keys)
        }

    def validate_inputs(self, texts: List[str]) -> None:
        """
        Validate input texts.

        Args:
            texts: Texts to validate

        Raises:
            ValidationError: If inputs are invalid
        """
        if not texts:
            raise ValidationError("Input list cannot be empty")
        if not all(isinstance(t, str) for t in texts):
            raise ValidationError("All inputs must be strings")
        if any(not t.strip() for t in texts):
            raise ValidationError("Input texts cannot be empty or whitespace only")


class EvaluationMetric(Versioned, ABC):
    """
    Abstract base class for evaluation metrics.

    Subclasses must implement:
    - compute(): Metric computation logic
    """

    VERSION = "1.0.0"
    METRIC_NAME = "base"
    REQUIRES_GROUND_TRUTH = False

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize evaluation metric.

        Args:
            config: Metric configuration
        """
        self.config = config or {}

    @abstractmethod
    async def compute(
        self,
        chunks: List[str],
        embeddings: Optional[List[List[float]]] = None,
        ground_truth: Optional[Any] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> MetricResult:
        """
        Compute metric score.

        Args:
            chunks: List of text chunks
            embeddings: Optional embeddings for chunks
            ground_truth: Optional ground truth data
            context: Optional additional context (query, document, etc.)

        Returns:
            MetricResult containing score and details

        Raises:
            EvaluationError: If computation fails
            ValidationError: If required inputs are missing
        """
        pass

    def get_info(self) -> Dict[str, Any]:
        """
        Get metric information.

        Returns:
            Dictionary containing metric metadata
        """
        return {
            "name": self.METRIC_NAME,
            "version": self.VERSION,
            "requires_ground_truth": self.REQUIRES_GROUND_TRUTH,
            "description": self.__doc__ or "",
        }

    def validate_inputs(
        self,
        chunks: List[str],
        embeddings: Optional[List[List[float]]] = None,
        ground_truth: Optional[Any] = None,
    ) -> None:
        """
        Validate metric inputs.

        Args:
            chunks: Chunks to validate
            embeddings: Optional embeddings to validate
            ground_truth: Optional ground truth to validate

        Raises:
            ValidationError: If inputs are invalid
        """
        if not chunks:
            raise ValidationError("Chunks list cannot be empty")

        if self.REQUIRES_GROUND_TRUTH and ground_truth is None:
            raise ValidationError(f"Metric {self.METRIC_NAME} requires ground truth data")

        if embeddings is not None and len(embeddings) != len(chunks):
            raise ValidationError("Number of embeddings must match number of chunks")


class TrainableChunkingStrategy(ChunkingStrategy, ABC):
    """
    Abstract base class for ML-based trainable chunking strategies.

    Extends ChunkingStrategy with training capabilities.
    """

    VERSION = "1.0.0-ml"

    def __init__(
        self, config: Optional[Dict[str, Any]] = None, model_path: Optional[str] = None
    ) -> None:
        """
        Initialize trainable strategy.

        Args:
            config: Strategy configuration
            model_path: Path to pre-trained model
        """
        super().__init__(config)
        self.model = self._load_model(model_path)

    @abstractmethod
    def _load_model(self, path: Optional[str]) -> Any:
        """
        Load pre-trained model.

        Args:
            path: Path to model file, or None to use default

        Returns:
            Loaded model object

        Raises:
            ConfigurationError: If model loading fails
        """
        pass

    @abstractmethod
    async def train(
        self, training_data: List[Dict[str, Any]], **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Train or fine-tune the model.

        Args:
            training_data: Training dataset
            **kwargs: Additional training parameters

        Returns:
            Dictionary containing training metrics and results

        Raises:
            TrainingError: If training fails
        """
        pass

    def save_model(self, path: str) -> None:
        """
        Save trained model to disk.

        Args:
            path: Path to save model

        Raises:
            IOError: If save fails
        """
        raise NotImplementedError("Subclasses must implement save_model()")

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.

        Returns:
            Dictionary containing model metadata
        """
        return {
            "model_type": self.__class__.__name__,
            "version": self.VERSION,
            "is_trained": hasattr(self, "model") and self.model is not None,
        }
