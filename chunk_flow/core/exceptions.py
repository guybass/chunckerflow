"""Custom exceptions for ChunkFlow."""


class ChunkFlowError(Exception):
    """Base exception for all ChunkFlow errors."""

    pass


class ConfigurationError(ChunkFlowError):
    """Raised when configuration is invalid or missing required fields."""

    pass


class ChunkingError(ChunkFlowError):
    """Raised when chunking operation fails."""

    pass


class EmbeddingError(ChunkFlowError):
    """Raised when embedding generation fails."""

    pass


class EvaluationError(ChunkFlowError):
    """Raised when evaluation computation fails."""

    pass


class VersionCompatibilityError(ChunkFlowError):
    """Raised when component versions are incompatible."""

    pass


class RegistryError(ChunkFlowError):
    """Raised when registry operations fail (e.g., unknown strategy name)."""

    pass


class ValidationError(ChunkFlowError):
    """Raised when input validation fails."""

    pass


class ProviderError(ChunkFlowError):
    """Raised when embedding provider encounters an error."""

    pass


class APIError(ChunkFlowError):
    """Raised when external API calls fail."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        """
        Initialize API error.

        Args:
            message: Error message
            status_code: HTTP status code if applicable
        """
        super().__init__(message)
        self.status_code = status_code


class RateLimitError(APIError):
    """Raised when API rate limit is exceeded."""

    pass


class AuthenticationError(APIError):
    """Raised when API authentication fails."""

    pass
