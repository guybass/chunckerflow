"""Factory for creating embedding providers."""

from typing import Any, Dict, List, Type

from chunk_flow.core.base import EmbeddingProvider
from chunk_flow.core.exceptions import RegistryError
from chunk_flow.utils.logging import get_logger

logger = get_logger(__name__)


class EmbeddingProviderFactory:
    """
    Factory for creating embedding providers.

    Provides provider discovery, registration, and instantiation.
    """

    _providers: Dict[str, Type[EmbeddingProvider]] = {}

    @classmethod
    def register(cls, name: str, provider_class: Type[EmbeddingProvider]) -> None:
        """
        Register an embedding provider.

        Args:
            name: Provider name
            provider_class: Provider class to register

        Raises:
            RegistryError: If provider already registered
        """
        if name in cls._providers:
            logger.warning("provider_already_registered", name=name)
            return  # Allow re-registration

        cls._providers[name] = provider_class
        logger.info(
            "provider_registered",
            name=name,
            version=provider_class.VERSION,
        )

    @classmethod
    def unregister(cls, name: str) -> None:
        """
        Unregister a provider.

        Args:
            name: Provider name to unregister
        """
        if name in cls._providers:
            del cls._providers[name]
            logger.info("provider_unregistered", name=name)

    @classmethod
    def get(cls, name: str) -> Type[EmbeddingProvider]:
        """
        Get provider class by name.

        Args:
            name: Provider name

        Returns:
            Provider class

        Raises:
            RegistryError: If provider not found
        """
        if name not in cls._providers:
            available = ", ".join(cls._providers.keys())
            raise RegistryError(
                f"Unknown embedding provider: {name}. Available: {available}"
            )
        return cls._providers[name]

    @classmethod
    def create(
        cls, name: str, config: Dict[str, Any] | None = None
    ) -> EmbeddingProvider:
        """
        Create provider instance.

        Args:
            name: Provider name
            config: Provider configuration

        Returns:
            Provider instance

        Raises:
            RegistryError: If provider not found
        """
        provider_class = cls.get(name)
        return provider_class(config=config)

    @classmethod
    def list_providers(cls) -> List[str]:
        """
        List all registered providers.

        Returns:
            List of provider names
        """
        return list(cls._providers.keys())

    @classmethod
    def get_provider_info(cls, name: str) -> Dict[str, Any]:
        """
        Get provider information.

        Args:
            name: Provider name

        Returns:
            Provider info dictionary
        """
        provider_class = cls.get(name)
        temp_instance = provider_class()  # Create temp instance
        return temp_instance.get_info()

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """
        Check if provider is registered.

        Args:
            name: Provider name

        Returns:
            True if registered, False otherwise
        """
        return name in cls._providers

    @classmethod
    def clear(cls) -> None:
        """Clear all registered providers (useful for testing)."""
        cls._providers.clear()
        logger.info("provider_registry_cleared")


# Auto-register built-in providers
def _register_builtin_providers() -> None:
    """Register all built-in providers."""
    try:
        # Register OpenAI
        try:
            from chunk_flow.embeddings.providers.openai_provider import (
                OpenAIEmbeddingProvider,
            )

            EmbeddingProviderFactory.register("openai", OpenAIEmbeddingProvider)
        except ImportError as e:
            logger.warning("openai_provider_not_available", error=str(e))

        # Register HuggingFace
        try:
            from chunk_flow.embeddings.providers.huggingface_provider import (
                HuggingFaceEmbeddingProvider,
            )

            EmbeddingProviderFactory.register("huggingface", HuggingFaceEmbeddingProvider)
        except ImportError as e:
            logger.warning("huggingface_provider_not_available", error=str(e))

        logger.info(
            "builtin_providers_registered",
            count=len(EmbeddingProviderFactory.list_providers()),
        )

    except Exception as e:
        logger.error("provider_registration_failed", error=str(e), exc_info=True)


# Register on import
_register_builtin_providers()
