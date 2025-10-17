"""Registry for chunking strategies."""

from typing import Any, Dict, List, Type

from chunk_flow.core.base import ChunkingStrategy
from chunk_flow.core.exceptions import RegistryError
from chunk_flow.core.models import StrategyInfo
from chunk_flow.utils.logging import get_logger

logger = get_logger(__name__)


class StrategyRegistry:
    """
    Registry for all chunking strategies.

    Provides strategy discovery, registration, and instantiation.
    """

    _strategies: Dict[str, Type[ChunkingStrategy]] = {}

    @classmethod
    def register(cls, name: str, strategy_class: Type[ChunkingStrategy]) -> None:
        """
        Register a chunking strategy.

        Args:
            name: Strategy name (must match strategy_class.NAME)
            strategy_class: Strategy class to register

        Raises:
            RegistryError: If strategy already registered or name mismatch
        """
        if name in cls._strategies:
            logger.warning("strategy_already_registered", name=name)
            return  # Allow re-registration for hot reload

        if hasattr(strategy_class, "NAME") and strategy_class.NAME != name:
            raise RegistryError(
                f"Strategy name mismatch: {name} != {strategy_class.NAME}"
            )

        cls._strategies[name] = strategy_class
        logger.info("strategy_registered", name=name, version=strategy_class.VERSION)

    @classmethod
    def unregister(cls, name: str) -> None:
        """
        Unregister a strategy.

        Args:
            name: Strategy name to unregister
        """
        if name in cls._strategies:
            del cls._strategies[name]
            logger.info("strategy_unregistered", name=name)

    @classmethod
    def get(cls, name: str) -> Type[ChunkingStrategy]:
        """
        Get strategy class by name.

        Args:
            name: Strategy name

        Returns:
            Strategy class

        Raises:
            RegistryError: If strategy not found
        """
        if name not in cls._strategies:
            available = ", ".join(cls._strategies.keys())
            raise RegistryError(
                f"Unknown strategy: {name}. Available strategies: {available}"
            )
        return cls._strategies[name]

    @classmethod
    def create(cls, name: str, config: Dict[str, Any] | None = None) -> ChunkingStrategy:
        """
        Create strategy instance.

        Args:
            name: Strategy name
            config: Strategy configuration

        Returns:
            Strategy instance

        Raises:
            RegistryError: If strategy not found
        """
        strategy_class = cls.get(name)
        return strategy_class(config=config)

    @classmethod
    def list_strategies(cls) -> List[StrategyInfo]:
        """
        List all registered strategies with metadata.

        Returns:
            List of StrategyInfo objects
        """
        infos: List[StrategyInfo] = []

        for name, strategy_class in cls._strategies.items():
            # Create temp instance to get info (with default config)
            temp_instance = strategy_class()
            infos.append(temp_instance.get_info())

        return infos

    @classmethod
    def get_strategy_names(cls) -> List[str]:
        """
        Get list of registered strategy names.

        Returns:
            List of strategy names
        """
        return list(cls._strategies.keys())

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """
        Check if strategy is registered.

        Args:
            name: Strategy name

        Returns:
            True if registered, False otherwise
        """
        return name in cls._strategies

    @classmethod
    def clear(cls) -> None:
        """Clear all registered strategies (useful for testing)."""
        cls._strategies.clear()
        logger.info("strategy_registry_cleared")


# Auto-register built-in strategies
def _register_builtin_strategies() -> None:
    """Register all built-in strategies."""
    count = 0

    try:
        from chunk_flow.chunking.strategies.fixed_size import FixedSizeChunker
        from chunk_flow.chunking.strategies.recursive import RecursiveCharacterChunker
        from chunk_flow.chunking.strategies.document_based import (
            MarkdownChunker,
            HTMLChunker,
        )

        StrategyRegistry.register("fixed_size", FixedSizeChunker)
        StrategyRegistry.register("recursive", RecursiveCharacterChunker)
        StrategyRegistry.register("markdown", MarkdownChunker)
        StrategyRegistry.register("html", HTMLChunker)
        count += 4

    except ImportError as e:
        logger.warning("basic_strategy_registration_failed", error=str(e))

    # Register advanced strategies (optional dependencies)
    try:
        from chunk_flow.chunking.strategies.semantic import SemanticChunker
        StrategyRegistry.register("semantic", SemanticChunker)
        count += 1
    except ImportError:
        logger.debug("semantic_chunker_not_available")

    try:
        from chunk_flow.chunking.strategies.late import LateChunker
        StrategyRegistry.register("late", LateChunker)
        count += 1
    except ImportError:
        logger.debug("late_chunker_not_available")

    logger.info("builtin_strategies_registered", count=count)


# Register on import
_register_builtin_strategies()
