"""Structured logging setup with structlog."""

import logging
import sys
from typing import Any, Dict

import structlog

from chunk_flow.core.config import get_settings


def configure_logging() -> None:
    """
    Configure structured logging for ChunkFlow.

    Sets up structlog with appropriate processors for development vs production.
    """
    settings = get_settings()

    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, settings.log_level.upper()),
    )

    # Common processors for both formats
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    # Choose renderer based on format setting
    if settings.log_format == "json":
        # Production: JSON logging for parsing by log aggregators
        processors = shared_processors + [
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ]
    else:
        # Development: Pretty console output
        processors = shared_processors + [
            structlog.dev.ConsoleRenderer(colors=True),
        ]

    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """
    Get a configured logger instance.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Structured logger instance

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("processing_started", doc_id="123", chunk_count=5)
    """
    return structlog.get_logger(name)


class LoggerMixin:
    """Mixin to add logging to classes."""

    @property
    def logger(self) -> structlog.stdlib.BoundLogger:
        """Get logger for this class."""
        if not hasattr(self, "_logger"):
            self._logger = get_logger(self.__class__.__module__)
        return self._logger


def log_function_call(func_name: str, **kwargs: Any) -> None:
    """
    Log a function call with parameters.

    Args:
        func_name: Name of the function
        **kwargs: Function parameters to log
    """
    logger = get_logger("chunk_flow.calls")
    logger.debug("function_called", function=func_name, **kwargs)


def log_performance(operation: str, duration_ms: float, **context: Any) -> None:
    """
    Log performance metrics.

    Args:
        operation: Name of operation
        duration_ms: Duration in milliseconds
        **context: Additional context
    """
    logger = get_logger("chunk_flow.performance")
    logger.info("performance", operation=operation, duration_ms=duration_ms, **context)


def log_error(error: Exception, **context: Any) -> None:
    """
    Log an error with context.

    Args:
        error: Exception that occurred
        **context: Additional context
    """
    logger = get_logger("chunk_flow.errors")
    logger.error(
        "error_occurred",
        error_type=type(error).__name__,
        error_message=str(error),
        **context,
        exc_info=True,
    )


def bind_context(**kwargs: Any) -> None:
    """
    Bind context variables that will be included in all subsequent logs.

    Useful for request IDs, user IDs, etc.

    Args:
        **kwargs: Context variables to bind

    Example:
        >>> bind_context(request_id="abc123", user_id="user456")
        >>> logger.info("processing")  # Will include request_id and user_id
    """
    structlog.contextvars.bind_contextvars(**kwargs)


def unbind_context(*keys: str) -> None:
    """
    Unbind context variables.

    Args:
        *keys: Context keys to unbind
    """
    structlog.contextvars.unbind_contextvars(*keys)


def clear_context() -> None:
    """Clear all context variables."""
    structlog.contextvars.clear_contextvars()


# Auto-configure logging on import in development
# In production, call configure_logging() explicitly
try:
    settings = get_settings()
    if settings.is_development():
        configure_logging()
except Exception:
    # Fallback if settings not available
    pass
