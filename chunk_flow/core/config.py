"""Configuration management with Pydantic settings."""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict

from chunk_flow.core.exceptions import ConfigurationError


class ChunkFlowSettings(BaseSettings):
    """
    ChunkFlow configuration loaded from environment variables and .env file.

    Environment variables take precedence over .env file.
    Prefix all env vars with CHUNK_FLOW_ (e.g., CHUNK_FLOW_LOG_LEVEL=DEBUG)
    """

    # Application settings
    env: str = "development"
    debug: bool = False

    # Logging
    log_level: str = "INFO"
    log_format: str = "json"  # json or console

    # API settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 4
    api_reload: bool = False

    # Cache settings
    cache_enabled: bool = True
    cache_ttl: int = 3600
    redis_url: Optional[str] = None

    # Processing settings
    max_concurrent_tasks: int = 10
    request_timeout: int = 300
    batch_size: int = 100

    # API Keys (loaded from environment only)
    openai_api_key: Optional[str] = None
    cohere_api_key: Optional[str] = None
    voyage_api_key: Optional[str] = None
    jina_api_key: Optional[str] = None
    google_cloud_project: Optional[str] = None
    google_application_credentials: Optional[str] = None

    # Database (optional)
    database_url: Optional[str] = None

    # Monitoring
    metrics_enabled: bool = True
    tracing_enabled: bool = False

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="CHUNK_FLOW_",
        case_sensitive=False,
        extra="allow",  # Allow extra fields for extensibility
    )

    def get_api_key(self, provider: str) -> str:
        """
        Get API key for a provider.

        Args:
            provider: Provider name (openai, cohere, voyage, jina)

        Returns:
            API key

        Raises:
            ConfigurationError: If API key not found
        """
        key_attr = f"{provider.lower()}_api_key"
        api_key = getattr(self, key_attr, None)

        if not api_key:
            raise ConfigurationError(
                f"API key for {provider} not found. "
                f"Set CHUNK_FLOW_{provider.upper()}_API_KEY environment variable."
            )

        return api_key

    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.env.lower() == "production"

    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.env.lower() == "development"


class ConfigLoader:
    """Load configuration from YAML files."""

    def __init__(self, config_dir: str = "config") -> None:
        """
        Initialize config loader.

        Args:
            config_dir: Directory containing config files
        """
        self.config_dir = Path(config_dir)
        self._config: Dict[str, Any] = {}
        self._load_default()

    def _load_default(self) -> None:
        """Load default configuration."""
        default_path = self.config_dir / "default.yaml"
        if default_path.exists():
            with open(default_path) as f:
                self._config = yaml.safe_load(f) or {}

    def load_env_config(self, env: str) -> None:
        """
        Load environment-specific configuration.

        Args:
            env: Environment name (development, production, test)
        """
        env_path = self.config_dir / f"{env}.yaml"
        if env_path.exists():
            with open(env_path) as f:
                env_config = yaml.safe_load(f) or {}
                self._deep_merge(self._config, env_config)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value with dot notation.

        Args:
            key: Configuration key (e.g., "strategies.recursive.chunk_size")
            default: Default value if key not found

        Returns:
            Configuration value

        Example:
            >>> config = ConfigLoader()
            >>> chunk_size = config.get("strategies.recursive.chunk_size", 512)
        """
        keys = key.split(".")
        value = self._config

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default

            if value is None:
                return default

        return value

    def get_strategy_config(self, strategy_name: str) -> Dict[str, Any]:
        """
        Get configuration for a chunking strategy.

        Args:
            strategy_name: Name of the strategy

        Returns:
            Strategy configuration dictionary
        """
        return self.get(f"strategies.{strategy_name}", {})

    def get_embedding_config(self, provider_name: str) -> Dict[str, Any]:
        """
        Get configuration for an embedding provider.

        Args:
            provider_name: Name of the provider

        Returns:
            Provider configuration dictionary
        """
        return self.get(f"embeddings.{provider_name}", {})

    def get_metric_config(self, metric_name: str) -> Dict[str, Any]:
        """
        Get configuration for an evaluation metric.

        Args:
            metric_name: Name of the metric

        Returns:
            Metric configuration dictionary
        """
        return self.get(f"metrics.{metric_name}", {})

    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> None:
        """
        Deep merge override into base dictionary.

        Args:
            base: Base dictionary to merge into
            override: Dictionary with override values
        """
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value


# Global configuration instances
_settings: Optional[ChunkFlowSettings] = None
_config_loader: Optional[ConfigLoader] = None


def get_settings() -> ChunkFlowSettings:
    """
    Get global settings instance (singleton).

    Returns:
        ChunkFlowSettings instance
    """
    global _settings
    if _settings is None:
        _settings = ChunkFlowSettings()
    return _settings


def get_config_loader(config_dir: str = "config") -> ConfigLoader:
    """
    Get global config loader instance (singleton).

    Args:
        config_dir: Configuration directory path

    Returns:
        ConfigLoader instance
    """
    global _config_loader
    if _config_loader is None:
        _config_loader = ConfigLoader(config_dir)
        # Load environment-specific config
        env = os.getenv("CHUNK_FLOW_ENV", "development")
        _config_loader.load_env_config(env)
    return _config_loader


def reset_config() -> None:
    """Reset global configuration (useful for testing)."""
    global _settings, _config_loader
    _settings = None
    _config_loader = None
