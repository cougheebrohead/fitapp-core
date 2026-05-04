class FitAppCoreError(Exception):
    """Base exception for fitapp_core."""


class ConfigError(FitAppCoreError):
    """Raised when an API key or required config is missing."""


class ProviderError(FitAppCoreError):
    """Raised when an upstream provider (OFF, USDA, AI) returns an error."""
