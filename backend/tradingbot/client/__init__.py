"""
API Client Module.

REST API client library for WallStreetBots.
"""

from .api_client import (
    ApiError,
    AuthenticationError,
    RateLimitError,
    NotFoundError,
    ValidationError,
    ApiResponse,
    ClientConfig,
    HttpMethod,
    TradingApiClient,
    SyncTradingApiClient,
    create_client,
)

__all__ = [
    # Exceptions
    "ApiError",
    "AuthenticationError",
    "RateLimitError",
    "NotFoundError",
    "ValidationError",
    # Response
    "ApiResponse",
    # Config
    "ClientConfig",
    "HttpMethod",
    # Clients
    "TradingApiClient",
    "SyncTradingApiClient",
    "create_client",
]
