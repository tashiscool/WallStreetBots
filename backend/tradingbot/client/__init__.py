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
    # Response
    "ApiResponse",
    "AuthenticationError",
    # Config
    "ClientConfig",
    "HttpMethod",
    "NotFoundError",
    "RateLimitError",
    "SyncTradingApiClient",
    # Clients
    "TradingApiClient",
    "ValidationError",
    "create_client",
]
