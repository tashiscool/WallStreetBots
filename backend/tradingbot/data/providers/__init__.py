"""Data providers and clients.

This module contains the data providers and clients for fetching
market data from various sources.
"""

from .client import MarketDataClient, BarSpec
from .corporate_actions import CorporateActionsAdjuster

__all__ = [
    "BarSpec",
    "CorporateActionsAdjuster",
    "MarketDataClient",
]
