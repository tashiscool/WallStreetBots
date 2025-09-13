"""Production Data Integration

Live data providers and integration components:
- ProductionDataProvider: Real - time market data provider
"""

from .production_data_integration import ReliableDataProvider as ProductionDataProvider

__all__ = [
    "ProductionDataProvider",
]
