"""Trading bot models.

This module contains all models for the trading bot system.
"""

# Import production models (always available)
from .production_models import (
    StrategyRiskLevel,
    TradeStatus,
    OrderSide,
    Strategy as ProductionStrategy,
    Position as ProductionPosition,
    Trade as ProductionTrade,
    RiskLimit as ProductionRiskLimit,
    Alert as ProductionAlert,
    MarketData as ProductionMarketData,
    EarningsData as ProductionEarningsData,
    SentimentData as ProductionSentimentData,
    PerformanceMetrics as ProductionPerformanceMetrics,
)

# Django models are imported separately when needed
# to avoid initialization issues outside Django context
DJANGO_MODELS_AVAILABLE = False

def get_django_models():
    """Import Django models when needed."""
    global DJANGO_MODELS_AVAILABLE
    try:
        from .models import Portfolio, Bot, Company, Stock, Order
        DJANGO_MODELS_AVAILABLE = True
        return Portfolio, Bot, Company, Stock, Order
    except ImportError:
        return None, None, None, None, None

# Django models - only import when explicitly requested
Portfolio = Bot = Company = Stock = Order = None
DJANGO_MODELS_AVAILABLE = False

# Check if Django is available and configured
try:
    import django
    from django.conf import settings
    if settings.configured:
        try:
            from .models import Portfolio, Bot, Company, Stock, Order
            DJANGO_MODELS_AVAILABLE = True
        except ImportError:
            pass
except ImportError:
    pass

__all__ = [
    # Production models (always available)
    "StrategyRiskLevel",
    "TradeStatus",
    "OrderSide",
    "ProductionStrategy",
    "ProductionPosition",
    "ProductionTrade",
    "ProductionRiskLimit",
    "ProductionAlert",
    "ProductionMarketData",
    "ProductionEarningsData",
    "ProductionSentimentData",
    "ProductionPerformanceMetrics",

    # Django models (available when Django is configured)
    "Portfolio",
    "Bot",
    "Company",
    "Stock",
    "Order",
    "get_django_models",
    "DJANGO_MODELS_AVAILABLE",
]