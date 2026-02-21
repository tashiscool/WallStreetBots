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
        from .models import Portfolio, Bot, Company, Stock, Order, TradeTransaction
        DJANGO_MODELS_AVAILABLE = True
        return Portfolio, Bot, Company, Stock, Order, TradeTransaction
    except ImportError:
        return None, None, None, None, None, None

# Django models - only import when explicitly requested
Portfolio = Bot = Company = Stock = Order = TradeTransaction = None
DJANGO_MODELS_AVAILABLE = False

# Check if Django is available and configured
try:
    import django
    from django.conf import settings
    if settings.configured:
        try:
            from .models import (
                Portfolio, Bot, Company, Stock, Order,
                TradeTransaction,
                ValidationRun, SignalValidationMetrics, DataQualityMetrics,
                ValidationParameterRegistry, TradeSignalSnapshot,
                SignalValidationHistory, StrategyAllocationLimit,
                AllocationReservation, CircuitBreakerEvent,
                CircuitBreakerState, CircuitBreakerHistory,
                StrategyPortfolio, UserProfile, DigestLog,
                TaxLot, TaxLotSale, StrategyPerformanceSnapshot,
                CustomStrategy
            )
            DJANGO_MODELS_AVAILABLE = True
        except ImportError:
            pass
except ImportError:
    pass

__all__ = [
    "DJANGO_MODELS_AVAILABLE",
    "AllocationReservation",
    "Bot",
    "CircuitBreakerEvent",
    "CircuitBreakerHistory",
    "CircuitBreakerState",
    "Company",
    "CustomStrategy",
    "DataQualityMetrics",
    "DigestLog",
    "Order",
    "OrderSide",
    "Portfolio",
    "ProductionAlert",
    "ProductionEarningsData",
    "ProductionMarketData",
    "ProductionPerformanceMetrics",
    "ProductionPosition",
    "ProductionRiskLimit",
    "ProductionSentimentData",
    "ProductionStrategy",
    "ProductionTrade",
    "SignalValidationHistory",
    "SignalValidationMetrics",
    "Stock",
    "StrategyAllocationLimit",
    "StrategyPerformanceSnapshot",
    "StrategyPortfolio",
    "StrategyRiskLevel",
    "TaxLot",
    "TaxLotSale",
    "TradeSignalSnapshot",
    "TradeStatus",
    "TradeTransaction",
    "UserProfile",
    "ValidationParameterRegistry",
    "ValidationRun",
    "get_django_models",
]
