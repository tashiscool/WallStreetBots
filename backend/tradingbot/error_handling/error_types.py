"""Custom Error Types for Trading System.

Defines specific error types for different failure scenarios in the trading system.
"""

from typing import Any


class TradingError(Exception):
    """Base exception for all trading - related errors."""

    def __init__(
        self,
        message: str,
        error_code: str | None = None,
        context: dict[str, Any] | None = None,
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.context = context or {}
        self.timestamp = None  # Will be set by error handler


class DataProviderError(TradingError):
    """Error related to data provider failures."""

    def __init__(
        self,
        message: str,
        provider: str | None = None,
        context: dict[str, Any] | None = None,
    ):
        super().__init__(message, "DATA_PROVIDER_ERROR", context)
        self.provider = provider


class BrokerConnectionError(TradingError):
    """Error related to broker API connection issues."""

    def __init__(
        self,
        message: str,
        broker: str | None = None,
        context: dict[str, Any] | None = None,
    ):
        super().__init__(message, "BROKER_CONNECTION_ERROR", context)
        self.broker = broker


class InsufficientFundsError(TradingError):
    """Error when account has insufficient funds for a trade."""

    def __init__(
        self,
        message: str,
        required_amount: float | None = None,
        available_amount: float | None = None,
        context: dict[str, Any] | None = None,
    ):
        super().__init__(message, "INSUFFICIENT_FUNDS_ERROR", context)
        self.required_amount = required_amount
        self.available_amount = available_amount


class PositionReconciliationError(TradingError):
    """Critical error when position reconciliation fails."""

    def __init__(
        self,
        message: str,
        discrepancies: list | None = None,
        context: dict[str, Any] | None = None,
    ):
        super().__init__(message, "POSITION_RECONCILIATION_ERROR", context)
        self.discrepancies = discrepancies or []


class RiskLimitExceededError(TradingError):
    """Error when risk limits are exceeded."""

    def __init__(
        self,
        message: str,
        limit_type: str | None = None,
        current_value: float | None = None,
        limit_value: float | None = None,
        context: dict[str, Any] | None = None,
    ):
        super().__init__(message, "RISK_LIMIT_EXCEEDED_ERROR", context)
        self.limit_type = limit_type
        self.current_value = current_value
        self.limit_value = limit_value


class OrderExecutionError(TradingError):
    """Error during order execution."""

    def __init__(
        self,
        message: str,
        order_id: str | None = None,
        ticker: str | None = None,
        context: dict[str, Any] | None = None,
    ):
        super().__init__(message, "ORDER_EXECUTION_ERROR", context)
        self.order_id = order_id
        self.ticker = ticker


class MarketDataError(TradingError):
    """Error related to market data issues."""

    def __init__(
        self,
        message: str,
        ticker: str | None = None,
        data_type: str | None = None,
        context: dict[str, Any] | None = None,
    ):
        super().__init__(message, "MARKET_DATA_ERROR", context)
        self.ticker = ticker
        self.data_type = data_type
