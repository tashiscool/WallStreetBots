"""Request validation schemas using Pydantic for API endpoints.

Provides type-safe validation for all trading API requests with:
- Automatic type coercion
- Clear validation error messages
- Serialization/deserialization support
"""

import re
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class OrderSide(str, Enum):
    """Valid order sides."""
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    """Valid order types."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class TimeInForce(str, Enum):
    """Valid time-in-force values."""
    DAY = "day"
    GTC = "gtc"  # Good 'til cancelled
    IOC = "ioc"  # Immediate or cancel
    FOK = "fok"  # Fill or kill


class OptionType(str, Enum):
    """Valid option types."""
    CALL = "call"
    PUT = "put"


# Regex patterns for validation
SYMBOL_PATTERN = re.compile(r"^[A-Z]{1,5}$")
OPTION_SYMBOL_PATTERN = re.compile(r"^[A-Z]{1,5}\d{6}[CP]\d{8}$")


class StockOrderRequest(BaseModel):
    """Validated stock order request."""

    symbol: str = Field(..., min_length=1, max_length=5, description="Stock symbol")
    side: OrderSide = Field(..., description="Buy or sell")
    quantity: int = Field(..., gt=0, le=10000, description="Number of shares")
    order_type: OrderType = Field(default=OrderType.MARKET, description="Order type")
    limit_price: Optional[Decimal] = Field(
        default=None, gt=0, description="Limit price for limit orders"
    )
    stop_price: Optional[Decimal] = Field(
        default=None, gt=0, description="Stop price for stop orders"
    )
    time_in_force: TimeInForce = Field(
        default=TimeInForce.DAY, description="Time in force"
    )
    extended_hours: bool = Field(
        default=False, description="Allow extended hours trading"
    )

    @field_validator("symbol")
    @classmethod
    def validate_symbol(cls, v: str) -> str:
        v = v.upper().strip()
        if not SYMBOL_PATTERN.match(v):
            raise ValueError(
                f"Invalid symbol format: {v}. Must be 1-5 uppercase letters."
            )
        return v

    @model_validator(mode="after")
    def validate_prices(self):
        """Validate limit/stop prices based on order type."""
        if self.order_type in (OrderType.LIMIT, OrderType.STOP_LIMIT):
            if self.limit_price is None:
                raise ValueError(
                    f"limit_price is required for {self.order_type.value} orders"
                )
        if self.order_type in (OrderType.STOP, OrderType.STOP_LIMIT):
            if self.stop_price is None:
                raise ValueError(
                    f"stop_price is required for {self.order_type.value} orders"
                )
        return self


class OptionOrderRequest(BaseModel):
    """Validated option order request."""

    symbol: str = Field(..., description="Option contract symbol (OCC format)")
    side: OrderSide = Field(..., description="Buy or sell")
    quantity: int = Field(
        ..., gt=0, le=100, description="Number of contracts (max 100)"
    )
    order_type: OrderType = Field(default=OrderType.LIMIT, description="Order type")
    limit_price: Optional[Decimal] = Field(
        default=None, gt=0, description="Limit price per contract"
    )
    time_in_force: TimeInForce = Field(
        default=TimeInForce.DAY, description="Time in force"
    )

    @field_validator("symbol")
    @classmethod
    def validate_option_symbol(cls, v: str) -> str:
        v = v.upper().strip()
        if not OPTION_SYMBOL_PATTERN.match(v):
            raise ValueError(
                f"Invalid option symbol format: {v}. Must be OCC format "
                "(e.g., AAPL240119C00150000)"
            )
        return v

    @model_validator(mode="after")
    def validate_option_order(self):
        """Options should generally use limit orders."""
        if self.order_type == OrderType.MARKET:
            # Market orders on options can be dangerous
            pass  # Allow but log warning
        elif self.order_type in (OrderType.LIMIT, OrderType.STOP_LIMIT):
            if self.limit_price is None:
                raise ValueError(
                    f"limit_price required for {self.order_type.value} orders"
                )
        return self


class PositionCloseRequest(BaseModel):
    """Request to close a position."""

    symbol: str = Field(..., description="Symbol to close")
    quantity: Optional[int] = Field(
        default=None, gt=0, description="Partial close quantity (None for full close)"
    )

    @field_validator("symbol")
    @classmethod
    def validate_symbol(cls, v: str) -> str:
        return v.upper().strip()


class WatchlistRequest(BaseModel):
    """Request to manage watchlist."""

    symbols: List[str] = Field(
        ..., min_length=1, max_length=50, description="Symbols to add/remove"
    )
    action: str = Field(
        ..., pattern="^(add|remove)$", description="Action: add or remove"
    )

    @field_validator("symbols")
    @classmethod
    def validate_symbols(cls, v: List[str]) -> List[str]:
        validated = []
        for symbol in v:
            s = symbol.upper().strip()
            if not SYMBOL_PATTERN.match(s):
                raise ValueError(f"Invalid symbol: {symbol}")
            validated.append(s)
        return validated


class StrategyConfigRequest(BaseModel):
    """Request to configure a trading strategy."""

    strategy_name: str = Field(..., min_length=1, max_length=50)
    enabled: bool = Field(default=True)
    max_position_size: Decimal = Field(
        default=Decimal("0.05"),
        gt=0,
        le=Decimal("0.25"),
        description="Max position as fraction of portfolio (0-0.25)",
    )
    risk_tolerance: str = Field(
        default="medium",
        pattern="^(low|medium|high)$",
        description="Risk tolerance level",
    )
    parameters: dict = Field(
        default_factory=dict, description="Strategy-specific parameters"
    )

    @field_validator("strategy_name")
    @classmethod
    def validate_strategy_name(cls, v: str) -> str:
        v = v.lower().strip().replace("-", "_")
        valid_strategies = {
            "wsb_dip_bot",
            "wheel_strategy",
            "momentum_weeklies",
            "earnings_protection",
            "debit_spreads",
            "leaps_tracker",
            "lotto_scanner",
            "swing_trading",
            "spx_credit_spreads",
            "index_baseline",
        }
        if v not in valid_strategies:
            raise ValueError(
                f"Unknown strategy: {v}. Valid: {', '.join(sorted(valid_strategies))}"
            )
        return v


class AlertConfigRequest(BaseModel):
    """Request to configure alerts."""

    alert_type: str = Field(
        ..., pattern="^(price|volume|news|earnings|risk)$"
    )
    symbol: Optional[str] = Field(default=None, description="Symbol (if applicable)")
    threshold: Optional[Decimal] = Field(
        default=None, description="Threshold value (if applicable)"
    )
    enabled: bool = Field(default=True)

    @field_validator("symbol")
    @classmethod
    def validate_symbol(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        v = v.upper().strip()
        if not SYMBOL_PATTERN.match(v):
            raise ValueError(f"Invalid symbol: {v}")
        return v


class BacktestRequest(BaseModel):
    """Request to run a backtest."""

    strategy_name: str = Field(..., description="Strategy to backtest")
    start_date: datetime = Field(..., description="Backtest start date")
    end_date: datetime = Field(..., description="Backtest end date")
    initial_capital: Decimal = Field(
        default=Decimal("100000"),
        gt=1000,
        le=10000000,
        description="Starting capital",
    )
    symbols: Optional[List[str]] = Field(
        default=None, description="Symbols to test (None for strategy default)"
    )

    @model_validator(mode="after")
    def validate_dates(self):
        if self.end_date <= self.start_date:
            raise ValueError("end_date must be after start_date")
        if (self.end_date - self.start_date).days > 365 * 5:
            raise ValueError("Backtest period cannot exceed 5 years")
        return self

    @field_validator("symbols")
    @classmethod
    def validate_symbols(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        if v is None:
            return None
        return [s.upper().strip() for s in v]


def validate_request(model_class: type[BaseModel], data: dict) -> BaseModel:
    """Validate request data against a Pydantic model.

    Args:
        model_class: The Pydantic model class to validate against
        data: Dictionary of request data

    Returns:
        Validated model instance

    Raises:
        ValueError: If validation fails with details about the errors
    """
    try:
        return model_class.model_validate(data)
    except Exception as e:
        # Re-raise with cleaner error message for API response
        raise ValueError(f"Validation failed: {e}")
