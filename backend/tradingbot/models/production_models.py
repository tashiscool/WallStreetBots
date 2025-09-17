"""Consolidated Production Database Models
Unified dataclass-based models for the trading system.
Supports both Django and standalone operation.
"""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any


class StrategyRiskLevel(Enum):
    """Risk levels for trading strategies."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class TradeStatus(Enum):
    """Status of trade execution."""
    PENDING = "pending"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class OrderSide(Enum):
    """Side of trade order."""
    BUY = "buy"
    SELL = "sell"


@dataclass
class Strategy:
    """Trading strategy configuration."""

    name: str
    description: str = ""
    risk_level: StrategyRiskLevel = StrategyRiskLevel.MEDIUM
    status: str = "active"
    max_position_risk: Decimal = Decimal("0.05")  # 5%
    max_account_risk: Decimal = Decimal("0.20")   # 20%
    max_total_risk: Decimal = Decimal("0.30")     # 30%
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    id: int | None = None

    def __str__(self):
        return f"{self.name} ({self.risk_level.value})"


@dataclass
class Position:
    """Current position in a security."""

    ticker: str = ""
    quantity: Decimal = Decimal("0")
    entry_price: Decimal = Decimal("0.00")
    current_price: Decimal = Decimal("0.00")
    market_value: Decimal = Decimal("0.00")
    unrealized_pnl: Decimal = Decimal("0.00")
    realized_pnl: Decimal = Decimal("0.00")
    total_pnl: Decimal = Decimal("0.00")
    risk_amount: Decimal = Decimal("0.00")
    entry_date: datetime = field(default_factory=datetime.now)
    last_update: datetime = field(default_factory=datetime.now)
    opened_at: datetime = field(default_factory=datetime.now)
    closed_at: datetime | None = None
    is_open: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)
    id: int | None = None
    strategy_id: int | None = None
    strategy: Strategy | None = None

    def __str__(self):
        return f"{self.ticker} ({self.quantity})"


@dataclass
class Trade:
    """Individual trade execution record."""

    ticker: str = ""
    side: OrderSide = OrderSide.BUY
    quantity: int = 0
    entry_price: Decimal = Decimal("0.00")
    exit_price: Decimal | None = None
    filled_price: Decimal | None = None
    limit_price: Decimal | None = None
    stop_price: Decimal | None = None
    pnl: Decimal | None = None
    commission: Decimal = Decimal("0.00")
    slippage: Decimal = Decimal("0.00")
    status: TradeStatus = TradeStatus.PENDING
    order_type: str = "market"
    entry_time: datetime = field(default_factory=datetime.now)
    exit_time: datetime | None = None
    filled_time: datetime | None = None
    transaction_time: datetime = field(default_factory=datetime.now)
    created_at: datetime = field(default_factory=datetime.now)
    filled_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    id: int | None = None
    strategy_id: int | None = None
    strategy: Strategy | None = None
    trade_id: str = ""
    broker_order_id: str | None = None
    filled_quantity: int = 0

    def __str__(self):
        return f"{self.side.value} {self.quantity} {self.ticker} ({self.status.value})"


@dataclass
class RiskLimit:
    """Risk management limits for strategies."""

    max_position_size: Decimal = Decimal("0.05")      # 5%
    max_total_risk: Decimal = Decimal("0.20")         # 20%
    max_daily_loss: Decimal = Decimal("0.02")         # 2%
    max_daily_loss_pct: Decimal = Decimal("0.02")     # 2%
    max_position_size_pct: Decimal = Decimal("0.05")  # 5%
    max_drawdown: Decimal = Decimal("0.15")           # 15%
    max_drawdown_pct: Decimal = Decimal("0.15")       # 15%
    stop_loss_pct: Decimal = Decimal("0.10")          # 10%
    take_profit_pct: Decimal = Decimal("0.20")        # 20%
    max_open_positions: int = 5
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    id: int | None = None
    strategy_id: int | None = None
    strategy: Strategy | None = None

    def __str__(self):
        return f"Risk Limits for {self.strategy.name if self.strategy else 'Unknown'}"


@dataclass
class Alert:
    """System alerts and notifications."""

    timestamp: datetime = field(default_factory=datetime.now)
    level: str = "info"  # info, warning, error, critical
    message: str = ""
    source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    acknowledged: bool = False
    id: int | None = None

    def __str__(self):
        return f"[{self.level.upper()}] {self.message}"


@dataclass
class MarketData:
    """Market data record."""

    ticker: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    open_price: Decimal = Decimal("0")
    high_price: Decimal = Decimal("0")
    low_price: Decimal = Decimal("0")
    close_price: Decimal = Decimal("0")
    volume: int = 0
    id: int | None = None

    def __str__(self):
        return f"{self.ticker} - {self.timestamp.isoformat()}"


@dataclass
class EarningsData:
    """Earnings data record."""

    ticker: str = ""
    report_date: datetime = field(default_factory=datetime.now)
    eps_estimate: Decimal | None = None
    eps_actual: Decimal | None = None
    revenue_estimate: Decimal | None = None
    revenue_actual: Decimal | None = None
    surprise_pct: Decimal | None = None
    fiscal_period: str | None = None
    id: int | None = None

    def __str__(self):
        return f"{self.ticker} Earnings on {self.report_date.date()}"


@dataclass
class SentimentData:
    """Sentiment data record."""

    ticker: str = ""
    source: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    sentiment_score: Decimal = Decimal("0")
    sentiment_magnitude: Decimal | None = None
    headline: str | None = None
    url: str | None = None
    id: int | None = None

    def __str__(self):
        return f"{self.ticker} Sentiment: {self.sentiment_score}"


@dataclass
class PerformanceMetrics:
    """Strategy performance metrics."""

    strategy_name: str = ""
    period_start: datetime = field(default_factory=datetime.now)
    period_end: datetime = field(default_factory=datetime.now)
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: Decimal = Decimal("0.00")
    total_pnl: Decimal = Decimal("0.00")
    total_return: Decimal = Decimal("0.00")
    daily_return: Decimal = Decimal("0.00")
    gross_profit: Decimal = Decimal("0.00")
    gross_loss: Decimal = Decimal("0.00")
    profit_factor: Decimal = Decimal("0.00")
    avg_win: Decimal = Decimal("0.00")
    avg_loss: Decimal = Decimal("0.00")
    largest_win: Decimal = Decimal("0.00")
    largest_loss: Decimal = Decimal("0.00")
    max_drawdown: Decimal = Decimal("0.00")
    sharpe_ratio: Decimal = Decimal("0.00")
    sortino_ratio: Decimal = Decimal("0.00")
    calmar_ratio: Decimal = Decimal("0.00")
    kelly_fraction: Decimal = Decimal("0.00")
    var_95: Decimal = Decimal("0.00")
    expected_shortfall: Decimal = Decimal("0.00")
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    date: datetime = field(default_factory=datetime.now)
    id: int | None = None
    strategy_id: int | None = None
    strategy: Strategy | None = None

    def __str__(self):
        return f"{self.strategy_name} Performance: {self.total_return:.2%}"


# Simple in-memory storage for demonstration
# In production, these would be stored in PostgreSQL
_strategies: list[Strategy] = []
_positions: list[Position] = []
_trades: list[Trade] = []
_risk_limits: list[RiskLimit] = []
_alerts: list[Alert] = []
_performance_metrics: list[PerformanceMetrics] = []


def create_strategy(strategy: Strategy) -> Strategy:
    """Create a new strategy."""
    strategy.id = len(_strategies) + 1
    _strategies.append(strategy)
    return strategy


def get_strategy(strategy_id: int) -> Strategy | None:
    """Get strategy by ID."""
    for strategy in _strategies:
        if strategy.id == strategy_id:
            return strategy
    return None


def create_position(position: Position) -> Position:
    """Create a new position."""
    position.id = len(_positions) + 1
    _positions.append(position)
    return position


def get_positions(strategy_id: int | None = None) -> list[Position]:
    """Get positions, optionally filtered by strategy."""
    if strategy_id:
        return [p for p in _positions if p.strategy_id == strategy_id]
    return _positions.copy()


def create_trade(trade: Trade) -> Trade:
    """Create a new trade."""
    trade.id = len(_trades) + 1
    _trades.append(trade)
    return trade


def get_trades(strategy_id: int | None = None) -> list[Trade]:
    """Get trades, optionally filtered by strategy."""
    if strategy_id:
        return [t for t in _trades if t.strategy_id == strategy_id]
    return _trades.copy()


def create_alert(alert: Alert) -> Alert:
    """Create a new alert."""
    alert.id = len(_alerts) + 1
    _alerts.append(alert)
    return alert


def get_alerts(level: str | None = None) -> list[Alert]:
    """Get alerts, optionally filtered by level."""
    if level:
        return [a for a in _alerts if a.level == level]
    return _alerts.copy()


def create_performance_metrics(metrics: PerformanceMetrics) -> PerformanceMetrics:
    """Create new performance metrics."""
    metrics.id = len(_performance_metrics) + 1
    _performance_metrics.append(metrics)
    return metrics


def get_performance_metrics(strategy_id: int | None = None) -> list[PerformanceMetrics]:
    """Get performance metrics, optionally filtered by strategy."""
    if strategy_id:
        return [m for m in _performance_metrics if m.strategy_id == strategy_id]
    return _performance_metrics.copy()