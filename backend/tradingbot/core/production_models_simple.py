"""Production Database Models - Simple Version
Works without Django dependencies
"""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal


@dataclass
class Strategy:
    """Trading strategy configuration"""

    name: str
    description: str = ""
    risk_level: str = "medium"
    status: str = "active"
    max_position_risk: Decimal = Decimal("0.10")
    max_total_risk: Decimal = Decimal("0.30")
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def __str__(self):
        return f"{self.name} ({self.risk_level})"


@dataclass
class Position:
    """Open trading position"""

    id: int | None = None
    strategy: Strategy | None = None
    ticker: str = ""
    quantity: Decimal = Decimal("0")
    entry_price: Decimal = Decimal("0")
    current_price: Decimal = Decimal("0")
    unrealized_pnl: Decimal = Decimal("0")
    realized_pnl: Decimal = Decimal("0")
    opened_at: datetime = field(default_factory=datetime.now)
    closed_at: datetime | None = None
    is_open: bool = True

    def __str__(self):
        return f"{self.ticker} ({self.quantity})"


@dataclass
class Trade:
    """Individual trade record"""

    id: int | None = None
    strategy: Strategy | None = None
    trade_id: str = ""
    ticker: str = ""
    side: str = ""  # 'buy' or 'sell'
    order_type: str = "market"
    quantity: int = 0
    limit_price: Decimal | None = None
    stop_price: Decimal | None = None
    status: str = "pending"
    filled_price: Decimal | None = None
    filled_quantity: int = 0
    commission: Decimal = Decimal("0")
    created_at: datetime = field(default_factory=datetime.now)
    filled_at: datetime | None = None

    def __str__(self):
        return f"{self.side} {self.quantity} {self.ticker} ({self.status})"


@dataclass
class RiskLimit:
    """Risk management limits"""

    id: int | None = None
    strategy: Strategy | None = None
    max_daily_loss_pct: Decimal = Decimal("0.01")  # 1%
    max_position_size_pct: Decimal = Decimal("0.05")  # 5%
    max_drawdown_pct: Decimal = Decimal("0.10")  # 10%
    max_open_positions: int = 5
    updated_at: datetime = field(default_factory=datetime.now)

    def __str__(self):
        return f"Risk Limits for {self.strategy.name if self.strategy else 'Unknown'}"


@dataclass
class MarketData:
    """Market data record"""

    id: int | None = None
    ticker: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    open_price: Decimal = Decimal("0")
    high_price: Decimal = Decimal("0")
    low_price: Decimal = Decimal("0")
    close_price: Decimal = Decimal("0")
    volume: int = 0

    def __str__(self):
        return f"{self.ticker} - {self.timestamp.isoformat()}"


@dataclass
class EarningsData:
    """Earnings data record"""

    id: int | None = None
    ticker: str = ""
    report_date: datetime = field(default_factory=datetime.now)
    eps_estimate: Decimal | None = None
    eps_actual: Decimal | None = None
    revenue_estimate: Decimal | None = None
    revenue_actual: Decimal | None = None
    surprise_pct: Decimal | None = None
    fiscal_period: str | None = None

    def __str__(self):
        return f"{self.ticker} Earnings on {self.report_date.date()}"


@dataclass
class SentimentData:
    """Sentiment data record"""

    id: int | None = None
    ticker: str = ""
    source: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    sentiment_score: Decimal = Decimal("0")
    sentiment_magnitude: Decimal | None = None
    headline: str | None = None
    url: str | None = None

    def __str__(self):
        return f"{self.ticker} Sentiment: {self.sentiment_score}"


@dataclass
class StrategyPerformance:
    """Strategy performance metrics"""

    id: int | None = None
    strategy: Strategy | None = None
    date: datetime = field(default_factory=datetime.now)
    total_return: Decimal = Decimal("0")
    daily_return: Decimal = Decimal("0")
    sharpe_ratio: Decimal = Decimal("0")
    max_drawdown: Decimal = Decimal("0")
    win_rate: Decimal = Decimal("0")
    profit_factor: Decimal = Decimal("0")
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0

    def __str__(self):
        return f"{self.strategy.name if self.strategy else 'Unknown'} Performance: {self.total_return:.2%}"
