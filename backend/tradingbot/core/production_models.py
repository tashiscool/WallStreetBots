"""
Production Database Models
Simple dataclass - based models that don't require Django
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any
from decimal import Decimal
from enum import Enum


class StrategyRiskLevel(Enum): 
    LOW = "low"
    MEDIUM = "medium" 
    HIGH = "high"


class TradeStatus(Enum): 
    PENDING = "pending"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class OrderSide(Enum): 
    BUY = "buy"
    SELL = "sell"


@dataclass
class Strategy: 
    """Trading strategy configuration"""
    id: Optional[int] = None
    name: str = ""
    description: str = ""
    risk_level: StrategyRiskLevel = StrategyRiskLevel.LOW
    status: str = "active"
    max_position_risk: Decimal = Decimal('0.02')
    max_account_risk: Decimal = Decimal('0.10')
    created_at: datetime = field(default_factory  =  datetime.now)
    updated_at: datetime = field(default_factory  =  datetime.now)


@dataclass  
class Position: 
    """Current position in a security"""
    id: Optional[int] = None
    strategy_id: Optional[int] = None
    ticker: str = ""
    quantity: int = 0
    avg_entry_price: Decimal = Decimal('0.00')
    current_price: Decimal = Decimal('0.00')
    market_value: Decimal = Decimal('0.00')
    unrealized_pnl: Decimal = Decimal('0.00')
    realized_pnl: Decimal = Decimal('0.00')
    total_pnl: Decimal = Decimal('0.00')
    risk_amount: Decimal = Decimal('0.00')
    entry_date: datetime = field(default_factory  =  datetime.now)
    last_update: datetime = field(default_factory  =  datetime.now)
    metadata: Dict[str, Any] = field(default_factory = dict)


@dataclass
class Trade: 
    """Individual trade execution record"""
    id: Optional[int] = None
    strategy_id: Optional[int] = None
    trade_id: str = ""
    ticker: str = ""
    side: OrderSide = OrderSide.BUY
    quantity: int = 0
    entry_price: Decimal = Decimal('0.00')
    exit_price: Optional[Decimal] = None
    pnl: Optional[Decimal] = None
    commission: Decimal = Decimal('0.00')
    slippage: Decimal = Decimal('0.00')
    status: TradeStatus = TradeStatus.PENDING
    entry_time: datetime = field(default_factory  =  datetime.now)
    exit_time: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory = dict)


@dataclass
class RiskLimit: 
    """Risk management limits for strategies"""
    id: Optional[int] = None
    strategy_id: Optional[int] = None
    max_position_size: Decimal = Decimal('0.02')
    max_total_risk: Decimal = Decimal('0.10')
    max_daily_loss: Decimal = Decimal('0.05')
    max_drawdown: Decimal = Decimal('0.20')
    stop_loss_pct: Decimal = Decimal('0.10')
    take_profit_pct: Decimal = Decimal('0.20')
    created_at: datetime = field(default_factory  =  datetime.now)


@dataclass
class Alert: 
    """System alerts and notifications"""
    id: Optional[int] = None
    timestamp: datetime = field(default_factory  =  datetime.now)
    level: str = "info"  # info, warning, error, critical
    message: str = ""
    source: str = ""
    metadata: Dict[str, Any] = field(default_factory = dict)
    acknowledged: bool = False


# Simple in - memory storage for demonstration
# In production, these would be stored in PostgreSQL
_strategies: List[Strategy] = []
_positions: List[Position] = []
_trades: List[Trade] = []
_risk_limits: List[RiskLimit] = []
_alerts: List[Alert] = []


def create_strategy(strategy: Strategy)->Strategy:
    """Create a new strategy"""
    strategy.id = len(_strategies) + 1
    _strategies.append(strategy)
    return strategy


def get_strategy(strategy_id: int)->Optional[Strategy]:
    """Get strategy by ID"""
    for strategy in _strategies: 
        if strategy.id ==  strategy_id: 
            return strategy
    return None


def create_position(position: Position)->Position:
    """Create a new position"""
    position.id = len(_positions) + 1
    _positions.append(position)
    return position


def get_positions(strategy_id: Optional[int] = None)->List[Position]:
    """Get positions, optionally filtered by strategy"""
    if strategy_id: 
        return [p for p in _positions if p.strategy_id ==  strategy_id]
    return _positions.copy()


def create_trade(trade: Trade)->Trade:
    """Create a new trade"""
    trade.id = len(_trades) + 1
    _trades.append(trade)
    return trade


def get_trades(strategy_id: Optional[int] = None)->List[Trade]:
    """Get trades, optionally filtered by strategy"""
    if strategy_id: 
        return [t for t in _trades if t.strategy_id ==  strategy_id]
    return _trades.copy()


def create_alert(alert: Alert)->Alert:
    """Create a new alert"""
    alert.id = len(_alerts) + 1
    _alerts.append(alert)
    return alert


def get_alerts(level: Optional[str] = None)->List[Alert]:
    """Get alerts, optionally filtered by level"""
    if level: 
        return [a for a in _alerts if a.level ==  level]
    return _alerts.copy()


@dataclass
class PerformanceMetrics: 
    """Strategy performance metrics"""
    id: Optional[int] = None
    strategy_id: Optional[int] = None
    strategy_name: str = ""
    period_start: datetime = field(default_factory  =  datetime.now)
    period_end: datetime = field(default_factory  =  datetime.now)
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: Decimal = Decimal('0.00')
    total_pnl: Decimal = Decimal('0.00')
    gross_profit: Decimal = Decimal('0.00')
    gross_loss: Decimal = Decimal('0.00')
    profit_factor: Decimal = Decimal('0.00')
    avg_win: Decimal = Decimal('0.00')
    avg_loss: Decimal = Decimal('0.00')
    largest_win: Decimal = Decimal('0.00')
    largest_loss: Decimal = Decimal('0.00')
    max_drawdown: Decimal = Decimal('0.00')
    sharpe_ratio: Decimal = Decimal('0.00')
    sortino_ratio: Decimal = Decimal('0.00')
    calmar_ratio: Decimal = Decimal('0.00')
    kelly_fraction: Decimal = Decimal('0.00')
    var_95: Decimal = Decimal('0.00')
    expected_shortfall: Decimal = Decimal('0.00')
    created_at: datetime = field(default_factory  =  datetime.now)
    updated_at: datetime = field(default_factory  =  datetime.now)