"""
Production Database Models - Dataclass Version
Fallback when Django is not available
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from decimal import Decimal
from datetime import datetime


@dataclass
class Strategy: 
    """Trading strategy configuration"""
    name: str
    description: str=""
    risk_level: str="medium"
    status: str="active"
    created_at: datetime=field(default_factory=datetime.now)
    updated_at: datetime=field(default_factory=datetime.now)
    
    def __str__(self): 
        return self.name


@dataclass
class Position: 
    """Open trading position"""
    id: Optional[int] = None
    strategy: Optional[Strategy] = None
    ticker: str=""
    quantity: Decimal=Decimal('0')
    entry_price: Decimal=Decimal('0')
    current_price: Decimal=Decimal('0')
    unrealized_pnl: Decimal=Decimal('0')
    realized_pnl: Decimal=Decimal('0')
    opened_at: datetime=field(default_factory=datetime.now)
    closed_at: Optional[datetime] = None
    is_open: bool = True
    
    def __str__(self): 
        return f"{self.ticker} ({self.quantity})"


@dataclass
class Trade: 
    """Individual trade record"""
    id: Optional[int] = None
    strategy: Optional[Strategy] = None
    ticker: str=""
    side: str=""  # 'buy' or 'sell'
    quantity: Decimal=Decimal('0')
    price: Decimal=Decimal('0')
    status: str="pending"
    broker_order_id: Optional[str] = None
    transaction_time: datetime=field(default_factory=datetime.now)
    filled_time: Optional[datetime] = None
    commission: Decimal=Decimal('0')
    
    def __str__(self): 
        return f"{self.side} {self.quantity} {self.ticker} ({self.status})"


@dataclass
class RiskLimit: 
    """Risk management limits"""
    id: Optional[int] = None
    strategy: Optional[Strategy] = None
    max_daily_loss_pct: Decimal=Decimal('0.01')  # 1%
    max_position_size_pct: Decimal=Decimal('0.05')  # 5%
    max_drawdown_pct: Decimal=Decimal('0.10')  # 10%
    max_open_positions: int = 5
    updated_at: datetime=field(default_factory=datetime.now)
    
    def __str__(self): 
        return f"Risk Limits for {self.strategy.name if self.strategy else 'Unknown'}"


@dataclass
class MarketData: 
    """Market data record"""
    id: Optional[int] = None
    ticker: str=""
    timestamp: datetime=field(default_factory=datetime.now)
    open_price: Decimal=Decimal('0')
    high_price: Decimal=Decimal('0')
    low_price: Decimal=Decimal('0')
    close_price: Decimal=Decimal('0')
    volume: int = 0
    
    def __str__(self): 
        return f"{self.ticker} - {self.timestamp.isoformat()}"


@dataclass
class EarningsData: 
    """Earnings data record"""
    id: Optional[int] = None
    ticker: str=""
    report_date: datetime=field(default_factory=datetime.now)
    eps_estimate: Optional[Decimal] = None
    eps_actual: Optional[Decimal] = None
    revenue_estimate: Optional[Decimal] = None
    revenue_actual: Optional[Decimal] = None
    surprise_pct: Optional[Decimal] = None
    fiscal_period: Optional[str] = None
    
    def __str__(self): 
        return f"{self.ticker} Earnings on {self.report_date.date()}"


@dataclass
class SentimentData: 
    """Sentiment data record"""
    id: Optional[int] = None
    ticker: str=""
    source: str=""
    timestamp: datetime=field(default_factory=datetime.now)
    sentiment_score: Decimal=Decimal('0')
    sentiment_magnitude: Optional[Decimal] = None
    headline: Optional[str] = None
    url: Optional[str] = None
    
    def __str__(self): 
        return f"{self.ticker} Sentiment: {self.sentiment_score}"


@dataclass
class StrategyPerformance: 
    """Strategy performance metrics"""
    id: Optional[int] = None
    strategy: Optional[Strategy] = None
    date: datetime=field(default_factory=datetime.now)
    total_return: Decimal=Decimal('0')
    daily_return: Decimal=Decimal('0')
    sharpe_ratio: Decimal=Decimal('0')
    max_drawdown: Decimal=Decimal('0')
    win_rate: Decimal=Decimal('0')
    profit_factor: Decimal=Decimal('0')
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    
    def __str__(self): 
        return f"{self.strategy.name if self.strategy else 'Unknown'} Performance: {self.total_return:.2%}"
