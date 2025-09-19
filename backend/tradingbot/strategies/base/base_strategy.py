"""Base strategy class and interfaces.

This module defines the core interfaces that all trading strategies
must implement, providing a consistent API across the system.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
import pandas as pd


class StrategyStatus(Enum):
    """Strategy execution status."""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"
    STOPPED = "stopped"


class SignalType(Enum):
    """Trading signal types."""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    CLOSE = "close"


@dataclass
class StrategyConfig:
    """Base configuration for all strategies."""
    name: str
    enabled: bool = True
    max_position_size: float = 10000.0
    max_total_risk: float = 50000.0
    stop_loss_pct: float = 0.05
    take_profit_multiplier: float = 2.0
    risk_free_rate: float = 0.02
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StrategyResult:
    """Result of strategy execution."""
    symbol: str
    signal: SignalType
    confidence: float
    price: float
    quantity: int
    timestamp: datetime
    reasoning: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseStrategy(ABC):
    """Abstract base class for all trading strategies.
    
    All strategies must implement the core methods defined here to ensure
    consistency and interoperability across the trading system.
    """
    
    def __init__(self, config: StrategyConfig):
        """Initialize strategy with configuration."""
        self.config = config
        self.status = StrategyStatus.IDLE
        self.last_update = datetime.now()
        self.performance_metrics = {}
        
    @abstractmethod
    def analyze(self, data: pd.DataFrame, symbol: str) -> Optional[StrategyResult]:
        """Analyze market data and generate trading signal.
        
        Args:
            data: Market data for the symbol
            symbol: Stock symbol to analyze
            
        Returns:
            StrategyResult if signal generated, None otherwise
        """
        pass
    
    @abstractmethod
    def get_required_data(self) -> List[str]:
        """Get list of required data fields for this strategy.
        
        Returns:
            List of required data column names
        """
        pass
    
    @abstractmethod
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate that data meets strategy requirements.
        
        Args:
            data: Market data to validate
            
        Returns:
            True if data is valid, False otherwise
        """
        pass
    
    def start(self) -> None:
        """Start strategy execution."""
        self.status = StrategyStatus.RUNNING
        self.last_update = datetime.now()
        
    def stop(self) -> None:
        """Stop strategy execution."""
        self.status = StrategyStatus.STOPPED
        self.last_update = datetime.now()
        
    def pause(self) -> None:
        """Pause strategy execution."""
        self.status = StrategyStatus.PAUSED
        self.last_update = datetime.now()
        
    def resume(self) -> None:
        """Resume strategy execution."""
        if self.status == StrategyStatus.PAUSED:
            self.status = StrategyStatus.RUNNING
            self.last_update = datetime.now()
    
    def get_status(self) -> Dict[str, Any]:
        """Get current strategy status.
        
        Returns:
            Dictionary containing status information
        """
        return {
            "name": self.config.name,
            "status": self.status.value,
            "last_update": self.last_update.isoformat(),
            "enabled": self.config.enabled,
            "performance_metrics": self.performance_metrics,
        }
    
    def update_config(self, new_config: StrategyConfig) -> None:
        """Update strategy configuration.
        
        Args:
            new_config: New configuration to apply
        """
        self.config = new_config
        self.last_update = datetime.now()
    
    def calculate_position_size(self, price: float, risk_amount: float) -> int:
        """Calculate position size based on risk management rules.
        
        Args:
            price: Current price of the asset
            risk_amount: Maximum amount to risk
            
        Returns:
            Number of shares to trade
        """
        if price <= 0:
            return 0
            
        max_shares = int(self.config.max_position_size / price)
        risk_shares = int(risk_amount / price)
        
        return min(max_shares, risk_shares)



