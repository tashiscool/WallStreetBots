"""Risk-managed strategy base class.

This module provides the base class for strategies that include
comprehensive risk management features.
"""

from abc import abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
import pandas as pd

from .base_strategy import BaseStrategy, StrategyConfig, StrategyResult


@dataclass
class RiskConfig:
    """Risk management configuration."""
    max_drawdown_pct: float = 0.10
    max_position_risk_pct: float = 0.05
    max_portfolio_risk_pct: float = 0.20
    var_confidence_level: float = 0.95
    stop_loss_pct: float = 0.05
    take_profit_pct: float = 0.10
    risk_free_rate: float = 0.02
    risk_metadata: Dict[str, Any] = field(default_factory=dict)


class RiskManagedStrategy(BaseStrategy):
    """Base class for strategies with comprehensive risk management.
    
    Risk-managed strategies include advanced risk controls, position sizing,
    and portfolio-level risk management.
    """
    
    def __init__(self, config: StrategyConfig, risk_config: RiskConfig):
        """Initialize risk-managed strategy."""
        super().__init__(config)
        self.risk_config = risk_config
        self.current_drawdown = 0.0
        self.portfolio_value = 100000.0  # Starting portfolio value
        self.peak_portfolio_value = self.portfolio_value
        self.risk_metrics = {}
        
    @abstractmethod
    def calculate_position_risk(self, symbol: str, price: float, quantity: int) -> float:
        """Calculate risk for a position.
        
        Args:
            symbol: Stock symbol
            price: Current price
            quantity: Number of shares
            
        Returns:
            Risk amount in dollars
        """
        pass
    
    @abstractmethod
    def calculate_portfolio_var(self) -> float:
        """Calculate portfolio Value at Risk.
        
        Returns:
            VaR amount in dollars
        """
        pass
    
    def update_portfolio_value(self, new_value: float) -> None:
        """Update portfolio value and calculate drawdown."""
        self.portfolio_value = new_value
        
        if new_value > self.peak_portfolio_value:
            self.peak_portfolio_value = new_value
            
        self.current_drawdown = (self.peak_portfolio_value - new_value) / self.peak_portfolio_value
    
    def check_risk_limits(self, symbol: str, price: float, quantity: int) -> bool:
        """Check if trade violates risk limits.
        
        Args:
            symbol: Stock symbol
            price: Current price
            quantity: Number of shares
            
        Returns:
            True if trade is within limits, False otherwise
        """
        # Check drawdown limit
        if self.current_drawdown > self.risk_config.max_drawdown_pct:
            return False
            
        # Check position risk limit
        position_risk = self.calculate_position_risk(symbol, price, quantity)
        position_risk_pct = position_risk / self.portfolio_value
        
        if position_risk_pct > self.risk_config.max_position_risk_pct:
            return False
            
        # Check portfolio VaR limit
        portfolio_var = self.calculate_portfolio_var()
        portfolio_var_pct = portfolio_var / self.portfolio_value
        
        if portfolio_var_pct > self.risk_config.max_portfolio_risk_pct:
            return False
            
        return True
    
    def calculate_risk_adjusted_position_size(self, symbol: str, price: float, base_quantity: int) -> int:
        """Calculate risk-adjusted position size.
        
        Args:
            symbol: Stock symbol
            price: Current price
            base_quantity: Base quantity from strategy
            
        Returns:
            Risk-adjusted quantity
        """
        max_risk_amount = self.portfolio_value * self.risk_config.max_position_risk_pct
        max_shares = int(max_risk_amount / price)
        
        return min(base_quantity, max_shares)
    
    def get_risk_status(self) -> Dict[str, Any]:
        """Get current risk status.
        
        Returns:
            Dictionary containing risk metrics
        """
        return {
            "current_drawdown": self.current_drawdown,
            "portfolio_value": self.portfolio_value,
            "peak_portfolio_value": self.peak_portfolio_value,
            "max_drawdown_limit": self.risk_config.max_drawdown_pct,
            "risk_metrics": self.risk_metrics,
        }


