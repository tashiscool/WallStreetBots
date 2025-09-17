"""Production strategy base class.

This module provides the base class for production-ready strategies
that integrate with real brokers and live data feeds.
"""

from abc import abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
import pandas as pd

from .base_strategy import BaseStrategy, StrategyConfig, StrategyResult


@dataclass
class ProductionStrategyConfig(StrategyConfig):
    """Configuration for production strategies."""
    paper_trading: bool = True
    broker_api_key: Optional[str] = None
    broker_secret_key: Optional[str] = None
    data_provider: str = "alpaca"
    execution_delay_ms: int = 100
    max_retries: int = 3
    timeout_seconds: int = 30
    production_metadata: Dict[str, Any] = field(default_factory=dict)


class ProductionStrategy(BaseStrategy):
    """Base class for production-ready trading strategies.
    
    Production strategies integrate with real brokers, live data feeds,
    and comprehensive risk management systems.
    """
    
    def __init__(self, config: ProductionStrategyConfig):
        """Initialize production strategy."""
        super().__init__(config)
        self.config: ProductionStrategyConfig = config
        self.broker_connected = False
        self.data_provider_connected = False
        self.last_execution_time = None
        self.execution_count = 0
        self.error_count = 0
        
    @abstractmethod
    def connect_to_broker(self) -> bool:
        """Connect to trading broker.
        
        Returns:
            True if connection successful, False otherwise
        """
        pass
    
    @abstractmethod
    def connect_to_data_provider(self) -> bool:
        """Connect to market data provider.
        
        Returns:
            True if connection successful, False otherwise
        """
        pass
    
    @abstractmethod
    def execute_trade(self, result: StrategyResult) -> bool:
        """Execute trade based on strategy result.
        
        Args:
            result: Strategy result containing trade information
            
        Returns:
            True if trade executed successfully, False otherwise
        """
        pass
    
    @abstractmethod
    def get_live_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Get live market data for symbol.
        
        Args:
            symbol: Stock symbol to get data for
            
        Returns:
            DataFrame with live data or None if unavailable
        """
        pass
    
    def initialize_production(self) -> bool:
        """Initialize production environment.
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            # Connect to broker
            self.broker_connected = self.connect_to_broker()
            if not self.broker_connected:
                return False
                
            # Connect to data provider
            self.data_provider_connected = self.connect_to_data_provider()
            if not self.data_provider_connected:
                return False
                
            # Start strategy
            self.start()
            return True
            
        except Exception as e:
            self.error_count += 1
            return False
    
    def shutdown_production(self) -> None:
        """Shutdown production environment."""
        self.stop()
        self.broker_connected = False
        self.data_provider_connected = False
    
    def get_production_status(self) -> Dict[str, Any]:
        """Get production-specific status information.
        
        Returns:
            Dictionary containing production status
        """
        base_status = self.get_status()
        base_status.update({
            "broker_connected": self.broker_connected,
            "data_provider_connected": self.data_provider_connected,
            "paper_trading": self.config.paper_trading,
            "execution_count": self.execution_count,
            "error_count": self.error_count,
            "last_execution_time": self.last_execution_time.isoformat() if self.last_execution_time else None,
        })
        return base_status
    
    def validate_production_environment(self) -> bool:
        """Validate that production environment is ready.
        
        Returns:
            True if environment is valid, False otherwise
        """
        return (
            self.broker_connected and
            self.data_provider_connected and
            self.status.value in ["running", "paused"]
        )
