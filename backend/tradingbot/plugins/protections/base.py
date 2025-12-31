"""Protection Plugin Base Class"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional


@dataclass
class ProtectionConfig:
    """Configuration for protection plugins."""
    enabled: bool = True
    lookback_period: timedelta = timedelta(hours=24)
    cooldown_period: timedelta = timedelta(hours=1)
    custom_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProtectionResult:
    """Result from protection check."""
    should_stop: bool = False
    stop_until: Optional[datetime] = None
    reason: str = ""
    affected_symbols: List[str] = field(default_factory=list)
    severity: str = "warning"  # warning, critical

    @property
    def is_global_stop(self) -> bool:
        """Check if this is a global trading stop."""
        return self.should_stop and not self.affected_symbols


class IProtection(ABC):
    """
    Interface for trading protection plugins.

    Protections can:
    - Stop all trading globally
    - Stop trading on specific symbols
    - Recommend position size reduction
    """

    def __init__(self, config: Optional[ProtectionConfig] = None):
        self.config = config or ProtectionConfig()
        self._trade_history: List[Dict[str, Any]] = []
        self._last_check: Optional[datetime] = None

    @abstractmethod
    def global_stop(self, trade_history: List[Dict]) -> Optional[ProtectionResult]:
        """
        Check if trading should stop globally.

        Args:
            trade_history: Recent trade history

        Returns:
            ProtectionResult if should stop, None otherwise
        """
        pass

    def stop_per_pair(
        self,
        symbol: str,
        trade_history: List[Dict],
    ) -> Optional[ProtectionResult]:
        """
        Check if trading should stop for a specific symbol.

        Args:
            symbol: Symbol to check
            trade_history: Trade history for this symbol

        Returns:
            ProtectionResult if should stop, None otherwise
        """
        return None  # Override in subclass

    def update_trade_history(self, trade_history: List[Dict]) -> None:
        """Update internal trade history."""
        self._trade_history = trade_history
        self._last_check = datetime.now()

    def filter_trades_by_period(
        self,
        trades: List[Dict],
        period: Optional[timedelta] = None,
    ) -> List[Dict]:
        """Filter trades to lookback period."""
        period = period or self.config.lookback_period
        cutoff = datetime.now() - period

        return [
            t for t in trades
            if t.get('timestamp', datetime.now()) > cutoff
        ]

    def count_losses(self, trades: List[Dict]) -> int:
        """Count losing trades."""
        return sum(1 for t in trades if t.get('pnl', 0) < 0)

    def count_stoplosses(self, trades: List[Dict]) -> int:
        """Count trades closed by stoploss."""
        return sum(1 for t in trades if t.get('exit_reason') == 'stoploss')

    def total_profit(self, trades: List[Dict]) -> float:
        """Calculate total profit from trades."""
        return sum(t.get('pnl', 0) for t in trades)

    @property
    def name(self) -> str:
        """Protection name."""
        return self.__class__.__name__
