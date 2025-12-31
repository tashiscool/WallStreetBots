"""Max Drawdown Protection"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional

from .base import IProtection, ProtectionConfig, ProtectionResult


class MaxDrawdownProtection(IProtection):
    """
    Stop trading when drawdown exceeds threshold.

    Protects against catastrophic losses by halting
    when cumulative losses reach a limit.
    """

    def __init__(
        self,
        max_drawdown: float = 0.10,  # 10%
        lookback_candles: int = 48,
        candle_minutes: int = 60,
        stop_duration_hours: int = 24,
        config: Optional[ProtectionConfig] = None,
    ):
        super().__init__(config)
        self.max_drawdown = max_drawdown
        self.lookback_candles = lookback_candles
        self.candle_minutes = candle_minutes
        self.stop_duration_hours = stop_duration_hours
        self._starting_balance: Optional[float] = None

    def set_starting_balance(self, balance: float) -> None:
        """Set starting balance for drawdown calculation."""
        self._starting_balance = balance

    def global_stop(self, trade_history: List[Dict]) -> Optional[ProtectionResult]:
        """Check if max drawdown exceeded."""
        lookback = timedelta(minutes=self.lookback_candles * self.candle_minutes)
        recent_trades = self.filter_trades_by_period(trade_history, lookback)

        if not recent_trades:
            return None

        # Calculate drawdown from trades
        total_pnl = self.total_profit(recent_trades)

        # Use starting balance if set, otherwise estimate from first trade
        if self._starting_balance:
            drawdown = abs(total_pnl) / self._starting_balance
        else:
            # Estimate from trade sizes
            avg_trade_value = sum(
                abs(t.get('value', 0)) for t in recent_trades
            ) / len(recent_trades) if recent_trades else 10000
            estimated_balance = avg_trade_value * 10  # Rough estimate
            drawdown = abs(total_pnl) / estimated_balance if estimated_balance > 0 else 0

        if total_pnl < 0 and drawdown >= self.max_drawdown:
            stop_until = datetime.now() + timedelta(hours=self.stop_duration_hours)

            return ProtectionResult(
                should_stop=True,
                stop_until=stop_until,
                reason=f"Max drawdown triggered: {drawdown:.1%} >= {self.max_drawdown:.1%} "
                       f"(loss: ${abs(total_pnl):.2f})",
                severity="critical",
            )

        return None
