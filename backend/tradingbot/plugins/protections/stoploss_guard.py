"""Stoploss Guard Protection"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional

from .base import IProtection, ProtectionConfig, ProtectionResult


class StoplossGuardProtection(IProtection):
    """
    Stop trading after multiple stoploss hits.

    If too many trades hit stoploss in a period, something
    may be wrong with market conditions or strategy.
    """

    def __init__(
        self,
        trade_limit: int = 4,
        lookback_candles: int = 24,
        candle_minutes: int = 60,
        stop_duration_candles: int = 12,
        config: Optional[ProtectionConfig] = None,
    ):
        super().__init__(config)
        self.trade_limit = trade_limit
        self.lookback_candles = lookback_candles
        self.candle_minutes = candle_minutes
        self.stop_duration_candles = stop_duration_candles

    def global_stop(self, trade_history: List[Dict]) -> Optional[ProtectionResult]:
        """Check if stoploss limit exceeded."""
        lookback = timedelta(minutes=self.lookback_candles * self.candle_minutes)
        recent_trades = self.filter_trades_by_period(trade_history, lookback)

        stoploss_count = self.count_stoplosses(recent_trades)

        if stoploss_count >= self.trade_limit:
            stop_duration = timedelta(
                minutes=self.stop_duration_candles * self.candle_minutes
            )
            stop_until = datetime.now() + stop_duration

            return ProtectionResult(
                should_stop=True,
                stop_until=stop_until,
                reason=f"Stoploss guard triggered: {stoploss_count} stoplosses "
                       f"in last {self.lookback_candles} candles",
                severity="critical",
            )

        return None
