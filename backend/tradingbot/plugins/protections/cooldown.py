"""Cooldown Protection"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional

from .base import IProtection, ProtectionConfig, ProtectionResult


class CooldownProtection(IProtection):
    """
    Wait period after loss before next trade.

    Prevents rapid-fire trading after losses which can
    compound losses due to emotional/revenge trading.
    """

    def __init__(
        self,
        cooldown_candles: int = 5,
        candle_minutes: int = 60,  # 1 hour candles
        after_loss_only: bool = True,
        config: Optional[ProtectionConfig] = None,
    ):
        super().__init__(config)
        self.cooldown_candles = cooldown_candles
        self.candle_minutes = candle_minutes
        self.after_loss_only = after_loss_only

    def global_stop(self, trade_history: List[Dict]) -> Optional[ProtectionResult]:
        """Check if in cooldown period."""
        if not trade_history:
            return None

        # Get last trade
        last_trade = max(trade_history, key=lambda t: t.get('timestamp', datetime.min))

        # Check if was a loss
        if self.after_loss_only and last_trade.get('pnl', 0) >= 0:
            return None

        # Calculate cooldown end time
        last_time = last_trade.get('timestamp', datetime.now())
        cooldown_duration = timedelta(minutes=self.cooldown_candles * self.candle_minutes)
        cooldown_end = last_time + cooldown_duration

        if datetime.now() < cooldown_end:
            return ProtectionResult(
                should_stop=True,
                stop_until=cooldown_end,
                reason=f"Cooldown after {'loss' if self.after_loss_only else 'trade'} - "
                       f"resumes at {cooldown_end.strftime('%H:%M')}",
                severity="warning",
            )

        return None
