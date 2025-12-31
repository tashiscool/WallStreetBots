"""Low Profit Pairs Protection"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional

from .base import IProtection, ProtectionConfig, ProtectionResult


class LowProfitPairsProtection(IProtection):
    """
    Disable trading on consistently unprofitable symbols.

    Automatically blacklists symbols that have been
    losing money over the lookback period.
    """

    def __init__(
        self,
        min_profit: float = 0.0,  # Minimum profit to continue trading
        trade_limit: int = 3,  # Minimum trades to evaluate
        lookback_days: int = 7,
        stop_duration_days: int = 7,
        config: Optional[ProtectionConfig] = None,
    ):
        super().__init__(config)
        self.min_profit = min_profit
        self.trade_limit = trade_limit
        self.lookback_days = lookback_days
        self.stop_duration_days = stop_duration_days

    def global_stop(self, trade_history: List[Dict]) -> Optional[ProtectionResult]:
        """Global stop is not used - this is per-pair only."""
        return None

    def stop_per_pair(
        self,
        symbol: str,
        trade_history: List[Dict],
    ) -> Optional[ProtectionResult]:
        """Check if symbol should be disabled."""
        lookback = timedelta(days=self.lookback_days)
        recent_trades = self.filter_trades_by_period(trade_history, lookback)

        # Filter to this symbol
        symbol_trades = [t for t in recent_trades if t.get('symbol') == symbol]

        if len(symbol_trades) < self.trade_limit:
            return None  # Not enough data

        profit = self.total_profit(symbol_trades)

        if profit < self.min_profit:
            stop_until = datetime.now() + timedelta(days=self.stop_duration_days)

            return ProtectionResult(
                should_stop=True,
                stop_until=stop_until,
                reason=f"Low profit for {symbol}: ${profit:.2f} in {len(symbol_trades)} trades",
                affected_symbols=[symbol],
                severity="warning",
            )

        return None

    def get_blacklisted_symbols(
        self,
        trade_history: List[Dict],
    ) -> List[str]:
        """Get list of symbols that should be blacklisted."""
        lookback = timedelta(days=self.lookback_days)
        recent_trades = self.filter_trades_by_period(trade_history, lookback)

        # Group by symbol
        by_symbol: Dict[str, List[Dict]] = {}
        for trade in recent_trades:
            symbol = trade.get('symbol', '')
            if symbol:
                if symbol not in by_symbol:
                    by_symbol[symbol] = []
                by_symbol[symbol].append(trade)

        # Check each symbol
        blacklist = []
        for symbol, trades in by_symbol.items():
            if len(trades) >= self.trade_limit:
                profit = self.total_profit(trades)
                if profit < self.min_profit:
                    blacklist.append(symbol)

        return blacklist
