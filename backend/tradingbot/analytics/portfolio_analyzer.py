"""
Portfolio Analyzer - Inspired by Nautilus Trader.

Comprehensive portfolio performance analysis with pluggable statistics.

Concepts from: https://github.com/nautechsystems/nautilus_trader
License: LGPL-3.0 (concepts only, clean-room implementation)
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
import numpy as np

from .statistics import (
    PortfolioStatistic,
    Trade,
    DEFAULT_STATISTICS,
)


@dataclass
class AnalysisResult:
    """Results from portfolio analysis."""
    total_pnl: float
    total_pnl_percent: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    statistics: Dict[str, Any]
    equity_curve: List[float]
    drawdown_curve: List[float]
    analysis_period_start: Optional[datetime] = None
    analysis_period_end: Optional[datetime] = None


class PortfolioAnalyzer:
    """
    Portfolio performance analyzer with pluggable statistics.

    Tracks trades, returns, and calculates comprehensive performance metrics.
    """

    def __init__(self):
        self._statistics: Dict[str, PortfolioStatistic] = {}
        self._trades: List[Trade] = []
        self._returns: List[float] = []
        self._return_timestamps: List[datetime] = []
        self._starting_balance: float = 0.0
        self._current_balance: float = 0.0

        # Register default statistics
        for stat in DEFAULT_STATISTICS:
            self.register_statistic(stat)

    def register_statistic(self, statistic: PortfolioStatistic) -> None:
        """Register a statistic with the analyzer."""
        self._statistics[statistic.name] = statistic

    def deregister_statistic(self, name: str) -> None:
        """Remove a statistic from the analyzer."""
        self._statistics.pop(name, None)

    def reset(self) -> None:
        """Reset all tracked data."""
        self._trades = []
        self._returns = []
        self._return_timestamps = []
        self._starting_balance = 0.0
        self._current_balance = 0.0

    def set_starting_balance(self, balance: float) -> None:
        """Set the starting account balance."""
        self._starting_balance = balance
        self._current_balance = balance

    def add_trade(self, trade: Trade) -> None:
        """Add a completed trade to the analysis."""
        self._trades.append(trade)
        self._current_balance += trade.pnl

    def add_return(self, timestamp: datetime, return_value: float) -> None:
        """Add a return observation (e.g., daily return)."""
        self._returns.append(return_value)
        self._return_timestamps.append(timestamp)

    def add_trades(self, trades: List[Trade]) -> None:
        """Add multiple trades."""
        for trade in trades:
            self.add_trade(trade)

    def add_returns(self, returns: List[float],
                   timestamps: Optional[List[datetime]] = None) -> None:
        """Add multiple return observations."""
        self._returns.extend(returns)
        if timestamps:
            self._return_timestamps.extend(timestamps)

    @property
    def trades(self) -> List[Trade]:
        """Return all tracked trades."""
        return self._trades.copy()

    @property
    def returns(self) -> List[float]:
        """Return all tracked returns."""
        return self._returns.copy()

    @property
    def total_pnl(self) -> float:
        """Total profit and loss."""
        return sum(t.pnl for t in self._trades)

    @property
    def total_pnl_percent(self) -> float:
        """Total P&L as percentage of starting balance."""
        if self._starting_balance == 0:
            return 0.0
        return (self.total_pnl / self._starting_balance) * 100

    def get_pnls(self) -> List[float]:
        """Get list of all trade P&Ls."""
        return [t.pnl for t in self._trades]

    def get_equity_curve(self) -> List[float]:
        """Calculate equity curve from trades."""
        if not self._trades:
            return [self._starting_balance]

        equity = [self._starting_balance]
        running = self._starting_balance

        for trade in sorted(self._trades, key=lambda t: t.exit_time):
            running += trade.pnl
            equity.append(running)

        return equity

    def get_drawdown_curve(self) -> List[float]:
        """Calculate drawdown curve (in percentage)."""
        equity = self.get_equity_curve()
        if not equity:
            return []

        equity_arr = np.array(equity)
        running_max = np.maximum.accumulate(equity_arr)
        drawdowns = (equity_arr - running_max) / running_max * 100

        return drawdowns.tolist()

    def calculate_pnl_statistics(self) -> Dict[str, Any]:
        """Calculate all statistics from P&Ls."""
        pnls = self.get_pnls()
        results = {}

        for name, stat in self._statistics.items():
            value = stat.calculate_from_pnls(pnls)
            if value is not None:
                results[name] = value

        return results

    def calculate_return_statistics(self) -> Dict[str, Any]:
        """Calculate all statistics from returns."""
        results = {}

        for name, stat in self._statistics.items():
            value = stat.calculate_from_returns(self._returns)
            if value is not None:
                results[name] = value

        return results

    def calculate_trade_statistics(self) -> Dict[str, Any]:
        """Calculate all statistics from trades."""
        results = {}

        for name, stat in self._statistics.items():
            value = stat.calculate_from_trades(self._trades)
            if value is not None:
                results[name] = value

        return results

    def get_all_statistics(self) -> Dict[str, Any]:
        """Calculate all available statistics."""
        results = {}

        # Combine all calculation methods
        results.update(self.calculate_pnl_statistics())
        results.update(self.calculate_return_statistics())
        results.update(self.calculate_trade_statistics())

        return results

    def analyze(self) -> AnalysisResult:
        """Perform full portfolio analysis."""
        winners = [t for t in self._trades if t.is_winner]
        losers = [t for t in self._trades if t.is_loser]

        period_start = None
        period_end = None
        if self._trades:
            sorted_trades = sorted(self._trades, key=lambda t: t.entry_time)
            period_start = sorted_trades[0].entry_time
            period_end = sorted_trades[-1].exit_time

        return AnalysisResult(
            total_pnl=self.total_pnl,
            total_pnl_percent=self.total_pnl_percent,
            total_trades=len(self._trades),
            winning_trades=len(winners),
            losing_trades=len(losers),
            statistics=self.get_all_statistics(),
            equity_curve=self.get_equity_curve(),
            drawdown_curve=self.get_drawdown_curve(),
            analysis_period_start=period_start,
            analysis_period_end=period_end,
        )

    def format_statistics(self, width: int = 30) -> List[str]:
        """Format statistics for display."""
        stats = self.get_all_statistics()
        output = []

        for name, value in stats.items():
            padding = width - len(name)
            if isinstance(value, float):
                output.append(f"{name}:{' ' * padding}{value:,.4f}")
            else:
                output.append(f"{name}:{' ' * padding}{value}")

        return output

    def get_summary(self) -> str:
        """Get a formatted summary string."""
        result = self.analyze()
        lines = [
            "=" * 50,
            "PORTFOLIO ANALYSIS SUMMARY",
            "=" * 50,
            f"Total P&L:          ${result.total_pnl:,.2f}",
            f"Total P&L %:        {result.total_pnl_percent:.2f}%",
            f"Total Trades:       {result.total_trades}",
            f"Winning Trades:     {result.winning_trades}",
            f"Losing Trades:      {result.losing_trades}",
            "-" * 50,
            "STATISTICS:",
            "-" * 50,
        ]
        lines.extend(self.format_statistics())
        lines.append("=" * 50)

        return "\n".join(lines)


def create_analyzer_with_trades(
    trades: List[Trade],
    starting_balance: float = 100000.0
) -> PortfolioAnalyzer:
    """Create and populate an analyzer from trades."""
    analyzer = PortfolioAnalyzer()
    analyzer.set_starting_balance(starting_balance)
    analyzer.add_trades(trades)
    return analyzer


def create_analyzer_with_returns(
    returns: List[float],
    timestamps: Optional[List[datetime]] = None
) -> PortfolioAnalyzer:
    """Create and populate an analyzer from returns."""
    analyzer = PortfolioAnalyzer()
    analyzer.add_returns(returns, timestamps)
    return analyzer
