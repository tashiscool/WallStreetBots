"""
Portfolio Statistics - Inspired by Nautilus Trader.

Provides 20+ portfolio performance statistics for strategy evaluation.

Inspired by Nautilus Trader concepts.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional, Any, Dict
import numpy as np
from enum import Enum


@dataclass
class Trade:
    """Represents a completed trade for analysis."""
    symbol: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    quantity: float
    side: str  # "long" or "short"
    pnl: float
    pnl_percent: float
    commission: float = 0.0

    @property
    def is_winner(self) -> bool:
        return self.pnl > 0

    @property
    def is_loser(self) -> bool:
        return self.pnl < 0

    @property
    def is_long(self) -> bool:
        return self.side == "long"

    @property
    def holding_period(self) -> timedelta:
        return self.exit_time - self.entry_time


class PortfolioStatistic(ABC):
    """
    Abstract base class for portfolio performance statistics.

    All statistics return JSON-serializable primitives.
    """

    @property
    def name(self) -> str:
        """Human-readable name for the statistic."""
        # Convert CamelCase to spaces
        import re
        klass = type(self).__name__
        matches = re.finditer(
            ".+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)",
            klass
        )
        return " ".join([m.group(0) for m in matches])

    def calculate_from_returns(self, returns: List[float]) -> Optional[Any]:
        """Calculate statistic from raw returns."""
        return None

    def calculate_from_pnls(self, pnls: List[float]) -> Optional[Any]:
        """Calculate statistic from realized P&Ls."""
        return None

    def calculate_from_trades(self, trades: List[Trade]) -> Optional[Any]:
        """Calculate statistic from trade list."""
        return None


class WinRate(PortfolioStatistic):
    """
    Win Rate - Percentage of winning trades.

    Formula: Winners / Total Trades * 100
    """

    def calculate_from_trades(self, trades: List[Trade]) -> Optional[float]:
        if not trades:
            return None
        winners = sum(1 for t in trades if t.is_winner)
        return round(winners / len(trades) * 100, 2)

    def calculate_from_pnls(self, pnls: List[float]) -> Optional[float]:
        if not pnls:
            return None
        winners = sum(1 for p in pnls if p > 0)
        return round(winners / len(pnls) * 100, 2)


class ProfitFactor(PortfolioStatistic):
    """
    Profit Factor - Ratio of gross profit to gross loss.

    Formula: Sum(Winning Trades) / |Sum(Losing Trades)|

    Values > 1.0 indicate profitability.
    Values > 2.0 are considered excellent.
    """

    def calculate_from_trades(self, trades: List[Trade]) -> Optional[float]:
        if not trades:
            return None
        gross_profit = sum(t.pnl for t in trades if t.is_winner)
        gross_loss = abs(sum(t.pnl for t in trades if t.is_loser))
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0
        return round(gross_profit / gross_loss, 3)

    def calculate_from_pnls(self, pnls: List[float]) -> Optional[float]:
        if not pnls:
            return None
        gross_profit = sum(p for p in pnls if p > 0)
        gross_loss = abs(sum(p for p in pnls if p < 0))
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0
        return round(gross_profit / gross_loss, 3)


class Expectancy(PortfolioStatistic):
    """
    Expectancy - Expected value per trade.

    Formula: (Win Rate * Avg Win) - (Loss Rate * Avg Loss)

    Positive values indicate a profitable system.
    """

    def calculate_from_trades(self, trades: List[Trade]) -> Optional[float]:
        if not trades:
            return None

        winners = [t.pnl for t in trades if t.is_winner]
        losers = [t.pnl for t in trades if t.is_loser]

        win_rate = len(winners) / len(trades)
        loss_rate = len(losers) / len(trades)

        avg_win = np.mean(winners) if winners else 0
        avg_loss = abs(np.mean(losers)) if losers else 0

        expectancy = (win_rate * avg_win) - (loss_rate * avg_loss)
        return round(expectancy, 2)

    def calculate_from_pnls(self, pnls: List[float]) -> Optional[float]:
        if not pnls:
            return None

        winners = [p for p in pnls if p > 0]
        losers = [p for p in pnls if p < 0]

        win_rate = len(winners) / len(pnls)
        loss_rate = len(losers) / len(pnls)

        avg_win = np.mean(winners) if winners else 0
        avg_loss = abs(np.mean(losers)) if losers else 0

        expectancy = (win_rate * avg_win) - (loss_rate * avg_loss)
        return round(expectancy, 2)


class SharpeRatio(PortfolioStatistic):
    """
    Sharpe Ratio - Risk-adjusted return.

    Formula: (Mean Return - Risk Free Rate) / Std(Returns)

    Values > 1.0 are good, > 2.0 are excellent.
    """

    def __init__(self, risk_free_rate: float = 0.02, periods_per_year: int = 252):
        self._risk_free_rate = risk_free_rate
        self._periods = periods_per_year

    def calculate_from_returns(self, returns: List[float]) -> Optional[float]:
        if not returns or len(returns) < 2:
            return None

        returns_arr = np.array(returns)
        mean_return = np.mean(returns_arr)
        std_return = np.std(returns_arr, ddof=1)

        if std_return == 0:
            return 0.0

        # Annualize
        daily_rf = self._risk_free_rate / self._periods
        excess_return = mean_return - daily_rf
        sharpe = (excess_return / std_return) * np.sqrt(self._periods)

        return round(sharpe, 3)


class SortinoRatio(PortfolioStatistic):
    """
    Sortino Ratio - Risk-adjusted return using downside deviation.

    Like Sharpe but only penalizes downside volatility.

    Formula: (Mean Return - Target) / Downside Deviation
    """

    def __init__(self, target_return: float = 0.0, periods_per_year: int = 252):
        self._target = target_return
        self._periods = periods_per_year

    def calculate_from_returns(self, returns: List[float]) -> Optional[float]:
        if not returns or len(returns) < 2:
            return None

        returns_arr = np.array(returns)
        mean_return = np.mean(returns_arr)

        # Downside deviation - only negative returns below target
        downside_returns = returns_arr[returns_arr < self._target] - self._target
        if len(downside_returns) == 0:
            return float('inf') if mean_return > self._target else 0.0

        downside_std = np.sqrt(np.mean(downside_returns ** 2))

        if downside_std == 0:
            return 0.0

        sortino = ((mean_return - self._target) / downside_std) * np.sqrt(self._periods)
        return round(sortino, 3)


class CalmarRatio(PortfolioStatistic):
    """
    Calmar Ratio - Return relative to maximum drawdown.

    Formula: CAGR / |Max Drawdown|

    Higher is better; measures return per unit of drawdown risk.
    """

    def calculate_from_returns(self, returns: List[float]) -> Optional[float]:
        if not returns or len(returns) < 2:
            return None

        # Calculate cumulative returns
        cumulative = np.cumprod(1 + np.array(returns))

        # Calculate max drawdown
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max
        max_dd = abs(np.min(drawdowns))

        if max_dd == 0:
            return float('inf')

        # Calculate annualized return
        total_return = cumulative[-1] - 1
        years = len(returns) / 252  # Assuming daily returns
        if years == 0:
            return 0.0
        cagr = (1 + total_return) ** (1 / years) - 1

        calmar = cagr / max_dd
        return round(calmar, 3)


class MaxDrawdown(PortfolioStatistic):
    """
    Maximum Drawdown - Largest peak-to-trough decline.

    Returns as a percentage (e.g., -25.5 for 25.5% drawdown).
    """

    def calculate_from_returns(self, returns: List[float]) -> Optional[float]:
        if not returns:
            return None

        cumulative = np.cumprod(1 + np.array(returns))
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max

        max_dd = np.min(drawdowns) * 100
        return round(max_dd, 2)


class CAGR(PortfolioStatistic):
    """
    Compound Annual Growth Rate.

    Formula: (Ending Value / Starting Value) ^ (1 / Years) - 1
    """

    def __init__(self, periods_per_year: int = 252):
        self._periods = periods_per_year

    def calculate_from_returns(self, returns: List[float]) -> Optional[float]:
        if not returns:
            return None

        cumulative = np.prod(1 + np.array(returns))
        years = len(returns) / self._periods

        if years == 0:
            return 0.0

        cagr = (cumulative ** (1 / years)) - 1
        return round(cagr * 100, 2)  # As percentage


class ReturnsVolatility(PortfolioStatistic):
    """
    Returns Volatility - Annualized standard deviation of returns.
    """

    def __init__(self, periods_per_year: int = 252):
        self._periods = periods_per_year

    def calculate_from_returns(self, returns: List[float]) -> Optional[float]:
        if not returns or len(returns) < 2:
            return None

        std = np.std(returns, ddof=1) * np.sqrt(self._periods)
        return round(std * 100, 2)  # As percentage


class ReturnsAverage(PortfolioStatistic):
    """
    Average Return - Mean of all returns.
    """

    def calculate_from_returns(self, returns: List[float]) -> Optional[float]:
        if not returns:
            return None
        return round(np.mean(returns) * 100, 4)  # As percentage


class ReturnsAvgWin(PortfolioStatistic):
    """
    Average Winning Return - Mean of positive returns.
    """

    def calculate_from_returns(self, returns: List[float]) -> Optional[float]:
        if not returns:
            return None
        winners = [r for r in returns if r > 0]
        if not winners:
            return 0.0
        return round(np.mean(winners) * 100, 4)


class ReturnsAvgLoss(PortfolioStatistic):
    """
    Average Losing Return - Mean of negative returns.
    """

    def calculate_from_returns(self, returns: List[float]) -> Optional[float]:
        if not returns:
            return None
        losers = [r for r in returns if r < 0]
        if not losers:
            return 0.0
        return round(np.mean(losers) * 100, 4)


class RiskReturnRatio(PortfolioStatistic):
    """
    Risk/Return Ratio - Average return divided by volatility.

    Similar to Sharpe but without risk-free rate adjustment.
    """

    def calculate_from_returns(self, returns: List[float]) -> Optional[float]:
        if not returns or len(returns) < 2:
            return None

        mean_return = np.mean(returns)
        std_return = np.std(returns, ddof=1)

        if std_return == 0:
            return 0.0

        ratio = mean_return / std_return
        return round(ratio, 4)


class WinnerAverage(PortfolioStatistic):
    """
    Average Winner - Mean P&L of winning trades.
    """

    def calculate_from_trades(self, trades: List[Trade]) -> Optional[float]:
        if not trades:
            return None
        winners = [t.pnl for t in trades if t.is_winner]
        if not winners:
            return 0.0
        return round(np.mean(winners), 2)

    def calculate_from_pnls(self, pnls: List[float]) -> Optional[float]:
        if not pnls:
            return None
        winners = [p for p in pnls if p > 0]
        if not winners:
            return 0.0
        return round(np.mean(winners), 2)


class WinnerMax(PortfolioStatistic):
    """
    Maximum Winner - Largest winning trade.
    """

    def calculate_from_trades(self, trades: List[Trade]) -> Optional[float]:
        if not trades:
            return None
        winners = [t.pnl for t in trades if t.is_winner]
        if not winners:
            return 0.0
        return round(max(winners), 2)

    def calculate_from_pnls(self, pnls: List[float]) -> Optional[float]:
        if not pnls:
            return None
        winners = [p for p in pnls if p > 0]
        if not winners:
            return 0.0
        return round(max(winners), 2)


class WinnerMin(PortfolioStatistic):
    """
    Minimum Winner - Smallest winning trade.
    """

    def calculate_from_trades(self, trades: List[Trade]) -> Optional[float]:
        if not trades:
            return None
        winners = [t.pnl for t in trades if t.is_winner]
        if not winners:
            return 0.0
        return round(min(winners), 2)

    def calculate_from_pnls(self, pnls: List[float]) -> Optional[float]:
        if not pnls:
            return None
        winners = [p for p in pnls if p > 0]
        if not winners:
            return 0.0
        return round(max(winners), 2)


class LoserAverage(PortfolioStatistic):
    """
    Average Loser - Mean P&L of losing trades (negative).
    """

    def calculate_from_trades(self, trades: List[Trade]) -> Optional[float]:
        if not trades:
            return None
        losers = [t.pnl for t in trades if t.is_loser]
        if not losers:
            return 0.0
        return round(np.mean(losers), 2)

    def calculate_from_pnls(self, pnls: List[float]) -> Optional[float]:
        if not pnls:
            return None
        losers = [p for p in pnls if p < 0]
        if not losers:
            return 0.0
        return round(np.mean(losers), 2)


class LoserMax(PortfolioStatistic):
    """
    Maximum Loser - Largest losing trade (most negative).
    """

    def calculate_from_trades(self, trades: List[Trade]) -> Optional[float]:
        if not trades:
            return None
        losers = [t.pnl for t in trades if t.is_loser]
        if not losers:
            return 0.0
        return round(min(losers), 2)  # Most negative

    def calculate_from_pnls(self, pnls: List[float]) -> Optional[float]:
        if not pnls:
            return None
        losers = [p for p in pnls if p < 0]
        if not losers:
            return 0.0
        return round(min(losers), 2)


class LoserMin(PortfolioStatistic):
    """
    Minimum Loser - Smallest losing trade.
    """

    def calculate_from_trades(self, trades: List[Trade]) -> Optional[float]:
        if not trades:
            return None
        losers = [t.pnl for t in trades if t.is_loser]
        if not losers:
            return 0.0
        return round(max(losers), 2)  # Closest to zero

    def calculate_from_pnls(self, pnls: List[float]) -> Optional[float]:
        if not pnls:
            return None
        losers = [p for p in pnls if p < 0]
        if not losers:
            return 0.0
        return round(max(losers), 2)


class LongRatio(PortfolioStatistic):
    """
    Long Ratio - Percentage of trades that were long.
    """

    def calculate_from_trades(self, trades: List[Trade]) -> Optional[float]:
        if not trades:
            return None
        longs = sum(1 for t in trades if t.is_long)
        return round(longs / len(trades) * 100, 2)


class PayoffRatio(PortfolioStatistic):
    """
    Payoff Ratio (Risk/Reward) - Average win / Average loss.

    Values > 1.0 mean winners are larger than losers on average.
    """

    def calculate_from_trades(self, trades: List[Trade]) -> Optional[float]:
        if not trades:
            return None

        winners = [t.pnl for t in trades if t.is_winner]
        losers = [abs(t.pnl) for t in trades if t.is_loser]

        avg_win = np.mean(winners) if winners else 0
        avg_loss = np.mean(losers) if losers else 0

        if avg_loss == 0:
            return float('inf') if avg_win > 0 else 0.0

        return round(avg_win / avg_loss, 3)


class AvgHoldingPeriod(PortfolioStatistic):
    """
    Average Holding Period - Mean time trades are held.
    """

    def calculate_from_trades(self, trades: List[Trade]) -> Optional[str]:
        if not trades:
            return None

        total_seconds = sum(t.holding_period.total_seconds() for t in trades)
        avg_seconds = total_seconds / len(trades)

        # Format nicely
        hours = int(avg_seconds // 3600)
        minutes = int((avg_seconds % 3600) // 60)

        if hours > 24:
            days = hours // 24
            hours = hours % 24
            return f"{days}d {hours}h {minutes}m"
        return f"{hours}h {minutes}m"


class TradeCount(PortfolioStatistic):
    """
    Total Trade Count.
    """

    def calculate_from_trades(self, trades: List[Trade]) -> Optional[int]:
        return len(trades) if trades else 0


class MaxConsecutiveWins(PortfolioStatistic):
    """
    Maximum Consecutive Wins - Longest winning streak.
    """

    def calculate_from_trades(self, trades: List[Trade]) -> Optional[int]:
        if not trades:
            return 0

        max_streak = 0
        current_streak = 0

        for trade in trades:
            if trade.is_winner:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0

        return max_streak


class MaxConsecutiveLosses(PortfolioStatistic):
    """
    Maximum Consecutive Losses - Longest losing streak.
    """

    def calculate_from_trades(self, trades: List[Trade]) -> Optional[int]:
        if not trades:
            return 0

        max_streak = 0
        current_streak = 0

        for trade in trades:
            if trade.is_loser:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0

        return max_streak


# Default statistics set
DEFAULT_STATISTICS = [
    WinRate(),
    ProfitFactor(),
    Expectancy(),
    SharpeRatio(),
    SortinoRatio(),
    CalmarRatio(),
    MaxDrawdown(),
    CAGR(),
    ReturnsVolatility(),
    ReturnsAverage(),
    ReturnsAvgWin(),
    ReturnsAvgLoss(),
    RiskReturnRatio(),
    WinnerAverage(),
    WinnerMax(),
    LoserAverage(),
    LoserMax(),
    LongRatio(),
    PayoffRatio(),
    TradeCount(),
    MaxConsecutiveWins(),
    MaxConsecutiveLosses(),
]
