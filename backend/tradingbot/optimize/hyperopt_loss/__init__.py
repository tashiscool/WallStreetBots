"""
Hyperopt Loss Functions

Various objective functions for strategy optimization.
Lower values are better (minimization).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional
import numpy as np
from decimal import Decimal


@dataclass
class BacktestResult:
    """Results from a backtest run."""
    total_trades: int
    winning_trades: int
    losing_trades: int
    total_profit: float
    total_profit_pct: float
    avg_profit_pct: float
    max_drawdown: float
    max_drawdown_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    profit_factor: float
    win_rate: float
    avg_trade_duration: float  # in hours
    trade_count_long: int
    trade_count_short: int
    profit_long: float
    profit_short: float
    equity_curve: List[float]
    drawdown_curve: List[float]
    start_date: datetime
    end_date: datetime


class IHyperoptLoss(ABC):
    """Interface for hyperopt loss functions."""

    @staticmethod
    @abstractmethod
    def hyperopt_loss_function(
        results: BacktestResult,
        trade_count: int,
        min_date: datetime,
        max_date: datetime,
        config: Dict[str, Any],
        processed: Optional[Dict[str, Any]] = None,
        backtest_stats: Optional[Dict[str, Any]] = None,
    ) -> float:
        """
        Calculate loss for hyperopt optimization.

        Lower values are better (this is a minimization problem).

        Args:
            results: BacktestResult object
            trade_count: Number of trades
            min_date: Start date of backtest
            max_date: End date of backtest
            config: Strategy configuration
            processed: Processed data (optional)
            backtest_stats: Additional backtest statistics (optional)

        Returns:
            float: Loss value (lower is better)
        """
        pass


class SharpeHyperoptLoss(IHyperoptLoss):
    """
    Optimize for Sharpe Ratio.

    Sharpe = (Return - Risk-free rate) / Std Dev of Returns
    """

    @staticmethod
    def hyperopt_loss_function(
        results: BacktestResult,
        trade_count: int,
        min_date: datetime,
        max_date: datetime,
        config: Dict[str, Any],
        processed: Optional[Dict[str, Any]] = None,
        backtest_stats: Optional[Dict[str, Any]] = None,
    ) -> float:
        # Penalize low trade count
        if trade_count < 10:
            return 10.0

        sharpe = results.sharpe_ratio if hasattr(results, 'sharpe_ratio') else 0.0

        # Negate because we minimize (higher Sharpe = lower loss)
        return -sharpe


class SortinoHyperoptLoss(IHyperoptLoss):
    """
    Optimize for Sortino Ratio.

    Sortino = (Return - Risk-free rate) / Downside Deviation
    Only penalizes downside volatility.
    """

    @staticmethod
    def hyperopt_loss_function(
        results: BacktestResult,
        trade_count: int,
        min_date: datetime,
        max_date: datetime,
        config: Dict[str, Any],
        processed: Optional[Dict[str, Any]] = None,
        backtest_stats: Optional[Dict[str, Any]] = None,
    ) -> float:
        if trade_count < 10:
            return 10.0

        sortino = results.sortino_ratio if hasattr(results, 'sortino_ratio') else 0.0
        return -sortino


class MaxDrawdownHyperoptLoss(IHyperoptLoss):
    """
    Optimize to minimize maximum drawdown.
    """

    @staticmethod
    def hyperopt_loss_function(
        results: BacktestResult,
        trade_count: int,
        min_date: datetime,
        max_date: datetime,
        config: Dict[str, Any],
        processed: Optional[Dict[str, Any]] = None,
        backtest_stats: Optional[Dict[str, Any]] = None,
    ) -> float:
        if trade_count < 10:
            return 1.0  # 100% drawdown equivalent

        max_dd = abs(results.max_drawdown_pct) if hasattr(results, 'max_drawdown_pct') else 1.0
        return max_dd


class CalmarHyperoptLoss(IHyperoptLoss):
    """
    Optimize for Calmar Ratio.

    Calmar = Annualized Return / Max Drawdown
    """

    @staticmethod
    def hyperopt_loss_function(
        results: BacktestResult,
        trade_count: int,
        min_date: datetime,
        max_date: datetime,
        config: Dict[str, Any],
        processed: Optional[Dict[str, Any]] = None,
        backtest_stats: Optional[Dict[str, Any]] = None,
    ) -> float:
        if trade_count < 10:
            return 10.0

        calmar = results.calmar_ratio if hasattr(results, 'calmar_ratio') else 0.0
        return -calmar


class ProfitHyperoptLoss(IHyperoptLoss):
    """
    Optimize for pure profit.
    """

    @staticmethod
    def hyperopt_loss_function(
        results: BacktestResult,
        trade_count: int,
        min_date: datetime,
        max_date: datetime,
        config: Dict[str, Any],
        processed: Optional[Dict[str, Any]] = None,
        backtest_stats: Optional[Dict[str, Any]] = None,
    ) -> float:
        if trade_count < 10:
            return 100.0

        profit = results.total_profit_pct if hasattr(results, 'total_profit_pct') else 0.0
        return -profit  # Negate for minimization


class WinRateHyperoptLoss(IHyperoptLoss):
    """
    Optimize for win rate while maintaining profitability.
    """

    @staticmethod
    def hyperopt_loss_function(
        results: BacktestResult,
        trade_count: int,
        min_date: datetime,
        max_date: datetime,
        config: Dict[str, Any],
        processed: Optional[Dict[str, Any]] = None,
        backtest_stats: Optional[Dict[str, Any]] = None,
    ) -> float:
        if trade_count < 10:
            return 1.0

        win_rate = results.win_rate if hasattr(results, 'win_rate') else 0.0

        # Penalize if not profitable
        if results.total_profit_pct <= 0:
            return 1.0 - (win_rate * 0.5)  # Still reward high win rate somewhat

        return -win_rate


class ShortTradeDurationHyperoptLoss(IHyperoptLoss):
    """
    Optimize for shorter trade duration while maintaining profit.
    Good for day trading strategies.
    """

    @staticmethod
    def hyperopt_loss_function(
        results: BacktestResult,
        trade_count: int,
        min_date: datetime,
        max_date: datetime,
        config: Dict[str, Any],
        processed: Optional[Dict[str, Any]] = None,
        backtest_stats: Optional[Dict[str, Any]] = None,
    ) -> float:
        if trade_count < 10:
            return 10.0

        # Must be profitable
        if results.total_profit_pct <= 0:
            return 10.0

        # Shorter duration = better (but must be profitable)
        avg_duration = results.avg_trade_duration if hasattr(results, 'avg_trade_duration') else 24.0

        # Normalize duration (1 hour = good, 24 hours = neutral, 168 hours (1 week) = bad)
        duration_score = avg_duration / 24.0

        # Combine with profit requirement
        profit_score = max(0, results.total_profit_pct) / 100.0

        return duration_score - profit_score


class MultiMetricHyperoptLoss(IHyperoptLoss):
    """
    Weighted combination of multiple metrics.

    Default weights:
    - Sharpe Ratio: 30%
    - Sortino Ratio: 20%
    - Max Drawdown: 20%
    - Profit: 20%
    - Win Rate: 10%
    """

    WEIGHTS = {
        'sharpe': 0.30,
        'sortino': 0.20,
        'max_drawdown': 0.20,
        'profit': 0.20,
        'win_rate': 0.10,
    }

    @staticmethod
    def hyperopt_loss_function(
        results: BacktestResult,
        trade_count: int,
        min_date: datetime,
        max_date: datetime,
        config: Dict[str, Any],
        processed: Optional[Dict[str, Any]] = None,
        backtest_stats: Optional[Dict[str, Any]] = None,
    ) -> float:
        if trade_count < 10:
            return 10.0

        weights = MultiMetricHyperoptLoss.WEIGHTS

        # Normalize each metric to 0-1 scale
        sharpe_score = min(max(results.sharpe_ratio / 3.0, -1), 1) if results.sharpe_ratio else 0
        sortino_score = min(max(results.sortino_ratio / 3.0, -1), 1) if results.sortino_ratio else 0
        drawdown_score = 1.0 - min(abs(results.max_drawdown_pct), 1.0)  # Lower DD = higher score
        profit_score = min(max(results.total_profit_pct / 100.0, -1), 1)
        winrate_score = results.win_rate if results.win_rate else 0

        # Weighted sum (higher = better, so negate for minimization)
        total_score = (
            weights['sharpe'] * sharpe_score +
            weights['sortino'] * sortino_score +
            weights['max_drawdown'] * drawdown_score +
            weights['profit'] * profit_score +
            weights['win_rate'] * winrate_score
        )

        return -total_score


class ExpectancyHyperoptLoss(IHyperoptLoss):
    """
    Optimize for trade expectancy.

    Expectancy = (Win Rate * Avg Win) - (Loss Rate * Avg Loss)
    """

    @staticmethod
    def hyperopt_loss_function(
        results: BacktestResult,
        trade_count: int,
        min_date: datetime,
        max_date: datetime,
        config: Dict[str, Any],
        processed: Optional[Dict[str, Any]] = None,
        backtest_stats: Optional[Dict[str, Any]] = None,
    ) -> float:
        if trade_count < 10:
            return 10.0

        win_rate = results.win_rate if results.win_rate else 0
        loss_rate = 1 - win_rate

        # Calculate average win/loss
        if results.winning_trades > 0:
            avg_win = results.profit_long / results.winning_trades if results.winning_trades else 0
        else:
            avg_win = 0

        if results.losing_trades > 0:
            avg_loss = abs(results.profit_short / results.losing_trades) if results.losing_trades else 0
        else:
            avg_loss = 0

        expectancy = (win_rate * avg_win) - (loss_rate * avg_loss)

        return -expectancy  # Negate for minimization


# Export all loss functions
__all__ = [
    'IHyperoptLoss',
    'BacktestResult',
    'SharpeHyperoptLoss',
    'SortinoHyperoptLoss',
    'MaxDrawdownHyperoptLoss',
    'CalmarHyperoptLoss',
    'ProfitHyperoptLoss',
    'WinRateHyperoptLoss',
    'ShortTradeDurationHyperoptLoss',
    'MultiMetricHyperoptLoss',
    'ExpectancyHyperoptLoss',
]
