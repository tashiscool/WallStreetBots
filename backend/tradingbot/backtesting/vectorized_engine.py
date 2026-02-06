"""
Vectorized Backtesting Engine

100x faster than event-driven backtesting for parameter sweeps.
All operations use numpy vectorized arithmetic on full price arrays.

Inspired by VectorBT's portfolio engine.
"""

from dataclasses import dataclass, field
from itertools import product
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class VectorizedResult:
    """Result from a vectorized backtest run."""

    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    profit_factor: float
    calmar_ratio: float
    sortino_ratio: float
    equity_curve: np.ndarray
    params: Dict[str, Any] = field(default_factory=dict)


class VectorizedBacktestEngine:
    """
    Vectorized backtesting engine for ultra-fast parameter sweeps.

    Instead of stepping through bars one by one, computes all metrics
    using numpy array operations on the full price/signal series.

    Usage:
        engine = VectorizedBacktestEngine(
            prices=close_prices,
            initial_capital=100000,
        )

        # Single run with pre-computed signals
        result = engine.run(signals)

        # Parameter sweep
        results = engine.parameter_sweep(
            param_grid={'sma_fast': [5,10,20], 'sma_slow': [20,50,100]},
            signal_fn=lambda prices, params: sma_crossover(prices, params),
        )
    """

    def __init__(
        self,
        prices: np.ndarray,
        initial_capital: float = 100000.0,
        commission: float = 0.001,
        slippage: float = 0.0005,
        risk_free_rate: float = 0.02,
    ):
        self.prices = np.asarray(prices, dtype=np.float64)
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.risk_free_rate = risk_free_rate / 252  # Daily

        # Pre-compute returns
        self.returns = np.diff(self.prices) / self.prices[:-1]

    def run(
        self,
        signals: np.ndarray,
        position_size: float = 1.0,
    ) -> VectorizedResult:
        """Run backtest with pre-computed signals.

        Args:
            signals: Array of +1 (long), -1 (short), 0 (flat).
                     Must be same length as prices.
            position_size: Fraction of capital per position (0-1).

        Returns:
            VectorizedResult with metrics and equity curve.
        """
        signals = np.asarray(signals, dtype=np.float64)
        if len(signals) != len(self.prices):
            raise ValueError(
                f"Signals length ({len(signals)}) must match prices ({len(self.prices)})"
            )

        # Position changes (where we trade)
        position_changes = np.diff(signals, prepend=0)
        trades = np.abs(position_changes) > 0

        # Transaction costs per trade
        costs = np.where(trades, self.commission + self.slippage, 0.0)

        # Strategy returns = signal * market_return - costs
        strategy_returns = signals[1:] * self.returns * position_size - costs[1:]

        # Equity curve
        equity = self.initial_capital * np.cumprod(1 + strategy_returns)
        equity = np.insert(equity, 0, self.initial_capital)

        return self._compute_metrics(strategy_returns, equity, signals)

    def parameter_sweep(
        self,
        param_grid: Dict[str, List[Any]],
        signal_fn: Callable[[np.ndarray, Dict[str, Any]], np.ndarray],
        position_size: float = 1.0,
        sort_by: str = "sharpe_ratio",
    ) -> List[VectorizedResult]:
        """Run backtest across all parameter combinations.

        Args:
            param_grid: Dict of param_name -> list of values to try.
            signal_fn: Function(prices, params) -> signals array.
            position_size: Fraction of capital per position.
            sort_by: Metric to sort results by (descending).

        Returns:
            List of VectorizedResult sorted by sort_by metric.
        """
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        results = []

        for combo in product(*values):
            params = dict(zip(keys, combo))

            try:
                signals = signal_fn(self.prices, params)
                result = self.run(signals, position_size)
                result.params = params
                results.append(result)
            except Exception:
                continue

        # Sort by metric
        results.sort(key=lambda r: getattr(r, sort_by, 0), reverse=True)
        return results

    def walk_forward(
        self,
        signal_fn: Callable[[np.ndarray, Dict[str, Any]], np.ndarray],
        param_grid: Dict[str, List[Any]],
        n_splits: int = 5,
        train_ratio: float = 0.7,
    ) -> List[VectorizedResult]:
        """Walk-forward optimization: train/test split across time.

        Args:
            signal_fn: Signal generation function.
            param_grid: Parameters to optimize.
            n_splits: Number of train/test splits.
            train_ratio: Fraction of each window for training.

        Returns:
            List of out-of-sample VectorizedResult for each split.
        """
        n = len(self.prices)
        window_size = n // n_splits
        oos_results = []

        for i in range(n_splits):
            start = i * window_size
            end = min(start + window_size, n)
            split = int(start + (end - start) * train_ratio)

            # Train: find best params
            train_prices = self.prices[start:split]
            train_engine = VectorizedBacktestEngine(
                train_prices, self.initial_capital,
                self.commission, self.slippage,
            )
            train_results = train_engine.parameter_sweep(
                param_grid, signal_fn, sort_by="sharpe_ratio"
            )

            if not train_results:
                continue

            best_params = train_results[0].params

            # Test: evaluate best params on out-of-sample
            test_prices = self.prices[split:end]
            test_engine = VectorizedBacktestEngine(
                test_prices, self.initial_capital,
                self.commission, self.slippage,
            )
            test_signals = signal_fn(test_prices, best_params)
            test_result = test_engine.run(test_signals)
            test_result.params = best_params
            oos_results.append(test_result)

        return oos_results

    def _compute_metrics(
        self,
        returns: np.ndarray,
        equity: np.ndarray,
        signals: np.ndarray,
    ) -> VectorizedResult:
        """Compute all performance metrics from returns."""
        # Total return
        total_return = (equity[-1] / equity[0]) - 1

        # Sharpe ratio
        mean_ret = np.mean(returns)
        std_ret = np.std(returns)
        sharpe = (
            (mean_ret - self.risk_free_rate) / (std_ret + 1e-10) * np.sqrt(252)
        )

        # Sortino ratio (downside deviation)
        downside = returns[returns < 0]
        downside_std = np.std(downside) if len(downside) > 0 else 1e-10
        sortino = (mean_ret - self.risk_free_rate) / (downside_std + 1e-10) * np.sqrt(252)

        # Max drawdown
        running_max = np.maximum.accumulate(equity)
        drawdowns = (equity - running_max) / (running_max + 1e-10)
        max_drawdown = abs(np.min(drawdowns))

        # Calmar ratio
        calmar = total_return / (max_drawdown + 1e-10)

        # Trade statistics
        position_changes = np.diff(signals, prepend=0)
        entries = np.where(np.abs(position_changes) > 0)[0]
        total_trades = len(entries)

        # Win rate and profit factor from trade returns
        if total_trades > 1:
            trade_returns = []
            for i in range(len(entries) - 1):
                start_idx = entries[i]
                end_idx = entries[i + 1]
                if start_idx < len(returns) and end_idx <= len(returns):
                    tr = np.sum(returns[start_idx:end_idx])
                    trade_returns.append(tr)

            if trade_returns:
                trade_returns = np.array(trade_returns)
                wins = trade_returns[trade_returns > 0]
                losses = trade_returns[trade_returns < 0]
                win_rate = len(wins) / len(trade_returns) if len(trade_returns) > 0 else 0
                gross_profit = np.sum(wins) if len(wins) > 0 else 0
                gross_loss = abs(np.sum(losses)) if len(losses) > 0 else 1e-10
                profit_factor = gross_profit / gross_loss
            else:
                win_rate = 0.0
                profit_factor = 0.0
        else:
            win_rate = 0.0
            profit_factor = 0.0

        return VectorizedResult(
            total_return=total_return,
            sharpe_ratio=sharpe,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            total_trades=total_trades,
            profit_factor=profit_factor,
            calmar_ratio=calmar,
            sortino_ratio=sortino,
            equity_curve=equity,
        )
