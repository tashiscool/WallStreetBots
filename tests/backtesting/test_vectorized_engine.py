"""Tests for Vectorized Backtesting Engine."""
import numpy as np
import pytest

from backend.tradingbot.backtesting.vectorized_engine import (
    VectorizedBacktestEngine, VectorizedResult,
)


def _trending_prices(n=200, start=100):
    """Generate upward-trending prices for testing."""
    np.random.seed(42)
    returns = np.random.normal(0.0005, 0.01, n - 1)
    prices = np.empty(n)
    prices[0] = start
    for i in range(1, n):
        prices[i] = prices[i - 1] * (1 + returns[i - 1])
    return prices


class TestVectorizedBacktestEngine:
    def test_init(self):
        prices = _trending_prices()
        engine = VectorizedBacktestEngine(prices)
        assert engine.initial_capital == 100000.0
        assert len(engine.returns) == len(prices) - 1

    def test_run_buy_and_hold(self):
        prices = _trending_prices()
        engine = VectorizedBacktestEngine(prices, commission=0.0, slippage=0.0)
        signals = np.ones(len(prices))  # Always long
        result = engine.run(signals)

        assert isinstance(result, VectorizedResult)
        assert isinstance(result.total_return, float)
        assert isinstance(result.sharpe_ratio, float)
        assert isinstance(result.max_drawdown, float)
        assert result.max_drawdown >= 0
        assert len(result.equity_curve) == len(prices)
        assert result.equity_curve[0] == 100000.0

    def test_run_flat(self):
        prices = _trending_prices()
        engine = VectorizedBacktestEngine(prices, commission=0.0, slippage=0.0)
        signals = np.zeros(len(prices))  # Always flat
        result = engine.run(signals)

        # Flat position should have ~0 return
        assert abs(result.total_return) < 0.01
        assert result.total_trades == 0

    def test_signal_length_mismatch_raises(self):
        prices = _trending_prices(100)
        engine = VectorizedBacktestEngine(prices)
        with pytest.raises(ValueError, match="Signals length"):
            engine.run(np.ones(50))

    def test_run_with_costs(self):
        prices = _trending_prices()
        engine_no_costs = VectorizedBacktestEngine(prices, commission=0.0, slippage=0.0)
        engine_costs = VectorizedBacktestEngine(prices, commission=0.002, slippage=0.001)

        # Use signals with many position changes so costs accumulate
        signals = np.ones(len(prices))
        signals[::3] = -1  # Switch direction every 3 bars
        result_no_costs = engine_no_costs.run(signals)
        result_costs = engine_costs.run(signals)

        # Costs should reduce returns
        assert result_costs.total_return < result_no_costs.total_return

    def test_result_metrics_types(self):
        prices = _trending_prices()
        engine = VectorizedBacktestEngine(prices)
        signals = np.ones(len(prices))
        result = engine.run(signals)

        assert isinstance(result.sharpe_ratio, float)
        assert isinstance(result.sortino_ratio, float)
        assert isinstance(result.calmar_ratio, float)
        assert isinstance(result.win_rate, float)
        assert isinstance(result.profit_factor, float)
        assert isinstance(result.total_trades, int)

    def test_short_signals(self):
        prices = _trending_prices()
        engine = VectorizedBacktestEngine(prices, commission=0.0, slippage=0.0)
        long_signals = np.ones(len(prices))
        short_signals = -np.ones(len(prices))

        long_result = engine.run(long_signals)
        short_result = engine.run(short_signals)

        # On trending-up data, short should underperform long
        assert short_result.total_return < long_result.total_return


class TestParameterSweep:
    def test_basic_sweep(self):
        prices = _trending_prices()
        engine = VectorizedBacktestEngine(prices)

        def signal_fn(prices, params):
            window = params["window"]
            sma = np.convolve(prices, np.ones(window) / window, mode="same")
            return np.where(prices > sma, 1.0, -1.0)

        results = engine.parameter_sweep(
            param_grid={"window": [5, 10, 20]},
            signal_fn=signal_fn,
        )

        assert len(results) == 3
        assert all(isinstance(r, VectorizedResult) for r in results)
        # Results should be sorted by sharpe_ratio descending
        for i in range(len(results) - 1):
            assert results[i].sharpe_ratio >= results[i + 1].sharpe_ratio

    def test_sweep_stores_params(self):
        prices = _trending_prices()
        engine = VectorizedBacktestEngine(prices)

        def signal_fn(prices, params):
            return np.ones(len(prices)) * (1 if params["direction"] == "long" else -1)

        results = engine.parameter_sweep(
            param_grid={"direction": ["long", "short"]},
            signal_fn=signal_fn,
        )

        assert all(r.params for r in results)
        directions = {r.params["direction"] for r in results}
        assert directions == {"long", "short"}


class TestWalkForward:
    def test_walk_forward(self):
        prices = _trending_prices(500)
        engine = VectorizedBacktestEngine(prices)

        def signal_fn(prices, params):
            window = params["window"]
            if len(prices) < window:
                return np.zeros(len(prices))
            sma = np.convolve(prices, np.ones(window) / window, mode="same")
            return np.where(prices > sma, 1.0, 0.0)

        results = engine.walk_forward(
            signal_fn=signal_fn,
            param_grid={"window": [5, 10, 20]},
            n_splits=3,
            train_ratio=0.7,
        )

        assert isinstance(results, list)
        # Each result should be out-of-sample
        for r in results:
            assert isinstance(r, VectorizedResult)
            assert r.params  # Should have params from training


class TestVectorizedResult:
    def test_dataclass(self):
        result = VectorizedResult(
            total_return=0.15,
            sharpe_ratio=1.5,
            max_drawdown=0.10,
            win_rate=0.55,
            total_trades=100,
            profit_factor=1.8,
            calmar_ratio=1.5,
            sortino_ratio=2.0,
            equity_curve=np.array([100000, 110000, 115000]),
        )
        assert result.total_return == 0.15
        assert result.params == {}
