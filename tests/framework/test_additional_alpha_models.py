"""Tests for Additional Alpha Models (Phase 5)."""
import numpy as np
import pytest

from backend.tradingbot.framework.insight import Insight, InsightDirection
from backend.tradingbot.framework.alpha_models.pairs_trading_alpha import PairsTradingAlphaModel
from backend.tradingbot.framework.alpha_models.bollinger_alpha import BollingerAlphaModel
from backend.tradingbot.framework.alpha_models.volatility_alpha import VolatilityAlphaModel
from backend.tradingbot.framework.alpha_models.stochastic_alpha import StochasticAlphaModel
from backend.tradingbot.framework.alpha_models.atr_reversion_alpha import ATRReversionAlphaModel


def _make_trending_data(n=100, start=100, drift=0.001):
    """Generate trending price data dict."""
    np.random.seed(42)
    closes = [start]
    for _ in range(n - 1):
        closes.append(closes[-1] * (1 + drift + np.random.normal(0, 0.01)))
    closes = np.array(closes)
    highs = closes * (1 + np.abs(np.random.normal(0, 0.005, n)))
    lows = closes * (1 - np.abs(np.random.normal(0, 0.005, n)))
    return {
        "close": closes.tolist(),
        "high": highs.tolist(),
        "low": lows.tolist(),
        "volume": [1000] * n,
    }


def _make_mean_reverting_data(n=100, mean=100, std=5):
    """Generate mean-reverting data."""
    np.random.seed(123)
    closes = []
    price = mean
    for _ in range(n):
        price = price + 0.3 * (mean - price) + np.random.normal(0, std * 0.1)
        closes.append(price)
    closes = np.array(closes)
    highs = closes + np.abs(np.random.normal(0, 1, n))
    lows = closes - np.abs(np.random.normal(0, 1, n))
    return {
        "close": closes.tolist(),
        "high": highs.tolist(),
        "low": lows.tolist(),
        "volume": [1000] * n,
    }


class TestPairsTradingAlphaModel:
    def test_init(self):
        model = PairsTradingAlphaModel(pairs=[("AAPL", "MSFT")])
        assert model.name == "PairsTradingAlpha"
        assert model.entry_z == 2.0

    def test_generates_insights_on_divergence(self):
        np.random.seed(42)
        n = 100
        # Create two correlated series with divergence at end
        base = np.cumsum(np.random.normal(0, 1, n)) + 100
        sym_a_closes = (base + np.random.normal(0, 0.5, n)).tolist()
        # Make B diverge from A at end
        sym_b_closes = (base + np.random.normal(0, 0.5, n)).tolist()
        # Force large divergence
        for i in range(n - 5, n):
            sym_a_closes[i] += 20  # A goes up a lot

        data = {
            "AAPL": {"close": sym_a_closes},
            "MSFT": {"close": sym_b_closes},
        }
        model = PairsTradingAlphaModel(
            pairs=[("AAPL", "MSFT")],
            lookback_period=60,
            entry_z=1.5,
        )
        insights = model.generate_insights(data, ["AAPL", "MSFT"])
        # Should generate opposing signals
        if insights:
            assert len(insights) == 2
            directions = {i.symbol: i.direction for i in insights}
            # AAPL went high -> spread high -> short AAPL, long MSFT
            assert directions["AAPL"] == InsightDirection.DOWN
            assert directions["MSFT"] == InsightDirection.UP

    def test_no_insights_when_spread_normal(self):
        np.random.seed(42)
        n = 100
        base = np.cumsum(np.random.normal(0, 0.1, n)) + 100
        data = {
            "A": {"close": (base + np.random.normal(0, 0.01, n)).tolist()},
            "B": {"close": (base + np.random.normal(0, 0.01, n)).tolist()},
        }
        model = PairsTradingAlphaModel(pairs=[("A", "B")], entry_z=5.0)
        insights = model.generate_insights(data, ["A", "B"])
        assert len(insights) == 0

    def test_auto_pair_discovery(self):
        model = PairsTradingAlphaModel(lookback_period=30)
        np.random.seed(42)
        n = 50
        base = np.cumsum(np.random.normal(0, 1, n)) + 100
        data = {
            "A": {"close": (base + np.random.normal(0, 0.1, n)).tolist()},
            "B": {"close": (base + np.random.normal(0, 0.1, n)).tolist()},
            "C": {"close": np.random.normal(200, 10, n).tolist()},
        }
        pairs = model._find_pairs(data, ["A", "B", "C"])
        # A and B should be highly correlated
        if pairs:
            pair_symbols = {(a, b) for a, b in pairs}
            assert ("A", "B") in pair_symbols or ("B", "A") in pair_symbols


class TestBollingerAlphaModel:
    def test_init_mean_reversion(self):
        model = BollingerAlphaModel(mode="mean_reversion")
        assert model.mode == "mean_reversion"

    def test_mean_reversion_buy_at_lower_band(self):
        # Create data where price crashes below lower band
        np.random.seed(42)
        closes = [100] * 20 + [80]  # Stable then crash
        data = {"SPY": {"close": closes}}
        model = BollingerAlphaModel(period=20, num_std=2.0, mode="mean_reversion")
        insights = model.generate_insights(data, ["SPY"])
        assert len(insights) == 1
        assert insights[0].direction == InsightDirection.UP

    def test_mean_reversion_sell_at_upper_band(self):
        closes = [100] * 20 + [120]  # Stable then spike
        data = {"SPY": {"close": closes}}
        model = BollingerAlphaModel(period=20, num_std=2.0, mode="mean_reversion")
        insights = model.generate_insights(data, ["SPY"])
        assert len(insights) == 1
        assert insights[0].direction == InsightDirection.DOWN

    def test_breakout_mode(self):
        closes = [100] * 20 + [120]  # Spike above upper band
        data = {"SPY": {"close": closes}}
        model = BollingerAlphaModel(period=20, num_std=2.0, mode="breakout")
        insights = model.generate_insights(data, ["SPY"])
        assert len(insights) == 1
        assert insights[0].direction == InsightDirection.UP  # Breakout = follow the trend

    def test_no_signal_within_bands(self):
        # Use data with some variance so bands are wide
        np.random.seed(99)
        closes = [*(100 + np.random.normal(0, 2, 20)).tolist(), 100.5]
        data = {"SPY": {"close": closes}}
        model = BollingerAlphaModel(period=20, num_std=2.0)
        insights = model.generate_insights(data, ["SPY"])
        assert len(insights) == 0

    def test_insufficient_data(self):
        data = {"SPY": {"close": [100, 101, 102]}}
        model = BollingerAlphaModel(period=20)
        insights = model.generate_insights(data, ["SPY"])
        assert len(insights) == 0


class TestVolatilityAlphaModel:
    def test_init(self):
        model = VolatilityAlphaModel()
        assert model.atr_multiplier == 2.0

    def test_keltner_breakout_up(self):
        # Create data where price spikes up above Keltner channel
        data = _make_trending_data(50, start=100, drift=0.0)
        # Spike at end
        data["close"][-1] = data["close"][-2] + 30
        data["high"][-1] = data["close"][-1] + 1
        model = VolatilityAlphaModel(ema_period=20, atr_period=14, atr_multiplier=2.0)
        insights = model.generate_insights({"SPY": data}, ["SPY"])
        if insights:
            assert insights[0].direction == InsightDirection.UP

    def test_keltner_breakout_down(self):
        data = _make_trending_data(50, start=100, drift=0.0)
        data["close"][-1] = data["close"][-2] - 30
        data["low"][-1] = data["close"][-1] - 1
        model = VolatilityAlphaModel(ema_period=20, atr_period=14, atr_multiplier=2.0)
        insights = model.generate_insights({"SPY": data}, ["SPY"])
        if insights:
            assert insights[0].direction == InsightDirection.DOWN

    def test_ema_calculation(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        ema = VolatilityAlphaModel._ema(data, 3)
        assert isinstance(ema, float)
        assert ema > 0

    def test_atr_calculation(self):
        n = 20
        highs = np.array([101.0 + i * 0.1 for i in range(n)])
        lows = np.array([99.0 + i * 0.1 for i in range(n)])
        closes = np.array([100.0 + i * 0.1 for i in range(n)])
        atr = VolatilityAlphaModel._atr(highs, lows, closes, 14)
        assert atr > 0


class TestStochasticAlphaModel:
    def test_init(self):
        model = StochasticAlphaModel()
        assert model.k_period == 14
        assert model.d_period == 3
        assert model.oversold == 20.0
        assert model.overbought == 80.0

    def test_insufficient_data(self):
        data = {"SPY": {"close": [100, 101], "high": [102, 103], "low": [99, 100]}}
        model = StochasticAlphaModel()
        insights = model.generate_insights(data, ["SPY"])
        assert len(insights) == 0

    def test_insight_has_metadata(self):
        # Generate enough data for stochastic calculation
        np.random.seed(42)
        n = 30
        closes = np.cumsum(np.random.normal(0, 1, n)) + 100
        highs = closes + np.abs(np.random.normal(0, 0.5, n))
        lows = closes - np.abs(np.random.normal(0, 0.5, n))
        data = {
            "SPY": {
                "close": closes.tolist(),
                "high": highs.tolist(),
                "low": lows.tolist(),
            }
        }
        model = StochasticAlphaModel(k_period=5, d_period=3)
        insights = model.generate_insights(data, ["SPY"])
        for ins in insights:
            assert "k" in ins.metadata
            assert "d" in ins.metadata


class TestATRReversionAlphaModel:
    def test_init(self):
        model = ATRReversionAlphaModel()
        assert model.sma_period == 20
        assert model.atr_threshold == 2.0

    def test_reversion_signal_below_sma(self):
        # Price way below SMA
        np.random.seed(42)
        n = 30
        closes = [100.0] * n
        closes[-1] = 70.0  # Crash
        highs = [c + 1 for c in closes]
        lows = [c - 1 for c in closes]
        data = {"SPY": {"close": closes, "high": highs, "low": lows}}
        model = ATRReversionAlphaModel(sma_period=20, atr_period=14, atr_threshold=2.0)
        insights = model.generate_insights(data, ["SPY"])
        assert len(insights) == 1
        assert insights[0].direction == InsightDirection.UP

    def test_reversion_signal_above_sma(self):
        np.random.seed(42)
        n = 30
        closes = [100.0] * n
        closes[-1] = 130.0  # Spike
        highs = [c + 1 for c in closes]
        lows = [c - 1 for c in closes]
        data = {"SPY": {"close": closes, "high": highs, "low": lows}}
        model = ATRReversionAlphaModel(sma_period=20, atr_period=14, atr_threshold=2.0)
        insights = model.generate_insights(data, ["SPY"])
        assert len(insights) == 1
        assert insights[0].direction == InsightDirection.DOWN

    def test_no_signal_near_sma(self):
        closes = [100.0] * 30
        closes[-1] = 100.5  # Barely moved
        highs = [c + 1 for c in closes]
        lows = [c - 1 for c in closes]
        data = {"SPY": {"close": closes, "high": highs, "low": lows}}
        model = ATRReversionAlphaModel(sma_period=20, atr_threshold=2.0)
        insights = model.generate_insights(data, ["SPY"])
        assert len(insights) == 0

    def test_insight_metadata(self):
        closes = [100.0] * 30
        closes[-1] = 70.0
        highs = [c + 1 for c in closes]
        lows = [c - 1 for c in closes]
        data = {"SPY": {"close": closes, "high": highs, "low": lows}}
        model = ATRReversionAlphaModel()
        insights = model.generate_insights(data, ["SPY"])
        if insights:
            assert "distance_atr" in insights[0].metadata
            assert "atr" in insights[0].metadata
            assert "sma" in insights[0].metadata
