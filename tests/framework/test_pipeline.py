"""Tests for Factor Pipeline (Phase 8)."""
import numpy as np
import pandas as pd
import pytest

from backend.tradingbot.framework.pipeline import (
    Factor, AverageDollarVolume, Returns, Volatility, MeanReversion,
    ScreenFilter, CombinedFilter, TopFilter, BottomFilter, PercentileFilter,
    Pipeline,
)


def _make_data(n_symbols=5, n_days=100):
    """Create sample data for pipeline testing."""
    np.random.seed(42)
    data = {}
    for i in range(n_symbols):
        sym = f"SYM{i}"
        base = 50 + i * 20
        closes = np.cumsum(np.random.normal(0.1, 1, n_days)) + base
        volumes = np.random.uniform(100000, 1000000, n_days)
        data[sym] = {
            "close": closes.tolist(),
            "volume": volumes.tolist(),
        }
    return data


# --- Factor Tests ---

class TestAverageDollarVolume:
    def test_computes_for_all_symbols(self):
        data = _make_data()
        factor = AverageDollarVolume(window=20)
        result = factor.compute(data)
        assert isinstance(result, pd.Series)
        assert len(result) == 5

    def test_all_values_positive(self):
        data = _make_data()
        factor = AverageDollarVolume(window=20)
        result = factor.compute(data)
        assert (result > 0).all()

    def test_skips_insufficient_data(self):
        data = {"SYM": {"close": [100, 101], "volume": [1000, 1000]}}
        factor = AverageDollarVolume(window=20)
        result = factor.compute(data)
        assert len(result) == 0


class TestReturns:
    def test_computes_returns(self):
        data = _make_data()
        factor = Returns(window=20)
        result = factor.compute(data)
        assert isinstance(result, pd.Series)
        assert len(result) == 5

    def test_positive_trending(self):
        # Strongly upward trending data
        closes = list(range(100, 200))
        data = {"UP": {"close": closes}}
        factor = Returns(window=20)
        result = factor.compute(data)
        assert result["UP"] > 0


class TestVolatility:
    def test_computes_volatility(self):
        data = _make_data()
        factor = Volatility(window=20)
        result = factor.compute(data)
        assert isinstance(result, pd.Series)
        assert (result > 0).all()

    def test_annualized(self):
        data = _make_data()
        factor = Volatility(window=20)
        result = factor.compute(data)
        # Annualized vol should be reasonable (not daily-scale)
        assert (result > 0.01).all()


class TestMeanReversion:
    def test_computes_distance_from_mean(self):
        data = _make_data()
        factor = MeanReversion(window=20)
        result = factor.compute(data)
        assert isinstance(result, pd.Series)

    def test_above_mean_positive(self):
        closes = [100] * 19 + [110]
        data = {"SYM": {"close": closes}}
        factor = MeanReversion(window=20)
        result = factor.compute(data)
        assert result["SYM"] > 0

    def test_below_mean_negative(self):
        closes = [100] * 19 + [90]
        data = {"SYM": {"close": closes}}
        factor = MeanReversion(window=20)
        result = factor.compute(data)
        assert result["SYM"] < 0


# --- Filter Tests ---

class TestScreenFilter:
    def test_greater_than(self):
        factor = AverageDollarVolume(window=20)
        filt = factor > 500000
        assert isinstance(filt, ScreenFilter)
        values = pd.Series({"A": 600000, "B": 400000, "C": 700000})
        mask = filt.apply(values)
        assert mask["A"] == True
        assert mask["B"] == False
        assert mask["C"] == True

    def test_less_than(self):
        factor = Returns(window=20)
        filt = factor < 0.1
        values = pd.Series({"A": 0.05, "B": 0.2})
        mask = filt.apply(values)
        assert mask["A"] == True
        assert mask["B"] == False


class TestTopFilter:
    def test_selects_top_n(self):
        factor = Returns(window=20)
        filt = factor.top(2)
        values = pd.Series({"A": 0.1, "B": 0.3, "C": 0.2, "D": 0.05})
        mask = filt.apply(values)
        assert mask["B"] == True
        assert mask["C"] == True
        assert mask.sum() == 2


class TestBottomFilter:
    def test_selects_bottom_n(self):
        factor = Volatility(window=20)
        filt = factor.bottom(2)
        values = pd.Series({"A": 0.1, "B": 0.3, "C": 0.05, "D": 0.2})
        mask = filt.apply(values)
        assert mask["C"] == True
        assert mask["A"] == True
        assert mask.sum() == 2


class TestPercentileFilter:
    def test_percentile_range(self):
        factor = Returns(window=20)
        filt = factor.percentile_between(25, 75)
        values = pd.Series({"A": 10, "B": 50, "C": 90, "D": 30, "E": 70})
        mask = filt.apply(values)
        # Middle 50% should be selected
        assert mask.sum() >= 2


# --- Pipeline Tests ---

class TestPipeline:
    def test_add_and_run(self):
        data = _make_data()
        pipe = Pipeline()
        pipe.add(AverageDollarVolume(window=20), "adv")
        pipe.add(Returns(window=20), "momentum")
        result = pipe.run(data)
        assert isinstance(result, pd.DataFrame)
        assert "adv" in result.columns
        assert "momentum" in result.columns

    def test_screen_filter(self):
        data = _make_data()
        pipe = Pipeline()
        adv = AverageDollarVolume(window=20)
        # Use factor's own name as key so screen lookup works
        pipe.add(adv, adv.name)
        pipe.add(Returns(window=20), "momentum")
        pipe.set_screen(adv > 0)  # All should pass since dollar volume > 0
        result = pipe.run(data)
        assert len(result) > 0

    def test_top_filter_screen(self):
        data = _make_data()
        pipe = Pipeline()
        ret_factor = Returns(window=20)
        pipe.add(AverageDollarVolume(window=20), "adv")
        pipe.add(ret_factor, "momentum")
        pipe.set_screen(ret_factor.top(3))
        result = pipe.run(data)
        assert len(result) <= 3

    def test_empty_pipeline(self):
        data = _make_data()
        pipe = Pipeline()
        result = pipe.run(data)
        assert isinstance(result, pd.DataFrame)
        assert len(result.columns) == 0

    def test_get_factor(self):
        pipe = Pipeline()
        adv = AverageDollarVolume(window=20)
        pipe.add(adv, "adv")
        assert pipe.get("adv") is adv

    def test_missing_factor_raises(self):
        pipe = Pipeline()
        with pytest.raises(KeyError):
            pipe.get("nonexistent")

    def test_multiple_factors(self):
        data = _make_data(n_symbols=10)
        pipe = Pipeline()
        pipe.add(AverageDollarVolume(window=20), "adv")
        pipe.add(Returns(window=20), "momentum")
        pipe.add(Volatility(window=20), "vol")
        pipe.add(MeanReversion(window=20), "mean_rev")
        result = pipe.run(data)
        assert len(result.columns) == 4
