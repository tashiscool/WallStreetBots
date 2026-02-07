"""Tests for OptionsFlowAlphaModel."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from backend.tradingbot.framework.alpha_models.options_flow_alpha import (
    OptionsFlowAlphaModel,
)
from backend.tradingbot.framework.insight import InsightDirection


@pytest.fixture
def model():
    return OptionsFlowAlphaModel(
        pc_ratio_bullish=0.5,
        pc_ratio_bearish=1.5,
        volume_oi_spike=3.0,
        block_trade_threshold=1000,
        min_total_volume=100,
    )


def _make_chain(calls, puts):
    """Build a mock option chain dict."""
    return {"calls": calls, "puts": puts}


def _contract(volume=100, open_interest=500, strike=100.0):
    return {"volume": volume, "open_interest": open_interest, "strike": strike}


class TestPutCallRatio:
    def test_low_pc_ratio_bullish(self, model):
        """Low put/call ratio → bullish (UP)."""
        chain = _make_chain(
            calls=[_contract(volume=500), _contract(volume=400)],
            puts=[_contract(volume=100), _contract(volume=50)],
        )
        data = {"options": {"AAPL": chain}}
        insights = model.generate_insights(data, ["AAPL"])
        assert len(insights) == 1
        assert insights[0].direction == InsightDirection.UP

    def test_high_pc_ratio_bearish(self, model):
        """High put/call ratio → bearish (DOWN)."""
        chain = _make_chain(
            calls=[_contract(volume=100)],
            puts=[_contract(volume=200), _contract(volume=100)],
        )
        data = {"options": {"AAPL": chain}}
        insights = model.generate_insights(data, ["AAPL"])
        assert len(insights) == 1
        assert insights[0].direction == InsightDirection.DOWN

    def test_neutral_ratio_no_signal(self, model):
        """Neutral ratio with no blocks or spikes → no signal."""
        chain = _make_chain(
            calls=[_contract(volume=100)],
            puts=[_contract(volume=100)],
        )
        data = {"options": {"AAPL": chain}}
        insights = model.generate_insights(data, ["AAPL"])
        assert len(insights) == 0


class TestVolumeSpikes:
    def test_volume_oi_spike_boosts_confidence(self, model):
        """Volume >> OI increases confidence."""
        chain = _make_chain(
            calls=[_contract(volume=2000, open_interest=100)],  # 20x spike
            puts=[_contract(volume=50, open_interest=500)],
        )
        data = {"options": {"SPY": chain}}
        insights = model.generate_insights(data, ["SPY"])
        assert len(insights) == 1
        assert insights[0].confidence >= 0.7
        assert "vol_oi_spikes" in str(insights[0].metadata.get("signals", []))


class TestBlockTrades:
    def test_call_blocks_bullish(self, model):
        """Large call block trades → bullish."""
        chain = _make_chain(
            calls=[_contract(volume=5000, open_interest=10000)],
            puts=[_contract(volume=50, open_interest=500)],
        )
        data = {"options": {"TSLA": chain}}
        insights = model.generate_insights(data, ["TSLA"])
        assert len(insights) == 1
        assert insights[0].direction == InsightDirection.UP

    def test_put_blocks_bearish(self, model):
        """Large put block trades → bearish."""
        chain = _make_chain(
            calls=[_contract(volume=50, open_interest=500)],
            puts=[_contract(volume=5000, open_interest=10000)],
        )
        data = {"options": {"META": chain}}
        insights = model.generate_insights(data, ["META"])
        assert len(insights) == 1
        assert insights[0].direction == InsightDirection.DOWN


class TestEmptyData:
    def test_missing_options_key(self, model):
        insights = model.generate_insights({}, ["AAPL"])
        assert insights == []

    def test_symbol_not_in_options(self, model):
        data = {"options": {"AAPL": _make_chain([], [])}}
        insights = model.generate_insights(data, ["MSFT"])
        assert insights == []

    def test_empty_chain(self, model):
        data = {"options": {"AAPL": _make_chain([], [])}}
        insights = model.generate_insights(data, ["AAPL"])
        assert insights == []

    def test_below_min_volume(self, model):
        chain = _make_chain(
            calls=[_contract(volume=10)],
            puts=[_contract(volume=5)],
        )
        data = {"options": {"AAPL": chain}}
        insights = model.generate_insights(data, ["AAPL"])
        assert insights == []


class TestMultipleSymbols:
    def test_generates_for_multiple(self, model):
        data = {
            "options": {
                "AAPL": _make_chain(
                    [_contract(volume=500)], [_contract(volume=50)]
                ),
                "TSLA": _make_chain(
                    [_contract(volume=50)], [_contract(volume=500)]
                ),
            }
        }
        insights = model.generate_insights(data, ["AAPL", "TSLA"])
        symbols = {i.symbol for i in insights}
        assert "AAPL" in symbols
        assert "TSLA" in symbols
