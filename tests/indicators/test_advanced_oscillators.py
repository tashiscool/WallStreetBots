"""Tests for Advanced Oscillator Indicators."""
from datetime import datetime, timedelta
import numpy as np
import pytest

from backend.tradingbot.indicators.base import TradeBar
from backend.tradingbot.indicators.advanced_oscillators import (
    ElderRayBull, ElderRayBear, DeMarker, ConnorsRSI,
    FisherTransform, ChandeMomentumOscillator, KlingerVolumeOscillator,
)


def make_bars(n=30, start_price=100.0, volatility=2.0):
    bars = []
    price = start_price
    t = datetime(2024, 1, 1)
    for i in range(n):
        change = np.random.randn() * volatility
        o = price
        c = price + change
        h = max(o, c) + abs(np.random.randn())
        low = min(o, c) - abs(np.random.randn())
        bars.append(TradeBar(time=t + timedelta(days=i), open=o, high=h, low=low, close=c, volume=1000 + i * 10))
        price = c
    return bars


class TestElderRay:
    def test_bull_power(self):
        indicator = ElderRayBull(period=13)
        for bar in make_bars(20):
            indicator.update_bar(bar)
        assert indicator.is_ready
        # Bull power can be positive or negative
        assert isinstance(indicator.value, float)

    def test_bear_power(self):
        indicator = ElderRayBear(period=13)
        for bar in make_bars(20):
            indicator.update_bar(bar)
        assert indicator.is_ready
        # Bear power is typically negative
        assert isinstance(indicator.value, float)


class TestDeMarker:
    def test_range(self):
        indicator = DeMarker(period=14)
        for bar in make_bars(20):
            indicator.update_bar(bar)
        if indicator.is_ready:
            assert 0 <= indicator.value <= 1


class TestConnorsRSI:
    def test_range(self):
        indicator = ConnorsRSI(rsi_period=3, streak_period=2, rank_period=20)
        for bar in make_bars(30):
            indicator.update(bar.time, bar.close)
        if indicator.is_ready:
            assert 0 <= indicator.value <= 100


class TestFisherTransform:
    def test_output(self):
        indicator = FisherTransform(period=10)
        for bar in make_bars(15):
            indicator.update_bar(bar)
        if indicator.is_ready:
            assert isinstance(indicator.value, float)


class TestChandeMomentumOscillator:
    def test_range(self):
        indicator = ChandeMomentumOscillator(period=14)
        for bar in make_bars(20):
            indicator.update(bar.time, bar.close)
        if indicator.is_ready:
            assert -100 <= indicator.value <= 100


class TestKlingerVolumeOscillator:
    def test_output(self):
        indicator = KlingerVolumeOscillator(fast_period=10, slow_period=20)
        for bar in make_bars(25):
            indicator.update_bar(bar)
        if indicator.is_ready:
            assert isinstance(indicator.value, float)
