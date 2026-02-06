"""Tests for Adverse Selection & Toxic Flow Detection module."""

import pytest
from datetime import datetime, timedelta

from backend.tradingbot.execution.adverse_selection import (
    AdverseSelectionMetrics,
    ToxicFlowDetector,
    TradeRecord,
    VPINCalculator,
    VPINResult,
)


class TestVPINCalculator:
    def test_initial_state(self):
        vpin = VPINCalculator(bucket_volume=1000, num_buckets=10)
        result = vpin.calculate()
        assert result.vpin == 0.0
        assert result.num_buckets == 0

    def test_balanced_flow(self):
        """Equal buy/sell volume should produce low VPIN."""
        vpin = VPINCalculator(bucket_volume=100, num_buckets=10)

        # Alternate buy and sell trades of equal size
        for i in range(100):
            side = 'buy' if i % 2 == 0 else 'sell'
            vpin.add_trade(price=100.0 + i * 0.01, volume=50, side=side)

        result = vpin.calculate()
        assert result.vpin < 0.3  # Low toxicity

    def test_one_sided_flow(self):
        """All-buy flow should produce high VPIN."""
        vpin = VPINCalculator(bucket_volume=100, num_buckets=10)

        # All buys
        for i in range(50):
            vpin.add_trade(price=100.0 + i * 0.01, volume=50, side='buy')

        result = vpin.calculate()
        assert result.vpin > 0.8  # High toxicity
        assert result.is_toxic

    def test_bucket_completion_returns_result(self):
        vpin = VPINCalculator(bucket_volume=100, num_buckets=5)

        # First trade doesn't complete a bucket
        result = vpin.add_trade(price=100.0, volume=50, side='buy')
        assert result is None

        # Second trade completes the bucket
        result = vpin.add_trade(price=100.0, volume=50, side='buy')
        assert result is not None
        assert result.num_buckets == 1

    def test_tick_classification(self):
        """Without explicit side, uses tick rule."""
        vpin = VPINCalculator(bucket_volume=100, num_buckets=5)

        # Uptick = buy
        vpin.add_trade(price=100.0, volume=50)
        vpin.add_trade(price=100.01, volume=50)  # Uptick -> buy

        result = vpin.calculate()
        assert result.buy_volume_imbalance > 0  # Skewed toward buys

    def test_toxicity_levels(self):
        result_low = VPINResult(vpin=0.2, num_buckets=10, bucket_volume=100, buy_volume_imbalance=0.0)
        assert result_low.toxicity_level == 'low'
        assert not result_low.is_toxic

        result_moderate = VPINResult(vpin=0.4, num_buckets=10, bucket_volume=100, buy_volume_imbalance=0.0)
        assert result_moderate.toxicity_level == 'moderate'

        result_elevated = VPINResult(vpin=0.6, num_buckets=10, bucket_volume=100, buy_volume_imbalance=0.0)
        assert result_elevated.toxicity_level == 'elevated'

        result_toxic = VPINResult(vpin=0.8, num_buckets=10, bucket_volume=100, buy_volume_imbalance=0.0)
        assert result_toxic.toxicity_level == 'toxic'
        assert result_toxic.is_toxic

    def test_reset(self):
        vpin = VPINCalculator(bucket_volume=100, num_buckets=5)
        vpin.add_trade(price=100.0, volume=200, side='buy')
        vpin.reset()
        result = vpin.calculate()
        assert result.num_buckets == 0


class TestToxicFlowDetector:
    def _make_detector(self):
        return ToxicFlowDetector(lookback_trades=100, toxic_threshold_bps=5.0)

    def test_record_fill(self):
        det = self._make_detector()
        det.record_fill("AAPL", "buy", 100, 150.05, mid_price=150.00)
        # Should have pending updates
        assert len(det._pending_updates.get("AAPL", [])) == 1

    def test_get_metrics_insufficient_data(self):
        det = self._make_detector()
        assert det.get_metrics("AAPL") is None

    def test_effective_spread_calculation(self):
        det = self._make_detector()
        # Record 10 buy fills all at 1bps above mid
        for i in range(10):
            det.record_fill(
                "AAPL", "buy", 100,
                fill_price=150.015,  # 1bps above mid
                mid_price=150.00,
                timestamp=datetime.now() - timedelta(seconds=i),
            )

        metrics = det.get_metrics("AAPL")
        assert metrics is not None
        assert metrics.num_trades == 10
        assert metrics.avg_effective_spread_bps == pytest.approx(1.0, rel=0.1)

    def test_adverse_selection_detected(self):
        """Price moving against us after fill = adverse selection."""
        det = self._make_detector()
        now = datetime.now()

        # Record fills where price drops after buying
        # effective_spread = (fill - mid)/mid = positive (paid above mid)
        # realized_spread = (fill - mid_30s)/mid = positive & large (price fell, so fill >> mid_30s)
        # adverse_selection_cost = effective - realized = negative when realized > effective
        # This means the price moved AWAY from our fill (adverse), captured in impact metrics
        for i in range(10):
            ts = now - timedelta(seconds=600 - i * 10)
            trade = TradeRecord(
                timestamp=ts,
                symbol="AAPL",
                side="buy",
                quantity=100,
                fill_price=150.05,
                mid_price_at_fill=150.00,
                mid_price_30s=149.90,  # Price dropped 10bps after fill
                mid_price_60s=149.85,
                mid_price_300s=149.80,
            )
            if "AAPL" not in det._trades:
                from collections import deque
                det._trades["AAPL"] = deque(maxlen=100)
            det._trades["AAPL"].append(trade)

        metrics = det.get_metrics("AAPL")
        assert metrics is not None
        # Price dropped after buy: post-fill impacts are negative
        assert metrics.impact_30s_bps < 0  # Price moved down after buy
        assert metrics.impact_60s_bps < 0
        assert metrics.impact_300s_bps < 0
        # Buy adverse should be significant
        assert abs(metrics.buy_adverse_bps) > 5.0

    def test_should_reduce_size(self):
        det = self._make_detector()
        # Not enough data
        should_reduce, reason = det.should_reduce_size("AAPL")
        assert not should_reduce
        assert reason == "insufficient_data"

    def test_update_post_fill_prices(self):
        det = self._make_detector()
        ts = datetime.now() - timedelta(seconds=120)
        det.record_fill("AAPL", "buy", 100, 150.05, mid_price=150.00, timestamp=ts)

        # Update with current price (120s later)
        det.update_post_fill_prices("AAPL", current_mid=149.95)

        # Check that post-fill prices were populated
        trade = list(det._trades["AAPL"])[0]
        assert trade.mid_price_1s is not None
        assert trade.mid_price_5s is not None
        assert trade.mid_price_30s is not None
        assert trade.mid_price_60s is not None

    def test_get_all_metrics(self):
        det = self._make_detector()
        now = datetime.now()

        for symbol in ["AAPL", "GOOG"]:
            for i in range(10):
                det.record_fill(
                    symbol, "buy", 100, 150.05, mid_price=150.00,
                    timestamp=now - timedelta(seconds=i),
                )

        all_metrics = det.get_all_metrics()
        assert "AAPL" in all_metrics
        assert "GOOG" in all_metrics

    def test_reset_symbol(self):
        det = self._make_detector()
        det.record_fill("AAPL", "buy", 100, 150.05, mid_price=150.00)
        det.record_fill("GOOG", "buy", 50, 140.00, mid_price=139.95)
        det.reset("AAPL")
        assert "AAPL" not in det._trades
        assert "GOOG" in det._trades

    def test_reset_all(self):
        det = self._make_detector()
        det.record_fill("AAPL", "buy", 100, 150.05, mid_price=150.00)
        det.reset()
        assert len(det._trades) == 0


class TestAdverseSelectionMetrics:
    def test_dataclass_fields(self):
        metrics = AdverseSelectionMetrics(
            symbol="AAPL",
            period_start=datetime.now(),
            period_end=datetime.now(),
            num_trades=100,
            avg_effective_spread_bps=2.0,
            avg_realized_spread_bps=1.0,
            impact_1s_bps=-0.5,
            impact_5s_bps=-1.0,
            impact_30s_bps=-2.0,
            impact_60s_bps=-3.0,
            impact_300s_bps=-4.0,
            adverse_selection_cost_bps=1.0,
            toxicity_score=0.1,
            buy_adverse_bps=1.5,
            sell_adverse_bps=0.5,
        )
        assert metrics.symbol == "AAPL"
        assert metrics.num_trades == 100
        assert metrics.toxicity_score == 0.1
