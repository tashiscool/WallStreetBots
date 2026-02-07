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


class TestBufferedMidPriceResolution:
    """Tests for the buffered interpolation approach to post-fill price measurement."""

    def _make_detector(self):
        return ToxicFlowDetector(lookback_trades=100, toxic_threshold_bps=5.0)

    def test_linear_interpolation_between_samples(self):
        """Mid price at exact horizon should be linearly interpolated."""
        det = self._make_detector()
        fill_ts = datetime(2025, 1, 1, 12, 0, 0)
        det.record_fill("AAPL", "buy", 100, 150.05, mid_price=150.00, timestamp=fill_ts)

        # Call at t+3s with mid=150.30
        det.update_post_fill_prices(
            "AAPL", current_mid=150.30,
            current_time=fill_ts + timedelta(seconds=3),
        )
        # 1s horizon: target = fill_ts + 1s
        # Buffer: [(fill_ts, 150.00), (fill_ts+3s, 150.30)]
        # Interpolation: 150.00 + (1/3)*(150.30 - 150.00) = 150.10
        trade = list(det._trades["AAPL"])[0]
        assert trade.mid_price_1s == pytest.approx(150.10, abs=0.001)
        assert trade.mid_price_1s_ts == fill_ts + timedelta(seconds=1)

    def test_interpolation_uses_bracketing_samples(self):
        """With many samples, interpolation should use the tightest bracket."""
        det = self._make_detector()
        fill_ts = datetime(2025, 1, 1, 12, 0, 0)
        det.record_fill("AAPL", "buy", 100, 150.05, mid_price=100.00, timestamp=fill_ts)

        # Build up a buffer with calls at 2s, 4s, 6s, 8s, 10s
        for sec in [2, 4, 6, 8, 10]:
            det.update_post_fill_prices(
                "AAPL", current_mid=100.0 + sec,
                current_time=fill_ts + timedelta(seconds=sec),
            )

        trade = list(det._trades["AAPL"])[0]
        # 5s horizon: target = fill_ts + 5s
        # Bracketing samples: (fill_ts+4s, 104.0) and (fill_ts+6s, 106.0)
        # Interpolation: 104.0 + (1/2)*(106.0 - 104.0) = 105.0
        assert trade.mid_price_5s == pytest.approx(105.0, abs=0.01)
        assert trade.mid_price_5s_ts == fill_ts + timedelta(seconds=5)

    def test_nearest_sample_fallback_no_before(self):
        """When no sample exists before target, use the earliest after."""
        det = self._make_detector()
        fill_ts = datetime(2025, 1, 1, 12, 0, 0)
        # Construct a trade with no fill-time mid (simulate edge case)
        # Actually, the seed always includes fill-time, so this would require
        # the target to be BEFORE the fill.  Instead, test by calling
        # _resolve_mid_price directly with a buffer missing the early sample.
        buffer = [
            (fill_ts + timedelta(seconds=3), 150.30),
            (fill_ts + timedelta(seconds=5), 150.50),
        ]
        target = fill_ts + timedelta(seconds=1)
        price, ts = ToxicFlowDetector._resolve_mid_price(buffer, target)
        # Should use nearest available (the first sample at 3s)
        assert price == 150.30
        assert ts == fill_ts + timedelta(seconds=3)

    def test_nearest_sample_fallback_no_after(self):
        """When target is beyond all samples, use the latest before."""
        buffer = [
            (datetime(2025, 1, 1, 12, 0, 0), 100.0),
            (datetime(2025, 1, 1, 12, 0, 4), 104.0),
        ]
        target = datetime(2025, 1, 1, 12, 0, 10)
        price, ts = ToxicFlowDetector._resolve_mid_price(buffer, target)
        assert price == 104.0
        assert ts == datetime(2025, 1, 1, 12, 0, 4)

    def test_observation_timestamps_recorded(self):
        """All horizon _ts fields should be populated after update."""
        det = self._make_detector()
        fill_ts = datetime(2025, 1, 1, 12, 0, 0)
        det.record_fill("AAPL", "buy", 100, 150.05, mid_price=150.00, timestamp=fill_ts)

        # Single call at 400s to trigger all horizons
        det.update_post_fill_prices(
            "AAPL", current_mid=149.50,
            current_time=fill_ts + timedelta(seconds=400),
        )

        trade = list(det._trades["AAPL"])[0]
        assert trade.mid_price_1s_ts is not None
        assert trade.mid_price_5s_ts is not None
        assert trade.mid_price_30s_ts is not None
        assert trade.mid_price_60s_ts is not None
        assert trade.mid_price_300s_ts is not None

    def test_progressive_updates_resolve_horizons_incrementally(self):
        """Horizons should resolve as elapsed time crosses each threshold."""
        det = self._make_detector()
        fill_ts = datetime(2025, 1, 1, 12, 0, 0)
        det.record_fill("AAPL", "buy", 100, 150.05, mid_price=150.00, timestamp=fill_ts)

        trade = list(det._trades["AAPL"])[0]

        # At t+2s: only 1s horizon should resolve
        det.update_post_fill_prices("AAPL", 150.10, fill_ts + timedelta(seconds=2))
        assert trade.mid_price_1s is not None
        assert trade.mid_price_5s is None

        # At t+10s: 5s should also resolve
        det.update_post_fill_prices("AAPL", 150.20, fill_ts + timedelta(seconds=10))
        assert trade.mid_price_5s is not None
        assert trade.mid_price_30s is None

        # At t+35s: 30s resolves
        det.update_post_fill_prices("AAPL", 150.15, fill_ts + timedelta(seconds=35))
        assert trade.mid_price_30s is not None
        assert trade.mid_price_60s is None

        # At t+65s: 60s resolves
        det.update_post_fill_prices("AAPL", 150.05, fill_ts + timedelta(seconds=65))
        assert trade.mid_price_60s is not None
        assert trade.mid_price_300s is None

        # At t+310s: 300s resolves, trade removed from pending
        det.update_post_fill_prices("AAPL", 149.90, fill_ts + timedelta(seconds=310))
        assert trade.mid_price_300s is not None
        assert len(det._pending_updates.get("AAPL", [])) == 0

    def test_buffer_cleaned_after_all_horizons_filled(self):
        """Buffer should be removed once the trade is fully resolved."""
        det = self._make_detector()
        fill_ts = datetime(2025, 1, 1, 12, 0, 0)
        det.record_fill("AAPL", "buy", 100, 150.05, mid_price=150.00, timestamp=fill_ts)

        trade = list(det._trades["AAPL"])[0]
        trade_id = id(trade)

        det.update_post_fill_prices("AAPL", 149.50, fill_ts + timedelta(seconds=400))
        # Buffer should be cleaned up
        assert trade_id not in det._mid_price_buffers

    def test_interpolation_accuracy_with_dense_samples(self):
        """Dense sampling should produce values close to the true price path."""
        det = self._make_detector()
        fill_ts = datetime(2025, 1, 1, 12, 0, 0)
        det.record_fill("AAPL", "buy", 100, 150.05, mid_price=100.00, timestamp=fill_ts)

        # Simulate a linear price path: mid = 100 + 0.1*t
        for sec in range(1, 62):
            det.update_post_fill_prices(
                "AAPL", current_mid=100.0 + 0.1 * sec,
                current_time=fill_ts + timedelta(seconds=sec),
            )

        trade = list(det._trades["AAPL"])[0]
        # 1s: expect 100.1 (exact sample at t=1)
        assert trade.mid_price_1s == pytest.approx(100.1, abs=0.01)
        # 5s: expect 100.5
        assert trade.mid_price_5s == pytest.approx(100.5, abs=0.01)
        # 30s: expect 103.0
        assert trade.mid_price_30s == pytest.approx(103.0, abs=0.01)
        # 60s: expect 106.0
        assert trade.mid_price_60s == pytest.approx(106.0, abs=0.01)

    def test_multiple_trades_independent_buffers(self):
        """Each trade should have its own independent buffer."""
        det = self._make_detector()
        fill_ts = datetime(2025, 1, 1, 12, 0, 0)

        det.record_fill("AAPL", "buy", 100, 150.05, mid_price=100.00,
                         timestamp=fill_ts)
        det.record_fill("AAPL", "sell", 50, 149.95, mid_price=100.00,
                         timestamp=fill_ts + timedelta(seconds=1))

        # Update at t+3s
        det.update_post_fill_prices("AAPL", 103.00, fill_ts + timedelta(seconds=3))

        trades = list(det._trades["AAPL"])
        # Trade 0 (fill at t=0): 1s horizon target is t+1s
        # Buffer: [(t+0, 100), (t+3, 103)] → interp at t+1: 100 + (1/3)*3 = 101
        assert trades[0].mid_price_1s == pytest.approx(101.0, abs=0.01)

        # Trade 1 (fill at t=1s): 1s horizon target is t+2s
        # Buffer: [(t+1, 100), (t+3, 103)] → interp at t+2: 100 + (1/2)*3 = 101.5
        assert trades[1].mid_price_1s == pytest.approx(101.5, abs=0.01)

    def test_reset_cleans_up_buffers(self):
        """reset() should also clear mid-price buffers."""
        det = self._make_detector()
        fill_ts = datetime(2025, 1, 1, 12, 0, 0)
        det.record_fill("AAPL", "buy", 100, 150.05, mid_price=150.00, timestamp=fill_ts)
        # Trigger buffer creation
        det.update_post_fill_prices("AAPL", 150.10, fill_ts + timedelta(seconds=2))
        assert len(det._mid_price_buffers) > 0

        det.reset("AAPL")
        assert len(det._mid_price_buffers) == 0

    def test_resolve_mid_price_empty_buffer_raises(self):
        """_resolve_mid_price should raise on empty buffer."""
        with pytest.raises(ValueError, match="Empty mid-price buffer"):
            ToxicFlowDetector._resolve_mid_price([], datetime.now())

    def test_resolve_mid_price_exact_match(self):
        """When a sample lands exactly on target, no interpolation needed."""
        ts = datetime(2025, 1, 1, 12, 0, 5)
        buffer = [
            (datetime(2025, 1, 1, 12, 0, 0), 100.0),
            (datetime(2025, 1, 1, 12, 0, 5), 105.0),
            (datetime(2025, 1, 1, 12, 0, 10), 110.0),
        ]
        price, obs_ts = ToxicFlowDetector._resolve_mid_price(buffer, ts)
        # Exact match at 5s: before=(5s, 105), after=(10s, 110)
        # Interpolation: alpha=0/5=0, so price=105.0
        assert price == pytest.approx(105.0)
        assert obs_ts == ts


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
