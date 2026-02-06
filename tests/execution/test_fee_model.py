"""Tests for Maker/Taker Fee Model module."""

import pytest

from backend.tradingbot.execution.fee_model import (
    ALPACA_CRYPTO,
    ALPACA_EQUITY,
    IBKR_FIXED_EQUITY,
    IBKR_OPTIONS,
    IBKR_TIERED_EQUITY,
    KNOWN_SCHEDULES,
    AssetClass,
    FeeEstimate,
    FeeModel,
    FeeOptimizer,
    FeeSchedule,
    FeeType,
    OrderTypeRecommendation,
)


class TestFeeSchedule:
    def test_alpaca_equity_is_zero_commission(self):
        assert ALPACA_EQUITY.maker_fee_type == FeeType.ZERO
        assert ALPACA_EQUITY.taker_fee_type == FeeType.ZERO

    def test_alpaca_crypto_has_maker_taker_spread(self):
        assert ALPACA_CRYPTO.maker_fee < ALPACA_CRYPTO.taker_fee

    def test_ibkr_tiered_has_rebate(self):
        assert IBKR_TIERED_EQUITY.maker_fee < 0  # Rebate

    def test_known_schedules_registry(self):
        assert len(KNOWN_SCHEDULES) >= 5
        assert "alpaca_equity" in KNOWN_SCHEDULES
        assert "ibkr_tiered_equity" in KNOWN_SCHEDULES

    def test_frozen(self):
        with pytest.raises(AttributeError):
            ALPACA_EQUITY.maker_fee = 0.01


class TestFeeModel:
    def test_alpaca_equity_buy_zero_commission(self):
        model = FeeModel(ALPACA_EQUITY)
        fee = model.calculate_fee(quantity=100, price=150.0, side='buy', is_maker=True)
        assert fee.commission == 0.0
        assert fee.regulatory == pytest.approx(0.0119, abs=0.01)  # TAF only (buy)

    def test_alpaca_equity_sell_has_sec_fee(self):
        model = FeeModel(ALPACA_EQUITY)
        fee = model.calculate_fee(quantity=100, price=150.0, side='sell', is_maker=False)
        # Sell has SEC fee + TAF
        assert fee.regulatory > 0
        assert 'sec_fee' in fee.breakdown
        assert 'taf_fee' in fee.breakdown

    def test_alpaca_crypto_maker_cheaper(self):
        model = FeeModel(ALPACA_CRYPTO)
        maker = model.calculate_fee(100, 50000.0, 'buy', is_maker=True)
        taker = model.calculate_fee(100, 50000.0, 'buy', is_maker=False)
        assert maker.total < taker.total
        assert maker.total_bps < taker.total_bps

    def test_ibkr_tiered_maker_rebate(self):
        model = FeeModel(IBKR_TIERED_EQUITY)
        fee = model.calculate_fee(quantity=1000, price=50.0, side='buy', is_maker=True)
        # Maker fee should be negative (rebate) for commission component
        assert fee.breakdown['commission'] < 0

    def test_ibkr_tiered_taker_fee(self):
        model = FeeModel(IBKR_TIERED_EQUITY)
        fee = model.calculate_fee(quantity=1000, price=50.0, side='buy', is_maker=False)
        assert fee.total > 0
        assert fee.total_bps > 0

    def test_ibkr_fixed_minimum(self):
        model = FeeModel(IBKR_FIXED_EQUITY)
        # Small order should hit minimum
        fee = model.calculate_fee(quantity=1, price=10.0, side='buy', is_maker=False)
        assert fee.breakdown['commission'] >= 1.0  # $1 minimum

    def test_compare_maker_taker(self):
        model = FeeModel(IBKR_TIERED_EQUITY)
        comparison = model.compare_maker_taker(500, 100.0, 'buy')
        assert 'maker' in comparison
        assert 'taker' in comparison
        assert comparison['maker'].is_maker is True
        assert comparison['taker'].is_maker is False

    def test_total_bps_calculation(self):
        model = FeeModel(ALPACA_CRYPTO)
        fee = model.calculate_fee(1.0, 50000.0, 'buy', is_maker=False)
        # 25bps taker fee
        assert fee.total_bps == pytest.approx(25.0, rel=0.01)

    def test_zero_quantity(self):
        model = FeeModel(ALPACA_EQUITY)
        fee = model.calculate_fee(0, 150.0, 'buy', is_maker=True)
        assert fee.total_bps == 0.0

    def test_custom_schedule(self):
        custom = FeeSchedule(
            venue="custom",
            maker_fee_type=FeeType.FLAT,
            maker_fee=1.50,
            taker_fee_type=FeeType.FLAT,
            taker_fee=3.00,
        )
        model = FeeModel(custom)
        maker = model.calculate_fee(100, 100.0, 'buy', is_maker=True)
        taker = model.calculate_fee(100, 100.0, 'buy', is_maker=False)
        assert maker.commission == 1.50
        assert taker.commission == 3.00


class TestFeeOptimizer:
    def test_high_urgency_recommends_market(self):
        optimizer = FeeOptimizer(fee_model=FeeModel(IBKR_TIERED_EQUITY))
        rec = optimizer.recommend(
            quantity=500, price=100.0, side='buy', urgency=0.95
        )
        assert rec.recommended_type == 'market'
        assert 'urgency' in rec.reasoning.lower()

    def test_low_urgency_with_fee_savings_recommends_limit(self):
        optimizer = FeeOptimizer(fee_model=FeeModel(IBKR_TIERED_EQUITY))
        rec = optimizer.recommend(
            quantity=500, price=100.0, side='buy', urgency=0.05,
            spread_bps=2.0,
        )
        assert rec.recommended_type == 'limit'

    def test_fee_savings_calculated(self):
        optimizer = FeeOptimizer(fee_model=FeeModel(IBKR_TIERED_EQUITY))
        rec = optimizer.recommend(quantity=1000, price=50.0, side='buy', urgency=0.5)
        # IBKR tiered has meaningful maker/taker spread
        assert rec.fee_savings_bps != 0

    def test_fill_probability_range(self):
        optimizer = FeeOptimizer()
        rec = optimizer.recommend(quantity=100, price=100.0, side='buy', urgency=0.5)
        assert 0.05 <= rec.fill_probability <= 0.95

    def test_recommendation_fields(self):
        optimizer = FeeOptimizer()
        rec = optimizer.recommend(quantity=100, price=100.0, side='buy')
        assert isinstance(rec.maker_fee_estimate, FeeEstimate)
        assert isinstance(rec.taker_fee_estimate, FeeEstimate)
        assert rec.recommended_type in ('limit', 'market')
        assert len(rec.reasoning) > 0

    def test_batch_recommend(self):
        optimizer = FeeOptimizer()
        orders = [
            {'quantity': 100, 'price': 150.0, 'side': 'buy', 'urgency': 0.9},
            {'quantity': 200, 'price': 50.0, 'side': 'sell', 'urgency': 0.1},
            {'quantity': 500, 'price': 100.0, 'side': 'buy', 'urgency': 0.5},
        ]
        recs = optimizer.batch_recommend(orders)
        assert len(recs) == 3
        # High urgency should recommend market
        assert recs[0].recommended_type == 'market'

    def test_wide_spread_reduces_fill_probability(self):
        optimizer = FeeOptimizer()
        tight = optimizer.recommend(
            quantity=100, price=100.0, side='buy', urgency=0.5, spread_bps=1.0
        )
        wide = optimizer.recommend(
            quantity=100, price=100.0, side='buy', urgency=0.5, spread_bps=20.0
        )
        assert tight.fill_probability > wide.fill_probability

    def test_high_volatility_increases_fill_probability(self):
        optimizer = FeeOptimizer()
        low_vol = optimizer.recommend(
            quantity=100, price=100.0, side='buy', urgency=0.5, volatility=0.005
        )
        high_vol = optimizer.recommend(
            quantity=100, price=100.0, side='buy', urgency=0.5, volatility=0.05
        )
        assert high_vol.fill_probability > low_vol.fill_probability

    def test_alpaca_zero_fee_still_considers_slippage(self):
        """Even with zero commissions, optimizer should still have an opinion."""
        optimizer = FeeOptimizer(fee_model=FeeModel(ALPACA_EQUITY))
        rec = optimizer.recommend(
            quantity=100, price=150.0, side='buy', urgency=0.5,
            expected_slippage_bps=5.0,
        )
        assert rec.recommended_type in ('limit', 'market')
        assert rec.expected_cost_market > 0  # Slippage included
