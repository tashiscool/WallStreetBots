"""Tests for Tail Hedge Manager."""

import pytest

from backend.tradingbot.risk.tail_hedge import TailHedgeConfig, TailHedgeManager


@pytest.fixture
def manager():
    return TailHedgeManager()


@pytest.fixture
def high_vix_manager():
    return TailHedgeManager(TailHedgeConfig(vix_threshold=20.0))


class TestShouldHedge:
    def test_above_threshold(self, manager):
        assert manager.should_hedge(30.0, 1_000_000) is True

    def test_below_threshold(self, manager):
        assert manager.should_hedge(20.0, 1_000_000) is False

    def test_at_threshold(self, manager):
        assert manager.should_hedge(25.0, 1_000_000) is True

    def test_zero_portfolio(self, manager):
        assert manager.should_hedge(30.0, 0) is False


class TestHedgeSizing:
    def test_basic_sizing(self, manager):
        size = manager.calculate_hedge_size(1_000_000, 30.0)
        assert size['notional'] > 0
        assert size['num_contracts'] >= 1
        assert size['estimated_cost'] > 0
        assert 0 < size['cost_pct'] < 1

    def test_cost_cap_respected(self):
        config = TailHedgeConfig(max_hedge_cost_pct=0.001)  # very low cap
        mgr = TailHedgeManager(config)
        size = mgr.calculate_hedge_size(1_000_000, 50.0)
        # Cost should not exceed cap (annualized, prorated)
        max_allowed = 1_000_000 * 0.001 * (30 / 365)
        assert size['estimated_cost'] <= max_allowed + 1  # small tolerance


class TestHedgeEffectiveness:
    def test_hedge_paid_off(self, manager):
        result = manager.evaluate_hedge_effectiveness(
            hedge_pnl=50_000,
            portfolio_pnl=-100_000,
        )
        assert result['protection_ratio'] == pytest.approx(0.5)
        assert result['net_benefit'] == -50_000
        assert result['cost_of_carry'] == 0.0

    def test_hedge_cost_no_crisis(self, manager):
        result = manager.evaluate_hedge_effectiveness(
            hedge_pnl=-5_000,
            portfolio_pnl=20_000,
        )
        assert result['protection_ratio'] == 0.0
        assert result['cost_of_carry'] == 5_000

    def test_zero_portfolio_pnl(self, manager):
        result = manager.evaluate_hedge_effectiveness(0, 0)
        assert result['protection_ratio'] == 0.0


class TestRecommendations:
    def test_add_when_vix_high_no_hedges(self, manager):
        rec = manager.get_hedge_recommendation(30.0, 1_000_000, current_hedges=[])
        assert rec['action'] == 'ADD'
        assert 'hedge_size' in rec

    def test_hold_when_vix_low(self, manager):
        rec = manager.get_hedge_recommendation(15.0, 1_000_000)
        assert rec['action'] == 'HOLD'

    def test_remove_existing_when_vix_drops(self, manager):
        rec = manager.get_hedge_recommendation(
            15.0, 1_000_000,
            current_hedges=[{'notional': 50_000, 'dte_remaining': 20}],
        )
        assert rec['action'] == 'REMOVE'

    def test_roll_near_expiry(self, manager):
        rec = manager.get_hedge_recommendation(
            30.0, 1_000_000,
            current_hedges=[{'notional': 50_000, 'dte_remaining': 5}],
        )
        assert rec['action'] == 'ROLL'

    def test_hold_adequate_hedges(self, manager):
        rec = manager.get_hedge_recommendation(
            30.0, 1_000_000,
            current_hedges=[{'notional': 50_000, 'dte_remaining': 25}],
        )
        assert rec['action'] == 'HOLD'
