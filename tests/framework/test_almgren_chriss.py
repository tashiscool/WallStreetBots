"""Tests for Almgren-Chriss optimal execution model."""

from decimal import Decimal
import math
import pytest

from backend.tradingbot.framework.execution_models.almgren_chriss import (
    AlmgrenChrissConfig,
    AlmgrenChrissModel,
    AlmgrenChrissExecutionModel,
)
from backend.tradingbot.framework.portfolio_target import PortfolioTarget


@pytest.fixture
def model():
    return AlmgrenChrissModel()


@pytest.fixture
def default_config():
    return AlmgrenChrissConfig(
        total_shares=Decimal("10000"),
        total_time=60.0,
        num_slices=10,
        volatility=0.02,
        daily_volume=1e6,
        permanent_impact=0.1,
        temporary_impact=0.01,
        risk_aversion=1e-6,
    )


class TestOptimalTrajectory:
    def test_trajectory_sums_to_total(self, model, default_config):
        trajectory = model.compute_optimal_trajectory(default_config)
        assert len(trajectory) == default_config.num_slices
        total = sum(trajectory)
        assert abs(total - default_config.total_shares) < Decimal("0.01")

    def test_single_slice(self, model):
        config = AlmgrenChrissConfig(
            total_shares=Decimal("500"),
            total_time=10.0,
            num_slices=1,
        )
        trajectory = model.compute_optimal_trajectory(config)
        assert len(trajectory) == 1
        assert trajectory[0] == Decimal("500")

    def test_zero_slices(self, model):
        config = AlmgrenChrissConfig(
            total_shares=Decimal("100"),
            total_time=10.0,
            num_slices=0,
        )
        assert model.compute_optimal_trajectory(config) == []

    def test_urgency_front_loads(self, model):
        """High risk aversion → front-loaded (first slice > last slice)."""
        config = AlmgrenChrissConfig(
            total_shares=Decimal("10000"),
            total_time=60.0,
            num_slices=5,
            volatility=0.02,
            temporary_impact=0.01,
            risk_aversion=1.0,  # Very high urgency
        )
        trajectory = model.compute_optimal_trajectory(config)
        assert trajectory[0] > trajectory[-1]

    def test_zero_risk_aversion_twap(self, model):
        """Zero risk aversion → uniform (TWAP-like) execution."""
        config = AlmgrenChrissConfig(
            total_shares=Decimal("1000"),
            total_time=60.0,
            num_slices=5,
            volatility=0.02,
            temporary_impact=0.01,
            risk_aversion=0.0,
        )
        trajectory = model.compute_optimal_trajectory(config)
        expected = Decimal("1000") / 5
        for t in trajectory:
            assert abs(t - expected) < Decimal("1")

    def test_all_slices_positive(self, model, default_config):
        trajectory = model.compute_optimal_trajectory(default_config)
        assert all(t >= 0 for t in trajectory)


class TestCostEstimation:
    def test_cost_keys(self, model, default_config):
        costs = model.estimate_execution_cost(default_config)
        assert 'expected_cost' in costs
        assert 'variance' in costs
        assert 'is_cost' in costs
        assert 'timing_risk' in costs

    def test_cost_increases_with_size(self, model):
        small = AlmgrenChrissConfig(total_shares=Decimal("100"), total_time=60.0)
        large = AlmgrenChrissConfig(total_shares=Decimal("100000"), total_time=60.0)
        c_small = model.estimate_execution_cost(small)
        c_large = model.estimate_execution_cost(large)
        assert c_large['expected_cost'] > c_small['expected_cost']

    def test_timing_risk_non_negative(self, model, default_config):
        costs = model.estimate_execution_cost(default_config)
        assert costs['timing_risk'] >= 0


class TestKappa:
    def test_kappa_zero_eta(self):
        config = AlmgrenChrissConfig(
            total_shares=Decimal("100"),
            total_time=10.0,
            temporary_impact=0.0,
        )
        assert AlmgrenChrissModel._kappa(config) == 0.0

    def test_kappa_positive(self, default_config):
        k = AlmgrenChrissModel._kappa(default_config)
        assert k > 0


class TestExecutionModel:
    def test_execute_returns_orders(self):
        exec_model = AlmgrenChrissExecutionModel(
            duration_minutes=30,
            num_slices=5,
        )
        exec_model.set_current_positions({})

        target = PortfolioTarget(symbol='AAPL', quantity=Decimal("100"))
        orders = exec_model.execute([target])

        assert len(orders) > 0
        total_qty = sum(o.quantity for o in orders)
        assert total_qty == Decimal("100")

    def test_execute_metadata(self):
        exec_model = AlmgrenChrissExecutionModel(num_slices=3)
        exec_model.set_current_positions({})

        target = PortfolioTarget(symbol='MSFT', quantity=Decimal("300"))
        orders = exec_model.execute([target])

        for o in orders:
            assert o.metadata['algorithm'] == 'AlmgrenChriss'

    def test_no_order_when_at_target(self):
        exec_model = AlmgrenChrissExecutionModel()
        exec_model.set_current_positions({'AAPL': Decimal("100")})

        target = PortfolioTarget(symbol='AAPL', quantity=Decimal("100"))
        orders = exec_model.execute([target])
        assert len(orders) == 0
