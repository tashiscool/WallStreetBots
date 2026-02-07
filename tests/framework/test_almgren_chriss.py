"""Tests for Almgren-Chriss optimal execution model."""

from decimal import Decimal
import math
import pytest

from backend.tradingbot.framework.execution_models.almgren_chriss import (
    AlmgrenChrissConfig,
    AlmgrenChrissModel,
    AlmgrenChrissExecutionModel,
    CalibrationRecord,
    ImpactCalibrator,
    IMPACT_PARAMS,
    LiquidityBucket,
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
    def test_execute_returns_first_slice_only(self):
        """execute() should return only the first slice (slice_index=0)."""
        exec_model = AlmgrenChrissExecutionModel(
            duration_minutes=30,
            num_slices=5,
        )
        exec_model.set_current_positions({})

        target = PortfolioTarget(symbol='AAPL', quantity=Decimal("100"))
        orders = exec_model.execute([target])

        # Should return exactly 1 order (the first slice)
        assert len(orders) == 1
        assert orders[0].metadata['slice_index'] == 0
        assert orders[0].metadata['algorithm'] == 'AlmgrenChriss'

    def test_execute_schedules_remaining_slices(self):
        """Remaining slices (index > 0) should be stored in _scheduled_slices."""
        exec_model = AlmgrenChrissExecutionModel(
            duration_minutes=30,
            num_slices=5,
        )
        exec_model.set_current_positions({})

        target = PortfolioTarget(symbol='AAPL', quantity=Decimal("1000"))
        orders = exec_model.execute([target])

        # First slice returned immediately
        assert len(orders) == 1

        # Remaining slices scheduled
        scheduled = exec_model.get_scheduled_slices()
        assert len(scheduled) >= 1  # At least some future slices

        # All scheduled slices have index > 0
        for s in scheduled:
            assert s['order'].metadata['slice_index'] > 0
            assert 'scheduled_time' in s

        # Total quantity across immediate + scheduled equals target
        total_qty = orders[0].quantity + sum(
            s['order'].quantity for s in scheduled
        )
        assert total_qty == Decimal("1000")

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


# ---------------------------------------------------------------------------
# ImpactCalibrator tests
# ---------------------------------------------------------------------------

import random
import numpy as np


class TestClassifyBucket:
    def test_mega_cap(self):
        assert ImpactCalibrator.classify_bucket(60_000_000) == LiquidityBucket.MEGA_CAP

    def test_large_cap(self):
        assert ImpactCalibrator.classify_bucket(15_000_000) == LiquidityBucket.LARGE_CAP

    def test_mid_cap(self):
        assert ImpactCalibrator.classify_bucket(5_000_000) == LiquidityBucket.MID_CAP

    def test_small_cap(self):
        assert ImpactCalibrator.classify_bucket(800_000) == LiquidityBucket.SMALL_CAP

    def test_micro_cap(self):
        assert ImpactCalibrator.classify_bucket(100_000) == LiquidityBucket.MICRO_CAP

    def test_crypto_major_btc(self):
        assert ImpactCalibrator.classify_bucket(1e9, 'BTC-USD') == LiquidityBucket.CRYPTO_MAJOR

    def test_crypto_major_eth(self):
        assert ImpactCalibrator.classify_bucket(5e8, 'ETHUSD') == LiquidityBucket.CRYPTO_MAJOR

    def test_crypto_alt(self):
        assert ImpactCalibrator.classify_bucket(1e7, 'SOL-USD') == LiquidityBucket.CRYPTO_ALT

    def test_boundary_mega(self):
        """Exactly at 50M threshold → MEGA_CAP."""
        assert ImpactCalibrator.classify_bucket(50_000_000) == LiquidityBucket.MEGA_CAP

    def test_below_boundary_mega(self):
        """Just below 50M → LARGE_CAP."""
        assert ImpactCalibrator.classify_bucket(49_999_999) == LiquidityBucket.LARGE_CAP


def _make_synthetic_records(
    n: int,
    bucket: LiquidityBucket,
    daily_volume: float = 20_000_000,
    seed: int = 42,
) -> list:
    """Generate synthetic CalibrationRecords with realistic linear relationship."""
    rng = np.random.RandomState(seed)
    records = []
    for _ in range(n):
        qty = rng.uniform(1000, 50000)
        participation = qty / daily_volume
        # slippage ~ 50 * participation + noise  (temporary impact ≈ 50)
        slippage = 50.0 * participation + rng.normal(0, 0.5)
        # 5min impact ~ 20 * participation + noise  (permanent impact ≈ 20)
        impact_5m = 20.0 * participation + rng.normal(0, 0.3)
        records.append(CalibrationRecord(
            symbol='AAPL',
            side='buy',
            filled_quantity=qty,
            daily_volume=daily_volume,
            slippage_bps=slippage,
            impact_5min_bps=impact_5m,
            market_cap_bucket=bucket,
        ))
    return records


class TestCalibrateWithSufficientData:
    def test_calibrated_params_differ_from_defaults(self):
        """With 50 synthetic records, calibrated params should differ from defaults."""
        calibrator = ImpactCalibrator()
        records = _make_synthetic_records(50, LiquidityBucket.LARGE_CAP)
        for r in records:
            calibrator.record(r)

        result = calibrator.calibrate()

        # Calibrated values should exist for LARGE_CAP
        assert LiquidityBucket.LARGE_CAP in result
        calibrated = result[LiquidityBucket.LARGE_CAP]
        defaults = IMPACT_PARAMS[LiquidityBucket.LARGE_CAP]

        # At least one param should differ from defaults
        assert (
            calibrated['permanent_impact'] != defaults['permanent_impact']
            or calibrated['temporary_impact'] != defaults['temporary_impact']
        )

    def test_temporary_impact_reasonable(self):
        """Calibrated temporary_impact should be in the right ballpark (~50 bps/participation)."""
        calibrator = ImpactCalibrator()
        records = _make_synthetic_records(100, LiquidityBucket.LARGE_CAP)
        for r in records:
            calibrator.record(r)

        result = calibrator.calibrate()
        temp = result[LiquidityBucket.LARGE_CAP]['temporary_impact']
        # Should be roughly 50 (our synthetic slope), allow wide tolerance
        assert 10 < temp < 200

    def test_permanent_impact_positive(self):
        """Calibrated permanent_impact should be positive and finite."""
        calibrator = ImpactCalibrator()
        records = _make_synthetic_records(100, LiquidityBucket.LARGE_CAP)
        for r in records:
            calibrator.record(r)

        result = calibrator.calibrate()
        perm = result[LiquidityBucket.LARGE_CAP]['permanent_impact']
        assert perm > 0
        assert np.isfinite(perm)


class TestCalibrateInsufficientData:
    def test_under_min_samples_keeps_defaults(self):
        """<30 records → returns IMPACT_PARAMS defaults."""
        calibrator = ImpactCalibrator()
        records = _make_synthetic_records(10, LiquidityBucket.MID_CAP)
        for r in records:
            calibrator.record(r)

        result = calibrator.calibrate()
        assert result[LiquidityBucket.MID_CAP] == IMPACT_PARAMS[LiquidityBucket.MID_CAP]

    def test_empty_calibrator_returns_all_defaults(self):
        """No records at all → every bucket gets defaults."""
        calibrator = ImpactCalibrator()
        result = calibrator.calibrate()
        for bucket in LiquidityBucket:
            assert result[bucket] == IMPACT_PARAMS[bucket]


class TestGetCalibratedParams:
    def test_returns_calibrated_after_calibration(self):
        calibrator = ImpactCalibrator()
        records = _make_synthetic_records(50, LiquidityBucket.LARGE_CAP)
        for r in records:
            calibrator.record(r)
        calibrator.calibrate()

        params = calibrator.get_calibrated_params(LiquidityBucket.LARGE_CAP)
        assert 'permanent_impact' in params
        assert 'temporary_impact' in params

    def test_returns_defaults_before_calibration(self):
        calibrator = ImpactCalibrator()
        params = calibrator.get_calibrated_params(LiquidityBucket.MEGA_CAP)
        assert params == IMPACT_PARAMS[LiquidityBucket.MEGA_CAP]


class TestCalibratedParamsUsedInExecution:
    def test_execution_model_uses_calibrator(self):
        """AlmgrenChrissExecutionModel with calibrator should use calibrated params."""
        calibrator = ImpactCalibrator()
        # Feed records into LARGE_CAP bucket (ADV=20M)
        records = _make_synthetic_records(50, LiquidityBucket.LARGE_CAP, daily_volume=20_000_000)
        for r in records:
            calibrator.record(r)
        calibrator.calibrate()

        # Verify calibrated params actually differ from defaults
        cal_params = calibrator.get_calibrated_params(LiquidityBucket.LARGE_CAP)
        defaults = IMPACT_PARAMS[LiquidityBucket.LARGE_CAP]
        assert (
            cal_params['permanent_impact'] != defaults['permanent_impact']
            or cal_params['temporary_impact'] != defaults['temporary_impact']
        ), "Calibrated params should differ from defaults for a valid test"

        # Use large order + many slices + high risk_aversion so trajectory
        # shape is sensitive to impact params and quantization doesn't mask
        exec_with = AlmgrenChrissExecutionModel(
            num_slices=10,
            daily_volume=20_000_000,
            risk_aversion=1.0,
            calibrator=calibrator,
        )
        exec_with.set_current_positions({})

        exec_without = AlmgrenChrissExecutionModel(
            num_slices=10,
            daily_volume=20_000_000,
            risk_aversion=1.0,
        )
        exec_without.set_current_positions({})

        target = PortfolioTarget(symbol='AAPL', quantity=Decimal("100000"))

        orders_with = exec_with.execute([target])
        orders_without = exec_without.execute([target])

        # Both should produce first-slice orders
        assert len(orders_with) == 1
        assert len(orders_without) == 1

        # The first slice quantity should differ due to different impact params
        # shaping the trajectory differently
        sched_with = exec_with.get_scheduled_slices()
        sched_without = exec_without.get_scheduled_slices()

        qty_with = [s['order'].quantity for s in sched_with]
        qty_without = [s['order'].quantity for s in sched_without]
        assert qty_with != qty_without
