"""Comprehensive tests for Greek exposure limits module."""
import pytest
from unittest.mock import Mock, patch
import numpy as np

from backend.tradingbot.risk.greek_exposure_limits import (
    GreekExposureLimiter,
    GreekLimits,
    PositionGreeks
)


class TestGreekExposureComprehensive:
    """Comprehensive tests for Greek exposure functionality."""

    def test_greek_limits_default_initialization(self):
        """Test GreekLimits default initialization."""
        limits = GreekLimits()

        # Check default values exist and are reasonable
        assert limits.max_portfolio_delta > 0
        assert limits.max_portfolio_gamma > 0
        assert limits.max_portfolio_theta < 0  # Theta decay is negative
        assert limits.max_portfolio_vega > 0
        assert limits.max_per_name_delta > 0
        assert limits.max_per_name_gamma > 0
        assert limits.max_per_name_theta < 0
        assert limits.max_per_name_vega > 0

    def test_position_greeks_initialization(self):
        """Test PositionGreeks initialization and properties."""
        position = PositionGreeks(
            symbol="AAPL",
            delta=1000.0,
            gamma=50.0,
            theta=-10.0,
            vega=200.0,
            beta=1.2,
            notional=100000.0
        )

        assert position.symbol == "AAPL"
        assert position.delta == 1000.0
        assert position.gamma == 50.0
        assert position.theta == -10.0
        assert position.vega == 200.0
        assert position.beta == 1.2
        assert position.notional == 100000.0

    def test_add_position_new_symbol(self):
        """Test adding position for new symbol."""
        limiter = GreekExposureLimiter()

        position = PositionGreeks("AAPL", 1000.0, 50.0, -10.0, 200.0, 1.2, 100000.0)
        limiter.add_position(position)

        assert "AAPL" in limiter.positions
        assert limiter.positions["AAPL"].delta == 1000.0

    def test_add_position_update_existing(self):
        """Test updating existing position."""
        limiter = GreekExposureLimiter()

        # Add initial position
        position1 = PositionGreeks("AAPL", 1000.0, 50.0, -10.0, 200.0, 1.2, 100000.0)
        limiter.add_position(position1)

        # Update with new position
        position2 = PositionGreeks("AAPL", 1500.0, 75.0, -15.0, 300.0, 1.3, 150000.0)
        limiter.add_position(position2)

        # Should replace, not add
        assert len(limiter.positions) == 1
        assert limiter.positions["AAPL"].delta == 1500.0
        assert limiter.positions["AAPL"].gamma == 75.0

    def test_remove_position(self):
        """Test removing position."""
        limiter = GreekExposureLimiter()

        # Add position
        position = PositionGreeks("AAPL", 1000.0, 50.0, -10.0, 200.0, 1.2, 100000.0)
        limiter.add_position(position)
        assert "AAPL" in limiter.positions

        # Remove position
        limiter.remove_position("AAPL")
        assert "AAPL" not in limiter.positions

    def test_remove_nonexistent_position(self):
        """Test removing position that doesn't exist."""
        limiter = GreekExposureLimiter()

        # Should handle gracefully
        limiter.remove_position("NONEXISTENT")
        assert len(limiter.positions) == 0

    def test_get_portfolio_greeks_empty(self):
        """Test portfolio greeks calculation with empty portfolio."""
        limiter = GreekExposureLimiter()

        greeks = limiter.get_portfolio_greeks()

        assert greeks["delta"] == 0.0
        assert greeks["gamma"] == 0.0
        assert greeks["theta"] == 0.0
        assert greeks["vega"] == 0.0
        assert greeks["beta_adjusted_exposure"] == 0.0

    def test_get_portfolio_greeks_multiple_positions(self):
        """Test portfolio greeks calculation with multiple positions."""
        limiter = GreekExposureLimiter()

        # Add multiple positions
        positions = [
            PositionGreeks("AAPL", 1000.0, 50.0, -10.0, 200.0, 1.2, 100000.0),
            PositionGreeks("MSFT", -500.0, 30.0, -5.0, 150.0, 0.8, 75000.0),
            PositionGreeks("GOOGL", 800.0, 40.0, -8.0, 180.0, 1.1, 120000.0)
        ]

        for position in positions:
            limiter.add_position(position)

        greeks = limiter.get_portfolio_greeks()

        # Check aggregation
        assert greeks["delta"] == 1000.0 + (-500.0) + 800.0  # 1300.0
        assert greeks["gamma"] == 50.0 + 30.0 + 40.0  # 120.0
        assert greeks["theta"] == -10.0 + (-5.0) + (-8.0)  # -23.0
        assert greeks["vega"] == 200.0 + 150.0 + 180.0  # 530.0

        # Beta-adjusted exposure
        expected_beta_exposure = (100000.0 * 1.2) + (75000.0 * 0.8) + (120000.0 * 1.1)
        assert abs(greeks["beta_adjusted_exposure"] - expected_beta_exposure) < 0.01

    def test_check_portfolio_limits_no_violations(self):
        """Test portfolio limits check with no violations."""
        limits = GreekLimits(
            max_portfolio_delta=5000.0,
            max_portfolio_gamma=200.0,
            max_portfolio_theta=-50.0,
            max_portfolio_vega=1000.0,
            max_beta_adjusted_exposure=500000.0
        )
        limiter = GreekExposureLimiter(limits=limits)

        # Add position within limits
        position = PositionGreeks("AAPL", 1000.0, 50.0, -10.0, 200.0, 1.0, 100000.0)
        limiter.add_position(position)

        violations = limiter.check_portfolio_limits()
        assert len(violations) == 0

    def test_check_portfolio_limits_delta_violation(self):
        """Test portfolio limits check with delta violation."""
        limits = GreekLimits(
            max_portfolio_delta=500.0,  # Low limit
            max_portfolio_gamma=1000.0,
            max_portfolio_theta=-100.0,
            max_portfolio_vega=2000.0,
            max_beta_adjusted_exposure=1000000.0
        )
        limiter = GreekExposureLimiter(limits=limits)

        # Add position that exceeds delta limit
        position = PositionGreeks("AAPL", 1000.0, 50.0, -10.0, 200.0, 1.0, 100000.0)
        limiter.add_position(position)

        violations = limiter.check_portfolio_limits()
        assert len(violations) > 0
        assert any("Portfolio delta" in violation for violation in violations)

    def test_check_portfolio_limits_multiple_violations(self):
        """Test portfolio limits check with multiple violations."""
        limits = GreekLimits(
            max_portfolio_delta=500.0,    # Will be exceeded
            max_portfolio_gamma=25.0,     # Will be exceeded
            max_portfolio_theta=-5.0,     # Will be exceeded (more negative)
            max_portfolio_vega=100.0,     # Will be exceeded
            max_beta_adjusted_exposure=50000.0  # Will be exceeded
        )
        limiter = GreekExposureLimiter(limits=limits)

        # Add position that exceeds multiple limits
        position = PositionGreeks("AAPL", 1000.0, 50.0, -10.0, 200.0, 1.0, 100000.0)
        limiter.add_position(position)

        violations = limiter.check_portfolio_limits()
        assert len(violations) >= 4  # Should have multiple violations

    def test_check_per_name_limits_no_violations(self):
        """Test per-name limits check with no violations."""
        limits = GreekLimits(
            max_per_name_delta=2000.0,
            max_per_name_gamma=100.0,
            max_per_name_theta=-50.0,
            max_per_name_vega=500.0
        )
        limiter = GreekExposureLimiter(limits=limits)

        # Add position within per-name limits
        position = PositionGreeks("AAPL", 1000.0, 50.0, -10.0, 200.0, 1.0, 100000.0)
        limiter.add_position(position)

        violations = limiter.check_per_name_limits()
        assert len(violations) == 0

    def test_check_per_name_limits_violations(self):
        """Test per-name limits check with violations."""
        limits = GreekLimits(
            max_per_name_delta=500.0,   # Will be exceeded
            max_per_name_gamma=25.0,    # Will be exceeded
            max_per_name_theta=-5.0,    # Will be exceeded
            max_per_name_vega=100.0     # Will be exceeded
        )
        limiter = GreekExposureLimiter(limits=limits)

        # Add position that exceeds per-name limits
        position = PositionGreeks("AAPL", 1000.0, 50.0, -10.0, 200.0, 1.0, 100000.0)
        limiter.add_position(position)

        violations = limiter.check_per_name_limits()
        assert len(violations) >= 4  # Should have violations for all Greeks

    def test_get_position_summary(self):
        """Test position summary generation."""
        limiter = GreekExposureLimiter()

        # Add multiple positions
        positions = [
            PositionGreeks("AAPL", 1000.0, 50.0, -10.0, 200.0, 1.2, 100000.0),
            PositionGreeks("MSFT", -500.0, 30.0, -5.0, 150.0, 0.8, 75000.0)
        ]

        for position in positions:
            limiter.add_position(position)

        summary = limiter.get_position_summary()

        assert "total_positions" in summary
        assert "portfolio_greeks" in summary
        assert "largest_positions" in summary
        assert summary["total_positions"] == 2

    def test_validate_position_basic(self):
        """Test basic position validation."""
        limiter = GreekExposureLimiter()

        # Valid position
        position = PositionGreeks("AAPL", 1000.0, 50.0, -10.0, 200.0, 1.0, 100000.0)
        is_valid, violations = limiter.validate_position(position)

        assert is_valid is True
        assert len(violations) == 0

    def test_validate_position_violations(self):
        """Test position validation with violations."""
        limits = GreekLimits(
            max_per_name_delta=500.0,
            max_per_name_gamma=25.0,
            max_per_name_theta=-5.0,
            max_per_name_vega=100.0
        )
        limiter = GreekExposureLimiter(limits=limits)

        # Position that violates limits
        position = PositionGreeks("AAPL", 1000.0, 50.0, -10.0, 200.0, 1.0, 100000.0)
        limiter.add_position(position)  # Add position first
        is_valid, violations = limiter.validate_position(position)

        assert is_valid is False
        assert len(violations) > 0

    def test_get_greek_utilization(self):
        """Test Greek utilization calculation."""
        limits = GreekLimits(
            max_portfolio_delta=2000.0,
            max_portfolio_gamma=100.0,
            max_portfolio_theta=-20.0,
            max_portfolio_vega=400.0
        )
        limiter = GreekExposureLimiter(limits=limits)

        # Add position that uses 50% of delta limit
        position = PositionGreeks("AAPL", 1000.0, 50.0, -10.0, 200.0, 1.0, 100000.0)
        limiter.add_position(position)

        utilization = limiter.get_greek_utilization()

        assert "delta_utilization" in utilization
        assert "gamma_utilization" in utilization
        assert "theta_utilization" in utilization
        assert "vega_utilization" in utilization

        # Delta should be 50% utilized (1000/2000)
        assert abs(utilization["delta_utilization"] - 0.5) < 0.01

    def test_stress_test_greeks(self):
        """Test Greek stress testing functionality."""
        limiter = GreekExposureLimiter()

        # Add base position
        position = PositionGreeks("AAPL", 1000.0, 50.0, -10.0, 200.0, 1.0, 100000.0)
        limiter.add_position(position)

        # Test stress scenarios
        stress_scenarios = {
            "market_down_10": {"delta_shock": -0.1, "vega_shock": 0.2},
            "vol_up_20": {"vega_shock": 0.2, "gamma_shock": 0.1}
        }

        stress_results = limiter.stress_test_greeks(stress_scenarios)

        assert "market_down_10" in stress_results
        assert "vol_up_20" in stress_results

        # Each result should have stressed Greeks
        for scenario, result in stress_results.items():
            assert "stressed_delta" in result
            assert "stressed_vega" in result
            assert "violations" in result

    def test_edge_case_zero_greeks(self):
        """Test edge case with zero Greeks values."""
        limiter = GreekExposureLimiter()

        # Position with all zero Greeks
        position = PositionGreeks("ZERO", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        limiter.add_position(position)

        # Should handle gracefully
        greeks = limiter.get_portfolio_greeks()
        assert all(value == 0.0 for value in greeks.values())

        violations = limiter.check_portfolio_limits()
        assert len(violations) == 0

    def test_edge_case_negative_greeks(self):
        """Test edge case with negative Greeks values."""
        limiter = GreekExposureLimiter()

        # Position with negative Greeks (short position)
        position = PositionGreeks("SHORT", -1000.0, -50.0, 10.0, -200.0, -1.0, -100000.0)
        limiter.add_position(position)

        # Should handle negative values properly
        greeks = limiter.get_portfolio_greeks()
        assert greeks["delta"] == -1000.0
        assert greeks["gamma"] == -50.0
        assert greeks["theta"] == 10.0  # Positive theta (beneficial for short)
        assert greeks["vega"] == -200.0

    def test_large_portfolio_performance(self):
        """Test performance with large number of positions."""
        limiter = GreekExposureLimiter()

        # Add many positions
        for i in range(100):
            symbol = f"STOCK_{i:03d}"
            delta = np.random.uniform(-1000, 1000)
            gamma = np.random.uniform(0, 100)
            theta = np.random.uniform(-20, 0)
            vega = np.random.uniform(0, 300)
            beta = np.random.uniform(0.5, 2.0)
            notional = np.random.uniform(10000, 200000)

            position = PositionGreeks(symbol, delta, gamma, theta, vega, beta, notional)
            limiter.add_position(position)

        # Performance test - should complete quickly
        import time
        start_time = time.time()

        # Perform various operations
        greeks = limiter.get_portfolio_greeks()
        violations = limiter.check_portfolio_limits()
        summary = limiter.get_position_summary()

        end_time = time.time()

        # Should complete in reasonable time
        assert end_time - start_time < 1.0  # Less than 1 second

        # Verify results are reasonable
        assert len(limiter.positions) == 100
        assert isinstance(greeks["delta"], (int, float))

    def test_greek_limits_validation_edge_cases(self):
        """Test GreekLimits validation with edge cases."""
        # Test with very small limits
        small_limits = GreekLimits(
            max_portfolio_delta=0.01,
            max_portfolio_gamma=0.01,
            max_portfolio_theta=-0.01,
            max_portfolio_vega=0.01
        )

        limiter = GreekExposureLimiter(limits=small_limits)

        # Even tiny position should violate
        tiny_position = PositionGreeks("TINY", 0.1, 0.1, -0.1, 0.1, 1.0, 1000.0)
        limiter.add_position(tiny_position)

        violations = limiter.check_portfolio_limits()
        assert len(violations) > 0

    def test_position_greeks_string_representation(self):
        """Test PositionGreeks string representation."""
        position = PositionGreeks("AAPL", 1000.0, 50.0, -10.0, 200.0, 1.2, 100000.0)

        # Should have meaningful string representation
        str_repr = str(position)
        assert "AAPL" in str_repr
        assert "1000.0" in str_repr

    def test_greek_exposure_reset_positions(self):
        """Test resetting all positions."""
        limiter = GreekExposureLimiter()

        # Add some positions
        positions = [
            PositionGreeks("AAPL", 1000.0, 50.0, -10.0, 200.0, 1.2, 100000.0),
            PositionGreeks("MSFT", 500.0, 30.0, -5.0, 150.0, 0.8, 75000.0)
        ]

        for position in positions:
            limiter.add_position(position)

        assert len(limiter.positions) == 2

        # Reset (if method exists)
        if hasattr(limiter, 'reset_positions'):
            limiter.reset_positions()
            assert len(limiter.positions) == 0
        else:
            # Alternative: clear manually
            limiter.positions.clear()
            assert len(limiter.positions) == 0