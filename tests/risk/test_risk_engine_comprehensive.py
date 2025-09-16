"""Comprehensive tests for risk engine module to achieve 100% coverage."""
import pytest
from unittest.mock import Mock, patch
import logging

from backend.tradingbot.risk.engine import RiskEngine, RiskLimits


class TestRiskEngineComprehensive:
    """Comprehensive tests to achieve 100% coverage of risk engine."""

    def test_risk_engine_kill_switch_reset(self):
        """Test kill switch reset functionality (missing coverage)."""
        limits = RiskLimits(
            max_drawdown=0.05,
            kill_switch_dd=0.03,
            max_position_size=10000.0
        )
        engine = RiskEngine(limits=limits, initial_equity=100000.0)

        # Trigger kill switch
        engine.check_drawdown(97000.0)  # 3% drawdown, triggers kill switch
        assert engine.kill_switch_active is True

        # Test reset functionality
        with patch('backend.tradingbot.risk.engine.log') as mock_log:
            engine.reset_kill_switch()

            # Verify kill switch is reset
            assert engine.kill_switch_active is False

            # Verify warning was logged
            mock_log.warning.assert_called_once_with("Kill switch manually reset")

    def test_risk_engine_edge_case_exact_drawdown_limits(self):
        """Test edge cases with exact drawdown limits."""
        limits = RiskLimits(
            max_drawdown=0.10,  # 10%
            kill_switch_dd=0.05,  # 5%
            max_position_size=10000.0
        )
        engine = RiskEngine(limits=limits, initial_equity=100000.0)

        # Test exactly at kill switch threshold
        result = engine.check_drawdown(95000.0)  # Exactly 5% drawdown
        assert result is False  # Should trigger kill switch
        assert engine.kill_switch_active is True

        # Reset for next test
        engine.reset_kill_switch()

        # Test exactly at max drawdown threshold (but not kill switch)
        limits2 = RiskLimits(
            max_drawdown=0.10,  # 10%
            kill_switch_dd=0.15,  # 15% (higher than max)
            max_position_size=10000.0
        )
        engine2 = RiskEngine(limits=limits2, initial_equity=100000.0)

        with patch('backend.tradingbot.risk.engine.log') as mock_log:
            result = engine2.check_drawdown(90000.0)  # Exactly 10% drawdown

            assert result is False  # Should fail max drawdown check
            assert engine2.kill_switch_active is False  # Kill switch not triggered

            # Verify warning was logged
            mock_log.warning.assert_called_once()
            call_args = mock_log.warning.call_args[0][0]
            assert "Max drawdown exceeded" in call_args
            assert "10.00%" in call_args

    def test_risk_engine_logging_scenarios(self):
        """Test various logging scenarios for complete coverage."""
        limits = RiskLimits(
            max_drawdown=0.10,
            kill_switch_dd=0.05,
            max_position_size=10000.0
        )
        engine = RiskEngine(limits=limits, initial_equity=100000.0)

        # Test kill switch logging
        with patch('backend.tradingbot.risk.engine.log') as mock_log:
            engine.check_drawdown(94000.0)  # 6% drawdown, triggers kill switch

            # Verify kill switch error was logged
            mock_log.error.assert_called_once()
            call_args = mock_log.error.call_args[0][0]
            assert "Kill switch activated" in call_args
            assert "6.00%" in call_args
            assert "5.00%" in call_args

    def test_risk_engine_position_size_validation(self):
        """Test position size validation edge cases."""
        limits = RiskLimits(
            max_drawdown=0.20,
            kill_switch_dd=0.15,
            max_position_size=5000.0
        )
        engine = RiskEngine(limits=limits, initial_equity=100000.0)

        # Test exactly at limit
        assert engine.validate_position_size(5000.0) is True

        # Test just over limit
        assert engine.validate_position_size(5000.01) is False

        # Test zero position
        assert engine.validate_position_size(0.0) is True

        # Test negative position (absolute value)
        assert engine.validate_position_size(-3000.0) is True

        # Test large negative position
        assert engine.validate_position_size(-6000.0) is False

    def test_risk_engine_drawdown_calculation_precision(self):
        """Test drawdown calculation with high precision scenarios."""
        limits = RiskLimits(
            max_drawdown=0.001,  # 0.1% very tight limit
            kill_switch_dd=0.0005,  # 0.05% very tight
            max_position_size=10000.0
        )
        engine = RiskEngine(limits=limits, initial_equity=100000.0)

        # Test very small drawdown that should trigger kill switch
        result = engine.check_drawdown(99949.99)  # 0.05001% drawdown
        assert result is False
        assert engine.kill_switch_active is True

        # Reset and test with different engine
        engine2 = RiskEngine(limits=limits, initial_equity=100000.0)

        # Test drawdown just under kill switch but over max
        result2 = engine2.check_drawdown(99900.0)  # 0.1% drawdown
        assert result2 is False
        assert engine2.kill_switch_active is True  # Should trigger kill switch since 0.1% >= 0.05%

    def test_risk_engine_property_access(self):
        """Test property access for kill switch status."""
        limits = RiskLimits(
            max_drawdown=0.10,
            kill_switch_dd=0.05,
            max_position_size=10000.0
        )
        engine = RiskEngine(limits=limits, initial_equity=100000.0)

        # Initially false
        assert engine.kill_switch_active is False

        # Trigger kill switch
        engine.check_drawdown(94000.0)  # 6% drawdown

        # Should be true
        assert engine.kill_switch_active is True

        # Reset
        engine.reset_kill_switch()

        # Should be false again
        assert engine.kill_switch_active is False

    def test_risk_engine_initialization_edge_cases(self):
        """Test risk engine initialization edge cases."""
        # Test with zero initial equity
        limits = RiskLimits(
            max_drawdown=0.10,
            kill_switch_dd=0.05,
            max_position_size=10000.0
        )

        engine = RiskEngine(limits=limits, initial_equity=0.0)
        assert engine.initial_equity == 0.0

        # Any current equity should result in infinite loss or gain
        # But the engine should handle it gracefully
        result = engine.check_drawdown(-1000.0)  # Negative equity
        # Should handle gracefully without crashing

    def test_risk_engine_extreme_values(self):
        """Test risk engine with extreme values."""
        limits = RiskLimits(
            max_drawdown=0.99,  # 99% drawdown allowed
            kill_switch_dd=0.95,  # 95% kill switch
            max_position_size=1e12  # Very large position size
        )
        engine = RiskEngine(limits=limits, initial_equity=1e6)  # $1M initial

        # Test very large drawdown that's still within limits
        result = engine.check_drawdown(60000.0)  # 94% drawdown
        assert result is True  # Should pass
        assert engine.kill_switch_active is False

        # Test kill switch trigger
        result = engine.check_drawdown(40000.0)  # 96% drawdown
        assert result is False  # Should fail
        assert engine.kill_switch_active is True

    def test_risk_engine_gains_scenario(self):
        """Test risk engine behavior with gains (negative drawdown)."""
        limits = RiskLimits(
            max_drawdown=0.10,
            kill_switch_dd=0.05,
            max_position_size=10000.0
        )
        engine = RiskEngine(limits=limits, initial_equity=100000.0)

        # Test with gains (negative drawdown)
        result = engine.check_drawdown(150000.0)  # 50% gain
        assert result is True  # Should pass
        assert engine.kill_switch_active is False

        # Test with even larger gains
        result = engine.check_drawdown(500000.0)  # 400% gain
        assert result is True  # Should pass
        assert engine.kill_switch_active is False

    def test_risk_limits_data_class(self):
        """Test RiskLimits data class functionality."""
        # Test default initialization
        limits = RiskLimits()
        assert hasattr(limits, 'max_drawdown')
        assert hasattr(limits, 'kill_switch_dd')
        assert hasattr(limits, 'max_position_size')

        # Test custom initialization
        custom_limits = RiskLimits(
            max_drawdown=0.15,
            kill_switch_dd=0.08,
            max_position_size=25000.0
        )
        assert custom_limits.max_drawdown == 0.15
        assert custom_limits.kill_switch_dd == 0.08
        assert custom_limits.max_position_size == 25000.0

    def test_risk_engine_sequence_of_operations(self):
        """Test sequence of risk operations for comprehensive coverage."""
        limits = RiskLimits(
            max_drawdown=0.10,
            kill_switch_dd=0.05,
            max_position_size=10000.0
        )
        engine = RiskEngine(limits=limits, initial_equity=100000.0)

        # 1. Start with valid position size
        assert engine.validate_position_size(5000.0) is True

        # 2. Check drawdown - still good
        assert engine.check_drawdown(98000.0) is True  # 2% drawdown

        # 3. Increase drawdown to trigger kill switch
        assert engine.check_drawdown(94000.0) is False  # 6% drawdown
        assert engine.kill_switch_active is True

        # 4. Position size should still validate (kill switch doesn't affect this)
        assert engine.validate_position_size(5000.0) is True

        # 5. Reset kill switch
        engine.reset_kill_switch()
        assert engine.kill_switch_active is False

        # 6. Now drawdown check should work again if within limits
        assert engine.check_drawdown(96000.0) is True  # 4% drawdown (under kill switch)

        # 7. Test invalid position size
        assert engine.validate_position_size(15000.0) is False  # Over limit


class TestRiskEngineErrorHandling:
    """Test error handling and edge cases."""

    def test_risk_engine_with_none_values(self):
        """Test risk engine behavior with None values."""
        limits = RiskLimits(
            max_drawdown=0.10,
            kill_switch_dd=0.05,
            max_position_size=10000.0
        )
        engine = RiskEngine(limits=limits, initial_equity=100000.0)

        # Test with None position size
        with pytest.raises(TypeError):
            engine.validate_position_size(None)

        # Test with None current equity
        with pytest.raises(TypeError):
            engine.check_drawdown(None)

    def test_risk_engine_with_string_values(self):
        """Test risk engine behavior with string values."""
        limits = RiskLimits(
            max_drawdown=0.10,
            kill_switch_dd=0.05,
            max_position_size=10000.0
        )
        engine = RiskEngine(limits=limits, initial_equity=100000.0)

        # Test with string position size
        with pytest.raises(TypeError):
            engine.validate_position_size("5000")

        # Test with string current equity
        with pytest.raises(TypeError):
            engine.check_drawdown("95000")

    def test_risk_engine_logging_integration(self):
        """Test logging integration and log levels."""
        limits = RiskLimits(
            max_drawdown=0.10,
            kill_switch_dd=0.05,
            max_position_size=10000.0
        )
        engine = RiskEngine(limits=limits, initial_equity=100000.0)

        # Capture logging
        with patch('backend.tradingbot.risk.engine.log') as mock_log:
            # Test kill switch logging (error level)
            engine.check_drawdown(94000.0)  # Triggers kill switch
            mock_log.error.assert_called()

            # Reset and test max drawdown logging (error level)
            engine.reset_kill_switch()
            mock_log.warning.assert_called()  # Reset warning

            # Test max drawdown without kill switch
            limits2 = RiskLimits(max_drawdown=0.08, kill_switch_dd=0.15, max_position_size=10000.0)
            engine2 = RiskEngine(limits=limits2, initial_equity=100000.0)

            engine2.check_drawdown(91000.0)  # 9% drawdown, over 8% limit
            # Should log max drawdown warning (not error since kill switch not triggered)
            mock_log.warning.assert_called()