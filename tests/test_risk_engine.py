"""Tests for the risk engine with kill-switch functionality"""
import pytest
from backend.tradingbot.risk.engine import RiskEngine, RiskLimits


class TestRiskEngine:

    def test_var_cvar_basic(self):
        """Test VaR and CVaR calculations"""
        limits = RiskLimits(max_total_risk=0.3, max_position_size=0.1)
        engine = RiskEngine(limits)

        samples = [-100, -20, 10, 15, -5, 30, 40]
        result = engine.var_cvar(samples, alpha=0.95)

        assert set(result.keys()) == {"var", "cvar"}
        assert result["var"] >= 0
        assert result["cvar"] >= 0

    def test_var_cvar_empty_samples(self):
        """Test VaR/CVaR with empty samples"""
        limits = RiskLimits(max_total_risk=0.3, max_position_size=0.1)
        engine = RiskEngine(limits)

        result = engine.var_cvar([])
        assert result == {"var": 0.0, "cvar": 0.0}

    def test_pretrade_check_total_risk(self):
        """Test pre-trade check blocks excessive total risk"""
        limits = RiskLimits(max_total_risk=0.3, max_position_size=0.1)
        engine = RiskEngine(limits)

        # Should pass - under limit
        assert engine.pretrade_check(current_exposure=0.2, new_position_risk=0.05)

        # Should fail - over total limit
        assert not engine.pretrade_check(current_exposure=0.25, new_position_risk=0.08)

    def test_pretrade_check_position_size(self):
        """Test pre-trade check blocks excessive position size"""
        limits = RiskLimits(max_total_risk=0.3, max_position_size=0.1)
        engine = RiskEngine(limits)

        # Should pass - under position limit
        assert engine.pretrade_check(current_exposure=0.05, new_position_risk=0.08)

        # Should fail - over position limit
        assert not engine.pretrade_check(current_exposure=0.05, new_position_risk=0.12)

    def test_drawdown_calculation(self):
        """Test drawdown calculation and peak tracking"""
        limits = RiskLimits(max_total_risk=0.3, max_position_size=0.1)
        engine = RiskEngine(limits)

        # Set initial peak
        engine.update_peak(100000)

        # Test normal drawdown
        dd = engine.drawdown(90000)
        assert abs(dd - 0.1) < 0.001  # 10% drawdown

        # Test new peak
        engine.update_peak(110000)
        dd = engine.drawdown(105000)
        assert abs(dd - (5000/110000)) < 0.001  # ~4.5% drawdown

    def test_posttrade_check_normal(self):
        """Test post-trade check with normal conditions"""
        limits = RiskLimits(max_total_risk=0.3, max_position_size=0.1,
                          max_drawdown=0.2, kill_switch_dd=0.25)
        engine = RiskEngine(limits)

        engine.update_peak(100000)

        # Should pass - small drawdown
        assert engine.posttrade_check(equity=95000)  # 5% DD

        # Should pass - at max drawdown
        assert engine.posttrade_check(equity=80000)  # 20% DD

    def test_posttrade_check_max_drawdown(self):
        """Test post-trade check fails with excessive drawdown"""
        limits = RiskLimits(max_total_risk=0.3, max_position_size=0.1,
                          max_drawdown=0.2, kill_switch_dd=0.25)
        engine = RiskEngine(limits)

        engine.update_peak(100000)

        # Should fail - over max drawdown but under kill switch
        assert not engine.posttrade_check(equity=75000)  # 25% DD > 20% max

    def test_kill_switch_activation(self):
        """Test kill switch activation and blocking"""
        limits = RiskLimits(max_total_risk=0.3, max_position_size=0.1,
                          max_drawdown=0.2, kill_switch_dd=0.25)
        engine = RiskEngine(limits)

        engine.update_peak(100000)

        # Should activate kill switch
        assert not engine.posttrade_check(equity=70000)  # 30% DD > 25% kill switch
        assert engine.kill_switch_active

        # Should block all trades when kill switch is active
        assert not engine.pretrade_check(current_exposure=0.05, new_position_risk=0.02)

    def test_kill_switch_reset(self):
        """Test manual kill switch reset"""
        limits = RiskLimits(max_total_risk=0.3, max_position_size=0.1,
                          max_drawdown=0.2, kill_switch_dd=0.25)
        engine = RiskEngine(limits)

        # Activate kill switch
        engine.update_peak(100000)
        engine.posttrade_check(equity=70000)  # Activates kill switch
        assert engine.kill_switch_active

        # Reset and verify
        engine.reset_kill_switch()
        assert not engine.kill_switch_active

        # Should allow trades again
        assert engine.pretrade_check(current_exposure=0.05, new_position_risk=0.02)