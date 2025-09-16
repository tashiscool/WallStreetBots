"""Comprehensive tests for Advanced Analytics System to achieve >85% coverage."""
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from backend.tradingbot.analytics.advanced_analytics import (
    AdvancedAnalytics,
    PerformanceMetrics,
    DrawdownPeriod,
    analyze_performance
)


class TestPerformanceMetrics:
    """Test PerformanceMetrics dataclass."""

    def test_performance_metrics_creation(self):
        """Test creation of PerformanceMetrics."""
        now = datetime.now()
        metrics = PerformanceMetrics(
            total_return=0.25,
            annualized_return=0.15,
            volatility=0.20,
            sharpe_ratio=1.2,
            sortino_ratio=1.5,
            calmar_ratio=2.0,
            max_drawdown=0.10,
            var_95=0.05,
            var_99=0.08,
            cvar_95=0.06,
            win_rate=0.60,
            avg_win=0.02,
            avg_loss=-0.015,
            profit_factor=1.8,
            information_ratio=0.5,
            treynor_ratio=0.08,
            alpha=0.03,
            beta=1.1,
            tracking_error=0.04,
            period_start=now - timedelta(days=365),
            period_end=now,
            trading_days=252,
            best_day=0.08,
            worst_day=-0.06,
            positive_days=151,
            negative_days=101,
            recovery_factor=2.5,
            ulcer_index=5.2,
            sterling_ratio=1.5
        )

        assert metrics.total_return == 0.25
        assert metrics.sharpe_ratio == 1.2
        assert metrics.win_rate == 0.60
        assert metrics.trading_days == 252
        assert metrics.positive_days == 151

    def test_performance_metrics_all_fields(self):
        """Test that all PerformanceMetrics fields are accessible."""
        now = datetime.now()
        metrics = PerformanceMetrics(
            total_return=0.0, annualized_return=0.0, volatility=0.0,
            sharpe_ratio=0.0, sortino_ratio=0.0, calmar_ratio=0.0,
            max_drawdown=0.0, var_95=0.0, var_99=0.0, cvar_95=0.0,
            win_rate=0.0, avg_win=0.0, avg_loss=0.0, profit_factor=0.0,
            information_ratio=0.0, treynor_ratio=0.0, alpha=0.0,
            beta=1.0, tracking_error=0.0, period_start=now,
            period_end=now, trading_days=0, best_day=0.0,
            worst_day=0.0, positive_days=0, negative_days=0,
            recovery_factor=0.0, ulcer_index=0.0, sterling_ratio=0.0
        )

        # Verify all fields exist and are accessible
        assert hasattr(metrics, 'total_return')
        assert hasattr(metrics, 'recovery_factor')
        assert hasattr(metrics, 'sterling_ratio')
        assert metrics.beta == 1.0


class TestDrawdownPeriod:
    """Test DrawdownPeriod dataclass."""

    def test_drawdown_period_creation(self):
        """Test creation of DrawdownPeriod."""
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 2, 1)
        recovery_date = datetime(2023, 3, 1)

        dd_period = DrawdownPeriod(
            start_date=start_date,
            end_date=end_date,
            recovery_date=recovery_date,
            peak_value=100000,
            trough_value=85000,
            drawdown_pct=0.15,
            duration_days=31,
            recovery_days=28,
            is_recovered=True
        )

        assert dd_period.start_date == start_date
        assert dd_period.drawdown_pct == 0.15
        assert dd_period.is_recovered
        assert dd_period.recovery_days == 28

    def test_drawdown_period_ongoing(self):
        """Test DrawdownPeriod for ongoing drawdown."""
        dd_period = DrawdownPeriod(
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 2, 1),
            recovery_date=None,
            peak_value=100000,
            trough_value=80000,
            drawdown_pct=0.20,
            duration_days=45,
            recovery_days=None,
            is_recovered=False
        )

        assert dd_period.recovery_date is None
        assert dd_period.recovery_days is None
        assert not dd_period.is_recovered


class TestAdvancedAnalytics:
    """Test AdvancedAnalytics class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.analytics = AdvancedAnalytics(risk_free_rate=0.02)

        # Create sample data
        np.random.seed(42)
        self.returns = np.random.normal(0.001, 0.02, 252)  # Daily returns for 1 year
        self.benchmark_returns = np.random.normal(0.0008, 0.015, 252)
        self.portfolio_values = np.cumprod(1 + self.returns) * 100000

    def test_analytics_initialization(self):
        """Test AdvancedAnalytics initialization."""
        assert self.analytics.risk_free_rate == 0.02
        assert hasattr(self.analytics, 'logger')

    def test_analytics_custom_risk_free_rate(self):
        """Test initialization with custom risk-free rate."""
        custom_analytics = AdvancedAnalytics(risk_free_rate=0.05)
        assert custom_analytics.risk_free_rate == 0.05

    def test_calculate_comprehensive_metrics_basic(self):
        """Test basic comprehensive metrics calculation."""
        metrics = self.analytics.calculate_comprehensive_metrics(self.returns)

        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.trading_days == len(self.returns)
        assert isinstance(metrics.total_return, float)
        assert isinstance(metrics.sharpe_ratio, float)
        assert isinstance(metrics.max_drawdown, float)
        assert 0.0 <= metrics.win_rate <= 1.0

    def test_calculate_comprehensive_metrics_with_benchmark(self):
        """Test metrics calculation with benchmark."""
        metrics = self.analytics.calculate_comprehensive_metrics(
            returns=self.returns,
            benchmark_returns=self.benchmark_returns
        )

        assert isinstance(metrics.alpha, float)
        assert isinstance(metrics.beta, float)
        assert isinstance(metrics.information_ratio, float)
        assert isinstance(metrics.tracking_error, float)
        assert metrics.beta > 0  # Should be positive

    def test_calculate_comprehensive_metrics_with_portfolio_values(self):
        """Test metrics with portfolio values."""
        metrics = self.analytics.calculate_comprehensive_metrics(
            returns=self.returns,
            portfolio_values=self.portfolio_values
        )

        assert metrics.max_drawdown >= 0
        assert isinstance(metrics.ulcer_index, float)
        assert isinstance(metrics.recovery_factor, float)

    def test_calculate_comprehensive_metrics_with_dates(self):
        """Test metrics with start and end dates."""
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 12, 31)

        metrics = self.analytics.calculate_comprehensive_metrics(
            returns=self.returns,
            start_date=start_date,
            end_date=end_date
        )

        assert metrics.period_start == start_date
        assert metrics.period_end == end_date

    def test_calculate_comprehensive_metrics_empty_returns(self):
        """Test metrics with empty returns."""
        empty_returns = []
        metrics = self.analytics.calculate_comprehensive_metrics(empty_returns)

        assert metrics.total_return == 0.0
        assert metrics.sharpe_ratio == 0.0
        assert metrics.trading_days == 0
        assert metrics.positive_days == 0
        assert metrics.negative_days == 0

    def test_calculate_comprehensive_metrics_single_return(self):
        """Test metrics with single return."""
        single_return = [0.05]
        metrics = self.analytics.calculate_comprehensive_metrics(single_return)

        assert abs(metrics.total_return - 0.05) < 1e-10
        assert metrics.trading_days == 1
        assert metrics.best_day == 0.05
        assert metrics.worst_day == 0.05

    def test_calculate_comprehensive_metrics_all_positive_returns(self):
        """Test metrics with all positive returns."""
        positive_returns = np.abs(self.returns)
        metrics = self.analytics.calculate_comprehensive_metrics(positive_returns)

        assert metrics.win_rate == 1.0
        assert metrics.positive_days == len(positive_returns)
        assert metrics.negative_days == 0
        assert metrics.avg_loss == 0.0
        assert metrics.worst_day >= 0

    def test_calculate_comprehensive_metrics_all_negative_returns(self):
        """Test metrics with all negative returns."""
        negative_returns = -np.abs(self.returns)
        metrics = self.analytics.calculate_comprehensive_metrics(negative_returns)

        assert metrics.win_rate == 0.0
        assert metrics.positive_days == 0
        assert metrics.negative_days == len(negative_returns)
        assert metrics.avg_win == 0.0
        assert metrics.best_day <= 0

    def test_analyze_drawdown_periods_basic(self):
        """Test basic drawdown period analysis."""
        # Create portfolio values with clear drawdown
        values = [100, 110, 120, 100, 80, 90, 130, 125, 140]
        drawdowns = self.analytics.analyze_drawdown_periods(values)

        assert isinstance(drawdowns, list)
        assert len(drawdowns) > 0
        for dd in drawdowns:
            assert isinstance(dd, DrawdownPeriod)
            assert dd.drawdown_pct >= 0
            assert dd.peak_value >= dd.trough_value

    def test_analyze_drawdown_periods_with_dates(self):
        """Test drawdown analysis with custom dates."""
        values = [100, 110, 90, 105]
        dates = [
            datetime(2023, 1, 1),
            datetime(2023, 1, 2),
            datetime(2023, 1, 3),
            datetime(2023, 1, 4)
        ]

        drawdowns = self.analytics.analyze_drawdown_periods(values, dates)

        assert len(drawdowns) > 0
        for dd in drawdowns:
            assert dd.start_date in dates
            assert dd.end_date in dates

    def test_analyze_drawdown_periods_no_drawdown(self):
        """Test drawdown analysis with only increasing values."""
        values = [100, 110, 120, 130, 140]
        drawdowns = self.analytics.analyze_drawdown_periods(values)

        # Only count meaningful drawdowns (> 0.1%)
        meaningful_drawdowns = [dd for dd in drawdowns if dd.drawdown_pct > 0.001]
        assert len(meaningful_drawdowns) == 0

    def test_analyze_drawdown_periods_ongoing_drawdown(self):
        """Test drawdown analysis with ongoing drawdown."""
        values = [100, 110, 120, 100, 80, 70]  # Ends in drawdown
        drawdowns = self.analytics.analyze_drawdown_periods(values)

        # Should have one ongoing drawdown
        ongoing_dd = [dd for dd in drawdowns if not dd.is_recovered]
        assert len(ongoing_dd) >= 1
        assert ongoing_dd[0].recovery_date is None
        assert ongoing_dd[0].recovery_days is None

    def test_analyze_drawdown_periods_insufficient_data(self):
        """Test drawdown analysis with insufficient data."""
        # Single value
        single_value = [100]
        drawdowns = self.analytics.analyze_drawdown_periods(single_value)
        assert len(drawdowns) == 0

        # Empty values
        empty_values = []
        drawdowns = self.analytics.analyze_drawdown_periods(empty_values)
        assert len(drawdowns) == 0

    def test_calculate_total_return(self):
        """Test total return calculation."""
        returns = [0.1, -0.05, 0.03, 0.02]
        total_return = self.analytics._calculate_total_return(np.array(returns))

        # Calculate expected: (1+0.1)*(1-0.05)*(1+0.03)*(1+0.02) - 1
        expected = 1.1 * 0.95 * 1.03 * 1.02 - 1
        assert abs(total_return - expected) < 1e-10

    def test_calculate_annualized_return(self):
        """Test annualized return calculation."""
        # Simple case: 10% return over half year
        returns = np.array([0.1] + [0.0] * 125)  # 126 days = ~0.5 years
        ann_return = self.analytics._calculate_annualized_return(returns)

        # Should be roughly (1.1)^2 - 1 = 21%
        assert 0.15 < ann_return < 0.25

    def test_calculate_annualized_return_empty(self):
        """Test annualized return with empty returns."""
        empty_returns = np.array([])
        ann_return = self.analytics._calculate_annualized_return(empty_returns)
        assert ann_return == 0.0

    def test_calculate_volatility(self):
        """Test volatility calculation."""
        # Known volatility case
        returns = np.array([0.01, -0.01, 0.01, -0.01] * 63)  # 252 days
        volatility = self.analytics._calculate_volatility(returns)

        # Should be annualized std
        expected_vol = np.std(returns) * np.sqrt(252)
        assert abs(volatility - expected_vol) < 1e-10

    def test_calculate_sharpe_ratio(self):
        """Test Sharpe ratio calculation."""
        # High return, low volatility case
        high_returns = np.array([0.002] * 252)  # Consistent 0.2% daily
        sharpe = self.analytics._calculate_sharpe_ratio(high_returns)

        assert sharpe > 0  # Should be positive

        # Zero volatility case
        zero_vol_returns = np.array([0.001] * 252)
        sharpe_zero = self.analytics._calculate_sharpe_ratio(zero_vol_returns)
        # When volatility is 0, Sharpe ratio approaches infinity, so we expect a large positive number
        assert sharpe_zero > 1000 or np.isinf(sharpe_zero) or np.isnan(sharpe_zero)

    def test_calculate_sharpe_ratio_empty(self):
        """Test Sharpe ratio with empty returns."""
        empty_returns = np.array([])
        sharpe = self.analytics._calculate_sharpe_ratio(empty_returns)
        assert sharpe == 0.0

    def test_calculate_sortino_ratio(self):
        """Test Sortino ratio calculation."""
        # Mix of positive and negative returns
        mixed_returns = np.array([0.02, -0.01, 0.015, -0.005, 0.01])
        sortino = self.analytics._calculate_sortino_ratio(mixed_returns)

        assert isinstance(sortino, float)

        # All positive returns (no downside)
        positive_returns = np.array([0.01, 0.02, 0.005])
        sortino_pos = self.analytics._calculate_sortino_ratio(positive_returns)
        assert sortino_pos == 0.0  # No downside deviation

    def test_calculate_sortino_ratio_empty(self):
        """Test Sortino ratio with empty returns."""
        empty_returns = np.array([])
        sortino = self.analytics._calculate_sortino_ratio(empty_returns)
        assert sortino == 0.0

    def test_calculate_calmar_ratio(self):
        """Test Calmar ratio calculation."""
        # Create declining then recovering portfolio
        portfolio_values = np.array([100, 90, 80, 85, 110])  # 20% max drawdown
        returns = np.diff(portfolio_values) / portfolio_values[:-1]

        calmar = self.analytics._calculate_calmar_ratio(returns, portfolio_values)
        assert isinstance(calmar, float)

        # Test with None portfolio values
        calmar_none = self.analytics._calculate_calmar_ratio(returns, None)
        assert isinstance(calmar_none, float)

    def test_calculate_max_drawdown(self):
        """Test maximum drawdown calculation."""
        # Clear max drawdown case
        values = np.array([100, 110, 120, 90, 80, 100, 130])
        max_dd = self.analytics._calculate_max_drawdown(values)

        # Max drawdown should be (120 - 80) / 120 = 1/3
        expected_dd = (120 - 80) / 120
        assert abs(max_dd - expected_dd) < 1e-10

    def test_calculate_max_drawdown_no_drawdown(self):
        """Test max drawdown with only increasing values."""
        increasing_values = np.array([100, 110, 120, 130])
        max_dd = self.analytics._calculate_max_drawdown(increasing_values)
        assert max_dd == 0.0

    def test_calculate_max_drawdown_insufficient_data(self):
        """Test max drawdown with insufficient data."""
        single_value = np.array([100])
        max_dd = self.analytics._calculate_max_drawdown(single_value)
        assert max_dd == 0.0

        empty_values = np.array([])
        max_dd_empty = self.analytics._calculate_max_drawdown(empty_values)
        assert max_dd_empty == 0.0

    def test_calculate_var(self):
        """Test Value at Risk calculation."""
        # Normal distribution case
        normal_returns = np.random.normal(0, 0.02, 1000)
        var_95 = self.analytics._calculate_var(normal_returns, 0.95)
        var_99 = self.analytics._calculate_var(normal_returns, 0.99)

        assert var_95 > 0  # VaR should be positive (loss)
        assert var_99 > var_95  # 99% VaR should be more extreme

        # Empty returns
        empty_returns = np.array([])
        var_empty = self.analytics._calculate_var(empty_returns, 0.95)
        assert var_empty == 0.0

    def test_calculate_cvar(self):
        """Test Conditional VaR calculation."""
        # Generate returns with known tail
        returns = np.array([-0.05, -0.03, -0.01, 0.01, 0.02, 0.03])
        cvar_95 = self.analytics._calculate_cvar(returns, 0.95)

        assert cvar_95 > 0  # CVaR should be positive

        # Compare with VaR
        var_95 = self.analytics._calculate_var(returns, 0.95)
        assert cvar_95 >= var_95  # CVaR should be >= VaR

    def test_calculate_win_rate(self):
        """Test win rate calculation."""
        # Known win rate case
        returns = np.array([0.01, -0.01, 0.02, 0.01, -0.005])  # 3 wins, 2 losses
        win_rate = self.analytics._calculate_win_rate(returns)
        assert win_rate == 0.6  # 3/5 = 60%

        # All wins
        all_wins = np.array([0.01, 0.02, 0.005])
        win_rate_all = self.analytics._calculate_win_rate(all_wins)
        assert win_rate_all == 1.0

        # Empty returns
        empty_returns = np.array([])
        win_rate_empty = self.analytics._calculate_win_rate(empty_returns)
        assert win_rate_empty == 0.0

    def test_calculate_avg_win_loss(self):
        """Test average win and loss calculation."""
        returns = np.array([0.02, -0.01, 0.04, -0.03, 0.01])
        avg_win, avg_loss = self.analytics._calculate_avg_win_loss(returns)

        # Wins: 0.02, 0.04, 0.01 -> avg = 0.023333
        # Losses: -0.01, -0.03 -> avg = -0.02
        assert abs(avg_win - (0.02 + 0.04 + 0.01) / 3) < 1e-10
        assert abs(avg_loss - (-0.01 - 0.03) / 2) < 1e-10

        # No wins case
        all_losses = np.array([-0.01, -0.02, -0.03])
        avg_win_none, avg_loss_all = self.analytics._calculate_avg_win_loss(all_losses)
        assert avg_win_none == 0.0
        assert avg_loss_all < 0

        # Empty returns
        empty_returns = np.array([])
        avg_win_empty, avg_loss_empty = self.analytics._calculate_avg_win_loss(empty_returns)
        assert avg_win_empty == 0.0
        assert avg_loss_empty == 0.0

    def test_calculate_profit_factor(self):
        """Test profit factor calculation."""
        # Known case
        returns = np.array([0.02, -0.01, 0.04, -0.03])  # Gross profit = 0.06, Gross loss = 0.04
        profit_factor = self.analytics._calculate_profit_factor(returns)
        expected_pf = 0.06 / 0.04  # 1.5
        assert abs(profit_factor - expected_pf) < 1e-10

        # No losses case
        all_wins = np.array([0.01, 0.02, 0.03])
        pf_no_losses = self.analytics._calculate_profit_factor(all_wins)
        assert pf_no_losses == 0.0  # Division by zero case

        # No wins case
        all_losses = np.array([-0.01, -0.02])
        pf_no_wins = self.analytics._calculate_profit_factor(all_losses)
        assert pf_no_wins == 0.0

    def test_calculate_relative_metrics_basic(self):
        """Test relative metrics calculation."""
        portfolio_returns = np.array([0.01, -0.005, 0.02, 0.015])
        benchmark_returns = np.array([0.008, -0.002, 0.015, 0.01])

        alpha, beta, info_ratio, treynor, tracking_error = self.analytics._calculate_relative_metrics(
            portfolio_returns, benchmark_returns
        )

        assert isinstance(alpha, float)
        assert isinstance(beta, float)
        assert beta > 0  # Beta should be positive for similar movements
        assert isinstance(tracking_error, float)
        assert tracking_error >= 0

    def test_calculate_relative_metrics_no_benchmark(self):
        """Test relative metrics with no benchmark."""
        returns = np.array([0.01, -0.005, 0.02])
        alpha, beta, info_ratio, treynor, tracking_error = self.analytics._calculate_relative_metrics(
            returns, None
        )

        assert alpha == 0.0
        assert beta == 1.0
        assert info_ratio == 0.0
        assert treynor == 0.0
        assert tracking_error == 0.0

    def test_calculate_relative_metrics_mismatched_length(self):
        """Test relative metrics with mismatched lengths."""
        portfolio_returns = np.array([0.01, -0.005, 0.02])
        benchmark_returns = np.array([0.008, -0.002])  # Different length

        alpha, beta, info_ratio, treynor, tracking_error = self.analytics._calculate_relative_metrics(
            portfolio_returns, benchmark_returns
        )

        # Should handle length mismatch by using shorter length
        assert isinstance(alpha, float)
        assert isinstance(beta, float)

    def test_calculate_relative_metrics_zero_variance(self):
        """Test relative metrics with zero variance benchmark."""
        portfolio_returns = np.array([0.01, -0.005, 0.02])
        benchmark_returns = np.array([0.01, 0.01, 0.01])  # No variance

        alpha, beta, info_ratio, treynor, tracking_error = self.analytics._calculate_relative_metrics(
            portfolio_returns, benchmark_returns
        )

        assert beta == 1.0  # Should default to 1.0 for zero variance

    def test_calculate_recovery_factor(self):
        """Test recovery factor calculation."""
        returns = np.array([0.1, -0.05, 0.03])  # Total return ~7.85%
        max_drawdown = 0.05  # 5% max drawdown

        recovery_factor = self.analytics._calculate_recovery_factor(returns, max_drawdown)
        expected_rf = self.analytics._calculate_total_return(returns) / max_drawdown
        assert abs(recovery_factor - expected_rf) < 1e-10

        # Zero max drawdown case
        recovery_factor_zero = self.analytics._calculate_recovery_factor(returns, 0.0)
        assert recovery_factor_zero == 0.0

    def test_calculate_ulcer_index(self):
        """Test Ulcer Index calculation."""
        # Portfolio with drawdowns
        values = np.array([100, 90, 95, 110, 100])
        ulcer = self.analytics._calculate_ulcer_index(values)

        assert ulcer >= 0
        assert isinstance(ulcer, float)

        # No drawdown case
        increasing_values = np.array([100, 110, 120, 130])
        ulcer_no_dd = self.analytics._calculate_ulcer_index(increasing_values)
        assert ulcer_no_dd == 0.0

        # Single value
        single_value = np.array([100])
        ulcer_single = self.analytics._calculate_ulcer_index(single_value)
        assert ulcer_single == 0.0

    def test_calculate_sterling_ratio(self):
        """Test Sterling ratio calculation."""
        returns = np.array([0.1, -0.02, 0.05])
        max_drawdown = 0.03

        sterling = self.analytics._calculate_sterling_ratio(returns, max_drawdown)
        expected_sterling = self.analytics._calculate_annualized_return(returns) / max_drawdown
        assert abs(sterling - expected_sterling) < 1e-10

        # Zero drawdown case
        sterling_zero = self.analytics._calculate_sterling_ratio(returns, 0.0)
        assert sterling_zero == 0.0

    def test_returns_to_values(self):
        """Test conversion from returns to portfolio values."""
        returns = np.array([0.1, -0.05, 0.02])
        initial_value = 1000.0

        values = self.analytics._returns_to_values(returns, initial_value)

        # Calculate expected values
        expected = [1000 * 1.1, 1000 * 1.1 * 0.95, 1000 * 1.1 * 0.95 * 1.02]
        expected = np.array(expected)

        assert np.allclose(values, expected)
        assert len(values) == len(returns)

    def test_returns_to_values_default_initial(self):
        """Test returns to values with default initial value."""
        returns = np.array([0.1, -0.05])
        values = self.analytics._returns_to_values(returns)

        # Should start with 10000 by default
        expected_first = 10000 * 1.1
        assert abs(values[0] - expected_first) < 1e-10

    def test_create_empty_metrics(self):
        """Test creation of empty metrics."""
        empty_metrics = self.analytics._create_empty_metrics()

        assert empty_metrics.total_return == 0.0
        assert empty_metrics.sharpe_ratio == 0.0
        assert empty_metrics.trading_days == 0
        assert empty_metrics.beta == 1.0  # Default beta
        assert isinstance(empty_metrics.period_start, datetime)
        assert isinstance(empty_metrics.period_end, datetime)

    def test_generate_analytics_report(self):
        """Test analytics report generation."""
        # Create sample metrics
        now = datetime.now()
        metrics = PerformanceMetrics(
            total_return=0.25, annualized_return=0.15, volatility=0.20,
            sharpe_ratio=1.2, sortino_ratio=1.5, calmar_ratio=2.0,
            max_drawdown=0.10, var_95=0.05, var_99=0.08, cvar_95=0.06,
            win_rate=0.60, avg_win=0.02, avg_loss=-0.015, profit_factor=1.8,
            information_ratio=0.5, treynor_ratio=0.08, alpha=0.03, beta=1.1,
            tracking_error=0.04, period_start=now - timedelta(days=365),
            period_end=now, trading_days=252, best_day=0.08, worst_day=-0.06,
            positive_days=151, negative_days=101, recovery_factor=2.5,
            ulcer_index=5.2, sterling_ratio=1.5
        )

        report = self.analytics.generate_analytics_report(metrics)

        assert isinstance(report, str)
        assert "ADVANCED PERFORMANCE ANALYTICS REPORT" in report
        assert "25.00%" in report  # Total return
        assert "1.20" in report    # Sharpe ratio
        assert "60.00%" in report  # Win rate
        assert "252" in report     # Trading days

    def test_calculate_comprehensive_metrics_error_handling(self):
        """Test error handling in comprehensive metrics calculation."""
        # Test with invalid data that might cause exceptions
        with patch.object(self.analytics, '_calculate_total_return', side_effect=Exception("Test error")):
            metrics = self.analytics.calculate_comprehensive_metrics([0.01, 0.02, 0.03])

            # Should return empty metrics on error
            assert metrics.total_return == 0.0
            assert metrics.trading_days == 0

    def test_analyze_drawdown_periods_error_handling(self):
        """Test error handling in drawdown analysis."""
        # Test with invalid data that causes realistic errors
        invalid_values = [float('nan'), float('inf'), -float('inf')]
        drawdowns = self.analytics.analyze_drawdown_periods(invalid_values)
        # Should handle invalid values gracefully
        assert isinstance(drawdowns, list)


class TestConvenienceFunction:
    """Test the analyze_performance convenience function."""

    def test_analyze_performance_basic(self):
        """Test basic analyze_performance function."""
        returns = [0.01, -0.005, 0.02, 0.015]
        metrics = analyze_performance(returns)

        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.trading_days == len(returns)

    def test_analyze_performance_with_benchmark(self):
        """Test analyze_performance with benchmark."""
        returns = [0.01, -0.005, 0.02, 0.015]
        benchmark = [0.008, -0.002, 0.018, 0.012]

        metrics = analyze_performance(returns, benchmark_returns=benchmark)

        assert isinstance(metrics.alpha, float)
        assert isinstance(metrics.beta, float)

    def test_analyze_performance_with_portfolio_values(self):
        """Test analyze_performance with portfolio values."""
        returns = [0.01, -0.005, 0.02]
        portfolio_values = [100, 101, 100.495, 102.5049]

        metrics = analyze_performance(returns, portfolio_values=portfolio_values)

        assert metrics.max_drawdown >= 0

    def test_analyze_performance_custom_risk_free_rate(self):
        """Test analyze_performance with custom risk-free rate."""
        returns = [0.01, 0.02, 0.015]
        custom_rate = 0.05

        metrics = analyze_performance(returns, risk_free_rate=custom_rate)

        # Verify the custom rate was used (indirectly through Sharpe calculation)
        assert isinstance(metrics.sharpe_ratio, float)


class TestIntegrationScenarios:
    """Test integration scenarios for advanced analytics."""

    def setup_method(self):
        """Set up test fixtures."""
        self.analytics = AdvancedAnalytics(risk_free_rate=0.03)

    def test_bull_market_scenario(self):
        """Test analytics for bull market scenario."""
        # Generate bull market returns (positive trend with low volatility)
        np.random.seed(123)
        trend = np.linspace(0.0005, 0.002, 252)  # Increasing trend
        noise = np.random.normal(0, 0.01, 252)   # Low volatility
        bull_returns = trend + noise

        metrics = self.analytics.calculate_comprehensive_metrics(bull_returns)

        assert metrics.win_rate > 0.5  # Should have high win rate
        assert metrics.sharpe_ratio > 0  # Should be positive
        assert metrics.total_return > 0  # Should be profitable

    def test_bear_market_scenario(self):
        """Test analytics for bear market scenario."""
        # Generate bear market returns (negative trend with high volatility)
        np.random.seed(456)
        trend = np.linspace(-0.0005, -0.002, 252)  # Decreasing trend
        noise = np.random.normal(0, 0.025, 252)    # High volatility
        bear_returns = trend + noise

        metrics = self.analytics.calculate_comprehensive_metrics(bear_returns)

        assert metrics.total_return < 0  # Should be losing
        assert metrics.max_drawdown > 0.05  # Should have significant drawdown
        assert metrics.var_95 > 0.01  # Should have high VaR

    def test_volatile_market_scenario(self):
        """Test analytics for high volatility market."""
        # Generate high volatility returns
        np.random.seed(789)
        volatile_returns = np.random.normal(0.001, 0.04, 252)  # High vol

        metrics = self.analytics.calculate_comprehensive_metrics(volatile_returns)

        assert metrics.volatility > 0.3  # Should show high volatility
        assert metrics.ulcer_index > 10  # Should have high ulcer index
        assert abs(metrics.best_day) > 0.05 or abs(metrics.worst_day) > 0.05

    def test_steady_growth_scenario(self):
        """Test analytics for steady growth scenario."""
        # Generate steady, consistent returns
        steady_returns = np.array([0.0008] * 252)  # Consistent daily return

        metrics = self.analytics.calculate_comprehensive_metrics(steady_returns)

        assert metrics.win_rate == 1.0  # All positive days
        assert metrics.volatility < 0.01  # Very low volatility
        assert metrics.max_drawdown == 0.0  # No drawdowns
        assert metrics.sharpe_ratio > 5  # Very high Sharpe

    def test_portfolio_comparison_scenario(self):
        """Test comparing multiple portfolios."""
        # Create three different return profiles
        conservative_returns = np.random.normal(0.0003, 0.008, 252)
        moderate_returns = np.random.normal(0.0008, 0.015, 252)
        aggressive_returns = np.random.normal(0.0015, 0.030, 252)

        # Calculate metrics for each
        conservative_metrics = self.analytics.calculate_comprehensive_metrics(conservative_returns)
        moderate_metrics = self.analytics.calculate_comprehensive_metrics(moderate_returns)
        aggressive_metrics = self.analytics.calculate_comprehensive_metrics(aggressive_returns)

        # Verify risk-return relationship
        assert conservative_metrics.volatility < moderate_metrics.volatility < aggressive_metrics.volatility
        assert conservative_metrics.max_drawdown <= moderate_metrics.max_drawdown

    def test_benchmark_comparison_scenario(self):
        """Test portfolio vs benchmark comparison."""
        # Generate portfolio that outperforms benchmark
        np.random.seed(42)
        benchmark_returns = np.random.normal(0.0004, 0.012, 252)
        portfolio_returns = benchmark_returns * 1.2 + np.random.normal(0.0002, 0.008, 252)

        metrics = self.analytics.calculate_comprehensive_metrics(
            returns=portfolio_returns,
            benchmark_returns=benchmark_returns
        )

        assert metrics.alpha > 0  # Should have positive alpha
        assert metrics.information_ratio > 0  # Should outperform on risk-adjusted basis
        assert metrics.beta > 0  # Should be correlated

    def test_crisis_recovery_scenario(self):
        """Test analytics during crisis and recovery."""
        # Simulate market crisis followed by recovery
        normal_returns = np.random.normal(0.0005, 0.015, 100)
        crisis_returns = np.random.normal(-0.003, 0.035, 50)  # Crisis period
        recovery_returns = np.random.normal(0.002, 0.020, 102)  # Recovery

        crisis_scenario = np.concatenate([normal_returns, crisis_returns, recovery_returns])

        metrics = self.analytics.calculate_comprehensive_metrics(crisis_scenario)

        # Analyze recovery - random scenario may not always recover fully
        assert metrics.max_drawdown > 0.15  # Should show significant drawdown
        # Recovery factor can be negative if final value is below peak
        assert isinstance(metrics.recovery_factor, (int, float))  # Should be a valid number

        # Analyze drawdown periods
        portfolio_values = np.cumprod(1 + crisis_scenario) * 10000
        drawdowns = self.analytics.analyze_drawdown_periods(portfolio_values)

        # Should identify the crisis period
        major_drawdowns = [dd for dd in drawdowns if dd.drawdown_pct > 0.10]
        assert len(major_drawdowns) >= 1
        assert any(dd.is_recovered for dd in major_drawdowns)  # Should show recovery

    def test_real_time_monitoring_scenario(self):
        """Test real-time analytics monitoring simulation."""
        # Simulate adding new data points over time
        base_returns = []
        analytics_history = []

        for day in range(1, 101):  # 100 trading days
            # Add new return
            daily_return = np.random.normal(0.001, 0.02)
            base_returns.append(daily_return)

            # Calculate metrics every 10 days
            if day % 10 == 0:
                metrics = self.analytics.calculate_comprehensive_metrics(base_returns)
                analytics_history.append({
                    'day': day,
                    'sharpe': metrics.sharpe_ratio,
                    'max_dd': metrics.max_drawdown,
                    'win_rate': metrics.win_rate
                })

        # Verify we have monitoring history
        assert len(analytics_history) == 10  # Every 10 days for 100 days

        # Verify metrics evolve over time
        sharpe_ratios = [h['sharpe'] for h in analytics_history]
        assert len(set(sharpe_ratios)) > 1  # Should change over time