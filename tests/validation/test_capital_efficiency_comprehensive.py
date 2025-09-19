"""
Comprehensive Tests for Capital Efficiency Analyzer
==================================================

Enhanced test coverage for Kelly sizing calculations, leverage analysis,
and capital allocation edge cases.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, MagicMock
from typing import Dict, Any, List

from backend.validation.capital_efficiency import (
    CapitalEfficiencyAnalyzer,
    KellyResult,
    _max_dd,
    _sharpe
)


class TestKellyResult:
    """Test KellyResult dataclass functionality."""

    def test_kelly_result_creation(self):
        """Test basic KellyResult creation."""
        result = KellyResult(
            kelly_fraction=0.25,
            conservative_kelly=0.125,
            recommended_position_size=0.125,
            win_rate=0.65,
            win_loss_ratio=1.8,
            expected_return=0.05
        )
        assert result.kelly_fraction == 0.25
        assert result.conservative_kelly == 0.125
        assert result.recommended_position_size == 0.125
        assert result.win_rate == 0.65
        assert result.win_loss_ratio == 1.8
        assert result.expected_return == 0.05

    def test_kelly_result_zero_values(self):
        """Test KellyResult with zero values."""
        result = KellyResult(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        assert all(value == 0.0 for value in [
            result.kelly_fraction,
            result.conservative_kelly,
            result.recommended_position_size,
            result.win_rate,
            result.win_loss_ratio,
            result.expected_return
        ])

    def test_kelly_result_negative_values(self):
        """Test KellyResult with negative values."""
        result = KellyResult(-0.1, -0.05, 0.0, 0.4, 0.8, -0.02)
        assert result.kelly_fraction == -0.1
        assert result.conservative_kelly == -0.05
        assert result.recommended_position_size == 0.0
        assert result.expected_return == -0.02


class TestUtilityFunctions:
    """Test utility functions for metrics calculation."""

    def test_max_dd_simple_case(self):
        """Test maximum drawdown with simple case."""
        returns = pd.Series([0.10, -0.05, 0.02, -0.15, 0.08])
        max_dd = _max_dd(returns)

        # Calculate expected: equity curve and max drawdown
        equity = (1 + returns).cumprod()
        expected_dd = (equity / equity.cummax() - 1).min()

        assert abs(max_dd - expected_dd) < 1e-10

    def test_max_dd_no_drawdown(self):
        """Test maximum drawdown with no drawdown."""
        returns = pd.Series([0.01, 0.02, 0.015, 0.03])
        max_dd = _max_dd(returns)
        assert max_dd == 0.0

    def test_max_dd_empty_series(self):
        """Test maximum drawdown with empty series."""
        returns = pd.Series([], dtype=float)
        max_dd = _max_dd(returns)
        assert max_dd == 0.0

    def test_max_dd_with_nans(self):
        """Test maximum drawdown with NaN values."""
        returns = pd.Series([0.10, np.nan, -0.05, np.nan, 0.02])
        max_dd = _max_dd(returns)
        # Should handle NaN values by dropping them
        assert isinstance(max_dd, float)

    def test_sharpe_simple_case(self):
        """Test Sharpe ratio calculation."""
        returns = pd.Series([0.01, 0.02, -0.005, 0.015, 0.008])
        sharpe = _sharpe(returns)

        # Calculate expected Sharpe
        expected = np.sqrt(252) * returns.mean() / returns.std(ddof=1)
        assert abs(sharpe - expected) < 1e-10

    def test_sharpe_zero_std(self):
        """Test Sharpe ratio with zero standard deviation."""
        returns = pd.Series([0.01, 0.01, 0.01, 0.01])
        sharpe = _sharpe(returns)
        assert sharpe == 0.0

    def test_sharpe_insufficient_data(self):
        """Test Sharpe ratio with insufficient data."""
        returns = pd.Series([0.01])
        sharpe = _sharpe(returns)
        assert sharpe == 0.0

    def test_sharpe_empty_series(self):
        """Test Sharpe ratio with empty series."""
        returns = pd.Series([], dtype=float)
        sharpe = _sharpe(returns)
        assert sharpe == 0.0

    def test_sharpe_with_nans(self):
        """Test Sharpe ratio with NaN values."""
        returns = pd.Series([0.01, np.nan, 0.02, np.nan, -0.005])
        sharpe = _sharpe(returns)
        # Should handle NaN values by dropping them
        assert isinstance(sharpe, float)


class TestCapitalEfficiencyAnalyzerInitialization:
    """Test CapitalEfficiencyAnalyzer initialization."""

    def test_default_initialization(self):
        """Test default initialization."""
        analyzer = CapitalEfficiencyAnalyzer()
        assert analyzer.daily_margin_rate == 0.00008

    def test_custom_margin_rate(self):
        """Test custom margin rate."""
        analyzer = CapitalEfficiencyAnalyzer(daily_margin_rate=0.0001)
        assert analyzer.daily_margin_rate == 0.0001

    def test_zero_margin_rate(self):
        """Test zero margin rate."""
        analyzer = CapitalEfficiencyAnalyzer(daily_margin_rate=0.0)
        assert analyzer.daily_margin_rate == 0.0

    def test_negative_margin_rate(self):
        """Test negative margin rate (edge case)."""
        analyzer = CapitalEfficiencyAnalyzer(daily_margin_rate=-0.0001)
        assert analyzer.daily_margin_rate == -0.0001


class TestKellySizingAnalysis:
    """Test Kelly sizing analysis functionality."""

    def test_kelly_sizing_perfect_strategy(self):
        """Test Kelly sizing for perfect winning strategy."""
        analyzer = CapitalEfficiencyAnalyzer()
        returns = pd.Series([0.05, 0.03, 0.08, 0.02, 0.06])  # All wins

        result = analyzer.kelly_sizing_analysis(returns)

        assert result.win_rate == 1.0
        assert result.kelly_fraction == 1.0  # Should recommend 100% for perfect strategy
        assert result.conservative_kelly == 0.5
        assert result.expected_return > 0

    def test_kelly_sizing_losing_strategy(self):
        """Test Kelly sizing for losing strategy."""
        analyzer = CapitalEfficiencyAnalyzer()
        returns = pd.Series([-0.05, -0.03, -0.08, -0.02, -0.06])  # All losses

        result = analyzer.kelly_sizing_analysis(returns)

        assert result.win_rate == 0.0
        assert result.kelly_fraction == 0.0
        assert result.conservative_kelly == 0.0
        assert result.recommended_position_size == 0.0
        assert result.expected_return < 0

    def test_kelly_sizing_mixed_returns(self):
        """Test Kelly sizing with mixed returns."""
        analyzer = CapitalEfficiencyAnalyzer()
        returns = pd.Series([0.10, -0.05, 0.08, -0.03, 0.06, -0.04])

        result = analyzer.kelly_sizing_analysis(returns)

        assert 0.0 < result.win_rate < 1.0
        assert result.kelly_fraction >= 0.0
        assert result.conservative_kelly == 0.5 * result.kelly_fraction
        assert result.win_loss_ratio > 0

    def test_kelly_sizing_empty_returns(self):
        """Test Kelly sizing with empty returns."""
        analyzer = CapitalEfficiencyAnalyzer()
        returns = pd.Series([], dtype=float)

        result = analyzer.kelly_sizing_analysis(returns)

        assert result.kelly_fraction == 0.0
        assert result.conservative_kelly == 0.0
        assert result.recommended_position_size == 0.0
        assert result.win_rate == 0.0
        assert result.expected_return == 0.0

    def test_kelly_sizing_with_nans(self):
        """Test Kelly sizing with NaN values."""
        analyzer = CapitalEfficiencyAnalyzer()
        returns = pd.Series([0.05, np.nan, -0.03, np.nan, 0.02])

        result = analyzer.kelly_sizing_analysis(returns)

        # Should handle NaN values by dropping them
        assert isinstance(result.kelly_fraction, float)
        assert isinstance(result.win_rate, float)

    def test_kelly_sizing_zero_average_loss(self):
        """Test Kelly sizing when average loss is zero."""
        analyzer = CapitalEfficiencyAnalyzer()
        returns = pd.Series([0.05, 0.03, 0.0, 0.02, 0.06])  # No negative returns

        result = analyzer.kelly_sizing_analysis(returns)

        # When no losses, should return conservative values
        assert result.kelly_fraction == 0.0
        assert result.conservative_kelly == 0.0

    def test_kelly_sizing_custom_cap(self):
        """Test Kelly sizing with custom cap."""
        analyzer = CapitalEfficiencyAnalyzer()
        returns = pd.Series([0.20, 0.15, 0.25, 0.18, 0.22])  # High returns

        result = analyzer.kelly_sizing_analysis(returns, cap_at=0.10)

        # Should cap at 0.10 even if Kelly suggests higher
        assert result.recommended_position_size <= 0.10

    def test_kelly_sizing_boundary_conditions(self):
        """Test Kelly sizing boundary conditions."""
        analyzer = CapitalEfficiencyAnalyzer()

        # Very small positive returns
        small_returns = pd.Series([0.001, 0.0005, -0.0002, 0.0008])
        result = analyzer.kelly_sizing_analysis(small_returns)
        assert result.kelly_fraction >= 0.0

        # Very large returns
        large_returns = pd.Series([1.0, -0.5, 0.8, -0.3])
        result = analyzer.kelly_sizing_analysis(large_returns)
        assert result.kelly_fraction <= 1.0


class TestLeverageEfficiencyAnalysis:
    """Test leverage efficiency analysis functionality."""

    def test_analyze_leverage_efficiency_basic(self):
        """Test basic leverage efficiency analysis."""
        analyzer = CapitalEfficiencyAnalyzer()

        # Mock strategy
        mock_strategy = Mock()
        mock_backtest = Mock()
        mock_backtest.returns = pd.Series([0.01, 0.02, -0.005, 0.015])
        mock_backtest.margin_calls = 0
        mock_strategy.backtest_with_capital.return_value = mock_backtest

        capital_levels = [100000, 200000]

        result = analyzer.analyze_leverage_efficiency(mock_strategy, capital_levels)

        assert 'detailed_results' in result
        assert 'optimal_setups' in result
        assert len(result['optimal_setups']) <= len(capital_levels)

    def test_analyze_leverage_efficiency_with_failures(self):
        """Test leverage analysis with some failures."""
        analyzer = CapitalEfficiencyAnalyzer()

        # Mock strategy that fails for certain conditions
        mock_strategy = Mock()

        def mock_backtest_func(capital):
            if capital > 150000:
                raise Exception("Insufficient margin")
            mock_bt = Mock()
            mock_bt.returns = pd.Series([0.01, 0.02, -0.005])
            mock_bt.margin_calls = 0
            return mock_bt

        mock_strategy.backtest_with_capital.side_effect = mock_backtest_func

        capital_levels = [100000, 200000]

        result = analyzer.analyze_leverage_efficiency(mock_strategy, capital_levels)

        # Should handle failures gracefully
        assert isinstance(result, dict)
        assert 'detailed_results' in result

    def test_analyze_leverage_efficiency_empty_capital_levels(self):
        """Test leverage analysis with empty capital levels."""
        analyzer = CapitalEfficiencyAnalyzer()
        mock_strategy = Mock()

        result = analyzer.analyze_leverage_efficiency(mock_strategy, [])

        assert result['detailed_results'] == {}
        assert result['optimal_setups'] == {}

    def test_analyze_leverage_efficiency_margin_cost_calculation(self):
        """Test margin cost calculation in leverage analysis."""
        analyzer = CapitalEfficiencyAnalyzer(daily_margin_rate=0.0001)

        mock_strategy = Mock()
        mock_backtest = Mock()
        mock_backtest.returns = pd.Series([0.02, 0.01, 0.015])
        mock_backtest.margin_calls = 0
        mock_strategy.backtest_with_capital.return_value = mock_backtest

        result = analyzer.analyze_leverage_efficiency(mock_strategy, [100000])

        # Verify margin costs are applied differently for different leverage levels
        detailed = result['detailed_results']

        # Should have results for different leverage levels
        leverage_levels = [key[1] for key in detailed.keys()]
        assert 1.0 in leverage_levels  # No leverage
        assert any(lev > 1.0 for lev in leverage_levels)  # Some leverage


class TestCapitalAllocationAnalysis:
    """Test capital allocation analysis functionality."""

    def test_analyze_capital_allocation_basic(self):
        """Test basic capital allocation analysis."""
        analyzer = CapitalEfficiencyAnalyzer()

        strategies = {
            'strategy_a': {
                'trade_returns': pd.Series([0.05, -0.02, 0.03, 0.01])
            },
            'strategy_b': {
                'trade_returns': pd.Series([0.08, -0.04, 0.06, -0.01])
            }
        }

        result = analyzer.analyze_capital_allocation(strategies, 1000000)

        assert 'strategy_allocations' in result
        assert 'total_capital' in result
        assert 'total_allocated' in result
        assert 'capital_utilization' in result

        assert result['total_capital'] == 1000000
        assert len(result['strategy_allocations']) == 2

        # Check each strategy allocation
        for allocation in result['strategy_allocations'].values():
            assert 'kelly_result' in allocation
            assert 'suggested_allocation' in allocation
            assert 'allocation_percentage' in allocation
            assert 'risk_adjusted_return' in allocation

    def test_analyze_capital_allocation_exceeds_capital(self):
        """Test capital allocation when suggested allocations exceed total capital."""
        analyzer = CapitalEfficiencyAnalyzer()

        # Create strategies that would suggest high allocations
        strategies = {
            'high_return_a': {
                'trade_returns': pd.Series([0.10, 0.08, 0.12, 0.09])  # All wins
            },
            'high_return_b': {
                'trade_returns': pd.Series([0.15, 0.12, 0.18, 0.14])  # All wins
            }
        }

        result = analyzer.analyze_capital_allocation(strategies, 100000)

        # Total allocated should not exceed total capital
        assert result['total_allocated'] <= result['total_capital']
        assert result['capital_utilization'] <= 1.0

    def test_analyze_capital_allocation_empty_strategies(self):
        """Test capital allocation with empty strategies."""
        analyzer = CapitalEfficiencyAnalyzer()

        result = analyzer.analyze_capital_allocation({}, 1000000)

        assert result['strategy_allocations'] == {}
        assert result['total_capital'] == 1000000
        assert result['total_allocated'] == 0
        assert result['capital_utilization'] == 0.0

    def test_analyze_capital_allocation_missing_returns(self):
        """Test capital allocation with missing trade_returns."""
        analyzer = CapitalEfficiencyAnalyzer()

        strategies = {
            'strategy_a': {
                'trade_returns': pd.Series([0.05, -0.02, 0.03])
            },
            'strategy_b': {
                'other_data': 'no_returns'  # Missing trade_returns
            }
        }

        result = analyzer.analyze_capital_allocation(strategies, 1000000)

        # Should only process strategies with trade_returns
        assert len(result['strategy_allocations']) == 1
        assert 'strategy_a' in result['strategy_allocations']
        assert 'strategy_b' not in result['strategy_allocations']

    def test_analyze_capital_allocation_zero_capital(self):
        """Test capital allocation with zero capital."""
        analyzer = CapitalEfficiencyAnalyzer()

        strategies = {
            'strategy_a': {
                'trade_returns': pd.Series([0.05, -0.02, 0.03])
            }
        }

        result = analyzer.analyze_capital_allocation(strategies, 0)

        assert result['total_capital'] == 0
        assert result['total_allocated'] == 0

        # Allocation percentages should still be calculated
        for allocation in result['strategy_allocations'].values():
            assert allocation['suggested_allocation'] == 0

    def test_analyze_capital_allocation_risk_adjusted_return(self):
        """Test risk-adjusted return calculation in capital allocation."""
        analyzer = CapitalEfficiencyAnalyzer()

        strategies = {
            'low_risk': {
                'trade_returns': pd.Series([0.02, 0.01, 0.02, 0.01])  # Low variance
            },
            'high_risk': {
                'trade_returns': pd.Series([0.10, -0.05, 0.08, -0.02])  # High variance
            }
        }

        result = analyzer.analyze_capital_allocation(strategies, 1000000)

        # Both should have risk-adjusted returns calculated
        for allocation in result['strategy_allocations'].values():
            assert 'risk_adjusted_return' in allocation
            assert isinstance(allocation['risk_adjusted_return'], (int, float))


class TestCapitalEfficiencyEdgeCases:
    """Test edge cases and error conditions."""

    def test_kelly_with_extreme_values(self):
        """Test Kelly calculation with extreme values."""
        analyzer = CapitalEfficiencyAnalyzer()

        # Extreme positive returns
        extreme_positive = pd.Series([10.0, 5.0, 15.0, 8.0])
        result = analyzer.kelly_sizing_analysis(extreme_positive)
        assert 0.0 <= result.kelly_fraction <= 1.0

        # Extreme negative returns
        extreme_negative = pd.Series([-0.99, -0.95, -0.98, -0.97])
        result = analyzer.kelly_sizing_analysis(extreme_negative)
        assert result.kelly_fraction == 0.0

    def test_kelly_with_single_value(self):
        """Test Kelly calculation with single return value."""
        analyzer = CapitalEfficiencyAnalyzer()

        single_win = pd.Series([0.05])
        result = analyzer.kelly_sizing_analysis(single_win)
        assert result.kelly_fraction == 1.0  # Single win is treated as perfect strategy

        single_loss = pd.Series([-0.05])
        result = analyzer.kelly_sizing_analysis(single_loss)
        assert result.kelly_fraction == 0.0

    def test_leverage_analysis_with_infinite_returns(self):
        """Test leverage analysis with infinite returns."""
        analyzer = CapitalEfficiencyAnalyzer()

        mock_strategy = Mock()
        mock_backtest = Mock()
        mock_backtest.returns = pd.Series([float('inf'), 0.01, 0.02])
        mock_backtest.margin_calls = 0
        mock_strategy.backtest_with_capital.return_value = mock_backtest

        # Should handle infinite values gracefully
        result = analyzer.analyze_leverage_efficiency(mock_strategy, [100000])
        assert isinstance(result, dict)

    def test_capital_allocation_with_zero_position_size(self):
        """Test capital allocation when recommended position size is zero."""
        analyzer = CapitalEfficiencyAnalyzer()

        strategies = {
            'losing_strategy': {
                'trade_returns': pd.Series([-0.05, -0.03, -0.08, -0.02])
            }
        }

        result = analyzer.analyze_capital_allocation(strategies, 1000000)

        allocation = result['strategy_allocations']['losing_strategy']
        assert allocation['suggested_allocation'] == 0
        assert allocation['allocation_percentage'] == 0

        # Risk-adjusted return should handle division by zero
        assert isinstance(allocation['risk_adjusted_return'], (int, float))

    def test_margin_rate_edge_cases(self):
        """Test margin rate edge cases."""
        # Very high margin rate
        high_margin_analyzer = CapitalEfficiencyAnalyzer(daily_margin_rate=0.01)
        assert high_margin_analyzer.daily_margin_rate == 0.01

        # Very small margin rate
        tiny_margin_analyzer = CapitalEfficiencyAnalyzer(daily_margin_rate=1e-10)
        assert tiny_margin_analyzer.daily_margin_rate == 1e-10


class TestCapitalEfficiencyIntegration:
    """Test integration scenarios."""

    def test_full_analysis_pipeline(self):
        """Test complete analysis pipeline."""
        analyzer = CapitalEfficiencyAnalyzer()

        # Simulate realistic trading data
        np.random.seed(42)  # For reproducible results
        returns_a = np.random.normal(0.001, 0.02, 100)  # Strategy A
        returns_b = np.random.normal(0.0005, 0.015, 100)  # Strategy B

        strategies = {
            'momentum_strategy': {'trade_returns': pd.Series(returns_a)},
            'mean_reversion': {'trade_returns': pd.Series(returns_b)}
        }

        # Run capital allocation analysis
        allocation_result = analyzer.analyze_capital_allocation(strategies, 1000000)

        # Verify complete results
        assert len(allocation_result['strategy_allocations']) == 2
        assert allocation_result['total_capital'] == 1000000
        assert 0 <= allocation_result['capital_utilization'] <= 1.0

        # Run Kelly analysis for each strategy
        for strategy_data in strategies.values():
            kelly_result = analyzer.kelly_sizing_analysis(strategy_data['trade_returns'])
            assert isinstance(kelly_result, KellyResult)
            assert 0 <= kelly_result.kelly_fraction <= 1.0

    def test_performance_with_large_datasets(self):
        """Test performance with large datasets."""
        analyzer = CapitalEfficiencyAnalyzer()

        # Large return series
        large_returns = pd.Series(np.random.normal(0.001, 0.02, 10000))

        # Should complete in reasonable time
        result = analyzer.kelly_sizing_analysis(large_returns)
        assert isinstance(result, KellyResult)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])