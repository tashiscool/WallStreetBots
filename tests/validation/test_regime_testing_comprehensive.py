#!/usr/bin/env python3
"""
Comprehensive tests for regime_testing module.
Tests all public methods, edge cases, and error handling.
Target: 80%+ coverage
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
from datetime import datetime

from backend.validation.regime_testing import (
    RegimeValidator,
    _max_drawdown,
    _sharpe,
    _calmar_ratio,
    _sortino_ratio
)


class TestUtilityFunctions:
    """Test utility functions for metrics calculation."""

    def test_max_drawdown_simple(self):
        """Test max drawdown calculation with simple data."""
        returns = pd.Series([0.01, 0.02, -0.03, -0.02, 0.04])
        dd = _max_drawdown(returns)

        assert dd < 0  # Drawdown should be negative
        assert dd >= -1  # Cannot exceed -100%

    def test_max_drawdown_no_drawdown(self):
        """Test with continuously positive returns."""
        returns = pd.Series([0.01, 0.02, 0.03, 0.04])
        dd = _max_drawdown(returns)

        assert dd == 0.0

    def test_max_drawdown_empty(self):
        """Test with empty series."""
        returns = pd.Series(dtype=float)
        dd = _max_drawdown(returns)

        assert dd == 0.0

    def test_sharpe_ratio(self):
        """Test Sharpe ratio calculation."""
        returns = pd.Series(np.random.normal(0.001, 0.01, 252))
        sharpe = _sharpe(returns)

        assert isinstance(sharpe, float)
        assert not np.isnan(sharpe)

    def test_sharpe_ratio_zero_volatility(self):
        """Test Sharpe with zero volatility."""
        returns = pd.Series([0.001] * 100)
        sharpe = _sharpe(returns)

        assert sharpe == 0.0

    def test_sharpe_ratio_insufficient_data(self):
        """Test Sharpe with insufficient data."""
        returns = pd.Series([0.001])
        sharpe = _sharpe(returns)

        assert sharpe == 0.0

    def test_calmar_ratio(self):
        """Test Calmar ratio calculation."""
        returns = pd.Series(np.random.normal(0.001, 0.015, 252))
        calmar = _calmar_ratio(returns)

        assert isinstance(calmar, float)

    def test_calmar_ratio_no_drawdown(self):
        """Test Calmar with no drawdown."""
        returns = pd.Series([0.01] * 100)
        calmar = _calmar_ratio(returns)

        assert calmar > 0 or calmar == float('inf')

    def test_calmar_ratio_negative_returns_no_drawdown(self):
        """Test Calmar with negative returns but no drawdown."""
        returns = pd.Series([-0.01] * 100)
        calmar = _calmar_ratio(returns)

        assert isinstance(calmar, float)

    def test_sortino_ratio(self):
        """Test Sortino ratio calculation."""
        returns = pd.Series(np.random.normal(0.001, 0.01, 252))
        sortino = _sortino_ratio(returns)

        assert isinstance(sortino, float)

    def test_sortino_ratio_no_downside(self):
        """Test Sortino with only positive returns."""
        returns = pd.Series([0.01, 0.02, 0.03, 0.015])
        sortino = _sortino_ratio(returns)

        assert sortino > 0 or sortino == float('inf')

    def test_sortino_ratio_insufficient_data(self):
        """Test Sortino with insufficient data."""
        returns = pd.Series([0.001])
        sortino = _sortino_ratio(returns)

        assert sortino == 0.0


class TestRegimeValidator:
    """Test RegimeValidator class."""

    @pytest.fixture
    def validator(self):
        """Create validator instance."""
        return RegimeValidator()

    @pytest.fixture
    def sample_market_data(self, validator):
        """Create sample market data."""
        return validator.create_synthetic_market_data('2022-01-01', '2023-12-31')

    @pytest.fixture
    def sample_strategy_returns(self, sample_market_data):
        """Create sample strategy returns."""
        np.random.seed(42)
        spy_returns = sample_market_data['SPY'].pct_change().dropna()

        returns = (
            0.0003 +  # Base alpha
            0.5 * spy_returns +  # Market exposure
            np.random.normal(0, 0.01, len(spy_returns))
        ).dropna()

        returns.name = 'strategy'
        return returns

    def test_initialization(self, validator):
        """Test validator initialization."""
        assert validator.logger is not None
        assert len(validator.regimes) > 0
        assert 'bull_market' in validator.regimes
        assert 'bear_market' in validator.regimes

    def test_regime_definitions(self, validator):
        """Test that all regime detection functions exist."""
        expected_regimes = [
            'bull_market', 'bear_market', 'high_vol', 'low_vol',
            'rate_hiking', 'rate_cutting', 'recession', 'expansion',
            'vix_spike', 'quiet_market'
        ]

        for regime in expected_regimes:
            assert regime in validator.regimes

    def test_test_edge_persistence_success(self, validator, sample_strategy_returns, sample_market_data):
        """Test successful edge persistence analysis."""
        results = validator.test_edge_persistence(
            sample_strategy_returns,
            sample_market_data,
            min_observations=20
        )

        assert 'regime_results' in results
        assert 'edge_is_robust' in results
        assert 'weakest_regime' in results
        assert 'strongest_regime' in results
        assert isinstance(results['edge_is_robust'], bool)

    def test_test_edge_persistence_multiple_regimes(self, validator, sample_strategy_returns, sample_market_data):
        """Test that multiple regimes are detected."""
        results = validator.test_edge_persistence(
            sample_strategy_returns,
            sample_market_data,
            min_observations=10
        )

        # Should detect multiple regimes
        assert len(results['regime_results']) > 0

    def test_test_edge_persistence_insufficient_data(self, validator):
        """Test with insufficient data."""
        short_returns = pd.Series([0.001] * 10)
        short_market = pd.DataFrame({
            'SPY': [100.0] * 10,
            'VIX': [20.0] * 10,
            'DGS10': [2.5] * 10
        })

        results = validator.test_edge_persistence(short_returns, short_market)

        assert 'error' in results
        assert results['edge_is_robust'] is False

    def test_test_edge_persistence_no_aligned_data(self, validator):
        """Test with no aligned data."""
        returns = pd.Series([0.001] * 100, index=pd.date_range('2023-01-01', periods=100))
        market = pd.DataFrame({
            'SPY': [100.0] * 100,
            'VIX': [20.0] * 100,
            'DGS10': [2.5] * 100
        }, index=pd.date_range('2024-01-01', periods=100))  # Different dates

        results = validator.test_edge_persistence(returns, market)

        assert 'error' in results or results['regime_results'] == {}

    def test_align_data(self, validator, sample_strategy_returns, sample_market_data):
        """Test data alignment."""
        aligned = validator._align_data(sample_strategy_returns, sample_market_data)

        assert aligned is not None
        assert 'strategy_returns' in aligned.columns
        assert 'SPY' in aligned.columns
        assert len(aligned) > 0

    def test_align_data_with_nans(self, validator):
        """Test alignment with NaN values."""
        # Need at least 50 clean observations after dropping NaNs
        np.random.seed(42)
        n_points = 60
        returns_data = np.random.normal(0.001, 0.01, n_points)
        returns_data[5] = np.nan  # Add some NaN values
        returns_data[15] = np.nan
        returns = pd.Series(returns_data, index=pd.date_range('2023-01-01', periods=n_points))

        spy_data = 100 + np.cumsum(np.random.normal(0, 1, n_points))
        spy_data[10] = np.nan
        market = pd.DataFrame({
            'SPY': spy_data,
            'VIX': np.random.uniform(15, 25, n_points),
            'DGS10': np.random.uniform(2, 3, n_points)
        }, index=pd.date_range('2023-01-01', periods=n_points))

        aligned = validator._align_data(returns, market)

        # Should drop rows with NaN
        assert aligned is not None
        assert len(aligned) < n_points

    def test_calculate_regime_metrics(self, validator):
        """Test regime metrics calculation."""
        returns = pd.Series(np.random.normal(0.001, 0.01, 100))
        metrics = validator._calculate_regime_metrics(returns)

        assert 'sample_size' in metrics
        assert 'avg_return' in metrics
        assert 'volatility' in metrics
        assert 'sharpe_ratio' in metrics
        assert 'win_rate' in metrics
        assert 'profit_factor' in metrics

        assert metrics['sample_size'] == 100

    def test_calculate_regime_metrics_empty(self, validator):
        """Test metrics with empty returns."""
        empty_returns = pd.Series(dtype=float)
        metrics = validator._calculate_regime_metrics(empty_returns)

        assert metrics['sample_size'] == 0
        assert metrics['avg_return'] == 0.0

    def test_calculate_regime_metrics_all_losses(self, validator):
        """Test metrics with all losing trades."""
        loss_returns = pd.Series([-0.01] * 50)
        metrics = validator._calculate_regime_metrics(loss_returns)

        assert metrics['win_rate'] == 0.0
        assert metrics['avg_win'] == 0.0

    def test_calculate_regime_metrics_all_wins(self, validator):
        """Test metrics with all winning trades."""
        win_returns = pd.Series([0.01] * 50)
        metrics = validator._calculate_regime_metrics(win_returns)

        assert metrics['win_rate'] == 1.0
        assert metrics['avg_loss'] == 0.0

    def test_calculate_profit_factor(self, validator):
        """Test profit factor calculation."""
        returns = pd.Series([0.02, -0.01, 0.03, -0.01, 0.01])
        pf = validator._calculate_profit_factor(returns)

        # Profit factor = (0.02 + 0.03 + 0.01) / (0.01 + 0.01) = 3.0
        assert pf > 0

    def test_calculate_profit_factor_no_losses(self, validator):
        """Test profit factor with no losses."""
        returns = pd.Series([0.01, 0.02, 0.03])
        pf = validator._calculate_profit_factor(returns)

        assert pf == float('inf')

    def test_calculate_profit_factor_no_gains(self, validator):
        """Test profit factor with no gains."""
        returns = pd.Series([-0.01, -0.02, -0.03])
        pf = validator._calculate_profit_factor(returns)

        assert pf == 1.0 or pf == 0.0

    def test_empty_metrics(self, validator):
        """Test empty metrics generation."""
        metrics = validator._empty_metrics()

        assert metrics['sample_size'] == 0
        assert metrics['sharpe_ratio'] == 0.0
        assert metrics['win_rate'] == 0.0
        assert len(metrics) > 0

    def test_analyze_edge_robustness(self, validator):
        """Test edge robustness analysis."""
        regime_results = {
            'bull_market': {
                'sharpe_ratio': 1.5,
                'win_rate': 0.60,
                'max_drawdown': -0.10
            },
            'bear_market': {
                'sharpe_ratio': 0.8,
                'win_rate': 0.52,
                'max_drawdown': -0.15
            },
            'high_vol': {
                'sharpe_ratio': 1.2,
                'win_rate': 0.55,
                'max_drawdown': -0.12
            }
        }

        analysis = validator._analyze_edge_robustness(regime_results)

        assert 'is_robust' in analysis
        assert 'weakest_regime' in analysis
        assert 'strongest_regime' in analysis
        assert 'robustness_score' in analysis
        assert 'consistency_metrics' in analysis

    def test_analyze_edge_robustness_empty(self, validator):
        """Test robustness analysis with empty results."""
        analysis = validator._analyze_edge_robustness({})

        assert analysis['is_robust'] is False
        assert analysis['weakest_regime'] is None
        assert analysis['robustness_score'] == 0.0

    def test_calculate_robustness_score(self, validator):
        """Test robustness score calculation."""
        metrics = {
            'sharpe_mean': 1.2,
            'sharpe_std': 0.3,
            'sharpe_min': 0.5,
            'positive_sharpe_pct': 0.8,
            'win_rate_mean': 0.55,
            'win_rate_min': 0.48,
            'max_drawdown_worst': -0.15,
            'regimes_tested': 5
        }

        score = validator._calculate_robustness_score(metrics)

        assert 0 <= score <= 1

    def test_calculate_robustness_score_perfect(self, validator):
        """Test robustness score with perfect metrics."""
        metrics = {
            'sharpe_mean': 2.0,
            'sharpe_std': 0.0,
            'sharpe_min': 2.0,
            'positive_sharpe_pct': 1.0,
            'win_rate_mean': 0.7,
            'win_rate_min': 0.7,
            'max_drawdown_worst': 0.0,
            'regimes_tested': 5
        }

        score = validator._calculate_robustness_score(metrics)

        assert score > 0.8

    def test_create_synthetic_market_data(self, validator):
        """Test synthetic market data creation."""
        market_data = validator.create_synthetic_market_data('2022-01-01', '2023-12-31')

        assert isinstance(market_data, pd.DataFrame)
        assert 'SPY' in market_data.columns
        assert 'VIX' in market_data.columns
        assert 'DGS10' in market_data.columns
        assert len(market_data) > 200

    def test_create_synthetic_market_data_short_period(self, validator):
        """Test synthetic data for short period."""
        market_data = validator.create_synthetic_market_data('2023-01-01', '2023-03-31')

        assert len(market_data) > 0
        assert all(col in market_data.columns for col in ['SPY', 'VIX', 'DGS10'])

    def test_create_synthetic_market_data_realistic_values(self, validator):
        """Test that synthetic data has realistic values."""
        market_data = validator.create_synthetic_market_data('2022-01-01', '2023-12-31')

        # VIX should be positive and reasonable
        assert (market_data['VIX'] > 0).all()
        assert (market_data['VIX'] < 100).all()

        # Interest rates should be positive
        assert (market_data['DGS10'] >= 0).all()

        # SPY should be positive
        assert (market_data['SPY'] > 0).all()

    def test_get_regime_summary(self, validator):
        """Test regime summary generation."""
        results = {
            'edge_is_robust': True,
            'robustness_score': 0.85,
            'strongest_regime': 'bull_market',
            'weakest_regime': 'bear_market',
            'regime_results': {
                'bull_market': {'sharpe_ratio': 1.5, 'win_rate': 0.60},
                'bear_market': {'sharpe_ratio': 0.8, 'win_rate': 0.52}
            },
            'consistency_metrics': {
                'positive_sharpe_pct': 0.75,
                'win_rate_min': 0.50,
                'max_drawdown_worst': -0.15
            }
        }

        summary = validator.get_regime_summary(results)

        assert isinstance(summary, str)
        assert 'ROBUST' in summary
        assert '0.85' in summary
        assert 'bull_market' in summary

    def test_get_regime_summary_not_robust(self, validator):
        """Test summary for non-robust strategy."""
        results = {
            'edge_is_robust': False,
            'robustness_score': 0.35,
            'strongest_regime': 'bull_market',
            'weakest_regime': 'bear_market',
            'regime_results': {
                'bull_market': {'sharpe_ratio': 1.0, 'win_rate': 0.55}
            },
            'consistency_metrics': {
                'positive_sharpe_pct': 0.50,
                'win_rate_min': 0.45,
                'max_drawdown_worst': -0.25
            }
        }

        summary = validator.get_regime_summary(results)

        assert 'REGIME DEPENDENCE' in summary or 'regime' in summary.lower()

    def test_get_regime_summary_no_results(self, validator):
        """Test summary with no results."""
        results = {'regime_results': {}}

        summary = validator.get_regime_summary(results)

        assert 'No regime analysis results' in summary

    def test_regime_detection_functions(self, validator, sample_market_data):
        """Test that regime detection functions work."""
        for regime_func in validator.regimes.values():
            mask = regime_func(sample_market_data)

            assert isinstance(mask, pd.Series)
            assert mask.dtype == bool or mask.dtype == object  # Some may return objects

    def test_robustness_criteria(self, validator):
        """Test robustness criteria thresholds."""
        # Create metrics that should pass
        passing_metrics = {
            'positive_sharpe_pct': 0.75,
            'sharpe_min': 0.5,
            'win_rate_min': 0.50,
            'max_drawdown_worst': -0.20
        }

        analysis = validator._analyze_edge_robustness({
            'regime1': {
                'sharpe_ratio': 1.0,
                'win_rate': 0.55,
                'max_drawdown': -0.15
            },
            'regime2': {
                'sharpe_ratio': 0.8,
                'win_rate': 0.52,
                'max_drawdown': -0.18
            },
            'regime3': {
                'sharpe_ratio': 1.2,
                'win_rate': 0.58,
                'max_drawdown': -0.12
            },
            'regime4': {
                'sharpe_ratio': 0.5,
                'win_rate': 0.50,
                'max_drawdown': -0.20
            }
        })

        # Verify criteria are evaluated
        assert 'consistency_metrics' in analysis
        metrics = analysis['consistency_metrics']
        assert 'positive_sharpe_pct' in metrics
        assert 'sharpe_min' in metrics
        assert 'win_rate_min' in metrics


class TestRegimeValidatorIntegration:
    """Integration tests for RegimeValidator."""

    def test_full_workflow(self):
        """Test complete regime validation workflow."""
        validator = RegimeValidator()

        # Create market data
        market_data = validator.create_synthetic_market_data('2022-01-01', '2023-12-31')

        # Create strategy returns - align all series properly
        np.random.seed(42)
        spy_returns = market_data['SPY'].pct_change().dropna()

        # Use aligned indices for all components
        aligned_vix = market_data['VIX'].loc[spy_returns.index]
        noise = pd.Series(np.random.normal(0, 0.01, len(spy_returns)), index=spy_returns.index)

        strategy_returns = (
            0.0005 +  # Alpha
            0.6 * spy_returns +
            -0.01 * (aligned_vix - 20) / 20 +
            noise
        ).dropna()

        # Run analysis
        results = validator.test_edge_persistence(strategy_returns, market_data, min_observations=15)

        # Verify complete results
        assert 'regime_results' in results
        assert 'edge_is_robust' in results
        assert 'robustness_score' in results

        # Generate summary
        summary = validator.get_regime_summary(results)
        assert len(summary) > 0

    def test_strategy_comparison(self):
        """Test comparing multiple strategies across regimes."""
        validator = RegimeValidator()
        market_data = validator.create_synthetic_market_data('2022-01-01', '2023-12-31')

        np.random.seed(42)
        spy_returns = market_data['SPY'].pct_change().dropna()

        # Use aligned indices for all components
        aligned_vix = market_data['VIX'].loc[spy_returns.index]
        noise1 = pd.Series(np.random.normal(0, 0.008, len(spy_returns)), index=spy_returns.index)
        noise2 = pd.Series(np.random.normal(0, 0.012, len(spy_returns)), index=spy_returns.index)

        # Strategy 1: Low volatility preference
        strategy1 = (
            0.0003 +
            0.5 * spy_returns -
            0.02 * (aligned_vix - 20) / 20 +
            noise1
        ).dropna()

        # Strategy 2: High volatility preference
        strategy2 = (
            0.0002 +
            0.7 * spy_returns +
            0.01 * (aligned_vix - 20) / 20 +
            noise2
        ).dropna()

        results1 = validator.test_edge_persistence(strategy1, market_data, min_observations=15)
        results2 = validator.test_edge_persistence(strategy2, market_data, min_observations=15)

        # Both should have results
        assert len(results1.get('regime_results', {})) > 0
        assert len(results2.get('regime_results', {})) > 0


class TestEdgeCasesAndErrors:
    """Test edge cases and error handling."""

    def test_single_observation_per_regime(self):
        """Test handling when regimes have very few observations."""
        validator = RegimeValidator()

        # Very short data
        returns = pd.Series([0.001] * 50)
        market = pd.DataFrame({
            'SPY': np.linspace(100, 110, 50),
            'VIX': [20] * 50,
            'DGS10': [2.5] * 50
        })

        results = validator.test_edge_persistence(returns, market, min_observations=100)

        # Should handle gracefully
        assert 'regime_results' in results

    def test_extreme_volatility_regime(self):
        """Test with extreme market volatility."""
        validator = RegimeValidator()

        # Create extreme volatility scenario
        market_data = pd.DataFrame({
            'SPY': [100] * 100,
            'VIX': [100] * 100,  # Extreme VIX
            'DGS10': [2.5] * 100
        }, index=pd.date_range('2023-01-01', periods=100))

        returns = pd.Series(np.random.normal(0, 0.05, 100), index=market_data.index)

        results = validator.test_edge_persistence(returns, market_data, min_observations=10)

        # Should still produce results
        assert isinstance(results, dict)

    def test_all_same_regime(self):
        """Test when all data falls into one regime."""
        validator = RegimeValidator()

        # Create stable market (all low vol)
        market_data = pd.DataFrame({
            'SPY': np.linspace(100, 110, 100),
            'VIX': [10] * 100,  # Always low
            'DGS10': [2.5] * 100
        }, index=pd.date_range('2023-01-01', periods=100))

        returns = pd.Series(np.random.normal(0.001, 0.005, 100), index=market_data.index)

        results = validator.test_edge_persistence(returns, market_data, min_observations=20)

        # Should handle limited regime diversity
        assert isinstance(results, dict)
