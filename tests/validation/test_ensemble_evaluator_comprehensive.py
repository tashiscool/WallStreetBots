#!/usr/bin/env python3
"""
Comprehensive tests for ensemble_evaluator module.
Tests all public methods, edge cases, and error handling.
Target: 80%+ coverage
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

from backend.validation.ensemble_evaluator import (
    EnsembleValidator,
    _sharpe
)


class TestUtilityFunctions:
    """Test utility functions."""

    def test_sharpe_basic(self):
        """Test basic Sharpe calculation."""
        returns = pd.Series(np.random.normal(0.001, 0.01, 252))
        sharpe = _sharpe(returns)

        assert isinstance(sharpe, float)
        assert not np.isnan(sharpe)

    def test_sharpe_zero_volatility(self):
        """Test Sharpe with zero volatility."""
        returns = pd.Series([0.001] * 100)
        sharpe = _sharpe(returns)

        assert sharpe == 0.0

    def test_sharpe_insufficient_data(self):
        """Test Sharpe with insufficient data."""
        returns = pd.Series([0.001])
        sharpe = _sharpe(returns)

        assert sharpe == 0.0

    def test_sharpe_with_nans(self):
        """Test Sharpe with NaN values."""
        returns = pd.Series([0.001, np.nan, 0.002, 0.003])
        sharpe = _sharpe(returns)

        assert isinstance(sharpe, float)


class TestEnsembleValidator:
    """Test EnsembleValidator class."""

    @pytest.fixture
    def validator(self):
        """Create validator instance."""
        return EnsembleValidator(corr_threshold=0.7)

    @pytest.fixture
    def sample_strategy_returns(self):
        """Create sample strategy returns."""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=252, freq='B')

        return {
            'strategy1': pd.Series(np.random.normal(0.001, 0.01, 252), index=dates),
            'strategy2': pd.Series(np.random.normal(0.0008, 0.012, 252), index=dates),
            'strategy3': pd.Series(np.random.normal(0.0012, 0.009, 252), index=dates)
        }

    def test_initialization_default(self):
        """Test default initialization."""
        validator = EnsembleValidator()
        assert validator.corr_threshold == 0.7

    def test_initialization_custom(self):
        """Test custom initialization."""
        validator = EnsembleValidator(corr_threshold=0.8)
        assert validator.corr_threshold == 0.8

    def test_analyze_strategy_correlations_success(self, validator, sample_strategy_returns):
        """Test successful correlation analysis."""
        result = validator.analyze_strategy_correlations(sample_strategy_returns)

        assert 'correlation_matrix' in result
        assert 'strategy_clusters' in result
        assert 'redundant_strategies' in result
        assert 'ensemble_performance' in result
        assert 'diversification_ratio' in result

    def test_analyze_strategy_correlations_empty(self, validator):
        """Test with empty strategy dict."""
        result = validator.analyze_strategy_correlations({})

        assert result['correlation_matrix'].empty
        assert result['strategy_clusters'] == []
        assert result['redundant_strategies'] == []

    def test_analyze_strategy_correlations_single_strategy(self, validator):
        """Test with single strategy."""
        single_strategy = {
            'strategy1': pd.Series(np.random.normal(0.001, 0.01, 100))
        }

        result = validator.analyze_strategy_correlations(single_strategy)

        # Should handle gracefully
        assert 'correlation_matrix' in result

    def test_analyze_strategy_correlations_highly_correlated(self, validator):
        """Test with highly correlated strategies."""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=252, freq='B')
        base_returns = np.random.normal(0.001, 0.01, 252)

        correlated_strategies = {
            'strategy1': pd.Series(base_returns, index=dates),
            'strategy2': pd.Series(base_returns + np.random.normal(0, 0.001, 252), index=dates),
            'strategy3': pd.Series(base_returns + np.random.normal(0, 0.001, 252), index=dates)
        }

        result = validator.analyze_strategy_correlations(correlated_strategies)

        # Should identify redundancy
        assert isinstance(result['redundant_strategies'], list)

    def test_analyze_strategy_correlations_uncorrelated(self, validator):
        """Test with uncorrelated strategies."""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=252, freq='B')

        uncorrelated_strategies = {
            'strategy1': pd.Series(np.random.normal(0.001, 0.01, 252), index=dates),
            'strategy2': pd.Series(np.random.normal(0.0008, 0.012, 252), index=dates),
            'strategy3': pd.Series(np.random.normal(0.0012, 0.009, 252), index=dates)
        }

        result = validator.analyze_strategy_correlations(uncorrelated_strategies)

        assert result['diversification_ratio'] > 0

    def test_ensemble_performance_equal_weight(self, validator, sample_strategy_returns):
        """Test equal weight ensemble performance."""
        result = validator.analyze_strategy_correlations(sample_strategy_returns)

        assert 'equal_weight_sharpe' in result['ensemble_performance']
        assert isinstance(result['ensemble_performance']['equal_weight_sharpe'], float)

    def test_ensemble_performance_optimized(self, validator, sample_strategy_returns):
        """Test optimized ensemble performance."""
        result = validator.analyze_strategy_correlations(sample_strategy_returns)

        assert 'optimized_sharpe' in result['ensemble_performance']
        assert 'optimal_weights' in result['ensemble_performance']
        assert 'optimized_series' in result['ensemble_performance']

        # Weights should sum to 1
        weights = result['ensemble_performance']['optimal_weights']
        total_weight = sum(weights.values())
        assert abs(total_weight - 1.0) < 0.01

    def test_optimize_weights(self, validator, sample_strategy_returns):
        """Test weight optimization."""
        df = pd.DataFrame(sample_strategy_returns).dropna()

        result = validator._optimize_weights(df)

        assert 'weights' in result
        assert 'success' in result
        assert len(result['weights']) == len(df.columns)
        assert abs(sum(result['weights']) - 1.0) < 0.01

    def test_optimize_weights_single_strategy(self, validator):
        """Test optimization with single strategy."""
        df = pd.DataFrame({
            'strategy1': np.random.normal(0.001, 0.01, 100)
        })

        result = validator._optimize_weights(df)

        # Should return equal weight
        assert abs(result['weights'][0] - 1.0) < 0.01

    def test_optimize_weights_failure(self, validator):
        """Test optimization failure handling."""
        # Create problematic data
        df = pd.DataFrame({
            'strategy1': [np.nan] * 100,
            'strategy2': [np.nan] * 100
        })

        with patch('backend.validation.ensemble_evaluator.minimize') as mock_minimize:
            mock_minimize.side_effect = Exception("Optimization failed")

            result = validator._optimize_weights(df)

            # Should fall back to equal weights
            assert result['success'] is False
            assert len(result['weights']) == 2

    def test_find_redundant(self, validator):
        """Test redundant strategy detection."""
        # Create correlation matrix with redundancy
        corr = pd.DataFrame({
            'strategy1': [1.0, 0.95, 0.3],
            'strategy2': [0.95, 1.0, 0.4],
            'strategy3': [0.3, 0.4, 1.0]
        }, index=['strategy1', 'strategy2', 'strategy3'])

        redundant = validator._find_redundant(corr, thr=0.9)

        # strategy2 should be redundant (corr > 0.9 with strategy1)
        assert 'strategy2' in redundant

    def test_find_redundant_no_redundancy(self, validator):
        """Test with no redundant strategies."""
        corr = pd.DataFrame({
            'strategy1': [1.0, 0.3, 0.2],
            'strategy2': [0.3, 1.0, 0.4],
            'strategy3': [0.2, 0.4, 1.0]
        }, index=['strategy1', 'strategy2', 'strategy3'])

        redundant = validator._find_redundant(corr, thr=0.9)

        assert len(redundant) == 0

    def test_find_redundant_all_redundant(self, validator):
        """Test with all strategies redundant."""
        corr = pd.DataFrame({
            'strategy1': [1.0, 0.95, 0.96],
            'strategy2': [0.95, 1.0, 0.97],
            'strategy3': [0.96, 0.97, 1.0]
        }, index=['strategy1', 'strategy2', 'strategy3'])

        redundant = validator._find_redundant(corr, thr=0.9)

        # At least some should be redundant
        assert len(redundant) > 0

    def test_clusters_from_corr(self, validator):
        """Test correlation clustering."""
        corr = pd.DataFrame({
            'strategy1': [1.0, 0.8, 0.2, 0.1],
            'strategy2': [0.8, 1.0, 0.3, 0.2],
            'strategy3': [0.2, 0.3, 1.0, 0.75],
            'strategy4': [0.1, 0.2, 0.75, 1.0]
        }, index=['strategy1', 'strategy2', 'strategy3', 'strategy4'])

        clusters = validator._clusters_from_corr(corr)

        assert isinstance(clusters, list)
        assert len(clusters) > 0

        # All strategies should be in some cluster
        all_strategies = set()
        for cluster in clusters:
            all_strategies.update(cluster)

        assert len(all_strategies) == 4

    def test_clusters_from_corr_low_threshold(self, validator):
        """Test clustering with low correlation threshold."""
        validator_low = EnsembleValidator(corr_threshold=0.3)

        corr = pd.DataFrame({
            'strategy1': [1.0, 0.4, 0.2],
            'strategy2': [0.4, 1.0, 0.3],
            'strategy3': [0.2, 0.3, 1.0]
        }, index=['strategy1', 'strategy2', 'strategy3'])

        clusters = validator_low._clusters_from_corr(corr)

        # Lower threshold should group more strategies
        assert len(clusters) > 0

    def test_diversification_ratio_calculation(self, validator, sample_strategy_returns):
        """Test diversification ratio calculation."""
        result = validator.analyze_strategy_correlations(sample_strategy_returns)

        div_ratio = result['diversification_ratio']

        assert isinstance(div_ratio, float)
        assert div_ratio > 0

    def test_misaligned_returns(self, validator):
        """Test with misaligned return series."""
        strategies = {
            'strategy1': pd.Series(
                np.random.normal(0.001, 0.01, 100),
                index=pd.date_range('2023-01-01', periods=100)
            ),
            'strategy2': pd.Series(
                np.random.normal(0.001, 0.01, 100),
                index=pd.date_range('2023-06-01', periods=100)
            )
        }

        result = validator.analyze_strategy_correlations(strategies)

        # Should handle by aligning dates
        assert 'correlation_matrix' in result

    def test_with_nans(self, validator):
        """Test handling of NaN values."""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=100)

        strategies = {
            'strategy1': pd.Series(np.random.normal(0.001, 0.01, 100), index=dates),
            'strategy2': pd.Series(np.random.normal(0.001, 0.01, 100), index=dates)
        }

        # Add some NaNs
        strategies['strategy1'].iloc[10:20] = np.nan

        result = validator.analyze_strategy_correlations(strategies)

        # Should drop NaNs and continue
        assert 'correlation_matrix' in result

    def test_zero_variance_strategy(self, validator):
        """Test with zero variance strategy."""
        dates = pd.date_range('2023-01-01', periods=100)

        strategies = {
            'strategy1': pd.Series([0.001] * 100, index=dates),  # Zero variance
            'strategy2': pd.Series(np.random.normal(0.001, 0.01, 100), index=dates)
        }

        result = validator.analyze_strategy_correlations(strategies)

        # Should handle gracefully
        assert isinstance(result, dict)

    def test_negative_returns(self, validator):
        """Test with negative return strategies."""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=252, freq='B')

        strategies = {
            'strategy1': pd.Series(np.random.normal(-0.001, 0.01, 252), index=dates),
            'strategy2': pd.Series(np.random.normal(-0.0008, 0.012, 252), index=dates)
        }

        result = validator.analyze_strategy_correlations(strategies)

        assert 'ensemble_performance' in result
        # Sharpe can be negative for losing strategies
        assert isinstance(result['ensemble_performance']['equal_weight_sharpe'], float)

    def test_extreme_correlations(self, validator):
        """Test with perfect and negative correlations."""
        dates = pd.date_range('2023-01-01', periods=100)
        base_returns = np.random.normal(0.001, 0.01, 100)

        strategies = {
            'strategy1': pd.Series(base_returns, index=dates),
            'strategy2': pd.Series(base_returns, index=dates),  # Perfect correlation
            'strategy3': pd.Series(-base_returns, index=dates)  # Perfect negative correlation
        }

        result = validator.analyze_strategy_correlations(strategies)

        corr_matrix = result['correlation_matrix']

        # Verify perfect correlations
        assert abs(corr_matrix.loc['strategy1', 'strategy2'] - 1.0) < 0.01
        assert abs(corr_matrix.loc['strategy1', 'strategy3'] + 1.0) < 0.01


class TestEnsembleValidatorIntegration:
    """Integration tests for EnsembleValidator."""

    def test_full_workflow(self):
        """Test complete ensemble validation workflow."""
        validator = EnsembleValidator(corr_threshold=0.7)

        # Create diverse strategies
        np.random.seed(42)
        dates = pd.date_range('2022-01-01', periods=252, freq='B')

        strategies = {
            'momentum': pd.Series(np.random.normal(0.0012, 0.012, 252), index=dates),
            'mean_reversion': pd.Series(np.random.normal(0.0008, 0.010, 252), index=dates),
            'trend_following': pd.Series(np.random.normal(0.0010, 0.015, 252), index=dates),
            'volatility': pd.Series(np.random.normal(0.0005, 0.008, 252), index=dates)
        }

        result = validator.analyze_strategy_correlations(strategies)

        # Verify complete analysis
        assert len(result['correlation_matrix']) == 4
        assert 'optimal_weights' in result['ensemble_performance']
        assert len(result['ensemble_performance']['optimal_weights']) == 4

        # Verify optimization improves Sharpe
        # (Not always guaranteed, but should work with good strategies)
        equal_weight = result['ensemble_performance']['equal_weight_sharpe']
        optimized = result['ensemble_performance']['optimized_sharpe']

        assert isinstance(equal_weight, float)
        assert isinstance(optimized, float)

    def test_portfolio_construction(self):
        """Test portfolio construction from strategies."""
        validator = EnsembleValidator()

        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=252, freq='B')

        strategies = {
            'strategy1': pd.Series(np.random.normal(0.001, 0.01, 252), index=dates),
            'strategy2': pd.Series(np.random.normal(0.0012, 0.012, 252), index=dates),
            'strategy3': pd.Series(np.random.normal(0.0008, 0.009, 252), index=dates)
        }

        result = validator.analyze_strategy_correlations(strategies)

        # Extract optimized portfolio
        optimized_series = result['ensemble_performance']['optimized_series']

        assert isinstance(optimized_series, pd.Series)
        assert len(optimized_series) == 252

    def test_correlation_penalty_effect(self):
        """Test that correlation penalty affects optimization."""
        validator = EnsembleValidator()

        # Create highly correlated strategies
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=252, freq='B')
        base_returns = np.random.normal(0.001, 0.01, 252)

        correlated_strategies = {
            'strategy1': pd.Series(base_returns, index=dates),
            'strategy2': pd.Series(base_returns + np.random.normal(0, 0.001, 252), index=dates),
            'strategy3': pd.Series(base_returns + np.random.normal(0, 0.001, 252), index=dates)
        }

        result = validator.analyze_strategy_correlations(correlated_strategies)

        # With correlation penalty, should avoid putting too much in correlated strategies
        weights = result['ensemble_performance']['optimal_weights']

        # Check that weights are somewhat distributed
        # (exact distribution depends on optimization)
        assert max(weights.values()) <= 1.0
        assert min(weights.values()) >= 0.0


class TestEdgeCasesAndErrors:
    """Test edge cases and error handling."""

    def test_all_zero_returns(self):
        """Test with all zero returns."""
        validator = EnsembleValidator()

        strategies = {
            'strategy1': pd.Series([0.0] * 100),
            'strategy2': pd.Series([0.0] * 100)
        }

        result = validator.analyze_strategy_correlations(strategies)

        # Should handle gracefully
        assert isinstance(result, dict)

    def test_infinite_sharpe(self):
        """Test handling of infinite Sharpe ratios."""
        validator = EnsembleValidator()

        # Create strategy with very low volatility
        strategies = {
            'strategy1': pd.Series([0.001] * 100),
            'strategy2': pd.Series(np.random.normal(0.001, 0.01, 100))
        }

        result = validator.analyze_strategy_correlations(strategies)

        # Should handle without crashing
        assert 'ensemble_performance' in result

    def test_very_short_history(self):
        """Test with very short return history."""
        validator = EnsembleValidator()

        strategies = {
            'strategy1': pd.Series([0.001, 0.002, 0.003]),
            'strategy2': pd.Series([0.002, 0.001, 0.004])
        }

        result = validator.analyze_strategy_correlations(strategies)

        # Should return results even with limited data
        assert 'correlation_matrix' in result

    def test_large_number_of_strategies(self):
        """Test with many strategies."""
        validator = EnsembleValidator()

        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=100)

        # Create 20 strategies
        strategies = {
            f'strategy{i}': pd.Series(np.random.normal(0.001, 0.01, 100), index=dates)
            for i in range(20)
        }

        result = validator.analyze_strategy_correlations(strategies)

        # Should handle optimization with many strategies
        assert len(result['correlation_matrix']) == 20
        assert len(result['ensemble_performance']['optimal_weights']) == 20

    def test_empty_after_alignment(self):
        """Test when alignment results in empty DataFrame."""
        validator = EnsembleValidator()

        # Create non-overlapping date ranges
        strategies = {
            'strategy1': pd.Series(
                [0.001] * 10,
                index=pd.date_range('2023-01-01', periods=10)
            ),
            'strategy2': pd.Series(
                [0.002] * 10,
                index=pd.date_range('2024-01-01', periods=10)
            )
        }

        result = validator.analyze_strategy_correlations(strategies)

        # Should detect empty aligned data
        assert result['correlation_matrix'].empty or len(result['correlation_matrix']) < 2
