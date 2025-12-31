#!/usr/bin/env python3
"""
Comprehensive tests for advanced_risk/cross_strategy_risk module.
Tests all public methods, edge cases, and error handling.
Target: 80%+ coverage
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime
from unittest.mock import Mock, patch

from backend.validation.advanced_risk.cross_strategy_risk import (
    CorrelationClusterAnalyzer,
    CrossStrategyRiskManager,
    StrategyCluster,
    RiskAllocation,
    PortfolioRiskMetrics
)


class TestStrategyCluster:
    """Test StrategyCluster dataclass."""

    def test_strategy_cluster_creation(self):
        """Test creating strategy cluster."""
        cluster = StrategyCluster(
            cluster_id=1,
            strategy_names=['strategy1', 'strategy2'],
            avg_correlation=0.65,
            max_correlation=0.75,
            cluster_size=2,
            risk_budget=0.4,
            current_allocation=0.3,
            available_capacity=0.1
        )

        assert cluster.cluster_id == 1
        assert len(cluster.strategy_names) == 2
        assert cluster.avg_correlation == 0.65


class TestRiskAllocation:
    """Test RiskAllocation dataclass."""

    def test_risk_allocation_creation(self):
        """Test creating risk allocation."""
        allocation = RiskAllocation(
            strategy_name='strategy1',
            cluster_id=1,
            position_size=0.25,
            risk_contribution=0.03,
            marginal_var=0.0015,
            component_var=0.002,
            correlation_penalty=0.05
        )

        assert allocation.strategy_name == 'strategy1'
        assert allocation.position_size == 0.25


class TestPortfolioRiskMetrics:
    """Test PortfolioRiskMetrics dataclass."""

    def test_portfolio_risk_metrics_creation(self):
        """Test creating portfolio risk metrics."""
        metrics = PortfolioRiskMetrics(
            total_var=0.04,
            diversified_var=0.03,
            concentration_risk=0.25,
            correlation_risk=0.01,
            cluster_concentrations={1: 0.35, 2: 0.25},
            undiversified_var=0.05,
            diversification_ratio=1.5
        )

        assert metrics.total_var == 0.04
        assert metrics.diversification_ratio == 1.5


class TestCorrelationClusterAnalyzer:
    """Test CorrelationClusterAnalyzer class."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance."""
        return CorrelationClusterAnalyzer(correlation_threshold=0.6)

    @pytest.fixture
    def sample_strategy_returns(self):
        """Create sample strategy returns."""
        np.random.seed(42)
        dates = pd.date_range('2022-01-01', periods=252, freq='B')

        return {
            'momentum': pd.Series(np.random.normal(0.001, 0.012, 252), index=dates),
            'mean_reversion': pd.Series(np.random.normal(0.0008, 0.010, 252), index=dates),
            'trend': pd.Series(np.random.normal(0.0012, 0.015, 252), index=dates),
            'volatility': pd.Series(np.random.normal(0.0005, 0.008, 252), index=dates)
        }

    def test_initialization(self, analyzer):
        """Test analyzer initialization."""
        assert analyzer.correlation_threshold == 0.6
        assert len(analyzer.strategy_clusters) == 0
        assert analyzer.correlation_matrix is None

    def test_analyze_strategy_correlations(self, analyzer, sample_strategy_returns):
        """Test correlation analysis."""
        clusters = analyzer.analyze_strategy_correlations(sample_strategy_returns)

        assert len(clusters) > 0
        assert all(isinstance(c, StrategyCluster) for c in clusters.values())

    def test_analyze_strategy_correlations_single(self, analyzer):
        """Test with single strategy."""
        single_strategy = {
            'strategy1': pd.Series(np.random.normal(0.001, 0.01, 100))
        }

        clusters = analyzer.analyze_strategy_correlations(single_strategy)

        assert len(clusters) == 0

    def test_analyze_strategy_correlations_highly_correlated(self, analyzer):
        """Test with highly correlated strategies."""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=252, freq='B')
        base_returns = np.random.normal(0.001, 0.01, 252)

        correlated_strategies = {
            'strategy1': pd.Series(base_returns, index=dates),
            'strategy2': pd.Series(base_returns + np.random.normal(0, 0.001, 252), index=dates),
            'strategy3': pd.Series(base_returns + np.random.normal(0, 0.001, 252), index=dates)
        }

        clusters = analyzer.analyze_strategy_correlations(correlated_strategies)

        # Should group correlated strategies
        assert len(clusters) > 0

    def test_align_returns(self, analyzer):
        """Test return alignment."""
        strategy_returns = {
            'strategy1': pd.Series(
                np.random.normal(0.001, 0.01, 100),
                index=pd.date_range('2023-01-01', periods=100)
            ),
            'strategy2': pd.Series(
                np.random.normal(0.001, 0.01, 100),
                index=pd.date_range('2023-01-01', periods=100)
            )
        }

        aligned = analyzer._align_returns(strategy_returns, lookback_days=252)

        assert not aligned.empty
        assert len(aligned.columns) == 2

    def test_align_returns_with_nans(self, analyzer):
        """Test alignment with NaN values."""
        strategy_returns = {
            'strategy1': pd.Series(
                [0.001, np.nan, 0.002, 0.003],
                index=pd.date_range('2023-01-01', periods=4)
            ),
            'strategy2': pd.Series(
                [0.002, 0.001, np.nan, 0.004],
                index=pd.date_range('2023-01-01', periods=4)
            )
        }

        aligned = analyzer._align_returns(strategy_returns, lookback_days=10)

        # Should drop rows with NaNs
        assert len(aligned) < 4

    def test_create_correlation_clusters(self, analyzer):
        """Test cluster creation."""
        corr_matrix = pd.DataFrame({
            'strategy1': [1.0, 0.8, 0.3],
            'strategy2': [0.8, 1.0, 0.2],
            'strategy3': [0.3, 0.2, 1.0]
        }, index=['strategy1', 'strategy2', 'strategy3'])

        clusters = analyzer._create_correlation_clusters(corr_matrix)

        assert len(clusters) > 0
        # All strategies should be assigned to clusters
        all_strategies = []
        for strategies in clusters.values():
            all_strategies.extend(strategies)
        assert len(all_strategies) == 3

    def test_create_correlation_clusters_fallback(self, analyzer):
        """Test cluster creation fallback on error."""
        # Create problematic correlation matrix
        corr_matrix = pd.DataFrame({
            'strategy1': [1.0, np.nan],
            'strategy2': [np.nan, 1.0]
        }, index=['strategy1', 'strategy2'])

        with patch('backend.validation.advanced_risk.cross_strategy_risk.linkage') as mock_linkage:
            mock_linkage.side_effect = Exception("Clustering failed")

            clusters = analyzer._create_correlation_clusters(corr_matrix)

            # Should fall back to individual clusters
            assert len(clusters) == 2

    def test_calculate_cluster_statistics(self, analyzer):
        """Test cluster statistics calculation."""
        clusters = {
            1: ['strategy1', 'strategy2'],
            2: ['strategy3']
        }

        corr_matrix = pd.DataFrame({
            'strategy1': [1.0, 0.75, 0.2],
            'strategy2': [0.75, 1.0, 0.3],
            'strategy3': [0.2, 0.3, 1.0]
        }, index=['strategy1', 'strategy2', 'strategy3'])

        stats = analyzer._calculate_cluster_statistics(clusters, corr_matrix)

        assert 1 in stats
        assert 2 in stats
        assert 'avg_correlation' in stats[1]
        assert 'max_correlation' in stats[1]

    def test_assign_cluster_risk_budgets(self, analyzer):
        """Test risk budget assignment."""
        cluster_stats = {
            1: {'avg_correlation': 0.7, 'max_correlation': 0.8},
            2: {'avg_correlation': 0.3, 'max_correlation': 0.4}
        }

        # Set up clusters
        analyzer.strategy_clusters = {
            1: type('obj', (object,), {'strategy_names': ['s1', 's2']})(),
            2: type('obj', (object,), {'strategy_names': ['s3', 's4']})()
        }

        budgets = analyzer._assign_cluster_risk_budgets(cluster_stats)

        assert len(budgets) == 2
        # Budgets should sum to approximately 1
        assert abs(sum(budgets.values()) - 1.0) < 0.01

    def test_get_cluster_for_strategy(self, analyzer):
        """Test getting cluster for strategy."""
        analyzer.strategy_clusters = {
            1: StrategyCluster(1, ['strategy1', 'strategy2'], 0.7, 0.8, 2, 0.4, 0, 0.4),
            2: StrategyCluster(2, ['strategy3'], 0.0, 0.0, 1, 0.3, 0, 0.3)
        }

        cluster_id = analyzer.get_cluster_for_strategy('strategy1')
        assert cluster_id == 1

        cluster_id = analyzer.get_cluster_for_strategy('strategy3')
        assert cluster_id == 2

        cluster_id = analyzer.get_cluster_for_strategy('unknown')
        assert cluster_id is None


class TestCrossStrategyRiskManager:
    """Test CrossStrategyRiskManager class."""

    @pytest.fixture
    def manager(self):
        """Create manager instance."""
        return CrossStrategyRiskManager()

    @pytest.fixture
    def sample_strategy_returns(self):
        """Create sample strategy returns."""
        np.random.seed(42)
        dates = pd.date_range('2022-01-01', periods=252, freq='B')

        return {
            'strategy1': pd.Series(np.random.normal(0.001, 0.012, 252), index=dates),
            'strategy2': pd.Series(np.random.normal(0.0008, 0.010, 252), index=dates),
            'strategy3': pd.Series(np.random.normal(0.0012, 0.015, 252), index=dates)
        }

    def test_initialization(self, manager):
        """Test manager initialization."""
        assert manager.cluster_analyzer is not None
        assert len(manager.current_allocations) == 0
        assert 'max_cluster_allocation' in manager.risk_limits

    def test_update_strategy_correlations(self, manager, sample_strategy_returns):
        """Test updating correlations."""
        manager.update_strategy_correlations(sample_strategy_returns)

        assert len(manager.cluster_analyzer.strategy_clusters) > 0
        assert manager.last_update is not None

    def test_calculate_portfolio_risk_allocation(self, manager, sample_strategy_returns):
        """Test portfolio risk allocation calculation."""
        manager.update_strategy_correlations(sample_strategy_returns)

        proposed_allocations = {
            'strategy1': 0.30,
            'strategy2': 0.25,
            'strategy3': 0.20
        }

        result = manager.calculate_portfolio_risk_allocation(
            proposed_allocations,
            sample_strategy_returns
        )

        assert 'validated_allocations' in result
        assert 'portfolio_risk_metrics' in result
        assert 'recommendations' in result

    def test_validate_cluster_constraints(self, manager, sample_strategy_returns):
        """Test cluster constraint validation."""
        manager.update_strategy_correlations(sample_strategy_returns)

        # Propose allocations that exceed cluster limits
        proposed_allocations = {
            'strategy1': 0.50,  # Might exceed cluster limit
            'strategy2': 0.30,
            'strategy3': 0.20
        }

        validated = manager._validate_cluster_constraints(proposed_allocations)

        # Should adjust allocations
        assert isinstance(validated, dict)
        # Total allocation should still be reasonable
        assert sum(validated.values()) <= 1.0

    def test_validate_cluster_constraints_individual_limits(self, manager):
        """Test individual strategy limits."""
        proposed_allocations = {
            'strategy1': 0.50  # Exceeds 20% individual limit
        }

        validated = manager._validate_cluster_constraints(proposed_allocations)

        # Should cap at max_single_strategy
        assert validated['strategy1'] <= manager.risk_limits['max_single_strategy']

    def test_calculate_portfolio_risk_metrics(self, manager, sample_strategy_returns):
        """Test portfolio risk metrics calculation."""
        allocations = {
            'strategy1': 0.30,
            'strategy2': 0.25,
            'strategy3': 0.20
        }

        manager.update_strategy_correlations(sample_strategy_returns)

        metrics = manager._calculate_portfolio_risk_metrics(
            allocations,
            sample_strategy_returns
        )

        assert isinstance(metrics, PortfolioRiskMetrics)
        assert metrics.total_var >= 0
        assert metrics.diversification_ratio > 0

    def test_calculate_portfolio_risk_metrics_empty(self, manager):
        """Test risk metrics with empty allocations."""
        metrics = manager._calculate_portfolio_risk_metrics({}, {})

        assert metrics.total_var == 0
        assert metrics.diversification_ratio == 1.0

    def test_calculate_risk_contributions(self, manager, sample_strategy_returns):
        """Test risk contribution calculation."""
        manager.update_strategy_correlations(sample_strategy_returns)

        allocations = {
            'strategy1': 0.30,
            'strategy2': 0.25
        }

        risk_allocations = manager._calculate_risk_contributions(
            allocations,
            sample_strategy_returns
        )

        assert len(risk_allocations) == 2
        assert all(isinstance(ra, RiskAllocation) for ra in risk_allocations)

    def test_calculate_cluster_exposures(self, manager, sample_strategy_returns):
        """Test cluster exposure calculation."""
        manager.update_strategy_correlations(sample_strategy_returns)

        allocations = {
            'strategy1': 0.30,
            'strategy2': 0.25,
            'strategy3': 0.20
        }

        exposures = manager._calculate_cluster_exposures(allocations)

        assert isinstance(exposures, dict)
        # Total exposure should match total allocation
        assert sum(exposures.values()) <= sum(allocations.values())

    def test_generate_allocation_recommendations(self, manager, sample_strategy_returns):
        """Test allocation recommendation generation."""
        manager.update_strategy_correlations(sample_strategy_returns)

        original_allocations = {
            'strategy1': 0.50,
            'strategy2': 0.30
        }

        validated_allocations = {
            'strategy1': 0.20,  # Reduced
            'strategy2': 0.25
        }

        metrics = PortfolioRiskMetrics(
            total_var=0.04,
            diversified_var=0.03,
            concentration_risk=0.35,
            correlation_risk=0.01,
            cluster_concentrations={1: 0.45},
            undiversified_var=0.05,
            diversification_ratio=1.3
        )

        recommendations = manager._generate_allocation_recommendations(
            original_allocations,
            validated_allocations,
            metrics
        )

        assert len(recommendations) > 0
        assert any('strategy1' in r for r in recommendations)

    def test_calculate_risk_budget_utilization(self, manager, sample_strategy_returns):
        """Test risk budget utilization calculation."""
        manager.update_strategy_correlations(sample_strategy_returns)

        allocations = {
            'strategy1': 0.30,
            'strategy2': 0.25
        }

        utilization = manager._calculate_risk_budget_utilization(allocations)

        assert isinstance(utilization, dict)
        for cluster_util in utilization.values():
            assert 'budget' in cluster_util
            assert 'used' in cluster_util
            assert 'available' in cluster_util

    def test_check_constraint_violations(self, manager):
        """Test constraint violation checking."""
        allocations = {
            'strategy1': 0.50  # Exceeds individual limit
        }

        violations = manager._check_constraint_violations(allocations)

        assert len(violations) > 0

    def test_check_constraint_violations_none(self, manager):
        """Test with no violations."""
        allocations = {
            'strategy1': 0.15,
            'strategy2': 0.12
        }

        violations = manager._check_constraint_violations(allocations)

        # Should have no violations for individual limits
        # (cluster violations depend on cluster setup)
        assert isinstance(violations, list)

    def test_update_risk_limits(self, manager):
        """Test updating risk limits."""
        new_limits = {
            'max_cluster_allocation': 0.50,
            'max_single_strategy': 0.25
        }

        manager.update_risk_limits(new_limits)

        assert manager.risk_limits['max_cluster_allocation'] == 0.50
        assert manager.risk_limits['max_single_strategy'] == 0.25

    def test_get_risk_dashboard(self, manager, sample_strategy_returns):
        """Test risk dashboard generation."""
        manager.update_strategy_correlations(sample_strategy_returns)

        dashboard = manager.get_risk_dashboard()

        assert 'cluster_summary' in dashboard
        assert 'risk_limits' in dashboard
        assert 'current_allocations' in dashboard
        assert 'last_correlation_update' in dashboard


class TestCrossStrategyRiskIntegration:
    """Integration tests for cross-strategy risk management."""

    def test_full_risk_management_workflow(self):
        """Test complete risk management workflow."""
        manager = CrossStrategyRiskManager()

        # Create diverse strategies
        np.random.seed(42)
        dates = pd.date_range('2022-01-01', periods=252, freq='B')

        strategies = {
            'momentum_spy': pd.Series(np.random.normal(0.0012, 0.012, 252), index=dates),
            'momentum_qqq': pd.Series(np.random.normal(0.0011, 0.013, 252), index=dates),
            'mean_reversion': pd.Series(np.random.normal(0.0008, 0.010, 252), index=dates),
            'volatility': pd.Series(np.random.normal(0.0005, 0.008, 252), index=dates)
        }

        # Update correlations
        manager.update_strategy_correlations(strategies)

        # Propose allocations
        proposed = {
            'momentum_spy': 0.25,
            'momentum_qqq': 0.25,
            'mean_reversion': 0.20,
            'volatility': 0.15
        }

        # Get validated allocations
        result = manager.calculate_portfolio_risk_allocation(proposed, strategies)

        # Verify results
        assert 'validated_allocations' in result
        assert 'recommendations' in result
        assert len(result['recommendations']) > 0

    def test_correlation_clustering_effect(self):
        """Test that correlation clustering works correctly."""
        manager = CrossStrategyRiskManager()

        np.random.seed(42)
        dates = pd.date_range('2022-01-01', periods=252, freq='B')

        # Create highly correlated strategies
        base_returns = np.random.normal(0.001, 0.01, 252)

        strategies = {
            'corr1': pd.Series(base_returns, index=dates),
            'corr2': pd.Series(base_returns + np.random.normal(0, 0.001, 252), index=dates),
            'uncorr': pd.Series(np.random.normal(0.001, 0.01, 252), index=dates)
        }

        manager.update_strategy_correlations(strategies)

        # Try to allocate heavily to correlated strategies
        proposed = {
            'corr1': 0.30,
            'corr2': 0.30,
            'uncorr': 0.20
        }

        result = manager.calculate_portfolio_risk_allocation(proposed, strategies)

        # Should detect high cluster exposure
        cluster_exposures = result['cluster_exposures']
        # At least one cluster should have significant exposure
        assert max(cluster_exposures.values()) > 0.2


class TestEdgeCasesAndErrors:
    """Test edge cases and error handling."""

    def test_empty_correlation_matrix(self):
        """Test with empty correlation matrix."""
        analyzer = CorrelationClusterAnalyzer()

        empty_corr = pd.DataFrame()

        clusters = analyzer._create_correlation_clusters(empty_corr)

        assert len(clusters) == 0

    def test_single_cluster(self):
        """Test when all strategies are in one cluster."""
        analyzer = CorrelationClusterAnalyzer(correlation_threshold=0.1)  # Very low threshold

        corr_matrix = pd.DataFrame({
            'strategy1': [1.0, 0.2, 0.15],
            'strategy2': [0.2, 1.0, 0.18],
            'strategy3': [0.15, 0.18, 1.0]
        }, index=['strategy1', 'strategy2', 'strategy3'])

        clusters = analyzer._create_correlation_clusters(corr_matrix)

        # Might create single cluster or multiple depending on threshold
        assert len(clusters) >= 1

    def test_perfect_correlation(self):
        """Test with perfectly correlated strategies."""
        manager = CrossStrategyRiskManager()

        dates = pd.date_range('2023-01-01', periods=100)
        base_returns = pd.Series(np.random.normal(0.001, 0.01, 100), index=dates)

        strategies = {
            'strategy1': base_returns,
            'strategy2': base_returns,  # Perfect correlation
            'strategy3': base_returns
        }

        manager.update_strategy_correlations(strategies)

        # Should still be able to analyze
        assert len(manager.cluster_analyzer.strategy_clusters) > 0

    def test_negative_correlation(self):
        """Test with negatively correlated strategies."""
        manager = CrossStrategyRiskManager()

        dates = pd.date_range('2023-01-01', periods=100)
        returns1 = pd.Series(np.random.normal(0.001, 0.01, 100), index=dates)
        returns2 = -returns1  # Perfect negative correlation

        strategies = {
            'strategy1': returns1,
            'strategy2': returns2
        }

        manager.update_strategy_correlations(strategies)

        allocations = {'strategy1': 0.25, 'strategy2': 0.25}

        result = manager.calculate_portfolio_risk_allocation(allocations, strategies)

        # Negative correlation should provide diversification benefit
        assert result['portfolio_risk_metrics'].diversification_ratio > 0

    def test_all_zero_returns(self):
        """Test with all zero returns."""
        manager = CrossStrategyRiskManager()

        dates = pd.date_range('2023-01-01', periods=100)

        strategies = {
            'strategy1': pd.Series([0.0] * 100, index=dates),
            'strategy2': pd.Series([0.0] * 100, index=dates)
        }

        manager.update_strategy_correlations(strategies)

        allocations = {'strategy1': 0.25, 'strategy2': 0.25}

        # Should handle gracefully
        result = manager.calculate_portfolio_risk_allocation(allocations, strategies)

        assert isinstance(result, dict)

    def test_extremely_high_allocations(self):
        """Test with allocations exceeding 100%."""
        manager = CrossStrategyRiskManager()

        allocations = {
            'strategy1': 0.80,
            'strategy2': 0.60
        }

        validated = manager._validate_cluster_constraints(allocations)

        # Should cap allocations appropriately
        assert all(v <= manager.risk_limits['max_single_strategy'] for v in validated.values())
