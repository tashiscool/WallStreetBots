"""Cross-strategy risk management and correlation clustering.

Implements hard caps per correlation cluster to prevent stacking
the same bet across multiple strategies.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Set
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import warnings

logger = logging.getLogger(__name__)


@dataclass
class StrategyCluster:
    """Represents a cluster of correlated strategies."""
    cluster_id: int
    strategy_names: List[str]
    avg_correlation: float
    max_correlation: float
    cluster_size: int
    risk_budget: float
    current_allocation: float
    available_capacity: float


@dataclass
class RiskAllocation:
    """Risk allocation for a strategy."""
    strategy_name: str
    cluster_id: int
    position_size: float
    risk_contribution: float
    marginal_var: float
    component_var: float
    correlation_penalty: float


@dataclass
class PortfolioRiskMetrics:
    """Portfolio-level risk metrics."""
    total_var: float
    diversified_var: float
    concentration_risk: float
    correlation_risk: float
    cluster_concentrations: Dict[int, float]
    undiversified_var: float
    diversification_ratio: float


class CorrelationClusterAnalyzer:
    """Analyzes strategy correlations and creates risk clusters."""

    def __init__(self, correlation_threshold: float = 0.6):
        self.correlation_threshold = correlation_threshold
        self.strategy_clusters = {}
        self.cluster_risk_budgets = {}
        self.correlation_matrix = None

    def analyze_strategy_correlations(self, strategy_returns: Dict[str, pd.Series],
                                    lookback_days: int = 252) -> Dict[int, StrategyCluster]:
        """Analyze correlations and create strategy clusters.

        Args:
            strategy_returns: Dictionary of strategy name -> return series
            lookback_days: Number of days to use for correlation calculation

        Returns:
            Dictionary of cluster_id -> StrategyCluster
        """
        if len(strategy_returns) < 2:
            logger.warning("Need at least 2 strategies for correlation analysis")
            return {}

        # Align return series and get recent data
        aligned_returns = self._align_returns(strategy_returns, lookback_days)

        if aligned_returns.empty:
            logger.error("No aligned return data available")
            return {}

        # Calculate correlation matrix
        self.correlation_matrix = aligned_returns.corr()

        # Create clusters using hierarchical clustering
        clusters = self._create_correlation_clusters(self.correlation_matrix)

        # Calculate cluster statistics
        cluster_stats = self._calculate_cluster_statistics(clusters, self.correlation_matrix)

        # Assign risk budgets
        cluster_risk_budgets = self._assign_cluster_risk_budgets(cluster_stats)

        # Create StrategyCluster objects
        strategy_clusters = {}
        for cluster_id, strategies in clusters.items():
            stats = cluster_stats[cluster_id]
            risk_budget = cluster_risk_budgets[cluster_id]

            strategy_clusters[cluster_id] = StrategyCluster(
                cluster_id=cluster_id,
                strategy_names=strategies,
                avg_correlation=stats['avg_correlation'],
                max_correlation=stats['max_correlation'],
                cluster_size=len(strategies),
                risk_budget=risk_budget,
                current_allocation=0.0,  # To be updated later
                available_capacity=risk_budget
            )

        self.strategy_clusters = strategy_clusters
        self.cluster_risk_budgets = cluster_risk_budgets

        logger.info(f"Created {len(strategy_clusters)} strategy clusters")
        return strategy_clusters

    def _align_returns(self, strategy_returns: Dict[str, pd.Series],
                      lookback_days: int) -> pd.DataFrame:
        """Align return series to common dates."""
        # Combine all series
        combined_df = pd.DataFrame(strategy_returns)

        # Get recent data
        if len(combined_df) > lookback_days:
            combined_df = combined_df.tail(lookback_days)

        # Remove rows with any NaN values
        clean_df = combined_df.dropna()

        if len(clean_df) < 30:
            logger.warning(f"Only {len(clean_df)} clean observations available")

        return clean_df

    def _create_correlation_clusters(self, correlation_matrix: pd.DataFrame) -> Dict[int, List[str]]:
        """Create strategy clusters using hierarchical clustering."""
        if correlation_matrix.empty or len(correlation_matrix) < 2:
            return {}

        try:
            # Convert correlation to distance
            distance_matrix = 1 - abs(correlation_matrix.values)

            # Handle NaN values
            distance_matrix = np.nan_to_num(distance_matrix, nan=1.0)

            # Ensure distance matrix is symmetric and has zero diagonal
            np.fill_diagonal(distance_matrix, 0)
            distance_matrix = (distance_matrix + distance_matrix.T) / 2

            # Convert to condensed form for linkage
            condensed_distances = squareform(distance_matrix, checks=False)

            # Perform hierarchical clustering
            linkage_matrix = linkage(condensed_distances, method='average')

            # Form clusters based on distance threshold
            distance_threshold = 1 - self.correlation_threshold
            cluster_labels = fcluster(linkage_matrix, distance_threshold, criterion='distance')

            # Group strategies by cluster
            clusters = {}
            for i, strategy in enumerate(correlation_matrix.index):
                cluster_id = cluster_labels[i]
                if cluster_id not in clusters:
                    clusters[cluster_id] = []
                clusters[cluster_id].append(strategy)

            return clusters

        except Exception as e:
            logger.error(f"Clustering failed: {e}")
            # Fallback: each strategy is its own cluster
            return {i + 1: [strategy] for i, strategy in enumerate(correlation_matrix.index)}

    def _calculate_cluster_statistics(self, clusters: Dict[int, List[str]],
                                    correlation_matrix: pd.DataFrame) -> Dict[int, Dict]:
        """Calculate statistics for each cluster."""
        cluster_stats = {}

        for cluster_id, strategies in clusters.items():
            if len(strategies) == 1:
                cluster_stats[cluster_id] = {
                    'avg_correlation': 0.0,
                    'max_correlation': 0.0,
                    'min_correlation': 0.0,
                    'correlation_std': 0.0
                }
            else:
                # Get correlation submatrix for this cluster
                cluster_corr = correlation_matrix.loc[strategies, strategies]

                # Calculate statistics (excluding diagonal)
                upper_triangle = np.triu(cluster_corr.values, k=1)
                correlations = upper_triangle[upper_triangle != 0]

                if len(correlations) > 0:
                    cluster_stats[cluster_id] = {
                        'avg_correlation': np.mean(correlations),
                        'max_correlation': np.max(correlations),
                        'min_correlation': np.min(correlations),
                        'correlation_std': np.std(correlations)
                    }
                else:
                    cluster_stats[cluster_id] = {
                        'avg_correlation': 0.0,
                        'max_correlation': 0.0,
                        'min_correlation': 0.0,
                        'correlation_std': 0.0
                    }

        return cluster_stats

    def _assign_cluster_risk_budgets(self, cluster_stats: Dict[int, Dict]) -> Dict[int, float]:
        """Assign risk budgets to clusters based on diversification benefit."""
        total_budget = 1.0  # 100% of risk budget
        cluster_risk_budgets = {}

        if not cluster_stats:
            return {}

        # Calculate base allocation (equal weight)
        num_clusters = len(cluster_stats)
        base_allocation = total_budget / num_clusters

        # Adjust based on cluster size and correlation
        adjusted_budgets = {}
        total_adjustment_weight = 0

        for cluster_id, stats in cluster_stats.items():
            # Clusters with lower correlation get higher budget
            correlation_discount = 1 - stats['avg_correlation']

            # Larger clusters get slightly higher budget (diversification within cluster)
            cluster_size = len(self.strategy_clusters.get(cluster_id, {}).get('strategy_names', [1]))
            size_bonus = min(1.2, 1 + 0.05 * (cluster_size - 1))

            # Combined adjustment weight
            adjustment_weight = correlation_discount * size_bonus
            adjusted_budgets[cluster_id] = adjustment_weight
            total_adjustment_weight += adjustment_weight

        # Normalize to total budget
        if total_adjustment_weight > 0:
            for cluster_id in adjusted_budgets:
                cluster_risk_budgets[cluster_id] = (
                    adjusted_budgets[cluster_id] / total_adjustment_weight * total_budget
                )
        else:
            # Fallback to equal allocation
            for cluster_id in cluster_stats:
                cluster_risk_budgets[cluster_id] = base_allocation

        return cluster_risk_budgets

    def get_cluster_for_strategy(self, strategy_name: str) -> Optional[int]:
        """Get cluster ID for a strategy."""
        for cluster_id, cluster in self.strategy_clusters.items():
            if strategy_name in cluster.strategy_names:
                return cluster_id
        return None


class CrossStrategyRiskManager:
    """Manages risk across multiple strategies with cluster-based limits."""

    def __init__(self):
        self.cluster_analyzer = CorrelationClusterAnalyzer()
        self.current_allocations = {}  # strategy_name -> allocation
        self.risk_limits = {
            'max_cluster_allocation': 0.40,    # 40% max per cluster
            'max_single_strategy': 0.20,       # 20% max per strategy
            'max_correlation_exposure': 0.60,  # 60% max in correlated strategies
            'min_diversification_ratio': 1.5   # Minimum diversification benefit
        }
        self.portfolio_var = 0.0
        self.last_update = None

    def update_strategy_correlations(self, strategy_returns: Dict[str, pd.Series]):
        """Update correlation analysis and cluster assignments."""
        self.cluster_analyzer.analyze_strategy_correlations(strategy_returns)
        self.last_update = datetime.now()
        logger.info("Updated strategy correlation clusters")

    def calculate_portfolio_risk_allocation(self,
                                          proposed_allocations: Dict[str, float],
                                          strategy_returns: Dict[str, pd.Series]) -> Dict[str, Any]:
        """Calculate optimal risk allocation across strategies with cluster constraints.

        Args:
            proposed_allocations: Dictionary of strategy_name -> proposed allocation
            strategy_returns: Dictionary of strategy_name -> return series

        Returns:
            Dictionary with allocation recommendations and risk analysis
        """
        # Update correlations if needed
        if not self.cluster_analyzer.strategy_clusters:
            self.update_strategy_correlations(strategy_returns)

        # Validate and adjust allocations
        validated_allocations = self._validate_cluster_constraints(proposed_allocations)

        # Calculate portfolio risk metrics
        portfolio_risk = self._calculate_portfolio_risk_metrics(
            validated_allocations, strategy_returns
        )

        # Generate risk allocations
        risk_allocations = self._calculate_risk_contributions(
            validated_allocations, strategy_returns
        )

        # Create recommendations
        recommendations = self._generate_allocation_recommendations(
            proposed_allocations, validated_allocations, portfolio_risk
        )

        return {
            'validated_allocations': validated_allocations,
            'original_allocations': proposed_allocations,
            'portfolio_risk_metrics': portfolio_risk,
            'risk_allocations': risk_allocations,
            'cluster_exposures': self._calculate_cluster_exposures(validated_allocations),
            'recommendations': recommendations,
            'risk_budget_utilization': self._calculate_risk_budget_utilization(validated_allocations),
            'constraint_violations': self._check_constraint_violations(validated_allocations)
        }

    def _validate_cluster_constraints(self, proposed_allocations: Dict[str, float]) -> Dict[str, float]:
        """Validate and adjust allocations to respect cluster constraints."""
        validated_allocations = proposed_allocations.copy()

        # Check cluster allocation limits
        cluster_allocations = {}
        for strategy_name, allocation in validated_allocations.items():
            cluster_id = self.cluster_analyzer.get_cluster_for_strategy(strategy_name)
            if cluster_id is not None:
                if cluster_id not in cluster_allocations:
                    cluster_allocations[cluster_id] = 0
                cluster_allocations[cluster_id] += allocation

        # Adjust allocations if cluster limits are exceeded
        for cluster_id, total_allocation in cluster_allocations.items():
            max_cluster_allocation = self.risk_limits['max_cluster_allocation']

            if total_allocation > max_cluster_allocation:
                # Scale down all strategies in this cluster proportionally
                cluster = self.cluster_analyzer.strategy_clusters.get(cluster_id)
                if cluster:
                    scale_factor = max_cluster_allocation / total_allocation

                    for strategy_name in cluster.strategy_names:
                        if strategy_name in validated_allocations:
                            validated_allocations[strategy_name] *= scale_factor

                    logger.warning(f"Scaled down cluster {cluster_id} allocation by {scale_factor:.2f}")

        # Check individual strategy limits
        max_strategy_allocation = self.risk_limits['max_single_strategy']
        for strategy_name, allocation in validated_allocations.items():
            if allocation > max_strategy_allocation:
                validated_allocations[strategy_name] = max_strategy_allocation
                logger.warning(f"Capped {strategy_name} allocation at {max_strategy_allocation:.1%}")

        return validated_allocations

    def _calculate_portfolio_risk_metrics(self, allocations: Dict[str, float],
                                        strategy_returns: Dict[str, pd.Series]) -> PortfolioRiskMetrics:
        """Calculate comprehensive portfolio risk metrics."""
        if not allocations or not strategy_returns:
            return PortfolioRiskMetrics(
                total_var=0, diversified_var=0, concentration_risk=0,
                correlation_risk=0, cluster_concentrations={},
                undiversified_var=0, diversification_ratio=1.0
            )

        # Align returns and get covariance matrix
        aligned_returns = self.cluster_analyzer._align_returns(strategy_returns, 252)

        if aligned_returns.empty:
            return PortfolioRiskMetrics(
                total_var=0, diversified_var=0, concentration_risk=0,
                correlation_risk=0, cluster_concentrations={},
                undiversified_var=0, diversification_ratio=1.0
            )

        # Filter to strategies with allocations
        strategy_names = [name for name in allocations.keys() if name in aligned_returns.columns]
        if not strategy_names:
            return PortfolioRiskMetrics(
                total_var=0, diversified_var=0, concentration_risk=0,
                correlation_risk=0, cluster_concentrations={},
                undiversified_var=0, diversification_ratio=1.0
            )

        strategy_returns_filtered = aligned_returns[strategy_names]
        weights = np.array([allocations[name] for name in strategy_names])

        # Calculate covariance matrix
        cov_matrix = strategy_returns_filtered.cov().values * 252  # Annualize

        # Portfolio variance
        portfolio_var = np.dot(weights, np.dot(cov_matrix, weights))

        # Undiversified variance (sum of individual variances)
        individual_vars = np.diag(cov_matrix)
        undiversified_var = np.dot(weights**2, individual_vars)

        # Concentration risk (Herfindahl index)
        concentration_risk = np.sum(weights**2)

        # Correlation risk (portfolio var - undiversified var)
        correlation_risk = portfolio_var - undiversified_var

        # Diversification ratio
        weighted_vol = np.dot(weights, np.sqrt(individual_vars))
        portfolio_vol = np.sqrt(portfolio_var)
        diversification_ratio = weighted_vol / portfolio_vol if portfolio_vol > 0 else 1.0

        # Cluster concentrations
        cluster_concentrations = self._calculate_cluster_exposures(allocations)

        return PortfolioRiskMetrics(
            total_var=portfolio_var,
            diversified_var=portfolio_var,
            concentration_risk=concentration_risk,
            correlation_risk=correlation_risk,
            cluster_concentrations=cluster_concentrations,
            undiversified_var=undiversified_var,
            diversification_ratio=diversification_ratio
        )

    def _calculate_risk_contributions(self, allocations: Dict[str, float],
                                    strategy_returns: Dict[str, pd.Series]) -> List[RiskAllocation]:
        """Calculate risk contributions for each strategy."""
        risk_allocations = []

        for strategy_name, allocation in allocations.items():
            cluster_id = self.cluster_analyzer.get_cluster_for_strategy(strategy_name)

            # Simplified risk contribution calculation
            if strategy_name in strategy_returns:
                strategy_vol = strategy_returns[strategy_name].std() * np.sqrt(252)
                risk_contribution = allocation * strategy_vol

                # Calculate correlation penalty
                correlation_penalty = 0
                if cluster_id and cluster_id in self.cluster_analyzer.strategy_clusters:
                    cluster = self.cluster_analyzer.strategy_clusters[cluster_id]
                    correlation_penalty = cluster.avg_correlation * allocation

                risk_allocations.append(RiskAllocation(
                    strategy_name=strategy_name,
                    cluster_id=cluster_id or 0,
                    position_size=allocation,
                    risk_contribution=risk_contribution,
                    marginal_var=risk_contribution,  # Simplified
                    component_var=risk_contribution,  # Simplified
                    correlation_penalty=correlation_penalty
                ))

        return risk_allocations

    def _calculate_cluster_exposures(self, allocations: Dict[str, float]) -> Dict[int, float]:
        """Calculate total exposure per cluster."""
        cluster_exposures = {}

        for strategy_name, allocation in allocations.items():
            cluster_id = self.cluster_analyzer.get_cluster_for_strategy(strategy_name)
            if cluster_id is not None:
                if cluster_id not in cluster_exposures:
                    cluster_exposures[cluster_id] = 0
                cluster_exposures[cluster_id] += allocation

        return cluster_exposures

    def _generate_allocation_recommendations(self, original_allocations: Dict[str, float],
                                           validated_allocations: Dict[str, float],
                                           portfolio_risk: PortfolioRiskMetrics) -> List[str]:
        """Generate allocation recommendations based on risk analysis."""
        recommendations = []

        # Check if allocations were adjusted
        for strategy_name, original_alloc in original_allocations.items():
            validated_alloc = validated_allocations.get(strategy_name, 0)
            if abs(original_alloc - validated_alloc) > 0.01:  # 1% threshold
                recommendations.append(
                    f"Reduced {strategy_name} allocation from {original_alloc:.1%} "
                    f"to {validated_alloc:.1%} due to risk constraints"
                )

        # Diversification recommendations
        if portfolio_risk.diversification_ratio < self.risk_limits['min_diversification_ratio']:
            recommendations.append(
                f"Portfolio diversification ratio ({portfolio_risk.diversification_ratio:.2f}) "
                f"below target ({self.risk_limits['min_diversification_ratio']:.2f}). "
                "Consider adding uncorrelated strategies."
            )

        # Concentration warnings
        if portfolio_risk.concentration_risk > 0.3:
            recommendations.append(
                f"High concentration risk ({portfolio_risk.concentration_risk:.2f}). "
                "Consider more balanced allocation across strategies."
            )

        # Cluster concentration warnings
        for cluster_id, exposure in portfolio_risk.cluster_concentrations.items():
            if exposure > self.risk_limits['max_cluster_allocation']:
                cluster = self.cluster_analyzer.strategy_clusters.get(cluster_id)
                if cluster:
                    recommendations.append(
                        f"Cluster {cluster_id} exposure ({exposure:.1%}) exceeds limit "
                        f"({self.risk_limits['max_cluster_allocation']:.1%}). "
                        f"Strategies: {', '.join(cluster.strategy_names)}"
                    )

        if not recommendations:
            recommendations.append("Allocation meets all risk constraints and diversification targets.")

        return recommendations

    def _calculate_risk_budget_utilization(self, allocations: Dict[str, float]) -> Dict[int, Dict[str, float]]:
        """Calculate how much of each cluster's risk budget is utilized."""
        utilization = {}

        cluster_exposures = self._calculate_cluster_exposures(allocations)

        for cluster_id, cluster in self.cluster_analyzer.strategy_clusters.items():
            current_exposure = cluster_exposures.get(cluster_id, 0)
            budget = cluster.risk_budget

            utilization[cluster_id] = {
                'budget': budget,
                'used': current_exposure,
                'available': max(0, budget - current_exposure),
                'utilization_pct': current_exposure / budget if budget > 0 else 0
            }

        return utilization

    def _check_constraint_violations(self, allocations: Dict[str, float]) -> List[str]:
        """Check for any constraint violations."""
        violations = []

        # Check individual strategy limits
        for strategy_name, allocation in allocations.items():
            if allocation > self.risk_limits['max_single_strategy']:
                violations.append(
                    f"{strategy_name} allocation ({allocation:.1%}) exceeds "
                    f"single strategy limit ({self.risk_limits['max_single_strategy']:.1%})"
                )

        # Check cluster limits
        cluster_exposures = self._calculate_cluster_exposures(allocations)
        for cluster_id, exposure in cluster_exposures.items():
            if exposure > self.risk_limits['max_cluster_allocation']:
                violations.append(
                    f"Cluster {cluster_id} exposure ({exposure:.1%}) exceeds "
                    f"cluster limit ({self.risk_limits['max_cluster_allocation']:.1%})"
                )

        return violations

    def update_risk_limits(self, new_limits: Dict[str, float]):
        """Update risk limits configuration."""
        self.risk_limits.update(new_limits)
        logger.info(f"Updated risk limits: {self.risk_limits}")

    def get_risk_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive risk dashboard."""
        return {
            'cluster_summary': {
                cluster_id: {
                    'strategies': cluster.strategy_names,
                    'avg_correlation': cluster.avg_correlation,
                    'risk_budget': cluster.risk_budget,
                    'current_allocation': cluster.current_allocation
                }
                for cluster_id, cluster in self.cluster_analyzer.strategy_clusters.items()
            },
            'risk_limits': self.risk_limits,
            'current_allocations': self.current_allocations,
            'last_correlation_update': self.last_update.isoformat() if self.last_update else None,
            'portfolio_var': self.portfolio_var
        }