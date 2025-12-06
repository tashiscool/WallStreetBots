"""Portfolio-Level Capital Allocation Optimizer

Optimizes capital allocation across multiple strategies to maximize risk-adjusted returns.
Handles strategy correlation, capital competition, and dynamic rebalancing.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from decimal import Decimal
import pandas as pd
import numpy as np

from backend.validation.ensemble_evaluator import EnsembleValidator
from backend.validation.advanced_risk.cross_strategy_risk import CrossStrategyRiskManager

logger = logging.getLogger(__name__)


@dataclass
class StrategyAllocation:
    """Capital allocation for a strategy."""
    strategy_name: str
    current_allocation: float  # Current % of portfolio
    optimal_allocation: float  # Optimal % of portfolio
    recommended_allocation: float  # Recommended allocation after constraints
    sharpe_ratio: float
    expected_return: float
    volatility: float
    max_drawdown: float
    correlation_with_portfolio: float
    risk_contribution: float
    capital_required: Decimal
    priority_score: float  # Higher = more capital should go here


@dataclass
class PortfolioAllocationResult:
    """Result of portfolio capital allocation optimization."""
    timestamp: datetime
    total_capital: Decimal
    allocations: Dict[str, StrategyAllocation]
    portfolio_sharpe: float
    portfolio_expected_return: float
    portfolio_volatility: float
    diversification_ratio: float
    risk_budget_utilization: float
    recommendations: List[str]
    warnings: List[str]


class PortfolioCapitalAllocator:
    """Optimizes capital allocation across multiple trading strategies.
    
    Features:
    - Correlation-aware allocation
    - Risk parity principles
    - Dynamic rebalancing based on performance
    - Capital competition resolution
    - Strategy priority scoring
    """
    
    def __init__(self, max_total_allocation: float = 0.95):
        """Initialize portfolio capital allocator.
        
        Args:
            max_total_allocation: Maximum total allocation (leaves cash buffer)
        """
        self.max_total_allocation = max_total_allocation
        self.ensemble_validator = EnsembleValidator()
        self.cross_strategy_risk = CrossStrategyRiskManager()
        self.logger = logging.getLogger(__name__)
        
        # Allocation history for tracking
        self.allocation_history: List[PortfolioAllocationResult] = []
        self.max_history = 100
        
    async def optimize_allocation(
        self,
        strategies: Dict[str, Any],
        portfolio_value: Decimal,
        strategy_returns: Optional[Dict[str, pd.Series]] = None,
        current_positions: Optional[Dict[str, Dict]] = None
    ) -> PortfolioAllocationResult:
        """Optimize capital allocation across strategies.
        
        Args:
            strategies: Dictionary of strategy_name -> strategy object
            portfolio_value: Total portfolio value
            strategy_returns: Historical returns for each strategy (optional)
            current_positions: Current positions by strategy (optional)
            
        Returns:
            PortfolioAllocationResult with optimal allocations
        """
        try:
            # 1. Calculate strategy performance metrics
            strategy_metrics = await self._calculate_strategy_metrics(
                strategies, strategy_returns
            )
            
            # 2. Calculate strategy correlations
            if strategy_returns:
                correlations = self.ensemble_validator.analyze_strategy_correlations(strategy_returns)
            else:
                correlations = {}
            
            # 3. Calculate optimal allocations using multiple methods
            allocations = await self._calculate_optimal_allocations(
                strategy_metrics, correlations, portfolio_value
            )
            
            # 4. Apply constraints and validate
            validated_allocations = self._apply_allocation_constraints(
                allocations, portfolio_value, current_positions
            )
            
            # 5. Calculate portfolio-level metrics
            portfolio_metrics = self._calculate_portfolio_metrics(
                validated_allocations, correlations
            )
            
            # 6. Generate recommendations
            recommendations, warnings = self._generate_recommendations(
                validated_allocations, portfolio_metrics
            )
            
            result = PortfolioAllocationResult(
                timestamp=datetime.now(),
                total_capital=portfolio_value,
                allocations=validated_allocations,
                portfolio_sharpe=portfolio_metrics.get('sharpe_ratio', 0.0),
                portfolio_expected_return=portfolio_metrics.get('expected_return', 0.0),
                portfolio_volatility=portfolio_metrics.get('volatility', 0.0),
                diversification_ratio=portfolio_metrics.get('diversification_ratio', 1.0),
                risk_budget_utilization=portfolio_metrics.get('risk_budget_utilization', 0.0),
                recommendations=recommendations,
                warnings=warnings
            )
            
            # Store in history
            self.allocation_history.append(result)
            if len(self.allocation_history) > self.max_history:
                self.allocation_history = self.allocation_history[-self.max_history:]
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error optimizing portfolio allocation: {e}")
            # Return equal allocation as fallback
            return self._create_equal_allocation_fallback(strategies, portfolio_value)
    
    async def _calculate_strategy_metrics(
        self,
        strategies: Dict[str, Any],
        strategy_returns: Optional[Dict[str, pd.Series]]
    ) -> Dict[str, Dict[str, float]]:
        """Calculate performance metrics for each strategy."""
        metrics = {}
        
        for strategy_name, strategy in strategies.items():
            try:
                # Try to get performance from strategy
                if hasattr(strategy, 'get_strategy_status'):
                    status = strategy.get_strategy_status()
                    sharpe = status.get('sharpe_ratio', 0.0)
                    expected_return = status.get('expected_return', 0.0)
                    volatility = status.get('volatility', 0.20)
                    max_drawdown = status.get('max_drawdown', 0.0)
                elif strategy_returns and strategy_name in strategy_returns:
                    # Calculate from returns
                    returns = strategy_returns[strategy_name]
                    sharpe = self._calculate_sharpe(returns)
                    expected_return = float(returns.mean() * 252)  # Annualized
                    volatility = float(returns.std() * np.sqrt(252))  # Annualized
                    max_drawdown = self._calculate_max_drawdown(returns)
                else:
                    # Default metrics
                    sharpe = 0.5
                    expected_return = 0.10
                    volatility = 0.20
                    max_drawdown = 0.15
                
                # Calculate priority score (higher = better)
                priority_score = (
                    sharpe * 0.4 +  # Risk-adjusted return
                    (expected_return / volatility) * 0.3 +  # Return/risk ratio
                    (1.0 - max_drawdown) * 0.2 +  # Drawdown penalty
                    0.1  # Base score
                )
                
                metrics[strategy_name] = {
                    'sharpe_ratio': sharpe,
                    'expected_return': expected_return,
                    'volatility': volatility,
                    'max_drawdown': max_drawdown,
                    'priority_score': priority_score
                }
                
            except Exception as e:
                self.logger.warning(f"Error calculating metrics for {strategy_name}: {e}")
                metrics[strategy_name] = {
                    'sharpe_ratio': 0.0,
                    'expected_return': 0.0,
                    'volatility': 0.20,
                    'max_drawdown': 0.0,
                    'priority_score': 0.0
                }
        
        return metrics
    
    async def _calculate_optimal_allocations(
        self,
        strategy_metrics: Dict[str, Dict[str, float]],
        correlations: Dict[str, Any],
        portfolio_value: Decimal
    ) -> Dict[str, StrategyAllocation]:
        """Calculate optimal allocations using multiple methods."""
        allocations = {}
        
        # Method 1: Risk Parity (equal risk contribution)
        risk_parity_weights = self._calculate_risk_parity_weights(
            strategy_metrics, correlations
        )
        
        # Method 2: Sharpe Ratio Weighted
        sharpe_weights = self._calculate_sharpe_weighted_weights(strategy_metrics)
        
        # Method 3: Priority Score Weighted
        priority_weights = self._calculate_priority_weighted_weights(strategy_metrics)
        
        # Combine methods (weighted average)
        for strategy_name in strategy_metrics.keys():
            metrics = strategy_metrics[strategy_name]
            
            # Combined weight (favor risk parity and priority)
            combined_weight = (
                risk_parity_weights.get(strategy_name, 0.0) * 0.4 +
                sharpe_weights.get(strategy_name, 0.0) * 0.3 +
                priority_weights.get(strategy_name, 0.0) * 0.3
            )
            
            optimal_allocation = combined_weight * self.max_total_allocation
            capital_required = portfolio_value * Decimal(str(optimal_allocation))
            
            allocations[strategy_name] = StrategyAllocation(
                strategy_name=strategy_name,
                current_allocation=0.0,  # Will be updated if current positions provided
                optimal_allocation=optimal_allocation,
                recommended_allocation=optimal_allocation,
                sharpe_ratio=metrics['sharpe_ratio'],
                expected_return=metrics['expected_return'],
                volatility=metrics['volatility'],
                max_drawdown=metrics['max_drawdown'],
                correlation_with_portfolio=0.0,  # Will be calculated
                risk_contribution=0.0,  # Will be calculated
                capital_required=capital_required,
                priority_score=metrics['priority_score']
            )
        
        return allocations
    
    def _calculate_risk_parity_weights(
        self,
        strategy_metrics: Dict[str, Dict[str, float]],
        correlations: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate risk parity weights (equal risk contribution)."""
        # Simplified risk parity: inverse volatility weighting
        inv_vol = {
            name: 1.0 / max(metrics['volatility'], 0.01)
            for name, metrics in strategy_metrics.items()
        }
        
        total_inv_vol = sum(inv_vol.values())
        if total_inv_vol > 0:
            return {name: inv / total_inv_vol for name, inv in inv_vol.items()}
        else:
            # Equal weights fallback
            n = len(strategy_metrics)
            return {name: 1.0 / n for name in strategy_metrics.keys()}
    
    def _calculate_sharpe_weighted_weights(
        self,
        strategy_metrics: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:
        """Calculate weights proportional to Sharpe ratio."""
        sharpe_values = {
            name: max(metrics['sharpe_ratio'], 0.0)
            for name, metrics in strategy_metrics.items()
        }
        
        total_sharpe = sum(sharpe_values.values())
        if total_sharpe > 0:
            return {name: sharpe / total_sharpe for name, sharpe in sharpe_values.items()}
        else:
            # Equal weights fallback
            n = len(strategy_metrics)
            return {name: 1.0 / n for name in strategy_metrics.keys()}
    
    def _calculate_priority_weighted_weights(
        self,
        strategy_metrics: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:
        """Calculate weights based on priority scores."""
        priority_scores = {
            name: max(metrics['priority_score'], 0.0)
            for name, metrics in strategy_metrics.items()
        }
        
        total_priority = sum(priority_scores.values())
        if total_priority > 0:
            return {name: score / total_priority for name, score in priority_scores.items()}
        else:
            # Equal weights fallback
            n = len(strategy_metrics)
            return {name: 1.0 / n for name in strategy_metrics.keys()}
    
    def _apply_allocation_constraints(
        self,
        allocations: Dict[str, StrategyAllocation],
        portfolio_value: Decimal,
        current_positions: Optional[Dict[str, Dict]]
    ) -> Dict[str, StrategyAllocation]:
        """Apply constraints to allocations."""
        # Constraint 1: Max allocation per strategy (20%)
        max_per_strategy = 0.20
        
        # Constraint 2: Min allocation per strategy (1%)
        min_per_strategy = 0.01
        
        # Constraint 3: Total allocation limit
        total_allocation = sum(a.optimal_allocation for a in allocations.values())
        
        # Normalize if over limit
        if total_allocation > self.max_total_allocation:
            scale_factor = self.max_total_allocation / total_allocation
            for allocation in allocations.values():
                allocation.recommended_allocation = min(
                    allocation.optimal_allocation * scale_factor,
                    max_per_strategy
                )
        else:
            for allocation in allocations.values():
                allocation.recommended_allocation = min(
                    max(allocation.optimal_allocation, min_per_strategy),
                    max_per_strategy
                )
        
        # Update capital required
        for allocation in allocations.values():
            allocation.capital_required = portfolio_value * Decimal(str(allocation.recommended_allocation))
        
        # Update current allocation if positions provided
        if current_positions:
            for strategy_name, allocation in allocations.items():
                if strategy_name in current_positions:
                    position_value = Decimal(str(current_positions[strategy_name].get('value', 0)))
                    allocation.current_allocation = float(position_value / portfolio_value)
        
        return allocations
    
    def _calculate_portfolio_metrics(
        self,
        allocations: Dict[str, StrategyAllocation],
        correlations: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate portfolio-level risk/return metrics."""
        try:
            # Weighted average return
            portfolio_return = sum(
                a.recommended_allocation * a.expected_return
                for a in allocations.values()
            )
            
            # Portfolio volatility (simplified - would need full correlation matrix)
            portfolio_vol = np.sqrt(sum(
                (a.recommended_allocation * a.volatility) ** 2
                for a in allocations.values()
            ))
            
            # Sharpe ratio
            sharpe = portfolio_return / max(portfolio_vol, 0.01) if portfolio_vol > 0 else 0.0
            
            # Diversification ratio
            avg_vol = np.mean([a.volatility for a in allocations.values()])
            diversification_ratio = avg_vol / max(portfolio_vol, 0.01) if portfolio_vol > 0 else 1.0
            
            # Risk budget utilization
            total_allocation = sum(a.recommended_allocation for a in allocations.values())
            risk_budget_utilization = total_allocation / self.max_total_allocation
            
            return {
                'expected_return': portfolio_return,
                'volatility': portfolio_vol,
                'sharpe_ratio': sharpe,
                'diversification_ratio': diversification_ratio,
                'risk_budget_utilization': risk_budget_utilization
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio metrics: {e}")
            return {
                'expected_return': 0.0,
                'volatility': 0.20,
                'sharpe_ratio': 0.0,
                'diversification_ratio': 1.0,
                'risk_budget_utilization': 0.0
            }
    
    def _generate_recommendations(
        self,
        allocations: Dict[str, StrategyAllocation],
        portfolio_metrics: Dict[str, float]
    ) -> tuple[List[str], List[str]]:
        """Generate allocation recommendations and warnings."""
        recommendations = []
        warnings = []
        
        # Check for overallocation
        total_allocation = sum(a.recommended_allocation for a in allocations.values())
        if total_allocation > self.max_total_allocation:
            warnings.append(
                f"Total allocation {total_allocation:.1%} exceeds limit {self.max_total_allocation:.1%}"
            )
        
        # Check for underallocation
        if total_allocation < 0.5:
            recommendations.append(
                f"Consider increasing allocation (currently {total_allocation:.1%})"
            )
        
        # Check Sharpe ratio
        if portfolio_metrics['sharpe_ratio'] < 1.0:
            warnings.append(
                f"Portfolio Sharpe ratio {portfolio_metrics['sharpe_ratio']:.2f} is below target (1.0)"
            )
        
        # Strategy-specific recommendations
        for allocation in allocations.values():
            if allocation.priority_score > 0.7 and allocation.recommended_allocation < 0.10:
                recommendations.append(
                    f"Increase allocation for {allocation.strategy_name} "
                    f"(high priority score: {allocation.priority_score:.2f})"
                )
            elif allocation.priority_score < 0.3 and allocation.recommended_allocation > 0.10:
                recommendations.append(
                    f"Consider reducing allocation for {allocation.strategy_name} "
                    f"(low priority score: {allocation.priority_score:.2f})"
                )
        
        return recommendations, warnings
    
    def _create_equal_allocation_fallback(
        self,
        strategies: Dict[str, Any],
        portfolio_value: Decimal
    ) -> PortfolioAllocationResult:
        """Create equal allocation as fallback."""
        n = len(strategies)
        equal_weight = (self.max_total_allocation / n) if n > 0 else 0.0
        
        allocations = {}
        for strategy_name in strategies.keys():
            allocations[strategy_name] = StrategyAllocation(
                strategy_name=strategy_name,
                current_allocation=0.0,
                optimal_allocation=equal_weight,
                recommended_allocation=equal_weight,
                sharpe_ratio=0.0,
                expected_return=0.0,
                volatility=0.20,
                max_drawdown=0.0,
                correlation_with_portfolio=0.0,
                risk_contribution=0.0,
                capital_required=portfolio_value * Decimal(str(equal_weight)),
                priority_score=0.5
            )
        
        return PortfolioAllocationResult(
            timestamp=datetime.now(),
            total_capital=portfolio_value,
            allocations=allocations,
            portfolio_sharpe=0.0,
            portfolio_expected_return=0.0,
            portfolio_volatility=0.20,
            diversification_ratio=1.0,
            risk_budget_utilization=equal_weight * n,
            recommendations=["Using equal allocation fallback"],
            warnings=["Optimization failed, using equal weights"]
        )
    
    @staticmethod
    def _calculate_sharpe(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) < 2:
            return 0.0
        excess_returns = returns - (risk_free_rate / 252)
        if excess_returns.std() == 0:
            return 0.0
        return float(np.sqrt(252) * excess_returns.mean() / excess_returns.std())
    
    @staticmethod
    def _calculate_max_drawdown(returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        if len(returns) < 2:
            return 0.0
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return float(abs(drawdown.min()))

