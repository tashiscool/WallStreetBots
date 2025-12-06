"""Allocation Weight Fine-Tuning System

Fine-tunes portfolio allocation weights based on backtesting results.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from decimal import Decimal
import pandas as pd
import numpy as np
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


@dataclass
class AllocationWeights:
    """Allocation weight configuration."""
    risk_parity_weight: float = 0.4
    sharpe_weight_weight: float = 0.3
    priority_weight_weight: float = 0.3
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            'risk_parity': self.risk_parity_weight,
            'sharpe_weighted': self.sharpe_weight_weight,
            'priority_weighted': self.priority_weight_weight
        }


@dataclass
class BacktestResult:
    """Result of a backtest run."""
    timestamp: datetime
    weights: AllocationWeights
    portfolio_sharpe: float
    portfolio_return: float
    portfolio_volatility: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    strategy_allocations: Dict[str, float]


@dataclass
class TuningResult:
    """Result of weight tuning."""
    timestamp: datetime
    optimal_weights: AllocationWeights
    backtest_results: List[BacktestResult]
    improvement: Dict[str, float]  # Improvement metrics
    recommendations: List[str]


class AllocationWeightTuner:
    """Fine-tune allocation weights based on backtesting."""
    
    def __init__(self):
        """Initialize tuner."""
        self.logger = logging.getLogger(__name__)
        self.backtest_results: List[BacktestResult] = []
        
    def tune_weights(
        self,
        historical_returns: Dict[str, pd.Series],
        current_weights: Optional[AllocationWeights] = None,
        optimization_objective: str = 'sharpe'
    ) -> TuningResult:
        """Tune allocation weights using historical data.
        
        Args:
            historical_returns: Dictionary of strategy_name -> returns Series
            current_weights: Current weight configuration
            optimization_objective: 'sharpe', 'return', or 'risk_adjusted'
            
        Returns:
            TuningResult with optimal weights
        """
        try:
            self.logger.info("Starting allocation weight tuning")
            
            if current_weights is None:
                current_weights = AllocationWeights()
            
            # Run backtests with different weight combinations
            backtest_results = self._run_backtest_grid_search(
                historical_returns,
                optimization_objective
            )
            
            # Find optimal weights
            optimal_weights = self._find_optimal_weights(
                backtest_results,
                optimization_objective
            )
            
            # Calculate improvement
            improvement = self._calculate_improvement(
                current_weights,
                optimal_weights,
                backtest_results
            )
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                optimal_weights,
                backtest_results,
                improvement
            )
            
            result = TuningResult(
                timestamp=datetime.now(),
                optimal_weights=optimal_weights,
                backtest_results=backtest_results,
                improvement=improvement,
                recommendations=recommendations
            )
            
            self.backtest_results.extend(backtest_results)
            
            self.logger.info(f"Tuning complete. Optimal weights: {optimal_weights.to_dict()}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error tuning weights: {e}")
            # Return current weights as fallback
            return TuningResult(
                timestamp=datetime.now(),
                optimal_weights=current_weights or AllocationWeights(),
                backtest_results=[],
                improvement={},
                recommendations=[f"Tuning failed: {e}"]
            )
    
    def _run_backtest_grid_search(
        self,
        historical_returns: Dict[str, pd.Series],
        objective: str
    ) -> List[BacktestResult]:
        """Run backtests with different weight combinations."""
        results = []
        
        # Generate weight combinations
        weight_combinations = self._generate_weight_combinations()
        
        for weights in weight_combinations:
            try:
                # Calculate portfolio returns with these weights
                portfolio_returns = self._calculate_portfolio_returns(
                    historical_returns,
                    weights
                )
                
                if len(portfolio_returns) < 10:  # Need minimum data
                    continue
                
                # Calculate metrics
                sharpe = self._calculate_sharpe(portfolio_returns)
                total_return = float(portfolio_returns.sum())
                volatility = float(portfolio_returns.std() * np.sqrt(252))
                max_dd = self._calculate_max_drawdown(portfolio_returns)
                win_rate = float((portfolio_returns > 0).sum() / len(portfolio_returns))
                
                # Calculate strategy allocations (simplified)
                strategy_allocations = self._calculate_strategy_allocations(
                    historical_returns,
                    weights
                )
                
                result = BacktestResult(
                    timestamp=datetime.now(),
                    weights=weights,
                    portfolio_sharpe=sharpe,
                    portfolio_return=total_return,
                    portfolio_volatility=volatility,
                    max_drawdown=max_dd,
                    win_rate=win_rate,
                    total_trades=len(portfolio_returns),
                    strategy_allocations=strategy_allocations
                )
                
                results.append(result)
                
            except Exception as e:
                self.logger.warning(f"Error in backtest with weights {weights.to_dict()}: {e}")
        
        return results
    
    def _generate_weight_combinations(self) -> List[AllocationWeights]:
        """Generate weight combinations to test."""
        combinations = []
        
        # Grid search over weight space
        for rp in [0.2, 0.3, 0.4, 0.5, 0.6]:
            for sw in [0.2, 0.3, 0.4, 0.5]:
                pw = 1.0 - rp - sw
                if pw >= 0.1:  # Minimum weight for each method
                    combinations.append(AllocationWeights(
                        risk_parity_weight=rp,
                        sharpe_weight_weight=sw,
                        priority_weight_weight=pw
                    ))
        
        return combinations
    
    def _calculate_portfolio_returns(
        self,
        strategy_returns: Dict[str, pd.Series],
        weights: AllocationWeights
    ) -> pd.Series:
        """Calculate portfolio returns using given weights."""
        # This is a simplified calculation
        # In reality, we'd need to apply the weights to actual allocations
        
        if not strategy_returns:
            return pd.Series(dtype=float)
        
        # Align all return series
        aligned_returns = pd.DataFrame(strategy_returns)
        aligned_returns = aligned_returns.fillna(0)
        
        # Equal weight for now (simplified)
        # In full implementation, would use actual allocation weights
        portfolio_returns = aligned_returns.mean(axis=1)
        
        return portfolio_returns
    
    def _calculate_strategy_allocations(
        self,
        strategy_returns: Dict[str, pd.Series],
        weights: AllocationWeights
    ) -> Dict[str, float]:
        """Calculate strategy allocations using given weights."""
        # Simplified: equal allocation
        # In full implementation, would use actual allocation logic
        n = len(strategy_returns)
        return {name: 1.0 / n for name in strategy_returns.keys()}
    
    def _find_optimal_weights(
        self,
        backtest_results: List[BacktestResult],
        objective: str
    ) -> AllocationWeights:
        """Find optimal weights based on backtest results."""
        if not backtest_results:
            return AllocationWeights()
        
        # Score each result based on objective
        scored_results = []
        for result in backtest_results:
            if objective == 'sharpe':
                score = result.portfolio_sharpe
            elif objective == 'return':
                score = result.portfolio_return
            elif objective == 'risk_adjusted':
                score = result.portfolio_sharpe * (1.0 - result.max_drawdown)
            else:
                score = result.portfolio_sharpe
            
            scored_results.append((score, result))
        
        # Sort by score
        scored_results.sort(key=lambda x: x[0], reverse=True)
        
        # Return weights from best result
        return scored_results[0][1].weights
    
    def _calculate_improvement(
        self,
        current_weights: AllocationWeights,
        optimal_weights: AllocationWeights,
        backtest_results: List[BacktestResult]
    ) -> Dict[str, float]:
        """Calculate improvement from tuning."""
        if not backtest_results:
            return {}
        
        # Find results with current and optimal weights
        current_result = next(
            (r for r in backtest_results if r.weights == current_weights),
            None
        )
        optimal_result = next(
            (r for r in backtest_results if r.weights == optimal_weights),
            None
        )
        
        if not current_result or not optimal_result:
            return {}
        
        improvement = {
            'sharpe_improvement': optimal_result.portfolio_sharpe - current_result.portfolio_sharpe,
            'return_improvement': optimal_result.portfolio_return - current_result.portfolio_return,
            'volatility_change': optimal_result.portfolio_volatility - current_result.portfolio_volatility,
            'drawdown_improvement': current_result.max_drawdown - optimal_result.max_drawdown,
            'sharpe_pct_improvement': (
                (optimal_result.portfolio_sharpe - current_result.portfolio_sharpe) /
                max(abs(current_result.portfolio_sharpe), 0.01) * 100
            )
        }
        
        return improvement
    
    def _generate_recommendations(
        self,
        optimal_weights: AllocationWeights,
        backtest_results: List[BacktestResult],
        improvement: Dict[str, float]
    ) -> List[str]:
        """Generate recommendations based on tuning results."""
        recommendations = []
        
        if improvement.get('sharpe_improvement', 0) > 0.1:
            recommendations.append(
                f"Optimal weights improve Sharpe by {improvement['sharpe_improvement']:.2f}. "
                "Consider updating allocation weights."
            )
        
        # Weight-specific recommendations
        if optimal_weights.risk_parity_weight > 0.5:
            recommendations.append(
                "Risk parity has high weight - good for diversification"
            )
        
        if optimal_weights.sharpe_weight_weight > 0.4:
            recommendations.append(
                "Sharpe-weighted allocation has high weight - good for risk-adjusted returns"
            )
        
        return recommendations
    
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

