"""Portfolio Allocator Testing Framework

Tests portfolio allocator with real strategy data and validates allocation decisions.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, Any, List
import pandas as pd
import numpy as np

from ..core.portfolio_capital_allocator import PortfolioCapitalAllocator, PortfolioAllocationResult
from ...production.core.production_integration import ProductionIntegrationManager

logger = logging.getLogger(__name__)


class PortfolioAllocatorTester:
    """Test portfolio allocator with real strategy data."""
    
    def __init__(self, integration_manager: ProductionIntegrationManager):
        """Initialize tester.
        
        Args:
            integration_manager: Production integration manager for accessing real data
        """
        self.integration_manager = integration_manager
        self.allocator = PortfolioCapitalAllocator(max_total_allocation=0.95)
        self.logger = logging.getLogger(__name__)
        
        # Test results storage
        self.test_results: List[Dict[str, Any]] = []
        
    async def test_with_real_strategies(
        self,
        strategies: Dict[str, Any],
        portfolio_value: Decimal
    ) -> Dict[str, Any]:
        """Test allocator with real strategy objects.
        
        Args:
            strategies: Dictionary of strategy_name -> strategy object
            portfolio_value: Current portfolio value
            
        Returns:
            Test results dictionary
        """
        try:
            self.logger.info("Starting portfolio allocator test with real strategies")
            
            # 1. Collect real strategy returns
            strategy_returns = await self._collect_real_strategy_returns(strategies)
            
            # 2. Get current positions
            current_positions = await self.integration_manager.get_all_positions()
            
            # 3. Run allocation optimization
            allocation_result = await self.allocator.optimize_allocation(
                strategies=strategies,
                portfolio_value=portfolio_value,
                strategy_returns=strategy_returns,
                current_positions=current_positions
            )
            
            # 4. Validate allocation result
            validation_result = self._validate_allocation_result(allocation_result)
            
            # 5. Calculate test metrics
            test_metrics = self._calculate_test_metrics(
                allocation_result,
                strategy_returns,
                validation_result
            )
            
            # 6. Store test result
            test_result = {
                'timestamp': datetime.now(),
                'allocation_result': allocation_result,
                'validation': validation_result,
                'metrics': test_metrics,
                'strategy_returns_available': len(strategy_returns),
                'strategies_tested': list(strategies.keys())
            }
            
            self.test_results.append(test_result)
            
            # 7. Log results
            self._log_test_results(test_result)
            
            return test_result
            
        except Exception as e:
            self.logger.error(f"Error testing portfolio allocator: {e}")
            return {'error': str(e)}
    
    async def _collect_real_strategy_returns(
        self,
        strategies: Dict[str, Any]
    ) -> Dict[str, pd.Series]:
        """Collect real strategy returns from trades and positions."""
        returns = {}
        
        for strategy_name, strategy in strategies.items():
            try:
                # Method 1: Try to get returns from strategy object
                if hasattr(strategy, 'get_returns_history'):
                    strategy_returns = await strategy.get_returns_history()
                    if strategy_returns is not None and isinstance(strategy_returns, pd.Series):
                        returns[strategy_name] = strategy_returns
                        continue
                
                # Method 2: Calculate from trades
                strategy_returns = await self._calculate_returns_from_trades(strategy_name)
                if strategy_returns is not None and len(strategy_returns) > 0:
                    returns[strategy_name] = strategy_returns
                    continue
                
                # Method 3: Calculate from positions
                strategy_returns = await self._calculate_returns_from_positions(strategy_name)
                if strategy_returns is not None and len(strategy_returns) > 0:
                    returns[strategy_name] = strategy_returns
                    
            except Exception as e:
                self.logger.warning(f"Error collecting returns for {strategy_name}: {e}")
        
        return returns
    
    async def _calculate_returns_from_trades(self, strategy_name: str) -> pd.Series | None:
        """Calculate returns from completed trades."""
        try:
            # Get all trades for this strategy
            all_trades = self.integration_manager.trades
            
            strategy_trades = [
                trade for trade in all_trades
                if trade.strategy_name == strategy_name and trade.exit_price is not None
            ]
            
            if len(strategy_trades) < 2:
                return None
            
            # Calculate daily returns
            returns_data = []
            dates = []
            
            for trade in strategy_trades:
                if trade.exit_timestamp and trade.fill_timestamp:
                    # Calculate return
                    entry_cost = float(trade.entry_price * Decimal(str(trade.quantity)) + 
                                     trade.commission + trade.slippage)
                    exit_proceeds = float(trade.exit_price * Decimal(str(trade.quantity)) - 
                                        trade.commission - trade.slippage)
                    
                    if entry_cost > 0:
                        trade_return = (exit_proceeds - entry_cost) / entry_cost
                        returns_data.append(trade_return)
                        dates.append(trade.fill_timestamp)
            
            if len(returns_data) > 0:
                # Create time series
                returns_series = pd.Series(returns_data, index=pd.DatetimeIndex(dates))
                returns_series = returns_series.sort_index()
                
                # Resample to daily if needed
                if len(returns_series) > 0:
                    returns_series = returns_series.resample('D').sum().fillna(0)
                
                return returns_series
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error calculating returns from trades: {e}")
            return None
    
    async def _calculate_returns_from_positions(self, strategy_name: str) -> pd.Series | None:
        """Calculate returns from current positions."""
        try:
            # Get positions for this strategy
            positions = [
                pos for pos in self.integration_manager.active_positions.values()
                if pos.strategy_name == strategy_name
            ]
            
            if len(positions) == 0:
                return None
            
            # Calculate unrealized returns
            returns_data = []
            dates = []
            
            for position in positions:
                if position.entry_price > 0:
                    current_return = float(
                        (position.current_price - position.entry_price) / position.entry_price
                    )
                    returns_data.append(current_return)
                    dates.append(position.created_at)
            
            if len(returns_data) > 0:
                returns_series = pd.Series(returns_data, index=pd.DatetimeIndex(dates))
                returns_series = returns_series.sort_index()
                return returns_series
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error calculating returns from positions: {e}")
            return None
    
    def _validate_allocation_result(
        self,
        allocation_result: PortfolioAllocationResult
    ) -> Dict[str, Any]:
        """Validate allocation result for correctness."""
        validation = {
            'is_valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Check 1: Total allocation doesn't exceed limit
        total_allocation = sum(
            a.recommended_allocation for a in allocation_result.allocations.values()
        )
        if total_allocation > self.allocator.max_total_allocation:
            validation['is_valid'] = False
            validation['errors'].append(
                f"Total allocation {total_allocation:.1%} exceeds limit "
                f"{self.allocator.max_total_allocation:.1%}"
            )
        
        # Check 2: No negative allocations
        for name, allocation in allocation_result.allocations.items():
            if allocation.recommended_allocation < 0:
                validation['is_valid'] = False
                validation['errors'].append(
                    f"Negative allocation for {name}: {allocation.recommended_allocation:.1%}"
                )
        
        # Check 3: Reasonable Sharpe ratio
        if allocation_result.portfolio_sharpe < 0:
            validation['warnings'].append(
                f"Negative portfolio Sharpe ratio: {allocation_result.portfolio_sharpe:.2f}"
            )
        
        # Check 4: Diversification ratio
        if allocation_result.diversification_ratio < 1.0:
            validation['warnings'].append(
                f"Low diversification ratio: {allocation_result.diversification_ratio:.2f}"
            )
        
        return validation
    
    def _calculate_test_metrics(
        self,
        allocation_result: PortfolioAllocationResult,
        strategy_returns: Dict[str, pd.Series],
        validation_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate test metrics."""
        metrics = {
            'total_strategies': len(allocation_result.allocations),
            'strategies_with_returns': len(strategy_returns),
            'total_allocation': sum(
                a.recommended_allocation for a in allocation_result.allocations.values()
            ),
            'portfolio_sharpe': allocation_result.portfolio_sharpe,
            'portfolio_expected_return': allocation_result.portfolio_expected_return,
            'portfolio_volatility': allocation_result.portfolio_volatility,
            'diversification_ratio': allocation_result.diversification_ratio,
            'allocation_valid': validation_result['is_valid'],
            'recommendations_count': len(allocation_result.recommendations),
            'warnings_count': len(allocation_result.warnings)
        }
        
        # Calculate allocation concentration
        allocations = [a.recommended_allocation for a in allocation_result.allocations.values()]
        if allocations:
            metrics['allocation_concentration'] = float(np.std(allocations))
            metrics['max_allocation'] = float(max(allocations))
            metrics['min_allocation'] = float(min(allocations))
        
        return metrics
    
    def _log_test_results(self, test_result: Dict[str, Any]):
        """Log test results."""
        self.logger.info("=" * 60)
        self.logger.info("PORTFOLIO ALLOCATOR TEST RESULTS")
        self.logger.info("=" * 60)
        
        metrics = test_result['metrics']
        self.logger.info(f"Strategies tested: {metrics['total_strategies']}")
        self.logger.info(f"Strategies with returns: {metrics['strategies_with_returns']}")
        self.logger.info(f"Total allocation: {metrics['total_allocation']:.1%}")
        self.logger.info(f"Portfolio Sharpe: {metrics['portfolio_sharpe']:.2f}")
        self.logger.info(f"Portfolio Expected Return: {metrics['portfolio_expected_return']:.1%}")
        self.logger.info(f"Portfolio Volatility: {metrics['portfolio_volatility']:.1%}")
        self.logger.info(f"Diversification Ratio: {metrics['diversification_ratio']:.2f}")
        
        # Log allocations
        self.logger.info("\nAllocations:")
        for name, allocation in test_result['allocation_result'].allocations.items():
            self.logger.info(
                f"  {name}: {allocation.recommended_allocation:.1%} "
                f"(Sharpe: {allocation.sharpe_ratio:.2f}, "
                f"Priority: {allocation.priority_score:.2f})"
            )
        
        # Log validation
        validation = test_result['validation']
        if validation['errors']:
            self.logger.error("Validation Errors:")
            for error in validation['errors']:
                self.logger.error(f"  - {error}")
        
        if validation['warnings']:
            self.logger.warning("Validation Warnings:")
            for warning in validation['warnings']:
                self.logger.warning(f"  - {warning}")
        
        self.logger.info("=" * 60)
    
    def get_test_summary(self) -> Dict[str, Any]:
        """Get summary of all test results."""
        if not self.test_results:
            return {'message': 'No tests run yet'}
        
        return {
            'total_tests': len(self.test_results),
            'latest_test': self.test_results[-1],
            'average_portfolio_sharpe': np.mean([
                r['metrics']['portfolio_sharpe'] for r in self.test_results
            ]),
            'average_diversification_ratio': np.mean([
                r['metrics']['diversification_ratio'] for r in self.test_results
            ])
        }

