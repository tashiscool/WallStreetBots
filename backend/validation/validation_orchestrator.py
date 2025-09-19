"""Validation orchestrator for comprehensive strategy validation and alpha discovery.

Coordinates all validation modules to provide complete strategy evaluation
and deployment readiness assessment.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
import logging
import asyncio
from datetime import datetime
import json
import os

from .factor_analysis import AlphaFactorAnalyzer
from .regime_testing import RegimeValidator
from .cross_market_validator import CrossMarketValidator
from .ensemble_evaluator import EnsembleValidator
from .capital_efficiency import CapitalEfficiencyAnalyzer
from .alpha_validation_gate import AlphaValidationGate, ValidationCriteria

logger = logging.getLogger(__name__)


class ValidationOrchestrator:
    """Orchestrates comprehensive strategy validation and alpha discovery."""

    def __init__(self, criteria: ValidationCriteria = None):
        self.criteria = criteria or ValidationCriteria()

        # Initialize validators
        self.factor_analyzer = AlphaFactorAnalyzer()
        self.regime_validator = RegimeValidator()
        self.cross_market_validator = CrossMarketValidator()
        self.ensemble_evaluator = EnsembleValidator()
        self.capital_analyzer = CapitalEfficiencyAnalyzer()
        self.validation_gate = AlphaValidationGate(self.criteria)

    def validate_strategy(self,
                         strategy,
                         start_date: str = '2020-01-01',
                         end_date: str = '2024-12-31',
                         market_data: Optional[Dict[str, pd.Series]] = None,
                         run_parallel: bool = True) -> Dict[str, Any]:
        """Run comprehensive validation for a single strategy.

        Args:
            strategy: Strategy object to validate
            start_date: Start date for validation
            end_date: End date for validation
            market_data: Optional market data for regime analysis
            run_parallel: Whether to run validations in parallel

        Returns:
            Comprehensive validation results
        """
        logger.info(f"Starting comprehensive validation for strategy: {getattr(strategy, 'name', 'Unknown')}")

        validation_results = {
            'strategy_info': self._extract_strategy_info(strategy),
            'validation_period': {'start': start_date, 'end': end_date},
            'validation_timestamp': datetime.now().isoformat()
        }

        try:
            # Get strategy returns
            strategy_returns = self._get_strategy_returns(strategy, start_date, end_date)
            if strategy_returns is None or len(strategy_returns) < 60:
                return self._create_insufficient_data_response(strategy_returns)

            validation_results['data_quality'] = self._assess_data_quality(strategy_returns)

            # Get market data if not provided
            if market_data is None:
                market_data = self._get_default_market_data(start_date, end_date)

            if run_parallel:
                # Run validations in parallel
                validation_results.update(
                    self._run_parallel_validations(strategy, strategy_returns, market_data, start_date, end_date)
                )
            else:
                # Run validations sequentially
                validation_results.update(
                    self._run_sequential_validations(strategy, strategy_returns, market_data, start_date, end_date)
                )

            # Final evaluation
            validation_results['deployment_decision'] = self.validation_gate.evaluate_go_no_go(validation_results)

            logger.info(f"Validation completed. Recommendation: {validation_results['deployment_decision']['recommendation']}")

        except Exception as e:
            logger.error(f"Validation failed: {e}")
            validation_results['error'] = str(e)
            validation_results['deployment_decision'] = {
                'recommendation': 'ERROR',
                'error': str(e)
            }

        return validation_results

    def validate_multiple_strategies(self,
                                   strategies: Dict[str, Any],
                                   start_date: str = '2020-01-01',
                                   end_date: str = '2024-12-31') -> Dict[str, Any]:
        """Validate multiple strategies and perform ensemble analysis.

        Args:
            strategies: Dictionary mapping strategy names to strategy objects
            start_date: Start date for validation
            end_date: End date for validation

        Returns:
            Comprehensive validation results for all strategies plus ensemble analysis
        """
        logger.info(f"Starting validation for {len(strategies)} strategies")

        results = {
            'individual_strategy_results': {},
            'ensemble_analysis': {},
            'comparative_analysis': {},
            'validation_summary': {}
        }

        strategy_returns = {}

        # Validate individual strategies
        for strategy_name, strategy in strategies.items():
            logger.info(f"Validating strategy: {strategy_name}")

            try:
                individual_results = self.validate_strategy(
                    strategy, start_date, end_date, run_parallel=False
                )
                results['individual_strategy_results'][strategy_name] = individual_results

                # Collect returns for ensemble analysis
                if 'data_quality' in individual_results and 'returns' in individual_results['data_quality']:
                    strategy_returns[strategy_name] = individual_results['data_quality']['returns']

            except Exception as e:
                logger.error(f"Failed to validate strategy {strategy_name}: {e}")
                results['individual_strategy_results'][strategy_name] = {'error': str(e)}

        # Ensemble analysis if we have multiple valid strategies
        if len(strategy_returns) > 1:
            try:
                logger.info("Running ensemble analysis")
                ensemble_results = self.ensemble_evaluator.analyze_strategy_correlations(strategy_returns)
                results['ensemble_analysis'] = ensemble_results

                # Build optimal ensemble
                optimal_ensemble = self.ensemble_evaluator.build_optimal_ensemble(strategy_returns)
                results['ensemble_analysis']['optimal_ensemble'] = optimal_ensemble

            except Exception as e:
                logger.error(f"Ensemble analysis failed: {e}")
                results['ensemble_analysis'] = {'error': str(e)}

        # Comparative analysis
        results['comparative_analysis'] = self._create_comparative_analysis(
            results['individual_strategy_results']
        )

        # Overall summary
        results['validation_summary'] = self._create_validation_summary(results)

        return results

    def _run_parallel_validations(self,
                                strategy: Any,
                                strategy_returns: pd.Series,
                                market_data: Dict[str, pd.Series],
                                start_date: str,
                                end_date: str) -> Dict[str, Any]:
        """Run validations in parallel using asyncio."""
        try:
            # Note: This is a simplified parallel approach
            # In production, you might want to use actual async implementations
            results = {}

            # Run factor analysis
            try:
                logger.info("Running factor analysis")
                factor_results = self.factor_analyzer.run_factor_regression(
                    strategy_returns, start_date, end_date
                )
                results['factor_analysis'] = factor_results
            except Exception as e:
                logger.warning(f"Factor analysis failed: {e}")
                results['factor_analysis'] = {'error': str(e)}

            # Run regime testing
            try:
                logger.info("Running regime testing")
                regime_results = self.regime_validator.test_edge_persistence(
                    strategy_returns, market_data
                )
                results['regime_testing'] = regime_results
            except Exception as e:
                logger.warning(f"Regime testing failed: {e}")
                results['regime_testing'] = {'error': str(e)}

            # Run cross-market validation
            try:
                logger.info("Running cross-market validation")
                cross_market_results = self.cross_market_validator.validate_across_markets(
                    type(strategy), self._extract_strategy_params(strategy), start_date, end_date
                )
                results['cross_market'] = cross_market_results
            except Exception as e:
                logger.warning(f"Cross-market validation failed: {e}")
                results['cross_market'] = {'error': str(e)}

            # Run capital efficiency analysis
            try:
                logger.info("Running capital efficiency analysis")
                capital_results = self._run_capital_efficiency_analysis(strategy, strategy_returns)
                results['capital_efficiency'] = capital_results
            except Exception as e:
                logger.warning(f"Capital efficiency analysis failed: {e}")
                results['capital_efficiency'] = {'error': str(e)}

            return results

        except Exception as e:
            logger.error(f"Parallel validation failed: {e}")
            return {'error': f'Parallel validation failed: {e!s}'}

    def _run_sequential_validations(self,
                                  strategy: Any,
                                  strategy_returns: pd.Series,
                                  market_data: Dict[str, pd.Series],
                                  start_date: str,
                                  end_date: str) -> Dict[str, Any]:
        """Run validations sequentially."""
        return self._run_parallel_validations(strategy, strategy_returns, market_data, start_date, end_date)

    def _run_capital_efficiency_analysis(self, strategy: Any, strategy_returns: pd.Series) -> Dict[str, Any]:
        """Run capital efficiency analysis with proper error handling."""
        results = {}

        try:
            # Kelly sizing analysis
            kelly_results = self.capital_analyzer.kelly_sizing_analysis(strategy_returns)
            results['kelly_analysis'] = kelly_results

            # Leverage efficiency (simplified version)
            if hasattr(strategy, 'backtest_with_capital') or hasattr(strategy, 'set_capital'):
                capital_levels = [10000, 50000, 100000]
                leverage_results = self.capital_analyzer.analyze_leverage_efficiency(
                    strategy, capital_levels
                )
                results.update(leverage_results)
            else:
                results['leverage_analysis'] = {
                    'error': 'Strategy does not support capital adjustment for leverage analysis'
                }

        except Exception as e:
            logger.error(f"Capital efficiency analysis failed: {e}")
            results['error'] = str(e)

        return results

    def _get_strategy_returns(self, strategy: Any, start_date: str, end_date: str) -> Optional[pd.Series]:
        """Extract strategy returns with multiple fallback methods."""
        try:
            # Method 1: Direct returns attribute
            if hasattr(strategy, 'returns'):
                returns = strategy.returns
                if isinstance(returns, pd.Series) and len(returns) > 0:
                    return returns

            # Method 2: Run backtest
            if hasattr(strategy, 'backtest'):
                backtest_results = strategy.backtest(start=start_date, end=end_date)
                if hasattr(backtest_results, 'returns'):
                    return backtest_results.returns
                elif isinstance(backtest_results, dict) and 'returns' in backtest_results:
                    return pd.Series(backtest_results['returns'])

            # Method 3: Alternative backtest method
            if hasattr(strategy, 'run_backtest'):
                backtest_results = strategy.run_backtest(start_date, end_date)
                if hasattr(backtest_results, 'returns'):
                    return backtest_results.returns
                elif isinstance(backtest_results, dict) and 'returns' in backtest_results:
                    return pd.Series(backtest_results['returns'])

            # Method 4: Generate returns if strategy supports it
            if hasattr(strategy, 'generate_returns'):
                returns = strategy.generate_returns(start_date, end_date)
                if isinstance(returns, (pd.Series, list, np.ndarray)):
                    return pd.Series(returns)

            logger.warning("Could not extract strategy returns")
            return None

        except Exception as e:
            logger.error(f"Failed to get strategy returns: {e}")
            return None

    def _get_default_market_data(self, start_date: str, end_date: str) -> Dict[str, pd.Series]:
        """Get default market data for regime analysis."""
        market_data = {}

        try:
            import yfinance as yf

            # Download key market indicators
            tickers = {
                'SPY': 'SPY',    # S&P 500
                'VIX': '^VIX',   # Volatility Index
                'DGS10': '^TNX'  # 10-Year Treasury (approximate)
            }

            for name, ticker in tickers.items():
                try:
                    data = yf.download(ticker, start=start_date, end=end_date, progress=False)
                    if not data.empty:
                        market_data[name] = data['Close']
                except Exception as e:
                    logger.warning(f"Could not download {name} data: {e}")

        except ImportError:
            logger.warning("yfinance not available, using synthetic market data")
            # Generate synthetic market data as fallback
            date_range = pd.date_range(start=start_date, end=end_date)
            market_data['SPY'] = pd.Series(
                100 * np.cumprod(1 + np.random.normal(0.0005, 0.02, len(date_range))),
                index=date_range
            )
            market_data['VIX'] = pd.Series(
                15 + 10 * np.random.beta(2, 5, len(date_range)),
                index=date_range
            )

        except Exception as e:
            logger.warning(f"Failed to get market data: {e}")

        return market_data

    def _extract_strategy_info(self, strategy: Any) -> Dict[str, Any]:
        """Extract strategy information for documentation."""
        info = {
            'class_name': strategy.__class__.__name__,
            'name': getattr(strategy, 'name', 'Unknown'),
            'description': getattr(strategy, 'description', 'No description available')
        }

        # Try to get additional strategy attributes
        for attr in ['version', 'author', 'parameters', 'asset_class']:
            if hasattr(strategy, attr):
                info[attr] = getattr(strategy, attr)

        return info

    def _extract_strategy_params(self, strategy: Any) -> Dict[str, Any]:
        """Extract strategy parameters for cross-market testing."""
        params = {}

        # Common parameter attributes to look for
        param_attrs = ['lookback', 'threshold', 'max_positions', 'stop_loss', 'take_profit']

        for attr in param_attrs:
            if hasattr(strategy, attr):
                params[attr] = getattr(strategy, attr)

        # If strategy has a parameters dict
        if hasattr(strategy, 'parameters'):
            params.update(strategy.parameters)

        return params

    def _assess_data_quality(self, returns: pd.Series) -> Dict[str, Any]:
        """Assess quality of strategy returns data."""
        if returns is None:
            return {'error': 'No returns data available'}

        quality_metrics = {
            'total_observations': len(returns),
            'non_null_observations': returns.count(),
            'null_percentage': (len(returns) - returns.count()) / len(returns),
            'date_range': {
                'start': returns.index[0].strftime('%Y-%m-%d') if len(returns) > 0 else None,
                'end': returns.index[-1].strftime('%Y-%m-%d') if len(returns) > 0 else None
            },
            'returns_statistics': {
                'mean': returns.mean(),
                'std': returns.std(),
                'min': returns.min(),
                'max': returns.max(),
                'skewness': returns.skew(),
                'kurtosis': returns.kurtosis()
            },
            'outliers': {
                'extreme_positive': (returns > returns.quantile(0.99)).sum(),
                'extreme_negative': (returns < returns.quantile(0.01)).sum()
            },
            'returns': returns  # Include for downstream analysis
        }

        # Data quality assessment
        quality_score = 1.0
        quality_issues = []

        if quality_metrics['null_percentage'] > 0.05:
            quality_score -= 0.2
            quality_issues.append('High percentage of missing data')

        if len(returns) < 252:
            quality_score -= 0.3
            quality_issues.append('Insufficient data for annual analysis')

        if abs(returns.skew()) > 3:
            quality_score -= 0.1
            quality_issues.append('Highly skewed returns distribution')

        quality_metrics['quality_score'] = max(0, quality_score)
        quality_metrics['quality_issues'] = quality_issues

        return quality_metrics

    def _create_comparative_analysis(self, individual_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create comparative analysis across strategies."""
        comparison = {
            'strategy_rankings': {},
            'best_performers': {},
            'risk_comparison': {},
            'deployment_readiness': {}
        }

        valid_strategies = {name: results for name, results in individual_results.items()
                          if 'error' not in results}

        if not valid_strategies:
            return {'error': 'No valid strategies for comparison'}

        # Extract key metrics for comparison
        metrics = {}
        for name, results in valid_strategies.items():
            strategy_metrics = {}

            # Risk management metrics
            if 'data_quality' in results:
                data_quality = results['data_quality']
                if 'returns_statistics' in data_quality:
                    stats = data_quality['returns_statistics']
                    returns = data_quality.get('returns')
                    if returns is not None and len(returns) > 0:
                        strategy_metrics['sharpe_ratio'] = self._calculate_sharpe(returns)
                        strategy_metrics['max_drawdown'] = self._calculate_max_drawdown(returns)
                        strategy_metrics['volatility'] = stats['std'] * np.sqrt(252)

            # Factor analysis metrics
            if 'factor_analysis' in results and 'annualized_alpha' in results['factor_analysis']:
                strategy_metrics['alpha'] = results['factor_analysis']['annualized_alpha']
                strategy_metrics['alpha_significant'] = results['factor_analysis'].get('alpha_significant', False)

            # Deployment decision
            if 'deployment_decision' in results:
                strategy_metrics['recommendation'] = results['deployment_decision']['recommendation']
                strategy_metrics['confidence'] = results['deployment_decision'].get('confidence_score', 0)

            metrics[name] = strategy_metrics

        # Create rankings
        for metric in ['sharpe_ratio', 'alpha', 'confidence']:
            if any(metric in m for m in metrics.values()):
                sorted_strategies = sorted(
                    [(name, m.get(metric, 0)) for name, m in metrics.items()],
                    key=lambda x: x[1], reverse=True
                )
                comparison['strategy_rankings'][metric] = sorted_strategies

        # Best performers
        comparison['best_performers'] = {
            'highest_sharpe': max(metrics.items(), key=lambda x: x[1].get('sharpe_ratio', 0))[0] if metrics else None,
            'highest_alpha': max(metrics.items(), key=lambda x: x[1].get('alpha', 0))[0] if metrics else None,
            'most_deployable': max(metrics.items(), key=lambda x: x[1].get('confidence', 0))[0] if metrics else None
        }

        # Deployment readiness summary
        deployment_counts = {
            'GO': 0,
            'CAUTIOUS_GO': 0,
            'NO_GO': 0,
            'INSUFFICIENT_VALIDATION': 0,
            'ERROR': 0
        }

        for strategy_metrics in metrics.values():
            recommendation = strategy_metrics.get('recommendation', 'ERROR')
            deployment_counts[recommendation] = deployment_counts.get(recommendation, 0) + 1

        comparison['deployment_readiness'] = deployment_counts

        return comparison

    def _create_validation_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Create overall validation summary."""
        summary = {
            'total_strategies': len(results.get('individual_strategy_results', {})),
            'validation_completion': {},
            'overall_assessment': {},
            'next_steps': []
        }

        individual_results = results.get('individual_strategy_results', {})

        # Count completion rates
        validation_components = [
            'factor_analysis', 'regime_testing', 'cross_market',
            'capital_efficiency', 'deployment_decision'
        ]

        completion_rates = {}
        for component in validation_components:
            completed = sum(1 for r in individual_results.values()
                          if component in r and 'error' not in r[component])
            completion_rates[component] = completed / len(individual_results) if individual_results else 0

        summary['validation_completion'] = completion_rates

        # Overall assessment
        if 'comparative_analysis' in results:
            comparative = results['comparative_analysis']
            if 'deployment_readiness' in comparative:
                deployment_readiness = comparative['deployment_readiness']
                ready_strategies = deployment_readiness.get('GO', 0) + deployment_readiness.get('CAUTIOUS_GO', 0)
                total_strategies = sum(deployment_readiness.values())

                summary['overall_assessment'] = {
                    'deployment_ready_strategies': ready_strategies,
                    'deployment_ready_percentage': ready_strategies / total_strategies if total_strategies > 0 else 0,
                    'validation_quality': np.mean(list(completion_rates.values())),
                    'ensemble_available': len(results.get('ensemble_analysis', {})) > 0
                }

        # Generate next steps
        if summary['overall_assessment'].get('deployment_ready_percentage', 0) == 0:
            summary['next_steps'].append('Address critical issues preventing deployment')
        elif summary['overall_assessment'].get('deployment_ready_percentage', 0) < 0.5:
            summary['next_steps'].append('Improve underperforming strategies before deployment')
        else:
            summary['next_steps'].append('Proceed with deployment of validated strategies')

        if summary['overall_assessment'].get('ensemble_available', False):
            summary['next_steps'].append('Consider ensemble deployment for risk reduction')

        return summary

    def _create_insufficient_data_response(self, returns: Optional[pd.Series]) -> Dict[str, Any]:
        """Create response for insufficient data scenarios."""
        return {
            'error': 'Insufficient data for validation',
            'data_available': len(returns) if returns is not None else 0,
            'minimum_required': 60,
            'deployment_decision': {
                'recommendation': 'INSUFFICIENT_DATA',
                'confidence_score': 0.0,
                'error': 'Need at least 60 data points for validation'
            }
        }

    def _calculate_sharpe(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio with error handling."""
        try:
            if len(returns) == 0:
                return np.nan

            excess_returns = returns - risk_free_rate / 252

            if excess_returns.std() == 0:
                return 0.0 if excess_returns.mean() == 0 else np.inf * np.sign(excess_returns.mean())

            return excess_returns.mean() / excess_returns.std() * np.sqrt(252)

        except Exception:
            return np.nan

    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        try:
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdowns = (cumulative - running_max) / running_max
            return drawdowns.min()
        except Exception:
            return np.nan

    def save_validation_results(self, results: Dict[str, Any], filepath: str) -> bool:
        """Save validation results to file."""
        try:
            # Convert pandas objects to serializable format
            serializable_results = self._make_serializable(results)

            os.makedirs(os.path.dirname(filepath), exist_ok=True)

            with open(filepath, 'w') as f:
                json.dump(serializable_results, f, indent=2, default=str)

            logger.info(f"Validation results saved to {filepath}")
            return True

        except Exception as e:
            logger.error(f"Failed to save validation results: {e}")
            return False

    def _make_serializable(self, obj: Any) -> Any:
        """Convert objects to JSON-serializable format."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, pd.Series):
            return {
                'type': 'pandas_series',
                'data': obj.tolist(),
                'index': obj.index.tolist()
            }
        elif isinstance(obj, pd.DataFrame):
            return {
                'type': 'pandas_dataframe',
                'data': obj.to_dict('records')
            }
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif pd.isna(obj):
            return None
        else:
            return obj