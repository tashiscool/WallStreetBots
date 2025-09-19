"""
Regime Testing with Proper Alignment and Drawdown Calculation
Tests strategy performance across different market regimes with correct statistical measures.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict, Callable, Any, Tuple
import logging


def _max_drawdown(returns: pd.Series) -> float:
    """Calculate maximum drawdown on cumulative equity curve."""
    if len(returns) == 0:
        return 0.0

    # Calculate cumulative equity
    equity = (1 + returns).cumprod()

    # Calculate running maximum
    peak = equity.cummax()

    # Calculate drawdown
    drawdown = (equity / peak) - 1.0

    return float(drawdown.min())


def _sharpe(returns: pd.Series, rf_daily: float = 0.0) -> float:
    """Calculate annualized Sharpe ratio."""
    excess_returns = returns - rf_daily

    if len(excess_returns) < 2 or excess_returns.std(ddof=1) == 0:
        return 0.0

    return float(np.sqrt(252) * excess_returns.mean() / excess_returns.std(ddof=1))


def _calmar_ratio(returns: pd.Series) -> float:
    """Calculate Calmar ratio (annual return / max drawdown)."""
    if len(returns) == 0:
        return 0.0

    annual_return = (1 + returns.mean()) ** 252 - 1
    max_dd = abs(_max_drawdown(returns))

    if max_dd == 0:
        return float('inf') if annual_return > 0 else 0.0

    return annual_return / max_dd


def _sortino_ratio(returns: pd.Series, rf_daily: float = 0.0) -> float:
    """Calculate Sortino ratio (downside deviation)."""
    excess_returns = returns - rf_daily

    if len(excess_returns) < 2:
        return 0.0

    downside_returns = excess_returns[excess_returns < 0]

    if len(downside_returns) == 0:
        return float('inf') if excess_returns.mean() > 0 else 0.0

    downside_deviation = downside_returns.std(ddof=1)

    if downside_deviation == 0:
        return 0.0

    return float(np.sqrt(252) * excess_returns.mean() / downside_deviation)


class RegimeValidator:
    """
    Test strategy performance across different market regimes.

    market_data columns expected: ['SPY','VIX','DGS10'] daily close or levels.
    strategy_returns: daily returns aligned to market_data.index
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Define regime detection functions
        # Each function takes aligned DataFrame and returns boolean Series
        self.regimes: Dict[str, Callable[[pd.DataFrame], pd.Series]] = {
            'bull_market': lambda d: (d['SPY'].pct_change(20).rolling(60).mean() > 0.02),
            'bear_market': lambda d: (d['SPY'].pct_change(20).rolling(60).mean() < -0.02),
            'high_vol': lambda d: (d['VIX'] > 20),
            'low_vol': lambda d: (d['VIX'] < 15),
            'rate_hiking': lambda d: (d['DGS10'].diff(60) > 0.5),
            'rate_cutting': lambda d: (d['DGS10'].diff(60) < -0.5),
            'recession': lambda d: (d['SPY'].pct_change(252).rolling(20).mean() < -0.15),
            'expansion': lambda d: (d['SPY'].pct_change(252).rolling(20).mean() > 0.05),
            'vix_spike': lambda d: (d['VIX'] > d['VIX'].rolling(60).quantile(0.9)),
            'quiet_market': lambda d: (d['VIX'] < d['VIX'].rolling(60).quantile(0.1))
        }

    def test_edge_persistence(self,
                             strategy_returns: pd.Series,
                             market_data: pd.DataFrame,
                             min_observations: int = 30) -> Dict[str, Any]:
        """
        Test if strategy edge persists across different market regimes.

        Args:
            strategy_returns: Daily strategy returns
            market_data: DataFrame with SPY, VIX, DGS10 columns
            min_observations: Minimum observations required per regime

        Returns:
            Dict with regime analysis results
        """
        try:
            # Align strategy returns with market data
            aligned = self._align_data(strategy_returns, market_data)

            if aligned is None or len(aligned) < 100:
                return {
                    'regime_results': {},
                    'edge_is_robust': False,
                    'weakest_regime': None,
                    'strongest_regime': None,
                    'error': 'Insufficient aligned data'
                }

            regime_results = {}
            valid_regimes = []

            # Test each regime
            for regime_name, regime_func in self.regimes.items():
                try:
                    # Get regime mask - must return aligned boolean Series
                    regime_mask = regime_func(aligned).fillna(False)

                    # Ensure mask is boolean and aligned
                    if not isinstance(regime_mask, pd.Series):
                        continue

                    regime_mask = regime_mask.reindex(aligned.index, fill_value=False)

                    # Get strategy returns during this regime
                    regime_returns = aligned.loc[regime_mask, 'strategy_returns']

                    if len(regime_returns) >= min_observations:
                        metrics = self._calculate_regime_metrics(regime_returns)
                        regime_results[regime_name] = metrics
                        valid_regimes.append(regime_name)

                        self.logger.info(f"Regime {regime_name}: {len(regime_returns)} obs, "
                                       f"Sharpe={metrics['sharpe_ratio']:.2f}")

                except Exception as e:
                    self.logger.warning(f"Failed to analyze regime {regime_name}: {e}")
                    continue

            if not regime_results:
                return {
                    'regime_results': {},
                    'edge_is_robust': False,
                    'weakest_regime': None,
                    'strongest_regime': None,
                    'error': 'No valid regimes found'
                }

            # Analyze robustness
            edge_analysis = self._analyze_edge_robustness(regime_results)

            return {
                'regime_results': regime_results,
                'edge_is_robust': edge_analysis['is_robust'],
                'weakest_regime': edge_analysis['weakest_regime'],
                'strongest_regime': edge_analysis['strongest_regime'],
                'robustness_score': edge_analysis['robustness_score'],
                'consistency_metrics': edge_analysis['consistency_metrics']
            }

        except Exception as e:
            self.logger.error(f"Regime testing failed: {e}")
            return {
                'regime_results': {},
                'edge_is_robust': False,
                'weakest_regime': None,
                'strongest_regime': None,
                'error': str(e)
            }

    def _align_data(self, strategy_returns: pd.Series, market_data: pd.DataFrame) -> pd.DataFrame:
        """Align strategy returns with market data."""
        try:
            # Combine data
            combined = pd.concat([
                strategy_returns.rename('strategy_returns'),
                market_data
            ], axis=1)

            # Drop rows with any NaN values
            aligned = combined.dropna()

            if len(aligned) < 50:
                self.logger.warning(f"Only {len(aligned)} aligned observations")
                return None

            return aligned

        except Exception as e:
            self.logger.error(f"Data alignment failed: {e}")
            return None

    def _calculate_regime_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate comprehensive metrics for a regime."""
        try:
            if len(returns) == 0:
                return self._empty_metrics()

            metrics = {
                'sample_size': len(returns),
                'avg_return': float(returns.mean()),
                'volatility': float(returns.std(ddof=1)),
                'sharpe_ratio': _sharpe(returns),
                'sortino_ratio': _sortino_ratio(returns),
                'calmar_ratio': _calmar_ratio(returns),
                'max_drawdown': _max_drawdown(returns),
                'win_rate': float((returns > 0).mean()),
                'avg_win': float(returns[returns > 0].mean()) if (returns > 0).any() else 0.0,
                'avg_loss': float(returns[returns < 0].mean()) if (returns < 0).any() else 0.0,
                'profit_factor': self._calculate_profit_factor(returns),
                'skewness': float(returns.skew()) if len(returns) > 2 else 0.0,
                'kurtosis': float(returns.kurtosis()) if len(returns) > 3 else 0.0,
                'var_95': float(returns.quantile(0.05)),
                'cvar_95': float(returns[returns <= returns.quantile(0.05)].mean()) if (returns <= returns.quantile(0.05)).any() else 0.0
            }

            # Annualized metrics
            if len(returns) > 0:
                metrics['annualized_return'] = float((1 + returns.mean()) ** 252 - 1)
                metrics['annualized_volatility'] = float(returns.std(ddof=1) * np.sqrt(252))
            else:
                metrics['annualized_return'] = 0.0
                metrics['annualized_volatility'] = 0.0

            return metrics

        except Exception as e:
            self.logger.error(f"Regime metrics calculation failed: {e}")
            return self._empty_metrics()

    def _calculate_profit_factor(self, returns: pd.Series) -> float:
        """Calculate profit factor (gross profit / gross loss)."""
        try:
            wins = returns[returns > 0]
            losses = returns[returns < 0]

            gross_profit = wins.sum() if len(wins) > 0 else 0.0
            gross_loss = abs(losses.sum()) if len(losses) > 0 else 0.0

            if gross_loss == 0:
                return float('inf') if gross_profit > 0 else 1.0

            return float(gross_profit / gross_loss)

        except Exception:
            return 1.0

    def _empty_metrics(self) -> Dict[str, float]:
        """Return empty metrics dict for error cases."""
        return {
            'sample_size': 0,
            'avg_return': 0.0,
            'volatility': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'calmar_ratio': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'profit_factor': 1.0,
            'skewness': 0.0,
            'kurtosis': 0.0,
            'var_95': 0.0,
            'cvar_95': 0.0,
            'annualized_return': 0.0,
            'annualized_volatility': 0.0
        }

    def _analyze_edge_robustness(self, regime_results: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """Analyze how robust the edge is across regimes."""
        try:
            if not regime_results:
                return {
                    'is_robust': False,
                    'weakest_regime': None,
                    'strongest_regime': None,
                    'robustness_score': 0.0,
                    'consistency_metrics': {}
                }

            # Extract key metrics across regimes
            sharpe_ratios = {k: v['sharpe_ratio'] for k, v in regime_results.items()}
            win_rates = {k: v['win_rate'] for k, v in regime_results.items()}
            max_drawdowns = {k: v['max_drawdown'] for k, v in regime_results.items()}

            # Find strongest and weakest regimes
            strongest_regime = max(sharpe_ratios.keys(), key=lambda k: sharpe_ratios[k])
            weakest_regime = min(sharpe_ratios.keys(), key=lambda k: sharpe_ratios[k])

            # Calculate consistency metrics
            sharpe_values = list(sharpe_ratios.values())
            win_rate_values = list(win_rates.values())

            consistency_metrics = {
                'sharpe_mean': float(np.mean(sharpe_values)),
                'sharpe_std': float(np.std(sharpe_values)),
                'sharpe_min': float(np.min(sharpe_values)),
                'positive_sharpe_pct': float(np.mean([s > 0 for s in sharpe_values])),
                'win_rate_mean': float(np.mean(win_rate_values)),
                'win_rate_min': float(np.min(win_rate_values)),
                'max_drawdown_worst': float(np.min(list(max_drawdowns.values()))),  # Most negative
                'regimes_tested': len(regime_results)
            }

            # Robustness criteria
            robustness_score = self._calculate_robustness_score(consistency_metrics)

            # Edge is robust if:
            # 1. Positive Sharpe in >= 70% of regimes
            # 2. Minimum Sharpe > 0.3
            # 3. Win rate > 45% in all regimes
            # 4. Max drawdown in any regime < 25%
            is_robust = (
                consistency_metrics['positive_sharpe_pct'] >= 0.7 and
                consistency_metrics['sharpe_min'] > 0.3 and
                consistency_metrics['win_rate_min'] > 0.45 and
                consistency_metrics['max_drawdown_worst'] > -0.25
            )

            return {
                'is_robust': is_robust,
                'weakest_regime': weakest_regime,
                'strongest_regime': strongest_regime,
                'robustness_score': robustness_score,
                'consistency_metrics': consistency_metrics
            }

        except Exception as e:
            self.logger.error(f"Robustness analysis failed: {e}")
            return {
                'is_robust': False,
                'weakest_regime': None,
                'strongest_regime': None,
                'robustness_score': 0.0,
                'consistency_metrics': {}
            }

    def _calculate_robustness_score(self, metrics: Dict[str, float]) -> float:
        """Calculate overall robustness score (0-1)."""
        try:
            # Scoring components (each 0-1)
            sharpe_score = min(1.0, max(0.0, (metrics['sharpe_mean'] - 0.5) / 1.0))
            consistency_score = max(0.0, 1.0 - metrics['sharpe_std'] / 2.0)
            positive_score = metrics['positive_sharpe_pct']
            win_rate_score = min(1.0, max(0.0, (metrics['win_rate_mean'] - 0.4) / 0.3))
            drawdown_score = min(1.0, max(0.0, (0.3 + metrics['max_drawdown_worst']) / 0.3))

            # Weighted average
            robustness_score = (
                sharpe_score * 0.3 +
                consistency_score * 0.2 +
                positive_score * 0.2 +
                win_rate_score * 0.15 +
                drawdown_score * 0.15
            )

            return float(robustness_score)

        except Exception:
            return 0.0

    def create_synthetic_market_data(self, start_date: str = '2020-01-01',
                                   end_date: str = '2024-12-31') -> pd.DataFrame:
        """Create synthetic market data for testing."""
        try:
            dates = pd.date_range(start_date, end_date, freq='D')
            np.random.seed(42)

            # Generate realistic market data
            n_days = len(dates)

            # SPY: simulate with regimes
            spy_returns = np.random.normal(0.0005, 0.015, n_days)

            # Add regime changes
            regime_changes = np.random.choice([0, 1], n_days, p=[0.98, 0.02])
            regime_multiplier = np.where(regime_changes, -3, 1)  # Bear market periods
            spy_returns *= regime_multiplier

            spy_prices = 100 * (1 + spy_returns).cumprod()

            # VIX: inverse correlation with SPY, mean reverting
            vix_base = 20
            vix_values = np.zeros(n_days)
            vix_values[0] = vix_base

            for i in range(1, n_days):
                mean_reversion = -0.1 * (vix_values[i-1] - vix_base)
                shock = -2 * spy_returns[i]  # Negative correlation with market
                vix_values[i] = max(5, vix_values[i-1] + mean_reversion + shock + np.random.normal(0, 2))

            # DGS10: random walk with mean reversion
            rate_base = 2.5
            rates = np.zeros(n_days)
            rates[0] = rate_base

            for i in range(1, n_days):
                mean_reversion = -0.01 * (rates[i-1] - rate_base)
                rates[i] = max(0, rates[i-1] + mean_reversion + np.random.normal(0, 0.05))

            market_data = pd.DataFrame({
                'SPY': spy_prices,
                'VIX': vix_values,
                'DGS10': rates
            }, index=dates)

            self.logger.info(f"Created synthetic market data for {len(dates)} days")
            return market_data

        except Exception as e:
            self.logger.error(f"Failed to create synthetic market data: {e}")
            raise

    def get_regime_summary(self, results: Dict[str, Any]) -> str:
        """Generate human-readable summary of regime analysis."""
        try:
            if not results.get('regime_results'):
                return "No regime analysis results available."

            regime_results = results['regime_results']
            summary_parts = []

            # Overall assessment
            if results['edge_is_robust']:
                summary_parts.append("✅ STRATEGY EDGE IS ROBUST ACROSS REGIMES")
            else:
                summary_parts.append("⚠️  STRATEGY EDGE SHOWS REGIME DEPENDENCE")

            # Robustness score
            score = results.get('robustness_score', 0)
            summary_parts.append(f"Robustness Score: {score:.2f}/1.0")

            # Best and worst regimes
            if results['strongest_regime']:
                strongest_sharpe = regime_results[results['strongest_regime']]['sharpe_ratio']
                summary_parts.append(f"Strongest Regime: {results['strongest_regime']} (Sharpe: {strongest_sharpe:.2f})")

            if results['weakest_regime']:
                weakest_sharpe = regime_results[results['weakest_regime']]['sharpe_ratio']
                summary_parts.append(f"Weakest Regime: {results['weakest_regime']} (Sharpe: {weakest_sharpe:.2f})")

            # Consistency metrics
            consistency = results.get('consistency_metrics', {})
            if consistency:
                summary_parts.append(f"Regimes with Positive Sharpe: {consistency.get('positive_sharpe_pct', 0):.1%}")
                summary_parts.append(f"Minimum Win Rate: {consistency.get('win_rate_min', 0):.1%}")
                summary_parts.append(f"Worst Max Drawdown: {consistency.get('max_drawdown_worst', 0):.1%}")

            return "\n".join(summary_parts)

        except Exception as e:
            return f"Error generating regime summary: {e}"


# Example usage and testing
if __name__ == "__main__":
    def test_regime_validation():
        print("=== Regime Validation Demo ===")

        validator = RegimeValidator()

        # Create synthetic market data
        market_data = validator.create_synthetic_market_data('2020-01-01', '2023-12-31')
        print(f"Created market data: {list(market_data.columns)}")

        # Create synthetic strategy returns with different regime sensitivities
        np.random.seed(42)
        spy_returns = market_data['SPY'].pct_change().dropna()

        # Strategy that performs better in low vol environments
        strategy_returns = (
            0.0003 +  # Base alpha
            0.5 * spy_returns +  # Market exposure
            -0.02 * (market_data['VIX'] - 20) / 20 +  # VIX sensitivity
            np.random.normal(0, 0.01, len(spy_returns))  # Noise
        ).dropna()

        strategy_returns.name = 'strategy'

        # Run regime analysis
        results = validator.test_edge_persistence(strategy_returns, market_data)

        print("\n=== Regime Analysis Results ===")
        print(f"Edge is Robust: {results['edge_is_robust']}")
        print(f"Robustness Score: {results.get('robustness_score', 0):.2f}")
        print(f"Strongest Regime: {results['strongest_regime']}")
        print(f"Weakest Regime: {results['weakest_regime']}")

        # Show individual regime results
        if results['regime_results']:
            print("\n=== Individual Regime Performance ===")
            for regime, metrics in results['regime_results'].items():
                print(f"{regime}:")
                print(f"  Sample Size: {metrics['sample_size']}")
                print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
                print(f"  Win Rate: {metrics['win_rate']:.1%}")
                print(f"  Max Drawdown: {metrics['max_drawdown']:.1%}")
                print(f"  Ann. Return: {metrics['annualized_return']:.1%}")

        # Generate summary
        summary = validator.get_regime_summary(results)
        print("\n=== Summary ===")
        print(summary)

    test_regime_validation()