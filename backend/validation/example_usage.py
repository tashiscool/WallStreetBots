"""Example usage of the validation framework.

Demonstrates how to use the validation framework to evaluate trading strategies
and make deployment decisions.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from backend.validation import ValidationOrchestrator, ValidationCriteria


class ExampleStrategy:
    """Example strategy for demonstration purposes."""

    def __init__(self, name: str = "Example Strategy", lookback: int = 20):
        self.name = name
        self.lookback = lookback
        self.description = "Example momentum strategy for validation testing"
        self.parameters = {"lookback": lookback}

    def backtest(self, start: str = "2020-01-01", end: str = "2024-12-31"):
        """Simulate strategy backtest with realistic returns."""
        date_range = pd.date_range(start=start, end=end, freq='D')

        # Generate realistic returns with momentum bias
        np.random.seed(42)  # For reproducible results

        # Base market returns
        market_returns = np.random.normal(0.0005, 0.015, len(date_range))

        # Add momentum factor
        momentum_signal = np.random.normal(0.0002, 0.005, len(date_range))

        # Strategy returns with some alpha
        strategy_returns = market_returns + momentum_signal * 0.3

        # Add some regime-dependent behavior
        volatility_regime = np.random.choice([0, 1], len(date_range), p=[0.8, 0.2])
        strategy_returns[volatility_regime == 1] *= 1.5  # Higher volatility periods

        returns = pd.Series(strategy_returns, index=date_range)

        # Create backtest results object
        class BacktestResults:
            def __init__(self, returns):
                self.returns = returns
                self.total_return = (1 + returns).prod() - 1
                self.sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)
                self.max_drawdown = self._calculate_max_drawdown(returns)

            def _calculate_max_drawdown(self, returns):
                cumulative = (1 + returns).cumprod()
                running_max = cumulative.expanding().max()
                drawdowns = (cumulative - running_max) / running_max
                return drawdowns.min()

        return BacktestResults(returns)

    def backtest_with_capital(self, capital: float):
        """Backtest with specific capital amount."""
        results = self.backtest()
        # Scale returns based on capital (simplified)
        return results


class PoorPerformingStrategy(ExampleStrategy):
    """Example of a poorly performing strategy."""

    def __init__(self):
        super().__init__("Poor Strategy", lookback=10)
        self.description = "Example strategy with poor performance"

    def backtest(self, start: str = "2020-01-01", end: str = "2024-12-31"):
        """Generate poor performing returns."""
        date_range = pd.date_range(start=start, end=end, freq='D')

        np.random.seed(123)
        # Generate consistently poor returns
        poor_returns = np.random.normal(-0.0002, 0.025, len(date_range))

        # Add occasional large losses
        large_loss_days = np.random.choice([0, 1], len(date_range), p=[0.95, 0.05])
        poor_returns[large_loss_days == 1] -= 0.05

        returns = pd.Series(poor_returns, index=date_range)

        class BacktestResults:
            def __init__(self, returns):
                self.returns = returns
                self.total_return = (1 + returns).prod() - 1
                self.sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)

        return BacktestResults(returns)


def example_single_strategy_validation():
    """Example of validating a single strategy."""
    logger.info("=== Single Strategy Validation Example ===")

    # Create a custom validation criteria
    criteria = ValidationCriteria(
        min_sharpe_ratio=0.8,  # Slightly lower threshold for demo
        max_drawdown=0.20,     # 20% max drawdown
        min_factor_adjusted_alpha=0.03  # 3% alpha
    )

    # Initialize orchestrator with custom criteria
    orchestrator = ValidationOrchestrator(criteria)

    # Create strategy
    strategy = ExampleStrategy("Momentum Strategy")

    # Run validation
    try:
        results = orchestrator.validate_strategy(
            strategy,
            start_date="2020-01-01",
            end_date="2024-12-31"
        )

        # Print key results
        print("\n" + "="*60)
        print("VALIDATION RESULTS")
        print("="*60)

        # Strategy info
        if 'strategy_info' in results:
            print(f"Strategy: {results['strategy_info']['name']}")

        # Deployment decision
        if 'deployment_decision' in results:
            decision = results['deployment_decision']
            print(f"Recommendation: {decision['recommendation']}")
            print(f"Confidence: {decision.get('confidence_score', 0):.2%}")

            if 'decision_rationale' in decision:
                print(f"Rationale: {decision['decision_rationale']}")

        # Data quality
        if 'data_quality' in results:
            quality = results['data_quality']
            if 'returns_statistics' in quality:
                stats = quality['returns_statistics']
                print("\nStrategy Performance:")
                print(f"  - Annual Return: {stats['mean'] * 252:.2%}")
                print(f"  - Volatility: {stats['std'] * np.sqrt(252):.2%}")
                print(f"  - Sharpe Ratio: {stats['mean'] / stats['std'] * np.sqrt(252):.2f}")

        # Factor analysis
        if 'factor_analysis' in results and 'annualized_alpha' in results['factor_analysis']:
            alpha_results = results['factor_analysis']
            print("\nAlpha Analysis:")
            print(f"  - Annualized Alpha: {alpha_results['annualized_alpha']:.2%}")
            print(f"  - Alpha Significant: {alpha_results.get('alpha_significant', False)}")
            print(f"  - Data Source: {alpha_results.get('data_source', 'Unknown')}")

        # Regime testing
        if 'regime_testing' in results and 'profitable_regime_rate' in results['regime_testing']:
            regime_results = results['regime_testing']
            print("\nRegime Analysis:")
            print(f"  - Profitable Regime Rate: {regime_results['profitable_regime_rate']:.1%}")
            print(f"  - Edge is Robust: {regime_results.get('edge_is_robust', False)}")

        # Recommendations
        if 'deployment_decision' in results and 'deployment_recommendations' in results['deployment_decision']:
            recommendations = results['deployment_decision']['deployment_recommendations']
            if recommendations.get('immediate_actions'):
                print("\nImmediate Actions:")
                for action in recommendations['immediate_actions'][:3]:
                    print(f"  - {action}")

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"validation_results_{timestamp}.json"
        orchestrator.save_validation_results(results, filename)
        print(f"\nResults saved to: {filename}")

    except Exception as e:
        logger.error(f"Validation failed: {e}")
        print(f"Validation failed: {e}")


def example_multiple_strategy_validation():
    """Example of validating multiple strategies."""
    logger.info("=== Multiple Strategy Validation Example ===")

    # Create orchestrator
    orchestrator = ValidationOrchestrator()

    # Create multiple strategies
    strategies = {
        "Momentum_20": ExampleStrategy("Momentum 20-day", lookback=20),
        "Momentum_50": ExampleStrategy("Momentum 50-day", lookback=50),
        "Poor_Strategy": PoorPerformingStrategy()
    }

    try:
        # Run multi-strategy validation
        results = orchestrator.validate_multiple_strategies(
            strategies,
            start_date="2020-01-01",
            end_date="2024-12-31"
        )

        print("\n" + "="*60)
        print("MULTI-STRATEGY VALIDATION RESULTS")
        print("="*60)

        # Validation summary
        if 'validation_summary' in results:
            summary = results['validation_summary']
            print(f"Total Strategies: {summary['total_strategies']}")

            if 'overall_assessment' in summary:
                assessment = summary['overall_assessment']
                ready_pct = assessment.get('deployment_ready_percentage', 0)
                print(f"Deployment Ready: {ready_pct:.1%}")

        # Strategy rankings
        if 'comparative_analysis' in results:
            comparative = results['comparative_analysis']

            if 'strategy_rankings' in comparative:
                rankings = comparative['strategy_rankings']

                if 'sharpe_ratio' in rankings:
                    print("\nSharpe Ratio Rankings:")
                    for i, (name, sharpe) in enumerate(rankings['sharpe_ratio'][:3], 1):
                        print(f"  {i}. {name}: {sharpe:.2f}")

            if 'best_performers' in comparative:
                best = comparative['best_performers']
                print("\nBest Performers:")
                if best.get('highest_sharpe'):
                    print(f"  - Highest Sharpe: {best['highest_sharpe']}")
                if best.get('most_deployable'):
                    print(f"  - Most Deployable: {best['most_deployable']}")

            if 'deployment_readiness' in comparative:
                readiness = comparative['deployment_readiness']
                print("\nDeployment Readiness:")
                for status, count in readiness.items():
                    if count > 0:
                        print(f"  - {status}: {count}")

        # Ensemble analysis
        if 'ensemble_analysis' in results and 'portfolio_comparison' in results['ensemble_analysis']:
            ensemble = results['ensemble_analysis']
            comparison = ensemble['portfolio_comparison']

            print("\nEnsemble Analysis:")
            print(f"  - Equal Weight Sharpe: {comparison.get('equal_weight_sharpe', 0):.2f}")
            print(f"  - Optimized Sharpe: {comparison.get('optimized_sharpe', 0):.2f}")
            print(f"  - Diversification Benefit: {comparison.get('diversification_benefit', 0):.2f}")

            if 'optimal_ensemble' in ensemble:
                optimal = ensemble['optimal_ensemble']
                if 'selected_strategies' in optimal:
                    print(f"  - Optimal Strategies: {', '.join(optimal['selected_strategies'])}")

    except Exception as e:
        logger.error(f"Multi-strategy validation failed: {e}")
        print(f"Multi-strategy validation failed: {e}")


def example_custom_validation_criteria():
    """Example of using custom validation criteria."""
    logger.info("=== Custom Validation Criteria Example ===")

    # Create strict criteria for institutional deployment
    strict_criteria = ValidationCriteria(
        min_sharpe_ratio=1.5,           # Higher Sharpe requirement
        max_drawdown=0.10,              # Lower drawdown tolerance
        min_win_rate=0.55,              # Higher win rate requirement
        min_factor_adjusted_alpha=0.08, # Higher alpha requirement
        alpha_t_stat_threshold=2.5,     # Stricter significance
        min_regime_consistency=0.8,     # Must work in 80% of regimes
        min_cross_market_success=0.7,   # Higher cross-market success
        min_return_on_capital=0.20,     # Higher ROC requirement
        max_margin_calls=0              # Zero tolerance for margin calls
    )

    orchestrator = ValidationOrchestrator(strict_criteria)
    strategy = ExampleStrategy("High-Bar Strategy")

    try:
        results = orchestrator.validate_strategy(strategy)

        print("\n" + "="*60)
        print("STRICT CRITERIA VALIDATION")
        print("="*60)

        if 'deployment_decision' in results:
            decision = results['deployment_decision']
            print(f"Recommendation: {decision['recommendation']}")
            print(f"Confidence: {decision.get('confidence_score', 0):.2%}")

            # Show which criteria failed
            if 'component_scores' in decision:
                print("\nCriteria Performance:")
                for component, scores in decision['component_scores'].items():
                    print(f"  {component}:")
                    for metric, data in scores.items():
                        status = "✓" if data['passed'] else "✗"
                        print(f"    {status} {metric}: {data['value']:.3f} (threshold: {data['threshold']})")

    except Exception as e:
        logger.error(f"Strict validation failed: {e}")
        print(f"Strict validation failed: {e}")


if __name__ == "__main__":
    print("WallStreetBots Validation Framework Examples")
    print("=" * 50)

    # Run examples
    example_single_strategy_validation()
    print("\n" + "="*80 + "\n")

    example_multiple_strategy_validation()
    print("\n" + "="*80 + "\n")

    example_custom_validation_criteria()

    print("\n" + "="*50)
    print("Examples completed!")
    print("\nTo use the validation framework in your code:")
    print("1. Import: from backend.validation import ValidationOrchestrator")
    print("2. Create orchestrator: orchestrator = ValidationOrchestrator()")
    print("3. Validate strategy: results = orchestrator.validate_strategy(your_strategy)")
    print("4. Check recommendation: results['deployment_decision']['recommendation']")