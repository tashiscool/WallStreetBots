"""Risk of ruin calculations and capital allocation optimization.

Provides analytical upper bounds for risk of ruin and optimizes
position sizing to maintain acceptable ruin probabilities.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
from scipy import stats, optimize
import warnings

logger = logging.getLogger(__name__)


@dataclass
class RiskOfRuinResult:
    """Results from risk of ruin calculation."""
    probability_of_ruin: float
    analytical_upper_bound: float
    monte_carlo_estimate: float
    time_to_ruin_median: Optional[float]
    safe_position_size: float
    current_position_size: float
    recommendation: str
    confidence_interval: Tuple[float, float]


@dataclass
class TailRiskMetrics:
    """Tail risk analysis results."""
    var_95: float
    var_99: float
    var_999: float
    cvar_95: float
    cvar_99: float
    expected_shortfall: float
    tail_ratio: float
    max_drawdown_estimate: float


class RiskOfRuinCalculator:
    """Calculate risk of ruin using multiple methods."""

    def __init__(self, initial_capital: float):
        self.initial_capital = initial_capital
        self.target_ruin_probability = 0.02  # 2% maximum acceptable ruin risk

    def calculate_risk_of_ruin(self, returns: pd.Series,
                             position_size: float,
                             confidence_level: float = 0.95) -> RiskOfRuinResult:
        """Calculate comprehensive risk of ruin analysis.

        Args:
            returns: Historical strategy returns
            position_size: Current position size (fraction of capital)
            confidence_level: Confidence level for estimates

        Returns:
            RiskOfRuinResult with multiple ruin probability estimates
        """
        if len(returns) < 30:
            raise ValueError("Need at least 30 return observations")

        # Calculate key statistics
        win_rate = (returns > 0).mean()
        avg_win = returns[returns > 0].mean() if (returns > 0).any() else 0
        avg_loss = returns[returns <= 0].mean() if (returns <= 0).any() else 0

        # Analytical approximation (Gambler's Ruin formula adaptation)
        analytical_ruin = self._analytical_risk_of_ruin(
            win_rate, avg_win, avg_loss, position_size
        )

        # Monte Carlo simulation
        mc_ruin, mc_confidence = self._monte_carlo_risk_of_ruin(
            returns, position_size, confidence_level
        )

        # Time to ruin analysis
        median_time_to_ruin = self._estimate_time_to_ruin(
            returns, position_size
        )

        # Calculate safe position size
        safe_size = self._calculate_safe_position_size(
            returns, self.target_ruin_probability
        )

        # Generate recommendation
        recommendation = self._generate_recommendation(
            analytical_ruin, mc_ruin, position_size, safe_size
        )

        return RiskOfRuinResult(
            probability_of_ruin=mc_ruin,
            analytical_upper_bound=analytical_ruin,
            monte_carlo_estimate=mc_ruin,
            time_to_ruin_median=median_time_to_ruin,
            safe_position_size=safe_size,
            current_position_size=position_size,
            recommendation=recommendation,
            confidence_interval=mc_confidence
        )

    def _analytical_risk_of_ruin(self, win_rate: float, avg_win: float,
                               avg_loss: float, position_size: float) -> float:
        """Calculate analytical risk of ruin approximation."""
        if avg_loss == 0:
            return 0.0  # No losses = no ruin

        # Calculate probability parameters
        p = win_rate  # Probability of win
        q = 1 - p     # Probability of loss

        # Average gain/loss per bet as fraction of capital
        a = avg_win * position_size    # Average gain when winning
        b = abs(avg_loss) * position_size  # Average loss when losing

        if a <= 0 or b <= 0:
            return 1.0  # Invalid parameters lead to certain ruin

        # Expected value per bet
        expected_return = p * a - q * b

        if expected_return <= 0:
            return 1.0  # Negative expectancy leads to certain ruin

        # Simplified gambler's ruin adaptation
        # This assumes discrete bets and uses the classic formula
        # R = (q*b / p*a)^(capital / unit_bet_size)

        # Risk ratio
        risk_ratio = (q * b) / (p * a)

        if risk_ratio >= 1:
            return 1.0  # Unfavorable game

        # Number of bets before ruin (simplified)
        # Assumes each bet risks fixed fraction of remaining capital
        ruin_threshold = 0.1  # 10% of initial capital remaining = "ruin"

        # Use logarithmic approximation for continuous betting
        # P(ruin) ≈ exp(-2μN/o²) where μ is drift, o is volatility, N is number of periods

        if risk_ratio > 0:
            # Approximate using exponential decay
            ruin_prob = min(1.0, risk_ratio ** (1.0 / position_size))
        else:
            ruin_prob = 0.0

        return min(1.0, max(0.0, ruin_prob))

    def _monte_carlo_risk_of_ruin(self, returns: pd.Series, position_size: float,
                                confidence_level: float, num_simulations: int = 10000) -> Tuple[float, Tuple[float, float]]:
        """Monte Carlo simulation of risk of ruin."""
        np.random.seed(42)  # For reproducible results

        # Parameters for return distribution
        mean_return = returns.mean()
        std_return = returns.std()

        # Number of periods to simulate (1 year)
        periods = 252

        # Track ruin events
        ruin_events = 0
        time_to_ruin = []

        for _ in range(num_simulations):
            capital = 1.0  # Start with normalized capital

            for period in range(periods):
                # Generate return for this period
                period_return = np.random.normal(mean_return, std_return)

                # Apply position sizing
                portfolio_return = period_return * position_size

                # Update capital
                capital *= (1 + portfolio_return)

                # Check for ruin (capital below 10% of initial)
                if capital <= 0.1:
                    ruin_events += 1
                    time_to_ruin.append(period)
                    break

        # Calculate ruin probability
        ruin_probability = ruin_events / num_simulations

        # Calculate confidence interval using Wilson score interval
        if ruin_events > 0:
            z = stats.norm.ppf((1 + confidence_level) / 2)
            n = num_simulations
            p_hat = ruin_probability

            denominator = 1 + z**2 / n
            center = (p_hat + z**2 / (2*n)) / denominator
            margin = z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4*n)) / n) / denominator

            ci_lower = max(0, center - margin)
            ci_upper = min(1, center + margin)
        else:
            ci_lower = 0
            ci_upper = 3 / num_simulations  # Rule of three for zero events

        return ruin_probability, (ci_lower, ci_upper)

    def _estimate_time_to_ruin(self, returns: pd.Series, position_size: float) -> Optional[float]:
        """Estimate median time to ruin if ruin occurs."""
        # Simple approximation based on negative drift
        mean_return = returns.mean()
        std_return = returns.std()

        if mean_return >= 0:
            return None  # Positive expectancy = no expected ruin

        # Portfolio level statistics
        portfolio_mean = mean_return * position_size
        portfolio_std = std_return * position_size

        # Time to hit 10% of capital (ln(0.1) = -2.3)
        # Using first passage time approximation
        barrier = np.log(0.1)  # Log of ruin level

        if portfolio_mean < 0:
            # Expected time to hit barrier (simplified)
            expected_time = -barrier / portfolio_mean
            return max(1, expected_time)  # At least 1 period

        return None

    def _calculate_safe_position_size(self, returns: pd.Series,
                                    target_ruin_prob: float) -> float:
        """Calculate maximum safe position size for target ruin probability."""

        def ruin_objective(position_size):
            """Objective function: minimize difference from target ruin probability."""
            if position_size <= 0 or position_size > 1:
                return 1.0  # Penalty for invalid sizes

            try:
                _, mc_confidence = self._monte_carlo_risk_of_ruin(
                    returns, position_size, 0.95, num_simulations=1000  # Faster for optimization
                )
                estimated_ruin = mc_confidence[1]  # Use upper bound of confidence interval
                return abs(estimated_ruin - target_ruin_prob)
            except Exception:
                return 1.0

        # Binary search for optimal position size
        try:
            result = optimize.minimize_scalar(
                ruin_objective,
                bounds=(0.01, 0.5),  # Search between 1% and 50% position size
                method='bounded'
            )

            optimal_size = result.x if result.success else 0.1

            # Verify the result makes sense
            if optimal_size > 0.5:
                optimal_size = 0.1  # Conservative fallback

            return optimal_size

        except Exception as e:
            logger.warning(f"Safe position size calculation failed: {e}")
            return 0.1  # Conservative default

    def _generate_recommendation(self, analytical_ruin: float, mc_ruin: float,
                               current_size: float, safe_size: float) -> str:
        """Generate risk management recommendation."""
        max_ruin = max(analytical_ruin, mc_ruin)

        if max_ruin > 0.05:  # 5% threshold
            return f"HIGH RISK: Reduce position size from {current_size:.1%} to {safe_size:.1%}"
        elif max_ruin > 0.02:  # 2% threshold
            return f"MODERATE RISK: Consider reducing position size to {safe_size:.1%}"
        elif current_size < safe_size * 0.8:  # Significantly under-leveraged
            return f"CONSERVATIVE: Could increase position size to {safe_size:.1%}"
        else:
            return "ACCEPTABLE RISK: Current position size is appropriate"


class TailRiskAnalyzer:
    """Analyzes tail risk and extreme loss scenarios."""

    def __init__(self):
        self.confidence_levels = [0.95, 0.99, 0.999]

    def analyze_tail_risk(self, returns: pd.Series, position_size: float = 1.0) -> TailRiskMetrics:
        """Comprehensive tail risk analysis.

        Args:
            returns: Strategy returns
            position_size: Position sizing factor

        Returns:
            TailRiskMetrics with VaR, CVaR, and other tail measures
        """
        if len(returns) < 30:
            raise ValueError("Need at least 30 return observations")

        # Scale returns by position size
        scaled_returns = returns * position_size

        # Calculate VaR at different confidence levels
        var_95 = self._calculate_var(scaled_returns, 0.95)
        var_99 = self._calculate_var(scaled_returns, 0.99)
        var_999 = self._calculate_var(scaled_returns, 0.999)

        # Calculate Conditional VaR (Expected Shortfall)
        cvar_95 = self._calculate_cvar(scaled_returns, 0.95)
        cvar_99 = self._calculate_cvar(scaled_returns, 0.99)

        # Expected shortfall (average of worst 5% outcomes)
        expected_shortfall = scaled_returns.quantile(0.05)

        # Tail ratio (90th percentile / 10th percentile)
        tail_ratio = scaled_returns.quantile(0.9) / abs(scaled_returns.quantile(0.1))

        # Maximum drawdown estimate
        max_dd_estimate = self._estimate_max_drawdown(scaled_returns)

        return TailRiskMetrics(
            var_95=var_95,
            var_99=var_99,
            var_999=var_999,
            cvar_95=cvar_95,
            cvar_99=cvar_99,
            expected_shortfall=expected_shortfall,
            tail_ratio=tail_ratio,
            max_drawdown_estimate=max_dd_estimate
        )

    def _calculate_var(self, returns: pd.Series, confidence_level: float) -> float:
        """Calculate Value at Risk."""
        return -returns.quantile(1 - confidence_level)

    def _calculate_cvar(self, returns: pd.Series, confidence_level: float) -> float:
        """Calculate Conditional Value at Risk (Expected Shortfall)."""
        var_threshold = returns.quantile(1 - confidence_level)
        tail_returns = returns[returns <= var_threshold]

        if len(tail_returns) == 0:
            return self._calculate_var(returns, confidence_level)

        return -tail_returns.mean()

    def _estimate_max_drawdown(self, returns: pd.Series) -> float:
        """Estimate maximum drawdown using Monte Carlo."""
        np.random.seed(42)

        # Parameters for return distribution
        mean_return = returns.mean()
        std_return = returns.std()

        max_drawdowns = []

        # Run multiple simulations
        for _ in range(1000):
            # Generate return path
            simulated_returns = np.random.normal(mean_return, std_return, len(returns))

            # Calculate cumulative returns
            cumulative = (1 + pd.Series(simulated_returns)).cumprod()

            # Calculate drawdowns
            running_max = cumulative.expanding().max()
            drawdowns = (cumulative - running_max) / running_max

            max_drawdowns.append(drawdowns.min())

        # Return 95th percentile of maximum drawdowns (conservative estimate)
        return -np.percentile(max_drawdowns, 95)


class DynamicRiskManager:
    """Dynamic risk management with real-time adjustments."""

    def __init__(self, initial_capital: float):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.risk_calculator = RiskOfRuinCalculator(initial_capital)
        self.tail_analyzer = TailRiskAnalyzer()

        # Risk limits
        self.risk_limits = {
            'max_ruin_probability': 0.02,  # 2%
            'max_var_95': 0.05,           # 5% daily VaR
            'max_drawdown': 0.20,         # 20% max drawdown
            'min_sharpe_ratio': 1.0       # Minimum Sharpe ratio
        }

        # Position sizing parameters
        self.base_position_size = 0.1     # 10% base allocation
        self.max_position_size = 0.3      # 30% maximum allocation
        self.min_position_size = 0.01     # 1% minimum allocation

    def calculate_optimal_position_size(self, strategy_returns: pd.Series,
                                      current_market_regime: str = 'normal') -> Dict[str, Any]:
        """Calculate optimal position size considering multiple risk factors.

        Args:
            strategy_returns: Historical strategy returns
            current_market_regime: Current market regime ('bull', 'bear', 'normal', 'volatile')

        Returns:
            Dictionary with position sizing recommendation and analysis
        """

        # Start with base position size
        recommended_size = self.base_position_size

        # Risk of ruin analysis
        ruin_analysis = self.risk_calculator.calculate_risk_of_ruin(
            strategy_returns, recommended_size
        )

        # Tail risk analysis
        tail_analysis = self.tail_analyzer.analyze_tail_risk(
            strategy_returns, recommended_size
        )

        # Adjust for risk of ruin
        if ruin_analysis.probability_of_ruin > self.risk_limits['max_ruin_probability']:
            recommended_size = ruin_analysis.safe_position_size

        # Adjust for VaR limits
        if tail_analysis.var_95 > self.risk_limits['max_var_95']:
            var_adjustment = self.risk_limits['max_var_95'] / tail_analysis.var_95
            recommended_size *= var_adjustment

        # Adjust for market regime
        regime_multiplier = self._get_regime_multiplier(current_market_regime)
        recommended_size *= regime_multiplier

        # Apply bounds
        recommended_size = max(self.min_position_size,
                             min(self.max_position_size, recommended_size))

        # Calculate expected metrics with recommended size
        final_ruin = self.risk_calculator.calculate_risk_of_ruin(
            strategy_returns, recommended_size
        )

        final_tail = self.tail_analyzer.analyze_tail_risk(
            strategy_returns, recommended_size
        )

        return {
            'recommended_position_size': recommended_size,
            'base_size': self.base_position_size,
            'regime_adjustment': regime_multiplier,
            'risk_of_ruin': final_ruin.probability_of_ruin,
            'var_95_percent': final_tail.var_95 * 100,
            'expected_sharpe': self._estimate_sharpe(strategy_returns, recommended_size),
            'max_drawdown_estimate': final_tail.max_drawdown_estimate,
            'recommendation_reason': self._explain_sizing_decision(
                ruin_analysis, tail_analysis, regime_multiplier
            ),
            'risk_metrics': {
                'ruin_probability': final_ruin.probability_of_ruin,
                'var_95': final_tail.var_95,
                'cvar_95': final_tail.cvar_95,
                'tail_ratio': final_tail.tail_ratio
            }
        }

    def _get_regime_multiplier(self, regime: str) -> float:
        """Get position size multiplier based on market regime."""
        multipliers = {
            'bull': 1.2,       # Increase size in bull markets
            'normal': 1.0,     # Normal size
            'bear': 0.7,       # Reduce size in bear markets
            'volatile': 0.6,   # Significantly reduce in volatile markets
            'crisis': 0.3      # Minimal size in crisis
        }

        return multipliers.get(regime, 1.0)

    def _estimate_sharpe(self, returns: pd.Series, position_size: float) -> float:
        """Estimate Sharpe ratio with given position size."""
        scaled_returns = returns * position_size

        if scaled_returns.std() == 0:
            return 0.0

        return scaled_returns.mean() / scaled_returns.std() * np.sqrt(252)

    def _explain_sizing_decision(self, ruin_analysis: RiskOfRuinResult,
                               tail_analysis: TailRiskMetrics,
                               regime_multiplier: float) -> str:
        """Explain the position sizing decision."""
        reasons = []

        if ruin_analysis.probability_of_ruin > 0.01:
            reasons.append(f"Risk of ruin ({ruin_analysis.probability_of_ruin:.1%}) considered")

        if tail_analysis.var_95 > 0.03:
            reasons.append(f"High VaR ({tail_analysis.var_95:.1%}) factored in")

        if regime_multiplier != 1.0:
            if regime_multiplier > 1.0:
                reasons.append("Increased for favorable market regime")
            else:
                reasons.append("Reduced for unfavorable market regime")

        if not reasons:
            return "Standard position sizing based on risk limits"

        return "; ".join(reasons)

    def update_capital(self, new_capital: float):
        """Update current capital level."""
        self.current_capital = new_capital
        self.risk_calculator = RiskOfRuinCalculator(new_capital)

    def get_risk_dashboard(self) -> Dict[str, Any]:
        """Get current risk dashboard metrics."""
        return {
            'current_capital': self.current_capital,
            'capital_change_pct': (self.current_capital - self.initial_capital) / self.initial_capital,
            'risk_limits': self.risk_limits,
            'position_size_range': {
                'min': self.min_position_size,
                'max': self.max_position_size,
                'base': self.base_position_size
            },
            'last_updated': datetime.now().isoformat()
        }