"""
Phase 3: Unified Risk Management System
======================================

Comprehensive risk management framework that integrates portfolio-level
controls, Greeks-based exposure management, and real-time monitoring
across all Index Baseline strategies.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import warnings
from abc import ABC, abstractmethod

try:
    from scipy import stats
    from scipy.optimize import minimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("scipy not available, using simplified optimization")


class RiskLevel(Enum):
    """Risk level enumeration."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class PositionRisk:
    """Risk metrics for individual position."""
    symbol: str
    strategy: str
    position_size: float
    market_value: float
    portfolio_weight: float
    delta: float
    gamma: float
    theta: float
    vega: float
    var_1d: float  # 1-day Value at Risk
    cvar_1d: float  # 1-day Conditional VaR
    volatility: float
    correlation_exposure: float
    concentration_risk: float
    risk_level: RiskLevel


@dataclass
class PortfolioRisk:
    """Portfolio-level risk metrics."""
    total_value: float
    total_delta: float
    total_gamma: float
    total_theta: float
    total_vega: float
    portfolio_var_1d: float
    portfolio_cvar_1d: float
    portfolio_volatility: float
    sharpe_ratio: float
    max_drawdown: float
    concentration_hhi: float  # Herfindahl-Hirschman Index
    correlation_risk: float
    leverage_ratio: float
    risk_level: RiskLevel
    positions: List[PositionRisk] = field(default_factory=list)


@dataclass
class RiskLimit:
    """Risk limit definition."""
    name: str
    metric: str
    limit_value: float
    warning_threshold: float  # Percentage of limit (e.g., 0.8 = 80%)
    limit_type: str  # 'absolute', 'percentage', 'ratio'
    scope: str  # 'position', 'strategy', 'portfolio'
    enabled: bool = True


@dataclass
class RiskAlert:
    """Risk alert notification."""
    timestamp: datetime
    alert_type: str
    severity: RiskLevel
    message: str
    affected_positions: List[str]
    current_value: float
    limit_value: float
    recommended_action: str
    auto_action_taken: bool = False


class RiskCalculator(ABC):
    """Abstract base class for risk calculations."""

    @abstractmethod
    def calculate_var(self, returns: np.ndarray, confidence: float = 0.95) -> float:
        """Calculate Value at Risk."""
        pass

    @abstractmethod
    def calculate_cvar(self, returns: np.ndarray, confidence: float = 0.95) -> float:
        """Calculate Conditional Value at Risk."""
        pass


class ParametricRiskCalculator(RiskCalculator):
    """Parametric risk calculation methods."""

    def calculate_var(self, returns: np.ndarray, confidence: float = 0.95) -> float:
        """Calculate parametric VaR assuming normal distribution."""
        if len(returns) == 0:
            return 0.0

        mean_return = np.mean(returns)
        std_return = np.std(returns, ddof=1)

        if SCIPY_AVAILABLE:
            z_score = stats.norm.ppf(1 - confidence)
        else:
            # Approximate z-scores for common confidence levels
            z_scores = {0.90: -1.28, 0.95: -1.65, 0.99: -2.33}
            z_score = z_scores.get(confidence, -1.65)

        var = -(mean_return + z_score * std_return)
        return max(0, var)

    def calculate_cvar(self, returns: np.ndarray, confidence: float = 0.95) -> float:
        """Calculate parametric CVaR (Expected Shortfall)."""
        if len(returns) == 0:
            return 0.0

        var = self.calculate_var(returns, confidence)

        # For normal distribution, CVaR = VaR + sigma * phi(z) / (1-confidence)
        std_return = np.std(returns, ddof=1)

        if SCIPY_AVAILABLE:
            z_score = stats.norm.ppf(1 - confidence)
            phi_z = stats.norm.pdf(z_score)
        else:
            # Approximation for common confidence levels
            if confidence == 0.95:
                phi_z = 0.0484  # phi(-1.65)
            elif confidence == 0.99:
                phi_z = 0.0277  # phi(-2.33)
            else:
                phi_z = 0.0484  # Default approximation

        cvar = var + std_return * phi_z / (1 - confidence)
        return max(0, cvar)


class HistoricalRiskCalculator(RiskCalculator):
    """Historical simulation risk calculation methods."""

    def calculate_var(self, returns: np.ndarray, confidence: float = 0.95) -> float:
        """Calculate historical VaR."""
        if len(returns) == 0:
            return 0.0

        # Sort returns and find percentile
        sorted_returns = np.sort(returns)
        index = int((1 - confidence) * len(sorted_returns))
        var = -sorted_returns[index] if index < len(sorted_returns) else 0
        return max(0, var)

    def calculate_cvar(self, returns: np.ndarray, confidence: float = 0.95) -> float:
        """Calculate historical CVaR (mean of worst tail losses)."""
        if len(returns) == 0:
            return 0.0

        sorted_returns = np.sort(returns)
        index = int((1 - confidence) * len(sorted_returns))

        if index > 0:
            tail_losses = sorted_returns[:index]
            cvar = -np.mean(tail_losses)
        else:
            cvar = -sorted_returns[0] if len(sorted_returns) > 0 else 0

        return max(0, cvar)


class GreeksCalculator:
    """Options Greeks calculation for risk management."""

    def __init__(self):
        self.risk_free_rate = 0.02  # 2% risk-free rate

    def estimate_delta(self, spot: float, strike: float, time_to_exp: float,
                      option_type: str = 'call', volatility: float = 0.20) -> float:
        """Estimate option delta using simplified Black-Scholes."""

        if time_to_exp <= 0:
            if option_type.lower() == 'call':
                return 1.0 if spot > strike else 0.0
            else:
                return -1.0 if spot < strike else 0.0

        try:
            if SCIPY_AVAILABLE:
                # Simplified Black-Scholes delta
                d1 = (np.log(spot / strike) +
                     (self.risk_free_rate + 0.5 * volatility**2) * time_to_exp) / \
                     (volatility * np.sqrt(time_to_exp))

                if option_type.lower() == 'call':
                    delta = stats.norm.cdf(d1)
                else:
                    delta = stats.norm.cdf(d1) - 1
            else:
                # Simple approximation based on moneyness
                moneyness = spot / strike
                if option_type.lower() == 'call':
                    if moneyness > 1.2:  # Deep ITM
                        delta = 0.9
                    elif moneyness > 1.0:  # ITM
                        delta = 0.6 + (moneyness - 1.0) * 1.5
                    elif moneyness > 0.8:  # OTM
                        delta = 0.3 + (moneyness - 0.8) * 1.5
                    else:  # Deep OTM
                        delta = 0.1
                else:  # Put
                    delta = self.estimate_delta(spot, strike, time_to_exp, 'call', volatility) - 1

            return np.clip(delta, -1.0, 1.0)

        except Exception:
            return 0.5 if option_type.lower() == 'call' else -0.5

    def estimate_gamma(self, spot: float, strike: float, time_to_exp: float,
                      volatility: float = 0.20) -> float:
        """Estimate option gamma."""

        if time_to_exp <= 0:
            return 0.0

        try:
            if SCIPY_AVAILABLE:
                d1 = (np.log(spot / strike) +
                     (self.risk_free_rate + 0.5 * volatility**2) * time_to_exp) / \
                     (volatility * np.sqrt(time_to_exp))

                gamma = stats.norm.pdf(d1) / (spot * volatility * np.sqrt(time_to_exp))
            else:
                # Simple approximation - highest gamma near ATM
                moneyness = spot / strike
                if 0.95 <= moneyness <= 1.05:  # Near ATM
                    gamma = 0.05 / np.sqrt(time_to_exp)
                else:
                    gamma = 0.01 / np.sqrt(time_to_exp)

            return max(0, gamma)

        except Exception:
            return 0.01

    def estimate_theta(self, spot: float, strike: float, time_to_exp: float,
                      option_type: str = 'call', volatility: float = 0.20) -> float:
        """Estimate option theta (time decay)."""

        if time_to_exp <= 0:
            return 0.0

        try:
            # Simplified theta approximation
            option_value = max(0, spot - strike) if option_type.lower() == 'call' else max(0, strike - spot)
            time_value = option_value * volatility * np.sqrt(time_to_exp / (2 * np.pi))

            # Theta is approximately -time_value / time_to_exp
            theta = -time_value / (365 * time_to_exp) if time_to_exp > 0 else 0

            return theta

        except Exception:
            return -0.01  # Default small negative theta

    def estimate_vega(self, spot: float, strike: float, time_to_exp: float,
                     volatility: float = 0.20) -> float:
        """Estimate option vega (volatility sensitivity)."""

        if time_to_exp <= 0:
            return 0.0

        try:
            if SCIPY_AVAILABLE:
                d1 = (np.log(spot / strike) +
                     (self.risk_free_rate + 0.5 * volatility**2) * time_to_exp) / \
                     (volatility * np.sqrt(time_to_exp))

                vega = spot * stats.norm.pdf(d1) * np.sqrt(time_to_exp) / 100
            else:
                # Simple approximation
                vega = spot * 0.4 * np.sqrt(time_to_exp) / 100

            return max(0, vega)

        except Exception:
            return 0.1


class UnifiedRiskManager:
    """
    Unified Risk Management System for Index Baseline strategies.

    Provides portfolio-level risk management with:
    - VaR and CVaR calculation
    - Greeks-based exposure management
    - Position concentration limits
    - Real-time monitoring and alerts
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Risk calculators
        self.parametric_calc = ParametricRiskCalculator()
        self.historical_calc = HistoricalRiskCalculator()
        self.greeks_calc = GreeksCalculator()

        # Portfolio state
        self.positions: Dict[str, Dict] = {}
        self.portfolio_history: List[PortfolioRisk] = []

        # Risk limits
        self.risk_limits = self._initialize_default_limits()

        # Alert system
        self.active_alerts: List[RiskAlert] = []
        self.alert_callbacks: List[Callable] = []

    def _initialize_default_limits(self) -> Dict[str, RiskLimit]:
        """Initialize default risk limits."""

        limits = {
            # Portfolio-level limits
            'portfolio_var_1d': RiskLimit(
                name='Portfolio VaR 1D',
                metric='portfolio_var_1d',
                limit_value=0.02,  # 2% daily VaR
                warning_threshold=0.8,
                limit_type='percentage',
                scope='portfolio'
            ),

            'portfolio_delta': RiskLimit(
                name='Portfolio Delta Exposure',
                metric='total_delta',
                limit_value=100000,  # $100k delta equivalent
                warning_threshold=0.8,
                limit_type='absolute',
                scope='portfolio'
            ),

            'concentration_hhi': RiskLimit(
                name='Portfolio Concentration',
                metric='concentration_hhi',
                limit_value=0.25,  # Maximum HHI
                warning_threshold=0.8,
                limit_type='ratio',
                scope='portfolio'
            ),

            # Position-level limits
            'position_size': RiskLimit(
                name='Maximum Position Size',
                metric='portfolio_weight',
                limit_value=0.10,  # 10% of portfolio
                warning_threshold=0.8,
                limit_type='percentage',
                scope='position'
            ),

            'position_var': RiskLimit(
                name='Position VaR',
                metric='var_1d',
                limit_value=0.05,  # 5% position VaR
                warning_threshold=0.8,
                limit_type='percentage',
                scope='position'
            ),

            # Strategy-level limits
            'strategy_allocation': RiskLimit(
                name='Strategy Allocation',
                metric='strategy_weight',
                limit_value=0.30,  # 30% per strategy
                warning_threshold=0.8,
                limit_type='percentage',
                scope='strategy'
            )
        }

        return limits

    def add_position(self, symbol: str, strategy: str, position_data: Dict[str, Any]) -> None:
        """
        Add or update a position in the risk management system.

        Args:
            symbol: Position symbol
            strategy: Strategy name
            position_data: Dictionary containing position details
        """

        position_key = f"{strategy}_{symbol}"

        # Extract position data with defaults
        position_info = {
            'symbol': symbol,
            'strategy': strategy,
            'quantity': position_data.get('quantity', 0),
            'market_price': position_data.get('market_price', 0),
            'position_value': position_data.get('position_value', 0),
            'option_type': position_data.get('option_type', 'stock'),
            'strike': position_data.get('strike', 0),
            'expiry': position_data.get('expiry', None),
            'implied_volatility': position_data.get('implied_volatility', 0.20),
            'returns_history': position_data.get('returns_history', []),
            'timestamp': datetime.now()
        }

        self.positions[position_key] = position_info

        self.logger.info(f"Added/updated position: {position_key} "
                        f"Value: ${position_info['position_value']:,.0f}")

    def calculate_position_risk(self, position_key: str) -> PositionRisk:
        """Calculate comprehensive risk metrics for a position."""

        position = self.positions[position_key]

        # Basic position data
        symbol = position['symbol']
        strategy = position['strategy']
        position_value = abs(position['position_value'])

        # Portfolio weight
        total_portfolio_value = self.get_total_portfolio_value()
        portfolio_weight = position_value / total_portfolio_value if total_portfolio_value > 0 else 0

        # Calculate Greeks
        if position['option_type'] in ['call', 'put']:
            spot = position['market_price']
            strike = position['strike']
            expiry = position.get('expiry')

            if expiry:
                time_to_exp = max(0, (expiry - datetime.now()).days / 365.0) if isinstance(expiry, datetime) else 0.25
            else:
                time_to_exp = 0.25  # Default 3 months

            iv = position.get('implied_volatility', 0.20)

            delta = self.greeks_calc.estimate_delta(spot, strike, time_to_exp, position['option_type'], iv)
            gamma = self.greeks_calc.estimate_gamma(spot, strike, time_to_exp, iv)
            theta = self.greeks_calc.estimate_theta(spot, strike, time_to_exp, position['option_type'], iv)
            vega = self.greeks_calc.estimate_vega(spot, strike, time_to_exp, iv)

            # Scale by position size
            quantity = position['quantity']
            delta *= quantity
            gamma *= quantity
            theta *= quantity
            vega *= quantity

        else:
            # Stock position
            delta = position['quantity']
            gamma = 0
            theta = 0
            vega = 0

        # Risk calculations
        returns = np.array(position.get('returns_history', []))

        if len(returns) > 10:
            var_1d = self.parametric_calc.calculate_var(returns, 0.95) * position_value
            cvar_1d = self.parametric_calc.calculate_cvar(returns, 0.95) * position_value
            volatility = np.std(returns, ddof=1) if len(returns) > 1 else 0.02
        else:
            # Use default risk estimates
            volatility = 0.02
            var_1d = position_value * volatility * 1.65  # Approximate 95% VaR
            cvar_1d = position_value * volatility * 2.0   # Approximate CVaR

        # Correlation and concentration risk (simplified)
        correlation_exposure = min(portfolio_weight * 2, 1.0)  # Higher weight = higher correlation risk
        concentration_risk = portfolio_weight

        # Risk level assessment
        risk_factors = []
        if portfolio_weight > 0.15:
            risk_factors.append("High concentration")
        if var_1d / position_value > 0.05:
            risk_factors.append("High volatility")
        if abs(delta) > position_value * 0.8:
            risk_factors.append("High delta exposure")

        if len(risk_factors) >= 2:
            risk_level = RiskLevel.HIGH
        elif len(risk_factors) == 1:
            risk_level = RiskLevel.MEDIUM
        else:
            risk_level = RiskLevel.LOW

        return PositionRisk(
            symbol=symbol,
            strategy=strategy,
            position_size=position['quantity'],
            market_value=position_value,
            portfolio_weight=portfolio_weight,
            delta=delta,
            gamma=gamma,
            theta=theta,
            vega=vega,
            var_1d=var_1d,
            cvar_1d=cvar_1d,
            volatility=volatility,
            correlation_exposure=correlation_exposure,
            concentration_risk=concentration_risk,
            risk_level=risk_level
        )

    def calculate_portfolio_risk(self) -> PortfolioRisk:
        """Calculate comprehensive portfolio risk metrics."""

        if not self.positions:
            return PortfolioRisk(
                total_value=0,
                total_delta=0,
                total_gamma=0,
                total_theta=0,
                total_vega=0,
                portfolio_var_1d=0,
                portfolio_cvar_1d=0,
                portfolio_volatility=0,
                sharpe_ratio=0,
                max_drawdown=0,
                concentration_hhi=0,
                correlation_risk=0,
                leverage_ratio=1.0,
                risk_level=RiskLevel.LOW
            )

        # Calculate position risks
        position_risks = []
        for position_key in self.positions:
            position_risk = self.calculate_position_risk(position_key)
            position_risks.append(position_risk)

        # Aggregate portfolio metrics
        total_value = sum(p.market_value for p in position_risks)
        total_delta = sum(p.delta for p in position_risks)
        total_gamma = sum(p.gamma for p in position_risks)
        total_theta = sum(p.theta for p in position_risks)
        total_vega = sum(p.vega for p in position_risks)

        # Portfolio VaR (simplified - assumes some diversification)
        individual_var = sum(p.var_1d**2 for p in position_risks)
        diversification_factor = 0.8  # Assume 20% diversification benefit
        portfolio_var = np.sqrt(individual_var) * diversification_factor

        # Portfolio CVaR
        individual_cvar = sum(p.cvar_1d**2 for p in position_risks)
        portfolio_cvar = np.sqrt(individual_cvar) * diversification_factor

        # Portfolio volatility (weighted average)
        if total_value > 0:
            portfolio_volatility = sum(p.volatility * p.portfolio_weight for p in position_risks)
        else:
            portfolio_volatility = 0

        # Concentration risk (HHI)
        weights = [p.portfolio_weight for p in position_risks]
        concentration_hhi = sum(w**2 for w in weights)

        # Correlation risk (simplified)
        strategy_weights = {}
        for p in position_risks:
            strategy_weights[p.strategy] = strategy_weights.get(p.strategy, 0) + p.portfolio_weight

        correlation_risk = max(strategy_weights.values()) if strategy_weights else 0

        # Risk level assessment
        risk_factors = []
        if portfolio_var / total_value > 0.02 if total_value > 0 else False:
            risk_factors.append("High VaR")
        if concentration_hhi > 0.25:
            risk_factors.append("High concentration")
        if correlation_risk > 0.4:
            risk_factors.append("High correlation risk")
        if abs(total_delta) > total_value * 0.1 if total_value > 0 else False:
            risk_factors.append("High delta exposure")

        if len(risk_factors) >= 3:
            risk_level = RiskLevel.CRITICAL
        elif len(risk_factors) >= 2:
            risk_level = RiskLevel.HIGH
        elif len(risk_factors) >= 1:
            risk_level = RiskLevel.MEDIUM
        else:
            risk_level = RiskLevel.LOW

        # Portfolio performance metrics (simplified)
        sharpe_ratio = 0  # Would need historical portfolio returns
        max_drawdown = 0  # Would need historical portfolio values

        portfolio_risk = PortfolioRisk(
            total_value=total_value,
            total_delta=total_delta,
            total_gamma=total_gamma,
            total_theta=total_theta,
            total_vega=total_vega,
            portfolio_var_1d=portfolio_var,
            portfolio_cvar_1d=portfolio_cvar,
            portfolio_volatility=portfolio_volatility,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            concentration_hhi=concentration_hhi,
            correlation_risk=correlation_risk,
            leverage_ratio=1.0,  # Simplified
            risk_level=risk_level,
            positions=position_risks
        )

        # Store in history
        self.portfolio_history.append(portfolio_risk)

        # Keep only recent history (last 100 calculations)
        if len(self.portfolio_history) > 100:
            self.portfolio_history = self.portfolio_history[-100:]

        return portfolio_risk

    def check_risk_limits(self) -> List[RiskAlert]:
        """Check all risk limits and generate alerts."""

        alerts = []
        current_portfolio = self.calculate_portfolio_risk()

        # Check portfolio-level limits
        for limit_name, limit in self.risk_limits.items():
            if not limit.enabled or limit.scope != 'portfolio':
                continue

            current_value = getattr(current_portfolio, limit.metric, 0)

            # Convert percentage limits
            if limit.limit_type == 'percentage':
                if limit.metric in ['portfolio_var_1d', 'portfolio_cvar_1d']:
                    # VaR limits as percentage of portfolio value
                    limit_absolute = limit.limit_value * current_portfolio.total_value
                    warning_absolute = limit_absolute * limit.warning_threshold
                    current_absolute = current_value
                else:
                    limit_absolute = limit.limit_value
                    warning_absolute = limit_absolute * limit.warning_threshold
                    current_absolute = current_value
            else:
                limit_absolute = limit.limit_value
                warning_absolute = limit_absolute * limit.warning_threshold
                current_absolute = current_value

            # Check for violations
            severity = None
            message = ""
            action = ""

            if current_absolute > limit_absolute:
                severity = RiskLevel.CRITICAL
                message = f"Portfolio limit exceeded: {limit.name}"
                action = f"Reduce exposure immediately - Current: {current_absolute:.4f}, Limit: {limit_absolute:.4f}"
            elif current_absolute > warning_absolute:
                severity = RiskLevel.HIGH
                message = f"Portfolio limit warning: {limit.name}"
                action = f"Monitor closely - Current: {current_absolute:.4f}, Warning: {warning_absolute:.4f}"

            if severity:
                alert = RiskAlert(
                    timestamp=datetime.now(),
                    alert_type=f"LIMIT_VIOLATION_{limit_name.upper()}",
                    severity=severity,
                    message=message,
                    affected_positions=[],
                    current_value=current_absolute,
                    limit_value=limit_absolute,
                    recommended_action=action
                )
                alerts.append(alert)

        # Check position-level limits
        for position_risk in current_portfolio.positions:
            for limit_name, limit in self.risk_limits.items():
                if not limit.enabled or limit.scope != 'position':
                    continue

                current_value = getattr(position_risk, limit.metric, 0)

                if current_value > limit.limit_value:
                    alert = RiskAlert(
                        timestamp=datetime.now(),
                        alert_type=f"POSITION_LIMIT_{limit_name.upper()}",
                        severity=RiskLevel.HIGH,
                        message=f"Position limit exceeded: {limit.name} for {position_risk.symbol}",
                        affected_positions=[position_risk.symbol],
                        current_value=current_value,
                        limit_value=limit.limit_value,
                        recommended_action=f"Reduce {position_risk.symbol} position size"
                    )
                    alerts.append(alert)

        # Update active alerts
        self.active_alerts.extend(alerts)

        # Remove old alerts (keep only last 24 hours)
        cutoff_time = datetime.now() - timedelta(hours=24)
        self.active_alerts = [a for a in self.active_alerts if a.timestamp >= cutoff_time]

        # Trigger alert callbacks
        for alert in alerts:
            for callback in self.alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    self.logger.error(f"Alert callback failed: {e}")

        if alerts:
            self.logger.warning(f"Generated {len(alerts)} risk alerts")

        return alerts

    def get_total_portfolio_value(self) -> float:
        """Get total portfolio value."""
        return sum(abs(pos.get('position_value', 0)) for pos in self.positions.values())

    def add_alert_callback(self, callback: Callable[[RiskAlert], None]) -> None:
        """Add callback function for risk alerts."""
        self.alert_callbacks.append(callback)

    def get_risk_summary(self) -> Dict[str, Any]:
        """Get comprehensive risk summary."""

        portfolio_risk = self.calculate_portfolio_risk()

        # Recent alerts summary
        recent_alerts = [a for a in self.active_alerts
                        if a.timestamp >= datetime.now() - timedelta(hours=1)]

        alert_summary = {
            'total_active_alerts': len(self.active_alerts),
            'recent_alerts': len(recent_alerts),
            'critical_alerts': len([a for a in self.active_alerts if a.severity == RiskLevel.CRITICAL]),
            'high_alerts': len([a for a in self.active_alerts if a.severity == RiskLevel.HIGH])
        }

        # Position summary
        position_summary = {
            'total_positions': len(self.positions),
            'high_risk_positions': len([p for p in portfolio_risk.positions if p.risk_level == RiskLevel.HIGH]),
            'strategies': list({p.strategy for p in portfolio_risk.positions}),
            'largest_position': max([p.portfolio_weight for p in portfolio_risk.positions], default=0)
        }

        return {
            'timestamp': datetime.now(),
            'portfolio_risk': portfolio_risk,
            'alert_summary': alert_summary,
            'position_summary': position_summary,
            'risk_limits_status': {name: limit.enabled for name, limit in self.risk_limits.items()}
        }

    def generate_risk_report(self) -> str:
        """Generate comprehensive risk management report."""

        summary = self.get_risk_summary()
        portfolio = summary['portfolio_risk']

        report = []
        report.append("=" * 80)
        report.append("UNIFIED RISK MANAGEMENT REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        # Portfolio Overview
        report.append("📊 PORTFOLIO OVERVIEW:")
        report.append(f"  Total Value: ${portfolio.total_value:,.0f}")
        report.append(f"  Number of Positions: {len(portfolio.positions)}")
        report.append(f"  Risk Level: {portfolio.risk_level.value}")
        report.append("")

        # Risk Metrics
        report.append("⚠️ RISK METRICS:")
        report.append(f"  Portfolio VaR (1D): ${portfolio.portfolio_var_1d:,.0f} "
                     f"({portfolio.portfolio_var_1d/portfolio.total_value:.2%} of portfolio)")
        report.append(f"  Portfolio CVaR (1D): ${portfolio.portfolio_cvar_1d:,.0f}")
        report.append(f"  Portfolio Volatility: {portfolio.portfolio_volatility:.2%}")
        report.append(f"  Concentration (HHI): {portfolio.concentration_hhi:.3f}")
        report.append("")

        # Greeks Exposure
        report.append("📈 GREEKS EXPOSURE:")
        report.append(f"  Total Delta: {portfolio.total_delta:,.0f}")
        report.append(f"  Total Gamma: {portfolio.total_gamma:,.0f}")
        report.append(f"  Total Theta: {portfolio.total_theta:,.0f}")
        report.append(f"  Total Vega: {portfolio.total_vega:,.0f}")
        report.append("")

        # Active Alerts
        alert_summary = summary['alert_summary']
        report.append("🚨 ALERT SUMMARY:")
        report.append(f"  Active Alerts: {alert_summary['total_active_alerts']}")
        report.append(f"  Critical: {alert_summary['critical_alerts']}")
        report.append(f"  High: {alert_summary['high_alerts']}")
        report.append("")

        # Top Risk Positions
        high_risk_positions = [p for p in portfolio.positions if p.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]]
        if high_risk_positions:
            report.append("⚠️ HIGH RISK POSITIONS:")
            for pos in high_risk_positions[:5]:  # Top 5
                report.append(f"  {pos.symbol} ({pos.strategy}): {pos.risk_level.value}")
                report.append(f"    Weight: {pos.portfolio_weight:.1%}, VaR: ${pos.var_1d:,.0f}")

        report.append("=" * 80)

        return "\n".join(report)


# Example usage and testing
if __name__ == "__main__":
    def test_unified_risk_manager():
        """Test the unified risk management system."""

        print("Testing Unified Risk Management System")
        print("=" * 60)

        # Create risk manager
        risk_manager = UnifiedRiskManager()

        # Add sample positions
        positions = [
            {
                'symbol': 'AAPL',
                'strategy': 'wheel_strategy',
                'quantity': 100,
                'market_price': 150.0,
                'position_value': 15000,
                'option_type': 'put',
                'strike': 145,
                'expiry': datetime.now() + timedelta(days=30),
                'implied_volatility': 0.25,
                'returns_history': np.random.normal(0.001, 0.02, 50).tolist()
            },
            {
                'symbol': 'SPY',
                'strategy': 'spx_credit_spreads',
                'quantity': -10,
                'market_price': 420.0,
                'position_value': 8000,
                'option_type': 'call',
                'strike': 425,
                'expiry': datetime.now() + timedelta(days=15),
                'implied_volatility': 0.18,
                'returns_history': np.random.normal(0.0005, 0.015, 50).tolist()
            },
            {
                'symbol': 'TSLA',
                'strategy': 'swing_trading',
                'quantity': 50,
                'market_price': 250.0,
                'position_value': 12500,
                'option_type': 'call',
                'strike': 260,
                'expiry': datetime.now() + timedelta(days=7),
                'implied_volatility': 0.35,
                'returns_history': np.random.normal(0.002, 0.03, 50).tolist()
            }
        ]

        print("1. Adding positions:")
        for pos in positions:
            risk_manager.add_position(pos['symbol'], pos['strategy'], pos)
            print(f"   Added {pos['symbol']} ({pos['strategy']}): ${pos['position_value']:,}")

        print("\n2. Calculating portfolio risk:")
        portfolio_risk = risk_manager.calculate_portfolio_risk()

        print(f"   Total Value: ${portfolio_risk.total_value:,.0f}")
        print(f"   Portfolio VaR: ${portfolio_risk.portfolio_var_1d:,.0f}")
        print(f"   Risk Level: {portfolio_risk.risk_level.value}")
        print(f"   Total Delta: {portfolio_risk.total_delta:,.0f}")
        print(f"   Concentration HHI: {portfolio_risk.concentration_hhi:.3f}")

        print("\n3. Checking risk limits:")
        alerts = risk_manager.check_risk_limits()

        if alerts:
            print(f"   Generated {len(alerts)} alerts:")
            for alert in alerts:
                print(f"   - {alert.alert_type}: {alert.severity.value}")
                print(f"     {alert.message}")
        else:
            print("   No risk limit violations")

        print("\n4. Risk summary:")
        summary = risk_manager.get_risk_summary()
        print(f"   Active alerts: {summary['alert_summary']['total_active_alerts']}")
        print(f"   High risk positions: {summary['position_summary']['high_risk_positions']}")
        print(f"   Strategies: {summary['position_summary']['strategies']}")

        print("\n5. Risk report:")
        report = risk_manager.generate_risk_report()
        print(report[:500] + "..." if len(report) > 500 else report)

        return risk_manager

    # Run test
    test_risk_manager = test_unified_risk_manager()