"""
Portfolio Greeks Aggregation.

Ported from QuantConnect/LEAN's option portfolio Greeks.
Aggregates Greeks across all positions for risk management.
"""

from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Callable
import logging
import math

logger = logging.getLogger(__name__)


class RiskMetric(Enum):
    """Types of risk metrics."""
    DELTA_DOLLARS = "delta_dollars"
    GAMMA_DOLLARS = "gamma_dollars"
    THETA_DOLLARS = "theta_dollars"
    VEGA_DOLLARS = "vega_dollars"
    VAR_95 = "var_95"
    VAR_99 = "var_99"
    EXPECTED_SHORTFALL = "expected_shortfall"


@dataclass
class OptionPosition:
    """Option position for Greeks calculation."""
    symbol: str
    underlying: str
    quantity: int
    option_type: str  # "call" or "put"
    strike: Decimal
    expiration: date
    current_price: Decimal
    underlying_price: Decimal

    # Greeks per contract
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0
    rho: float = 0.0
    vanna: float = 0.0  # d(delta)/d(vol)
    charm: float = 0.0  # d(delta)/d(time)
    vomma: float = 0.0  # d(vega)/d(vol)

    # Implied volatility
    implied_volatility: float = 0.0

    @property
    def contract_multiplier(self) -> int:
        """Standard contract multiplier."""
        return 100

    @property
    def notional_value(self) -> Decimal:
        """Notional value of position."""
        return self.strike * self.contract_multiplier * abs(self.quantity)

    @property
    def market_value(self) -> Decimal:
        """Current market value."""
        return self.current_price * self.contract_multiplier * self.quantity


@dataclass
class StockPosition:
    """Stock position for delta calculation."""
    symbol: str
    quantity: int
    current_price: Decimal

    @property
    def delta(self) -> float:
        """Stock delta is always 1.0 per share."""
        return float(self.quantity)

    @property
    def market_value(self) -> Decimal:
        """Current market value."""
        return self.current_price * abs(self.quantity)


@dataclass
class UnderlyingGreeks:
    """Aggregated Greeks for a single underlying."""
    symbol: str
    underlying_price: Decimal

    # Position totals
    total_option_positions: int = 0
    total_stock_shares: int = 0

    # Greeks (raw units)
    net_delta: float = 0.0  # Shares equivalent
    net_gamma: float = 0.0  # Delta change per $1 move
    net_theta: float = 0.0  # Daily decay in dollars
    net_vega: float = 0.0  # P&L change per 1% IV change
    net_rho: float = 0.0  # P&L change per 1% rate change

    # Second-order Greeks
    net_vanna: float = 0.0
    net_charm: float = 0.0
    net_vomma: float = 0.0

    # Dollar-weighted Greeks
    delta_dollars: Decimal = Decimal("0")  # Delta * underlying price
    gamma_dollars: Decimal = Decimal("0")  # Gamma * underlying^2 * 0.01
    theta_dollars: Decimal = Decimal("0")  # Theta in dollars
    vega_dollars: Decimal = Decimal("0")  # Vega * 0.01 (per 1% move)

    # Risk metrics
    beta_weighted_delta: float = 0.0  # Normalized to SPY
    portfolio_weight: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "underlying_price": float(self.underlying_price),
            "total_option_positions": self.total_option_positions,
            "total_stock_shares": self.total_stock_shares,
            "net_delta": self.net_delta,
            "net_gamma": self.net_gamma,
            "net_theta": self.net_theta,
            "net_vega": self.net_vega,
            "delta_dollars": float(self.delta_dollars),
            "gamma_dollars": float(self.gamma_dollars),
            "theta_dollars": float(self.theta_dollars),
            "vega_dollars": float(self.vega_dollars),
            "beta_weighted_delta": self.beta_weighted_delta,
        }


@dataclass
class PortfolioGreeks:
    """Aggregated Greeks for entire portfolio."""
    # Timestamp
    as_of: datetime = field(default_factory=datetime.now)

    # Total portfolio metrics
    total_delta: float = 0.0  # Shares equivalent
    total_gamma: float = 0.0
    total_theta: float = 0.0  # Daily decay
    total_vega: float = 0.0

    # Dollar-weighted totals
    total_delta_dollars: Decimal = Decimal("0")
    total_gamma_dollars: Decimal = Decimal("0")
    total_theta_dollars: Decimal = Decimal("0")
    total_vega_dollars: Decimal = Decimal("0")

    # Beta-weighted (normalized to SPY)
    beta_weighted_delta: float = 0.0

    # By underlying
    by_underlying: Dict[str, UnderlyingGreeks] = field(default_factory=dict)

    # Portfolio value
    total_market_value: Decimal = Decimal("0")
    total_notional: Decimal = Decimal("0")

    # Risk estimates
    estimated_daily_pnl_range: Tuple[Decimal, Decimal] = (Decimal("0"), Decimal("0"))
    var_95: Decimal = Decimal("0")
    var_99: Decimal = Decimal("0")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "as_of": self.as_of.isoformat(),
            "total_delta": self.total_delta,
            "total_gamma": self.total_gamma,
            "total_theta": self.total_theta,
            "total_vega": self.total_vega,
            "total_delta_dollars": float(self.total_delta_dollars),
            "total_gamma_dollars": float(self.total_gamma_dollars),
            "total_theta_dollars": float(self.total_theta_dollars),
            "total_vega_dollars": float(self.total_vega_dollars),
            "beta_weighted_delta": self.beta_weighted_delta,
            "total_market_value": float(self.total_market_value),
            "estimated_daily_pnl_range": (
                float(self.estimated_daily_pnl_range[0]),
                float(self.estimated_daily_pnl_range[1]),
            ),
            "var_95": float(self.var_95),
            "var_99": float(self.var_99),
            "by_underlying": {
                k: v.to_dict() for k, v in self.by_underlying.items()
            },
        }


class PortfolioGreeksCalculator:
    """
    Calculates and aggregates Greeks across portfolio.

    Features:
    - Per-underlying aggregation
    - Beta-weighted deltas (normalized to SPY)
    - Dollar-weighted Greeks
    - VaR estimation
    - Scenario analysis
    """

    # Default beta values for common underlyings
    DEFAULT_BETAS = {
        "SPY": 1.0,
        "QQQ": 1.1,
        "IWM": 1.2,
        "DIA": 0.9,
        "AAPL": 1.2,
        "MSFT": 1.1,
        "GOOGL": 1.1,
        "AMZN": 1.3,
        "TSLA": 1.8,
        "NVDA": 1.5,
        "META": 1.2,
    }

    # Standard deviations for VaR (normal distribution)
    VAR_95_MULT = 1.645
    VAR_99_MULT = 2.326

    def __init__(
        self,
        spy_price: Optional[Decimal] = None,
        beta_lookup: Optional[Callable[[str], float]] = None,
    ):
        """
        Initialize calculator.

        Args:
            spy_price: Current SPY price for beta weighting
            beta_lookup: Function to look up beta for a symbol
        """
        self.spy_price = spy_price or Decimal("450")  # Default
        self.beta_lookup = beta_lookup or self._default_beta_lookup

    def _default_beta_lookup(self, symbol: str) -> float:
        """Default beta lookup using hardcoded values."""
        return self.DEFAULT_BETAS.get(symbol, 1.0)

    def calculate_position_greeks(
        self,
        position: OptionPosition,
    ) -> Dict[str, float]:
        """
        Calculate dollar-weighted Greeks for a single position.

        Args:
            position: Option position

        Returns:
            Dictionary of calculated metrics
        """
        multiplier = position.contract_multiplier
        qty = position.quantity
        price = float(position.underlying_price)

        # Raw Greeks * quantity * multiplier
        delta = position.delta * qty * multiplier
        gamma = position.gamma * qty * multiplier
        theta = position.theta * qty * multiplier
        vega = position.vega * qty * multiplier

        # Dollar-weighted
        delta_dollars = delta * price
        gamma_dollars = gamma * price * price * 0.01  # Per 1% move
        theta_dollars = theta  # Already in dollars/day
        vega_dollars = vega * 0.01  # Per 1% IV change

        return {
            "delta": delta,
            "gamma": gamma,
            "theta": theta,
            "vega": vega,
            "delta_dollars": delta_dollars,
            "gamma_dollars": gamma_dollars,
            "theta_dollars": theta_dollars,
            "vega_dollars": vega_dollars,
        }

    def aggregate_greeks(
        self,
        option_positions: List[OptionPosition],
        stock_positions: Optional[List[StockPosition]] = None,
    ) -> PortfolioGreeks:
        """
        Aggregate Greeks across all positions.

        Args:
            option_positions: List of option positions
            stock_positions: Optional list of stock positions

        Returns:
            PortfolioGreeks with aggregated metrics
        """
        stock_positions = stock_positions or []

        result = PortfolioGreeks()
        by_underlying: Dict[str, UnderlyingGreeks] = {}

        # Process option positions
        for pos in option_positions:
            underlying = pos.underlying

            # Initialize underlying if needed
            if underlying not in by_underlying:
                by_underlying[underlying] = UnderlyingGreeks(
                    symbol=underlying,
                    underlying_price=pos.underlying_price,
                )

            ug = by_underlying[underlying]
            ug.total_option_positions += abs(pos.quantity)

            # Calculate position Greeks
            pos_greeks = self.calculate_position_greeks(pos)

            # Aggregate raw Greeks
            ug.net_delta += pos_greeks["delta"]
            ug.net_gamma += pos_greeks["gamma"]
            ug.net_theta += pos_greeks["theta"]
            ug.net_vega += pos_greeks["vega"]

            # Aggregate dollar Greeks
            ug.delta_dollars += Decimal(str(pos_greeks["delta_dollars"]))
            ug.gamma_dollars += Decimal(str(pos_greeks["gamma_dollars"]))
            ug.theta_dollars += Decimal(str(pos_greeks["theta_dollars"]))
            ug.vega_dollars += Decimal(str(pos_greeks["vega_dollars"]))

            # Second-order Greeks
            ug.net_vanna += pos.vanna * pos.quantity * pos.contract_multiplier
            ug.net_charm += pos.charm * pos.quantity * pos.contract_multiplier
            ug.net_vomma += pos.vomma * pos.quantity * pos.contract_multiplier

            # Market value
            result.total_market_value += pos.market_value
            result.total_notional += pos.notional_value

        # Process stock positions
        for pos in stock_positions:
            symbol = pos.symbol

            if symbol not in by_underlying:
                by_underlying[symbol] = UnderlyingGreeks(
                    symbol=symbol,
                    underlying_price=pos.current_price,
                )

            ug = by_underlying[symbol]
            ug.total_stock_shares += pos.quantity

            # Stock has delta = quantity (shares)
            ug.net_delta += pos.delta
            delta_dollars = pos.delta * float(pos.current_price)
            ug.delta_dollars += Decimal(str(delta_dollars))

            # Market value
            result.total_market_value += pos.market_value

        # Calculate beta-weighted deltas
        for symbol, ug in by_underlying.items():
            beta = self.beta_lookup(symbol)
            spy_equivalent = ug.net_delta * beta * float(ug.underlying_price) / float(self.spy_price)
            ug.beta_weighted_delta = spy_equivalent

        # Aggregate totals
        for ug in by_underlying.values():
            result.total_delta += ug.net_delta
            result.total_gamma += ug.net_gamma
            result.total_theta += ug.net_theta
            result.total_vega += ug.net_vega

            result.total_delta_dollars += ug.delta_dollars
            result.total_gamma_dollars += ug.gamma_dollars
            result.total_theta_dollars += ug.theta_dollars
            result.total_vega_dollars += ug.vega_dollars

            result.beta_weighted_delta += ug.beta_weighted_delta

        # Calculate portfolio weights
        if result.total_market_value > 0:
            for symbol, ug in by_underlying.items():
                underlying_value = sum(
                    pos.market_value for pos in option_positions
                    if pos.underlying == symbol
                ) + sum(
                    pos.market_value for pos in stock_positions
                    if pos.symbol == symbol
                )
                ug.portfolio_weight = float(underlying_value / result.total_market_value)

        # Estimate VaR
        result.var_95, result.var_99 = self._estimate_var(result)

        # Estimate daily P&L range
        result.estimated_daily_pnl_range = self._estimate_daily_pnl(result)

        result.by_underlying = by_underlying
        result.as_of = datetime.now()

        return result

    def _estimate_var(
        self,
        greeks: PortfolioGreeks,
        daily_vol: float = 0.01,  # 1% daily volatility
    ) -> Tuple[Decimal, Decimal]:
        """
        Estimate Value at Risk using delta-gamma approximation.

        Args:
            greeks: Portfolio Greeks
            daily_vol: Assumed daily volatility

        Returns:
            Tuple of (VaR 95%, VaR 99%)
        """
        # Delta-normal VaR
        delta_var = float(greeks.total_delta_dollars) * daily_vol

        # Add gamma adjustment
        gamma_adj = 0.5 * float(greeks.total_gamma_dollars) * (daily_vol ** 2)

        # Total portfolio VaR
        portfolio_std = abs(delta_var) + abs(gamma_adj)

        var_95 = Decimal(str(portfolio_std * self.VAR_95_MULT))
        var_99 = Decimal(str(portfolio_std * self.VAR_99_MULT))

        return var_95, var_99

    def _estimate_daily_pnl(
        self,
        greeks: PortfolioGreeks,
    ) -> Tuple[Decimal, Decimal]:
        """
        Estimate daily P&L range including theta.

        Returns (worst_case, best_case) estimates.
        """
        # Theta is guaranteed daily loss/gain
        theta = greeks.total_theta_dollars

        # Add 1 standard deviation move
        delta_impact = abs(greeks.total_delta_dollars) * Decimal("0.01")

        # Worst case: theta + adverse delta move
        worst = theta - delta_impact
        # Best case: theta + favorable delta move
        best = theta + delta_impact

        return (worst, best)

    def scenario_analysis(
        self,
        greeks: PortfolioGreeks,
        price_change_pct: float,
        iv_change_pct: float = 0.0,
        days_forward: int = 0,
    ) -> Dict[str, Any]:
        """
        Run scenario analysis on portfolio.

        Args:
            greeks: Portfolio Greeks
            price_change_pct: Percentage price change (e.g., 0.05 for 5%)
            iv_change_pct: Percentage IV change (e.g., 0.10 for 10%)
            days_forward: Days to project forward

        Returns:
            Scenario analysis results
        """
        # Delta P&L
        delta_pnl = float(greeks.total_delta_dollars) * price_change_pct

        # Gamma P&L (second order)
        gamma_pnl = 0.5 * float(greeks.total_gamma_dollars) * (price_change_pct ** 2)

        # Vega P&L
        vega_pnl = float(greeks.total_vega_dollars) * iv_change_pct

        # Theta P&L
        theta_pnl = float(greeks.total_theta_dollars) * days_forward

        total_pnl = delta_pnl + gamma_pnl + vega_pnl + theta_pnl

        return {
            "scenario": {
                "price_change_pct": price_change_pct,
                "iv_change_pct": iv_change_pct,
                "days_forward": days_forward,
            },
            "pnl_breakdown": {
                "delta_pnl": delta_pnl,
                "gamma_pnl": gamma_pnl,
                "vega_pnl": vega_pnl,
                "theta_pnl": theta_pnl,
            },
            "total_pnl": total_pnl,
            "pnl_pct": total_pnl / float(greeks.total_market_value) if greeks.total_market_value > 0 else 0,
        }

    def stress_test(
        self,
        greeks: PortfolioGreeks,
        scenarios: Optional[List[Dict[str, float]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Run multiple stress test scenarios.

        Args:
            greeks: Portfolio Greeks
            scenarios: List of scenario dictionaries

        Returns:
            List of scenario results
        """
        if scenarios is None:
            scenarios = [
                {"name": "Market crash -10%", "price_change_pct": -0.10, "iv_change_pct": 0.50},
                {"name": "Market crash -20%", "price_change_pct": -0.20, "iv_change_pct": 0.80},
                {"name": "Market rally +10%", "price_change_pct": 0.10, "iv_change_pct": -0.20},
                {"name": "IV spike +50%", "price_change_pct": 0.0, "iv_change_pct": 0.50},
                {"name": "IV crush -30%", "price_change_pct": 0.0, "iv_change_pct": -0.30},
                {"name": "Flat 30 days", "price_change_pct": 0.0, "iv_change_pct": 0.0, "days_forward": 30},
            ]

        results = []
        for scenario in scenarios:
            name = scenario.pop("name", "Unnamed")
            result = self.scenario_analysis(greeks, **scenario)
            result["name"] = name
            results.append(result)
            scenario["name"] = name  # Restore

        return results


class GreeksMonitor:
    """
    Real-time Greeks monitoring with alerts.

    Monitors portfolio Greeks and triggers alerts when
    thresholds are breached.
    """

    def __init__(
        self,
        max_delta: float = 500.0,  # Max delta exposure
        max_gamma: float = 100.0,  # Max gamma
        min_theta: float = -100.0,  # Min theta (daily)
        max_vega: float = 200.0,  # Max vega
    ):
        """
        Initialize monitor with thresholds.

        Args:
            max_delta: Maximum absolute delta
            max_gamma: Maximum absolute gamma
            min_theta: Minimum theta (daily decay)
            max_vega: Maximum absolute vega
        """
        self.max_delta = max_delta
        self.max_gamma = max_gamma
        self.min_theta = min_theta
        self.max_vega = max_vega

        self._alerts: List[Dict[str, Any]] = []
        self._callbacks: List[Callable[[Dict[str, Any]], None]] = []

    def on_alert(
        self,
        callback: Callable[[Dict[str, Any]], None],
    ) -> None:
        """Register callback for alerts."""
        self._callbacks.append(callback)

    def _emit_alert(self, alert: Dict[str, Any]) -> None:
        """Emit alert to callbacks."""
        self._alerts.append(alert)
        for callback in self._callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")

    def check_thresholds(
        self,
        greeks: PortfolioGreeks,
    ) -> List[Dict[str, Any]]:
        """
        Check Greeks against thresholds.

        Args:
            greeks: Portfolio Greeks

        Returns:
            List of threshold breaches
        """
        breaches = []

        # Delta check
        if abs(greeks.total_delta) > self.max_delta:
            breach = {
                "type": "delta",
                "value": greeks.total_delta,
                "threshold": self.max_delta,
                "severity": "high" if abs(greeks.total_delta) > self.max_delta * 1.5 else "medium",
                "message": f"Delta ({greeks.total_delta:.0f}) exceeds threshold ({self.max_delta})",
                "timestamp": datetime.now(),
            }
            breaches.append(breach)
            self._emit_alert(breach)

        # Gamma check
        if abs(greeks.total_gamma) > self.max_gamma:
            breach = {
                "type": "gamma",
                "value": greeks.total_gamma,
                "threshold": self.max_gamma,
                "severity": "medium",
                "message": f"Gamma ({greeks.total_gamma:.0f}) exceeds threshold ({self.max_gamma})",
                "timestamp": datetime.now(),
            }
            breaches.append(breach)
            self._emit_alert(breach)

        # Theta check
        if greeks.total_theta < self.min_theta:
            breach = {
                "type": "theta",
                "value": greeks.total_theta,
                "threshold": self.min_theta,
                "severity": "medium",
                "message": f"Theta ({greeks.total_theta:.0f}) below threshold ({self.min_theta})",
                "timestamp": datetime.now(),
            }
            breaches.append(breach)
            self._emit_alert(breach)

        # Vega check
        if abs(greeks.total_vega) > self.max_vega:
            breach = {
                "type": "vega",
                "value": greeks.total_vega,
                "threshold": self.max_vega,
                "severity": "medium",
                "message": f"Vega ({greeks.total_vega:.0f}) exceeds threshold ({self.max_vega})",
                "timestamp": datetime.now(),
            }
            breaches.append(breach)
            self._emit_alert(breach)

        return breaches

    def get_hedging_recommendations(
        self,
        greeks: PortfolioGreeks,
    ) -> List[Dict[str, Any]]:
        """
        Generate hedging recommendations based on Greeks.

        Args:
            greeks: Portfolio Greeks

        Returns:
            List of hedging recommendations
        """
        recommendations = []

        # Delta hedging
        if abs(greeks.total_delta) > self.max_delta * 0.8:
            shares_to_hedge = -int(greeks.total_delta)
            recommendations.append({
                "type": "delta_hedge",
                "action": "buy" if shares_to_hedge > 0 else "sell",
                "instrument": "SPY",
                "quantity": abs(shares_to_hedge),
                "reason": f"Reduce delta exposure from {greeks.total_delta:.0f} to near zero",
                "urgency": "high" if abs(greeks.total_delta) > self.max_delta else "medium",
            })

        # Gamma hedging (complex - suggest closing positions)
        if abs(greeks.total_gamma) > self.max_gamma:
            # Find underlying with highest gamma contribution
            max_gamma_underlying = max(
                greeks.by_underlying.values(),
                key=lambda x: abs(x.net_gamma),
                default=None
            )
            if max_gamma_underlying:
                recommendations.append({
                    "type": "gamma_reduction",
                    "action": "reduce_position",
                    "instrument": max_gamma_underlying.symbol,
                    "reason": f"Reduce gamma exposure in {max_gamma_underlying.symbol}",
                    "urgency": "medium",
                })

        # Vega hedging
        if abs(greeks.total_vega) > self.max_vega:
            recommendations.append({
                "type": "vega_hedge",
                "action": "sell_straddle" if greeks.total_vega > 0 else "buy_straddle",
                "instrument": "VIX_options",
                "reason": f"Reduce vega exposure from {greeks.total_vega:.0f}",
                "urgency": "medium",
            })

        return recommendations

    def get_alerts(
        self,
        since: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """Get historical alerts."""
        if since:
            return [a for a in self._alerts if a["timestamp"] >= since]
        return self._alerts.copy()
