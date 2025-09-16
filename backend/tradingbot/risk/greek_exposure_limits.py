"""Greek and beta exposure limits for portfolio risk management.

This module enforces exposure limits on portfolio Greeks (Delta, Gamma, Theta, Vega)
and beta-adjusted exposure at both portfolio and per-name levels.
"""
from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional
from decimal import Decimal

log = logging.getLogger("wsb.greek_limits")


@dataclass
class GreekLimits:
    """Configuration for Greek exposure limits."""
    max_portfolio_delta: float = 1000000.0  # $1M delta exposure
    max_portfolio_gamma: float = 10000.0   # $10K gamma exposure
    max_portfolio_theta: float = -1000.0   # -$1K theta exposure (negative is decay)
    max_portfolio_vega: float = 50000.0    # $50K vega exposure
    
    max_per_name_delta: float = 100000.0  # $100K per name delta
    max_per_name_gamma: float = 1000.0    # $1K per name gamma
    max_per_name_theta: float = -100.0    # -$100 per name theta
    max_per_name_vega: float = 5000.0     # $5K per name vega
    
    max_beta_adjusted_exposure: float = 2000000.0  # $2M beta-adjusted exposure
    max_per_name_beta_exposure: float = 200000.0  # $200K per name beta exposure


@dataclass
class PositionGreeks:
    """Greeks for a single position."""
    symbol: str
    delta: float
    gamma: float
    theta: float
    vega: float
    beta: float
    notional: float


class GreekExposureLimiter:
    """Enforces Greek and beta exposure limits.
    
    Provides pre-trade validation and real-time monitoring
    of portfolio Greek exposures.
    """

    def __init__(self, limits: GreekLimits = GreekLimits()):
        """Initialize Greek exposure limiter.
        
        Args:
            limits: GreekLimits configuration
        """
        self.limits = limits
        self.positions: Dict[str, PositionGreeks] = {}

    def add_position(self, position: PositionGreeks) -> None:
        """Add or update position Greeks.
        
        Args:
            position: Position Greeks to add/update
        """
        self.positions[position.symbol] = position
        log.debug(f"Added position Greeks for {position.symbol}")

    def remove_position(self, symbol: str) -> None:
        """Remove position.
        
        Args:
            symbol: Symbol to remove
        """
        if symbol in self.positions:
            del self.positions[symbol]
            log.debug(f"Removed position Greeks for {symbol}")

    def get_portfolio_greeks(self) -> Dict[str, float]:
        """Calculate portfolio-level Greeks.
        
        Returns:
            Dictionary with portfolio Greek exposures
        """
        portfolio_delta = sum(pos.delta for pos in self.positions.values())
        portfolio_gamma = sum(pos.gamma for pos in self.positions.values())
        portfolio_theta = sum(pos.theta for pos in self.positions.values())
        portfolio_vega = sum(pos.vega for pos in self.positions.values())
        
        # Beta-adjusted exposure
        beta_adjusted_exposure = sum(
            pos.notional * pos.beta for pos in self.positions.values()
        )
        
        return {
            "delta": portfolio_delta,
            "gamma": portfolio_gamma,
            "theta": portfolio_theta,
            "vega": portfolio_vega,
            "beta_adjusted_exposure": beta_adjusted_exposure,
        }

    def check_portfolio_limits(self) -> List[str]:
        """Check portfolio-level limits.
        
        Returns:
            List of limit violations
        """
        violations = []
        greeks = self.get_portfolio_greeks()
        
        if greeks["delta"] > self.limits.max_portfolio_delta:
            violations.append(
                f"Portfolio delta {greeks['delta']:.0f} exceeds limit {self.limits.max_portfolio_delta:.0f}"
            )
        
        if greeks["gamma"] > self.limits.max_portfolio_gamma:
            violations.append(
                f"Portfolio gamma {greeks['gamma']:.0f} exceeds limit {self.limits.max_portfolio_gamma:.0f}"
            )
        
        if greeks["theta"] < self.limits.max_portfolio_theta:
            violations.append(
                f"Portfolio theta {greeks['theta']:.0f} below limit {self.limits.max_portfolio_theta:.0f}"
            )
        
        if greeks["vega"] > self.limits.max_portfolio_vega:
            violations.append(
                f"Portfolio vega {greeks['vega']:.0f} exceeds limit {self.limits.max_portfolio_vega:.0f}"
            )
        
        if greeks["beta_adjusted_exposure"] > self.limits.max_beta_adjusted_exposure:
            violations.append(
                f"Beta-adjusted exposure {greeks['beta_adjusted_exposure']:.0f} exceeds limit {self.limits.max_beta_adjusted_exposure:.0f}"
            )
        
        return violations

    def check_per_name_limits(self, symbol: str = None) -> List[str]:
        """Check per-name limits for a symbol or all positions.
        
        Args:
            symbol: Symbol to check, or None to check all positions
            
        Returns:
            List of limit violations
        """
        violations = []
        
        if symbol is not None:
            # Check specific symbol
            if symbol not in self.positions:
                return violations
            
            pos = self.positions[symbol]
            
            if abs(pos.delta) > self.limits.max_per_name_delta:
                violations.append(
                    f"{symbol} delta {pos.delta:.0f} exceeds limit {self.limits.max_per_name_delta:.0f}"
                )
            
            if abs(pos.gamma) > self.limits.max_per_name_gamma:
                violations.append(
                    f"{symbol} gamma {pos.gamma:.0f} exceeds limit {self.limits.max_per_name_gamma:.0f}"
                )
            
            if pos.theta < self.limits.max_per_name_theta:
                violations.append(
                    f"{symbol} theta {pos.theta:.0f} below limit {self.limits.max_per_name_theta:.0f}"
                )
            
            if abs(pos.vega) > self.limits.max_per_name_vega:
                violations.append(
                    f"{symbol} vega {pos.vega:.0f} exceeds limit {self.limits.max_per_name_vega:.0f}"
                )
            
            beta_exposure = pos.notional * pos.beta
            if abs(beta_exposure) > self.limits.max_per_name_beta_exposure:
                violations.append(
                    f"{symbol} beta exposure {beta_exposure:.0f} exceeds limit {self.limits.max_per_name_beta_exposure:.0f}"
                )
        else:
            # Check all positions
            for symbol in self.positions:
                violations.extend(self.check_per_name_limits(symbol))
        
        return violations

    def validate_new_position(
        self, 
        symbol: str, 
        new_delta: float,
        new_gamma: float,
        new_theta: float,
        new_vega: float,
        new_beta: float,
        new_notional: float
    ) -> List[str]:
        """Validate new position against limits.
        
        Args:
            symbol: Symbol for new position
            new_delta: New position delta
            new_gamma: New position gamma
            new_theta: New position theta
            new_vega: New position vega
            new_beta: New position beta
            new_notional: New position notional
            
        Returns:
            List of validation errors
        """
        violations = []
        
        # Check per-name limits for new position
        temp_pos = PositionGreeks(
            symbol=symbol,
            delta=new_delta,
            gamma=new_gamma,
            theta=new_theta,
            vega=new_vega,
            beta=new_beta,
            notional=new_notional
        )
        
        violations.extend(self.check_per_name_limits(symbol))
        
        # Check portfolio limits with new position
        current_greeks = self.get_portfolio_greeks()
        new_portfolio_delta = current_greeks["delta"] + new_delta
        new_portfolio_gamma = current_greeks["gamma"] + new_gamma
        new_portfolio_theta = current_greeks["theta"] + new_theta
        new_portfolio_vega = current_greeks["vega"] + new_vega
        new_beta_exposure = current_greeks["beta_adjusted_exposure"] + (new_notional * new_beta)
        
        if new_portfolio_delta > self.limits.max_portfolio_delta:
            violations.append(
                f"New position would exceed portfolio delta limit: {new_portfolio_delta:.0f} > {self.limits.max_portfolio_delta:.0f}"
            )
        
        if new_portfolio_gamma > self.limits.max_portfolio_gamma:
            violations.append(
                f"New position would exceed portfolio gamma limit: {new_portfolio_gamma:.0f} > {self.limits.max_portfolio_gamma:.0f}"
            )
        
        if new_portfolio_theta < self.limits.max_portfolio_theta:
            violations.append(
                f"New position would exceed portfolio theta limit: {new_portfolio_theta:.0f} < {self.limits.max_portfolio_theta:.0f}"
            )
        
        if new_portfolio_vega > self.limits.max_portfolio_vega:
            violations.append(
                f"New position would exceed portfolio vega limit: {new_portfolio_vega:.0f} > {self.limits.max_portfolio_vega:.0f}"
            )
        
        if new_beta_exposure > self.limits.max_beta_adjusted_exposure:
            violations.append(
                f"New position would exceed beta-adjusted exposure limit: {new_beta_exposure:.0f} > {self.limits.max_beta_adjusted_exposure:.0f}"
            )
        
        return violations

    def require_ok(self) -> None:
        """Require all limits to be OK, raise if not.
        
        Raises:
            RuntimeError: If any limits are violated
        """
        violations = self.check_portfolio_limits()
        
        # Check per-name limits for all positions
        for symbol in self.positions:
            violations.extend(self.check_per_name_limits(symbol))
        
        if violations:
            error_msg = "Greek exposure limits violated:\n" + "\n".join(violations)
            raise RuntimeError(error_msg)

    def status(self) -> Dict:
        """Get current status.
        
        Returns:
            Dictionary with current exposures and limits
        """
        greeks = self.get_portfolio_greeks()
        
        return {
            "portfolio_greeks": greeks,
            "limits": {
                "max_portfolio_delta": self.limits.max_portfolio_delta,
                "max_portfolio_gamma": self.limits.max_portfolio_gamma,
                "max_portfolio_theta": self.limits.max_portfolio_theta,
                "max_portfolio_vega": self.limits.max_portfolio_vega,
                "max_beta_adjusted_exposure": self.limits.max_beta_adjusted_exposure,
            },
            "positions_count": len(self.positions),
            "violations": self.check_portfolio_limits(),
        }

    def get_position_summary(self) -> Dict:
        """Get summary of all positions.
        
        Returns:
            Dictionary with position summaries
        """
        positions_summary = {}
        for symbol, position in self.positions.items():
            positions_summary[symbol] = {
                "delta": position.delta,
                "gamma": position.gamma,
                "theta": position.theta,
                "vega": position.vega,
                "beta": position.beta,
                "notional": position.notional,
                "beta_exposure": position.notional * position.beta,
            }
        
        # Get portfolio Greeks
        portfolio_greeks = self.get_portfolio_greeks()
        
        # Find largest positions by absolute delta
        largest_positions = sorted(
            positions_summary.items(),
            key=lambda x: abs(x[1]["delta"]),
            reverse=True
        )[:5]  # Top 5 largest positions
        
        return {
            "total_positions": len(self.positions),
            "portfolio_greeks": portfolio_greeks,
            "largest_positions": dict(largest_positions),
            "positions": positions_summary,
        }

    def validate_position(self, position: PositionGreeks) -> tuple[bool, List[str]]:
        """Validate a position against limits.
        
        Args:
            position: Position to validate
            
        Returns:
            Tuple of (is_valid, violations)
        """
        violations = self.validate_new_position(
            position.symbol,
            position.delta,
            position.gamma,
            position.theta,
            position.vega,
            position.beta,
            position.notional
        )
        return len(violations) == 0, violations

    def get_greek_utilization(self) -> Dict[str, float]:
        """Get Greek utilization as decimal of limits.
        
        Returns:
            Dictionary with utilization as decimals (0.5 = 50%)
        """
        greeks = self.get_portfolio_greeks()
        
        return {
            "delta_utilization": greeks["delta"] / self.limits.max_portfolio_delta,
            "gamma_utilization": greeks["gamma"] / self.limits.max_portfolio_gamma,
            "theta_utilization": abs(greeks["theta"] / self.limits.max_portfolio_theta),
            "vega_utilization": greeks["vega"] / self.limits.max_portfolio_vega,
            "beta_utilization": greeks["beta_adjusted_exposure"] / self.limits.max_beta_adjusted_exposure,
        }

    def stress_test_greeks(self, stress_scenarios: Dict[str, Dict]) -> Dict[str, Dict]:
        """Run stress tests on portfolio Greeks.
        
        Args:
            stress_scenarios: Dictionary of scenario names to stress parameters
            
        Returns:
            Dictionary of stress test results
        """
        results = {}
        current_greeks = self.get_portfolio_greeks()
        
        for scenario_name, scenario_params in stress_scenarios.items():
            # Apply stress scenario - always include all Greeks
            stressed_greeks = {}
            
            # Initialize all Greeks with current values
            for greek_name in ["delta", "gamma", "theta", "vega"]:
                stressed_greeks[f"stressed_{greek_name}"] = current_greeks[greek_name]
            
            # Apply stress factors
            for greek, stress_factor in scenario_params.items():
                # Map scenario parameter names to Greek names
                greek_mapping = {
                    "delta_shock": "delta",
                    "gamma_shock": "gamma", 
                    "theta_shock": "theta",
                    "vega_shock": "vega"
                }
                
                if greek in greek_mapping:
                    actual_greek = greek_mapping[greek]
                    if actual_greek in current_greeks:
                        stressed_greeks[f"stressed_{actual_greek}"] = current_greeks[actual_greek] * (1 + stress_factor)
            
            # Check if stressed scenario violates limits
            violations = []
            if stressed_greeks.get("stressed_delta", 0) > self.limits.max_portfolio_delta:
                violations.append("Delta limit exceeded")
            if stressed_greeks.get("stressed_gamma", 0) > self.limits.max_portfolio_gamma:
                violations.append("Gamma limit exceeded")
            if stressed_greeks.get("stressed_theta", 0) < self.limits.max_portfolio_theta:
                violations.append("Theta limit exceeded")
            if stressed_greeks.get("stressed_vega", 0) > self.limits.max_portfolio_vega:
                violations.append("Vega limit exceeded")
            
            # Merge stressed Greeks directly into result
            result = {
                "scenario": scenario_params,
                "violations": violations,
                "passed": len(violations) == 0
            }
            result.update(stressed_greeks)
            results[scenario_name] = result
        
        return results

