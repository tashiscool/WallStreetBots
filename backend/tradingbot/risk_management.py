"""Position Sizing and Risk Management System
Implements sophisticated risk management to prevent existential bets while maintaining edge.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

import numpy as np


class RiskLevel(Enum):
    """Risk level classification."""

    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"
    EXISTENTIAL = "existential"  # To be avoided


class PositionStatus(Enum):
    """Position status for tracking."""

    OPEN = "open"
    CLOSED = "closed"
    STOPPED_OUT = "stopped_out"
    EXPIRED = "expired"


@dataclass
class RiskParameters:
    """Risk management configuration."""

    # Account - level risk limits
    max_single_position_risk: float = 0.15  # Never risk more than 15% on single trade
    recommended_position_risk: float = 0.10  # Recommended 10% per position
    max_total_risk: float = 0.30  # Max 30% of account at risk across all positions
    max_concentration_per_ticker: float = 0.20  # Max 20% in any single ticker

    # Kelly Criterion limits
    max_kelly_fraction: float = 0.50  # Never exceed 50% Kelly (typically use 25% Kelly)
    kelly_multiplier: float = 0.25  # Use quarter Kelly for safety

    # Position sizing tiers
    risk_tiers: dict[str, float] = field(
        default_factory=lambda: {
            "conservative": 0.05,  # 5% risk for uncertain setups
            "moderate": 0.10,  # 10% risk for solid setups
            "aggressive": 0.15,  # 15% risk for high - conviction setups
        }
    )

    # Stop loss and take profit levels
    max_loss_per_position: float = 0.50  # Stop at 50% loss
    profit_take_levels: list[float] = field(
        default_factory=lambda: [1.0, 2.0, 2.5]
    )  # 100%, 200%, 250%
    trailing_stop_trigger: float = 1.0  # Start trailing after 100% gain
    trailing_stop_distance: float = 0.25  # Trail 25% behind peak

    # Time-based risk controls
    max_position_hold_days: int = 45  # Force exit before expiry
    earnings_blackout_days: int = 7  # No new positions ±7 days from earnings

    # Correlation limits
    max_correlated_exposure: float = 0.25  # Max 25% in highly correlated positions


@dataclass
class Position:
    """Individual position tracking."""

    ticker: str
    position_type: str  # 'call', 'put', 'spread'
    entry_date: datetime
    expiry_date: datetime
    strike: float
    contracts: int
    entry_premium: float  # Per contract
    current_premium: float  # Current market value per contract
    total_cost: float  # Total premium paid
    current_value: float  # Current position value
    stop_loss_level: float  # Stop loss price per contract
    profit_targets: list[float]  # Profit target levels
    status: PositionStatus = PositionStatus.OPEN

    # Risk metrics
    initial_risk: float = 0.0  # Initial $ at risk
    current_risk: float = 0.0  # Current $ at risk
    unrealized_pnl: float = 0.0  # Current P & L
    max_profit: float = 0.0  # Peak profit achieved

    def __post_init__(self):
        self.total_cost = self.contracts * self.entry_premium
        self.current_value = self.contracts * self.current_premium
        self.initial_risk = self.total_cost
        self.current_risk = max(0, self.total_cost - self.current_value)
        self.unrealized_pnl = self.current_value - self.total_cost

        # Set default stop loss at 50% of premium
        if self.stop_loss_level == 0:
            self.stop_loss_level = self.entry_premium * 0.50

    @property
    def days_to_expiry(self) -> int:
        """Calculate days remaining to expiry."""
        return max(0, (self.expiry_date - datetime.now()).days)

    @property
    def unrealized_roi(self) -> float:
        """Calculate unrealized return on investment."""
        return (self.unrealized_pnl / self.total_cost) if self.total_cost > 0 else 0.0

    def update_current_premium(self, new_premium: float):
        """Update position with current market premium."""
        self.current_premium = new_premium
        self.current_value = self.contracts * new_premium
        self.unrealized_pnl = self.current_value - self.total_cost

        # Track peak profit for trailing stops
        self.max_profit = max(self.max_profit, self.unrealized_pnl)


@dataclass
class PortfolioRisk:
    """Portfolio - level risk metrics."""

    account_value: float
    total_cash: float
    total_positions_value: float
    total_risk_amount: float  # Total $ at risk across all positions
    unrealized_pnl: float

    # Risk percentages
    cash_utilization: float = 0.0  # % of account in positions
    risk_utilization: float = 0.0  # % of account at risk

    # Concentration metrics
    ticker_concentrations: dict[str, float] = field(default_factory=dict)
    sector_concentrations: dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        self.cash_utilization = (
            self.total_positions_value / self.account_value if self.account_value > 0 else 0
        )
        self.risk_utilization = (
            self.total_risk_amount / self.account_value if self.account_value > 0 else 0
        )


class KellyCalculator:
    """Kelly Criterion calculator for optimal position sizing."""

    @staticmethod
    def calculate_kelly_fraction(
        win_probability: float, avg_win_pct: float, avg_loss_pct: float
    ) -> float:
        """Calculate Kelly fraction for optimal position sizing.

        Args:
            win_probability: Probability of winning (0 to 1)
            avg_win_pct: Average winning percentage return
            avg_loss_pct: Average losing percentage (positive number)

        Returns:
            Optimal Kelly fraction (can be negative if negative edge)
        """
        if avg_loss_pct <= 0:
            return 0.0

        # Kelly formula: f*=(bp - q) / b
        # where b=avg_win_pct / avg_loss_pct, p=win_prob, q = loss_prob
        b = avg_win_pct / avg_loss_pct
        p = win_probability
        q = 1 - p

        kelly_fraction = (b * p - q) / b
        return max(0.0, kelly_fraction)  # Don't go negative

    @staticmethod
    def calculate_from_historical_trades(trades: list[dict]) -> tuple[float, dict]:
        """Calculate Kelly fraction from historical trade results.

        Args:
            trades: List of trade dictionaries with 'return_pct' key

        Returns:
            Tuple of (kelly_fraction, statistics)
        """
        if not trades:
            return 0.0, {}

        returns = [trade["return_pct"] for trade in trades]
        wins = [r for r in returns if r > 0]
        losses = [-r for r in returns if r < 0]  # Make positive

        if not wins or not losses:
            return 0.0, {"error": "Need both wins and losses"}

        win_prob = len(wins) / len(returns)
        avg_win = sum(wins) / len(wins)
        avg_loss = sum(losses) / len(losses)

        kelly = KellyCalculator.calculate_kelly_fraction(win_prob, avg_win, avg_loss)

        stats = {
            "win_probability": win_prob,
            "avg_win_pct": avg_win,
            "avg_loss_pct": avg_loss,
            "total_trades": len(returns),
            "kelly_fraction": kelly,
        }

        return kelly, stats


class PositionSizer:
    """Advanced position sizing with multiple risk models."""

    def __init__(self, risk_params: RiskParameters = None):
        self.risk_params = risk_params or RiskParameters()
        self.kelly_calc = KellyCalculator()

    def calculate_position_size(
        self,
        account_value: float,
        setup_confidence: float,  # 0 to 1
        premium_per_contract: float,
        expected_win_rate: float = 0.60,  # Default from successful track record
        expected_avg_win: float = 1.50,  # Average 150% gain on winners
        expected_avg_loss: float = 0.45,  # Average 45% loss on losers (stop loss)
        risk_tier: str = "moderate",
    ) -> dict:
        """Calculate optimal position size using multiple methods.

        Returns:
            Dictionary with position sizing recommendations
        """
        if account_value <= 0 or premium_per_contract <= 0:
            raise ValueError("Account value and premium must be positive")

        results = {}

        # 1. Fixed fractional sizing based on risk tier
        max_risk_amount = account_value * self.risk_params.risk_tiers.get(risk_tier, 0.10)
        fixed_fractional_contracts = int(max_risk_amount / premium_per_contract)

        # 2. Kelly Criterion sizing
        kelly_fraction = self.kelly_calc.calculate_kelly_fraction(
            expected_win_rate, expected_avg_win, expected_avg_loss
        )

        # Apply Kelly multiplier for safety (typically 0.25x Kelly)
        safe_kelly_fraction = kelly_fraction * self.risk_params.kelly_multiplier
        kelly_risk_amount = account_value * safe_kelly_fraction
        kelly_contracts = int(kelly_risk_amount / premium_per_contract)

        # 3. Confidence-adjusted sizing
        confidence_adjusted_risk = max_risk_amount * setup_confidence
        confidence_contracts = int(confidence_adjusted_risk / premium_per_contract)

        # 4. Volatility - adjusted sizing (for high IV environments)
        # Reduce size when IV is high to account for potential IV crush
        base_iv = 0.25  # Baseline 25% IV
        current_iv = 0.30  # This would come from market data
        iv_adjustment = base_iv / current_iv if current_iv > 0 else 1.0
        iv_adjusted_contracts = int(fixed_fractional_contracts * iv_adjustment)

        # Take the minimum of all methods for safety
        recommended_contracts = min(
            fixed_fractional_contracts, kelly_contracts, confidence_contracts, iv_adjusted_contracts
        )

        # Ensure we don't exceed absolute limits
        max_absolute_risk = account_value * self.risk_params.max_single_position_risk
        absolute_max_contracts = int(max_absolute_risk / premium_per_contract)
        recommended_contracts = min(recommended_contracts, absolute_max_contracts)

        # Calculate final metrics
        final_cost = recommended_contracts * premium_per_contract
        final_risk_pct = (final_cost / account_value) * 100

        results = {
            "recommended_contracts": max(0, recommended_contracts),
            "total_cost": final_cost,
            "risk_amount": final_cost,
            "risk_percentage": final_risk_pct,
            "risk_tier": risk_tier,
            # Individual method results
            "fixed_fractional_contracts": fixed_fractional_contracts,
            "kelly_contracts": kelly_contracts,
            "confidence_contracts": confidence_contracts,
            "iv_adjusted_contracts": iv_adjusted_contracts,
            # Risk metrics
            "kelly_fraction": kelly_fraction,
            "safe_kelly_fraction": safe_kelly_fraction,
            "setup_confidence": setup_confidence,
            "expected_metrics": {
                "win_rate": expected_win_rate,
                "avg_win": expected_avg_win,
                "avg_loss": expected_avg_loss,
            },
        }

        return results


class RiskManager:
    """Comprehensive risk management system."""

    def __init__(self, risk_params: RiskParameters = None):
        self.risk_params = risk_params or RiskParameters()
        self.position_sizer = PositionSizer(risk_params)
        self.positions: list[Position] = []

    def add_position(self, position: Position) -> bool:
        """Add position after risk checks."""
        # Check if position passes risk limits
        if self._validate_new_position(position):
            self.positions.append(position)
            return True
        return False

    def _validate_new_position(self, position: Position) -> bool:
        """Validate new position against risk limits."""
        current_portfolio = self.calculate_portfolio_risk(position.total_cost)

        # Check individual position size limit
        account_value = current_portfolio.account_value
        position_risk_pct = position.total_cost / account_value

        if position_risk_pct > self.risk_params.max_single_position_risk:
            return False

        # Check total portfolio risk limit
        if current_portfolio.risk_utilization > self.risk_params.max_total_risk:
            return False

        # Check ticker concentration
        ticker_exposure = self._calculate_ticker_exposure(position.ticker)
        return not ticker_exposure > self.risk_params.max_concentration_per_ticker

    def calculate_portfolio_risk(self, additional_risk: float = 0) -> PortfolioRisk:
        """Calculate current portfolio risk metrics."""
        if not self.positions:
            # Assume some baseline account value for empty portfolio
            account_value = 500000.0  # This should come from account data
            return PortfolioRisk(
                account_value=account_value,
                total_cash=account_value,
                total_positions_value=additional_risk,
                total_risk_amount=additional_risk,
                unrealized_pnl=0.0,
            )

        total_positions_value = sum(
            pos.current_value for pos in self.positions if pos.status == PositionStatus.OPEN
        )
        total_risk_amount = sum(
            pos.current_risk for pos in self.positions if pos.status == PositionStatus.OPEN
        )
        unrealized_pnl = sum(
            pos.unrealized_pnl for pos in self.positions if pos.status == PositionStatus.OPEN
        )

        # Calculate account value (this should come from broker API)
        account_value = 500000.0  # Placeholder
        total_cash = account_value - total_positions_value

        # Calculate concentrations
        ticker_concentrations = self._calculate_concentrations()

        return PortfolioRisk(
            account_value=account_value,
            total_cash=total_cash,
            total_positions_value=total_positions_value + additional_risk,
            total_risk_amount=total_risk_amount + additional_risk,
            unrealized_pnl=unrealized_pnl,
            ticker_concentrations=ticker_concentrations,
        )

    def _calculate_ticker_exposure(self, ticker: str) -> float:
        """Calculate current exposure to a specific ticker."""
        ticker_value = sum(
            pos.current_value
            for pos in self.positions
            if pos.ticker == ticker and pos.status == PositionStatus.OPEN
        )
        portfolio_risk = self.calculate_portfolio_risk()
        return (
            ticker_value / portfolio_risk.account_value if portfolio_risk.account_value > 0 else 0
        )

    def _calculate_concentrations(self) -> dict[str, float]:
        """Calculate ticker concentration percentages."""
        concentrations = {}

        for position in self.positions:
            if position.status == PositionStatus.OPEN:
                ticker = position.ticker
                if ticker not in concentrations:
                    concentrations[ticker] = 0.0
                concentrations[ticker] += position.current_value

        # Convert to percentages
        total_value = sum(concentrations.values())
        if total_value > 0:
            concentrations = {
                ticker: value / total_value for ticker, value in concentrations.items()
            }

        return concentrations

    def check_stop_losses(self) -> list[Position]:
        """Check which positions should be stopped out."""
        positions_to_stop = []

        for position in self.positions:
            if position.status != PositionStatus.OPEN:
                continue

            # Price-based stop loss
            if position.current_premium <= position.stop_loss_level:
                positions_to_stop.append(position)
                continue

            # Time-based stop loss
            if position.days_to_expiry <= 7:  # Force exit 1 week before expiry
                positions_to_stop.append(position)
                continue

            # Maximum hold period
            hold_days = (datetime.now() - position.entry_date).days
            if hold_days >= self.risk_params.max_position_hold_days:
                positions_to_stop.append(position)
                continue

        return positions_to_stop

    def check_profit_targets(self) -> list[tuple[Position, float]]:
        """Check which positions hit profit targets."""
        profit_exits = []

        for position in self.positions:
            if position.status != PositionStatus.OPEN:
                continue

            current_roi = position.unrealized_roi

            # Check each profit target level
            for _i, target_roi in enumerate(self.risk_params.profit_take_levels):
                if current_roi >= target_roi:
                    # Determine what fraction to close (1 / 3 each level)
                    fraction_to_close = 1.0 / len(self.risk_params.profit_take_levels)
                    profit_exits.append((position, fraction_to_close))
                    break

        return profit_exits

    def generate_risk_report(self) -> dict:
        """Generate comprehensive risk report."""
        portfolio_risk = self.calculate_portfolio_risk()

        open_positions = [pos for pos in self.positions if pos.status == PositionStatus.OPEN]

        report = {
            "portfolio_metrics": {
                "account_value": portfolio_risk.account_value,
                "total_positions_value": portfolio_risk.total_positions_value,
                "cash_available": portfolio_risk.total_cash,
                "total_risk_amount": portfolio_risk.total_risk_amount,
                "risk_utilization_pct": portfolio_risk.risk_utilization * 100,
                "unrealized_pnl": portfolio_risk.unrealized_pnl,
            },
            "position_summary": {
                "total_positions": len(open_positions),
                "avg_days_to_expiry": np.mean([pos.days_to_expiry for pos in open_positions])
                if open_positions
                else 0,
                "positions_at_profit": len(
                    [pos for pos in open_positions if pos.unrealized_pnl > 0]
                ),
                "positions_at_loss": len([pos for pos in open_positions if pos.unrealized_pnl < 0]),
            },
            "risk_alerts": [],
            "recommendations": [],
        }

        # Generate alerts
        if portfolio_risk.risk_utilization > self.risk_params.max_total_risk:
            report["risk_alerts"].append(
                f"Portfolio risk utilization ({portfolio_risk.risk_utilization: .1%}) exceeds limit ({self.risk_params.max_total_risk: .1%})"
            )

        # Check for over - concentration
        for ticker, concentration in portfolio_risk.ticker_concentrations.items():
            if concentration > self.risk_params.max_concentration_per_ticker:
                report["risk_alerts"].append(
                    f"Over - concentrated in {ticker}: {concentration:.1%}"
                )

        # Check positions needing action
        stop_positions = self.check_stop_losses()
        profit_positions = self.check_profit_targets()

        if stop_positions:
            report["recommendations"].append(
                f"Consider stopping out {len(stop_positions)} positions"
            )

        if profit_positions:
            report["recommendations"].append(
                f"Consider taking profits on {len(profit_positions)} positions"
            )

        return report


if __name__ == "__main__":  # Test the risk management system
    print("=== RISK MANAGEMENT SYSTEM TEST ===")

    # Test position sizing
    sizer = PositionSizer()
    account_value = 500000
    premium = 4.70

    sizing = sizer.calculate_position_size(
        account_value=account_value,
        setup_confidence=0.8,  # High confidence setup
        premium_per_contract=premium,
        risk_tier="moderate",
    )

    print("Position Sizing Results: ")
    print(f"Recommended contracts: {sizing['recommended_contracts']:,}")
    print(f"Total cost: ${sizing['total_cost']:,.0f}")
    print(f"Risk percentage: {sizing['risk_percentage']:.1f}%")
    print(f"Kelly fraction: {sizing['kelly_fraction']:.3f}")

    # Test Kelly calculator with sample trades
    kelly_calc = KellyCalculator()
    sample_trades = [
        {"return_pct": 2.40},  # The 240% winner
        {"return_pct": -0.45},  # 45% loss
        {"return_pct": 0.50},  # 50% win
        {"return_pct": -0.30},  # 30% loss
        {"return_pct": 1.20},  # 120% win
    ]

    kelly_fraction, stats = kelly_calc.calculate_from_historical_trades(sample_trades)
    print("\nKelly Analysis from Historical Trades: ")
    print(f"Win rate: {stats['win_probability']:.1%}")
    print(f"Average win: {stats['avg_win_pct']:.1%}")
    print(f"Average loss: {stats['avg_loss_pct']:.1%}")
    print(f"Kelly fraction: {stats['kelly_fraction']:.3f}")

    # Test risk manager
    risk_manager = RiskManager()

    # Create sample position
    sample_position = Position(
        ticker="GOOGL",
        position_type="call",
        entry_date=datetime.now(),
        expiry_date=datetime.now() + timedelta(days=30),
        strike=220.0,
        contracts=950,
        entry_premium=4.70,
        current_premium=16.00,
        total_cost=446500,
        current_value=1520000,
        stop_loss_level=2.35,
        profit_targets=[1.0, 2.0, 2.5],
    )

    risk_report = risk_manager.generate_risk_report()
    print("\nRisk Report Summary: ")
    print(f"Account Value: ${risk_report['portfolio_metrics']['account_value']:,.0f}")
    print(f"Risk Utilization: {risk_report['portfolio_metrics']['risk_utilization_pct']:.1f}%")

    if risk_report["risk_alerts"]:
        print("Risk Alerts: ")
        for alert in risk_report["risk_alerts"]:
            print(f"  ⚠️  {alert}")

    if risk_report["recommendations"]:
        print("Recommendations: ")
        for rec in risk_report["recommendations"]:
            print(f"  💡 {rec}")
