"""
Position Sizing Strategies for Prediction Market Arbitrage.

Synthesized from:
- RichardFeynmanEnthusiast: Kelly-inspired sqrt sizing
- dexorynLabs: Conservative percentage-based sizing
- antevorta: Proportional position scaling

Risk-aware position sizing is critical for survival.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from decimal import Decimal, ROUND_DOWN
from enum import Enum
from typing import Optional
import logging
import math

logger = logging.getLogger(__name__)


class SizingStrategy(Enum):
    """Available sizing strategies."""
    FIXED = "fixed"
    PERCENTAGE = "percentage"
    KELLY = "kelly"
    SQRT = "sqrt"
    PROPORTIONAL = "proportional"


@dataclass
class WalletBalance:
    """Wallet balance information."""
    platform: str
    available: Decimal
    reserved: Decimal = Decimal("0")

    @property
    def total(self) -> Decimal:
        return self.available + self.reserved

    @property
    def utilization(self) -> Decimal:
        if self.total == 0:
            return Decimal("0")
        return self.reserved / self.total


@dataclass
class SizingContext:
    """Context for position sizing decisions."""
    # Wallet balances
    polymarket_balance: WalletBalance
    kalshi_balance: WalletBalance

    # Opportunity details
    opportunity_size: int  # Max tradeable size from order books
    profit_margin: Decimal  # Expected profit per contract
    win_probability: Decimal = Decimal("1.0")  # For Kelly

    # Fees
    estimated_fees: Decimal = Decimal("0")

    @property
    def min_available(self) -> Decimal:
        """Minimum available balance across platforms."""
        return min(
            self.polymarket_balance.available,
            self.kalshi_balance.available
        )


@dataclass
class SizingResult:
    """Result of position sizing calculation."""
    size: int
    strategy_used: SizingStrategy
    capped_by: Optional[str] = None  # "wallet", "opportunity", "risk"
    details: str = ""


class PositionSizer(ABC):
    """Abstract base class for position sizing strategies."""

    @abstractmethod
    def calculate_size(self, context: SizingContext) -> SizingResult:
        """Calculate position size."""
        pass


class FixedSizer(PositionSizer):
    """
    Fixed position size.

    Simple but effective for learning/testing.
    """

    def __init__(self, fixed_size: int = 100):
        self._fixed_size = fixed_size

    def calculate_size(self, context: SizingContext) -> SizingResult:
        """Return fixed size, capped by available."""
        size = min(
            self._fixed_size,
            context.opportunity_size,
            int(context.min_available)
        )

        capped_by = None
        if size < self._fixed_size:
            if size == context.opportunity_size:
                capped_by = "opportunity"
            else:
                capped_by = "wallet"

        return SizingResult(
            size=size,
            strategy_used=SizingStrategy.FIXED,
            capped_by=capped_by,
            details=f"Fixed size {self._fixed_size}, actual {size}"
        )


class PercentageSizer(PositionSizer):
    """
    Percentage-based position sizing.

    From dexorynLabs: Conservative 15% of opportunity.
    From RichardFeynmanEnthusiast: 95% of min wallet.
    """

    def __init__(
        self,
        opportunity_pct: Decimal = Decimal("0.15"),
        wallet_pct: Decimal = Decimal("0.95"),
        min_size: int = 1,
        max_size: int = 10000,
    ):
        self._opportunity_pct = opportunity_pct
        self._wallet_pct = wallet_pct
        self._min_size = min_size
        self._max_size = max_size

    def calculate_size(self, context: SizingContext) -> SizingResult:
        """Calculate percentage-based size."""
        # 15% of opportunity
        opp_based = int(context.opportunity_size * self._opportunity_pct)

        # 95% of min wallet (leaves buffer)
        wallet_based = int(context.min_available * self._wallet_pct)

        # Take minimum
        size = min(opp_based, wallet_based, self._max_size)
        size = max(size, 0)  # Can't be negative

        capped_by = None
        if size == opp_based and opp_based < wallet_based:
            capped_by = "opportunity"
        elif size == wallet_based:
            capped_by = "wallet"
        elif size == self._max_size:
            capped_by = "risk"

        # Check minimum
        if size < self._min_size:
            return SizingResult(
                size=0,
                strategy_used=SizingStrategy.PERCENTAGE,
                capped_by="minimum",
                details=f"Size {size} below minimum {self._min_size}"
            )

        return SizingResult(
            size=size,
            strategy_used=SizingStrategy.PERCENTAGE,
            capped_by=capped_by,
            details=f"Opp: {opp_based}, Wallet: {wallet_based}, Final: {size}"
        )


class KellySizer(PositionSizer):
    """
    Kelly Criterion-inspired position sizing.

    Kelly formula: f* = (bp - q) / b
    Where:
        f* = fraction of capital to bet
        b = odds received (net profit / cost)
        p = probability of winning
        q = probability of losing (1 - p)

    For arbitrage with guaranteed profit, p = 1.0.
    """

    def __init__(
        self,
        fraction: Decimal = Decimal("0.5"),  # Half-Kelly for safety
        max_fraction: Decimal = Decimal("0.25"),  # Max 25% of capital
    ):
        self._fraction = fraction
        self._max_fraction = max_fraction

    def calculate_size(self, context: SizingContext) -> SizingResult:
        """Calculate Kelly-based size."""
        # For arbitrage: guaranteed profit, so simplified Kelly
        # Edge = profit_margin / cost_per_contract
        # Assume cost ~= 1.0 for prediction markets

        if context.profit_margin <= 0:
            return SizingResult(
                size=0,
                strategy_used=SizingStrategy.KELLY,
                capped_by="no_edge",
                details="No positive edge"
            )

        # Full Kelly = edge (for even-money bets with guaranteed win)
        full_kelly = float(context.profit_margin)

        # Apply Kelly fraction (half-Kelly common for safety)
        kelly_fraction = full_kelly * float(self._fraction)

        # Cap at maximum fraction
        kelly_fraction = min(kelly_fraction, float(self._max_fraction))

        # Calculate size from fraction of capital
        capital = float(context.min_available)
        kelly_size = int(capital * kelly_fraction)

        # Cap by opportunity
        size = min(kelly_size, context.opportunity_size)

        capped_by = None
        if size < kelly_size:
            capped_by = "opportunity"
        elif kelly_fraction == float(self._max_fraction):
            capped_by = "risk"

        return SizingResult(
            size=size,
            strategy_used=SizingStrategy.KELLY,
            capped_by=capped_by,
            details=f"Kelly fraction: {kelly_fraction:.4f}, Size: {size}"
        )


class SqrtSizer(PositionSizer):
    """
    Square root position sizing.

    From RichardFeynmanEnthusiast: Kelly-Criterion-inspired sqrt
    reduces position when opportunity size is large.

    Rationale: Large opportunities often have worse execution
    (slippage, partial fills), so scale down.
    """

    def __init__(
        self,
        multiplier: Decimal = Decimal("1.0"),
        wallet_buffer: Decimal = Decimal("0.95"),
    ):
        self._multiplier = multiplier
        self._wallet_buffer = wallet_buffer

    def calculate_size(self, context: SizingContext) -> SizingResult:
        """Calculate sqrt-based size."""
        if context.opportunity_size <= 0:
            return SizingResult(
                size=0,
                strategy_used=SizingStrategy.SQRT,
                capped_by="no_opportunity",
                details="No available opportunity size"
            )

        # Sqrt of opportunity size
        sqrt_size = int(math.sqrt(context.opportunity_size) * float(self._multiplier))

        # Cap by wallet (with buffer)
        wallet_cap = int(context.min_available * self._wallet_buffer)
        size = min(sqrt_size, wallet_cap, context.opportunity_size)

        capped_by = None
        if size == wallet_cap:
            capped_by = "wallet"
        elif size < sqrt_size:
            capped_by = "opportunity"

        return SizingResult(
            size=size,
            strategy_used=SizingStrategy.SQRT,
            capped_by=capped_by,
            details=f"Sqrt({context.opportunity_size}) = {sqrt_size}, Final: {size}"
        )


class ProportionalSizer(PositionSizer):
    """
    Proportional position sizing.

    From antevorta: Match position size relative to a target.
    Useful for copy trading or following specific traders.
    """

    def __init__(
        self,
        target_position: int = 0,
        target_capital: Decimal = Decimal("0"),
        max_ratio: Decimal = Decimal("0.01"),  # Max 1% of target
    ):
        self._target_position = target_position
        self._target_capital = target_capital
        self._max_ratio = max_ratio

    def calculate_size(self, context: SizingContext) -> SizingResult:
        """Calculate proportional size."""
        if self._target_capital <= 0:
            return SizingResult(
                size=0,
                strategy_used=SizingStrategy.PROPORTIONAL,
                capped_by="no_target",
                details="No target capital configured"
            )

        # Calculate ratio based on our capital vs target capital
        our_capital = context.min_available
        ratio = min(
            our_capital / self._target_capital,
            self._max_ratio
        )

        # Apply ratio to target position
        prop_size = int(self._target_position * float(ratio))

        # Cap by opportunity
        size = min(prop_size, context.opportunity_size)

        capped_by = None
        if ratio == float(self._max_ratio):
            capped_by = "risk"
        elif size < prop_size:
            capped_by = "opportunity"

        return SizingResult(
            size=size,
            strategy_used=SizingStrategy.PROPORTIONAL,
            capped_by=capped_by,
            details=f"Ratio: {ratio:.6f}, Prop size: {prop_size}, Final: {size}"
        )


class CompositePositionSizer(PositionSizer):
    """
    Combines multiple sizing strategies.

    Takes the minimum across all strategies for maximum safety.
    """

    def __init__(self, sizers: list[PositionSizer]):
        self._sizers = sizers

    def calculate_size(self, context: SizingContext) -> SizingResult:
        """Calculate minimum across all sizers."""
        if not self._sizers:
            return SizingResult(
                size=0,
                strategy_used=SizingStrategy.FIXED,
                capped_by="no_sizers",
                details="No sizers configured"
            )

        results = [sizer.calculate_size(context) for sizer in self._sizers]
        min_result = min(results, key=lambda r: r.size)

        return SizingResult(
            size=min_result.size,
            strategy_used=min_result.strategy_used,
            capped_by=min_result.capped_by,
            details=f"Composite min from {len(self._sizers)} sizers: {min_result.details}"
        )


def create_position_sizer(
    strategy: SizingStrategy,
    **kwargs
) -> PositionSizer:
    """Factory function to create position sizers."""
    sizers = {
        SizingStrategy.FIXED: FixedSizer,
        SizingStrategy.PERCENTAGE: PercentageSizer,
        SizingStrategy.KELLY: KellySizer,
        SizingStrategy.SQRT: SqrtSizer,
        SizingStrategy.PROPORTIONAL: ProportionalSizer,
    }

    sizer_class = sizers.get(strategy)
    if sizer_class is None:
        raise ValueError(f"Unknown sizing strategy: {strategy}")

    return sizer_class(**kwargs)


def create_conservative_sizer() -> CompositePositionSizer:
    """
    Create a conservative composite sizer.

    Combines multiple strategies for maximum safety.
    """
    return CompositePositionSizer([
        PercentageSizer(opportunity_pct=Decimal("0.10")),  # 10% of opportunity
        SqrtSizer(),  # Sqrt scaling
        FixedSizer(fixed_size=500),  # Hard cap at 500
    ])
