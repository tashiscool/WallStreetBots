"""
Portfolio Allocator - Manages target allocations and capital distribution.

Provides utilities for calculating position sizes, managing allocation
targets, and determining how to distribute capital across assets.
"""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class AllocationMethod(Enum):
    """Method for calculating allocations."""
    EQUAL_WEIGHT = "equal_weight"
    MARKET_CAP_WEIGHT = "market_cap_weight"
    RISK_PARITY = "risk_parity"
    INVERSE_VOLATILITY = "inverse_volatility"
    CUSTOM = "custom"


@dataclass
class AllocationTarget:
    """Target allocation for a single asset."""
    symbol: str
    target_weight: float  # 0-1
    min_weight: float = 0.0
    max_weight: float = 1.0
    is_core: bool = True  # Core positions vs tactical
    sector: Optional[str] = None
    notes: Optional[str] = None

    def __post_init__(self):
        """Validate allocation."""
        if not 0 <= self.target_weight <= 1:
            raise ValueError(f"Target weight must be 0-1, got {self.target_weight}")
        if not self.min_weight <= self.target_weight <= self.max_weight:
            raise ValueError("Target weight must be between min and max")


@dataclass
class AllocationResult:
    """Result of allocation calculation."""
    allocations: Dict[str, float]
    total_value: Decimal
    cash_allocation: float
    timestamp: datetime = field(default_factory=datetime.now)
    method: AllocationMethod = AllocationMethod.CUSTOM
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_allocation(self) -> float:
        """Sum of all allocations including cash."""
        return sum(self.allocations.values()) + self.cash_allocation

    def get_position_size(self, symbol: str) -> Decimal:
        """Get dollar size for a position."""
        weight = self.allocations.get(symbol, 0)
        return self.total_value * Decimal(str(weight))


class PortfolioAllocator:
    """
    Manages portfolio allocation targets and calculations.

    Usage:
        allocator = PortfolioAllocator()

        # Add targets
        allocator.add_target("SPY", 0.60, sector="equity")
        allocator.add_target("BND", 0.30, sector="bonds")
        allocator.add_target("GLD", 0.10, sector="commodities")

        # Calculate allocations
        result = allocator.calculate_allocations(portfolio_value=100000)

        # Or use equal weight
        result = allocator.calculate_equal_weight(
            symbols=["AAPL", "MSFT", "GOOGL", "AMZN"],
            portfolio_value=100000
        )
    """

    def __init__(
        self,
        targets: Optional[List[AllocationTarget]] = None,
        cash_buffer: float = 0.02,  # Keep 2% cash
        sector_limits: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize allocator.

        Args:
            targets: List of allocation targets
            cash_buffer: Percentage to keep in cash
            sector_limits: Maximum weight per sector
        """
        self.targets: Dict[str, AllocationTarget] = {}
        self.cash_buffer = cash_buffer
        self.sector_limits = sector_limits or {}

        if targets:
            for target in targets:
                self.targets[target.symbol] = target

    def add_target(
        self,
        symbol: str,
        weight: float,
        min_weight: float = 0.0,
        max_weight: float = 1.0,
        is_core: bool = True,
        sector: Optional[str] = None,
    ) -> "PortfolioAllocator":
        """Add or update an allocation target."""
        self.targets[symbol] = AllocationTarget(
            symbol=symbol,
            target_weight=weight,
            min_weight=min_weight,
            max_weight=max_weight,
            is_core=is_core,
            sector=sector,
        )
        return self

    def remove_target(self, symbol: str) -> "PortfolioAllocator":
        """Remove an allocation target."""
        self.targets.pop(symbol, None)
        return self

    def clear_targets(self) -> "PortfolioAllocator":
        """Clear all targets."""
        self.targets.clear()
        return self

    def get_target_allocations(self) -> Dict[str, float]:
        """Get dict of symbol -> target weight."""
        return {t.symbol: t.target_weight for t in self.targets.values()}

    def validate_allocations(self) -> List[str]:
        """
        Validate that allocations are valid.

        Returns:
            List of warning/error messages
        """
        warnings = []

        total = sum(t.target_weight for t in self.targets.values())
        total_with_cash = total + self.cash_buffer

        if total_with_cash > 1.0:
            warnings.append(
                f"Total allocation ({total_with_cash:.1%}) exceeds 100%"
            )

        if total_with_cash < 0.95:
            warnings.append(
                f"Total allocation ({total_with_cash:.1%}) is below 95%"
            )

        # Check sector limits
        sector_totals: Dict[str, float] = {}
        for target in self.targets.values():
            if target.sector:
                sector_totals[target.sector] = (
                    sector_totals.get(target.sector, 0) + target.target_weight
                )

        for sector, total in sector_totals.items():
            limit = self.sector_limits.get(sector, 1.0)
            if total > limit:
                warnings.append(
                    f"Sector '{sector}' allocation ({total:.1%}) exceeds limit ({limit:.1%})"
                )

        return warnings

    def calculate_allocations(
        self,
        portfolio_value: Decimal,
        available_cash: Optional[Decimal] = None,
    ) -> AllocationResult:
        """
        Calculate allocations based on targets.

        Args:
            portfolio_value: Total portfolio value
            available_cash: Available cash (uses cash_buffer if not provided)

        Returns:
            AllocationResult with calculated allocations
        """
        allocations = {}

        # Calculate actual weights (accounting for cash buffer)
        investable_pct = 1.0 - self.cash_buffer
        for target in self.targets.values():
            allocations[target.symbol] = target.target_weight * investable_pct

        return AllocationResult(
            allocations=allocations,
            total_value=portfolio_value,
            cash_allocation=self.cash_buffer,
            method=AllocationMethod.CUSTOM,
        )

    def calculate_equal_weight(
        self,
        symbols: List[str],
        portfolio_value: Decimal,
    ) -> AllocationResult:
        """
        Calculate equal-weight allocations.

        Args:
            symbols: List of symbols to allocate to
            portfolio_value: Total portfolio value

        Returns:
            AllocationResult with equal weights
        """
        if not symbols:
            return AllocationResult(
                allocations={},
                total_value=portfolio_value,
                cash_allocation=1.0,
                method=AllocationMethod.EQUAL_WEIGHT,
            )

        investable_pct = 1.0 - self.cash_buffer
        weight_per_symbol = investable_pct / len(symbols)

        allocations = dict.fromkeys(symbols, weight_per_symbol)

        return AllocationResult(
            allocations=allocations,
            total_value=portfolio_value,
            cash_allocation=self.cash_buffer,
            method=AllocationMethod.EQUAL_WEIGHT,
        )

    def calculate_inverse_volatility(
        self,
        symbols: List[str],
        volatilities: Dict[str, float],
        portfolio_value: Decimal,
    ) -> AllocationResult:
        """
        Calculate inverse-volatility weighted allocations.

        Lower volatility assets get higher weights.

        Args:
            symbols: Symbols to allocate to
            volatilities: Dict of symbol -> annualized volatility
            portfolio_value: Total portfolio value

        Returns:
            AllocationResult with inverse-vol weights
        """
        if not symbols:
            return AllocationResult(
                allocations={},
                total_value=portfolio_value,
                cash_allocation=1.0,
                method=AllocationMethod.INVERSE_VOLATILITY,
            )

        # Calculate inverse volatilities
        inv_vols = {}
        for symbol in symbols:
            vol = volatilities.get(symbol, 0.20)  # Default 20% vol
            if vol > 0:
                inv_vols[symbol] = 1.0 / vol
            else:
                inv_vols[symbol] = 0

        total_inv_vol = sum(inv_vols.values())
        if total_inv_vol == 0:
            # Fall back to equal weight
            return self.calculate_equal_weight(symbols, portfolio_value)

        # Calculate weights
        investable_pct = 1.0 - self.cash_buffer
        allocations = {}
        for symbol in symbols:
            weight = (inv_vols[symbol] / total_inv_vol) * investable_pct
            allocations[symbol] = weight

        return AllocationResult(
            allocations=allocations,
            total_value=portfolio_value,
            cash_allocation=self.cash_buffer,
            method=AllocationMethod.INVERSE_VOLATILITY,
            metadata={"volatilities": volatilities},
        )

    def calculate_risk_parity(
        self,
        symbols: List[str],
        volatilities: Dict[str, float],
        correlations: Optional[Dict[str, Dict[str, float]]] = None,
        portfolio_value: Decimal = Decimal("100000"),
        target_volatility: float = 0.10,
    ) -> AllocationResult:
        """
        Calculate risk parity allocations.

        Each asset contributes equally to portfolio risk.
        Simplified version without correlation matrix.

        Args:
            symbols: Symbols to allocate to
            volatilities: Dict of symbol -> annualized volatility
            correlations: Optional correlation matrix
            portfolio_value: Total portfolio value
            target_volatility: Target portfolio volatility

        Returns:
            AllocationResult with risk parity weights
        """
        # Simplified risk parity (inverse vol squared)
        if not symbols:
            return AllocationResult(
                allocations={},
                total_value=portfolio_value,
                cash_allocation=1.0,
                method=AllocationMethod.RISK_PARITY,
            )

        inv_var = {}
        for symbol in symbols:
            vol = volatilities.get(symbol, 0.20)
            if vol > 0:
                inv_var[symbol] = 1.0 / (vol ** 2)
            else:
                inv_var[symbol] = 0

        total_inv_var = sum(inv_var.values())
        if total_inv_var == 0:
            return self.calculate_equal_weight(symbols, portfolio_value)

        # Calculate raw weights
        investable_pct = 1.0 - self.cash_buffer
        allocations = {}
        for symbol in symbols:
            weight = (inv_var[symbol] / total_inv_var) * investable_pct
            allocations[symbol] = weight

        return AllocationResult(
            allocations=allocations,
            total_value=portfolio_value,
            cash_allocation=self.cash_buffer,
            method=AllocationMethod.RISK_PARITY,
            metadata={
                "volatilities": volatilities,
                "target_volatility": target_volatility,
            },
        )

    def calculate_position_sizes(
        self,
        portfolio_value: Decimal,
        prices: Dict[str, Decimal],
        round_to_shares: bool = True,
    ) -> Dict[str, int]:
        """
        Calculate number of shares for each position.

        Args:
            portfolio_value: Total portfolio value
            prices: Current prices
            round_to_shares: Round to whole shares

        Returns:
            Dict of symbol -> number of shares
        """
        allocations = self.calculate_allocations(portfolio_value)
        shares = {}

        for symbol, weight in allocations.allocations.items():
            dollar_amount = portfolio_value * Decimal(str(weight))
            price = prices.get(symbol, Decimal("0"))

            if price > 0:
                if round_to_shares:
                    shares[symbol] = int(dollar_amount / price)
                else:
                    shares[symbol] = float(dollar_amount / price)
            else:
                shares[symbol] = 0

        return shares

    def adjust_for_constraints(
        self,
        allocations: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Adjust allocations to respect min/max constraints.

        Args:
            allocations: Raw allocations

        Returns:
            Adjusted allocations
        """
        adjusted = {}
        excess = 0.0

        # First pass: apply min/max constraints
        for symbol, weight in allocations.items():
            target = self.targets.get(symbol)
            if target:
                if weight < target.min_weight:
                    adjusted[symbol] = target.min_weight
                    excess -= (target.min_weight - weight)
                elif weight > target.max_weight:
                    adjusted[symbol] = target.max_weight
                    excess += (weight - target.max_weight)
                else:
                    adjusted[symbol] = weight
            else:
                adjusted[symbol] = weight

        # Second pass: distribute excess to unconstrained positions
        if abs(excess) > 0.001:
            unconstrained = [
                s for s, w in adjusted.items()
                if self.targets.get(s) and
                self.targets[s].min_weight < w < self.targets[s].max_weight
            ]

            if unconstrained:
                adjustment = excess / len(unconstrained)
                for symbol in unconstrained:
                    adjusted[symbol] += adjustment

        return adjusted

    def to_dict(self) -> Dict[str, Any]:
        """Export allocator configuration."""
        return {
            "targets": [
                {
                    "symbol": t.symbol,
                    "target_weight": t.target_weight,
                    "min_weight": t.min_weight,
                    "max_weight": t.max_weight,
                    "is_core": t.is_core,
                    "sector": t.sector,
                }
                for t in self.targets.values()
            ],
            "cash_buffer": self.cash_buffer,
            "sector_limits": self.sector_limits,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PortfolioAllocator":
        """Create allocator from configuration dict."""
        targets = [
            AllocationTarget(**t)
            for t in data.get("targets", [])
        ]
        return cls(
            targets=targets,
            cash_buffer=data.get("cash_buffer", 0.02),
            sector_limits=data.get("sector_limits", {}),
        )
