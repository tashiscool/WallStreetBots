"""
Tiered Fee Calculator for Prediction Markets.

Synthesized from:
- dexorynLabs: Tiered fee structure based on price
- RichardFeynmanEnthusiast: Kalshi fee model (7% rate, quadratic formula)
- CarlosIbCu: Maker vs taker differentiation

Accurate fee modeling is critical for profitable arbitrage.
"""

from dataclasses import dataclass
from decimal import Decimal, ROUND_CEILING, ROUND_HALF_UP
from enum import Enum
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class FeeType(Enum):
    """Type of fee being charged."""
    TAKER = "taker"
    MAKER = "maker"


class Platform(Enum):
    """Supported platforms."""
    POLYMARKET = "polymarket"
    KALSHI = "kalshi"


@dataclass
class FeeResult:
    """Result of fee calculation."""
    fee_amount: Decimal
    fee_rate: Decimal
    fee_type: FeeType
    platform: Platform
    details: str = ""


class FeeCalculator:
    """
    Universal fee calculator for prediction markets.

    Implements tiered fee structures from dexorynLabs:
    - Fees vary based on contract price
    - Maker fees typically lower than taker
    - Some platforms have quadratic fee formulas
    """

    # Polymarket tiered fee schedule (percentage, by price range)
    # From dexorynLabs: fees highest at 50 cents, lowest at extremes
    POLYMARKET_FEE_SCHEDULE: List[Tuple[Tuple[int, int], Decimal]] = [
        ((0, 5), Decimal("0.01")),      # 1% at extremes
        ((5, 10), Decimal("0.015")),
        ((10, 20), Decimal("0.02")),
        ((20, 30), Decimal("0.025")),
        ((30, 40), Decimal("0.03")),
        ((40, 50), Decimal("0.035")),   # 3.5% near 50 cents (max)
        ((50, 60), Decimal("0.035")),
        ((60, 70), Decimal("0.03")),
        ((70, 80), Decimal("0.025")),
        ((80, 90), Decimal("0.02")),
        ((90, 95), Decimal("0.015")),
        ((95, 100), Decimal("0.01")),   # 1% at extremes
    ]

    # Maker fee multiplier (makers pay 50% of taker fees on Polymarket)
    POLYMARKET_MAKER_MULTIPLIER = Decimal("0.5")

    # Kalshi fee rate (7% base rate, from RichardFeynmanEnthusiast)
    KALSHI_BASE_RATE = Decimal("0.07")

    def __init__(self):
        """Initialize fee calculator."""
        self._cache: Dict[str, FeeResult] = {}

    def calculate_polymarket_fee(
        self,
        price: Decimal,
        quantity: int,
        is_maker: bool = False,
    ) -> FeeResult:
        """
        Calculate Polymarket fee using tiered schedule.

        Args:
            price: Contract price (0-1 scale)
            quantity: Number of contracts
            is_maker: Whether this is a maker order

        Returns:
            FeeResult with calculated fee
        """
        # Convert to cents for schedule lookup
        price_cents = int(price * 100)

        # Find applicable fee rate
        fee_rate = Decimal("0.02")  # Default
        for (low, high), rate in self.POLYMARKET_FEE_SCHEDULE:
            if low <= price_cents < high:
                fee_rate = rate
                break

        # Apply maker discount
        if is_maker:
            fee_rate *= self.POLYMARKET_MAKER_MULTIPLIER

        # Calculate fee
        notional = price * quantity
        fee_amount = (notional * fee_rate).quantize(
            Decimal("0.01"),
            rounding=ROUND_CEILING
        )

        return FeeResult(
            fee_amount=fee_amount,
            fee_rate=fee_rate,
            fee_type=FeeType.MAKER if is_maker else FeeType.TAKER,
            platform=Platform.POLYMARKET,
            details=f"Price {price_cents}¢ -> {fee_rate*100:.1f}% rate"
        )

    def calculate_kalshi_fee(
        self,
        price: Decimal,
        quantity: int,
        rate: Optional[Decimal] = None,
    ) -> FeeResult:
        """
        Calculate Kalshi fee using quadratic formula.

        From RichardFeynmanEnthusiast:
        Fee = rate × quantity × price × (1 - price)

        This formula means:
        - Fees are highest at price = 0.50 (max uncertainty)
        - Fees approach zero at price = 0 or 1 (certainty)

        Args:
            price: Contract price (0-1 scale)
            quantity: Number of contracts
            rate: Fee rate (default 7%)

        Returns:
            FeeResult with calculated fee
        """
        if rate is None:
            rate = self.KALSHI_BASE_RATE

        # Validate price bounds
        if price <= Decimal("0") or price >= Decimal("1"):
            return FeeResult(
                fee_amount=Decimal("0"),
                fee_rate=rate,
                fee_type=FeeType.TAKER,
                platform=Platform.KALSHI,
                details="Price at boundary, zero fee"
            )

        # Quadratic formula: rate * quantity * price * (1 - price)
        raw_fee = rate * quantity * price * (Decimal("1") - price)

        # Round UP to nearest cent (conservative)
        fee_amount = raw_fee.quantize(
            Decimal("0.01"),
            rounding=ROUND_CEILING
        )

        return FeeResult(
            fee_amount=fee_amount,
            fee_rate=rate,
            fee_type=FeeType.TAKER,
            platform=Platform.KALSHI,
            details=f"Quadratic: {rate}*{quantity}*{price}*(1-{price})"
        )

    def calculate_arbitrage_fees(
        self,
        poly_yes_price: Decimal,
        poly_no_price: Decimal,
        kalshi_yes_price: Decimal,
        kalshi_no_price: Decimal,
        quantity: int,
        poly_is_maker: bool = False,
    ) -> Dict[str, FeeResult]:
        """
        Calculate all fees for an arbitrage trade.

        From dexorynLabs: Always calculate net profit after fees.

        Args:
            poly_yes_price: Polymarket YES price
            poly_no_price: Polymarket NO price
            kalshi_yes_price: Kalshi YES price
            kalshi_no_price: Kalshi NO price
            quantity: Number of contracts
            poly_is_maker: Whether Polymarket orders are maker

        Returns:
            Dict with fees for each leg
        """
        return {
            "poly_yes": self.calculate_polymarket_fee(
                poly_yes_price, quantity, poly_is_maker
            ),
            "poly_no": self.calculate_polymarket_fee(
                poly_no_price, quantity, poly_is_maker
            ),
            "kalshi_yes": self.calculate_kalshi_fee(
                kalshi_yes_price, quantity
            ),
            "kalshi_no": self.calculate_kalshi_fee(
                kalshi_no_price, quantity
            ),
        }

    def calculate_net_profit(
        self,
        gross_profit: Decimal,
        fees: Dict[str, FeeResult],
        gas_cost: Decimal = Decimal("0"),
    ) -> Decimal:
        """
        Calculate net profit after all fees.

        From dexorynLabs: Dynamic threshold based on actual fees.

        Args:
            gross_profit: Gross profit before fees
            fees: Dict of FeeResult from calculate_arbitrage_fees
            gas_cost: Blockchain gas costs (Polymarket only)

        Returns:
            Net profit after fees
        """
        total_fees = sum(f.fee_amount for f in fees.values())
        return gross_profit - total_fees - gas_cost

    def estimate_break_even_quantity(
        self,
        edge_per_contract: Decimal,
        poly_price: Decimal,
        kalshi_price: Decimal,
        fixed_costs: Decimal = Decimal("0"),
    ) -> int:
        """
        Estimate minimum quantity for profitable trade.

        Args:
            edge_per_contract: Gross edge per contract
            poly_price: Average Polymarket price
            kalshi_price: Average Kalshi price
            fixed_costs: Fixed costs (gas, etc.)

        Returns:
            Minimum quantity to break even
        """
        # Estimate per-contract fee
        poly_fee_rate = self._get_poly_rate(poly_price)
        kalshi_fee_per = self.KALSHI_BASE_RATE * kalshi_price * (1 - kalshi_price)

        total_fee_per = float(poly_fee_rate * poly_price + kalshi_fee_per)
        edge = float(edge_per_contract)

        if edge <= total_fee_per:
            return 0  # Never profitable

        # Break-even: edge * qty = fixed + fee_per * qty
        # qty = fixed / (edge - fee_per)
        if edge - total_fee_per <= 0:
            return 0

        break_even = float(fixed_costs) / (edge - total_fee_per)
        return int(break_even) + 1

    def _get_poly_rate(self, price: Decimal) -> Decimal:
        """Get Polymarket fee rate for price."""
        price_cents = int(price * 100)
        for (low, high), rate in self.POLYMARKET_FEE_SCHEDULE:
            if low <= price_cents < high:
                return rate
        return Decimal("0.02")


class GasCostEstimator:
    """
    Estimates blockchain gas costs for Polymarket.

    From RichardFeynmanEnthusiast: Include gas in profitability calcs.
    """

    # Average gas costs in USD (varies with network congestion)
    DEFAULT_GAS_PER_ORDER = Decimal("0.50")

    def __init__(
        self,
        gas_per_order: Decimal = None,
    ):
        self._gas_per_order = gas_per_order or self.DEFAULT_GAS_PER_ORDER

    def estimate_trade_gas(
        self,
        num_orders: int = 2,
    ) -> Decimal:
        """
        Estimate gas cost for trade.

        Args:
            num_orders: Number of orders (typically 2 for arb)

        Returns:
            Estimated gas cost in USD
        """
        return self._gas_per_order * num_orders

    def set_gas_price(self, gas_per_order: Decimal) -> None:
        """Update gas price estimate."""
        self._gas_per_order = gas_per_order


# Global fee calculator instance
_fee_calculator: Optional[FeeCalculator] = None


def get_fee_calculator() -> FeeCalculator:
    """Get or create the global fee calculator."""
    global _fee_calculator
    if _fee_calculator is None:
        _fee_calculator = FeeCalculator()
    return _fee_calculator
