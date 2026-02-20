"""
Position Group Margin Model.

Ported from QuantConnect/LEAN's option position group margin.
Calculates risk-based margin for option spreads instead of per-leg margin.
"""

from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class OptionType(Enum):
    """Option type."""
    CALL = "call"
    PUT = "put"


class PositionType(Enum):
    """Position type for margin calculation."""
    LONG_STOCK = "long_stock"
    SHORT_STOCK = "short_stock"
    LONG_CALL = "long_call"
    SHORT_CALL = "short_call"
    LONG_PUT = "long_put"
    SHORT_PUT = "short_put"


class SpreadType(Enum):
    """Type of option spread."""
    NAKED_CALL = "naked_call"
    NAKED_PUT = "naked_put"
    COVERED_CALL = "covered_call"
    CASH_SECURED_PUT = "cash_secured_put"
    VERTICAL_CALL = "vertical_call"
    VERTICAL_PUT = "vertical_put"
    IRON_CONDOR = "iron_condor"
    IRON_BUTTERFLY = "iron_butterfly"
    STRADDLE = "straddle"
    STRANGLE = "strangle"
    CALENDAR = "calendar"
    DIAGONAL = "diagonal"
    RATIO = "ratio"
    UNKNOWN = "unknown"


@dataclass
class OptionLeg:
    """Single option leg in a position group."""
    symbol: str
    underlying: str
    option_type: OptionType
    strike: Decimal
    expiration: date
    quantity: int
    premium: Decimal = Decimal("0")
    current_price: Decimal = Decimal("0")
    delta: Optional[float] = None
    gamma: Optional[float] = None

    @property
    def is_long(self) -> bool:
        """True if long position."""
        return self.quantity > 0

    @property
    def is_short(self) -> bool:
        """True if short position."""
        return self.quantity < 0

    @property
    def is_call(self) -> bool:
        """True if call option."""
        return self.option_type == OptionType.CALL

    @property
    def is_put(self) -> bool:
        """True if put option."""
        return self.option_type == OptionType.PUT

    @property
    def notional_value(self) -> Decimal:
        """Notional value (strike * 100 * quantity)."""
        return self.strike * Decimal("100") * abs(self.quantity)

    @property
    def market_value(self) -> Decimal:
        """Current market value."""
        return self.current_price * Decimal("100") * self.quantity


@dataclass
class StockLeg:
    """Stock leg in a position group."""
    symbol: str
    quantity: int
    current_price: Decimal

    @property
    def is_long(self) -> bool:
        """True if long position."""
        return self.quantity > 0

    @property
    def market_value(self) -> Decimal:
        """Current market value."""
        return self.current_price * abs(self.quantity)


@dataclass
class PositionGroup:
    """Group of related positions for margin calculation."""
    underlying: str
    option_legs: List[OptionLeg] = field(default_factory=list)
    stock_leg: Optional[StockLeg] = None
    spread_type: SpreadType = SpreadType.UNKNOWN

    @property
    def is_defined_risk(self) -> bool:
        """True if spread has defined (limited) risk."""
        defined_risk_types = {
            SpreadType.COVERED_CALL,
            SpreadType.CASH_SECURED_PUT,
            SpreadType.VERTICAL_CALL,
            SpreadType.VERTICAL_PUT,
            SpreadType.IRON_CONDOR,
            SpreadType.IRON_BUTTERFLY,
        }
        return self.spread_type in defined_risk_types

    @property
    def net_delta(self) -> Optional[float]:
        """Net delta of the position group."""
        if not all(leg.delta is not None for leg in self.option_legs):
            return None

        delta = sum(
            (leg.delta or 0) * leg.quantity * 100
            for leg in self.option_legs
        )

        if self.stock_leg:
            delta += self.stock_leg.quantity

        return delta


@dataclass
class MarginResult:
    """Result of margin calculation."""
    initial_margin: Decimal
    maintenance_margin: Decimal
    max_loss: Optional[Decimal] = None
    spread_type: SpreadType = SpreadType.UNKNOWN
    details: Dict[str, Any] = field(default_factory=dict)


class PositionGroupMarginModel:
    """
    Calculate risk-based margin for option position groups.

    Uses max loss as margin for defined-risk spreads instead of
    summing individual leg margins.
    """

    # Contract multiplier
    MULTIPLIER = Decimal("100")

    # Margin requirements for undefined risk
    NAKED_MARGIN_PCT = Decimal("0.20")  # 20% of underlying
    NAKED_MIN_PCT = Decimal("0.10")     # 10% minimum

    def __init__(
        self,
        use_portfolio_margin: bool = False,
    ):
        """
        Initialize margin model.

        Args:
            use_portfolio_margin: Use portfolio margin (lower requirements)
        """
        self.use_portfolio_margin = use_portfolio_margin

        if use_portfolio_margin:
            self.NAKED_MARGIN_PCT = Decimal("0.15")
            self.NAKED_MIN_PCT = Decimal("0.075")

    def identify_spread_type(
        self,
        group: PositionGroup,
    ) -> SpreadType:
        """
        Identify the type of spread from its legs.

        Args:
            group: Position group to analyze

        Returns:
            SpreadType enum
        """
        legs = group.option_legs
        stock = group.stock_leg

        if len(legs) == 0:
            return SpreadType.UNKNOWN

        if len(legs) == 1:
            leg = legs[0]

            # Check for covered call
            if leg.is_call and leg.is_short and stock:
                if stock.is_long and stock.quantity >= abs(leg.quantity) * 100:
                    return SpreadType.COVERED_CALL

            # Check for cash-secured put
            if leg.is_put and leg.is_short:
                return SpreadType.CASH_SECURED_PUT

            # Naked options
            if leg.is_short:
                return SpreadType.NAKED_CALL if leg.is_call else SpreadType.NAKED_PUT

            return SpreadType.UNKNOWN

        if len(legs) == 2:
            # Vertical spreads
            if legs[0].option_type == legs[1].option_type:
                if legs[0].expiration == legs[1].expiration:
                    is_call = legs[0].is_call
                    return SpreadType.VERTICAL_CALL if is_call else SpreadType.VERTICAL_PUT

            # Straddle (same strike, same expiration)
            if (legs[0].strike == legs[1].strike and
                legs[0].expiration == legs[1].expiration and
                legs[0].option_type != legs[1].option_type):
                return SpreadType.STRADDLE

            # Strangle (different strikes, same expiration)
            if (legs[0].strike != legs[1].strike and
                legs[0].expiration == legs[1].expiration and
                legs[0].option_type != legs[1].option_type):
                return SpreadType.STRANGLE

            # Calendar spread (same strike, different expiration)
            if (legs[0].strike == legs[1].strike and
                legs[0].expiration != legs[1].expiration and
                legs[0].option_type == legs[1].option_type):
                return SpreadType.CALENDAR

            # Diagonal spread
            if (legs[0].strike != legs[1].strike and
                legs[0].expiration != legs[1].expiration and
                legs[0].option_type == legs[1].option_type):
                return SpreadType.DIAGONAL

        if len(legs) == 4:
            # Iron condor / Iron butterfly
            calls = [leg for leg in legs if leg.is_call]
            puts = [leg for leg in legs if leg.is_put]

            if len(calls) == 2 and len(puts) == 2:
                # Check if iron condor (4 different strikes)
                strikes = {leg.strike for leg in legs}
                if len(strikes) == 4:
                    return SpreadType.IRON_CONDOR

                # Iron butterfly (2 strikes at center)
                if len(strikes) == 3:
                    return SpreadType.IRON_BUTTERFLY

        return SpreadType.UNKNOWN

    def calculate_max_loss(
        self,
        group: PositionGroup,
        underlying_price: Decimal,
    ) -> Optional[Decimal]:
        """
        Calculate maximum possible loss for a position group.

        Args:
            group: Position group
            underlying_price: Current underlying price

        Returns:
            Maximum loss or None if undefined risk
        """
        spread_type = group.spread_type

        if spread_type == SpreadType.COVERED_CALL:
            # Max loss is downside of stock minus premium received
            leg = group.option_legs[0]
            stock = group.stock_leg
            if stock:
                premium_received = abs(leg.quantity) * leg.premium * self.MULTIPLIER
                stock_cost = stock.quantity * stock.current_price
                # Max loss if stock goes to zero minus premium
                return stock_cost - premium_received

        elif spread_type == SpreadType.CASH_SECURED_PUT:
            leg = group.option_legs[0]
            # Max loss is strike price minus premium (if stock goes to 0)
            max_loss = leg.strike * self.MULTIPLIER * abs(leg.quantity)
            premium_received = leg.premium * self.MULTIPLIER * abs(leg.quantity)
            return max_loss - premium_received

        elif spread_type in (SpreadType.VERTICAL_CALL, SpreadType.VERTICAL_PUT):
            # Max loss is width of strikes minus net credit/plus net debit
            legs = sorted(group.option_legs, key=lambda leg: leg.strike)
            width = (legs[1].strike - legs[0].strike) * self.MULTIPLIER

            # Net premium (positive = credit, negative = debit)
            net_premium = sum(
                leg.premium * self.MULTIPLIER * leg.quantity
                for leg in legs
            )

            if net_premium > 0:
                # Credit spread: max loss = width - credit
                return (width - net_premium) * abs(legs[0].quantity)
            else:
                # Debit spread: max loss = debit paid
                return abs(net_premium) * abs(legs[0].quantity)

        elif spread_type == SpreadType.IRON_CONDOR:
            # Max loss is width of wider spread minus net credit
            calls = sorted([leg for leg in group.option_legs if leg.is_call], key=lambda leg: leg.strike)
            puts = sorted([leg for leg in group.option_legs if leg.is_put], key=lambda leg: leg.strike)

            call_width = (calls[1].strike - calls[0].strike) * self.MULTIPLIER
            put_width = (puts[1].strike - puts[0].strike) * self.MULTIPLIER

            max_width = max(call_width, put_width)

            # Net credit received
            net_premium = sum(
                leg.premium * self.MULTIPLIER * leg.quantity
                for leg in group.option_legs
            )

            return (max_width - net_premium) * abs(calls[0].quantity)

        elif spread_type == SpreadType.IRON_BUTTERFLY:
            # Similar to iron condor
            calls = sorted([leg for leg in group.option_legs if leg.is_call], key=lambda leg: leg.strike)
            puts = sorted([leg for leg in group.option_legs if leg.is_put], key=lambda leg: leg.strike)

            # Width from center to wing
            if len(calls) >= 2 and len(puts) >= 2:
                call_width = (calls[1].strike - calls[0].strike) * self.MULTIPLIER
                put_width = (puts[1].strike - puts[0].strike) * self.MULTIPLIER
                max_width = max(call_width, put_width)

                net_premium = sum(
                    leg.premium * self.MULTIPLIER * leg.quantity
                    for leg in group.option_legs
                )

                return (max_width - net_premium) * abs(calls[0].quantity)

        # Undefined risk
        return None

    def calculate_naked_margin(
        self,
        leg: OptionLeg,
        underlying_price: Decimal,
    ) -> Decimal:
        """
        Calculate margin for a naked option.

        Formula: Max(20% underlying - OTM amount, 10% underlying) + premium

        Args:
            leg: Naked option leg
            underlying_price: Current underlying price

        Returns:
            Required margin
        """
        multiplier = self.MULTIPLIER * abs(leg.quantity)

        # Calculate OTM amount
        if leg.is_call:
            otm_amount = max(Decimal("0"), leg.strike - underlying_price)
        else:
            otm_amount = max(Decimal("0"), underlying_price - leg.strike)

        # 20% of underlying minus OTM
        margin_1 = (
            underlying_price * self.NAKED_MARGIN_PCT - otm_amount
        ) * multiplier

        # 10% minimum
        margin_2 = underlying_price * self.NAKED_MIN_PCT * multiplier

        # Premium value
        premium_value = leg.current_price * multiplier

        return max(margin_1, margin_2) + premium_value

    def calculate_margin(
        self,
        group: PositionGroup,
        underlying_price: Decimal,
    ) -> MarginResult:
        """
        Calculate margin for a position group.

        Args:
            group: Position group to calculate margin for
            underlying_price: Current price of underlying

        Returns:
            MarginResult with initial and maintenance margin
        """
        # Identify spread type if not already set
        if group.spread_type == SpreadType.UNKNOWN:
            group.spread_type = self.identify_spread_type(group)

        spread_type = group.spread_type

        # For defined risk spreads, margin = max loss
        if group.is_defined_risk:
            max_loss = self.calculate_max_loss(group, underlying_price)

            if max_loss is not None:
                return MarginResult(
                    initial_margin=max_loss,
                    maintenance_margin=max_loss,
                    max_loss=max_loss,
                    spread_type=spread_type,
                    details={"method": "risk_based"},
                )

        # For undefined risk, use naked option margin
        total_margin = Decimal("0")

        for leg in group.option_legs:
            if leg.is_short:
                leg_margin = self.calculate_naked_margin(leg, underlying_price)
                total_margin += leg_margin

        # Long options reduce margin slightly
        long_premium = sum(
            leg.current_price * self.MULTIPLIER * abs(leg.quantity)
            for leg in group.option_legs
            if leg.is_long
        )

        # Stock leg margin
        if group.stock_leg and group.stock_leg.quantity < 0:
            # Short stock requires margin
            stock_margin = (
                abs(group.stock_leg.quantity) *
                group.stock_leg.current_price *
                Decimal("0.50")  # 50% for short stock
            )
            total_margin += stock_margin

        initial_margin = total_margin
        maintenance_margin = total_margin * Decimal("0.75")  # Lower maintenance

        return MarginResult(
            initial_margin=initial_margin,
            maintenance_margin=maintenance_margin,
            max_loss=None,
            spread_type=spread_type,
            details={
                "method": "standard",
                "long_premium_offset": long_premium,
            },
        )

    def calculate_portfolio_margin(
        self,
        groups: List[PositionGroup],
        underlying_prices: Dict[str, Decimal],
        account_value: Decimal,
    ) -> Dict[str, MarginResult]:
        """
        Calculate margin for all position groups in a portfolio.

        Args:
            groups: List of position groups
            underlying_prices: Dict of underlying symbol -> price
            account_value: Total account value

        Returns:
            Dict of underlying symbol -> MarginResult
        """
        results = {}

        for group in groups:
            underlying_price = underlying_prices.get(
                group.underlying, Decimal("0")
            )

            if underlying_price > 0:
                results[group.underlying] = self.calculate_margin(
                    group, underlying_price
                )

        return results

    def get_total_margin(
        self,
        results: Dict[str, MarginResult],
    ) -> Tuple[Decimal, Decimal]:
        """
        Get total margin across all positions.

        Args:
            results: Margin results by underlying

        Returns:
            Tuple of (total_initial, total_maintenance)
        """
        total_initial = sum(
            r.initial_margin for r in results.values()
        )
        total_maintenance = sum(
            r.maintenance_margin for r in results.values()
        )

        return total_initial, total_maintenance

