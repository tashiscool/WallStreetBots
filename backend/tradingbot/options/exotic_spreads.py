"""
Exotic Option Spreads

Dataclasses and utilities for multi-leg option strategies:
- Iron Condor
- Butterfly (Iron & Broken Wing)
- Calendar Spread
- Diagonal Spread
- Straddle
- Strangle
- Ratio Spread
"""

from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from enum import Enum
from typing import List, Optional, Tuple


class SpreadType(Enum):
    """Types of option spreads."""
    IRON_CONDOR = "iron_condor"
    IRON_BUTTERFLY = "iron_butterfly"
    BUTTERFLY = "butterfly"
    BROKEN_WING_BUTTERFLY = "broken_wing_butterfly"
    CALENDAR = "calendar"
    DIAGONAL = "diagonal"
    STRADDLE = "straddle"
    STRANGLE = "strangle"
    RATIO_SPREAD = "ratio_spread"
    VERTICAL_CALL = "vertical_call"
    VERTICAL_PUT = "vertical_put"


class SpreadDirection(Enum):
    """Direction of spread (bullish/bearish/neutral)."""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class LegType(Enum):
    """Type of option leg."""
    LONG_CALL = "long_call"
    SHORT_CALL = "short_call"
    LONG_PUT = "long_put"
    SHORT_PUT = "short_put"


@dataclass
class SpreadLeg:
    """Individual leg of an option spread."""
    leg_type: LegType
    strike: Decimal
    expiry: date
    contracts: int  # Positive for long, negative for short
    premium: Optional[Decimal] = None
    delta: Optional[Decimal] = None
    gamma: Optional[Decimal] = None
    theta: Optional[Decimal] = None
    vega: Optional[Decimal] = None

    @property
    def is_long(self) -> bool:
        return self.contracts > 0

    @property
    def is_call(self) -> bool:
        return self.leg_type in (LegType.LONG_CALL, LegType.SHORT_CALL)

    @property
    def option_type(self) -> str:
        return "call" if self.is_call else "put"


@dataclass
class SpreadGreeks:
    """Aggregate Greeks for the entire spread."""
    delta: Decimal = Decimal("0")
    gamma: Decimal = Decimal("0")
    theta: Decimal = Decimal("0")
    vega: Decimal = Decimal("0")

    def to_dict(self) -> dict:
        return {
            "delta": float(self.delta),
            "gamma": float(self.gamma),
            "theta": float(self.theta),
            "vega": float(self.vega),
        }


@dataclass
class OptionSpread:
    """Base class for all option spreads."""
    spread_type: SpreadType
    ticker: str
    legs: List[SpreadLeg]
    direction: SpreadDirection = SpreadDirection.NEUTRAL
    created_at: Optional[date] = None

    @property
    def total_contracts(self) -> int:
        """Total absolute number of contracts."""
        return sum(abs(leg.contracts) for leg in self.legs)

    @property
    def net_premium(self) -> Decimal:
        """Net premium received (positive) or paid (negative)."""
        total = Decimal("0")
        for leg in self.legs:
            if leg.premium:
                # Short legs receive premium, long legs pay premium
                multiplier = -1 if leg.is_long else 1
                total += leg.premium * abs(leg.contracts) * multiplier
        return total

    @property
    def is_credit(self) -> bool:
        """Returns True if this is a credit spread (premium received)."""
        return self.net_premium > 0

    @property
    def aggregate_greeks(self) -> SpreadGreeks:
        """Calculate aggregate Greeks for the spread."""
        greeks = SpreadGreeks()
        for leg in self.legs:
            multiplier = leg.contracts  # Already signed
            if leg.delta:
                greeks.delta += leg.delta * multiplier
            if leg.gamma:
                greeks.gamma += leg.gamma * multiplier
            if leg.theta:
                greeks.theta += leg.theta * multiplier
            if leg.vega:
                greeks.vega += leg.vega * multiplier
        return greeks

    def get_max_profit(self) -> Optional[Decimal]:
        """Calculate maximum profit. Override in subclasses."""
        return None

    def get_max_loss(self) -> Optional[Decimal]:
        """Calculate maximum loss. Override in subclasses."""
        return None

    def get_breakeven_points(self) -> List[Decimal]:
        """Calculate breakeven points. Override in subclasses."""
        return []


@dataclass
class IronCondor(OptionSpread):
    """
    Iron Condor: 4-leg neutral income strategy.

    Structure:
    - Sell OTM Put (lower strike)
    - Buy OTM Put (even lower strike)
    - Sell OTM Call (higher strike)
    - Buy OTM Call (even higher strike)

    Profit zone: Between the two short strikes.
    """
    put_long_strike: Decimal = Decimal("0")
    put_short_strike: Decimal = Decimal("0")
    call_short_strike: Decimal = Decimal("0")
    call_long_strike: Decimal = Decimal("0")
    expiry: Optional[date] = None

    def __post_init__(self):
        self.spread_type = SpreadType.IRON_CONDOR
        self.direction = SpreadDirection.NEUTRAL

    @property
    def wing_width_put(self) -> Decimal:
        """Width of the put spread wing."""
        return self.put_short_strike - self.put_long_strike

    @property
    def wing_width_call(self) -> Decimal:
        """Width of the call spread wing."""
        return self.call_long_strike - self.call_short_strike

    @property
    def profit_zone_width(self) -> Decimal:
        """Width between short strikes (profit zone)."""
        return self.call_short_strike - self.put_short_strike

    def get_max_profit(self) -> Decimal:
        """Max profit = net premium received."""
        return self.net_premium * 100  # Per contract

    def get_max_loss(self) -> Decimal:
        """Max loss = wider wing width - premium received."""
        max_wing = max(self.wing_width_put, self.wing_width_call)
        return (max_wing * 100) - self.net_premium * 100

    def get_breakeven_points(self) -> List[Decimal]:
        """Two breakeven points."""
        lower_be = self.put_short_strike - self.net_premium
        upper_be = self.call_short_strike + self.net_premium
        return [lower_be, upper_be]


@dataclass
class IronButterfly(OptionSpread):
    """
    Iron Butterfly: 4-leg neutral strategy.

    Structure:
    - Sell ATM Put
    - Buy OTM Put (lower strike)
    - Sell ATM Call (same strike as ATM put)
    - Buy OTM Call (higher strike)

    Similar to iron condor but short strikes are at the same level.
    """
    center_strike: Decimal = Decimal("0")
    lower_strike: Decimal = Decimal("0")
    upper_strike: Decimal = Decimal("0")
    expiry: Optional[date] = None

    def __post_init__(self):
        self.spread_type = SpreadType.IRON_BUTTERFLY
        self.direction = SpreadDirection.NEUTRAL

    @property
    def wing_width(self) -> Decimal:
        """Width of the wings."""
        return min(
            self.center_strike - self.lower_strike,
            self.upper_strike - self.center_strike
        )

    def get_max_profit(self) -> Decimal:
        """Max profit at center strike."""
        return self.net_premium * 100

    def get_max_loss(self) -> Decimal:
        """Max loss if price moves to either wing."""
        return (self.wing_width * 100) - self.net_premium * 100

    def get_breakeven_points(self) -> List[Decimal]:
        """Two breakeven points."""
        lower_be = self.center_strike - self.net_premium
        upper_be = self.center_strike + self.net_premium
        return [lower_be, upper_be]


@dataclass
class Butterfly(OptionSpread):
    """
    Butterfly Spread: 3-strike strategy (using only calls or puts).

    Call Butterfly:
    - Buy 1 lower strike call
    - Sell 2 middle strike calls
    - Buy 1 higher strike call

    Put Butterfly:
    - Buy 1 lower strike put
    - Sell 2 middle strike puts
    - Buy 1 higher strike put
    """
    lower_strike: Decimal = Decimal("0")
    middle_strike: Decimal = Decimal("0")
    upper_strike: Decimal = Decimal("0")
    option_type: str = "call"  # "call" or "put"
    expiry: Optional[date] = None

    def __post_init__(self):
        self.spread_type = SpreadType.BUTTERFLY
        self.direction = SpreadDirection.NEUTRAL

    @property
    def wing_width(self) -> Decimal:
        """Width of wings (should be equal for balanced butterfly)."""
        return self.middle_strike - self.lower_strike

    def get_max_profit(self) -> Decimal:
        """Max profit at middle strike."""
        return (self.wing_width - abs(self.net_premium)) * 100

    def get_max_loss(self) -> Decimal:
        """Max loss = premium paid for debit butterfly."""
        return abs(self.net_premium) * 100

    def get_breakeven_points(self) -> List[Decimal]:
        """Two breakeven points."""
        lower_be = self.lower_strike + abs(self.net_premium)
        upper_be = self.upper_strike - abs(self.net_premium)
        return [lower_be, upper_be]


@dataclass
class BrokenWingButterfly(OptionSpread):
    """
    Broken Wing Butterfly: Unbalanced butterfly with skewed risk.

    Useful for directional bias with limited risk.
    """
    lower_strike: Decimal = Decimal("0")
    middle_strike: Decimal = Decimal("0")
    upper_strike: Decimal = Decimal("0")
    option_type: str = "call"
    expiry: Optional[date] = None
    skip_strikes: int = 1  # How many strikes to skip on the far wing

    def __post_init__(self):
        self.spread_type = SpreadType.BROKEN_WING_BUTTERFLY
        # Direction depends on which wing is wider
        if self.upper_strike - self.middle_strike > self.middle_strike - self.lower_strike:
            self.direction = SpreadDirection.BULLISH
        else:
            self.direction = SpreadDirection.BEARISH


@dataclass
class CalendarSpread(OptionSpread):
    """
    Calendar Spread (Horizontal Spread): Same strike, different expirations.

    Structure:
    - Sell near-term option
    - Buy longer-term option (same strike)

    Profits from time decay differential.
    """
    strike: Decimal = Decimal("0")
    near_expiry: Optional[date] = None
    far_expiry: Optional[date] = None
    option_type: str = "call"

    def __post_init__(self):
        self.spread_type = SpreadType.CALENDAR
        self.direction = SpreadDirection.NEUTRAL

    @property
    def days_between(self) -> int:
        """Days between expiration dates."""
        if self.near_expiry and self.far_expiry:
            return (self.far_expiry - self.near_expiry).days
        return 0

    def get_max_loss(self) -> Decimal:
        """Max loss = premium paid for the spread."""
        return abs(self.net_premium) * 100


@dataclass
class DiagonalSpread(OptionSpread):
    """
    Diagonal Spread: Different strikes AND different expirations.

    Structure:
    - Sell near-term option at one strike
    - Buy longer-term option at different strike

    Combines elements of vertical and calendar spreads.
    """
    near_strike: Decimal = Decimal("0")
    far_strike: Decimal = Decimal("0")
    near_expiry: Optional[date] = None
    far_expiry: Optional[date] = None
    option_type: str = "call"

    def __post_init__(self):
        self.spread_type = SpreadType.DIAGONAL
        # Determine direction based on strike relationship
        if self.option_type == "call":
            if self.far_strike > self.near_strike:
                self.direction = SpreadDirection.BULLISH
            else:
                self.direction = SpreadDirection.BEARISH
        else:
            if self.far_strike < self.near_strike:
                self.direction = SpreadDirection.BULLISH
            else:
                self.direction = SpreadDirection.BEARISH


@dataclass
class Straddle(OptionSpread):
    """
    Straddle: ATM call + ATM put at same strike and expiry.

    Long Straddle: Buy both (profit from big moves either direction)
    Short Straddle: Sell both (profit from low volatility)
    """
    strike: Decimal = Decimal("0")
    expiry: Optional[date] = None
    is_long: bool = True  # Long straddle or short straddle

    def __post_init__(self):
        self.spread_type = SpreadType.STRADDLE
        self.direction = SpreadDirection.NEUTRAL

    def get_breakeven_points(self) -> List[Decimal]:
        """Two breakeven points for long straddle."""
        premium = abs(self.net_premium)
        lower_be = self.strike - premium
        upper_be = self.strike + premium
        return [lower_be, upper_be]

    def get_max_loss(self) -> Decimal:
        """Max loss for long straddle = premium paid."""
        if self.is_long:
            return abs(self.net_premium) * 100
        else:
            return Decimal("999999")  # Unlimited for short straddle

    def get_max_profit(self) -> Optional[Decimal]:
        """Max profit for short straddle = premium received."""
        if self.is_long:
            return None  # Unlimited for long straddle
        else:
            return self.net_premium * 100


@dataclass
class Strangle(OptionSpread):
    """
    Strangle: OTM call + OTM put at different strikes.

    Long Strangle: Buy both (cheaper than straddle, needs bigger move)
    Short Strangle: Sell both (wider profit zone, but more risk)
    """
    put_strike: Decimal = Decimal("0")
    call_strike: Decimal = Decimal("0")
    expiry: Optional[date] = None
    is_long: bool = True

    def __post_init__(self):
        self.spread_type = SpreadType.STRANGLE
        self.direction = SpreadDirection.NEUTRAL

    @property
    def width(self) -> Decimal:
        """Width between strikes."""
        return self.call_strike - self.put_strike

    def get_breakeven_points(self) -> List[Decimal]:
        """Two breakeven points."""
        premium = abs(self.net_premium)
        lower_be = self.put_strike - premium
        upper_be = self.call_strike + premium
        return [lower_be, upper_be]

    def get_max_loss(self) -> Decimal:
        """Max loss for long strangle = premium paid."""
        if self.is_long:
            return abs(self.net_premium) * 100
        else:
            return Decimal("999999")  # Unlimited for short strangle


@dataclass
class RatioSpread(OptionSpread):
    """
    Ratio Spread: Unequal quantities of options.

    Common ratios: 1:2, 1:3, 2:3

    Example 1:2 Call Ratio:
    - Buy 1 ATM call
    - Sell 2 OTM calls

    Can be done for credit, but has unlimited risk on one side.
    """
    long_strike: Decimal = Decimal("0")
    short_strike: Decimal = Decimal("0")
    long_contracts: int = 1
    short_contracts: int = 2
    option_type: str = "call"
    expiry: Optional[date] = None

    def __post_init__(self):
        self.spread_type = SpreadType.RATIO_SPREAD
        # Direction depends on option type and strike relationship
        if self.option_type == "call":
            self.direction = SpreadDirection.BEARISH if self.short_contracts > self.long_contracts else SpreadDirection.BULLISH
        else:
            self.direction = SpreadDirection.BULLISH if self.short_contracts > self.long_contracts else SpreadDirection.BEARISH

    @property
    def ratio(self) -> str:
        """Return the ratio as a string."""
        return f"{self.long_contracts}:{self.short_contracts}"

    @property
    def net_short(self) -> int:
        """Net short contracts (exposed to unlimited risk)."""
        return max(0, self.short_contracts - self.long_contracts)

    def get_breakeven_points(self) -> List[Decimal]:
        """Calculate breakeven points for ratio spread."""
        # Lower breakeven (for call ratio spread)
        if self.option_type == "call":
            lower_be = self.long_strike + abs(self.net_premium)
            # Upper breakeven depends on ratio
            spread_width = self.short_strike - self.long_strike
            upper_be = self.short_strike + (spread_width * self.net_short) - abs(self.net_premium)
            return [lower_be, upper_be]
        else:
            # Put ratio spread breakevens
            upper_be = self.long_strike - abs(self.net_premium)
            spread_width = self.long_strike - self.short_strike
            lower_be = self.short_strike - (spread_width * self.net_short) + abs(self.net_premium)
            return [lower_be, upper_be]


@dataclass
class SpreadAnalysis:
    """Analysis results for a spread."""
    spread: OptionSpread
    max_profit: Optional[Decimal] = None
    max_loss: Optional[Decimal] = None
    breakeven_points: List[Decimal] = field(default_factory=list)
    probability_of_profit: Optional[float] = None
    expected_value: Optional[Decimal] = None
    risk_reward_ratio: Optional[float] = None
    greeks: Optional[SpreadGreeks] = None
    recommendation: str = ""
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert analysis to dictionary."""
        return {
            "spread_type": self.spread.spread_type.value,
            "ticker": self.spread.ticker,
            "direction": self.spread.direction.value,
            "max_profit": float(self.max_profit) if self.max_profit else None,
            "max_loss": float(self.max_loss) if self.max_loss else None,
            "breakeven_points": [float(be) for be in self.breakeven_points],
            "probability_of_profit": self.probability_of_profit,
            "expected_value": float(self.expected_value) if self.expected_value else None,
            "risk_reward_ratio": self.risk_reward_ratio,
            "greeks": self.greeks.to_dict() if self.greeks else None,
            "recommendation": self.recommendation,
            "notes": self.notes,
        }
