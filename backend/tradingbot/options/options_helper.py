"""
Options Helper - Multi-leg Spread Builder with Greeks Aggregation

Ported from Lumibot's options_helper.py, adapted for stock/options trading
with Alpaca integration.

Supports:
- Vertical spreads (bull call, bear put, bull put, bear call)
- Straddles and strangles
- Iron condors and iron butterflies
- Calendar spreads
- Ratio spreads
- Custom multi-leg combinations
"""

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)


class OptionType(Enum):
    """Option type enum."""
    CALL = "call"
    PUT = "put"


class OptionSide(Enum):
    """Option position side."""
    LONG = "long"
    SHORT = "short"


class SpreadType(Enum):
    """Common spread types."""
    # Vertical spreads
    BULL_CALL_SPREAD = "bull_call_spread"
    BEAR_CALL_SPREAD = "bear_call_spread"
    BULL_PUT_SPREAD = "bull_put_spread"
    BEAR_PUT_SPREAD = "bear_put_spread"

    # Straddles/Strangles
    LONG_STRADDLE = "long_straddle"
    SHORT_STRADDLE = "short_straddle"
    LONG_STRANGLE = "long_strangle"
    SHORT_STRANGLE = "short_strangle"

    # Iron spreads
    IRON_CONDOR = "iron_condor"
    IRON_BUTTERFLY = "iron_butterfly"

    # Calendar/Diagonal
    CALENDAR_SPREAD = "calendar_spread"
    DIAGONAL_SPREAD = "diagonal_spread"

    # Other
    RATIO_SPREAD = "ratio_spread"
    BUTTERFLY = "butterfly"
    CUSTOM = "custom"


@dataclass
class OptionLeg:
    """Single leg of an options spread."""
    symbol: str
    option_type: OptionType
    strike: Decimal
    expiration: date
    side: OptionSide
    quantity: int = 1

    # Pricing (filled after market data lookup)
    bid: Optional[Decimal] = None
    ask: Optional[Decimal] = None
    last: Optional[Decimal] = None
    mid: Optional[Decimal] = None

    # Greeks (filled after market data lookup)
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None
    rho: Optional[float] = None
    implied_volatility: Optional[float] = None

    # Additional data
    open_interest: Optional[int] = None
    volume: Optional[int] = None

    def __post_init__(self):
        """Convert types if needed."""
        if isinstance(self.strike, (int, float)):
            self.strike = Decimal(str(self.strike))
        if isinstance(self.option_type, str):
            self.option_type = OptionType(self.option_type.lower())
        if isinstance(self.side, str):
            self.side = OptionSide(self.side.lower())
        if isinstance(self.expiration, str):
            self.expiration = datetime.strptime(self.expiration, "%Y-%m-%d").date()

    @property
    def multiplier(self) -> int:
        """Position multiplier (positive for long, negative for short)."""
        return self.quantity if self.side == OptionSide.LONG else -self.quantity

    @property
    def contract_symbol(self) -> str:
        """
        Generate OCC option symbol.
        Format: SYMBOL + YYMMDD + C/P + Strike (8 digits, 3 decimals implied)
        Example: AAPL240119C00185000 for AAPL Jan 19 2024 185 Call
        """
        exp_str = self.expiration.strftime("%y%m%d")
        type_char = "C" if self.option_type == OptionType.CALL else "P"
        # Strike is stored as whole dollars, convert to 8 digits with 3 implied decimals
        strike_int = int(self.strike * 1000)
        strike_str = f"{strike_int:08d}"
        return f"{self.symbol}{exp_str}{type_char}{strike_str}"

    def get_cost(self) -> Optional[Decimal]:
        """Get cost to open this leg (negative = credit)."""
        if self.mid is None:
            return None
        # Long positions cost money (debit), short positions receive money (credit)
        return self.mid * self.multiplier * 100  # Options are 100 shares per contract


@dataclass
class OptionsSpread:
    """
    Multi-leg options spread with aggregated Greeks and P&L calculations.
    """
    symbol: str
    spread_type: SpreadType
    legs: List[OptionLeg] = field(default_factory=list)
    name: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)

    # Calculated fields
    _net_delta: Optional[float] = None
    _net_gamma: Optional[float] = None
    _net_theta: Optional[float] = None
    _net_vega: Optional[float] = None
    _max_profit: Optional[Decimal] = None
    _max_loss: Optional[Decimal] = None
    _breakeven_points: List[Decimal] = field(default_factory=list)

    def add_leg(self, leg: OptionLeg) -> None:
        """Add a leg to the spread."""
        self.legs.append(leg)
        self._invalidate_cache()

    def remove_leg(self, index: int) -> Optional[OptionLeg]:
        """Remove a leg by index."""
        if 0 <= index < len(self.legs):
            leg = self.legs.pop(index)
            self._invalidate_cache()
            return leg
        return None

    def _invalidate_cache(self) -> None:
        """Clear cached calculations."""
        self._net_delta = None
        self._net_gamma = None
        self._net_theta = None
        self._net_vega = None
        self._max_profit = None
        self._max_loss = None
        self._breakeven_points = []

    @property
    def net_delta(self) -> Optional[float]:
        """Net delta of the spread."""
        if self._net_delta is None:
            self._calculate_greeks()
        return self._net_delta

    @property
    def net_gamma(self) -> Optional[float]:
        """Net gamma of the spread."""
        if self._net_gamma is None:
            self._calculate_greeks()
        return self._net_gamma

    @property
    def net_theta(self) -> Optional[float]:
        """Net theta of the spread (daily decay)."""
        if self._net_theta is None:
            self._calculate_greeks()
        return self._net_theta

    @property
    def net_vega(self) -> Optional[float]:
        """Net vega of the spread."""
        if self._net_vega is None:
            self._calculate_greeks()
        return self._net_vega

    def _calculate_greeks(self) -> None:
        """Calculate aggregate Greeks for the spread."""
        self._net_delta = 0.0
        self._net_gamma = 0.0
        self._net_theta = 0.0
        self._net_vega = 0.0

        for leg in self.legs:
            multiplier = leg.multiplier
            if leg.delta is not None:
                self._net_delta += leg.delta * multiplier * 100
            if leg.gamma is not None:
                self._net_gamma += leg.gamma * multiplier * 100
            if leg.theta is not None:
                self._net_theta += leg.theta * multiplier * 100
            if leg.vega is not None:
                self._net_vega += leg.vega * multiplier * 100

    @property
    def net_cost(self) -> Optional[Decimal]:
        """
        Net cost to open the spread.
        Positive = debit spread (pay to open)
        Negative = credit spread (receive to open)
        """
        total = Decimal("0")
        for leg in self.legs:
            cost = leg.get_cost()
            if cost is None:
                return None
            total += cost
        return total

    @property
    def is_debit_spread(self) -> bool:
        """True if this is a debit spread (costs money to open)."""
        cost = self.net_cost
        return cost is not None and cost > 0

    @property
    def is_credit_spread(self) -> bool:
        """True if this is a credit spread (receive money to open)."""
        cost = self.net_cost
        return cost is not None and cost < 0

    @property
    def days_to_expiration(self) -> Optional[int]:
        """Days to nearest expiration."""
        if not self.legs:
            return None
        min_exp = min(leg.expiration for leg in self.legs)
        return (min_exp - date.today()).days

    @property
    def expiration_dates(self) -> List[date]:
        """All expiration dates in the spread."""
        return sorted({leg.expiration for leg in self.legs})

    @property
    def strikes(self) -> List[Decimal]:
        """All strikes in the spread, sorted."""
        return sorted({leg.strike for leg in self.legs})

    @property
    def width(self) -> Optional[Decimal]:
        """Width of the spread (difference between strikes)."""
        strikes = self.strikes
        if len(strikes) < 2:
            return None
        return max(strikes) - min(strikes)

    def calculate_profit_at_expiry(self, underlying_price: Decimal) -> Decimal:
        """
        Calculate profit/loss at expiration for a given underlying price.

        Args:
            underlying_price: Price of underlying at expiration

        Returns:
            Profit/loss in dollars (positive = profit)
        """
        total_value = Decimal("0")

        for leg in self.legs:
            # Calculate intrinsic value at expiration
            if leg.option_type == OptionType.CALL:
                intrinsic = max(underlying_price - leg.strike, Decimal("0"))
            else:  # PUT
                intrinsic = max(leg.strike - underlying_price, Decimal("0"))

            # Long positions have positive value, short have negative
            total_value += intrinsic * leg.multiplier * 100

        # Subtract the cost paid to open (or add credit received)
        net_cost = self.net_cost or Decimal("0")
        return total_value - net_cost

    def get_profit_loss_range(
        self,
        price_range: Optional[Tuple[Decimal, Decimal]] = None,
        num_points: int = 50
    ) -> List[Tuple[Decimal, Decimal]]:
        """
        Calculate P&L across a range of underlying prices.

        Args:
            price_range: (min_price, max_price) or None to auto-calculate
            num_points: Number of price points to calculate

        Returns:
            List of (price, profit) tuples
        """
        if price_range is None:
            strikes = self.strikes
            if not strikes:
                return []
            min_strike = min(strikes)
            max_strike = max(strikes)
            margin = (max_strike - min_strike) * Decimal("0.5")
            if margin == 0:
                margin = min_strike * Decimal("0.1")
            price_range = (min_strike - margin, max_strike + margin)

        min_price, max_price = price_range
        step = (max_price - min_price) / (num_points - 1)

        results = []
        for i in range(num_points):
            price = min_price + step * i
            profit = self.calculate_profit_at_expiry(price)
            results.append((price, profit))

        return results

    def get_breakeven_points(self) -> List[Decimal]:
        """
        Find breakeven points where P&L = 0.

        Returns:
            List of underlying prices where spread breaks even
        """
        if self._breakeven_points:
            return self._breakeven_points

        # Get P&L range
        pl_range = self.get_profit_loss_range(num_points=200)

        breakevens = []
        for i in range(1, len(pl_range)):
            prev_price, prev_pl = pl_range[i - 1]
            curr_price, curr_pl = pl_range[i]

            # Check for sign change (crossing zero)
            if (prev_pl > 0 and curr_pl < 0) or (prev_pl < 0 and curr_pl > 0):
                # Linear interpolation to find approximate breakeven
                ratio = abs(prev_pl) / (abs(prev_pl) + abs(curr_pl))
                breakeven = prev_price + (curr_price - prev_price) * Decimal(str(ratio))
                breakevens.append(breakeven.quantize(Decimal("0.01")))

        self._breakeven_points = breakevens
        return breakevens

    def get_max_profit(self) -> Optional[Decimal]:
        """Calculate maximum possible profit."""
        if self._max_profit is not None:
            return self._max_profit

        pl_range = self.get_profit_loss_range(num_points=200)
        if not pl_range:
            return None

        self._max_profit = max(pl for _, pl in pl_range)
        return self._max_profit

    def get_max_loss(self) -> Optional[Decimal]:
        """Calculate maximum possible loss."""
        if self._max_loss is not None:
            return self._max_loss

        pl_range = self.get_profit_loss_range(num_points=200)
        if not pl_range:
            return None

        self._max_loss = min(pl for _, pl in pl_range)
        return self._max_loss

    @property
    def risk_reward_ratio(self) -> Optional[float]:
        """Risk/reward ratio (max_loss / max_profit)."""
        max_profit = self.get_max_profit()
        max_loss = self.get_max_loss()

        if max_profit is None or max_loss is None:
            return None
        if max_profit <= 0:
            return None

        return float(abs(max_loss) / max_profit)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "symbol": self.symbol,
            "spread_type": self.spread_type.value,
            "name": self.name,
            "legs": [
                {
                    "symbol": leg.symbol,
                    "option_type": leg.option_type.value,
                    "strike": float(leg.strike),
                    "expiration": leg.expiration.isoformat(),
                    "side": leg.side.value,
                    "quantity": leg.quantity,
                    "contract_symbol": leg.contract_symbol,
                }
                for leg in self.legs
            ],
            "net_cost": float(self.net_cost) if self.net_cost else None,
            "net_delta": self.net_delta,
            "net_gamma": self.net_gamma,
            "net_theta": self.net_theta,
            "net_vega": self.net_vega,
            "days_to_expiration": self.days_to_expiration,
            "max_profit": float(self.get_max_profit()) if self.get_max_profit() else None,
            "max_loss": float(self.get_max_loss()) if self.get_max_loss() else None,
            "breakeven_points": [float(b) for b in self.get_breakeven_points()],
            "risk_reward_ratio": self.risk_reward_ratio,
        }


class OptionsHelper:
    """
    Helper class for building and analyzing options spreads.

    Provides factory methods for common spread types and utilities
    for Greeks aggregation, P&L analysis, and position management.
    """

    def __init__(self, data_client=None):
        """
        Initialize OptionsHelper.

        Args:
            data_client: Optional data client for fetching option chains and Greeks
        """
        self.data_client = data_client

    # ==================== Vertical Spreads ====================

    def create_bull_call_spread(
        self,
        symbol: str,
        lower_strike: Decimal,
        upper_strike: Decimal,
        expiration: date,
        quantity: int = 1,
    ) -> OptionsSpread:
        """
        Create a bull call spread (debit spread).
        Buy lower strike call, sell higher strike call.

        Max profit: (upper_strike - lower_strike - net_debit) * 100
        Max loss: net_debit * 100
        """
        spread = OptionsSpread(
            symbol=symbol,
            spread_type=SpreadType.BULL_CALL_SPREAD,
            name=f"{symbol} Bull Call {lower_strike}/{upper_strike}",
        )
        spread.add_leg(OptionLeg(
            symbol=symbol,
            option_type=OptionType.CALL,
            strike=lower_strike,
            expiration=expiration,
            side=OptionSide.LONG,
            quantity=quantity,
        ))
        spread.add_leg(OptionLeg(
            symbol=symbol,
            option_type=OptionType.CALL,
            strike=upper_strike,
            expiration=expiration,
            side=OptionSide.SHORT,
            quantity=quantity,
        ))
        return spread

    def create_bear_put_spread(
        self,
        symbol: str,
        lower_strike: Decimal,
        upper_strike: Decimal,
        expiration: date,
        quantity: int = 1,
    ) -> OptionsSpread:
        """
        Create a bear put spread (debit spread).
        Buy higher strike put, sell lower strike put.

        Max profit: (upper_strike - lower_strike - net_debit) * 100
        Max loss: net_debit * 100
        """
        spread = OptionsSpread(
            symbol=symbol,
            spread_type=SpreadType.BEAR_PUT_SPREAD,
            name=f"{symbol} Bear Put {lower_strike}/{upper_strike}",
        )
        spread.add_leg(OptionLeg(
            symbol=symbol,
            option_type=OptionType.PUT,
            strike=upper_strike,
            expiration=expiration,
            side=OptionSide.LONG,
            quantity=quantity,
        ))
        spread.add_leg(OptionLeg(
            symbol=symbol,
            option_type=OptionType.PUT,
            strike=lower_strike,
            expiration=expiration,
            side=OptionSide.SHORT,
            quantity=quantity,
        ))
        return spread

    def create_bull_put_spread(
        self,
        symbol: str,
        lower_strike: Decimal,
        upper_strike: Decimal,
        expiration: date,
        quantity: int = 1,
    ) -> OptionsSpread:
        """
        Create a bull put spread (credit spread).
        Sell higher strike put, buy lower strike put.

        Max profit: net_credit * 100
        Max loss: (upper_strike - lower_strike - net_credit) * 100
        """
        spread = OptionsSpread(
            symbol=symbol,
            spread_type=SpreadType.BULL_PUT_SPREAD,
            name=f"{symbol} Bull Put {lower_strike}/{upper_strike}",
        )
        spread.add_leg(OptionLeg(
            symbol=symbol,
            option_type=OptionType.PUT,
            strike=upper_strike,
            expiration=expiration,
            side=OptionSide.SHORT,
            quantity=quantity,
        ))
        spread.add_leg(OptionLeg(
            symbol=symbol,
            option_type=OptionType.PUT,
            strike=lower_strike,
            expiration=expiration,
            side=OptionSide.LONG,
            quantity=quantity,
        ))
        return spread

    def create_bear_call_spread(
        self,
        symbol: str,
        lower_strike: Decimal,
        upper_strike: Decimal,
        expiration: date,
        quantity: int = 1,
    ) -> OptionsSpread:
        """
        Create a bear call spread (credit spread).
        Sell lower strike call, buy higher strike call.

        Max profit: net_credit * 100
        Max loss: (upper_strike - lower_strike - net_credit) * 100
        """
        spread = OptionsSpread(
            symbol=symbol,
            spread_type=SpreadType.BEAR_CALL_SPREAD,
            name=f"{symbol} Bear Call {lower_strike}/{upper_strike}",
        )
        spread.add_leg(OptionLeg(
            symbol=symbol,
            option_type=OptionType.CALL,
            strike=lower_strike,
            expiration=expiration,
            side=OptionSide.SHORT,
            quantity=quantity,
        ))
        spread.add_leg(OptionLeg(
            symbol=symbol,
            option_type=OptionType.CALL,
            strike=upper_strike,
            expiration=expiration,
            side=OptionSide.LONG,
            quantity=quantity,
        ))
        return spread

    # ==================== Straddles & Strangles ====================

    def create_long_straddle(
        self,
        symbol: str,
        strike: Decimal,
        expiration: date,
        quantity: int = 1,
    ) -> OptionsSpread:
        """
        Create a long straddle.
        Buy call and put at same strike.

        Profits from large move in either direction.
        Max loss: net_debit * 100
        """
        spread = OptionsSpread(
            symbol=symbol,
            spread_type=SpreadType.LONG_STRADDLE,
            name=f"{symbol} Long Straddle {strike}",
        )
        spread.add_leg(OptionLeg(
            symbol=symbol,
            option_type=OptionType.CALL,
            strike=strike,
            expiration=expiration,
            side=OptionSide.LONG,
            quantity=quantity,
        ))
        spread.add_leg(OptionLeg(
            symbol=symbol,
            option_type=OptionType.PUT,
            strike=strike,
            expiration=expiration,
            side=OptionSide.LONG,
            quantity=quantity,
        ))
        return spread

    def create_short_straddle(
        self,
        symbol: str,
        strike: Decimal,
        expiration: date,
        quantity: int = 1,
    ) -> OptionsSpread:
        """
        Create a short straddle.
        Sell call and put at same strike.

        Profits from low volatility / no movement.
        Max profit: net_credit * 100
        Max loss: Unlimited
        """
        spread = OptionsSpread(
            symbol=symbol,
            spread_type=SpreadType.SHORT_STRADDLE,
            name=f"{symbol} Short Straddle {strike}",
        )
        spread.add_leg(OptionLeg(
            symbol=symbol,
            option_type=OptionType.CALL,
            strike=strike,
            expiration=expiration,
            side=OptionSide.SHORT,
            quantity=quantity,
        ))
        spread.add_leg(OptionLeg(
            symbol=symbol,
            option_type=OptionType.PUT,
            strike=strike,
            expiration=expiration,
            side=OptionSide.SHORT,
            quantity=quantity,
        ))
        return spread

    def create_long_strangle(
        self,
        symbol: str,
        put_strike: Decimal,
        call_strike: Decimal,
        expiration: date,
        quantity: int = 1,
    ) -> OptionsSpread:
        """
        Create a long strangle.
        Buy OTM call and OTM put.

        Cheaper than straddle, needs larger move.
        Max loss: net_debit * 100
        """
        spread = OptionsSpread(
            symbol=symbol,
            spread_type=SpreadType.LONG_STRANGLE,
            name=f"{symbol} Long Strangle {put_strike}/{call_strike}",
        )
        spread.add_leg(OptionLeg(
            symbol=symbol,
            option_type=OptionType.CALL,
            strike=call_strike,
            expiration=expiration,
            side=OptionSide.LONG,
            quantity=quantity,
        ))
        spread.add_leg(OptionLeg(
            symbol=symbol,
            option_type=OptionType.PUT,
            strike=put_strike,
            expiration=expiration,
            side=OptionSide.LONG,
            quantity=quantity,
        ))
        return spread

    def create_short_strangle(
        self,
        symbol: str,
        put_strike: Decimal,
        call_strike: Decimal,
        expiration: date,
        quantity: int = 1,
    ) -> OptionsSpread:
        """
        Create a short strangle.
        Sell OTM call and OTM put.

        Profits from low volatility.
        Max profit: net_credit * 100
        Max loss: Unlimited
        """
        spread = OptionsSpread(
            symbol=symbol,
            spread_type=SpreadType.SHORT_STRANGLE,
            name=f"{symbol} Short Strangle {put_strike}/{call_strike}",
        )
        spread.add_leg(OptionLeg(
            symbol=symbol,
            option_type=OptionType.CALL,
            strike=call_strike,
            expiration=expiration,
            side=OptionSide.SHORT,
            quantity=quantity,
        ))
        spread.add_leg(OptionLeg(
            symbol=symbol,
            option_type=OptionType.PUT,
            strike=put_strike,
            expiration=expiration,
            side=OptionSide.SHORT,
            quantity=quantity,
        ))
        return spread

    # ==================== Iron Spreads ====================

    def create_iron_condor(
        self,
        symbol: str,
        put_lower: Decimal,
        put_upper: Decimal,
        call_lower: Decimal,
        call_upper: Decimal,
        expiration: date,
        quantity: int = 1,
    ) -> OptionsSpread:
        """
        Create an iron condor (credit spread).
        Bull put spread + Bear call spread.

        Profits from low volatility within range.
        Max profit: net_credit * 100
        Max loss: (width - net_credit) * 100
        """
        spread = OptionsSpread(
            symbol=symbol,
            spread_type=SpreadType.IRON_CONDOR,
            name=f"{symbol} Iron Condor {put_lower}/{put_upper}/{call_lower}/{call_upper}",
        )
        # Bull put spread (lower side)
        spread.add_leg(OptionLeg(
            symbol=symbol,
            option_type=OptionType.PUT,
            strike=put_lower,
            expiration=expiration,
            side=OptionSide.LONG,
            quantity=quantity,
        ))
        spread.add_leg(OptionLeg(
            symbol=symbol,
            option_type=OptionType.PUT,
            strike=put_upper,
            expiration=expiration,
            side=OptionSide.SHORT,
            quantity=quantity,
        ))
        # Bear call spread (upper side)
        spread.add_leg(OptionLeg(
            symbol=symbol,
            option_type=OptionType.CALL,
            strike=call_lower,
            expiration=expiration,
            side=OptionSide.SHORT,
            quantity=quantity,
        ))
        spread.add_leg(OptionLeg(
            symbol=symbol,
            option_type=OptionType.CALL,
            strike=call_upper,
            expiration=expiration,
            side=OptionSide.LONG,
            quantity=quantity,
        ))
        return spread

    def create_iron_butterfly(
        self,
        symbol: str,
        lower_strike: Decimal,
        middle_strike: Decimal,
        upper_strike: Decimal,
        expiration: date,
        quantity: int = 1,
    ) -> OptionsSpread:
        """
        Create an iron butterfly (credit spread).
        Short straddle + Long strangle protection.

        Max profit at middle strike.
        Max profit: net_credit * 100
        Max loss: (wing_width - net_credit) * 100
        """
        spread = OptionsSpread(
            symbol=symbol,
            spread_type=SpreadType.IRON_BUTTERFLY,
            name=f"{symbol} Iron Butterfly {lower_strike}/{middle_strike}/{upper_strike}",
        )
        # Long put wing
        spread.add_leg(OptionLeg(
            symbol=symbol,
            option_type=OptionType.PUT,
            strike=lower_strike,
            expiration=expiration,
            side=OptionSide.LONG,
            quantity=quantity,
        ))
        # Short put body
        spread.add_leg(OptionLeg(
            symbol=symbol,
            option_type=OptionType.PUT,
            strike=middle_strike,
            expiration=expiration,
            side=OptionSide.SHORT,
            quantity=quantity,
        ))
        # Short call body
        spread.add_leg(OptionLeg(
            symbol=symbol,
            option_type=OptionType.CALL,
            strike=middle_strike,
            expiration=expiration,
            side=OptionSide.SHORT,
            quantity=quantity,
        ))
        # Long call wing
        spread.add_leg(OptionLeg(
            symbol=symbol,
            option_type=OptionType.CALL,
            strike=upper_strike,
            expiration=expiration,
            side=OptionSide.LONG,
            quantity=quantity,
        ))
        return spread

    # ==================== Butterflies ====================

    def create_call_butterfly(
        self,
        symbol: str,
        lower_strike: Decimal,
        middle_strike: Decimal,
        upper_strike: Decimal,
        expiration: date,
        quantity: int = 1,
    ) -> OptionsSpread:
        """
        Create a call butterfly (debit spread).
        Buy 1 lower call, sell 2 middle calls, buy 1 upper call.

        Max profit at middle strike.
        """
        spread = OptionsSpread(
            symbol=symbol,
            spread_type=SpreadType.BUTTERFLY,
            name=f"{symbol} Call Butterfly {lower_strike}/{middle_strike}/{upper_strike}",
        )
        spread.add_leg(OptionLeg(
            symbol=symbol,
            option_type=OptionType.CALL,
            strike=lower_strike,
            expiration=expiration,
            side=OptionSide.LONG,
            quantity=quantity,
        ))
        spread.add_leg(OptionLeg(
            symbol=symbol,
            option_type=OptionType.CALL,
            strike=middle_strike,
            expiration=expiration,
            side=OptionSide.SHORT,
            quantity=quantity * 2,
        ))
        spread.add_leg(OptionLeg(
            symbol=symbol,
            option_type=OptionType.CALL,
            strike=upper_strike,
            expiration=expiration,
            side=OptionSide.LONG,
            quantity=quantity,
        ))
        return spread

    def create_put_butterfly(
        self,
        symbol: str,
        lower_strike: Decimal,
        middle_strike: Decimal,
        upper_strike: Decimal,
        expiration: date,
        quantity: int = 1,
    ) -> OptionsSpread:
        """
        Create a put butterfly (debit spread).
        Buy 1 lower put, sell 2 middle puts, buy 1 upper put.

        Max profit at middle strike.
        """
        spread = OptionsSpread(
            symbol=symbol,
            spread_type=SpreadType.BUTTERFLY,
            name=f"{symbol} Put Butterfly {lower_strike}/{middle_strike}/{upper_strike}",
        )
        spread.add_leg(OptionLeg(
            symbol=symbol,
            option_type=OptionType.PUT,
            strike=lower_strike,
            expiration=expiration,
            side=OptionSide.LONG,
            quantity=quantity,
        ))
        spread.add_leg(OptionLeg(
            symbol=symbol,
            option_type=OptionType.PUT,
            strike=middle_strike,
            expiration=expiration,
            side=OptionSide.SHORT,
            quantity=quantity * 2,
        ))
        spread.add_leg(OptionLeg(
            symbol=symbol,
            option_type=OptionType.PUT,
            strike=upper_strike,
            expiration=expiration,
            side=OptionSide.LONG,
            quantity=quantity,
        ))
        return spread

    # ==================== Calendar & Diagonal Spreads ====================

    def create_calendar_spread(
        self,
        symbol: str,
        strike: Decimal,
        near_expiration: date,
        far_expiration: date,
        option_type: OptionType = OptionType.CALL,
        quantity: int = 1,
    ) -> OptionsSpread:
        """
        Create a calendar spread (debit spread).
        Sell near-term, buy far-term at same strike.

        Profits from time decay differential.
        """
        spread = OptionsSpread(
            symbol=symbol,
            spread_type=SpreadType.CALENDAR_SPREAD,
            name=f"{symbol} Calendar {strike} {option_type.value}",
        )
        # Sell near-term
        spread.add_leg(OptionLeg(
            symbol=symbol,
            option_type=option_type,
            strike=strike,
            expiration=near_expiration,
            side=OptionSide.SHORT,
            quantity=quantity,
        ))
        # Buy far-term
        spread.add_leg(OptionLeg(
            symbol=symbol,
            option_type=option_type,
            strike=strike,
            expiration=far_expiration,
            side=OptionSide.LONG,
            quantity=quantity,
        ))
        return spread

    def create_diagonal_spread(
        self,
        symbol: str,
        near_strike: Decimal,
        far_strike: Decimal,
        near_expiration: date,
        far_expiration: date,
        option_type: OptionType = OptionType.CALL,
        quantity: int = 1,
    ) -> OptionsSpread:
        """
        Create a diagonal spread.
        Sell near-term at one strike, buy far-term at different strike.

        Combines calendar and vertical spread characteristics.
        """
        spread = OptionsSpread(
            symbol=symbol,
            spread_type=SpreadType.DIAGONAL_SPREAD,
            name=f"{symbol} Diagonal {near_strike}/{far_strike} {option_type.value}",
        )
        # Sell near-term
        spread.add_leg(OptionLeg(
            symbol=symbol,
            option_type=option_type,
            strike=near_strike,
            expiration=near_expiration,
            side=OptionSide.SHORT,
            quantity=quantity,
        ))
        # Buy far-term
        spread.add_leg(OptionLeg(
            symbol=symbol,
            option_type=option_type,
            strike=far_strike,
            expiration=far_expiration,
            side=OptionSide.LONG,
            quantity=quantity,
        ))
        return spread

    # ==================== Utilities ====================

    def create_custom_spread(
        self,
        symbol: str,
        legs: List[Dict[str, Any]],
        name: Optional[str] = None,
    ) -> OptionsSpread:
        """
        Create a custom spread from a list of leg definitions.

        Args:
            symbol: Underlying symbol
            legs: List of dicts with keys: option_type, strike, expiration, side, quantity
            name: Optional name for the spread

        Returns:
            OptionsSpread object
        """
        spread = OptionsSpread(
            symbol=symbol,
            spread_type=SpreadType.CUSTOM,
            name=name or f"{symbol} Custom Spread",
        )
        for leg_def in legs:
            spread.add_leg(OptionLeg(
                symbol=symbol,
                option_type=leg_def.get("option_type", OptionType.CALL),
                strike=Decimal(str(leg_def["strike"])),
                expiration=leg_def["expiration"],
                side=leg_def.get("side", OptionSide.LONG),
                quantity=leg_def.get("quantity", 1),
            ))
        return spread

    async def populate_greeks(self, spread: OptionsSpread) -> OptionsSpread:
        """
        Populate Greeks for all legs using data client.

        Args:
            spread: Spread to populate

        Returns:
            Spread with Greeks populated
        """
        if self.data_client is None:
            logger.warning("No data client configured, cannot populate Greeks")
            return spread

        for leg in spread.legs:
            try:
                # Get option chain data for this contract
                contract_data = await self.data_client.get_option_quote(
                    leg.contract_symbol
                )
                if contract_data:
                    leg.bid = Decimal(str(contract_data.get("bid", 0)))
                    leg.ask = Decimal(str(contract_data.get("ask", 0)))
                    leg.last = Decimal(str(contract_data.get("last", 0)))
                    leg.mid = (leg.bid + leg.ask) / 2 if leg.bid and leg.ask else leg.last
                    leg.delta = contract_data.get("delta")
                    leg.gamma = contract_data.get("gamma")
                    leg.theta = contract_data.get("theta")
                    leg.vega = contract_data.get("vega")
                    leg.implied_volatility = contract_data.get("implied_volatility")
                    leg.open_interest = contract_data.get("open_interest")
                    leg.volume = contract_data.get("volume")
            except Exception as e:
                logger.error(f"Error fetching Greeks for {leg.contract_symbol}: {e}")

        # Recalculate aggregate Greeks
        spread._invalidate_cache()
        return spread

    def find_strikes_by_delta(
        self,
        option_chain: List[Dict],
        target_delta: float,
        option_type: OptionType,
        tolerance: float = 0.05,
    ) -> List[Decimal]:
        """
        Find strikes with delta closest to target.

        Args:
            option_chain: List of option data dicts
            target_delta: Target delta (e.g., 0.30 for 30 delta)
            option_type: CALL or PUT
            tolerance: Delta tolerance for matching

        Returns:
            List of matching strikes sorted by delta proximity
        """
        matches = []
        for opt in option_chain:
            if opt.get("option_type") != option_type.value:
                continue
            delta = abs(opt.get("delta", 0))
            if abs(delta - target_delta) <= tolerance:
                matches.append((Decimal(str(opt["strike"])), delta))

        # Sort by proximity to target delta
        matches.sort(key=lambda x: abs(x[1] - target_delta))
        return [strike for strike, _ in matches]

    def suggest_strikes_for_spread(
        self,
        underlying_price: Decimal,
        spread_type: SpreadType,
        width: Decimal = Decimal("5"),
    ) -> Dict[str, Decimal]:
        """
        Suggest strike prices for a given spread type based on underlying price.

        Args:
            underlying_price: Current price of underlying
            spread_type: Type of spread
            width: Width between strikes (for verticals)

        Returns:
            Dict of suggested strikes
        """
        # Round to nearest 5
        atm_strike = round(underlying_price / 5) * 5

        if spread_type in (SpreadType.BULL_CALL_SPREAD, SpreadType.BEAR_PUT_SPREAD):
            return {
                "lower_strike": atm_strike - width,
                "upper_strike": atm_strike,
            }
        elif spread_type in (SpreadType.BEAR_CALL_SPREAD, SpreadType.BULL_PUT_SPREAD):
            return {
                "lower_strike": atm_strike,
                "upper_strike": atm_strike + width,
            }
        elif spread_type in (SpreadType.LONG_STRADDLE, SpreadType.SHORT_STRADDLE):
            return {"strike": atm_strike}
        elif spread_type in (SpreadType.LONG_STRANGLE, SpreadType.SHORT_STRANGLE):
            return {
                "put_strike": atm_strike - width,
                "call_strike": atm_strike + width,
            }
        elif spread_type == SpreadType.IRON_CONDOR:
            return {
                "put_lower": atm_strike - width * 3,
                "put_upper": atm_strike - width,
                "call_lower": atm_strike + width,
                "call_upper": atm_strike + width * 3,
            }
        elif spread_type == SpreadType.IRON_BUTTERFLY:
            return {
                "lower_strike": atm_strike - width * 2,
                "middle_strike": atm_strike,
                "upper_strike": atm_strike + width * 2,
            }
        elif spread_type == SpreadType.BUTTERFLY:
            return {
                "lower_strike": atm_strike - width,
                "middle_strike": atm_strike,
                "upper_strike": atm_strike + width,
            }
        else:
            return {"strike": atm_strike}
