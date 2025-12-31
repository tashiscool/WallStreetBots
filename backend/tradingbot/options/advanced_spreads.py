"""
Advanced Option Spreads

Extended spread types beyond basic spreads:
- Box Spread (arbitrage)
- Jade Lizard
- Poor Man's Covered Call (PMCC)
- Collar
- Protective Put
- Christmas Tree
- Double Diagonal
- Zebra (Zero Extrinsic Back Ratio)
- Broken Wing Condor
"""

from dataclasses import dataclass, field
from datetime import date, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import logging

from .exotic_spreads import (
    OptionSpread,
    SpreadType,
    SpreadDirection,
    SpreadLeg,
    LegType,
    SpreadGreeks,
)

logger = logging.getLogger(__name__)


class AdvancedSpreadType(Enum):
    """Advanced spread types."""
    BOX_SPREAD = "box_spread"
    JADE_LIZARD = "jade_lizard"
    POOR_MANS_COVERED_CALL = "pmcc"
    COLLAR = "collar"
    PROTECTIVE_PUT = "protective_put"
    CHRISTMAS_TREE = "christmas_tree"
    DOUBLE_DIAGONAL = "double_diagonal"
    ZEBRA = "zebra"
    BROKEN_WING_CONDOR = "broken_wing_condor"
    SYNTHETIC_LONG = "synthetic_long"
    SYNTHETIC_SHORT = "synthetic_short"
    CONVERSION = "conversion"
    REVERSAL = "reversal"


@dataclass
class BoxSpread(OptionSpread):
    """
    Box Spread: Arbitrage strategy combining bull call spread + bear put spread.

    Structure:
    - Buy call at lower strike, sell call at higher strike (bull call)
    - Buy put at higher strike, sell put at lower strike (bear put)

    Profit = Strike difference - premium paid (should equal risk-free rate)
    Used for arbitrage or financing.
    """
    lower_strike: Decimal = Decimal("0")
    upper_strike: Decimal = Decimal("0")
    expiry: Optional[date] = None

    def __post_init__(self):
        self.spread_type = SpreadType.IRON_CONDOR  # Closest enum
        self.direction = SpreadDirection.NEUTRAL

    @property
    def box_value(self) -> Decimal:
        """Theoretical value at expiration."""
        return (self.upper_strike - self.lower_strike) * 100

    def get_max_profit(self) -> Decimal:
        """Profit = box value - premium paid."""
        return self.box_value - abs(self.net_premium) * 100

    def get_max_loss(self) -> Decimal:
        """Loss if box doesn't converge (should be ~0 for true arbitrage)."""
        return abs(self.net_premium) * 100


@dataclass
class JadeLizard(OptionSpread):
    """
    Jade Lizard: Short strangle with embedded protection.

    Structure:
    - Sell OTM put (naked)
    - Sell OTM call spread (short call + long higher call)

    No upside risk beyond call spread width.
    Ideal when: Bullish to neutral, want premium, limit call side risk.
    """
    put_strike: Decimal = Decimal("0")
    call_short_strike: Decimal = Decimal("0")
    call_long_strike: Decimal = Decimal("0")
    expiry: Optional[date] = None

    def __post_init__(self):
        self.spread_type = SpreadType.STRANGLE  # Closest
        self.direction = SpreadDirection.NEUTRAL

    @property
    def call_spread_width(self) -> Decimal:
        """Width of the embedded call spread."""
        return self.call_long_strike - self.call_short_strike

    def get_max_profit(self) -> Decimal:
        """Max profit = net premium received."""
        return self.net_premium * 100

    def get_max_loss(self) -> Decimal:
        """Max loss on upside = call spread width - premium."""
        upside_risk = (self.call_spread_width * 100) - self.net_premium * 100
        # Downside is unlimited (naked put)
        return max(upside_risk, Decimal("999999"))

    def get_breakeven_points(self) -> List[Decimal]:
        """Lower breakeven (put side)."""
        lower_be = self.put_strike - self.net_premium
        return [lower_be]


@dataclass
class PoorMansCoveredCall(OptionSpread):
    """
    Poor Man's Covered Call (PMCC): LEAPS call + short near-term call.

    Structure:
    - Buy deep ITM LEAPS call (acts like stock)
    - Sell OTM near-term call (covered by LEAPS)

    Lower capital requirement than covered call with stock.
    Key: LEAPS delta should be >0.70 (deep ITM).
    """
    leaps_strike: Decimal = Decimal("0")
    leaps_expiry: Optional[date] = None
    short_strike: Decimal = Decimal("0")
    short_expiry: Optional[date] = None

    def __post_init__(self):
        self.spread_type = SpreadType.DIAGONAL
        self.direction = SpreadDirection.BULLISH

    @property
    def width(self) -> Decimal:
        """Effective covered call width."""
        return self.short_strike - self.leaps_strike

    def get_max_profit(self) -> Decimal:
        """Max profit = width - net debit (if assigned)."""
        return (self.width * 100) - abs(self.net_premium) * 100

    def get_max_loss(self) -> Decimal:
        """Max loss = debit paid for LEAPS (minus any premiums collected)."""
        return abs(self.net_premium) * 100


@dataclass
class Collar(OptionSpread):
    """
    Collar: Stock + protective put + covered call.

    Structure:
    - Long stock (100 shares)
    - Buy OTM put (protection)
    - Sell OTM call (finances the put)

    Zero-cost collar: Premium from call = cost of put.
    """
    stock_price: Decimal = Decimal("0")
    put_strike: Decimal = Decimal("0")
    call_strike: Decimal = Decimal("0")
    expiry: Optional[date] = None
    shares: int = 100

    def __post_init__(self):
        self.spread_type = SpreadType.VERTICAL_PUT  # Closest
        self.direction = SpreadDirection.NEUTRAL

    @property
    def downside_protection(self) -> Decimal:
        """Maximum loss per share with protection."""
        return self.stock_price - self.put_strike

    @property
    def upside_cap(self) -> Decimal:
        """Maximum gain per share."""
        return self.call_strike - self.stock_price

    def get_max_profit(self) -> Decimal:
        """Max profit if called away."""
        return self.upside_cap * self.shares + self.net_premium * 100

    def get_max_loss(self) -> Decimal:
        """Max loss with put protection."""
        return self.downside_protection * self.shares - self.net_premium * 100


@dataclass
class ProtectivePut(OptionSpread):
    """
    Protective Put: Stock + long put for downside protection.

    Structure:
    - Long stock (100 shares)
    - Buy put (insurance)

    Also called "married put" or portfolio insurance.
    """
    stock_price: Decimal = Decimal("0")
    put_strike: Decimal = Decimal("0")
    expiry: Optional[date] = None
    shares: int = 100

    def __post_init__(self):
        self.spread_type = SpreadType.VERTICAL_PUT  # Closest
        self.direction = SpreadDirection.BULLISH

    def get_max_loss(self) -> Decimal:
        """Max loss = stock to strike + premium paid."""
        loss_to_strike = (self.stock_price - self.put_strike) * self.shares
        return loss_to_strike + abs(self.net_premium) * 100

    def get_max_profit(self) -> Optional[Decimal]:
        """Unlimited upside potential (minus premium)."""
        return None  # Unlimited


@dataclass
class ChristmasTree(OptionSpread):
    """
    Christmas Tree: Unbalanced butterfly with more OTM wings.

    Call Christmas Tree:
    - Buy 1 ATM call
    - Sell 1 OTM call (first level)
    - Sell 1 OTM call (second level, further out)

    Profits from moderate upward move, cheaper than butterfly.
    """
    lower_strike: Decimal = Decimal("0")
    middle_strike: Decimal = Decimal("0")
    upper_strike: Decimal = Decimal("0")
    option_type: str = "call"
    expiry: Optional[date] = None

    def __post_init__(self):
        self.spread_type = SpreadType.BROKEN_WING_BUTTERFLY
        self.direction = SpreadDirection.BULLISH if self.option_type == "call" else SpreadDirection.BEARISH

    def get_max_profit(self) -> Decimal:
        """Max profit at first short strike."""
        return (self.middle_strike - self.lower_strike) * 100 - abs(self.net_premium) * 100

    def get_max_loss(self) -> Decimal:
        """Max loss = debit paid."""
        return abs(self.net_premium) * 100


@dataclass
class DoubleDiagonal(OptionSpread):
    """
    Double Diagonal: Calendar spread on both sides.

    Structure:
    - Sell near-term OTM put + OTM call
    - Buy far-term OTM put + OTM call (same or different strikes)

    Profits from time decay and moderate price movement.
    """
    near_put_strike: Decimal = Decimal("0")
    near_call_strike: Decimal = Decimal("0")
    near_expiry: Optional[date] = None
    far_put_strike: Decimal = Decimal("0")
    far_call_strike: Decimal = Decimal("0")
    far_expiry: Optional[date] = None

    def __post_init__(self):
        self.spread_type = SpreadType.DIAGONAL
        self.direction = SpreadDirection.NEUTRAL

    @property
    def put_width(self) -> Decimal:
        """Width of put diagonal."""
        return abs(self.far_put_strike - self.near_put_strike)

    @property
    def call_width(self) -> Decimal:
        """Width of call diagonal."""
        return abs(self.far_call_strike - self.near_call_strike)


@dataclass
class Zebra(OptionSpread):
    """
    ZEBRA (Zero Extrinsic Back Ratio): Synthetic stock position.

    Structure (call zebra):
    - Buy 2 ATM calls
    - Sell 1 ITM call

    Creates ~100 delta position with less capital than stock.
    Zero extrinsic value at entry if done correctly.
    """
    atm_strike: Decimal = Decimal("0")
    itm_strike: Decimal = Decimal("0")
    expiry: Optional[date] = None
    option_type: str = "call"
    ratio: Tuple[int, int] = (2, 1)  # (long, short)

    def __post_init__(self):
        self.spread_type = SpreadType.RATIO_SPREAD
        self.direction = SpreadDirection.BULLISH if self.option_type == "call" else SpreadDirection.BEARISH

    @property
    def effective_delta(self) -> float:
        """Should be approximately 100 (like owning stock)."""
        return 100.0  # By design


@dataclass
class BrokenWingCondor(OptionSpread):
    """
    Broken Wing Iron Condor: Unbalanced condor with directional bias.

    Structure:
    - Standard iron condor but one wing is wider
    - Creates directional bias while collecting premium

    Use when you want condor income but have directional view.
    """
    put_long_strike: Decimal = Decimal("0")
    put_short_strike: Decimal = Decimal("0")
    call_short_strike: Decimal = Decimal("0")
    call_long_strike: Decimal = Decimal("0")
    expiry: Optional[date] = None

    def __post_init__(self):
        self.spread_type = SpreadType.IRON_CONDOR
        # Direction based on which wing is wider
        put_wing = self.put_short_strike - self.put_long_strike
        call_wing = self.call_long_strike - self.call_short_strike
        if put_wing > call_wing:
            self.direction = SpreadDirection.BULLISH
        elif call_wing > put_wing:
            self.direction = SpreadDirection.BEARISH
        else:
            self.direction = SpreadDirection.NEUTRAL

    @property
    def put_wing_width(self) -> Decimal:
        """Width of put spread."""
        return self.put_short_strike - self.put_long_strike

    @property
    def call_wing_width(self) -> Decimal:
        """Width of call spread."""
        return self.call_long_strike - self.call_short_strike


@dataclass
class SyntheticLong(OptionSpread):
    """
    Synthetic Long Stock: Long call + short put at same strike.

    Replicates stock ownership without owning shares.
    Useful for leveraged exposure or hard-to-borrow stocks.
    """
    strike: Decimal = Decimal("0")
    expiry: Optional[date] = None

    def __post_init__(self):
        self.spread_type = SpreadType.STRADDLE  # Closest
        self.direction = SpreadDirection.BULLISH

    def get_breakeven_points(self) -> List[Decimal]:
        """Breakeven = strike +/- net premium."""
        return [self.strike + self.net_premium]


@dataclass
class SyntheticShort(OptionSpread):
    """
    Synthetic Short Stock: Short call + long put at same strike.

    Replicates short stock without borrowing shares.
    """
    strike: Decimal = Decimal("0")
    expiry: Optional[date] = None

    def __post_init__(self):
        self.spread_type = SpreadType.STRADDLE  # Closest
        self.direction = SpreadDirection.BEARISH

    def get_breakeven_points(self) -> List[Decimal]:
        """Breakeven = strike +/- net premium."""
        return [self.strike - self.net_premium]


class SpreadRollType(Enum):
    """Types of spread roll adjustments."""
    ROLL_UP = "roll_up"
    ROLL_DOWN = "roll_down"
    ROLL_OUT = "roll_out"  # Same strikes, later expiry
    ROLL_UP_AND_OUT = "roll_up_and_out"
    ROLL_DOWN_AND_OUT = "roll_down_and_out"
    ROLL_DIAGONAL = "roll_diagonal"


@dataclass
class RollRecommendation:
    """Recommendation for rolling a spread."""
    roll_type: SpreadRollType
    new_strikes: Dict[str, Decimal]
    new_expiry: date
    estimated_credit_debit: Decimal  # Positive = credit
    reason: str
    urgency: str  # "low", "medium", "high"
    profit_potential: Decimal
    risk_change: str


class SpreadAdjustmentEngine:
    """
    Engine for managing spread adjustments and rolls.

    Provides automated adjustment recommendations based on:
    - Delta/Greek changes
    - Time to expiration
    - Profit/loss thresholds
    - Market conditions
    """

    # Thresholds for adjustment triggers
    PROFIT_TARGET_PCT = 0.50  # Close at 50% profit
    LOSS_TRIGGER_PCT = 2.0  # Adjust at 2x credit
    DTE_ROLL_TRIGGER = 21  # Roll when DTE <= 21
    DELTA_ADJUSTMENT_THRESHOLD = 0.30  # Adjust when delta moves beyond

    def __init__(
        self,
        profit_target: float = 0.50,
        loss_trigger: float = 2.0,
        dte_roll_trigger: int = 21,
    ):
        """
        Initialize adjustment engine.

        Args:
            profit_target: Take profit at this % of max profit
            loss_trigger: Adjust when loss reaches this multiple of credit
            dte_roll_trigger: Roll when DTE falls below this
        """
        self.profit_target = profit_target
        self.loss_trigger = loss_trigger
        self.dte_roll_trigger = dte_roll_trigger

    def check_adjustment_needed(
        self,
        spread: OptionSpread,
        current_price: Decimal,
        current_pnl: Decimal,
    ) -> Tuple[bool, List[str]]:
        """
        Check if a spread needs adjustment.

        Args:
            spread: The spread to check
            current_price: Current underlying price
            current_pnl: Current P&L of position

        Returns:
            Tuple of (needs_adjustment, reasons)
        """
        reasons = []
        needs_adjustment = False

        # Check profit target
        max_profit = spread.get_max_profit()
        if max_profit and current_pnl >= max_profit * Decimal(str(self.profit_target)):
            reasons.append(f"Profit target reached ({self.profit_target * 100:.0f}% of max)")
            needs_adjustment = True

        # Check loss trigger
        if spread.is_credit:
            credit = spread.net_premium * 100
            if current_pnl < -credit * Decimal(str(self.loss_trigger)):
                reasons.append(f"Loss exceeds {self.loss_trigger}x credit received")
                needs_adjustment = True

        # Check DTE
        if hasattr(spread, 'expiry') and spread.expiry:
            dte = (spread.expiry - date.today()).days
            if dte <= self.dte_roll_trigger:
                reasons.append(f"DTE ({dte}) below roll trigger ({self.dte_roll_trigger})")
                needs_adjustment = True

        # Check delta (for spreads with greeks)
        greeks = spread.aggregate_greeks
        if abs(float(greeks.delta)) > self.DELTA_ADJUSTMENT_THRESHOLD * 100:
            reasons.append(f"Position delta ({float(greeks.delta):.0f}) exceeds threshold")
            needs_adjustment = True

        return needs_adjustment, reasons

    def recommend_iron_condor_adjustment(
        self,
        spread,  # IronCondor
        current_price: Decimal,
        current_pnl: Decimal,
    ) -> Optional[RollRecommendation]:
        """
        Recommend adjustment for iron condor.

        Strategies:
        1. Price threatening short put: Roll down
        2. Price threatening short call: Roll up
        3. Low DTE with profit: Close
        4. Low DTE underwater: Roll out
        """
        if not hasattr(spread, 'put_short_strike'):
            return None

        dte = (spread.expiry - date.today()).days if spread.expiry else 999

        # Price near short put (bearish threat)
        if current_price <= spread.put_short_strike * Decimal("1.02"):
            # Roll put side down
            new_put_short = spread.put_short_strike - Decimal("5")
            new_put_long = spread.put_long_strike - Decimal("5")

            return RollRecommendation(
                roll_type=SpreadRollType.ROLL_DOWN,
                new_strikes={
                    "put_short": new_put_short,
                    "put_long": new_put_long,
                },
                new_expiry=spread.expiry,
                estimated_credit_debit=Decimal("-0.50"),  # Usually debit to roll down
                reason="Price approaching short put strike",
                urgency="high",
                profit_potential=Decimal("0"),
                risk_change="increased_downside_protection",
            )

        # Price near short call (bullish threat)
        if current_price >= spread.call_short_strike * Decimal("0.98"):
            # Roll call side up
            new_call_short = spread.call_short_strike + Decimal("5")
            new_call_long = spread.call_long_strike + Decimal("5")

            return RollRecommendation(
                roll_type=SpreadRollType.ROLL_UP,
                new_strikes={
                    "call_short": new_call_short,
                    "call_long": new_call_long,
                },
                new_expiry=spread.expiry,
                estimated_credit_debit=Decimal("-0.30"),
                reason="Price approaching short call strike",
                urgency="high",
                profit_potential=Decimal("0"),
                risk_change="increased_upside_protection",
            )

        # Low DTE - roll out
        if dte <= self.dte_roll_trigger and current_pnl < spread.get_max_profit() * Decimal("0.5"):
            new_expiry = date.today() + timedelta(days=45)
            while new_expiry.weekday() != 4:
                new_expiry += timedelta(days=1)

            return RollRecommendation(
                roll_type=SpreadRollType.ROLL_OUT,
                new_strikes={
                    "put_long": spread.put_long_strike,
                    "put_short": spread.put_short_strike,
                    "call_short": spread.call_short_strike,
                    "call_long": spread.call_long_strike,
                },
                new_expiry=new_expiry,
                estimated_credit_debit=Decimal("0.30"),  # Usually credit to roll out
                reason=f"Low DTE ({dte} days) with unrealized profit",
                urgency="medium",
                profit_potential=Decimal("30"),
                risk_change="extended_time_exposure",
            )

        return None

    def recommend_calendar_adjustment(
        self,
        spread,  # CalendarSpread
        current_price: Decimal,
    ) -> Optional[RollRecommendation]:
        """
        Recommend adjustment for calendar spread.

        Strategies:
        1. Price moved away from strike: Roll strikes to follow
        2. Near expiry approaching: Roll to next cycle
        """
        if not hasattr(spread, 'strike'):
            return None

        # Price moved significantly from strike
        price_diff_pct = abs(float(current_price - spread.strike) / float(spread.strike))

        if price_diff_pct > 0.05:  # 5% move
            # Roll to follow price
            new_strike = Decimal(str(round(float(current_price), 0)))

            roll_type = SpreadRollType.ROLL_UP if current_price > spread.strike else SpreadRollType.ROLL_DOWN

            return RollRecommendation(
                roll_type=roll_type,
                new_strikes={"strike": new_strike},
                new_expiry=spread.far_expiry,
                estimated_credit_debit=Decimal("-0.20"),  # Small debit to adjust
                reason=f"Price moved {price_diff_pct * 100:.1f}% from strike",
                urgency="medium",
                profit_potential=Decimal("0"),
                risk_change="repositioned_to_atm",
            )

        return None

    def recommend_vertical_adjustment(
        self,
        spread: OptionSpread,
        current_price: Decimal,
        current_pnl: Decimal,
        is_call_spread: bool,
    ) -> Optional[RollRecommendation]:
        """
        Recommend adjustment for vertical spread (credit or debit).

        Strategies:
        1. ITM threat: Roll up/down
        2. Profit target: Close
        3. DTE low: Roll out
        """
        legs = spread.legs
        if len(legs) < 2:
            return None

        short_leg = next((l for l in legs if l.contracts < 0), None)
        if not short_leg:
            return None

        dte = (short_leg.expiry - date.today()).days if short_leg.expiry else 999

        # Short leg threatened
        if is_call_spread and current_price >= short_leg.strike * Decimal("0.95"):
            # Roll up
            return RollRecommendation(
                roll_type=SpreadRollType.ROLL_UP,
                new_strikes={
                    "short": short_leg.strike + Decimal("5"),
                    "long": short_leg.strike + Decimal("10"),
                },
                new_expiry=short_leg.expiry,
                estimated_credit_debit=Decimal("-0.40"),
                reason="Short call strike threatened",
                urgency="high",
                profit_potential=Decimal("0"),
                risk_change="moved_strikes_higher",
            )

        if not is_call_spread and current_price <= short_leg.strike * Decimal("1.05"):
            # Roll down
            return RollRecommendation(
                roll_type=SpreadRollType.ROLL_DOWN,
                new_strikes={
                    "short": short_leg.strike - Decimal("5"),
                    "long": short_leg.strike - Decimal("10"),
                },
                new_expiry=short_leg.expiry,
                estimated_credit_debit=Decimal("-0.40"),
                reason="Short put strike threatened",
                urgency="high",
                profit_potential=Decimal("0"),
                risk_change="moved_strikes_lower",
            )

        # Low DTE
        if dte <= self.dte_roll_trigger:
            new_expiry = date.today() + timedelta(days=30)
            while new_expiry.weekday() != 4:
                new_expiry += timedelta(days=1)

            return RollRecommendation(
                roll_type=SpreadRollType.ROLL_OUT,
                new_strikes={},
                new_expiry=new_expiry,
                estimated_credit_debit=Decimal("0.20"),
                reason=f"Low DTE ({dte})",
                urgency="low",
                profit_potential=Decimal("20"),
                risk_change="extended_expiry",
            )

        return None

    def get_adjustment_orders(
        self,
        spread: OptionSpread,
        recommendation: RollRecommendation,
    ) -> List[Dict[str, Any]]:
        """
        Generate orders to execute an adjustment.

        Returns list of orders to close existing and open new position.
        """
        orders = []

        # Close existing position
        for leg in spread.legs:
            orders.append({
                "action": "CLOSE",
                "symbol": f"{spread.ticker}_{leg.expiry}_{leg.strike}_{leg.option_type}",
                "quantity": -leg.contracts,  # Opposite to close
                "leg_type": leg.leg_type.value,
            })

        # Open new position (based on recommendation)
        for strike_key, strike_value in recommendation.new_strikes.items():
            # Determine leg type from key
            if "put_long" in strike_key or "long_put" in strike_key:
                leg_type = "long_put"
                quantity = 1
            elif "put_short" in strike_key or "short_put" in strike_key:
                leg_type = "short_put"
                quantity = -1
            elif "call_long" in strike_key or "long_call" in strike_key:
                leg_type = "long_call"
                quantity = 1
            elif "call_short" in strike_key or "short_call" in strike_key:
                leg_type = "short_call"
                quantity = -1
            else:
                continue

            option_type = "call" if "call" in leg_type else "put"

            orders.append({
                "action": "OPEN",
                "symbol": f"{spread.ticker}_{recommendation.new_expiry}_{strike_value}_{option_type}",
                "strike": float(strike_value),
                "expiry": recommendation.new_expiry.isoformat(),
                "quantity": quantity,
                "leg_type": leg_type,
            })

        return orders


class SpreadScanner:
    """
    Scanner for finding optimal spread opportunities.

    Scans for spreads meeting criteria:
    - Minimum probability of profit
    - Target credit/debit
    - Risk/reward ratio
    - Greeks constraints
    """

    def __init__(
        self,
        min_pop: float = 0.50,
        min_credit: Decimal = Decimal("0.30"),
        max_risk_reward: float = 3.0,
        max_vega: float = 50.0,
    ):
        """
        Initialize spread scanner.

        Args:
            min_pop: Minimum probability of profit
            min_credit: Minimum credit for credit spreads
            max_risk_reward: Maximum risk/reward ratio
            max_vega: Maximum vega exposure
        """
        self.min_pop = min_pop
        self.min_credit = min_credit
        self.max_risk_reward = max_risk_reward
        self.max_vega = max_vega

    def scan_iron_condors(
        self,
        ticker: str,
        current_price: Decimal,
        available_strikes: List[Decimal],
        available_expiries: List[date],
        iv_data: Optional[Dict[str, float]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Scan for optimal iron condor setups.

        Returns list of potential iron condors ranked by quality.
        """
        candidates = []

        for expiry in available_expiries:
            dte = (expiry - date.today()).days
            if dte < 21 or dte > 60:
                continue

            # Find strikes around current price
            strikes = sorted(available_strikes)
            atm_idx = min(
                range(len(strikes)),
                key=lambda i: abs(strikes[i] - current_price)
            )

            # Try different wing widths
            for wing_width in range(2, 6):
                put_short_idx = atm_idx - wing_width
                put_long_idx = put_short_idx - wing_width
                call_short_idx = atm_idx + wing_width
                call_long_idx = call_short_idx + wing_width

                if put_long_idx < 0 or call_long_idx >= len(strikes):
                    continue

                setup = {
                    "ticker": ticker,
                    "expiry": expiry,
                    "dte": dte,
                    "put_long": strikes[put_long_idx],
                    "put_short": strikes[put_short_idx],
                    "call_short": strikes[call_short_idx],
                    "call_long": strikes[call_long_idx],
                    "wing_width": wing_width,
                }

                # Calculate simple PoP estimate
                profit_zone = float(setup["call_short"] - setup["put_short"])
                pop = min(0.95, profit_zone / float(current_price) * 2)
                setup["estimated_pop"] = pop

                if pop >= self.min_pop:
                    candidates.append(setup)

        # Sort by PoP (highest first)
        candidates.sort(key=lambda x: x["estimated_pop"], reverse=True)

        return candidates[:10]  # Return top 10

    def scan_calendars(
        self,
        ticker: str,
        current_price: Decimal,
        available_strikes: List[Decimal],
        available_expiries: List[date],
    ) -> List[Dict[str, Any]]:
        """
        Scan for optimal calendar spread setups.

        Returns list of potential calendars ranked by quality.
        """
        candidates = []

        # Find ATM strike
        strikes = sorted(available_strikes)
        atm_strike = min(strikes, key=lambda s: abs(s - current_price))

        # Find expiry pairs
        expiries = sorted(available_expiries)

        for i, near_exp in enumerate(expiries[:-1]):
            near_dte = (near_exp - date.today()).days
            if near_dte < 14 or near_dte > 30:
                continue

            for far_exp in expiries[i + 1:]:
                far_dte = (far_exp - date.today()).days
                days_between = far_dte - near_dte

                if days_between < 21 or days_between > 60:
                    continue

                setup = {
                    "ticker": ticker,
                    "strike": atm_strike,
                    "near_expiry": near_exp,
                    "far_expiry": far_exp,
                    "near_dte": near_dte,
                    "far_dte": far_dte,
                    "days_between": days_between,
                }

                # Calendars work best with 28-35 days between expiries
                if 28 <= days_between <= 35:
                    setup["quality"] = "optimal"
                else:
                    setup["quality"] = "acceptable"

                candidates.append(setup)

        # Sort by quality
        candidates.sort(key=lambda x: (x["quality"] == "optimal", x["days_between"]), reverse=True)

        return candidates[:10]
