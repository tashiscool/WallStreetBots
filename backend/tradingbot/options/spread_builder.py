"""
Spread Builder

Factory for constructing optimal option spreads based on market conditions,
volatility, and user parameters.
"""

import logging
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal
from typing import List, Optional, Tuple

from .exotic_spreads import (
    IronCondor,
    IronButterfly,
    Butterfly,
    CalendarSpread,
    DiagonalSpread,
    Straddle,
    Strangle,
    RatioSpread,
    SpreadLeg,
    LegType,
    SpreadAnalysis,
    SpreadGreeks,
    SpreadType,
)
from .pricing_engine import (
    BlackScholesEngine,
    RealOptionsPricingEngine,
    OptionsContract,
)

logger = logging.getLogger(__name__)


@dataclass
class SpreadBuilderConfig:
    """Configuration for spread building."""
    min_dte: int = 21  # Minimum days to expiration
    max_dte: int = 45  # Maximum days to expiration
    min_premium_credit: Decimal = Decimal("0.30")  # Minimum net credit
    max_risk_reward_ratio: float = 3.0  # Maximum risk/reward
    min_probability_of_profit: float = 0.50  # Minimum PoP
    wing_width_min: int = 1  # Minimum wing width (in strikes)
    wing_width_max: int = 5  # Maximum wing width (in strikes)
    target_delta_short: float = 0.16  # Target delta for short strikes
    max_vega_exposure: Decimal = Decimal("50.0")  # Max vega per spread
    prefer_weekly_expiry: bool = True  # Prefer weekly options


class SpreadBuilder:
    """
    Builds optimal option spreads based on market data and parameters.
    """

    def __init__(
        self,
        pricing_engine: Optional[RealOptionsPricingEngine] = None,
        config: Optional[SpreadBuilderConfig] = None,
    ):
        self.pricing_engine = pricing_engine or RealOptionsPricingEngine()
        self.config = config or SpreadBuilderConfig()
        self.bs_engine = BlackScholesEngine()

    async def _get_strike_ladder(
        self,
        ticker: str,
        current_price: Decimal,
        expiry: date,
    ) -> List[Decimal]:
        """Get available strikes for a ticker and expiry."""
        try:
            chain = await self.pricing_engine.get_options_chain_yahoo(ticker, expiry)
            strikes = sorted({c.strike for c in chain})
            return strikes
        except Exception as e:
            logger.error(f"Error getting strike ladder: {e}")
            # Generate synthetic strikes
            base = round(float(current_price), 0)
            return [Decimal(str(base + i)) for i in range(-20, 21)]

    async def _find_strike_by_delta(
        self,
        ticker: str,
        current_price: Decimal,
        expiry: date,
        target_delta: float,
        option_type: str,
    ) -> Optional[Decimal]:
        """Find strike with closest delta to target."""
        try:
            chain = await self.pricing_engine.get_options_chain_yahoo(ticker, expiry)

            best_strike = None
            best_diff = float('inf')

            for contract in chain:
                if contract.option_type != option_type:
                    continue
                if contract.delta is None:
                    continue

                diff = abs(float(contract.delta) - target_delta)
                if diff < best_diff:
                    best_diff = diff
                    best_strike = contract.strike

            return best_strike

        except Exception as e:
            logger.warning(f"Error finding delta-based strike: {e}")
            # Fallback: estimate strike based on delta
            if option_type == "call":
                # OTM call: strike above current price
                offset = int((1 - target_delta) * 10)
                return Decimal(str(round(float(current_price) * (1 + offset * 0.02), 0)))
            else:
                # OTM put: strike below current price
                offset = int((1 - abs(target_delta)) * 10)
                return Decimal(str(round(float(current_price) * (1 - offset * 0.02), 0)))

    async def _get_option_premium(
        self,
        ticker: str,
        strike: Decimal,
        expiry: date,
        option_type: str,
        current_price: Decimal,
    ) -> Decimal:
        """Get theoretical premium for an option."""
        return await self.pricing_engine.calculate_theoretical_price(
            ticker, strike, expiry, option_type, current_price
        )

    async def build_iron_condor(
        self,
        ticker: str,
        current_price: Decimal,
        expiry: Optional[date] = None,
        wing_width: int = 5,
        target_short_delta: float = 0.16,
    ) -> Optional[Tuple[IronCondor, SpreadAnalysis]]:
        """
        Build an iron condor spread.

        Args:
            ticker: Stock ticker
            current_price: Current stock price
            expiry: Expiration date (or auto-select)
            wing_width: Width of each wing in dollars
            target_short_delta: Target delta for short strikes

        Returns:
            Tuple of (IronCondor, SpreadAnalysis) or None
        """
        try:
            # Select expiry if not provided
            if expiry is None:
                expiry = self._select_optimal_expiry()

            # Find short strikes by delta
            put_short = await self._find_strike_by_delta(
                ticker, current_price, expiry, -target_short_delta, "put"
            )
            call_short = await self._find_strike_by_delta(
                ticker, current_price, expiry, target_short_delta, "call"
            )

            if not put_short or not call_short:
                return None

            # Calculate wing strikes
            put_long = put_short - Decimal(str(wing_width))
            call_long = call_short + Decimal(str(wing_width))

            # Get premiums
            put_short_prem = await self._get_option_premium(
                ticker, put_short, expiry, "put", current_price
            )
            put_long_prem = await self._get_option_premium(
                ticker, put_long, expiry, "put", current_price
            )
            call_short_prem = await self._get_option_premium(
                ticker, call_short, expiry, "call", current_price
            )
            call_long_prem = await self._get_option_premium(
                ticker, call_long, expiry, "call", current_price
            )

            # Build legs
            legs = [
                SpreadLeg(LegType.LONG_PUT, put_long, expiry, 1, put_long_prem),
                SpreadLeg(LegType.SHORT_PUT, put_short, expiry, -1, put_short_prem),
                SpreadLeg(LegType.SHORT_CALL, call_short, expiry, -1, call_short_prem),
                SpreadLeg(LegType.LONG_CALL, call_long, expiry, 1, call_long_prem),
            ]

            # Create iron condor
            ic = IronCondor(
                spread_type=SpreadType.IRON_CONDOR,
                ticker=ticker,
                legs=legs,
                put_long_strike=put_long,
                put_short_strike=put_short,
                call_short_strike=call_short,
                call_long_strike=call_long,
                expiry=expiry,
            )

            # Analyze
            analysis = self._analyze_spread(ic, current_price)

            return ic, analysis

        except Exception as e:
            logger.error(f"Error building iron condor: {e}")
            return None

    async def build_iron_butterfly(
        self,
        ticker: str,
        current_price: Decimal,
        expiry: Optional[date] = None,
        wing_width: int = 5,
    ) -> Optional[Tuple[IronButterfly, SpreadAnalysis]]:
        """
        Build an iron butterfly spread.

        Centered at ATM with equal wings.
        """
        try:
            if expiry is None:
                expiry = self._select_optimal_expiry()

            # ATM strike
            center = Decimal(str(round(float(current_price), 0)))
            lower = center - Decimal(str(wing_width))
            upper = center + Decimal(str(wing_width))

            # Get premiums
            center_put_prem = await self._get_option_premium(
                ticker, center, expiry, "put", current_price
            )
            center_call_prem = await self._get_option_premium(
                ticker, center, expiry, "call", current_price
            )
            lower_put_prem = await self._get_option_premium(
                ticker, lower, expiry, "put", current_price
            )
            upper_call_prem = await self._get_option_premium(
                ticker, upper, expiry, "call", current_price
            )

            legs = [
                SpreadLeg(LegType.LONG_PUT, lower, expiry, 1, lower_put_prem),
                SpreadLeg(LegType.SHORT_PUT, center, expiry, -1, center_put_prem),
                SpreadLeg(LegType.SHORT_CALL, center, expiry, -1, center_call_prem),
                SpreadLeg(LegType.LONG_CALL, upper, expiry, 1, upper_call_prem),
            ]

            ib = IronButterfly(
                spread_type=SpreadType.IRON_BUTTERFLY,
                ticker=ticker,
                legs=legs,
                center_strike=center,
                lower_strike=lower,
                upper_strike=upper,
                expiry=expiry,
            )

            analysis = self._analyze_spread(ib, current_price)
            return ib, analysis

        except Exception as e:
            logger.error(f"Error building iron butterfly: {e}")
            return None

    async def build_straddle(
        self,
        ticker: str,
        current_price: Decimal,
        expiry: Optional[date] = None,
        is_long: bool = True,
    ) -> Optional[Tuple[Straddle, SpreadAnalysis]]:
        """
        Build a straddle (ATM call + put).

        Args:
            ticker: Stock ticker
            current_price: Current stock price
            expiry: Expiration date
            is_long: True for long straddle, False for short
        """
        try:
            if expiry is None:
                expiry = self._select_optimal_expiry()

            strike = Decimal(str(round(float(current_price), 0)))

            call_prem = await self._get_option_premium(
                ticker, strike, expiry, "call", current_price
            )
            put_prem = await self._get_option_premium(
                ticker, strike, expiry, "put", current_price
            )

            if is_long:
                legs = [
                    SpreadLeg(LegType.LONG_CALL, strike, expiry, 1, call_prem),
                    SpreadLeg(LegType.LONG_PUT, strike, expiry, 1, put_prem),
                ]
            else:
                legs = [
                    SpreadLeg(LegType.SHORT_CALL, strike, expiry, -1, call_prem),
                    SpreadLeg(LegType.SHORT_PUT, strike, expiry, -1, put_prem),
                ]

            straddle = Straddle(
                spread_type=SpreadType.STRADDLE,
                ticker=ticker,
                legs=legs,
                strike=strike,
                expiry=expiry,
                is_long=is_long,
            )

            analysis = self._analyze_spread(straddle, current_price)
            return straddle, analysis

        except Exception as e:
            logger.error(f"Error building straddle: {e}")
            return None

    async def build_strangle(
        self,
        ticker: str,
        current_price: Decimal,
        expiry: Optional[date] = None,
        width: int = 5,
        is_long: bool = True,
    ) -> Optional[Tuple[Strangle, SpreadAnalysis]]:
        """
        Build a strangle (OTM call + OTM put).

        Args:
            ticker: Stock ticker
            current_price: Current stock price
            expiry: Expiration date
            width: Distance from ATM for each leg
            is_long: True for long strangle, False for short
        """
        try:
            if expiry is None:
                expiry = self._select_optimal_expiry()

            atm = Decimal(str(round(float(current_price), 0)))
            put_strike = atm - Decimal(str(width))
            call_strike = atm + Decimal(str(width))

            call_prem = await self._get_option_premium(
                ticker, call_strike, expiry, "call", current_price
            )
            put_prem = await self._get_option_premium(
                ticker, put_strike, expiry, "put", current_price
            )

            if is_long:
                legs = [
                    SpreadLeg(LegType.LONG_CALL, call_strike, expiry, 1, call_prem),
                    SpreadLeg(LegType.LONG_PUT, put_strike, expiry, 1, put_prem),
                ]
            else:
                legs = [
                    SpreadLeg(LegType.SHORT_CALL, call_strike, expiry, -1, call_prem),
                    SpreadLeg(LegType.SHORT_PUT, put_strike, expiry, -1, put_prem),
                ]

            strangle = Strangle(
                spread_type=SpreadType.STRANGLE,
                ticker=ticker,
                legs=legs,
                put_strike=put_strike,
                call_strike=call_strike,
                expiry=expiry,
                is_long=is_long,
            )

            analysis = self._analyze_spread(strangle, current_price)
            return strangle, analysis

        except Exception as e:
            logger.error(f"Error building strangle: {e}")
            return None

    async def build_calendar_spread(
        self,
        ticker: str,
        current_price: Decimal,
        strike: Optional[Decimal] = None,
        near_expiry: Optional[date] = None,
        far_expiry: Optional[date] = None,
        option_type: str = "call",
    ) -> Optional[Tuple[CalendarSpread, SpreadAnalysis]]:
        """
        Build a calendar spread (same strike, different expirations).

        Args:
            ticker: Stock ticker
            current_price: Current stock price
            strike: Strike price (defaults to ATM)
            near_expiry: Near-term expiration
            far_expiry: Far-term expiration
            option_type: "call" or "put"
        """
        try:
            if strike is None:
                strike = Decimal(str(round(float(current_price), 0)))

            if near_expiry is None:
                near_expiry = date.today() + timedelta(days=self.config.min_dte)
                # Adjust to Friday
                while near_expiry.weekday() != 4:
                    near_expiry += timedelta(days=1)

            if far_expiry is None:
                far_expiry = near_expiry + timedelta(days=28)
                while far_expiry.weekday() != 4:
                    far_expiry += timedelta(days=1)

            near_prem = await self._get_option_premium(
                ticker, strike, near_expiry, option_type, current_price
            )
            far_prem = await self._get_option_premium(
                ticker, strike, far_expiry, option_type, current_price
            )

            leg_type_near = LegType.SHORT_CALL if option_type == "call" else LegType.SHORT_PUT
            leg_type_far = LegType.LONG_CALL if option_type == "call" else LegType.LONG_PUT

            legs = [
                SpreadLeg(leg_type_near, strike, near_expiry, -1, near_prem),
                SpreadLeg(leg_type_far, strike, far_expiry, 1, far_prem),
            ]

            calendar = CalendarSpread(
                spread_type=SpreadType.CALENDAR,
                ticker=ticker,
                legs=legs,
                strike=strike,
                near_expiry=near_expiry,
                far_expiry=far_expiry,
                option_type=option_type,
            )

            analysis = self._analyze_spread(calendar, current_price)
            return calendar, analysis

        except Exception as e:
            logger.error(f"Error building calendar spread: {e}")
            return None

    async def build_ratio_spread(
        self,
        ticker: str,
        current_price: Decimal,
        expiry: Optional[date] = None,
        long_strike: Optional[Decimal] = None,
        short_strike: Optional[Decimal] = None,
        ratio: Tuple[int, int] = (1, 2),
        option_type: str = "call",
    ) -> Optional[Tuple[RatioSpread, SpreadAnalysis]]:
        """
        Build a ratio spread.

        Args:
            ticker: Stock ticker
            current_price: Current stock price
            expiry: Expiration date
            long_strike: Strike for long leg (defaults to ATM)
            short_strike: Strike for short leg (defaults to OTM)
            ratio: (long_contracts, short_contracts) e.g., (1, 2)
            option_type: "call" or "put"
        """
        try:
            if expiry is None:
                expiry = self._select_optimal_expiry()

            if long_strike is None:
                long_strike = Decimal(str(round(float(current_price), 0)))

            if short_strike is None:
                if option_type == "call":
                    short_strike = long_strike + Decimal("5")
                else:
                    short_strike = long_strike - Decimal("5")

            long_prem = await self._get_option_premium(
                ticker, long_strike, expiry, option_type, current_price
            )
            short_prem = await self._get_option_premium(
                ticker, short_strike, expiry, option_type, current_price
            )

            if option_type == "call":
                leg_type_long = LegType.LONG_CALL
                leg_type_short = LegType.SHORT_CALL
            else:
                leg_type_long = LegType.LONG_PUT
                leg_type_short = LegType.SHORT_PUT

            legs = [
                SpreadLeg(leg_type_long, long_strike, expiry, ratio[0], long_prem),
                SpreadLeg(leg_type_short, short_strike, expiry, -ratio[1], short_prem),
            ]

            ratio_spread = RatioSpread(
                spread_type=SpreadType.RATIO_SPREAD,
                ticker=ticker,
                legs=legs,
                long_strike=long_strike,
                short_strike=short_strike,
                long_contracts=ratio[0],
                short_contracts=ratio[1],
                option_type=option_type,
                expiry=expiry,
            )

            analysis = self._analyze_spread(ratio_spread, current_price)
            return ratio_spread, analysis

        except Exception as e:
            logger.error(f"Error building ratio spread: {e}")
            return None

    def _select_optimal_expiry(self) -> date:
        """Select optimal expiration date based on config."""
        target_dte = (self.config.min_dte + self.config.max_dte) // 2
        expiry = date.today() + timedelta(days=target_dte)

        # Adjust to Friday (standard options expiration)
        while expiry.weekday() != 4:
            expiry += timedelta(days=1)

        return expiry

    def _analyze_spread(
        self,
        spread,
        current_price: Decimal,
    ) -> SpreadAnalysis:
        """Analyze a spread and return analysis results."""
        max_profit = spread.get_max_profit()
        max_loss = spread.get_max_loss()
        breakevens = spread.get_breakeven_points()
        greeks = spread.aggregate_greeks

        # Calculate risk/reward ratio
        risk_reward = None
        if max_profit and max_loss and max_loss > 0:
            risk_reward = float(max_loss / max_profit)

        # Estimate probability of profit (simplified)
        pop = None
        if breakevens and len(breakevens) >= 2:
            lower_be, upper_be = breakevens[0], breakevens[-1]
            profit_zone = float(upper_be - lower_be)
            # Rough estimate based on profit zone width relative to price
            pop = min(0.95, profit_zone / float(current_price))

        # Generate recommendation
        notes = []
        recommendation = "consider"

        if risk_reward and risk_reward > self.config.max_risk_reward_ratio:
            notes.append(f"High risk/reward ratio: {risk_reward:.2f}")
            recommendation = "caution"
        elif risk_reward and risk_reward < 1.5:
            notes.append(f"Favorable risk/reward: {risk_reward:.2f}")
            recommendation = "favorable"

        if spread.is_credit:
            notes.append(f"Net credit: ${float(spread.net_premium):.2f}")
        else:
            notes.append(f"Net debit: ${abs(float(spread.net_premium)):.2f}")

        if max_profit:
            notes.append(f"Max profit: ${float(max_profit):.2f}")
        if max_loss and max_loss != Decimal("999999"):
            notes.append(f"Max loss: ${float(max_loss):.2f}")

        return SpreadAnalysis(
            spread=spread,
            max_profit=max_profit,
            max_loss=max_loss if max_loss != Decimal("999999") else None,
            breakeven_points=breakevens,
            probability_of_profit=pop,
            risk_reward_ratio=risk_reward,
            greeks=greeks,
            recommendation=recommendation,
            notes=notes,
        )


async def create_spread_builder(
    config: Optional[SpreadBuilderConfig] = None,
) -> SpreadBuilder:
    """Factory function to create a spread builder."""
    return SpreadBuilder(config=config)
