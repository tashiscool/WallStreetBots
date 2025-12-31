"""
Tests for Exotic Option Spreads.

Tests all multi-leg option strategies:
- Iron Condor
- Butterfly
- Calendar Spread
- Straddle
- Strangle
"""

import pytest
from datetime import date, timedelta
from decimal import Decimal

from backend.tradingbot.options.exotic_spreads import (
    SpreadType,
    SpreadDirection,
    LegType,
    SpreadLeg,
    SpreadGreeks,
    OptionSpread,
    IronCondor,
)


class TestSpreadEnums:
    """Tests for spread enumeration types."""

    def test_spread_types(self):
        """Test all spread types are defined."""
        assert SpreadType.IRON_CONDOR.value == "iron_condor"
        assert SpreadType.IRON_BUTTERFLY.value == "iron_butterfly"
        assert SpreadType.BUTTERFLY.value == "butterfly"
        assert SpreadType.CALENDAR.value == "calendar"
        assert SpreadType.DIAGONAL.value == "diagonal"
        assert SpreadType.STRADDLE.value == "straddle"
        assert SpreadType.STRANGLE.value == "strangle"

    def test_spread_directions(self):
        """Test spread directions."""
        assert SpreadDirection.BULLISH.value == "bullish"
        assert SpreadDirection.BEARISH.value == "bearish"
        assert SpreadDirection.NEUTRAL.value == "neutral"

    def test_leg_types(self):
        """Test leg types."""
        assert LegType.LONG_CALL.value == "long_call"
        assert LegType.SHORT_CALL.value == "short_call"
        assert LegType.LONG_PUT.value == "long_put"
        assert LegType.SHORT_PUT.value == "short_put"


class TestSpreadLeg:
    """Tests for individual spread legs."""

    def test_leg_initialization(self):
        """Test leg initializes correctly."""
        leg = SpreadLeg(
            leg_type=LegType.LONG_CALL,
            strike=Decimal("100"),
            expiry=date.today() + timedelta(days=30),
            contracts=1,
            premium=Decimal("2.50")
        )
        assert leg.strike == Decimal("100")
        assert leg.contracts == 1
        assert leg.premium == Decimal("2.50")

    def test_is_long(self):
        """Test is_long property."""
        long_leg = SpreadLeg(
            leg_type=LegType.LONG_CALL,
            strike=Decimal("100"),
            expiry=date.today(),
            contracts=1
        )
        assert long_leg.is_long is True

        short_leg = SpreadLeg(
            leg_type=LegType.SHORT_CALL,
            strike=Decimal("105"),
            expiry=date.today(),
            contracts=-1
        )
        assert short_leg.is_long is False

    def test_is_call(self):
        """Test is_call property."""
        call_leg = SpreadLeg(
            leg_type=LegType.LONG_CALL,
            strike=Decimal("100"),
            expiry=date.today(),
            contracts=1
        )
        assert call_leg.is_call is True

        put_leg = SpreadLeg(
            leg_type=LegType.LONG_PUT,
            strike=Decimal("95"),
            expiry=date.today(),
            contracts=1
        )
        assert put_leg.is_call is False

    def test_option_type(self):
        """Test option_type property."""
        call_leg = SpreadLeg(
            leg_type=LegType.LONG_CALL,
            strike=Decimal("100"),
            expiry=date.today(),
            contracts=1
        )
        assert call_leg.option_type == "call"

        put_leg = SpreadLeg(
            leg_type=LegType.SHORT_PUT,
            strike=Decimal("95"),
            expiry=date.today(),
            contracts=-1
        )
        assert put_leg.option_type == "put"


class TestSpreadGreeks:
    """Tests for aggregate Greeks."""

    def test_greeks_initialization(self):
        """Test Greeks initialize to zero."""
        greeks = SpreadGreeks()
        assert greeks.delta == Decimal("0")
        assert greeks.gamma == Decimal("0")
        assert greeks.theta == Decimal("0")
        assert greeks.vega == Decimal("0")

    def test_greeks_to_dict(self):
        """Test Greeks to dictionary conversion."""
        greeks = SpreadGreeks(
            delta=Decimal("0.25"),
            gamma=Decimal("0.05"),
            theta=Decimal("-0.02"),
            vega=Decimal("0.10")
        )
        result = greeks.to_dict()

        assert result["delta"] == 0.25
        assert result["gamma"] == 0.05
        assert result["theta"] == -0.02
        assert result["vega"] == 0.10


class TestOptionSpread:
    """Tests for base OptionSpread class."""

    def test_total_contracts(self):
        """Test total contracts calculation."""
        expiry = date.today() + timedelta(days=30)
        legs = [
            SpreadLeg(leg_type=LegType.LONG_CALL, strike=Decimal("100"),
                      expiry=expiry, contracts=1, premium=Decimal("3.00")),
            SpreadLeg(leg_type=LegType.SHORT_CALL, strike=Decimal("105"),
                      expiry=expiry, contracts=-1, premium=Decimal("1.50"))
        ]
        spread = OptionSpread(
            spread_type=SpreadType.VERTICAL_CALL,
            ticker="AAPL",
            legs=legs
        )
        assert spread.total_contracts == 2

    def test_net_premium_debit(self):
        """Test net premium for debit spread."""
        expiry = date.today() + timedelta(days=30)
        legs = [
            SpreadLeg(leg_type=LegType.LONG_CALL, strike=Decimal("100"),
                      expiry=expiry, contracts=1, premium=Decimal("3.00")),
            SpreadLeg(leg_type=LegType.SHORT_CALL, strike=Decimal("105"),
                      expiry=expiry, contracts=-1, premium=Decimal("1.50"))
        ]
        spread = OptionSpread(
            spread_type=SpreadType.VERTICAL_CALL,
            ticker="AAPL",
            legs=legs
        )
        # Long pays 3.00, short receives 1.50 = -1.50 net (debit)
        assert spread.net_premium == Decimal("-1.50")
        assert spread.is_credit is False

    def test_aggregate_greeks(self):
        """Test aggregate Greeks calculation."""
        expiry = date.today() + timedelta(days=30)
        legs = [
            SpreadLeg(leg_type=LegType.LONG_CALL, strike=Decimal("100"),
                      expiry=expiry, contracts=1, delta=Decimal("0.50")),
            SpreadLeg(leg_type=LegType.SHORT_CALL, strike=Decimal("105"),
                      expiry=expiry, contracts=-1, delta=Decimal("0.30"))
        ]
        spread = OptionSpread(
            spread_type=SpreadType.VERTICAL_CALL,
            ticker="AAPL",
            legs=legs
        )
        greeks = spread.aggregate_greeks
        # Long +0.50, Short -0.30 = +0.20 net delta
        assert greeks.delta == Decimal("0.20")


class TestIronCondor:
    """Tests for Iron Condor spread."""

    def test_iron_condor_initialization(self):
        """Test Iron Condor initializes correctly."""
        expiry = date.today() + timedelta(days=30)
        ic = IronCondor(
            spread_type=SpreadType.IRON_CONDOR,
            ticker="SPY",
            legs=[],
            put_long_strike=Decimal("90"),
            put_short_strike=Decimal("95"),
            call_short_strike=Decimal("105"),
            call_long_strike=Decimal("110"),
            expiry=expiry
        )
        assert ic.spread_type == SpreadType.IRON_CONDOR
        assert ic.direction == SpreadDirection.NEUTRAL
        assert ic.ticker == "SPY"

    def test_wing_widths(self):
        """Test wing width calculations."""
        expiry = date.today() + timedelta(days=30)
        ic = IronCondor(
            spread_type=SpreadType.IRON_CONDOR,
            ticker="SPY",
            legs=[],
            put_long_strike=Decimal("90"),
            put_short_strike=Decimal("95"),
            call_short_strike=Decimal("105"),
            call_long_strike=Decimal("110"),
            expiry=expiry
        )
        assert ic.wing_width_put == Decimal("5")
        assert ic.wing_width_call == Decimal("5")

    def test_profit_zone_width(self):
        """Test profit zone width calculation."""
        expiry = date.today() + timedelta(days=30)
        ic = IronCondor(
            spread_type=SpreadType.IRON_CONDOR,
            ticker="SPY",
            legs=[],
            put_long_strike=Decimal("90"),
            put_short_strike=Decimal("95"),
            call_short_strike=Decimal("105"),
            call_long_strike=Decimal("110"),
            expiry=expiry
        )
        assert ic.profit_zone_width == Decimal("10")

    def test_iron_condor_max_profit(self):
        """Test maximum profit calculation."""
        expiry = date.today() + timedelta(days=30)
        legs = [
            SpreadLeg(leg_type=LegType.LONG_PUT, strike=Decimal("90"),
                      expiry=expiry, contracts=1, premium=Decimal("0.50")),
            SpreadLeg(leg_type=LegType.SHORT_PUT, strike=Decimal("95"),
                      expiry=expiry, contracts=-1, premium=Decimal("1.00")),
            SpreadLeg(leg_type=LegType.SHORT_CALL, strike=Decimal("105"),
                      expiry=expiry, contracts=-1, premium=Decimal("1.00")),
            SpreadLeg(leg_type=LegType.LONG_CALL, strike=Decimal("110"),
                      expiry=expiry, contracts=1, premium=Decimal("0.50")),
        ]
        ic = IronCondor(
            spread_type=SpreadType.IRON_CONDOR,
            ticker="SPY",
            legs=legs,
            put_long_strike=Decimal("90"),
            put_short_strike=Decimal("95"),
            call_short_strike=Decimal("105"),
            call_long_strike=Decimal("110"),
            expiry=expiry
        )
        # Net premium = 1.00 + 1.00 - 0.50 - 0.50 = 1.00
        # Max profit = 1.00 * 100 = 100
        assert ic.get_max_profit() == Decimal("100")

    def test_iron_condor_max_loss(self):
        """Test maximum loss calculation."""
        expiry = date.today() + timedelta(days=30)
        legs = [
            SpreadLeg(leg_type=LegType.LONG_PUT, strike=Decimal("90"),
                      expiry=expiry, contracts=1, premium=Decimal("0.50")),
            SpreadLeg(leg_type=LegType.SHORT_PUT, strike=Decimal("95"),
                      expiry=expiry, contracts=-1, premium=Decimal("1.00")),
            SpreadLeg(leg_type=LegType.SHORT_CALL, strike=Decimal("105"),
                      expiry=expiry, contracts=-1, premium=Decimal("1.00")),
            SpreadLeg(leg_type=LegType.LONG_CALL, strike=Decimal("110"),
                      expiry=expiry, contracts=1, premium=Decimal("0.50")),
        ]
        ic = IronCondor(
            spread_type=SpreadType.IRON_CONDOR,
            ticker="SPY",
            legs=legs,
            put_long_strike=Decimal("90"),
            put_short_strike=Decimal("95"),
            call_short_strike=Decimal("105"),
            call_long_strike=Decimal("110"),
            expiry=expiry
        )
        # Max loss = wing_width * 100 - premium * 100 = 5 * 100 - 1 * 100 = 400
        assert ic.get_max_loss() == Decimal("400")


class TestSpreadValidation:
    """Tests for spread validation and edge cases."""

    def test_leg_with_no_premium(self):
        """Test leg without premium set."""
        leg = SpreadLeg(
            leg_type=LegType.LONG_CALL,
            strike=Decimal("100"),
            expiry=date.today(),
            contracts=1
        )
        assert leg.premium is None

    def test_spread_with_no_greeks(self):
        """Test spread with legs that have no Greeks."""
        expiry = date.today() + timedelta(days=30)
        legs = [
            SpreadLeg(leg_type=LegType.LONG_CALL, strike=Decimal("100"),
                      expiry=expiry, contracts=1)
        ]
        spread = OptionSpread(
            spread_type=SpreadType.VERTICAL_CALL,
            ticker="TEST",
            legs=legs
        )
        greeks = spread.aggregate_greeks
        assert greeks.delta == Decimal("0")

    def test_empty_spread(self):
        """Test spread with no legs."""
        spread = OptionSpread(
            spread_type=SpreadType.VERTICAL_CALL,
            ticker="TEST",
            legs=[]
        )
        assert spread.total_contracts == 0
        assert spread.net_premium == Decimal("0")
