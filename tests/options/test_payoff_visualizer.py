"""Tests for options payoff visualization."""

import pytest
from decimal import Decimal
from datetime import date

from backend.tradingbot.options.exotic_spreads import (
    OptionSpread, SpreadLeg, SpreadType, LegType, SpreadDirection,
    IronCondor, Straddle, Butterfly,
)
from backend.tradingbot.options.payoff_visualizer import (
    _leg_pnl_at_price,
    _black_scholes_value,
    _leg_value_before_expiry,
    PayoffDiagramGenerator,
    PayoffDiagramConfig,
    GreeksDashboard,
    generate_pnl_heatmap,
)


class TestLegPnlAtPrice:
    """Test individual leg P&L calculations."""

    def test_long_call_itm(self):
        leg = SpreadLeg(
            leg_type=LegType.LONG_CALL,
            strike=Decimal("100"),
            expiry=date(2025, 3, 21),
            contracts=1,
            premium=Decimal("5.00"),
        )
        # Price at 110: intrinsic 10, paid 5, profit = (10-5)*100 = 500
        pnl = _leg_pnl_at_price(leg, 110.0)
        assert pnl == 500.0

    def test_long_call_otm(self):
        leg = SpreadLeg(
            leg_type=LegType.LONG_CALL,
            strike=Decimal("100"),
            expiry=date(2025, 3, 21),
            contracts=1,
            premium=Decimal("5.00"),
        )
        # Price at 90: intrinsic 0, paid 5, loss = -500
        pnl = _leg_pnl_at_price(leg, 90.0)
        assert pnl == -500.0

    def test_short_put_otm(self):
        leg = SpreadLeg(
            leg_type=LegType.SHORT_PUT,
            strike=Decimal("100"),
            expiry=date(2025, 3, 21),
            contracts=1,
            premium=Decimal("4.00"),
        )
        # Price at 110: intrinsic 0, received 4, profit = 400
        pnl = _leg_pnl_at_price(leg, 110.0)
        assert pnl == 400.0

    def test_long_put_itm(self):
        leg = SpreadLeg(
            leg_type=LegType.LONG_PUT,
            strike=Decimal("100"),
            expiry=date(2025, 3, 21),
            contracts=1,
            premium=Decimal("3.00"),
        )
        # Price at 90: intrinsic 10, paid 3, profit = 700
        pnl = _leg_pnl_at_price(leg, 90.0)
        assert pnl == 700.0


class TestBlackScholesValue:
    """Test Black-Scholes valuation."""

    def test_call_at_expiry(self):
        val = _black_scholes_value(110, 100, 0, 0.3, 0.05, True)
        assert val == 10.0

    def test_put_at_expiry(self):
        val = _black_scholes_value(90, 100, 0, 0.3, 0.05, False)
        assert val == 10.0

    def test_call_before_expiry(self):
        val = _black_scholes_value(100, 100, 30, 0.3, 0.05, True)
        assert val > 0  # ATM call should have time value

    def test_deep_otm_call(self):
        val = _black_scholes_value(50, 100, 30, 0.3, 0.05, True)
        assert val < 1.0  # Deep OTM, near zero


class TestPayoffDiagramGenerator:
    """Test payoff diagram generation."""

    def _make_bull_call_spread(self):
        return OptionSpread(
            spread_type=SpreadType.VERTICAL_CALL,
            ticker="AAPL",
            legs=[
                SpreadLeg(LegType.LONG_CALL, Decimal("150"), date(2025, 3, 21), 1, Decimal("5.00")),
                SpreadLeg(LegType.SHORT_CALL, Decimal("160"), date(2025, 3, 21), -1, Decimal("2.00")),
            ],
            direction=SpreadDirection.BULLISH,
        )

    def test_generate_html(self):
        gen = PayoffDiagramGenerator()
        spread = self._make_bull_call_spread()
        result = gen.generate(spread, current_price=155.0, days_to_expiry=30)
        assert isinstance(result, str)
        assert 'plotly' in result.lower() or 'not available' in result.lower()

    def test_config_defaults(self):
        config = PayoffDiagramConfig()
        assert config.price_range_pct == 0.30
        assert config.num_points == 200
        assert config.show_breakevens is True

    def test_iron_condor_payoff(self):
        spread = IronCondor(
            spread_type=SpreadType.IRON_CONDOR,
            ticker="SPY",
            legs=[
                SpreadLeg(LegType.LONG_PUT, Decimal("440"), date(2025, 3, 21), 1, Decimal("2.00")),
                SpreadLeg(LegType.SHORT_PUT, Decimal("450"), date(2025, 3, 21), -1, Decimal("4.00")),
                SpreadLeg(LegType.SHORT_CALL, Decimal("470"), date(2025, 3, 21), -1, Decimal("4.00")),
                SpreadLeg(LegType.LONG_CALL, Decimal("480"), date(2025, 3, 21), 1, Decimal("2.00")),
            ],
            put_long_strike=Decimal("440"),
            put_short_strike=Decimal("450"),
            call_short_strike=Decimal("470"),
            call_long_strike=Decimal("480"),
        )
        gen = PayoffDiagramGenerator()
        result = gen.generate(spread, current_price=460.0)
        assert isinstance(result, str)


class TestGreeksDashboard:
    """Test Greeks dashboard generation."""

    def test_generate_html(self):
        spread = OptionSpread(
            spread_type=SpreadType.STRADDLE,
            ticker="AAPL",
            legs=[
                SpreadLeg(LegType.LONG_CALL, Decimal("150"), date(2025, 3, 21), 1, Decimal("5.00")),
                SpreadLeg(LegType.LONG_PUT, Decimal("150"), date(2025, 3, 21), 1, Decimal("4.50")),
            ],
        )
        dashboard = GreeksDashboard()
        result = dashboard.generate(spread, current_price=150.0, days_to_expiry=30)
        assert isinstance(result, str)

    def test_greeks_calculation(self):
        dashboard = GreeksDashboard()
        delta, gamma, theta, vega = dashboard._calculate_greeks(
            100.0, 100.0, 30, 0.30, 0.05, True
        )
        # ATM call delta should be ~0.5
        assert 0.4 < delta < 0.6
        # Gamma should be positive
        assert gamma > 0


class TestPnlHeatmap:
    """Test P&L heatmap generation."""

    def test_generate(self):
        spread = OptionSpread(
            spread_type=SpreadType.STRADDLE,
            ticker="AAPL",
            legs=[
                SpreadLeg(LegType.LONG_CALL, Decimal("150"), date(2025, 3, 21), 1, Decimal("5.00")),
                SpreadLeg(LegType.LONG_PUT, Decimal("150"), date(2025, 3, 21), 1, Decimal("4.50")),
            ],
        )
        result = generate_pnl_heatmap(spread, current_price=150.0, days_to_expiry=30)
        assert isinstance(result, str)
