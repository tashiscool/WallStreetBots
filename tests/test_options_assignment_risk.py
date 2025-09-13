#!/usr/bin/env python
"""Test options assignment risk management."""

import unittest
import datetime as dt
from backend.tradingbot.options import (
    OptionContract,
    UnderlyingState,
    auto_exercise_likely,
    early_assignment_risk,
    pin_risk,
)


class TestOptionsAssignmentRisk(unittest.TestCase):
    def setUp(self):
        self.today = dt.date.today()
        self.tomorrow = self.today + dt.timedelta(days=1)

        # Mock option contracts
        self.call_itm = OptionContract(
            symbol="AAPL 2025-01-17 180 C",
            underlying="AAPL",
            strike=180.0,
            right="C",
            expiry=self.today,
        )

        self.call_otm = OptionContract(
            symbol="AAPL 2025-01-17 200 C",
            underlying="AAPL",
            strike=200.0,
            right="C",
            expiry=self.today,
        )

        self.put_itm = OptionContract(
            symbol="AAPL 2025-01-17 200 P",
            underlying="AAPL",
            strike=200.0,
            right="P",
            expiry=self.today,
        )

    def test_auto_exercise_calls_itm(self):
        """ITM calls at expiration should trigger auto-exercise."""
        underlying = UnderlyingState(price=185.0)

        # ITM call should be auto-exercised
        self.assertTrue(auto_exercise_likely(self.call_itm, underlying))

        # OTM call should not be auto-exercised
        self.assertFalse(auto_exercise_likely(self.call_otm, underlying))

    def test_auto_exercise_puts_itm(self):
        """ITM puts at expiration should trigger auto-exercise."""
        underlying = UnderlyingState(price=195.0)

        # ITM put should be auto-exercised
        self.assertTrue(auto_exercise_likely(self.put_itm, underlying))

    def test_auto_exercise_threshold(self):
        """Auto-exercise should respect threshold."""
        # Just barely ITM (less than $0.01)
        underlying = UnderlyingState(price=180.005)

        # Should not trigger with default $0.01 threshold
        self.assertFalse(auto_exercise_likely(self.call_itm, underlying))

        # Should trigger with lower threshold
        self.assertTrue(
            auto_exercise_likely(self.call_itm, underlying, threshold=0.001)
        )

    def test_auto_exercise_non_expiry(self):
        """Auto-exercise should only apply on expiry date."""
        # Use option expiring tomorrow
        future_call = OptionContract(
            symbol="AAPL 2025-01-18 180 C",
            underlying="AAPL",
            strike=180.0,
            right="C",
            expiry=self.tomorrow,
        )

        underlying = UnderlyingState(price=185.0)

        # Should not auto-exercise before expiry
        self.assertFalse(auto_exercise_likely(future_call, underlying))

    def test_early_assignment_risk_ex_div(self):
        """Early assignment risk should increase before ex-dividend."""
        # Deep ITM call with ex-div tomorrow
        underlying = UnderlyingState(
            price=200.0, next_ex_div_date=self.tomorrow, div_amount=2.0
        )

        call_deep_itm = OptionContract(
            symbol="AAPL 2025-03-21 180 C",
            underlying="AAPL",
            strike=180.0,
            right="C",
            expiry=dt.date(2025, 3, 21),
        )

        # Should indicate early assignment risk
        self.assertTrue(early_assignment_risk(call_deep_itm, underlying))

    def test_early_assignment_risk_no_dividend(self):
        """No early assignment risk without dividend."""
        # Deep ITM call with no dividend
        underlying = UnderlyingState(price=200.0, next_ex_div_date=None, div_amount=0.0)

        call_deep_itm = OptionContract(
            symbol="AAPL 2025-03-21 180 C",
            underlying="AAPL",
            strike=180.0,
            right="C",
            expiry=dt.date(2025, 3, 21),
        )

        # Should not indicate early assignment risk
        self.assertFalse(early_assignment_risk(call_deep_itm, underlying))

    def test_early_assignment_risk_puts(self):
        """Puts should not have early assignment risk around ex-div."""
        underlying = UnderlyingState(
            price=150.0, next_ex_div_date=self.tomorrow, div_amount=2.0
        )

        put_deep_itm = OptionContract(
            symbol="AAPL 2025-03-21 200 P",
            underlying="AAPL",
            strike=200.0,
            right="P",
            expiry=dt.date(2025, 3, 21),
        )

        # Puts should not have early assignment risk for dividends
        self.assertFalse(early_assignment_risk(put_deep_itm, underlying))

    def test_pin_risk_at_strike(self):
        """Pin risk should trigger when spot is near strike at expiry."""
        # Spot exactly at strike
        underlying = UnderlyingState(price=180.0)

        self.assertTrue(pin_risk(self.call_itm, underlying))

    def test_pin_risk_within_band(self):
        """Pin risk should trigger within specified band."""
        # Within 10 bps (default band)
        # 10 bps of $180 = $0.018
        underlying = UnderlyingState(price=180.01)  # Just within band

        self.assertTrue(pin_risk(self.call_itm, underlying, band_bps=10.0))

    def test_pin_risk_outside_band(self):
        """Pin risk should not trigger outside band."""
        # Outside 10 bps band
        underlying = UnderlyingState(price=182.0)

        self.assertFalse(pin_risk(self.call_itm, underlying, band_bps=10.0))

    def test_pin_risk_non_expiry(self):
        """Pin risk should only apply on expiry date."""
        # Use option expiring tomorrow
        future_call = OptionContract(
            symbol="AAPL 2025-01-18 180 C",
            underlying="AAPL",
            strike=180.0,
            right="C",
            expiry=self.tomorrow,
        )

        underlying = UnderlyingState(price=180.0)

        # Should not indicate pin risk before expiry
        self.assertFalse(pin_risk(future_call, underlying))

    def test_pin_risk_custom_band(self):
        """Pin risk should respect custom band width."""
        underlying = UnderlyingState(price=180.05)  # $0.05 away from strike

        # Should trigger with wider band (50 bps = $0.90 for $180 strike)
        self.assertTrue(pin_risk(self.call_itm, underlying, band_bps=50.0))

        # Should not trigger with narrower band (10 bps = $0.18 for $180 strike)
        # Use price further away: $180.25 is $0.25 away, which is > $0.18
        underlying_far = UnderlyingState(price=180.25)
        self.assertFalse(pin_risk(self.call_itm, underlying_far, band_bps=10.0))


if __name__ == "__main__":
    unittest.main()
