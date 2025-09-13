#!/usr/bin/env python
"""Test compliance guard for trading constraints."""

import unittest
import datetime as dt
from backend.tradingbot.compliance import (
    ComplianceGuard,
    ComplianceError,
    PDTViolation,
    SSRViolation,
    HaltViolation,
    SessionViolation,
    SessionCalendar,
    DayTradeEvent,
)


class TestComplianceGuard(unittest.TestCase):
    def setUp(self):
        self.guard = ComplianceGuard(min_equity_for_day_trading=25_000.0)

    def test_pdt_check_high_equity(self):
        """High equity accounts should pass PDT checks."""
        # Should not raise with equity >= $25k
        self.guard.check_pdt(
            account_equity=50_000.0, pending_day_trades_count=5, now=dt.datetime.now()
        )

    def test_pdt_check_low_equity_violation(self):
        """Low equity accounts with >= 4 day trades should violate PDT."""
        with self.assertRaises(PDTViolation):
            self.guard.check_pdt(
                account_equity=10_000.0,
                pending_day_trades_count=4,
                now=dt.datetime.now(),
            )

    def test_pdt_check_low_equity_safe(self):
        """Low equity accounts with < 4 day trades should pass."""
        self.guard.check_pdt(
            account_equity=10_000.0, pending_day_trades_count=3, now=dt.datetime.now()
        )

    def test_halt_check(self):
        """Halted symbols should raise violations."""
        self.guard.set_halt("SPY", "News pending")

        with self.assertRaises(HaltViolation):
            self.guard.check_halt("SPY")

        # Other symbols should pass
        self.guard.check_halt("QQQ")

        # Clear halt and it should pass
        self.guard.clear_halt("SPY")
        self.guard.check_halt("SPY")

    def test_luld_check(self):
        """LULD limit checks should prevent orders outside bands."""
        self.guard.set_luld("SPY", lower=400.0, upper=500.0)

        # Price within band should pass
        self.guard.check_luld("SPY", limit_price=450.0)

        # Price outside band should fail
        with self.assertRaises(HaltViolation):
            self.guard.check_luld("SPY", limit_price=600.0)

        with self.assertRaises(HaltViolation):
            self.guard.check_luld("SPY", limit_price=300.0)

        # Market orders (no limit) should pass
        self.guard.check_luld("SPY", limit_price=None)

    def test_ssr_check(self):
        """SSR should block short sales."""
        today = dt.date.today()
        self.guard.set_ssr("SPY", today)

        with self.assertRaises(SSRViolation):
            self.guard.check_ssr("SPY", side="short", now=dt.datetime.now())

        # Other sides should pass
        self.guard.check_ssr("SPY", side="buy", now=dt.datetime.now())
        self.guard.check_ssr("SPY", side="sell", now=dt.datetime.now())

    def test_day_trade_recording(self):
        """Day trade events should be recorded and pruned."""
        now = dt.datetime.now()
        opened = now - dt.timedelta(hours=2)
        closed = now

        self.guard.record_day_trade("SPY", opened, closed)

        # Should have one day trade
        self.assertEqual(len(self.guard.day_trades), 1)
        self.assertEqual(self.guard.day_trades[0].symbol, "SPY")

    def test_session_calendar(self):
        """Session calendar should correctly identify trading sessions."""
        calendar = SessionCalendar()

        # Test regular session (1:30 PM UTC = 9:30 AM ET)
        regular_time = dt.datetime(
            2025, 1, 15, 15, 0, tzinfo=dt.timezone.utc
        )  # 3 PM UTC
        self.assertEqual(calendar.session(regular_time), "regular")

        # Test pre-market (10:00 AM UTC = 6:00 AM ET)
        pre_time = dt.datetime(2025, 1, 15, 10, 0, tzinfo=dt.timezone.utc)
        self.assertEqual(calendar.session(pre_time), "pre")

        # Test after hours (21:00 UTC = 5:00 PM ET)
        post_time = dt.datetime(2025, 1, 15, 21, 0, tzinfo=dt.timezone.utc)
        self.assertEqual(calendar.session(post_time), "post")

    def test_session_check(self):
        """Session checks should enforce trading hours."""
        # Mock time during regular session
        regular_time = dt.datetime(2025, 1, 15, 15, 0, tzinfo=dt.timezone.utc)

        # Regular session should always pass
        self.guard.check_session(regular_time, allow_pre=False, allow_post=False)

        # Pre-market time
        pre_time = dt.datetime(2025, 1, 15, 10, 0, tzinfo=dt.timezone.utc)

        # Should pass if pre-market allowed
        self.guard.check_session(pre_time, allow_pre=True, allow_post=False)

        # Should fail if pre-market not allowed
        with self.assertRaises(SessionViolation):
            self.guard.check_session(pre_time, allow_pre=False, allow_post=False)


if __name__ == "__main__":
    unittest.main()
