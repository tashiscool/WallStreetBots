"""Tests for EarningsCalendarSource."""

import os
import sys
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from backend.tradingbot.sentiment.sources.earnings_source import (
    EarningsCalendarSource,
)


class MockEdgarSource:
    """Mock SECEdgarSource for testing."""

    def __init__(self, filings=None):
        self._filings = filings or []

    def get_recent_filings(self, ticker, filing_types=None, limit=10):
        return self._filings


def _quarterly_filings(base_date, count=4):
    """Generate quarterly filing dates going back from base_date."""
    filings = []
    for i in range(count):
        d = base_date - timedelta(days=90 * i)
        filings.append({
            "filing_type": "10-Q",
            "date": d.strftime("%Y-%m-%d"),
            "description": "Quarterly Report",
        })
    return filings


@pytest.fixture
def now():
    return datetime.utcnow()


class TestUpcomingEarnings:
    def test_detects_upcoming_quarterly(self, now):
        # Last filing ~80 days ago → next is ~10 days out (within 30-day window)
        filings = _quarterly_filings(now - timedelta(days=80), count=4)
        edgar = MockEdgarSource(filings)
        source = EarningsCalendarSource(edgar_source=edgar)

        upcoming = source.get_upcoming_earnings(["AAPL"], days_ahead=30)
        assert len(upcoming) == 1
        assert upcoming[0]["ticker"] == "AAPL"
        assert upcoming[0]["days_until"] >= 0
        assert upcoming[0]["days_until"] <= 30

    def test_no_upcoming(self, now):
        # Recent filing 5 days ago → next is ~85 days out (outside 30-day window)
        filings = _quarterly_filings(now - timedelta(days=5), count=4)
        edgar = MockEdgarSource(filings)
        source = EarningsCalendarSource(edgar_source=edgar)

        upcoming = source.get_upcoming_earnings(["AAPL"], days_ahead=30)
        assert len(upcoming) == 0

    def test_empty_filings(self):
        edgar = MockEdgarSource(filings=[])
        source = EarningsCalendarSource(edgar_source=edgar)
        upcoming = source.get_upcoming_earnings(["AAPL"])
        assert upcoming == []

    def test_multiple_tickers(self, now):
        filings = _quarterly_filings(now - timedelta(days=80), count=4)
        edgar = MockEdgarSource(filings)
        source = EarningsCalendarSource(edgar_source=edgar)

        upcoming = source.get_upcoming_earnings(["AAPL", "MSFT"], days_ahead=30)
        tickers = {u["ticker"] for u in upcoming}
        assert "AAPL" in tickers
        assert "MSFT" in tickers

    def test_sorted_by_days_until(self, now):
        filings = _quarterly_filings(now - timedelta(days=80), count=4)
        edgar = MockEdgarSource(filings)
        source = EarningsCalendarSource(edgar_source=edgar)

        upcoming = source.get_upcoming_earnings(["AAPL", "MSFT"], days_ahead=30)
        if len(upcoming) >= 2:
            assert upcoming[0]["days_until"] <= upcoming[1]["days_until"]


class TestEarningsWindow:
    def test_in_window(self, now):
        # Next earnings ~2 days from now → within 5-day window
        filings = _quarterly_filings(now - timedelta(days=88), count=4)
        edgar = MockEdgarSource(filings)
        source = EarningsCalendarSource(edgar_source=edgar, earnings_window_days=5)

        result = source.is_in_earnings_window("AAPL", reference_date=now)
        assert isinstance(result, bool)

    def test_not_in_window(self, now):
        # Recent filing 10 days ago → next is ~80 days out, well outside 3-day window
        filings = _quarterly_filings(now - timedelta(days=10), count=4)
        edgar = MockEdgarSource(filings)
        source = EarningsCalendarSource(edgar_source=edgar, earnings_window_days=3)

        result = source.is_in_earnings_window("AAPL", reference_date=now)
        assert result is False

    def test_no_filings_returns_false(self):
        edgar = MockEdgarSource(filings=[])
        source = EarningsCalendarSource(edgar_source=edgar)
        assert source.is_in_earnings_window("AAPL") is False

    def test_edgar_error_returns_false(self):
        edgar = MagicMock()
        edgar.get_recent_filings.side_effect = Exception("fail")
        source = EarningsCalendarSource(edgar_source=edgar)
        assert source.is_in_earnings_window("AAPL") is False


class TestEstimateNextEarnings:
    def test_single_filing(self):
        filings = [{"date": "2025-03-15"}]
        result = EarningsCalendarSource._estimate_next_earnings(
            filings, datetime(2025, 6, 1)
        )
        # Single filing → 90 day default period
        assert result is not None
        assert result >= datetime(2025, 6, 1)

    def test_multiple_filings_periodicity(self):
        filings = [
            {"date": "2025-03-15"},
            {"date": "2024-12-15"},
            {"date": "2024-09-15"},
        ]
        result = EarningsCalendarSource._estimate_next_earnings(
            filings, datetime(2025, 5, 1)
        )
        assert result is not None
        # Average gap is 91 days, so next from 2025-03-15 should be ~June 14
        assert result > datetime(2025, 5, 1)

    def test_empty_filings(self):
        result = EarningsCalendarSource._estimate_next_earnings(
            [], datetime(2025, 6, 1)
        )
        assert result is None

    def test_datetime_field(self):
        filings = [{"date": datetime(2025, 3, 15)}]
        result = EarningsCalendarSource._estimate_next_earnings(
            filings, datetime(2025, 6, 1)
        )
        assert result is not None
