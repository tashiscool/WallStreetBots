"""
Earnings Calendar Source — Upcoming earnings dates and earnings window detection.

Leverages the existing SECEdgarSource for filing date information.
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class EarningsCalendarSource:
    """
    Provides earnings calendar data by analyzing SEC filing patterns.

    Uses ``SECEdgarSource`` to find 10-Q / 10-K filing dates as
    earnings proxies.
    """

    def __init__(
        self,
        edgar_source: Optional[Any] = None,
        earnings_window_days: int = 3,
    ) -> None:
        """
        Args:
            edgar_source: An existing ``SECEdgarSource`` instance (one will
                be created if not provided).
            earnings_window_days: Number of days before/after earnings
                that constitute the "earnings window".
        """
        if edgar_source is None:
            from .sec_edgar_source import SECEdgarSource
            edgar_source = SECEdgarSource()
        self._edgar = edgar_source
        self.earnings_window_days = earnings_window_days

    def get_upcoming_earnings(
        self,
        tickers: List[str],
        days_ahead: int = 30,
    ) -> List[Dict[str, Any]]:
        """Find tickers with upcoming earnings within *days_ahead*.

        Estimates next earnings based on historical 10-Q/10-K filing
        periodicity.

        Args:
            tickers: Symbols to check.
            days_ahead: Look-ahead window in days.

        Returns:
            List of dicts with ``ticker``, ``estimated_date``, ``filing_type``,
            ``days_until``.
        """
        upcoming: List[Dict[str, Any]] = []
        now = datetime.utcnow()

        for ticker in tickers:
            try:
                filings = self._edgar.get_recent_filings(
                    ticker,
                    filing_types=["10-Q", "10-K"],
                    limit=8,
                )
            except Exception as exc:
                logger.debug("Could not fetch filings for %s: %s", ticker, exc)
                continue

            if not filings:
                continue

            next_date = self._estimate_next_earnings(filings, now)
            if next_date is None:
                continue

            days_until = (next_date - now).days
            if 0 <= days_until <= days_ahead:
                upcoming.append({
                    "ticker": ticker,
                    "estimated_date": next_date,
                    "filing_type": filings[0].get("filing_type", "10-Q"),
                    "days_until": days_until,
                })

        upcoming.sort(key=lambda x: x["days_until"])
        return upcoming

    def is_in_earnings_window(
        self,
        ticker: str,
        reference_date: Optional[datetime] = None,
    ) -> bool:
        """Check if *ticker* is currently within an earnings window.

        Args:
            ticker: Symbol to check.
            reference_date: Date to check against (default: now).

        Returns:
            ``True`` if within ``earnings_window_days`` of an estimated
            earnings date.
        """
        ref = reference_date or datetime.utcnow()
        try:
            filings = self._edgar.get_recent_filings(
                ticker, filing_types=["10-Q", "10-K"], limit=8,
            )
        except Exception:
            return False

        if not filings:
            return False

        next_date = self._estimate_next_earnings(filings, ref)
        if next_date is None:
            return False

        delta = abs((next_date - ref).days)
        return delta <= self.earnings_window_days

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _estimate_next_earnings(
        filings: List[Dict[str, Any]],
        reference: datetime,
    ) -> Optional[datetime]:
        """Estimate next earnings date from historical filing periodicity."""
        dates: List[datetime] = []
        for f in filings:
            raw = f.get("date") or f.get("filing_date")
            if raw is None:
                continue
            if isinstance(raw, str):
                try:
                    dt = datetime.strptime(raw, "%Y-%m-%d")
                except ValueError:
                    continue
            elif isinstance(raw, datetime):
                dt = raw
            else:
                continue
            dates.append(dt)

        if not dates:
            return None

        dates.sort()

        # Estimate periodicity (typically ~90 days for 10-Q)
        if len(dates) >= 2:
            gaps = [
                (dates[i + 1] - dates[i]).days
                for i in range(len(dates) - 1)
            ]
            avg_gap = sum(gaps) / len(gaps)
        else:
            avg_gap = 90  # Default quarterly

        # Project next from most recent
        latest = dates[-1]
        next_date = latest + timedelta(days=avg_gap)

        # If projected date is in the past, keep adding periods
        while next_date < reference:
            next_date += timedelta(days=avg_gap)

        return next_date
