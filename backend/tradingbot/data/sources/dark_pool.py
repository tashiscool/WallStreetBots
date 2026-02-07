"""
Dark Pool Data Source — FINRA ATS (Alternative Trading System) weekly data.

Public, free data from FINRA's ATS transparency reports.
"""

import csv
import io
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import requests

    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

# FINRA ATS weekly data endpoint
_FINRA_ATS_URL = "https://api.finra.org/data/group/otcmarket/name/weeklysummary"
_DEFAULT_USER_AGENT = "WallStreetBots/1.0"


@dataclass
class DarkPoolData:
    """Weekly dark pool (ATS) trading data."""

    ticker: str
    week_ending: datetime
    total_shares: int
    total_trades: int = 0
    ats_name: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class DarkPoolDataSource:
    """
    FINRA ATS weekly transparency data.

    Provides dark pool volume data for detecting institutional
    accumulation/distribution patterns.
    """

    def __init__(self, user_agent: str = _DEFAULT_USER_AGENT) -> None:
        self._user_agent = user_agent
        self._session: Optional[Any] = None

        if HAS_REQUESTS:
            self._session = requests.Session()
            self._session.headers.update({
                "User-Agent": self._user_agent,
                "Accept": "application/json",
            })

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_ats_data(
        self,
        ticker: str,
        weeks_back: int = 12,
    ) -> List[DarkPoolData]:
        """Fetch weekly ATS data for a ticker.

        Args:
            ticker: Stock ticker symbol.
            weeks_back: Number of weeks to look back.

        Returns:
            List of ``DarkPoolData`` objects, most recent first.
        """
        if self._session is None:
            logger.warning("requests not installed; DarkPoolDataSource disabled")
            return []

        try:
            payload = {
                "fields": ["weekStartDate", "totalWeeklyShareQuantity",
                           "totalWeeklyTradeCount", "issueSymbolIdentifier",
                           "lastUpdateDate"],
                "compareFilters": [
                    {
                        "fieldName": "issueSymbolIdentifier",
                        "fieldValue": ticker.upper(),
                        "compareType": "EQUAL",
                    }
                ],
                "limit": weeks_back,
                "sortFields": ["-weekStartDate"],
            }
            resp = self._session.post(
                _FINRA_ATS_URL,
                json=payload,
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.error("FINRA ATS request failed for %s: %s", ticker, exc)
            return []

        return self._parse_ats_data(ticker, data)

    def get_dark_pool_ratio(
        self,
        ticker: str,
        exchange_volume: int,
    ) -> Optional[float]:
        """Calculate dark pool volume as ratio of total exchange volume.

        Args:
            ticker: Stock ticker.
            exchange_volume: Total exchange volume for comparison.

        Returns:
            Ratio (0.0–1.0) or None if data unavailable.
        """
        if exchange_volume <= 0:
            return None

        ats_data = self.get_ats_data(ticker, weeks_back=1)
        if not ats_data:
            return None

        dp_volume = ats_data[0].total_shares
        total = dp_volume + exchange_volume
        return dp_volume / total if total > 0 else None

    def detect_accumulation(
        self,
        ticker: str,
        weeks_back: int = 8,
        threshold_pct: float = 20.0,
    ) -> Dict[str, Any]:
        """Detect institutional accumulation via rising dark pool volume.

        Compares recent dark pool volume to historical average.

        Args:
            ticker: Stock ticker.
            weeks_back: Historical window.
            threshold_pct: Minimum % increase above average to flag.

        Returns:
            Dict with ``detected``, ``current_volume``, ``avg_volume``,
            ``change_pct``.
        """
        ats_data = self.get_ats_data(ticker, weeks_back=weeks_back)

        if len(ats_data) < 3:
            return {
                "detected": False,
                "current_volume": 0,
                "avg_volume": 0,
                "change_pct": 0.0,
            }

        current = ats_data[0].total_shares
        historical = ats_data[1:]
        avg_volume = sum(d.total_shares for d in historical) / len(historical)

        if avg_volume == 0:
            change_pct = 0.0
        else:
            change_pct = ((current - avg_volume) / avg_volume) * 100

        return {
            "detected": change_pct > threshold_pct,
            "current_volume": current,
            "avg_volume": avg_volume,
            "change_pct": round(change_pct, 2),
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_ats_data(
        ticker: str, data: Any
    ) -> List[DarkPoolData]:
        """Parse FINRA ATS JSON response."""
        results: List[DarkPoolData] = []

        items = data if isinstance(data, list) else []

        for item in items:
            try:
                week_str = item.get("weekStartDate", "")
                if isinstance(week_str, str) and week_str:
                    week_date = datetime.strptime(week_str, "%Y-%m-%d")
                else:
                    continue

                shares = int(item.get("totalWeeklyShareQuantity", 0))
                trades = int(item.get("totalWeeklyTradeCount", 0))

                results.append(DarkPoolData(
                    ticker=ticker.upper(),
                    week_ending=week_date,
                    total_shares=shares,
                    total_trades=trades,
                ))
            except (ValueError, TypeError) as exc:
                logger.debug("Skipping malformed ATS entry: %s", exc)
                continue

        return results
