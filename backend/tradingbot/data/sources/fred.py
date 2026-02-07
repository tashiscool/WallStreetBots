"""
FRED Data Source — Federal Reserve Economic Data.

Free API (key required via ``FRED_API_KEY`` env var).
Series: yield curve, unemployment, GDP, Fed funds rate, VIX, CPI.
"""

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import requests

    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

_FRED_BASE_URL = "https://api.stlouisfed.org/fred"

# Key FRED series IDs
FRED_SERIES = {
    "DGS10": "10-Year Treasury Constant Maturity Rate",
    "DGS2": "2-Year Treasury Constant Maturity Rate",
    "T10Y2Y": "10-Year Treasury Minus 2-Year (spread)",
    "UNRATE": "Unemployment Rate",
    "GDP": "Gross Domestic Product",
    "FEDFUNDS": "Effective Federal Funds Rate",
    "VIXCLS": "CBOE Volatility Index (VIX)",
    "CPIAUCSL": "Consumer Price Index for All Urban Consumers",
}


@dataclass
class FREDObservation:
    """A single FRED data observation."""

    series_id: str
    date: datetime
    value: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class FREDDataSource:
    """
    Federal Reserve Economic Data provider.

    Requires a free API key from https://fred.stlouisfed.org/docs/api/api_key.html.
    Set via ``FRED_API_KEY`` environment variable.
    """

    def __init__(self, api_key: Optional[str] = None) -> None:
        self.api_key = api_key or os.environ.get("FRED_API_KEY", "")
        self._session: Optional[Any] = None

        if HAS_REQUESTS and self.api_key:
            self._session = requests.Session()
        elif not self.api_key:
            logger.warning(
                "FRED_API_KEY not set. FREDDataSource disabled. "
                "Get a free key at https://fred.stlouisfed.org/docs/api/api_key.html"
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_series(
        self,
        series_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[FREDObservation]:
        """Fetch observations for a FRED series.

        Args:
            series_id: FRED series ID (e.g. 'DGS10', 'UNRATE').
            start_date: Start of observation window.
            end_date: End of observation window.
            limit: Max observations.

        Returns:
            List of ``FREDObservation`` objects.
        """
        if self._session is None:
            return []

        params: Dict[str, Any] = {
            "series_id": series_id,
            "api_key": self.api_key,
            "file_type": "json",
            "sort_order": "desc",
            "limit": limit,
        }
        if start_date:
            params["observation_start"] = start_date.strftime("%Y-%m-%d")
        if end_date:
            params["observation_end"] = end_date.strftime("%Y-%m-%d")

        try:
            resp = self._session.get(
                f"{_FRED_BASE_URL}/series/observations",
                params=params,
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.error("FRED request failed for %s: %s", series_id, exc)
            return []

        return self._parse_observations(series_id, data)

    def get_yield_curve(self) -> Dict[str, Optional[float]]:
        """Get current yield curve data points.

        Returns:
            Dict with keys ``dgs2``, ``dgs10``, ``spread``, ``inverted``.
        """
        dgs2 = self._latest_value("DGS2")
        dgs10 = self._latest_value("DGS10")

        spread = None
        inverted = False
        if dgs2 is not None and dgs10 is not None:
            spread = dgs10 - dgs2
            inverted = spread < 0

        return {
            "dgs2": dgs2,
            "dgs10": dgs10,
            "spread": spread,
            "inverted": inverted,
        }

    def is_yield_curve_inverted(self) -> bool:
        """Check if the yield curve is currently inverted (2Y > 10Y)."""
        curve = self.get_yield_curve()
        return bool(curve.get("inverted", False))

    def get_macro_snapshot(self) -> Dict[str, Optional[float]]:
        """Get a snapshot of key macroeconomic indicators.

        Returns:
            Dict with latest values for unemployment, Fed funds, VIX, CPI.
        """
        return {
            "unemployment": self._latest_value("UNRATE"),
            "fed_funds": self._latest_value("FEDFUNDS"),
            "vix": self._latest_value("VIXCLS"),
            "cpi": self._latest_value("CPIAUCSL"),
            "yield_spread": self.get_yield_curve().get("spread"),
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _latest_value(self, series_id: str) -> Optional[float]:
        """Fetch the most recent observation for a series."""
        obs = self.get_series(series_id, limit=1)
        if obs:
            return obs[0].value
        return None

    @staticmethod
    def _parse_observations(
        series_id: str, data: Dict[str, Any]
    ) -> List[FREDObservation]:
        """Parse FRED API JSON into FREDObservation list."""
        observations: List[FREDObservation] = []

        for item in data.get("observations", []):
            raw_value = item.get("value", ".")
            if raw_value == ".":
                continue  # FRED uses "." for missing data

            try:
                value = float(raw_value)
            except (ValueError, TypeError):
                continue

            try:
                obs_date = datetime.strptime(item["date"], "%Y-%m-%d")
            except (ValueError, KeyError):
                continue

            observations.append(FREDObservation(
                series_id=series_id,
                date=obs_date,
                value=value,
            ))

        return observations
