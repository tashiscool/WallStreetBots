"""
SEC EDGAR Source

Fetches recent filings from the SEC EDGAR full-text search API
(efts.sec.gov).  Uses only the ``requests`` library -- no optional
dependencies.
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import requests

    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

# SEC EDGAR full-text search endpoint
_EFTS_BASE_URL = "https://efts.sec.gov/LATEST/search-index"

# Friendly company-filings search endpoint (JSON)
_EDGAR_SEARCH_URL = "https://efts.sec.gov/LATEST/search-index"
_EDGAR_FULL_TEXT_URL = "https://efts.sec.gov/LATEST/search-index"

# Actually the most reliable public API endpoint:
_EDGAR_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"

# Full-text search
_EDGAR_FTS_URL = "https://efts.sec.gov/LATEST/search-index"

# We use the newer EDGAR full-text search API
_SEARCH_URL = "https://efts.sec.gov/LATEST/search-index"

# SEC requires a User-Agent header identifying the requester
_DEFAULT_USER_AGENT = "WallStreetBots/1.0 (contact@wallstreetbots.dev)"

# EDGAR full-text search API (the one that actually works well for keyword search)
EDGAR_FULLTEXT_SEARCH = "https://efts.sec.gov/LATEST/search-index"

# Use the simpler, publicly documented endpoint
EDGAR_SEARCH_API = "https://efts.sec.gov/LATEST/search-index"


class SECEdgarSource:
    """
    Fetches recent SEC filings (8-K, 10-K, 10-Q, etc.) using the
    public EDGAR full-text search API.

    No API key is required.  The SEC asks that callers set a descriptive
    User-Agent and limit request frequency to ~10 req/sec.
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
        else:
            logger.warning(
                "requests library not installed; SECEdgarSource disabled. "
                "Install with: pip install requests"
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_recent_filings(
        self,
        ticker: str,
        filing_types: Optional[List[str]] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Fetch recent filings for *ticker* from SEC EDGAR.

        Args:
            ticker: Company ticker symbol (e.g. ``AAPL``).
            filing_types: Filing form types to include (default ``['8-K']``).
            limit: Maximum number of filings to return.

        Returns:
            List of dicts with keys: ``filing_type``, ``date``,
            ``description``, ``url``, ``company_name``, ``source``.
        """
        if filing_types is None:
            filing_types = ["8-K"]

        if self._session is None:
            return []

        filings: List[Dict[str, Any]] = []

        for form_type in filing_types:
            fetched = self._search_filings(ticker, form_type, limit)
            filings.extend(fetched)

        # Sort by date descending and cap at limit
        filings.sort(key=lambda f: f.get("date", ""), reverse=True)
        return filings[:limit]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _search_filings(
        self,
        ticker: str,
        form_type: str,
        limit: int,
    ) -> List[Dict[str, Any]]:
        """Query EDGAR full-text search for a specific form type."""
        url = "https://efts.sec.gov/LATEST/search-index"
        params: Dict[str, Any] = {
            "q": f'"{ticker}"',
            "dateRange": "custom",
            "forms": form_type,
            "from": 0,
            "size": limit,
        }

        try:
            resp = self._session.get(url, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            # Fallback: try the documented search API
            return self._search_filings_fallback(ticker, form_type, limit)

        results: List[Dict[str, Any]] = []
        hits = data.get("hits", {}).get("hits", [])

        for hit in hits:
            source = hit.get("_source", {})
            filing_url = self._build_filing_url(source)

            results.append({
                "filing_type": source.get("form_type", form_type),
                "date": source.get("file_date", source.get("period_of_report", "")),
                "description": source.get("display_names", [ticker])[0] if source.get("display_names") else ticker,
                "url": filing_url,
                "company_name": source.get("entity_name", ""),
                "ticker": ticker,
                "source": "sec_edgar",
            })

        return results

    def _search_filings_fallback(
        self,
        ticker: str,
        form_type: str,
        limit: int,
    ) -> List[Dict[str, Any]]:
        """Fallback using the documented EDGAR full-text search API."""
        url = "https://efts.sec.gov/LATEST/search-index"
        params = {
            "q": ticker,
            "forms": form_type,
        }

        try:
            resp = self._session.get(url, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            logger.exception("EDGAR fallback search failed for %s %s", ticker, form_type)
            return []

        results: List[Dict[str, Any]] = []
        for filing in data.get("filings", []):
            results.append({
                "filing_type": filing.get("type", form_type),
                "date": filing.get("dateFiled", ""),
                "description": filing.get("title", ""),
                "url": filing.get("url", ""),
                "company_name": filing.get("companyName", ""),
                "ticker": ticker,
                "source": "sec_edgar",
            })

        return results[:limit]

    @staticmethod
    def _build_filing_url(source: Dict[str, Any]) -> str:
        """Construct an EDGAR filing URL from search-index _source data."""
        file_num = source.get("file_num", "")
        accession = source.get("accession_no", "")
        if accession:
            clean = accession.replace("-", "")
            return f"https://www.sec.gov/Archives/edgar/data/{file_num}/{clean}/{accession}-index.htm"
        return ""
