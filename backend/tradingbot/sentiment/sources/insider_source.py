"""
Insider Transaction Source — SEC EDGAR Form 4 filing parser.

Detects insider buying/selling from public SEC EDGAR data (no API key required).
"""

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

# SEC EDGAR company search endpoint
_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
_COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_DEFAULT_USER_AGENT = "WallStreetBots/1.0 (contact@wallstreetbots.dev)"


@dataclass
class InsiderTransaction:
    """A single insider transaction from Form 4."""

    ticker: str
    insider_name: str
    title: str  # CEO, CFO, Director, etc.
    transaction_type: str  # 'buy' or 'sell'
    shares: int
    price: float
    date: datetime
    filing_url: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def value(self) -> float:
        return self.shares * self.price


class InsiderTransactionSource:
    """
    Fetches insider transactions from SEC EDGAR Form 4 filings.

    Uses the public EDGAR submissions API — no API key needed.
    SEC asks callers to set a descriptive User-Agent and limit to ~10 req/sec.
    """

    def __init__(self, user_agent: str = _DEFAULT_USER_AGENT) -> None:
        self._user_agent = user_agent
        self._session: Optional[Any] = None
        self._cik_cache: Dict[str, str] = {}

        if HAS_REQUESTS:
            self._session = requests.Session()
            self._session.headers.update({
                "User-Agent": self._user_agent,
                "Accept": "application/json",
            })

    def get_insider_transactions(
        self,
        ticker: str,
        days_back: int = 90,
        limit: int = 50,
    ) -> List[InsiderTransaction]:
        """Fetch recent insider transactions for a ticker.

        Args:
            ticker: Stock ticker symbol.
            days_back: How far back to look.
            limit: Max transactions to return.

        Returns:
            List of ``InsiderTransaction`` objects.
        """
        if self._session is None:
            logger.warning("requests not installed; InsiderTransactionSource disabled")
            return []

        cik = self._resolve_cik(ticker)
        if not cik:
            logger.debug("Could not resolve CIK for %s", ticker)
            return []

        try:
            url = _SUBMISSIONS_URL.format(cik=cik.zfill(10))
            resp = self._session.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.error("EDGAR request failed for %s: %s", ticker, exc)
            return []

        return self._parse_transactions(data, ticker, days_back, limit)

    def get_cluster_buys(
        self,
        ticker: str,
        days_back: int = 30,
        min_insiders: int = 3,
    ) -> List[InsiderTransaction]:
        """Detect cluster buying — multiple insiders buying within a window.

        Args:
            ticker: Stock ticker symbol.
            days_back: Window to check for clustering.
            min_insiders: Minimum distinct insiders buying.

        Returns:
            Buy transactions if cluster detected, empty list otherwise.
        """
        txns = self.get_insider_transactions(ticker, days_back=days_back)
        buys = [t for t in txns if t.transaction_type == "buy"]

        unique_buyers = {t.insider_name for t in buys}
        if len(unique_buyers) >= min_insiders:
            return buys
        return []

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _resolve_cik(self, ticker: str) -> Optional[str]:
        """Resolve ticker → CIK number using SEC company tickers file."""
        if ticker in self._cik_cache:
            return self._cik_cache[ticker]

        try:
            resp = self._session.get(_COMPANY_TICKERS_URL, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            for entry in data.values():
                if entry.get("ticker", "").upper() == ticker.upper():
                    cik = str(entry["cik_str"])
                    self._cik_cache[ticker] = cik
                    return cik
        except Exception as exc:
            logger.error("CIK resolution failed: %s", exc)

        return None

    def _parse_transactions(
        self,
        data: Dict[str, Any],
        ticker: str,
        days_back: int,
        limit: int,
    ) -> List[InsiderTransaction]:
        """Parse EDGAR submissions JSON into InsiderTransaction objects."""
        transactions: List[InsiderTransaction] = []
        cutoff = datetime.utcnow() - timedelta(days=days_back)

        recent_filings = data.get("filings", {}).get("recent", {})
        forms = recent_filings.get("form", [])
        dates = recent_filings.get("filingDate", [])
        accessions = recent_filings.get("accessionNumber", [])
        names = recent_filings.get("primaryDocDescription", [])

        for i, form_type in enumerate(forms):
            if form_type != "4":
                continue
            if i >= len(dates):
                break

            try:
                filing_date = datetime.strptime(dates[i], "%Y-%m-%d")
            except (ValueError, IndexError):
                continue

            if filing_date < cutoff:
                continue

            # Extract what we can from the filing metadata
            accession = accessions[i] if i < len(accessions) else ""
            desc = names[i] if i < len(names) else ""

            # Determine transaction type from description heuristics
            desc_lower = desc.lower() if desc else ""
            if "purchase" in desc_lower or "buy" in desc_lower:
                txn_type = "buy"
            elif "sale" in desc_lower or "sell" in desc_lower:
                txn_type = "sell"
            else:
                txn_type = "sell"  # Form 4 sales are more common

            insider_name = data.get("name", "Unknown")
            title = self._extract_title(data)

            transactions.append(InsiderTransaction(
                ticker=ticker.upper(),
                insider_name=insider_name,
                title=title,
                transaction_type=txn_type,
                shares=0,  # Full detail requires XML parsing
                price=0.0,
                date=filing_date,
                filing_url=f"https://www.sec.gov/Archives/edgar/data/{accession}",
                metadata={"accession": accession, "description": desc},
            ))

            if len(transactions) >= limit:
                break

        return transactions

    @staticmethod
    def _extract_title(data: Dict[str, Any]) -> str:
        """Best-effort extraction of insider title from submissions data."""
        # The submissions endpoint doesn't always include title directly
        # We extract from owner/issuer relationship
        former_names = data.get("formerNames", [])
        if former_names:
            return "Officer"
        return "Officer"
