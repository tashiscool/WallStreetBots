"""
News Aggregator

Collects articles / posts from multiple sentiment data sources,
deduplicates by fuzzy title matching, and enforces per-source rate
limits so that upstream APIs are not hammered.
"""

import logging
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Minimum seconds between collections from the same source
_DEFAULT_RATE_LIMIT_SECS = 60.0


class NewsAggregator:
    """
    Orchestrates collection from several data sources (Reddit, Twitter,
    SEC EDGAR, etc.) and returns a deduplicated, merged list of articles.

    Deduplication uses a simple substring check on normalised titles:
    if one title is a substring of another (or they share >60 % of
    tokens) the later duplicate is dropped.

    Rate limiting prevents calling the same source more than once every
    ``rate_limit_seconds`` seconds.
    """

    def __init__(
        self,
        sources: Optional[List[Any]] = None,
        rate_limit_seconds: float = _DEFAULT_RATE_LIMIT_SECS,
    ) -> None:
        """
        Args:
            sources: Iterable of source instances. Each must have either
                ``get_posts(subreddits, limit)`` or ``search_tweets(query, limit)``
                or ``get_recent_filings(ticker, ...)``.
            rate_limit_seconds: Minimum interval between calls to the same source.
        """
        self._sources: List[Any] = list(sources or [])
        self._rate_limit = rate_limit_seconds
        # source id -> last call timestamp
        self._last_call: Dict[int, float] = {}
        # source id -> cached results
        self._cache: Dict[int, List[Dict[str, Any]]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_source(self, source: Any) -> None:
        """Register an additional data source."""
        self._sources.append(source)

    def collect(
        self,
        symbols: List[str],
        limit_per_source: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Collect articles/posts for the given *symbols* from all sources.

        Returns a deduplicated list of article dicts.
        """
        all_items: List[Dict[str, Any]] = []

        for source in self._sources:
            items = self._collect_from_source(source, symbols, limit_per_source)
            all_items.extend(items)

        deduplicated = self._deduplicate(all_items)
        logger.info(
            "Aggregator collected %d items (%d after dedup) for %s",
            len(all_items),
            len(deduplicated),
            symbols,
        )
        return deduplicated

    # ------------------------------------------------------------------
    # Per-source collection with rate limiting
    # ------------------------------------------------------------------

    def _collect_from_source(
        self,
        source: Any,
        symbols: List[str],
        limit: int,
    ) -> List[Dict[str, Any]]:
        """Fetch from a single source, respecting the rate limit."""
        source_id = id(source)
        now = time.monotonic()

        last = self._last_call.get(source_id, 0.0)
        if now - last < self._rate_limit:
            logger.debug("Rate-limited; returning cached results for source %s", type(source).__name__)
            return list(self._cache.get(source_id, []))

        items: List[Dict[str, Any]] = []

        try:
            # Reddit-like source
            if hasattr(source, "get_posts"):
                items = source.get_posts(limit=limit)

            # Twitter-like source
            elif hasattr(source, "search_tweets"):
                for sym in symbols:
                    cashtag = f"${sym.upper()}"
                    items.extend(source.search_tweets(query=cashtag, limit=limit))

            # SEC EDGAR-like source
            elif hasattr(source, "get_recent_filings"):
                for sym in symbols:
                    items.extend(source.get_recent_filings(ticker=sym, limit=limit))

            else:
                logger.warning("Unknown source type: %s", type(source).__name__)

        except Exception:
            logger.exception("Error collecting from %s", type(source).__name__)

        self._last_call[source_id] = now
        self._cache[source_id] = items
        return items

    # ------------------------------------------------------------------
    # Deduplication
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise(text: str) -> str:
        """Lower-case, strip whitespace, collapse spaces."""
        return " ".join(text.lower().split())

    @classmethod
    def _are_duplicates(cls, title_a: str, title_b: str) -> bool:
        """
        Fuzzy duplicate check.

        Returns True if one normalised title is a substring of the other
        or they share more than 60 % of their tokens.
        """
        a = cls._normalise(title_a)
        b = cls._normalise(title_b)

        if not a or not b:
            return False

        # Substring check
        if a in b or b in a:
            return True

        # Token overlap check
        tokens_a = set(a.split())
        tokens_b = set(b.split())
        if not tokens_a or not tokens_b:
            return False

        overlap = len(tokens_a & tokens_b)
        min_len = min(len(tokens_a), len(tokens_b))
        if min_len > 0 and overlap / min_len > 0.6:
            return True

        return False

    @classmethod
    def _deduplicate(cls, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate items based on fuzzy title matching."""
        unique: List[Dict[str, Any]] = []
        seen_titles: List[str] = []

        for item in items:
            title = item.get("title") or item.get("text", "")
            if not title:
                unique.append(item)
                continue

            is_dup = False
            for existing_title in seen_titles:
                if cls._are_duplicates(title, existing_title):
                    is_dup = True
                    break

            if not is_dup:
                unique.append(item)
                seen_titles.append(title)

        return unique
