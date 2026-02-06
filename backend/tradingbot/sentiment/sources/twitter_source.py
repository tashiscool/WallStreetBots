"""
Twitter / X Source

Collects tweets mentioning financial tickers or cashtags using the
tweepy library.  tweepy is an optional dependency; when it is not
installed this source returns empty results gracefully.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import tweepy  # noqa: F401

    HAS_TWEEPY = True
except ImportError:
    HAS_TWEEPY = False


class TwitterSource:
    """
    Searches for tweets referencing cashtags (e.g. ``$AAPL``) using
    the Twitter v2 API through tweepy.
    """

    def __init__(
        self,
        bearer_token: str = "",
        api_key: str = "",
        api_secret: str = "",
        access_token: str = "",
        access_secret: str = "",
    ) -> None:
        """
        Provide **either** a bearer_token (app-only auth) or the full
        set of OAuth1 credentials.
        """
        self._client: Optional[Any] = None

        if not HAS_TWEEPY:
            logger.warning("tweepy not installed; TwitterSource is disabled. Install with: pip install tweepy")
            return

        try:
            if bearer_token:
                self._client = tweepy.Client(bearer_token=bearer_token)
            elif api_key and api_secret and access_token and access_secret:
                self._client = tweepy.Client(
                    consumer_key=api_key,
                    consumer_secret=api_secret,
                    access_token=access_token,
                    access_token_secret=access_secret,
                )
            else:
                logger.warning("TwitterSource: no credentials provided; source will be inactive.")
        except Exception:
            logger.exception("Failed to initialise tweepy client.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search_tweets(
        self,
        query: str,
        limit: int = 25,
    ) -> List[Dict[str, Any]]:
        """
        Search recent tweets matching *query*.

        A typical *query* is a cashtag such as ``$AAPL`` or a
        space-separated list of cashtags.

        Args:
            query: Twitter search query string.
            limit: Maximum number of tweets to return (max 100 per API page).

        Returns:
            List of dicts, each with keys:
              ``text``, ``created_at``, ``user``, ``retweet_count``, ``url``,
              ``source``.
        """
        if self._client is None:
            logger.debug("TwitterSource: no client configured; returning empty list.")
            return []

        tweets: List[Dict[str, Any]] = []

        try:
            response = self._client.search_recent_tweets(
                query=query,
                max_results=min(limit, 100),
                tweet_fields=["created_at", "public_metrics", "author_id"],
            )

            if response.data is None:
                return []

            for tweet in response.data:
                metrics = tweet.public_metrics or {}
                created_at = tweet.created_at
                if isinstance(created_at, datetime):
                    created_at_str = created_at.isoformat()
                elif created_at is not None:
                    created_at_str = str(created_at)
                else:
                    created_at_str = datetime.now(timezone.utc).isoformat()

                tweets.append({
                    "text": tweet.text,
                    "created_at": created_at_str,
                    "user": str(tweet.author_id or ""),
                    "retweet_count": metrics.get("retweet_count", 0),
                    "url": f"https://twitter.com/i/web/status/{tweet.id}",
                    "source": "twitter",
                })
        except Exception:
            logger.exception("Error searching tweets for query: %s", query)

        return tweets

    def search_symbol(self, symbol: str, limit: int = 25) -> List[Dict[str, Any]]:
        """Convenience: search for cashtag ``$SYMBOL``."""
        cashtag = f"${symbol.upper()}"
        return self.search_tweets(query=cashtag, limit=limit)
