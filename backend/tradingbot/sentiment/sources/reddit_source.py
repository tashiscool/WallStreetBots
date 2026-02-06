"""
Reddit Source

Collects posts and comments from financial subreddits using the PRAW
(Python Reddit API Wrapper) library.  PRAW is an optional dependency;
when it is not installed this source returns empty results gracefully.
"""

import logging
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import praw  # noqa: F401

    HAS_PRAW = True
except ImportError:
    HAS_PRAW = False

# Default subreddits to monitor
DEFAULT_SUBREDDITS: List[str] = [
    "wallstreetbets",
    "stocks",
    "investing",
    "options",
]

# Regex for cashtag mentions (e.g. $AAPL, $TSLA)
_CASHTAG_RE = re.compile(r"\$([A-Z]{1,5})\b")


class RedditSource:
    """
    Collects posts from financial subreddits via PRAW.

    Posts are returned as plain dicts so downstream consumers do not
    need to depend on PRAW types.
    """

    def __init__(
        self,
        client_id: str = "",
        client_secret: str = "",
        user_agent: str = "WallStreetBots/1.0",
        subreddits: Optional[List[str]] = None,
    ) -> None:
        """
        Args:
            client_id: Reddit API client id.
            client_secret: Reddit API client secret.
            user_agent: User agent string for the Reddit API.
            subreddits: Subreddits to monitor (defaults to DEFAULT_SUBREDDITS).
        """
        self.subreddits = subreddits or list(DEFAULT_SUBREDDITS)
        self._reddit: Optional[Any] = None

        if HAS_PRAW and client_id and client_secret:
            try:
                self._reddit = praw.Reddit(
                    client_id=client_id,
                    client_secret=client_secret,
                    user_agent=user_agent,
                )
                logger.info("RedditSource initialised with PRAW (read-only).")
            except Exception:
                logger.exception("Failed to initialise PRAW Reddit instance.")
        elif not HAS_PRAW:
            logger.warning("PRAW not installed; RedditSource is disabled. Install with: pip install praw")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_posts(
        self,
        subreddits: Optional[List[str]] = None,
        limit: int = 25,
    ) -> List[Dict[str, Any]]:
        """
        Fetch recent hot posts from the specified subreddits.

        Args:
            subreddits: Override subreddits list (uses instance default if *None*).
            limit: Maximum posts to fetch **per subreddit**.

        Returns:
            List of dicts, each with keys:
              ``title``, ``text``, ``score``, ``created_utc``, ``url``, ``subreddit``,
              ``mentioned_tickers``.
        """
        target_subs = subreddits or self.subreddits

        if self._reddit is None:
            logger.debug("RedditSource: no PRAW client; returning empty list.")
            return []

        posts: List[Dict[str, Any]] = []

        for sub_name in target_subs:
            try:
                subreddit = self._reddit.subreddit(sub_name)
                for submission in subreddit.hot(limit=limit):
                    body = submission.selftext or ""
                    full_text = f"{submission.title} {body}"
                    tickers = _CASHTAG_RE.findall(full_text)

                    posts.append({
                        "title": submission.title,
                        "text": body,
                        "score": submission.score,
                        "created_utc": datetime.fromtimestamp(
                            submission.created_utc, tz=timezone.utc
                        ).isoformat(),
                        "url": f"https://reddit.com{submission.permalink}",
                        "subreddit": sub_name,
                        "mentioned_tickers": list(set(tickers)),
                        "source": "reddit",
                    })
            except Exception:
                logger.exception("Error fetching posts from r/%s", sub_name)

        return posts

    def get_posts_for_symbol(
        self,
        symbol: str,
        subreddits: Optional[List[str]] = None,
        limit: int = 25,
    ) -> List[Dict[str, Any]]:
        """
        Convenience wrapper: fetch posts then filter to those mentioning *symbol*.
        """
        all_posts = self.get_posts(subreddits=subreddits, limit=limit)
        sym_upper = symbol.upper()
        return [
            p for p in all_posts
            if sym_upper in p.get("mentioned_tickers", [])
            or sym_upper in (p.get("title", "") + " " + p.get("text", "")).upper()
        ]
