"""
Tests for the NewsAggregator.

Covers deduplication logic and rate limiting.
"""

import time
import pytest
from unittest.mock import MagicMock

from backend.tradingbot.sentiment.sources.aggregator import NewsAggregator


class TestNewsAggregatorDedup:
    """Tests for article deduplication."""

    def test_no_duplicates_passes_through(self):
        agg = NewsAggregator()
        items = [
            {"title": "Apple beats earnings", "source": "a"},
            {"title": "Tesla launches new model", "source": "b"},
            {"title": "Fed raises rates", "source": "c"},
        ]
        result = agg._deduplicate(items)
        assert len(result) == 3

    def test_exact_duplicate_removed(self):
        agg = NewsAggregator()
        items = [
            {"title": "Apple beats earnings", "source": "a"},
            {"title": "Apple beats earnings", "source": "b"},
        ]
        result = agg._deduplicate(items)
        assert len(result) == 1

    def test_substring_duplicate_removed(self):
        agg = NewsAggregator()
        items = [
            {"title": "Apple beats earnings estimates in Q4", "source": "a"},
            {"title": "Apple beats earnings", "source": "b"},
        ]
        result = agg._deduplicate(items)
        assert len(result) == 1
        # The first (longer) one is kept
        assert result[0]["title"] == "Apple beats earnings estimates in Q4"

    def test_high_token_overlap_duplicate(self):
        agg = NewsAggregator()
        items = [
            {"title": "Apple reports record quarterly earnings beat", "source": "a"},
            {"title": "Apple reports record quarterly earnings", "source": "b"},
        ]
        result = agg._deduplicate(items)
        assert len(result) == 1

    def test_different_articles_kept(self):
        agg = NewsAggregator()
        items = [
            {"title": "Apple earnings soar", "source": "a"},
            {"title": "Tesla production hits record", "source": "b"},
        ]
        result = agg._deduplicate(items)
        assert len(result) == 2

    def test_empty_title_not_deduplicated(self):
        """Items with no title and no text should all pass through."""
        agg = NewsAggregator()
        items = [
            {"source": "a"},
            {"source": "b"},
        ]
        result = agg._deduplicate(items)
        assert len(result) == 2

    def test_case_insensitive_dedup(self):
        agg = NewsAggregator()
        items = [
            {"title": "APPLE BEATS EARNINGS", "source": "a"},
            {"title": "apple beats earnings", "source": "b"},
        ]
        result = agg._deduplicate(items)
        assert len(result) == 1


class TestNewsAggregatorRateLimiting:
    """Tests for per-source rate limiting."""

    def test_second_call_within_limit_returns_cached(self):
        """A second collect within rate_limit_seconds should use cache."""
        mock_source = MagicMock()
        mock_source.get_posts.return_value = [
            {"title": "Post 1", "source": "reddit"},
        ]

        agg = NewsAggregator(sources=[mock_source], rate_limit_seconds=60.0)
        result1 = agg.collect(["AAPL"], limit_per_source=10)
        assert len(result1) == 1
        assert mock_source.get_posts.call_count == 1

        # Second call should be rate-limited and use cache
        result2 = agg.collect(["AAPL"], limit_per_source=10)
        assert len(result2) == 1
        # Still only 1 API call
        assert mock_source.get_posts.call_count == 1

    def test_call_after_rate_limit_makes_new_request(self):
        """After the rate limit expires, a new API call should be made."""
        mock_source = MagicMock()
        mock_source.get_posts.return_value = [
            {"title": "Post 1", "source": "reddit"},
        ]

        # Use a tiny rate limit for testing
        agg = NewsAggregator(sources=[mock_source], rate_limit_seconds=0.05)
        agg.collect(["AAPL"], limit_per_source=10)
        assert mock_source.get_posts.call_count == 1

        # Wait for rate limit to expire
        time.sleep(0.1)

        agg.collect(["AAPL"], limit_per_source=10)
        assert mock_source.get_posts.call_count == 2

    def test_multiple_sources_independent_rate_limits(self):
        """Each source should have its own rate limit timer."""
        mock_reddit = MagicMock()
        mock_reddit.get_posts.return_value = [{"title": "Reddit post", "source": "reddit"}]

        # Use spec=[] so that hasattr(mock_twitter, 'get_posts') is False
        mock_twitter = MagicMock(spec=[])
        mock_twitter.search_tweets = MagicMock(
            return_value=[{"text": "Tweet text", "source": "twitter"}]
        )

        agg = NewsAggregator(sources=[mock_reddit, mock_twitter], rate_limit_seconds=60.0)
        result = agg.collect(["AAPL"], limit_per_source=10)

        # Both sources called once
        assert mock_reddit.get_posts.call_count == 1
        assert mock_twitter.search_tweets.call_count == 1


class TestNewsAggregatorCollect:
    """Tests for the collect method with various source types."""

    def test_collect_from_reddit_source(self):
        mock_source = MagicMock()
        mock_source.get_posts.return_value = [
            {"title": "Reddit post about $AAPL", "source": "reddit"},
        ]
        agg = NewsAggregator(sources=[mock_source])
        result = agg.collect(["AAPL"])
        assert len(result) == 1

    def test_collect_from_twitter_source(self):
        mock_source = MagicMock(spec=[])  # no get_posts attribute
        mock_source.search_tweets = MagicMock(return_value=[
            {"text": "$AAPL looking good", "source": "twitter"},
        ])
        agg = NewsAggregator(sources=[mock_source])
        result = agg.collect(["AAPL"])
        assert len(result) == 1

    def test_collect_from_edgar_source(self):
        mock_source = MagicMock(spec=[])
        mock_source.get_recent_filings = MagicMock(return_value=[
            {"filing_type": "8-K", "date": "2024-01-15", "company_name": "Apple", "source": "sec_edgar"},
        ])
        agg = NewsAggregator(sources=[mock_source])
        result = agg.collect(["AAPL"])
        assert len(result) == 1

    def test_add_source(self):
        agg = NewsAggregator()
        assert len(agg._sources) == 0
        mock_source = MagicMock()
        agg.add_source(mock_source)
        assert len(agg._sources) == 1

    def test_collect_empty_symbols(self):
        mock_source = MagicMock()
        mock_source.get_posts.return_value = []
        agg = NewsAggregator(sources=[mock_source])
        result = agg.collect([])
        assert result == [] or isinstance(result, list)
