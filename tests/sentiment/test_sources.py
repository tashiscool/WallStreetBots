"""
Tests for sentiment data sources (Reddit, Twitter, SEC EDGAR).

All external API calls are mocked.
"""

import json
import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch, PropertyMock

from backend.tradingbot.sentiment.sources.reddit_source import RedditSource, DEFAULT_SUBREDDITS
from backend.tradingbot.sentiment.sources.twitter_source import TwitterSource
from backend.tradingbot.sentiment.sources.sec_edgar_source import SECEdgarSource


# =====================================================================
# RedditSource
# =====================================================================


class TestRedditSource:
    """Tests for RedditSource with mocked PRAW."""

    def test_no_client_returns_empty(self):
        """Without credentials, get_posts returns an empty list."""
        source = RedditSource()
        assert source.get_posts() == []

    def test_default_subreddits(self):
        source = RedditSource()
        assert source.subreddits == DEFAULT_SUBREDDITS

    def test_custom_subreddits(self):
        source = RedditSource(subreddits=["test_sub"])
        assert source.subreddits == ["test_sub"]

    @patch("backend.tradingbot.sentiment.sources.reddit_source.HAS_PRAW", True)
    def test_get_posts_with_mocked_praw(self):
        """Simulate PRAW returning posts."""
        source = RedditSource()

        # Build mock submissions
        mock_sub1 = MagicMock()
        mock_sub1.title = "$AAPL is surging today"
        mock_sub1.selftext = "Apple stock is up 5% on earnings beat"
        mock_sub1.score = 1500
        mock_sub1.created_utc = 1700000000.0
        mock_sub1.permalink = "/r/wallstreetbets/comments/abc123/aapl_surging"

        mock_sub2 = MagicMock()
        mock_sub2.title = "Market outlook for this week"
        mock_sub2.selftext = "Expecting bullish momentum"
        mock_sub2.score = 300
        mock_sub2.created_utc = 1700000100.0
        mock_sub2.permalink = "/r/stocks/comments/def456/market_outlook"

        # Mock subreddit.hot()
        mock_subreddit = MagicMock()
        mock_subreddit.hot.return_value = [mock_sub1, mock_sub2]

        # Mock reddit instance
        mock_reddit = MagicMock()
        mock_reddit.subreddit.return_value = mock_subreddit
        source._reddit = mock_reddit

        posts = source.get_posts(subreddits=["wallstreetbets"], limit=10)

        assert len(posts) == 2
        assert posts[0]["title"] == "$AAPL is surging today"
        assert posts[0]["source"] == "reddit"
        assert posts[0]["subreddit"] == "wallstreetbets"
        assert "AAPL" in posts[0]["mentioned_tickers"]
        assert posts[0]["score"] == 1500

    @patch("backend.tradingbot.sentiment.sources.reddit_source.HAS_PRAW", True)
    def test_get_posts_for_symbol(self):
        source = RedditSource()

        mock_sub = MagicMock()
        mock_sub.title = "$TSLA to the moon"
        mock_sub.selftext = "Tesla is looking good"
        mock_sub.score = 500
        mock_sub.created_utc = 1700000000.0
        mock_sub.permalink = "/r/wallstreetbets/comments/x/tsla"

        mock_other = MagicMock()
        mock_other.title = "General market discussion"
        mock_other.selftext = "SPY looking flat"
        mock_other.score = 100
        mock_other.created_utc = 1700000100.0
        mock_other.permalink = "/r/wallstreetbets/comments/y/market"

        mock_subreddit = MagicMock()
        mock_subreddit.hot.return_value = [mock_sub, mock_other]
        mock_reddit = MagicMock()
        mock_reddit.subreddit.return_value = mock_subreddit
        source._reddit = mock_reddit

        posts = source.get_posts_for_symbol("TSLA", subreddits=["wallstreetbets"])
        assert len(posts) == 1
        assert "TSLA" in posts[0]["title"]


# =====================================================================
# TwitterSource
# =====================================================================


class TestTwitterSource:
    """Tests for TwitterSource with mocked tweepy."""

    def test_no_client_returns_empty(self):
        source = TwitterSource()
        assert source.search_tweets("$AAPL") == []

    @patch("backend.tradingbot.sentiment.sources.twitter_source.HAS_TWEEPY", True)
    def test_search_tweets_with_mock(self):
        source = TwitterSource()

        mock_tweet1 = MagicMock()
        mock_tweet1.text = "$AAPL looking strong after earnings"
        mock_tweet1.created_at = datetime(2024, 1, 15, 12, 0, tzinfo=timezone.utc)
        mock_tweet1.author_id = "12345"
        mock_tweet1.public_metrics = {"retweet_count": 50, "like_count": 200}
        mock_tweet1.id = "9999"

        mock_tweet2 = MagicMock()
        mock_tweet2.text = "$AAPL might drop tomorrow"
        mock_tweet2.created_at = datetime(2024, 1, 15, 13, 0, tzinfo=timezone.utc)
        mock_tweet2.author_id = "67890"
        mock_tweet2.public_metrics = {"retweet_count": 10, "like_count": 30}
        mock_tweet2.id = "8888"

        mock_response = MagicMock()
        mock_response.data = [mock_tweet1, mock_tweet2]

        mock_client = MagicMock()
        mock_client.search_recent_tweets.return_value = mock_response
        source._client = mock_client

        tweets = source.search_tweets("$AAPL", limit=25)

        assert len(tweets) == 2
        assert tweets[0]["text"] == "$AAPL looking strong after earnings"
        assert tweets[0]["source"] == "twitter"
        assert tweets[0]["retweet_count"] == 50
        assert "twitter.com" in tweets[0]["url"]

    @patch("backend.tradingbot.sentiment.sources.twitter_source.HAS_TWEEPY", True)
    def test_search_tweets_no_data(self):
        source = TwitterSource()
        mock_response = MagicMock()
        mock_response.data = None
        mock_client = MagicMock()
        mock_client.search_recent_tweets.return_value = mock_response
        source._client = mock_client

        tweets = source.search_tweets("$AAPL")
        assert tweets == []

    @patch("backend.tradingbot.sentiment.sources.twitter_source.HAS_TWEEPY", True)
    def test_search_symbol_builds_cashtag(self):
        source = TwitterSource()
        mock_response = MagicMock()
        mock_response.data = None
        mock_client = MagicMock()
        mock_client.search_recent_tweets.return_value = mock_response
        source._client = mock_client

        source.search_symbol("aapl")
        mock_client.search_recent_tweets.assert_called_once()
        call_args = mock_client.search_recent_tweets.call_args
        assert call_args.kwargs.get("query") == "$AAPL" or call_args[1].get("query") == "$AAPL"


# =====================================================================
# SECEdgarSource
# =====================================================================


class TestSECEdgarSource:
    """Tests for SECEdgarSource with mocked requests."""

    def _sample_edgar_response(self):
        return {
            "hits": {
                "total": {"value": 2},
                "hits": [
                    {
                        "_source": {
                            "form_type": "8-K",
                            "file_date": "2024-01-15",
                            "entity_name": "Apple Inc.",
                            "display_names": ["AAPL"],
                            "file_num": "0000320193",
                            "accession_no": "0000320193-24-000001",
                        }
                    },
                    {
                        "_source": {
                            "form_type": "8-K",
                            "file_date": "2024-01-10",
                            "entity_name": "Apple Inc.",
                            "display_names": ["AAPL"],
                            "file_num": "0000320193",
                            "accession_no": "0000320193-24-000002",
                        }
                    },
                ],
            }
        }

    def test_no_requests_returns_empty(self):
        """If requests is missing (unlikely), returns empty."""
        source = SECEdgarSource()
        source._session = None
        assert source.get_recent_filings("AAPL") == []

    def test_get_recent_filings_with_mock(self):
        source = SECEdgarSource()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = self._sample_edgar_response()
        mock_response.raise_for_status = MagicMock()

        mock_session = MagicMock()
        mock_session.get.return_value = mock_response
        source._session = mock_session

        filings = source.get_recent_filings("AAPL", filing_types=["8-K"], limit=10)

        assert len(filings) == 2
        assert filings[0]["filing_type"] == "8-K"
        assert filings[0]["company_name"] == "Apple Inc."
        assert filings[0]["source"] == "sec_edgar"
        assert filings[0]["date"] == "2024-01-15"

    def test_filings_sorted_by_date_descending(self):
        source = SECEdgarSource()

        resp_data = {
            "hits": {
                "total": {"value": 2},
                "hits": [
                    {
                        "_source": {
                            "form_type": "8-K",
                            "file_date": "2024-01-10",
                            "entity_name": "Test Corp",
                            "file_num": "123",
                            "accession_no": "123-24-001",
                        }
                    },
                    {
                        "_source": {
                            "form_type": "8-K",
                            "file_date": "2024-01-15",
                            "entity_name": "Test Corp",
                            "file_num": "123",
                            "accession_no": "123-24-002",
                        }
                    },
                ],
            }
        }

        mock_response = MagicMock()
        mock_response.json.return_value = resp_data
        mock_response.raise_for_status = MagicMock()
        mock_session = MagicMock()
        mock_session.get.return_value = mock_response
        source._session = mock_session

        filings = source.get_recent_filings("TEST")
        # Most recent first
        assert filings[0]["date"] == "2024-01-15"
        assert filings[1]["date"] == "2024-01-10"

    def test_limit_caps_results(self):
        source = SECEdgarSource()

        hits = [
            {
                "_source": {
                    "form_type": "8-K",
                    "file_date": f"2024-01-{i:02d}",
                    "entity_name": "Corp",
                    "file_num": "1",
                    "accession_no": f"1-24-{i:03d}",
                }
            }
            for i in range(1, 11)
        ]
        resp_data = {"hits": {"total": {"value": 10}, "hits": hits}}
        mock_response = MagicMock()
        mock_response.json.return_value = resp_data
        mock_response.raise_for_status = MagicMock()
        mock_session = MagicMock()
        mock_session.get.return_value = mock_response
        source._session = mock_session

        filings = source.get_recent_filings("X", limit=3)
        assert len(filings) == 3

    def test_default_filing_types(self):
        """Default should query 8-K filings."""
        source = SECEdgarSource()
        mock_response = MagicMock()
        mock_response.json.return_value = {"hits": {"total": {"value": 0}, "hits": []}}
        mock_response.raise_for_status = MagicMock()
        mock_session = MagicMock()
        mock_session.get.return_value = mock_response
        source._session = mock_session

        source.get_recent_filings("AAPL")
        # Check that the request was made with forms=8-K
        call_kwargs = mock_session.get.call_args
        params = call_kwargs.kwargs.get("params") or call_kwargs[1].get("params", {})
        assert params.get("forms") == "8-K"
