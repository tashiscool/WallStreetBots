"""Sentiment data source connectors."""

from .reddit_source import RedditSource
from .twitter_source import TwitterSource
from .sec_edgar_source import SECEdgarSource
from .aggregator import NewsAggregator

__all__ = [
    'RedditSource',
    'TwitterSource',
    'SECEdgarSource',
    'NewsAggregator',
]
