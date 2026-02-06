"""
NLP Sentiment Analysis Engine.

Provides multi-source sentiment scoring for trading signal generation.
"""

from .scoring.base import SentimentScorer, SentimentScore
from .scoring.ensemble_scorer import EnsembleSentimentScorer, ScoringMode
from .sources.aggregator import NewsAggregator
from .pipeline import SentimentPipeline

__all__ = [
    'SentimentScorer',
    'SentimentScore',
    'EnsembleSentimentScorer',
    'ScoringMode',
    'NewsAggregator',
    'SentimentPipeline',
]
