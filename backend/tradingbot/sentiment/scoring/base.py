"""
Base Sentiment Scorer

Defines the abstract interface and shared data structures for all
sentiment scoring implementations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class SentimentScore:
    """
    Result of scoring a text for sentiment.

    Attributes:
        score: Sentiment polarity from -1.0 (most negative) to 1.0 (most positive).
        confidence: Confidence in the score from 0.0 to 1.0.
        model_used: Identifier of the model/method that produced this score.
        metadata: Extra information such as per-class probabilities.
    """

    score: float = 0.0
    confidence: float = 0.0
    model_used: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.score = max(-1.0, min(1.0, self.score))
        self.confidence = max(0.0, min(1.0, self.confidence))


class SentimentScorer(ABC):
    """
    Abstract base class for sentiment scoring implementations.

    All scorers accept raw text and return a ``SentimentScore`` containing
    a polarity value in [-1, 1] and an associated confidence.
    """

    @abstractmethod
    def score_text(self, text: str) -> SentimentScore:
        """
        Score sentiment of a text passage.

        Args:
            text: The text to analyze.

        Returns:
            A SentimentScore with polarity, confidence, and model metadata.
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for this scorer."""
        ...
