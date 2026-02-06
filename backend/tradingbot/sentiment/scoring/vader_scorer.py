"""
VADER-based Sentiment Scorer

Uses the vaderSentiment library augmented with a financial-domain lexicon.
Falls back to a simple keyword-based approach if vaderSentiment is not installed.
"""

import logging
import re
from typing import Dict, Optional

from .base import SentimentScore, SentimentScorer
from .financial_lexicon import FINANCIAL_LEXICON

logger = logging.getLogger(__name__)

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    HAS_VADER = True
except ImportError:
    HAS_VADER = False


class _KeywordFallbackScorer:
    """
    Minimal keyword scorer used when vaderSentiment is not installed.

    Splits text into tokens, looks each up in the financial lexicon (and
    a small set of general-purpose sentiment words), then returns a
    normalised compound score in [-1, 1].
    """

    _GENERAL_LEXICON: Dict[str, float] = {
        "good": 1.5,
        "great": 2.0,
        "excellent": 2.5,
        "amazing": 2.5,
        "terrible": -2.5,
        "bad": -1.5,
        "awful": -2.5,
        "horrible": -2.5,
        "poor": -1.5,
        "wonderful": 2.5,
        "fantastic": 2.5,
        "best": 2.0,
        "worst": -2.0,
        "up": 0.5,
        "down": -0.5,
        "high": 0.5,
        "low": -0.5,
        "increase": 1.0,
        "decrease": -1.0,
        "positive": 1.5,
        "negative": -1.5,
    }

    def __init__(self) -> None:
        self._lexicon: Dict[str, float] = {}
        self._lexicon.update(self._GENERAL_LEXICON)
        self._lexicon.update(FINANCIAL_LEXICON)

    def polarity_scores(self, text: str) -> Dict[str, float]:
        tokens = re.findall(r"[a-z][\w'-]*", text.lower())
        if not tokens:
            return {"compound": 0.0, "pos": 0.0, "neg": 0.0, "neu": 1.0}

        total = 0.0
        matched = 0
        for token in tokens:
            if token in self._lexicon:
                total += self._lexicon[token]
                matched += 1

        if matched == 0:
            return {"compound": 0.0, "pos": 0.0, "neg": 0.0, "neu": 1.0}

        # Normalise into [-1, 1] using a tanh-like sigmoid
        raw = total / (len(tokens) ** 0.5)
        compound = raw / ((raw ** 2 + 15) ** 0.5)

        pos = max(compound, 0.0)
        neg = abs(min(compound, 0.0))
        neu = 1.0 - pos - neg
        return {"compound": compound, "pos": pos, "neg": neg, "neu": max(neu, 0.0)}


class VaderSentimentScorer(SentimentScorer):
    """
    Sentiment scorer using VADER (Valence Aware Dictionary and sEntiment
    Reasoner) augmented with financial-domain vocabulary.

    If ``vaderSentiment`` is not installed the scorer falls back to a
    lightweight keyword-based approach using the same financial lexicon.
    """

    def __init__(self) -> None:
        self._analyzer: Optional[object] = None
        self._fallback: Optional[_KeywordFallbackScorer] = None
        self._using_vader = HAS_VADER

        if HAS_VADER:
            self._analyzer = SentimentIntensityAnalyzer()
            # Augment VADER's built-in lexicon with financial terms
            for term, intensity in FINANCIAL_LEXICON.items():
                self._analyzer.lexicon[term] = intensity
            logger.info("VaderSentimentScorer initialised with vaderSentiment + financial lexicon")
        else:
            self._fallback = _KeywordFallbackScorer()
            logger.warning(
                "vaderSentiment not installed; using keyword fallback. "
                "Install with: pip install vaderSentiment"
            )

    @property
    def name(self) -> str:
        return "vader" if self._using_vader else "keyword_fallback"

    def score_text(self, text: str) -> SentimentScore:
        """
        Score sentiment of *text*.

        Returns a ``SentimentScore`` whose ``score`` is the VADER compound
        value (already in [-1, 1]) and whose ``confidence`` is derived from
        the intensity of the signal (how far from zero the compound score is).
        """
        if not text or not text.strip():
            return SentimentScore(
                score=0.0,
                confidence=0.0,
                model_used=self.name,
                metadata={"raw_scores": {"compound": 0.0, "pos": 0.0, "neg": 0.0, "neu": 1.0}},
            )

        if self._using_vader and self._analyzer is not None:
            scores = self._analyzer.polarity_scores(text)
        else:
            scores = self._fallback.polarity_scores(text)

        compound = scores["compound"]

        # Confidence is based on how extreme the compound score is.
        # A compound of +/-1 gives confidence 1; near-zero gives low confidence.
        confidence = abs(compound)

        return SentimentScore(
            score=compound,
            confidence=confidence,
            model_used=self.name,
            metadata={"raw_scores": scores},
        )
