"""
Ensemble Sentiment Scorer

Combines multiple scoring backends with configurable weighting.

Two modes are offered:
  * FAST   -- VADER only (~<1 ms per call).
  * ACCURATE -- VADER (weight 0.30) + FinBERT (weight 0.70).  Falls back
    gracefully to FAST mode if FinBERT / transformers is unavailable.
"""

import logging
from enum import Enum
from typing import List, Tuple

from .base import SentimentScore, SentimentScorer
from .vader_scorer import VaderSentimentScorer
from .transformer_scorer import TransformerSentimentScorer, HAS_TRANSFORMERS

logger = logging.getLogger(__name__)


class ScoringMode(Enum):
    """Scoring strategy selector."""

    FAST = "fast"          # VADER only
    ACCURATE = "accurate"  # VADER + FinBERT weighted blend


class EnsembleSentimentScorer(SentimentScorer):
    """
    Ensemble of VADER and (optionally) FinBERT scorers.

    In ACCURATE mode the final score is a weighted average:
        score = w_vader * vader_score + w_finbert * finbert_score

    If FinBERT is unavailable the scorer silently degrades to FAST mode.
    """

    # Default weights for ACCURATE mode
    _VADER_WEIGHT = 0.30
    _FINBERT_WEIGHT = 0.70

    def __init__(self, mode: ScoringMode = ScoringMode.FAST) -> None:
        self._mode = mode
        self._vader = VaderSentimentScorer()
        self._transformer: TransformerSentimentScorer | None = None

        if mode == ScoringMode.ACCURATE:
            if HAS_TRANSFORMERS:
                self._transformer = TransformerSentimentScorer()
            else:
                logger.warning(
                    "ScoringMode.ACCURATE requested but transformers not installed; "
                    "falling back to FAST (VADER-only) mode."
                )

    @property
    def name(self) -> str:
        if self._mode == ScoringMode.ACCURATE and self._transformer is not None:
            return "ensemble_vader_finbert"
        return "ensemble_vader"

    @property
    def mode(self) -> ScoringMode:
        return self._mode

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _collect_scores(self, text: str) -> List[Tuple[SentimentScore, float]]:
        """Return ``(score, weight)`` pairs from all active backends."""
        pairs: List[Tuple[SentimentScore, float]] = []

        vader_result = self._vader.score_text(text)
        pairs.append((vader_result, self._VADER_WEIGHT if self._transformer else 1.0))

        if self._transformer is not None:
            finbert_result = self._transformer.score_text(text)
            # Only use FinBERT result when it succeeded (confidence > 0)
            if finbert_result.confidence > 0:
                pairs.append((finbert_result, self._FINBERT_WEIGHT))
            else:
                # FinBERT failed; give VADER full weight
                pairs[0] = (vader_result, 1.0)

        return pairs

    @staticmethod
    def _weighted_average(pairs: List[Tuple[SentimentScore, float]]) -> SentimentScore:
        """Compute weight-normalised average score and confidence."""
        total_weight = sum(w for _, w in pairs)
        if total_weight == 0:
            return SentimentScore(score=0.0, confidence=0.0, model_used="ensemble_empty")

        avg_score = sum(s.score * w for s, w in pairs) / total_weight
        avg_conf = sum(s.confidence * w for s, w in pairs) / total_weight

        models_used = "+".join(s.model_used for s, _ in pairs)
        metadata = {
            "components": [
                {"model": s.model_used, "score": s.score, "confidence": s.confidence, "weight": w}
                for s, w in pairs
            ],
        }

        return SentimentScore(
            score=avg_score,
            confidence=avg_conf,
            model_used=models_used,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score_text(self, text: str) -> SentimentScore:
        """Score *text* using the configured ensemble."""
        if not text or not text.strip():
            return SentimentScore(score=0.0, confidence=0.0, model_used=self.name)

        pairs = self._collect_scores(text)
        result = self._weighted_average(pairs)
        return result
