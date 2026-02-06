"""
Sentiment Alpha Model

Generates trading insights based on aggregated sentiment scores from
news, social media, and SEC filings.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from ..alpha_model import AlphaModel
from ..insight import Insight, InsightDirection

logger = logging.getLogger(__name__)

# Late-import guard: the sentiment package uses optional deps, so we
# avoid importing it at module level to prevent import errors when
# those deps are absent.
_ScoringMode = None


def _get_scoring_mode():
    global _ScoringMode
    if _ScoringMode is None:
        from ...sentiment.scoring.ensemble_scorer import ScoringMode
        _ScoringMode = ScoringMode
    return _ScoringMode


class SentimentAlphaModel(AlphaModel):
    """
    Alpha model that converts sentiment data into directional insights.

    The model expects sentiment data to be pre-loaded in the ``data``
    dict under the key ``"sentiment"``, structured as::

        data["sentiment"] = {
            "AAPL": [SentimentData(...), ...],
            "TSLA": [SentimentData(...), ...],
        }

    Alternatively the model can pull live data from a
    :class:`SentimentPipeline` instance if one is provided.

    Thresholds
    ----------
    * ``bullish_threshold`` (default 0.15): average score above this emits UP.
    * ``bearish_threshold`` (default -0.15): average score below this emits DOWN.
    * ``min_confidence`` (default 0.3): minimum average confidence to emit a signal.
    * ``min_articles`` (default 3): minimum number of scored articles required.
    """

    def __init__(
        self,
        bullish_threshold: float = 0.15,
        bearish_threshold: float = -0.15,
        min_confidence: float = 0.3,
        min_articles: int = 3,
        scoring_mode: Any = None,
        lookback_hours: float = 24.0,
        pipeline: Optional[Any] = None,
        name: str = "SentimentAlpha",
    ) -> None:
        super().__init__(name)
        self.bullish_threshold = bullish_threshold
        self.bearish_threshold = bearish_threshold
        self.min_confidence = min_confidence
        self.min_articles = min_articles
        self.lookback_hours = lookback_hours
        self._pipeline = pipeline

        # Resolve scoring mode (use FAST as default)
        if scoring_mode is None:
            try:
                ScoringMode = _get_scoring_mode()
                self.scoring_mode = ScoringMode.FAST
            except Exception:
                self.scoring_mode = None
        else:
            self.scoring_mode = scoring_mode

    def generate_insights(
        self,
        data: Dict[str, Any],
        symbols: List[str],
    ) -> List[Insight]:
        """
        Generate sentiment-based trading insights.

        Looks up recent sentiment data for each symbol, computes
        the average score and confidence, and emits UP / DOWN / FLAT
        insights based on the configured thresholds.
        """
        insights: List[Insight] = []
        sentiment_data = data.get("sentiment", {})

        for symbol in symbols:
            records = self._get_records(symbol, sentiment_data)
            if not records:
                continue

            # Filter to recent records within lookback window
            cutoff = datetime.now(timezone.utc) - timedelta(hours=self.lookback_hours)
            recent = self._filter_recent(records, cutoff)

            if len(recent) < self.min_articles:
                logger.debug(
                    "%s: only %d articles for %s (need %d); skipping",
                    self.name, len(recent), symbol, self.min_articles,
                )
                continue

            avg_score, avg_confidence, num = self._aggregate(recent)

            if avg_confidence < self.min_confidence:
                logger.debug(
                    "%s: low confidence %.3f for %s; skipping",
                    self.name, avg_confidence, symbol,
                )
                continue

            direction, magnitude = self._determine_direction(avg_score)

            if direction == InsightDirection.FLAT:
                continue  # No actionable signal

            insights.append(Insight(
                symbol=symbol,
                direction=direction,
                magnitude=magnitude,
                confidence=avg_confidence,
                period=timedelta(days=1),
                source_model=self.name,
                metadata={
                    "avg_sentiment": round(avg_score, 4),
                    "avg_confidence": round(avg_confidence, 4),
                    "num_articles": num,
                    "lookback_hours": self.lookback_hours,
                },
            ))

        return insights

    # ------------------------------------------------------------------
    # Record retrieval
    # ------------------------------------------------------------------

    def _get_records(
        self,
        symbol: str,
        sentiment_data: Dict[str, Any],
    ) -> List[Any]:
        """Get sentiment records for *symbol* from data dict or pipeline."""
        records = sentiment_data.get(symbol, [])
        if records:
            return records

        # Try pipeline if available
        if self._pipeline is not None:
            try:
                return self._pipeline.get_latest_for_symbol(symbol)
            except Exception:
                logger.debug("Pipeline lookup failed for %s", symbol)

        return []

    @staticmethod
    def _filter_recent(records: List[Any], cutoff: datetime) -> List[Any]:
        """Keep only records with timestamp >= cutoff."""
        recent: List[Any] = []
        for r in records:
            ts = getattr(r, "timestamp", None)
            if ts is None:
                recent.append(r)  # No timestamp -> include
                continue
            # Handle naive datetimes by assuming UTC
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            if ts >= cutoff:
                recent.append(r)
        return recent

    @staticmethod
    def _aggregate(records: List[Any]) -> tuple:
        """Compute average score and confidence from records."""
        scores: List[float] = []
        confidences: List[float] = []

        for r in records:
            score = getattr(r, "sentiment_score", None)
            if score is None:
                continue
            scores.append(float(score))

            magnitude = getattr(r, "sentiment_magnitude", None)
            if magnitude is not None:
                confidences.append(float(magnitude))

        if not scores:
            return 0.0, 0.0, 0

        avg_score = sum(scores) / len(scores)
        avg_conf = sum(confidences) / len(confidences) if confidences else abs(avg_score)

        return avg_score, avg_conf, len(scores)

    def _determine_direction(self, avg_score: float) -> tuple:
        """Map an average sentiment score to a direction and magnitude."""
        if avg_score > self.bullish_threshold:
            return InsightDirection.UP, abs(avg_score)
        elif avg_score < self.bearish_threshold:
            return InsightDirection.DOWN, abs(avg_score)
        else:
            return InsightDirection.FLAT, 0.0
