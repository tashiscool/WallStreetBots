"""
Tests for the SentimentAlphaModel.

Covers insight generation based on pre-loaded sentiment data,
threshold behaviour, min_articles filter, and registry integration.
"""

import pytest
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import List

from backend.tradingbot.framework.alpha_models.sentiment_alpha import SentimentAlphaModel
from backend.tradingbot.framework.insight import Insight, InsightDirection
from backend.tradingbot.framework.alpha_models import ALPHA_MODELS


# ---------------------------------------------------------------------------
# Helper: lightweight SentimentData stand-in for tests (avoids Django deps)
# ---------------------------------------------------------------------------

@dataclass
class _FakeSentiment:
    """Minimal stand-in matching SentimentData fields used by the alpha model."""
    ticker: str = ""
    source: str = "test"
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    sentiment_score: Decimal = Decimal("0")
    sentiment_magnitude: Decimal = Decimal("0.5")
    headline: str = ""
    url: str = ""
    model_used: str = "test_model"


def _make_records(ticker: str, score: float, count: int, hours_ago: float = 1.0) -> List[_FakeSentiment]:
    """Create *count* sentiment records for *ticker* with the given score."""
    ts = datetime.now(timezone.utc) - timedelta(hours=hours_ago)
    return [
        _FakeSentiment(
            ticker=ticker,
            sentiment_score=Decimal(str(score)),
            sentiment_magnitude=Decimal("0.7"),
            timestamp=ts,
        )
        for _ in range(count)
    ]


# =====================================================================
# Bullish / Bearish / Neutral thresholds
# =====================================================================


class TestSentimentAlphaThresholds:
    """Test that insights match expected directions based on sentiment scores."""

    def test_bullish_signal(self):
        model = SentimentAlphaModel(bullish_threshold=0.15, min_articles=3, min_confidence=0.1)
        records = _make_records("AAPL", 0.35, 5)
        data = {"sentiment": {"AAPL": records}}

        insights = model.generate_insights(data, ["AAPL"])
        assert len(insights) == 1
        assert insights[0].direction == InsightDirection.UP
        assert insights[0].symbol == "AAPL"
        assert insights[0].magnitude > 0
        assert insights[0].confidence > 0

    def test_bearish_signal(self):
        model = SentimentAlphaModel(bearish_threshold=-0.15, min_articles=3, min_confidence=0.1)
        records = _make_records("TSLA", -0.40, 5)
        data = {"sentiment": {"TSLA": records}}

        insights = model.generate_insights(data, ["TSLA"])
        assert len(insights) == 1
        assert insights[0].direction == InsightDirection.DOWN
        assert insights[0].symbol == "TSLA"
        assert insights[0].magnitude > 0

    def test_neutral_no_insight(self):
        """Scores near zero should not emit an insight."""
        model = SentimentAlphaModel(
            bullish_threshold=0.15,
            bearish_threshold=-0.15,
            min_articles=3,
            min_confidence=0.1,
        )
        records = _make_records("SPY", 0.05, 5)
        data = {"sentiment": {"SPY": records}}

        insights = model.generate_insights(data, ["SPY"])
        assert len(insights) == 0

    def test_exactly_at_threshold_no_signal(self):
        """Score exactly at the threshold is not greater-than, so no signal."""
        model = SentimentAlphaModel(bullish_threshold=0.15, min_articles=3, min_confidence=0.1)
        records = _make_records("AAPL", 0.15, 5)
        data = {"sentiment": {"AAPL": records}}

        insights = model.generate_insights(data, ["AAPL"])
        assert len(insights) == 0


# =====================================================================
# Min articles filter
# =====================================================================


class TestSentimentAlphaMinArticles:
    """Test the min_articles threshold."""

    def test_below_min_articles_no_insight(self):
        model = SentimentAlphaModel(min_articles=5, min_confidence=0.1)
        records = _make_records("AAPL", 0.5, 3)  # Only 3, need 5
        data = {"sentiment": {"AAPL": records}}

        insights = model.generate_insights(data, ["AAPL"])
        assert len(insights) == 0

    def test_exactly_min_articles_emits_insight(self):
        model = SentimentAlphaModel(min_articles=3, min_confidence=0.1)
        records = _make_records("AAPL", 0.5, 3)
        data = {"sentiment": {"AAPL": records}}

        insights = model.generate_insights(data, ["AAPL"])
        assert len(insights) == 1

    def test_no_records_no_insight(self):
        model = SentimentAlphaModel(min_articles=1)
        data = {"sentiment": {}}

        insights = model.generate_insights(data, ["AAPL"])
        assert len(insights) == 0


# =====================================================================
# Min confidence filter
# =====================================================================


class TestSentimentAlphaMinConfidence:
    """Test the min_confidence threshold."""

    def test_low_confidence_skipped(self):
        model = SentimentAlphaModel(min_confidence=0.8, min_articles=1)
        records = [
            _FakeSentiment(
                ticker="AAPL",
                sentiment_score=Decimal("0.5"),
                sentiment_magnitude=Decimal("0.2"),  # Low confidence
                timestamp=datetime.now(timezone.utc),
            )
            for _ in range(5)
        ]
        data = {"sentiment": {"AAPL": records}}

        insights = model.generate_insights(data, ["AAPL"])
        assert len(insights) == 0


# =====================================================================
# Lookback window
# =====================================================================


class TestSentimentAlphaLookback:
    """Test that old records are filtered out."""

    def test_old_records_excluded(self):
        model = SentimentAlphaModel(
            lookback_hours=24.0,
            min_articles=3,
            min_confidence=0.1,
        )
        # Records from 48 hours ago
        old_records = _make_records("AAPL", 0.5, 5, hours_ago=48.0)
        data = {"sentiment": {"AAPL": old_records}}

        insights = model.generate_insights(data, ["AAPL"])
        assert len(insights) == 0

    def test_recent_records_included(self):
        model = SentimentAlphaModel(
            lookback_hours=24.0,
            min_articles=3,
            min_confidence=0.1,
        )
        recent = _make_records("AAPL", 0.5, 5, hours_ago=1.0)
        data = {"sentiment": {"AAPL": recent}}

        insights = model.generate_insights(data, ["AAPL"])
        assert len(insights) == 1


# =====================================================================
# Multiple symbols
# =====================================================================


class TestSentimentAlphaMultipleSymbols:
    """Test with multiple symbols."""

    def test_multiple_symbols(self):
        model = SentimentAlphaModel(min_articles=2, min_confidence=0.1)
        data = {
            "sentiment": {
                "AAPL": _make_records("AAPL", 0.5, 5),
                "TSLA": _make_records("TSLA", -0.5, 3),
                "SPY": _make_records("SPY", 0.0, 4),
            }
        }

        insights = model.generate_insights(data, ["AAPL", "TSLA", "SPY"])
        directions = {i.symbol: i.direction for i in insights}

        assert directions.get("AAPL") == InsightDirection.UP
        assert directions.get("TSLA") == InsightDirection.DOWN
        assert "SPY" not in directions  # Neutral, no insight emitted

    def test_symbol_not_in_data_skipped(self):
        model = SentimentAlphaModel(min_articles=1)
        data = {"sentiment": {"AAPL": _make_records("AAPL", 0.5, 5)}}

        insights = model.generate_insights(data, ["AAPL", "GOOG"])
        assert len(insights) == 1
        assert insights[0].symbol == "AAPL"


# =====================================================================
# Insight metadata
# =====================================================================


class TestSentimentAlphaInsightMetadata:
    """Test that insights carry useful metadata."""

    def test_insight_has_metadata(self):
        model = SentimentAlphaModel(min_articles=1, min_confidence=0.1)
        data = {"sentiment": {"AAPL": _make_records("AAPL", 0.4, 3)}}

        insights = model.generate_insights(data, ["AAPL"])
        assert len(insights) == 1
        meta = insights[0].metadata
        assert "avg_sentiment" in meta
        assert "num_articles" in meta
        assert meta["num_articles"] == 3

    def test_source_model_set(self):
        model = SentimentAlphaModel(name="TestSentiment", min_articles=1, min_confidence=0.1)
        data = {"sentiment": {"AAPL": _make_records("AAPL", 0.4, 3)}}

        insights = model.generate_insights(data, ["AAPL"])
        assert insights[0].source_model == "TestSentiment"


# =====================================================================
# AlphaModel ABC compliance
# =====================================================================


class TestSentimentAlphaModelInterface:
    """Verify SentimentAlphaModel is a proper AlphaModel."""

    def test_inherits_alpha_model(self):
        from backend.tradingbot.framework.alpha_model import AlphaModel
        assert issubclass(SentimentAlphaModel, AlphaModel)

    def test_has_name(self):
        model = SentimentAlphaModel()
        assert model.name == "SentimentAlpha"

    def test_update_method_works(self):
        """The inherited update() method should call generate_insights."""
        model = SentimentAlphaModel(min_articles=1, min_confidence=0.1)
        model.state.symbols.add("AAPL")
        data = {"sentiment": {"AAPL": _make_records("AAPL", 0.4, 3)}}

        insights = model.update(data)
        assert len(insights) == 1
        assert insights[0].source_model == "SentimentAlpha"


# =====================================================================
# Registry integration
# =====================================================================


class TestSentimentAlphaRegistry:
    """Verify registration in the ALPHA_MODELS dict."""

    def test_registered_in_alpha_models(self):
        assert "sentiment" in ALPHA_MODELS
        assert ALPHA_MODELS["sentiment"] is SentimentAlphaModel

    def test_instantiable_via_registry(self):
        from backend.tradingbot.framework.alpha_models import get_alpha_model
        model = get_alpha_model("sentiment")
        assert isinstance(model, SentimentAlphaModel)

    def test_listed_in_alpha_models(self):
        from backend.tradingbot.framework.alpha_models import list_alpha_models
        listing = list_alpha_models()
        assert "sentiment" in listing
        assert listing["sentiment"] == "SentimentAlphaModel"
