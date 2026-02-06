"""
Tests for the sentiment scoring subsystem.

Covers SentimentScore, VaderSentimentScorer, TransformerSentimentScorer,
EnsembleSentimentScorer, and the financial lexicon.
"""

import pytest
from unittest.mock import MagicMock, patch

from backend.tradingbot.sentiment.scoring.base import SentimentScore, SentimentScorer
from backend.tradingbot.sentiment.scoring.financial_lexicon import FINANCIAL_LEXICON
from backend.tradingbot.sentiment.scoring.vader_scorer import VaderSentimentScorer, HAS_VADER
from backend.tradingbot.sentiment.scoring.transformer_scorer import (
    TransformerSentimentScorer,
    HAS_TRANSFORMERS,
)
from backend.tradingbot.sentiment.scoring.ensemble_scorer import (
    EnsembleSentimentScorer,
    ScoringMode,
)


# =====================================================================
# SentimentScore dataclass
# =====================================================================


class TestSentimentScore:
    """Tests for the SentimentScore dataclass."""

    def test_creation_defaults(self):
        s = SentimentScore()
        assert s.score == 0.0
        assert s.confidence == 0.0
        assert s.model_used == ""
        assert s.metadata == {}

    def test_creation_with_values(self):
        s = SentimentScore(score=0.75, confidence=0.9, model_used="test", metadata={"k": "v"})
        assert s.score == 0.75
        assert s.confidence == 0.9
        assert s.model_used == "test"
        assert s.metadata == {"k": "v"}

    def test_score_clamped_high(self):
        s = SentimentScore(score=2.5)
        assert s.score == 1.0

    def test_score_clamped_low(self):
        s = SentimentScore(score=-3.0)
        assert s.score == -1.0

    def test_confidence_clamped_high(self):
        s = SentimentScore(confidence=5.0)
        assert s.confidence == 1.0

    def test_confidence_clamped_low(self):
        s = SentimentScore(confidence=-0.5)
        assert s.confidence == 0.0

    def test_boundary_values(self):
        s = SentimentScore(score=1.0, confidence=1.0)
        assert s.score == 1.0
        assert s.confidence == 1.0

        s2 = SentimentScore(score=-1.0, confidence=0.0)
        assert s2.score == -1.0
        assert s2.confidence == 0.0


# =====================================================================
# Financial Lexicon
# =====================================================================


class TestFinancialLexicon:
    """Tests for the financial lexicon dictionary."""

    def test_has_at_least_50_terms(self):
        assert len(FINANCIAL_LEXICON) >= 50, (
            f"Expected at least 50 terms, got {len(FINANCIAL_LEXICON)}"
        )

    def test_bullish_terms_positive(self):
        for term in ("bullish", "upgrade", "surge", "rally", "soar"):
            assert FINANCIAL_LEXICON[term] > 0, f"'{term}' should be positive"

    def test_bearish_terms_negative(self):
        for term in ("bearish", "downgrade", "bankruptcy", "recession", "crash"):
            assert FINANCIAL_LEXICON[term] < 0, f"'{term}' should be negative"

    def test_all_values_in_range(self):
        for term, score in FINANCIAL_LEXICON.items():
            assert -4.0 <= score <= 4.0, f"'{term}' has out-of-range score {score}"

    def test_bankruptcy_strongly_negative(self):
        assert FINANCIAL_LEXICON["bankruptcy"] <= -3.0


# =====================================================================
# VaderSentimentScorer
# =====================================================================


class TestVaderSentimentScorer:
    """Tests for VaderSentimentScorer (with fallback if VADER missing)."""

    def test_name(self):
        scorer = VaderSentimentScorer()
        assert scorer.name in ("vader", "keyword_fallback")

    def test_empty_text_returns_zero(self):
        scorer = VaderSentimentScorer()
        result = scorer.score_text("")
        assert result.score == 0.0
        assert result.confidence == 0.0

    def test_whitespace_text_returns_zero(self):
        scorer = VaderSentimentScorer()
        result = scorer.score_text("   ")
        assert result.score == 0.0

    def test_positive_text(self):
        scorer = VaderSentimentScorer()
        result = scorer.score_text("Company reports record profits and strong growth")
        assert result.score > 0, "Positive financial text should have positive score"
        assert result.confidence > 0

    def test_negative_text(self):
        scorer = VaderSentimentScorer()
        result = scorer.score_text("Massive losses, bankruptcy filing imminent")
        assert result.score < 0, "Negative financial text should have negative score"
        assert result.confidence > 0

    def test_score_in_range(self):
        scorer = VaderSentimentScorer()
        result = scorer.score_text("Markets are doing things today.")
        assert -1.0 <= result.score <= 1.0
        assert 0.0 <= result.confidence <= 1.0

    def test_model_used_set(self):
        scorer = VaderSentimentScorer()
        result = scorer.score_text("test")
        assert result.model_used in ("vader", "keyword_fallback")

    def test_metadata_has_raw_scores(self):
        scorer = VaderSentimentScorer()
        result = scorer.score_text("test")
        assert "raw_scores" in result.metadata

    @pytest.mark.skipif(not HAS_VADER, reason="vaderSentiment not installed")
    def test_vader_augmented_lexicon(self):
        """VADER should incorporate financial terms from the lexicon."""
        scorer = VaderSentimentScorer()
        assert scorer._using_vader is True
        # "bullish" is in our lexicon but not in default VADER
        result = scorer.score_text("The outlook is extremely bullish")
        assert result.score > 0

    @pytest.mark.skipif(not HAS_VADER, reason="vaderSentiment not installed")
    def test_vader_bearish_term(self):
        scorer = VaderSentimentScorer()
        result = scorer.score_text("bankruptcy looms, massive crash expected")
        assert result.score < -0.3


# =====================================================================
# TransformerSentimentScorer
# =====================================================================


class TestTransformerSentimentScorer:
    """Tests for TransformerSentimentScorer using mocks."""

    def test_name(self):
        scorer = TransformerSentimentScorer()
        assert scorer.name == "finbert"

    def test_empty_text(self):
        scorer = TransformerSentimentScorer()
        result = scorer.score_text("")
        assert result.score == 0.0
        assert result.confidence == 0.0
        assert result.metadata.get("error") == "empty_input"

    def test_model_unavailable_without_transformers(self):
        scorer = TransformerSentimentScorer()
        scorer._loaded = True
        scorer._pipeline = None
        result = scorer.score_text("Some text")
        assert result.score == 0.0
        assert result.metadata.get("error") == "model_unavailable"

    def test_score_with_mocked_pipeline_positive(self):
        scorer = TransformerSentimentScorer()
        scorer._loaded = True
        mock_pipeline = MagicMock()
        mock_pipeline.return_value = [{"label": "positive", "score": 0.92}]
        scorer._pipeline = mock_pipeline

        result = scorer.score_text("Incredible earnings beat expectations")
        assert result.score == pytest.approx(0.92)
        assert result.confidence == pytest.approx(0.92)
        assert result.model_used == "finbert"

    def test_score_with_mocked_pipeline_negative(self):
        scorer = TransformerSentimentScorer()
        scorer._loaded = True
        mock_pipeline = MagicMock()
        mock_pipeline.return_value = [{"label": "negative", "score": 0.85}]
        scorer._pipeline = mock_pipeline

        result = scorer.score_text("Company faces bankruptcy")
        assert result.score == pytest.approx(-0.85)
        assert result.confidence == pytest.approx(0.85)

    def test_score_with_mocked_pipeline_neutral(self):
        scorer = TransformerSentimentScorer()
        scorer._loaded = True
        mock_pipeline = MagicMock()
        mock_pipeline.return_value = [{"label": "neutral", "score": 0.78}]
        scorer._pipeline = mock_pipeline

        result = scorer.score_text("Company reports quarterly results")
        assert result.score == 0.0
        assert result.confidence == pytest.approx(0.78)

    def test_pipeline_exception_returns_zero(self):
        scorer = TransformerSentimentScorer()
        scorer._loaded = True
        mock_pipeline = MagicMock(side_effect=RuntimeError("GPU OOM"))
        scorer._pipeline = mock_pipeline

        result = scorer.score_text("Some text")
        assert result.score == 0.0
        assert result.metadata.get("error") == "scoring_failed"

    def test_map_label_to_score_empty(self):
        score, confidence = TransformerSentimentScorer._map_label_to_score([])
        assert score == 0.0
        assert confidence == 0.0


# =====================================================================
# EnsembleSentimentScorer
# =====================================================================


class TestEnsembleSentimentScorer:
    """Tests for the ensemble scorer."""

    def test_fast_mode_uses_vader_only(self):
        scorer = EnsembleSentimentScorer(mode=ScoringMode.FAST)
        assert scorer._transformer is None
        assert "vader" in scorer.name

    def test_fast_mode_scores_text(self):
        scorer = EnsembleSentimentScorer(mode=ScoringMode.FAST)
        result = scorer.score_text("Stocks surge on great earnings")
        assert result.score > 0
        assert -1.0 <= result.score <= 1.0

    def test_empty_text_returns_zero(self):
        scorer = EnsembleSentimentScorer(mode=ScoringMode.FAST)
        result = scorer.score_text("")
        assert result.score == 0.0
        assert result.confidence == 0.0

    def test_accurate_mode_without_transformers(self):
        """If transformers not installed, should degrade to FAST."""
        with patch(
            "backend.tradingbot.sentiment.scoring.ensemble_scorer.HAS_TRANSFORMERS",
            False,
        ):
            scorer = EnsembleSentimentScorer.__new__(EnsembleSentimentScorer)
            scorer._mode = ScoringMode.ACCURATE
            scorer._vader = VaderSentimentScorer()
            scorer._transformer = None  # No transformers available
            result = scorer.score_text("Good news for the market")
            assert -1.0 <= result.score <= 1.0

    def test_accurate_mode_with_mocked_transformer(self):
        """Test weighted combination of VADER + FinBERT."""
        scorer = EnsembleSentimentScorer(mode=ScoringMode.FAST)
        scorer._mode = ScoringMode.ACCURATE

        # Mock the transformer scorer
        mock_transformer = MagicMock()
        mock_transformer.score_text.return_value = SentimentScore(
            score=0.8, confidence=0.9, model_used="finbert"
        )
        scorer._transformer = mock_transformer

        result = scorer.score_text("Strong quarterly earnings beat estimates")
        # Should be a blend of VADER (positive) and FinBERT (0.8)
        assert result.score > 0
        assert result.confidence > 0

    def test_scoring_mode_enum(self):
        assert ScoringMode.FAST.value == "fast"
        assert ScoringMode.ACCURATE.value == "accurate"

    def test_fast_mode_negative(self):
        scorer = EnsembleSentimentScorer(mode=ScoringMode.FAST)
        result = scorer.score_text("Terrible crash, bankruptcy imminent, massive losses")
        assert result.score < 0
