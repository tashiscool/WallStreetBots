"""
Tests for the SentimentPipeline.

Covers process_symbols, callback emission, and background loop control.
"""

import pytest
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import MagicMock, patch

from backend.tradingbot.sentiment.pipeline import SentimentPipeline
from backend.tradingbot.sentiment.scoring.base import SentimentScore
from backend.tradingbot.sentiment.scoring.ensemble_scorer import EnsembleSentimentScorer, ScoringMode
from backend.tradingbot.sentiment.sources.aggregator import NewsAggregator


class TestSentimentPipelineProcessing:
    """Tests for process_symbols."""

    def _make_pipeline(self):
        """Create a pipeline with mocked aggregator and scorer."""
        mock_aggregator = MagicMock(spec=NewsAggregator)
        mock_scorer = MagicMock(spec=EnsembleSentimentScorer)
        mock_scorer.name = "test_ensemble"

        pipeline = SentimentPipeline(
            aggregator=mock_aggregator,
            scorer=mock_scorer,
        )
        return pipeline, mock_aggregator, mock_scorer

    def test_process_symbols_empty_articles(self):
        pipeline, mock_agg, mock_scorer = self._make_pipeline()
        mock_agg.collect.return_value = []

        records = pipeline.process_symbols(["AAPL"])
        assert records == []
        mock_scorer.score_text.assert_not_called()

    def test_process_symbols_scores_articles(self):
        pipeline, mock_agg, mock_scorer = self._make_pipeline()
        mock_agg.collect.return_value = [
            {"title": "AAPL beats earnings", "text": "Great quarter", "source": "reddit", "url": "http://example.com"},
            {"title": "AAPL guidance raised", "text": "", "source": "twitter", "url": "http://example2.com"},
        ]
        mock_scorer.score_text.return_value = SentimentScore(
            score=0.65, confidence=0.8, model_used="test"
        )

        records = pipeline.process_symbols(["AAPL"])

        assert len(records) == 2
        assert mock_scorer.score_text.call_count == 2

        # Check record fields
        record = records[0]
        assert record.ticker == "AAPL"
        assert float(record.sentiment_score) == pytest.approx(0.65)
        assert record.source == "reddit"
        assert record.model_used == "test"

    def test_process_symbols_assigns_tickers(self):
        """Articles mentioning specific tickers get assigned correctly."""
        pipeline, mock_agg, mock_scorer = self._make_pipeline()
        mock_agg.collect.return_value = [
            {
                "title": "TSLA production numbers",
                "text": "Tesla hit record production",
                "source": "news",
                "url": "",
                "mentioned_tickers": ["TSLA"],
            },
        ]
        mock_scorer.score_text.return_value = SentimentScore(score=0.5, confidence=0.7, model_used="test")

        records = pipeline.process_symbols(["AAPL", "TSLA"])
        # Should only be assigned to TSLA
        tickers = [r.ticker for r in records]
        assert "TSLA" in tickers
        assert "AAPL" not in tickers

    def test_process_symbols_empty_text_skipped(self):
        pipeline, mock_agg, mock_scorer = self._make_pipeline()
        mock_agg.collect.return_value = [
            {"title": "", "text": "", "source": "x", "url": ""},
        ]
        records = pipeline.process_symbols(["AAPL"])
        assert records == []
        mock_scorer.score_text.assert_not_called()


class TestSentimentPipelineCallbacks:
    """Tests for callback emission."""

    def test_callback_called_with_records(self):
        pipeline, mock_agg, mock_scorer = self._make_pipeline_with_data()
        callback = MagicMock()
        pipeline.on_sentiment.append(callback)

        records = pipeline.process_symbols(["AAPL"])
        callback.assert_called_once()
        args = callback.call_args[0]
        assert len(args[0]) > 0

    def test_multiple_callbacks(self):
        pipeline, mock_agg, mock_scorer = self._make_pipeline_with_data()
        cb1 = MagicMock()
        cb2 = MagicMock()
        pipeline.on_sentiment.extend([cb1, cb2])

        pipeline.process_symbols(["AAPL"])
        cb1.assert_called_once()
        cb2.assert_called_once()

    def test_callback_exception_does_not_break_pipeline(self):
        pipeline, mock_agg, mock_scorer = self._make_pipeline_with_data()
        bad_cb = MagicMock(side_effect=RuntimeError("callback error"))
        good_cb = MagicMock()
        pipeline.on_sentiment.extend([bad_cb, good_cb])

        # Should not raise
        records = pipeline.process_symbols(["AAPL"])
        assert len(records) > 0
        bad_cb.assert_called_once()
        good_cb.assert_called_once()

    def _make_pipeline_with_data(self):
        mock_aggregator = MagicMock(spec=NewsAggregator)
        mock_scorer = MagicMock(spec=EnsembleSentimentScorer)
        mock_scorer.name = "test"

        mock_aggregator.collect.return_value = [
            {"title": "AAPL news", "text": "Some content about AAPL", "source": "test", "url": ""},
        ]
        mock_scorer.score_text.return_value = SentimentScore(
            score=0.5, confidence=0.8, model_used="test"
        )

        pipeline = SentimentPipeline(aggregator=mock_aggregator, scorer=mock_scorer)
        return pipeline, mock_aggregator, mock_scorer


class TestSentimentPipelineLifecycle:
    """Tests for start/stop/is_running."""

    def test_default_not_running(self):
        pipeline = SentimentPipeline()
        assert pipeline.is_running is False

    def test_start_stop(self):
        pipeline = SentimentPipeline(poll_interval_seconds=0.1)
        pipeline.start(symbols=["AAPL"])
        assert pipeline.is_running is True

        pipeline.stop()
        assert pipeline.is_running is False

    def test_update_symbols(self):
        pipeline = SentimentPipeline()
        pipeline.update_symbols(["AAPL", "TSLA"])
        assert pipeline._symbols == ["AAPL", "TSLA"]

    def test_latest_results(self):
        pipeline, mock_agg, mock_scorer = TestSentimentPipelineCallbacks()._make_pipeline_with_data()
        assert pipeline.latest_results == []

        pipeline.process_symbols(["AAPL"])
        assert len(pipeline.latest_results) > 0

    def test_get_latest_for_symbol(self):
        pipeline, mock_agg, mock_scorer = TestSentimentPipelineCallbacks()._make_pipeline_with_data()
        pipeline.process_symbols(["AAPL"])

        aapl = pipeline.get_latest_for_symbol("AAPL")
        assert len(aapl) > 0
        assert all(r.ticker == "AAPL" for r in aapl)

        tsla = pipeline.get_latest_for_symbol("TSLA")
        assert tsla == []
