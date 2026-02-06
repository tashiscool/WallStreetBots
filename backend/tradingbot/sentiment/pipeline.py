"""
Sentiment Pipeline

Orchestrates the full sentiment workflow: collect articles from
sources, score them, persist SentimentData records, and fire callbacks.

The pipeline can run as a periodic background loop or be invoked
on-demand via ``process_symbols``.
"""

import logging
import threading
import time
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Callable, Dict, List, Optional

from .scoring.ensemble_scorer import EnsembleSentimentScorer, ScoringMode
from .sources.aggregator import NewsAggregator

logger = logging.getLogger(__name__)


class SentimentPipeline:
    """
    End-to-end sentiment processing pipeline.

    Workflow
    -------
    1. Collect articles/posts via :class:`NewsAggregator`.
    2. Score each item with :class:`EnsembleSentimentScorer`.
    3. Convert scores into :class:`SentimentData` records.
    4. Emit ``on_sentiment`` callbacks with the new records.

    The pipeline can either be driven externally (call
    ``process_symbols`` in your own loop) or run autonomously via
    ``start`` / ``stop`` using a background daemon thread.
    """

    def __init__(
        self,
        aggregator: Optional[NewsAggregator] = None,
        scorer: Optional[EnsembleSentimentScorer] = None,
        scoring_mode: ScoringMode = ScoringMode.FAST,
        poll_interval_seconds: float = 300.0,
    ) -> None:
        """
        Args:
            aggregator: News aggregator instance (created empty if None).
            scorer: Sentiment scorer (created with *scoring_mode* if None).
            scoring_mode: Default scoring mode when creating a new scorer.
            poll_interval_seconds: Seconds between automatic collection cycles.
        """
        self._aggregator = aggregator or NewsAggregator()
        self._scorer = scorer or EnsembleSentimentScorer(mode=scoring_mode)
        self._poll_interval = poll_interval_seconds

        # Callbacks: each is called with (List[SentimentData])
        self.on_sentiment: List[Callable[[List[Any]], None]] = []

        # Background loop state
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._symbols: List[str] = []

        # Accumulated results (latest batch)
        self._latest_results: List[Any] = []

    # ------------------------------------------------------------------
    # Background loop
    # ------------------------------------------------------------------

    def start(self, symbols: Optional[List[str]] = None) -> None:
        """Start the background polling loop.

        Args:
            symbols: Symbols to monitor.  Can be updated later via
                ``update_symbols``.
        """
        if self._thread is not None and self._thread.is_alive():
            logger.warning("SentimentPipeline is already running.")
            return

        if symbols:
            self._symbols = list(symbols)

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            daemon=True,
            name="SentimentPipeline",
        )
        self._thread.start()
        logger.info("SentimentPipeline started (poll every %.0fs).", self._poll_interval)

    def stop(self) -> None:
        """Signal the background loop to stop and wait for it to finish."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self._poll_interval + 5)
            self._thread = None
        logger.info("SentimentPipeline stopped.")

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def update_symbols(self, symbols: List[str]) -> None:
        """Update the list of symbols being monitored."""
        self._symbols = list(symbols)

    def _run_loop(self) -> None:
        """Internal polling loop executed on the daemon thread."""
        while not self._stop_event.is_set():
            if self._symbols:
                try:
                    self.process_symbols(self._symbols)
                except Exception:
                    logger.exception("Error in SentimentPipeline processing loop.")
            self._stop_event.wait(timeout=self._poll_interval)

    # ------------------------------------------------------------------
    # On-demand processing
    # ------------------------------------------------------------------

    def process_symbols(
        self,
        symbols: List[str],
        limit_per_source: int = 20,
    ) -> List[Any]:
        """
        Run a single collection-and-scoring cycle for *symbols*.

        Returns a list of :class:`SentimentData` records (one per scored
        article).  The same records are also passed to all registered
        ``on_sentiment`` callbacks.
        """
        # 1. Collect
        articles = self._aggregator.collect(symbols, limit_per_source=limit_per_source)
        if not articles:
            logger.debug("No articles collected for %s", symbols)
            return []

        # 2. Score each article and build SentimentData records
        records = self._score_articles(articles, symbols)

        # 3. Store latest results
        self._latest_results = records

        # 4. Fire callbacks
        self._emit(records)

        logger.info("Processed %d articles -> %d sentiment records.", len(articles), len(records))
        return records

    # ------------------------------------------------------------------
    # Internal scoring
    # ------------------------------------------------------------------

    def _score_articles(
        self,
        articles: List[Dict[str, Any]],
        symbols: List[str],
    ) -> List[Any]:
        """Score each article and wrap in a SentimentData dataclass."""
        # Late import to avoid circular deps with Django models
        from ..models.production_models import SentimentData

        records: List[SentimentData] = []

        for article in articles:
            text = article.get("title", "") + " " + article.get("text", "")
            text = text.strip()
            if not text:
                continue

            score_result = self._scorer.score_text(text)

            # Determine which ticker(s) this article relates to
            tickers = self._extract_tickers(article, symbols)
            source_name = article.get("source", "unknown")

            for ticker in tickers:
                record = SentimentData(
                    ticker=ticker,
                    source=source_name,
                    timestamp=datetime.now(timezone.utc),
                    sentiment_score=Decimal(str(round(score_result.score, 6))),
                    sentiment_magnitude=Decimal(str(round(score_result.confidence, 6))),
                    headline=article.get("title", article.get("text", ""))[:500],
                    url=article.get("url", ""),
                    model_used=score_result.model_used,
                )
                records.append(record)

        return records

    @staticmethod
    def _extract_tickers(
        article: Dict[str, Any],
        symbols: List[str],
    ) -> List[str]:
        """Determine which of *symbols* are mentioned in *article*."""
        # Check explicit mentioned_tickers field (set by RedditSource)
        mentioned = article.get("mentioned_tickers", [])
        matched = [s for s in symbols if s.upper() in [m.upper() for m in mentioned]]
        if matched:
            return matched

        # Fall back to scanning title + text for symbol mentions
        full_text = (article.get("title", "") + " " + article.get("text", "")).upper()
        matched = [s for s in symbols if s.upper() in full_text or f"${s.upper()}" in full_text]
        if matched:
            return matched

        # If no specific match, assign to all symbols (broad market news)
        return list(symbols)

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _emit(self, records: List[Any]) -> None:
        """Fire all registered on_sentiment callbacks."""
        for callback in self.on_sentiment:
            try:
                callback(records)
            except Exception:
                logger.exception("Error in on_sentiment callback %s", callback)

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    @property
    def latest_results(self) -> List[Any]:
        """Return the most recent batch of SentimentData records."""
        return list(self._latest_results)

    def get_latest_for_symbol(self, symbol: str) -> List[Any]:
        """Filter latest results to a specific symbol."""
        return [r for r in self._latest_results if getattr(r, "ticker", "") == symbol]
