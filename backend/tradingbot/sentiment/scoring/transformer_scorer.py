"""
Transformer-based Sentiment Scorer

Uses HuggingFace Transformers with the FinBERT model (ProsusAI/finbert)
for financial sentiment classification.  The model outputs three classes
-- positive, negative, neutral -- which are mapped to a [-1, 1] score.

The model is lazy-loaded on the first call to ``score_text`` to avoid
importing heavyweight ML libraries at module-import time.
"""

import logging
from typing import Any, Dict, List, Optional

from .base import SentimentScore, SentimentScorer

logger = logging.getLogger(__name__)

try:
    import torch  # noqa: F401 -- only needed to verify availability
    from transformers import pipeline as hf_pipeline

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

# Default model; can be overridden via constructor
_DEFAULT_MODEL = "ProsusAI/finbert"

# Maximum token length accepted by FinBERT
_MAX_LENGTH = 512


class TransformerSentimentScorer(SentimentScorer):
    """
    Financial sentiment scorer powered by FinBERT.

    Lazy-loads the model on the first invocation of ``score_text`` so that
    importing this module is cheap even when ``transformers`` is installed.

    If ``transformers`` (or ``torch``) is not available, ``score_text``
    returns a zero-score with zero confidence.
    """

    def __init__(self, model_name: str = _DEFAULT_MODEL, device: int = -1) -> None:
        """
        Args:
            model_name: HuggingFace model identifier.
            device: Torch device ordinal (-1 for CPU).
        """
        self._model_name = model_name
        self._device = device
        self._pipeline: Optional[Any] = None
        self._loaded = False

    @property
    def name(self) -> str:
        return "finbert"

    # ------------------------------------------------------------------
    # Lazy initialisation
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> bool:
        """Load the HuggingFace pipeline if not yet done. Returns True on success."""
        if self._loaded:
            return self._pipeline is not None

        self._loaded = True

        if not HAS_TRANSFORMERS:
            logger.warning(
                "transformers / torch not installed; TransformerSentimentScorer "
                "is disabled.  Install with: pip install transformers torch"
            )
            return False

        try:
            logger.info("Loading FinBERT model %s (this may take a moment)...", self._model_name)
            self._pipeline = hf_pipeline(
                "sentiment-analysis",
                model=self._model_name,
                device=self._device,
                truncation=True,
                max_length=_MAX_LENGTH,
            )
            logger.info("FinBERT model loaded successfully.")
            return True
        except Exception:
            logger.exception("Failed to load FinBERT model %s", self._model_name)
            self._pipeline = None
            return False

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    @staticmethod
    def _map_label_to_score(results: List[Dict[str, Any]]) -> tuple:
        """
        Map FinBERT's 3-class output to a single [-1, 1] score.

        FinBERT returns one result per input, each with label in
        {positive, negative, neutral} and a softmax probability.

        Strategy:
        - positive  -> +prob
        - negative  -> -prob
        - neutral   ->  0
        Then combine with the full distribution if available.
        """
        if not results:
            return 0.0, 0.0

        result = results[0]
        label = result["label"].lower()
        prob = result["score"]  # softmax probability for the winning class

        if label == "positive":
            score = prob
        elif label == "negative":
            score = -prob
        else:
            score = 0.0

        # Confidence = winning class probability (higher prob -> more confident)
        confidence = prob

        return score, confidence

    def score_text(self, text: str) -> SentimentScore:
        """Score sentiment using FinBERT.

        Returns a zero-confidence score if the model could not be loaded
        or if the input text is empty.
        """
        if not text or not text.strip():
            return SentimentScore(
                score=0.0,
                confidence=0.0,
                model_used=self.name,
                metadata={"error": "empty_input"},
            )

        if not self._ensure_loaded():
            return SentimentScore(
                score=0.0,
                confidence=0.0,
                model_used=self.name,
                metadata={"error": "model_unavailable"},
            )

        try:
            # Truncate to keep within model limits
            truncated = text[:_MAX_LENGTH * 4]  # rough char estimate
            results = self._pipeline(truncated)
            score_val, confidence = self._map_label_to_score(results)

            return SentimentScore(
                score=score_val,
                confidence=confidence,
                model_used=self.name,
                metadata={
                    "raw_results": results,
                    "model": self._model_name,
                },
            )
        except Exception:
            logger.exception("FinBERT scoring failed for text: %.80s...", text)
            return SentimentScore(
                score=0.0,
                confidence=0.0,
                model_used=self.name,
                metadata={"error": "scoring_failed"},
            )
