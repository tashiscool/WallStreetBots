"""
Alpha Model Base Class

AlphaModels generate trading insights (predictions) based on market data.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set
import logging

from .insight import Insight, InsightDirection, InsightType

logger = logging.getLogger(__name__)


@dataclass
class AlphaModelState:
    """State tracking for AlphaModel."""
    symbols: Set[str] = field(default_factory=set)
    last_update: Optional[datetime] = None
    insights_generated: int = 0
    correct_predictions: int = 0
    total_scored: int = 0

    @property
    def accuracy(self) -> float:
        """Get prediction accuracy."""
        if self.total_scored == 0:
            return 0.0
        return self.correct_predictions / self.total_scored


class AlphaModel(ABC):
    """
    Base class for signal generation models.

    AlphaModels analyze market data and generate Insights (predictions)
    about future price movements.

    Override generate_insights() to implement your signal logic.

    Example:
        class RSIAlphaModel(AlphaModel):
            def __init__(self, rsi_period=14, oversold=30, overbought=70):
                super().__init__("RSIAlpha")
                self.rsi_period = rsi_period
                self.oversold = oversold
                self.overbought = overbought

            def generate_insights(self, data, symbols):
                insights = []
                for symbol in symbols:
                    rsi = self.calculate_rsi(data[symbol])
                    if rsi < self.oversold:
                        insights.append(Insight(
                            symbol=symbol,
                            direction=InsightDirection.UP,
                            magnitude=0.02,
                            confidence=0.7,
                        ))
                return insights
    """

    def __init__(self, name: str = "AlphaModel"):
        """
        Initialize AlphaModel.

        Args:
            name: Name of this alpha model (for tracking)
        """
        self.name = name
        self.state = AlphaModelState()
        self._active_insights: Dict[str, Insight] = {}  # symbol -> latest insight

    @abstractmethod
    def generate_insights(
        self,
        data: Dict[str, Any],
        symbols: List[str],
    ) -> List[Insight]:
        """
        Generate trading insights from market data.

        This is the main method to override in subclasses.

        Args:
            data: Market data dictionary
                {
                    'AAPL': {
                        'close': [...],
                        'volume': [...],
                        'high': [...],
                        'low': [...],
                        ...
                    },
                    ...
                }
            symbols: List of symbols to analyze

        Returns:
            List of Insight objects
        """
        pass

    def on_securities_changed(
        self,
        added: List[str],
        removed: List[str],
    ) -> None:
        """
        Called when the universe changes.

        Override to handle additions/removals of securities.

        Args:
            added: Symbols added to universe
            removed: Symbols removed from universe
        """
        self.state.symbols.update(added)
        self.state.symbols -= set(removed)

        # Clear insights for removed symbols
        for symbol in removed:
            self._active_insights.pop(symbol, None)

        logger.debug(f"{self.name}: Universe changed. Added {len(added)}, removed {len(removed)}")

    def update(
        self,
        data: Dict[str, Any],
        symbols: Optional[List[str]] = None,
    ) -> List[Insight]:
        """
        Update model and generate new insights.

        Args:
            data: Market data
            symbols: Symbols to analyze (defaults to state.symbols)

        Returns:
            List of new insights
        """
        symbols = symbols or list(self.state.symbols)
        if not symbols:
            return []

        # Generate insights
        insights = self.generate_insights(data, symbols)

        # Tag insights with source model
        for insight in insights:
            insight.source_model = self.name

        # Update state
        self.state.last_update = datetime.now()
        self.state.insights_generated += len(insights)

        # Track active insights
        for insight in insights:
            self._active_insights[insight.symbol] = insight

        logger.debug(f"{self.name}: Generated {len(insights)} insights")
        return insights

    def score_insight(self, insight: Insight, actual_return: float) -> None:
        """
        Score an expired insight.

        Args:
            insight: The insight to score
            actual_return: Actual return during insight period
        """
        insight.update_score(actual_return)

        self.state.total_scored += 1
        if insight.score and insight.score.direction_score > 0:
            self.state.correct_predictions += 1

    def get_active_insight(self, symbol: str) -> Optional[Insight]:
        """Get the active insight for a symbol."""
        insight = self._active_insights.get(symbol)
        if insight and not insight.is_expired:
            return insight
        return None

    def get_all_active_insights(self) -> List[Insight]:
        """Get all non-expired insights."""
        return [
            insight for insight in self._active_insights.values()
            if not insight.is_expired
        ]

    def get_stats(self) -> Dict[str, Any]:
        """Get model statistics."""
        return {
            'name': self.name,
            'symbols_tracked': len(self.state.symbols),
            'insights_generated': self.state.insights_generated,
            'accuracy': self.state.accuracy,
            'active_insights': len(self.get_all_active_insights()),
            'last_update': self.state.last_update.isoformat() if self.state.last_update else None,
        }

    def reset(self) -> None:
        """Reset model state."""
        self.state = AlphaModelState()
        self._active_insights.clear()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, accuracy={self.state.accuracy:.1%})"
