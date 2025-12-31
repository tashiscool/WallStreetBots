"""
Insight - Trading Signal Representation

An Insight represents a prediction about future price movement,
inspired by QuantConnect Lean's Insight class.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, Optional
import uuid


class InsightDirection(Enum):
    """Direction of predicted price movement."""
    UP = 1
    DOWN = -1
    FLAT = 0


class InsightType(Enum):
    """Type of insight (what is being predicted)."""
    PRICE = "price"           # Price movement prediction
    VOLATILITY = "volatility"  # Volatility prediction
    MOMENTUM = "momentum"      # Momentum prediction
    ALPHA = "alpha"           # Alpha/excess return prediction


@dataclass
class InsightScore:
    """Real-time scoring of insight accuracy."""
    direction_score: float = 0.0  # 1.0 if direction correct, -1.0 if wrong
    magnitude_score: float = 0.0  # How accurate was magnitude prediction
    time_score: float = 0.0       # Did prediction occur in expected timeframe
    final_score: float = 0.0      # Weighted combination

    def calculate_final(
        self,
        direction_weight: float = 0.5,
        magnitude_weight: float = 0.3,
        time_weight: float = 0.2
    ) -> float:
        """Calculate weighted final score."""
        self.final_score = (
            direction_weight * self.direction_score +
            magnitude_weight * self.magnitude_score +
            time_weight * self.time_score
        )
        return self.final_score


@dataclass
class Insight:
    """
    A prediction about future price movement.

    Insights are the output of AlphaModels and serve as input to
    PortfolioConstructionModels.

    Attributes:
        symbol: The ticker symbol this insight applies to
        direction: Expected price direction (UP, DOWN, FLAT)
        magnitude: Expected percentage move (e.g., 0.05 for 5%)
        confidence: Confidence level 0-1 (1 = highly confident)
        period: How long the insight is valid
        source_model: Name of the AlphaModel that generated this
        insight_type: Type of prediction (price, volatility, etc.)
        weight: Suggested portfolio weight (optional)
        stop_loss: Suggested stop-loss level (optional)
        take_profit: Suggested take-profit level (optional)
    """
    symbol: str
    direction: InsightDirection
    magnitude: float = 0.0  # Expected % move
    confidence: float = 0.5  # 0-1 confidence
    period: timedelta = field(default_factory=lambda: timedelta(days=1))
    source_model: str = ""
    insight_type: InsightType = InsightType.PRICE
    weight: Optional[float] = None  # Suggested portfolio weight
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

    # Metadata
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    generated_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    tag: str = ""  # Custom tag for filtering
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Scoring (updated after insight expires)
    score: Optional[InsightScore] = None
    actual_direction: Optional[InsightDirection] = None
    actual_return: Optional[float] = None

    def __post_init__(self):
        """Set expiration time based on period."""
        if self.expires_at is None:
            self.expires_at = self.generated_at + self.period

    @property
    def is_expired(self) -> bool:
        """Check if insight has expired."""
        return datetime.now() > self.expires_at

    @property
    def is_long(self) -> bool:
        """Check if this is a bullish insight."""
        return self.direction == InsightDirection.UP

    @property
    def is_short(self) -> bool:
        """Check if this is a bearish insight."""
        return self.direction == InsightDirection.DOWN

    @property
    def is_neutral(self) -> bool:
        """Check if this is a neutral insight."""
        return self.direction == InsightDirection.FLAT

    @property
    def time_remaining(self) -> timedelta:
        """Get time remaining until expiration."""
        remaining = self.expires_at - datetime.now()
        return max(remaining, timedelta(0))

    @property
    def weighted_magnitude(self) -> float:
        """Get magnitude weighted by confidence."""
        return self.magnitude * self.confidence

    def update_score(
        self,
        actual_return: float,
        actual_direction: Optional[InsightDirection] = None
    ) -> InsightScore:
        """
        Update insight score based on actual outcome.

        Args:
            actual_return: Actual % return during insight period
            actual_direction: Actual direction (inferred from return if not provided)
        """
        self.actual_return = actual_return

        # Infer direction from return if not provided
        if actual_direction is None:
            if actual_return > 0.001:
                actual_direction = InsightDirection.UP
            elif actual_return < -0.001:
                actual_direction = InsightDirection.DOWN
            else:
                actual_direction = InsightDirection.FLAT

        self.actual_direction = actual_direction

        # Calculate scores
        score = InsightScore()

        # Direction score: +1 if correct, -1 if wrong, 0 if flat
        if self.direction == InsightDirection.FLAT:
            score.direction_score = 0.5 if actual_direction == InsightDirection.FLAT else 0.0
        else:
            if self.direction == actual_direction:
                score.direction_score = 1.0
            elif actual_direction == InsightDirection.FLAT:
                score.direction_score = 0.0
            else:
                score.direction_score = -1.0

        # Magnitude score: how close was prediction
        if self.magnitude > 0:
            mag_error = abs(abs(actual_return) - self.magnitude) / self.magnitude
            score.magnitude_score = max(0, 1.0 - mag_error)
        else:
            score.magnitude_score = 0.5

        # Time score: did it happen in expected timeframe?
        # (This would require tracking when the move occurred)
        score.time_score = 0.5  # Default to neutral

        score.calculate_final()
        self.score = score
        return score

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'id': self.id,
            'symbol': self.symbol,
            'direction': self.direction.name,
            'magnitude': self.magnitude,
            'confidence': self.confidence,
            'period_seconds': self.period.total_seconds(),
            'source_model': self.source_model,
            'insight_type': self.insight_type.value,
            'weight': self.weight,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'generated_at': self.generated_at.isoformat(),
            'expires_at': self.expires_at.isoformat(),
            'tag': self.tag,
            'metadata': self.metadata,
            'score': {
                'direction_score': self.score.direction_score,
                'magnitude_score': self.score.magnitude_score,
                'final_score': self.score.final_score,
            } if self.score else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Insight':
        """Create Insight from dictionary."""
        return cls(
            id=data.get('id', str(uuid.uuid4())),
            symbol=data['symbol'],
            direction=InsightDirection[data['direction']],
            magnitude=data.get('magnitude', 0.0),
            confidence=data.get('confidence', 0.5),
            period=timedelta(seconds=data.get('period_seconds', 86400)),
            source_model=data.get('source_model', ''),
            insight_type=InsightType(data.get('insight_type', 'price')),
            weight=data.get('weight'),
            stop_loss=data.get('stop_loss'),
            take_profit=data.get('take_profit'),
            generated_at=datetime.fromisoformat(data['generated_at']) if 'generated_at' in data else datetime.now(),
            expires_at=datetime.fromisoformat(data['expires_at']) if 'expires_at' in data else None,
            tag=data.get('tag', ''),
            metadata=data.get('metadata', {}),
        )

    def __repr__(self) -> str:
        return (
            f"Insight({self.symbol}, {self.direction.name}, "
            f"mag={self.magnitude:.2%}, conf={self.confidence:.1%}, "
            f"source={self.source_model})"
        )
