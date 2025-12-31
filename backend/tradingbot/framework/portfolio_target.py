"""
Portfolio Target

Represents a target position for a symbol, output from PortfolioConstructionModel.
"""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, Optional
import uuid


class TargetStatus(Enum):
    """Status of portfolio target execution."""
    PENDING = "pending"
    EXECUTING = "executing"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class PortfolioTarget:
    """
    Target position for a symbol.

    Output from PortfolioConstructionModel, input to ExecutionModel.

    Attributes:
        symbol: The ticker symbol
        quantity: Target number of shares/contracts (positive=long, negative=short)
        target_weight: Target portfolio weight (0.0 to 1.0)
        source_insight_id: ID of the insight that generated this target
        minimum_order_margin: Minimum margin required for order
    """
    symbol: str
    quantity: Decimal
    target_weight: float = 0.0
    source_insight_id: Optional[str] = None

    # Order constraints
    minimum_order_margin: Optional[Decimal] = None
    maximum_quantity: Optional[Decimal] = None
    limit_price: Optional[Decimal] = None
    stop_price: Optional[Decimal] = None

    # Metadata
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=datetime.now)
    status: TargetStatus = TargetStatus.PENDING
    tag: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Execution tracking
    filled_quantity: Decimal = field(default_factory=lambda: Decimal("0"))
    avg_fill_price: Optional[Decimal] = None
    execution_started_at: Optional[datetime] = None
    execution_completed_at: Optional[datetime] = None

    @property
    def is_long(self) -> bool:
        """Check if this is a long position target."""
        return self.quantity > 0

    @property
    def is_short(self) -> bool:
        """Check if this is a short position target."""
        return self.quantity < 0

    @property
    def is_liquidate(self) -> bool:
        """Check if this target liquidates the position."""
        return self.quantity == 0

    @property
    def remaining_quantity(self) -> Decimal:
        """Get remaining quantity to fill."""
        return abs(self.quantity) - abs(self.filled_quantity)

    @property
    def fill_percentage(self) -> float:
        """Get percentage of order filled."""
        if self.quantity == 0:
            return 100.0
        return float(abs(self.filled_quantity) / abs(self.quantity)) * 100

    @property
    def is_complete(self) -> bool:
        """Check if target is fully executed."""
        return self.status in (TargetStatus.FILLED, TargetStatus.CANCELLED, TargetStatus.REJECTED)

    def update_fill(self, filled_qty: Decimal, fill_price: Decimal) -> None:
        """
        Update with a fill.

        Args:
            filled_qty: Quantity filled in this update
            fill_price: Price of this fill
        """
        # Update average fill price
        if self.avg_fill_price is None:
            self.avg_fill_price = fill_price
        else:
            # Weighted average
            total_filled = self.filled_quantity + filled_qty
            if total_filled > 0:
                self.avg_fill_price = (
                    (self.avg_fill_price * self.filled_quantity + fill_price * filled_qty) /
                    total_filled
                )

        self.filled_quantity += filled_qty

        # Update status
        if self.filled_quantity >= abs(self.quantity):
            self.status = TargetStatus.FILLED
            self.execution_completed_at = datetime.now()
        else:
            self.status = TargetStatus.PARTIALLY_FILLED

    def mark_executing(self) -> None:
        """Mark target as executing."""
        self.status = TargetStatus.EXECUTING
        self.execution_started_at = datetime.now()

    def mark_cancelled(self) -> None:
        """Mark target as cancelled."""
        self.status = TargetStatus.CANCELLED
        self.execution_completed_at = datetime.now()

    def mark_rejected(self, reason: str = "") -> None:
        """Mark target as rejected."""
        self.status = TargetStatus.REJECTED
        self.execution_completed_at = datetime.now()
        self.metadata['rejection_reason'] = reason

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'id': self.id,
            'symbol': self.symbol,
            'quantity': str(self.quantity),
            'target_weight': self.target_weight,
            'source_insight_id': self.source_insight_id,
            'status': self.status.value,
            'created_at': self.created_at.isoformat(),
            'filled_quantity': str(self.filled_quantity),
            'avg_fill_price': str(self.avg_fill_price) if self.avg_fill_price else None,
            'tag': self.tag,
            'metadata': self.metadata,
        }

    @classmethod
    def liquidate(cls, symbol: str, source_insight_id: Optional[str] = None) -> 'PortfolioTarget':
        """Create a target that liquidates a position."""
        return cls(
            symbol=symbol,
            quantity=Decimal("0"),
            target_weight=0.0,
            source_insight_id=source_insight_id,
            tag="liquidate",
        )

    @classmethod
    def from_weight(
        cls,
        symbol: str,
        target_weight: float,
        portfolio_value: Decimal,
        current_price: Decimal,
        source_insight_id: Optional[str] = None,
    ) -> 'PortfolioTarget':
        """
        Create target from portfolio weight.

        Args:
            symbol: Ticker symbol
            target_weight: Target portfolio weight (e.g., 0.10 for 10%)
            portfolio_value: Total portfolio value
            current_price: Current price of the asset
            source_insight_id: Optional source insight ID
        """
        target_value = portfolio_value * Decimal(str(target_weight))
        quantity = (target_value / current_price).quantize(Decimal("1"))

        return cls(
            symbol=symbol,
            quantity=quantity,
            target_weight=target_weight,
            source_insight_id=source_insight_id,
        )

    def __repr__(self) -> str:
        return (
            f"PortfolioTarget({self.symbol}, qty={self.quantity}, "
            f"weight={self.target_weight:.1%}, status={self.status.value})"
        )
