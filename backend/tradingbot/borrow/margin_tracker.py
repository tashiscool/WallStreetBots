"""
Margin Tracker

Tracks margin requirements, buying power, and maintenance margins
for short selling and leveraged positions.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


class MarginStatus(Enum):
    """Margin account status."""
    HEALTHY = "healthy"  # > 50% equity
    CAUTION = "caution"  # 35-50% equity
    WARNING = "warning"  # 25-35% equity
    MARGIN_CALL = "margin_call"  # < 25% equity


class PositionType(Enum):
    """Type of position for margin calculation."""
    LONG_STOCK = "long_stock"
    SHORT_STOCK = "short_stock"
    LONG_OPTION = "long_option"
    SHORT_OPTION = "short_option"
    SPREAD = "spread"


@dataclass
class MarginRequirement:
    """Margin requirement for a position."""
    symbol: str
    position_type: PositionType
    initial_margin: Decimal  # Required to open
    maintenance_margin: Decimal  # Required to hold
    current_value: Decimal
    margin_used: Decimal

    @property
    def maintenance_ratio(self) -> float:
        if self.current_value > 0:
            return float(self.margin_used / self.current_value)
        return 0.0


@dataclass
class MarginSummary:
    """Summary of account margin status."""
    total_equity: Decimal
    total_positions_value: Decimal
    buying_power: Decimal
    margin_used: Decimal
    margin_available: Decimal
    maintenance_margin: Decimal
    maintenance_excess: Decimal  # Equity above maintenance
    margin_ratio: float  # Equity / Total Position Value
    status: MarginStatus
    timestamp: datetime

    @property
    def is_margin_call(self) -> bool:
        return self.status == MarginStatus.MARGIN_CALL

    @property
    def can_open_positions(self) -> bool:
        return self.margin_available > Decimal("0") and not self.is_margin_call


@dataclass
class MarginCallInfo:
    """Information about a margin call."""
    call_amount: Decimal
    due_by: datetime
    positions_at_risk: List[str]
    recommended_action: str


# Regulation T margin requirements
REG_T_INITIAL_MARGIN = Decimal("0.50")  # 50% initial for stocks
REG_T_MAINTENANCE_MARGIN = Decimal("0.25")  # 25% maintenance for stocks

# Short selling margins (typically higher)
SHORT_INITIAL_MARGIN = Decimal("0.50")  # 50% initial
SHORT_MAINTENANCE_MARGIN = Decimal("0.30")  # 30% maintenance

# Options margins (simplified)
OPTION_MARGIN_MULTIPLIER = Decimal("100")  # Per contract


class MarginTracker:
    """
    Tracks margin requirements and buying power.

    Features:
    - Real-time margin calculation
    - Maintenance margin monitoring
    - Margin call detection
    - Buying power calculation
    - Position-level margin requirements
    """

    def __init__(
        self,
        broker_client: Optional[Any] = None,
        initial_equity: Decimal = Decimal("100000"),
    ):
        """
        Initialize margin tracker.

        Args:
            broker_client: Optional broker for real margin data
            initial_equity: Starting account equity
        """
        self.broker = broker_client
        self.equity = initial_equity

        # Position tracking
        self._positions: Dict[str, MarginRequirement] = {}
        self._margin_history: List[MarginSummary] = []

        # Thresholds
        self.margin_call_threshold = Decimal("0.25")
        self.warning_threshold = Decimal("0.35")
        self.caution_threshold = Decimal("0.50")

    async def get_margin_summary(self, refresh: bool = True) -> MarginSummary:
        """
        Get current margin summary.

        Args:
            refresh: Refresh from broker if available

        Returns:
            MarginSummary with current status
        """
        if refresh and self.broker:
            await self._refresh_from_broker()

        # Calculate totals
        total_positions_value = sum(p.current_value for p in self._positions.values())
        margin_used = sum(p.margin_used for p in self._positions.values())
        maintenance_margin = sum(p.maintenance_margin for p in self._positions.values())

        # Calculate metrics
        margin_available = max(Decimal("0"), self.equity - margin_used)
        maintenance_excess = self.equity - maintenance_margin
        buying_power = self.equity * Decimal("2") - margin_used  # 2:1 leverage typical

        # Calculate margin ratio
        if total_positions_value > 0:
            margin_ratio = float(self.equity / total_positions_value)
        else:
            margin_ratio = 1.0

        # Determine status
        status = self._determine_status(margin_ratio, maintenance_excess)

        summary = MarginSummary(
            total_equity=self.equity,
            total_positions_value=total_positions_value,
            buying_power=max(Decimal("0"), buying_power),
            margin_used=margin_used,
            margin_available=margin_available,
            maintenance_margin=maintenance_margin,
            maintenance_excess=maintenance_excess,
            margin_ratio=margin_ratio,
            status=status,
            timestamp=datetime.now(),
        )

        # Store history
        self._margin_history.append(summary)

        return summary

    async def _refresh_from_broker(self) -> None:
        """Refresh data from broker."""
        try:
            if hasattr(self.broker, 'get_account'):
                account = await self.broker.get_account()
                if hasattr(account, 'equity'):
                    self.equity = Decimal(str(account.equity))
        except Exception as e:
            logger.warning(f"Failed to refresh from broker: {e}")

    def _determine_status(
        self,
        margin_ratio: float,
        maintenance_excess: Decimal,
    ) -> MarginStatus:
        """Determine margin account status."""
        if maintenance_excess < 0 or margin_ratio < float(self.margin_call_threshold):
            return MarginStatus.MARGIN_CALL
        elif margin_ratio < float(self.warning_threshold):
            return MarginStatus.WARNING
        elif margin_ratio < float(self.caution_threshold):
            return MarginStatus.CAUTION
        else:
            return MarginStatus.HEALTHY

    def calculate_short_margin(
        self,
        symbol: str,
        qty: int,
        price: Decimal,
    ) -> MarginRequirement:
        """
        Calculate margin requirement for a short position.

        Args:
            symbol: Stock symbol
            qty: Quantity to short
            price: Current price

        Returns:
            MarginRequirement for the position
        """
        position_value = price * qty

        # Reg T: 50% initial, 30% maintenance for shorts
        initial_margin = position_value * SHORT_INITIAL_MARGIN
        maintenance_margin = position_value * SHORT_MAINTENANCE_MARGIN

        return MarginRequirement(
            symbol=symbol,
            position_type=PositionType.SHORT_STOCK,
            initial_margin=initial_margin,
            maintenance_margin=maintenance_margin,
            current_value=position_value,
            margin_used=initial_margin,
        )

    def calculate_long_margin(
        self,
        symbol: str,
        qty: int,
        price: Decimal,
        use_margin: bool = True,
    ) -> MarginRequirement:
        """
        Calculate margin requirement for a long position.

        Args:
            symbol: Stock symbol
            qty: Quantity to buy
            price: Current price
            use_margin: Whether buying on margin

        Returns:
            MarginRequirement for the position
        """
        position_value = price * qty

        if use_margin:
            initial_margin = position_value * REG_T_INITIAL_MARGIN
            maintenance_margin = position_value * REG_T_MAINTENANCE_MARGIN
        else:
            # Cash account
            initial_margin = position_value
            maintenance_margin = Decimal("0")

        return MarginRequirement(
            symbol=symbol,
            position_type=PositionType.LONG_STOCK,
            initial_margin=initial_margin,
            maintenance_margin=maintenance_margin,
            current_value=position_value,
            margin_used=initial_margin if use_margin else position_value,
        )

    def add_position(self, requirement: MarginRequirement) -> bool:
        """
        Add a position to track.

        Args:
            requirement: Margin requirement for position

        Returns:
            True if position can be added (enough margin)
        """
        # Check if we have enough margin
        total_margin_used = sum(p.margin_used for p in self._positions.values())
        new_total = total_margin_used + requirement.margin_used

        if new_total > self.equity:
            logger.warning(
                f"Insufficient margin to add {requirement.symbol}: "
                f"need ${requirement.margin_used}, available ${self.equity - total_margin_used}"
            )
            return False

        self._positions[requirement.symbol] = requirement
        return True

    def update_position(
        self,
        symbol: str,
        current_price: Decimal,
    ) -> Optional[MarginRequirement]:
        """
        Update position value and margin requirement.

        Args:
            symbol: Stock symbol
            current_price: Current market price

        Returns:
            Updated MarginRequirement or None
        """
        if symbol not in self._positions:
            return None

        position = self._positions[symbol]
        qty = int(position.current_value / position.initial_margin * Decimal("2"))  # Estimate qty

        # Recalculate based on position type
        if position.position_type == PositionType.SHORT_STOCK:
            updated = self.calculate_short_margin(symbol, qty, current_price)
        elif position.position_type == PositionType.LONG_STOCK:
            updated = self.calculate_long_margin(symbol, qty, current_price)
        else:
            updated = position

        self._positions[symbol] = updated
        return updated

    def remove_position(self, symbol: str) -> Optional[MarginRequirement]:
        """Remove a position from tracking."""
        return self._positions.pop(symbol, None)

    def can_open_position(self, requirement: MarginRequirement) -> bool:
        """Check if a new position can be opened."""
        total_margin_used = sum(p.margin_used for p in self._positions.values())
        available = self.equity - total_margin_used
        return requirement.margin_used <= available

    def get_buying_power_for_short(self) -> Decimal:
        """Calculate available buying power for short selling."""
        summary = MarginSummary(
            total_equity=self.equity,
            total_positions_value=sum(p.current_value for p in self._positions.values()),
            buying_power=Decimal("0"),
            margin_used=sum(p.margin_used for p in self._positions.values()),
            margin_available=Decimal("0"),
            maintenance_margin=sum(p.maintenance_margin for p in self._positions.values()),
            maintenance_excess=Decimal("0"),
            margin_ratio=0.0,
            status=MarginStatus.HEALTHY,
            timestamp=datetime.now(),
        )

        # Short selling requires 50% margin
        available_margin = self.equity - summary.margin_used
        return available_margin * Decimal("2")  # 2:1 leverage

    async def check_margin_call(self) -> Optional[MarginCallInfo]:
        """
        Check if account is in margin call.

        Returns:
            MarginCallInfo if in margin call, None otherwise
        """
        summary = await self.get_margin_summary()

        if not summary.is_margin_call:
            return None

        # Calculate call amount
        call_amount = abs(summary.maintenance_excess)

        # Find positions at risk (largest first)
        positions_by_value = sorted(
            self._positions.items(),
            key=lambda x: x[1].current_value,
            reverse=True,
        )
        at_risk = [p[0] for p in positions_by_value[:3]]

        return MarginCallInfo(
            call_amount=call_amount,
            due_by=datetime.now() + timedelta(days=5),  # T+5 typical
            positions_at_risk=at_risk,
            recommended_action=(
                f"Deposit ${call_amount} or reduce positions by "
                f"${call_amount * Decimal('2')} in market value"
            ),
        )

    def get_position_margins(self) -> List[MarginRequirement]:
        """Get all position margin requirements."""
        return list(self._positions.values())

    def get_margin_history(
        self,
        hours: int = 24,
    ) -> List[MarginSummary]:
        """Get margin history."""
        cutoff = datetime.now() - timedelta(hours=hours)
        return [h for h in self._margin_history if h.timestamp >= cutoff]

    def estimate_liquidation_price(
        self,
        symbol: str,
    ) -> Optional[Decimal]:
        """
        Estimate price at which position would be liquidated.

        Args:
            symbol: Stock symbol

        Returns:
            Estimated liquidation price or None
        """
        if symbol not in self._positions:
            return None

        position = self._positions[symbol]

        if position.position_type == PositionType.SHORT_STOCK:
            # For shorts, liquidation when price rises too much
            # Simplified: maintenance margin = position value * 0.30
            # Equity = initial_margin - (current_price - entry_price) * qty
            # Margin call when equity < position_value * 0.25
            entry_value = position.current_value
            margin = position.margin_used

            # Solve for price where we hit margin call
            # This is simplified - actual calculation depends on account state
            max_loss = margin - (entry_value * self.margin_call_threshold)
            qty = int(entry_value / position.current_value)
            if qty > 0:
                return (entry_value + max_loss) / qty

        return None


def create_margin_tracker(
    broker_client: Optional[Any] = None,
    initial_equity: Decimal = Decimal("100000"),
) -> MarginTracker:
    """Factory function to create margin tracker."""
    return MarginTracker(broker_client, initial_equity)
