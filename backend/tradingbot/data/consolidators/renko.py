"""
Renko and Range-based Data Consolidators.

Price-based consolidation that filters out noise.
"""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import List, Optional

from .base import (
    Bar,
    BarType,
    IDataConsolidator,
    Tick,
)


class RenkoType(Enum):
    """Type of Renko brick calculation."""
    CLASSIC = "classic"       # Traditional Renko
    ATR = "atr"               # ATR-based brick size
    VOLUME = "volume"         # Volume-weighted


@dataclass
class RenkoBrick:
    """
    Single Renko brick.

    Represents a fixed-size price movement.
    """
    symbol: str
    open: Decimal
    close: Decimal
    high: Decimal
    low: Decimal
    volume: int = 0
    timestamp: datetime = field(default_factory=datetime.now)
    direction: int = 1  # 1 = up, -1 = down

    @property
    def is_up(self) -> bool:
        """True if bullish brick."""
        return self.direction > 0

    @property
    def is_down(self) -> bool:
        """True if bearish brick."""
        return self.direction < 0

    @property
    def size(self) -> Decimal:
        """Brick size."""
        return abs(self.close - self.open)

    def to_bar(self) -> Bar:
        """Convert to standard Bar."""
        return Bar(
            symbol=self.symbol,
            open=self.open,
            high=self.high,
            low=self.low,
            close=self.close,
            volume=self.volume,
            timestamp=self.timestamp,
            bar_type=BarType.TRADE,
        )


class RenkoConsolidator(IDataConsolidator[Bar, RenkoBrick]):
    """
    Classic Renko consolidator.

    Creates fixed-size bricks based on price movement.
    Filters out time and focuses on significant price moves.
    """

    def __init__(
        self,
        symbol: str,
        brick_size: Decimal,
        use_close_price: bool = True,
    ):
        """
        Initialize Renko consolidator.

        Args:
            symbol: Symbol to consolidate
            brick_size: Size of each brick in price units
            use_close_price: Use close price (True) or high/low (False)
        """
        super().__init__(symbol)
        self.brick_size = brick_size
        self.use_close_price = use_close_price
        self._last_brick: Optional[RenkoBrick] = None
        self._pending_volume = 0

    def _reference_price(self, data: Bar) -> Decimal:
        """Get the reference price for brick calculation.

        When ``use_close_price`` is True (default), uses the bar's close.
        Otherwise uses the HL/2 midpoint ``(high + low) / 2``, which smooths
        out intrabar wicks and reduces noise-driven bricks.
        """
        if self.use_close_price:
            return data.close
        return (data.high + data.low) / 2

    def should_consolidate(self, data: Bar) -> bool:
        """Check if price moved enough for new brick."""
        if self._last_brick is None:
            return True

        price = self._reference_price(data)
        move = abs(price - self._last_brick.close)
        return move >= self.brick_size

    def update(self, data: Bar) -> Optional[RenkoBrick]:
        """
        Process a bar.

        May produce multiple bricks if price moved significantly.

        Args:
            data: Input bar

        Returns:
            First new brick, or None
        """
        self._input_count += 1
        self._pending_volume += data.volume

        price = self._reference_price(data)

        # First bar - establish starting point
        if self._last_brick is None:
            # Round to nearest brick level
            brick_level = (price // self.brick_size) * self.brick_size
            self._last_brick = RenkoBrick(
                symbol=self.symbol,
                open=brick_level,
                close=brick_level + self.brick_size,
                high=brick_level + self.brick_size,
                low=brick_level,
                volume=self._pending_volume,
                timestamp=data.timestamp,
                direction=1,
            )
            self._pending_volume = 0
            return None

        # Calculate price change
        up_move = price - self._last_brick.close
        down_move = self._last_brick.open - price

        result = None

        # Check for upward bricks
        while up_move >= self.brick_size:
            new_brick = RenkoBrick(
                symbol=self.symbol,
                open=self._last_brick.close,
                close=self._last_brick.close + self.brick_size,
                high=self._last_brick.close + self.brick_size,
                low=self._last_brick.close,
                volume=self._pending_volume,
                timestamp=data.timestamp,
                direction=1,
            )
            self._last_brick = new_brick
            self._pending_volume = 0
            self.emit(new_brick)
            if result is None:
                result = new_brick
            up_move -= self.brick_size

        # Check for downward bricks
        while down_move >= self.brick_size:
            new_brick = RenkoBrick(
                symbol=self.symbol,
                open=self._last_brick.open,
                close=self._last_brick.open - self.brick_size,
                high=self._last_brick.open,
                low=self._last_brick.open - self.brick_size,
                volume=self._pending_volume,
                timestamp=data.timestamp,
                direction=-1,
            )
            self._last_brick = new_brick
            self._pending_volume = 0
            self.emit(new_brick)
            if result is None:
                result = new_brick
            down_move -= self.brick_size

        return result


class ClassicRenkoConsolidator(RenkoConsolidator):
    """
    Classic Renko with reversal requirement.

    Requires 2x brick size to reverse direction.
    """

    def __init__(
        self,
        symbol: str,
        brick_size: Decimal,
    ):
        """
        Initialize classic Renko.

        Args:
            symbol: Symbol to consolidate
            brick_size: Size of each brick
        """
        super().__init__(
            symbol=symbol,
            brick_size=brick_size,
            use_close_price=True,
        )

    def update(self, data: Bar) -> Optional[RenkoBrick]:
        """Process bar with 2x reversal requirement."""
        self._input_count += 1
        self._pending_volume += data.volume
        price = data.close

        # First bar
        if self._last_brick is None:
            brick_level = (price // self.brick_size) * self.brick_size
            self._last_brick = RenkoBrick(
                symbol=self.symbol,
                open=brick_level,
                close=brick_level + self.brick_size,
                high=brick_level + self.brick_size,
                low=brick_level,
                volume=self._pending_volume,
                timestamp=data.timestamp,
                direction=1,
            )
            self._pending_volume = 0
            return None

        result = None

        # Continue in same direction
        if self._last_brick.is_up:
            up_move = price - self._last_brick.close
            down_move = self._last_brick.open - price  # Need 2x to reverse

            while up_move >= self.brick_size:
                new_brick = RenkoBrick(
                    symbol=self.symbol,
                    open=self._last_brick.close,
                    close=self._last_brick.close + self.brick_size,
                    high=self._last_brick.close + self.brick_size,
                    low=self._last_brick.close,
                    volume=self._pending_volume,
                    timestamp=data.timestamp,
                    direction=1,
                )
                self._last_brick = new_brick
                self._pending_volume = 0
                self.emit(new_brick)
                if result is None:
                    result = new_brick
                up_move -= self.brick_size

            # Reversal requires 2 bricks worth
            if down_move >= 2 * self.brick_size:
                new_brick = RenkoBrick(
                    symbol=self.symbol,
                    open=self._last_brick.open,
                    close=self._last_brick.open - self.brick_size,
                    high=self._last_brick.open,
                    low=self._last_brick.open - self.brick_size,
                    volume=self._pending_volume,
                    timestamp=data.timestamp,
                    direction=-1,
                )
                self._last_brick = new_brick
                self._pending_volume = 0
                self.emit(new_brick)
                if result is None:
                    result = new_brick

        else:  # Last brick was down
            down_move = self._last_brick.open - price
            up_move = price - self._last_brick.close  # Need 2x to reverse

            while down_move >= self.brick_size:
                new_brick = RenkoBrick(
                    symbol=self.symbol,
                    open=self._last_brick.open,
                    close=self._last_brick.open - self.brick_size,
                    high=self._last_brick.open,
                    low=self._last_brick.open - self.brick_size,
                    volume=self._pending_volume,
                    timestamp=data.timestamp,
                    direction=-1,
                )
                self._last_brick = new_brick
                self._pending_volume = 0
                self.emit(new_brick)
                if result is None:
                    result = new_brick
                down_move -= self.brick_size

            # Reversal requires 2 bricks worth
            if up_move >= 2 * self.brick_size:
                new_brick = RenkoBrick(
                    symbol=self.symbol,
                    open=self._last_brick.close,
                    close=self._last_brick.close + self.brick_size,
                    high=self._last_brick.close + self.brick_size,
                    low=self._last_brick.close,
                    volume=self._pending_volume,
                    timestamp=data.timestamp,
                    direction=1,
                )
                self._last_brick = new_brick
                self._pending_volume = 0
                self.emit(new_brick)
                if result is None:
                    result = new_brick

        return result


class RangeConsolidator(IDataConsolidator[Bar, Bar]):
    """
    Range bar consolidator.

    Creates bars of fixed price range (high - low).
    """

    def __init__(
        self,
        symbol: str,
        range_size: Decimal,
    ):
        """
        Initialize range consolidator.

        Args:
            symbol: Symbol to consolidate
            range_size: Fixed range for each bar
        """
        super().__init__(symbol)
        self.range_size = range_size
        self._bar_high: Optional[Decimal] = None
        self._bar_low: Optional[Decimal] = None
        self._bar_open: Optional[Decimal] = None
        self._bar_close: Optional[Decimal] = None
        self._bar_volume: int = 0
        self._bar_start: Optional[datetime] = None

    def should_consolidate(self, data: Bar) -> bool:
        """Check if range exceeded."""
        if self._bar_high is None:
            return False

        potential_high = max(self._bar_high, data.high)
        potential_low = min(self._bar_low, data.low)

        return (potential_high - potential_low) > self.range_size

    def update(self, data: Bar) -> Optional[Bar]:
        """Process a bar."""
        self._input_count += 1
        result = None

        # Check if bar complete
        if self.should_consolidate(data):
            result = Bar(
                symbol=self.symbol,
                open=self._bar_open,
                high=self._bar_high,
                low=self._bar_low,
                close=self._bar_close,
                volume=self._bar_volume,
                timestamp=self._bar_start,
                end_time=data.timestamp,
            )
            self.emit(result)
            self._bar_high = None

        # Start new bar or update existing
        if self._bar_high is None:
            self._bar_open = data.open
            self._bar_high = data.high
            self._bar_low = data.low
            self._bar_close = data.close
            self._bar_volume = data.volume
            self._bar_start = data.timestamp
        else:
            self._bar_high = max(self._bar_high, data.high)
            self._bar_low = min(self._bar_low, data.low)
            self._bar_close = data.close
            self._bar_volume += data.volume

        return result

    def reset(self) -> None:
        """Reset state."""
        super().reset()
        self._bar_high = None
        self._bar_low = None
        self._bar_open = None
        self._bar_close = None
        self._bar_volume = 0
        self._bar_start = None


class TickRangeConsolidator(IDataConsolidator[Tick, Bar]):
    """
    Range bar consolidator from ticks.

    Creates bars of fixed price range from raw ticks.
    """

    def __init__(
        self,
        symbol: str,
        range_size: Decimal,
    ):
        """
        Initialize tick range consolidator.

        Args:
            symbol: Symbol to consolidate
            range_size: Fixed range for each bar
        """
        super().__init__(symbol)
        self.range_size = range_size
        self._bar_high: Optional[Decimal] = None
        self._bar_low: Optional[Decimal] = None
        self._bar_open: Optional[Decimal] = None
        self._bar_close: Optional[Decimal] = None
        self._bar_volume: int = 0
        self._bar_start: Optional[datetime] = None

    def should_consolidate(self, data: Tick) -> bool:
        """Check if range exceeded."""
        if self._bar_high is None:
            return False

        potential_high = max(self._bar_high, data.price)
        potential_low = min(self._bar_low, data.price)

        return (potential_high - potential_low) > self.range_size

    def update(self, data: Tick) -> Optional[Bar]:
        """Process a tick."""
        self._input_count += 1
        result = None

        # Check if bar complete
        if self.should_consolidate(data):
            result = Bar(
                symbol=self.symbol,
                open=self._bar_open,
                high=self._bar_high,
                low=self._bar_low,
                close=self._bar_close,
                volume=self._bar_volume,
                timestamp=self._bar_start,
                end_time=data.timestamp,
            )
            self.emit(result)
            self._bar_high = None

        # Start new bar or update existing
        if self._bar_high is None:
            self._bar_open = data.price
            self._bar_high = data.price
            self._bar_low = data.price
            self._bar_close = data.price
            self._bar_volume = data.quantity
            self._bar_start = data.timestamp
        else:
            self._bar_high = max(self._bar_high, data.price)
            self._bar_low = min(self._bar_low, data.price)
            self._bar_close = data.price
            self._bar_volume += data.quantity

        return result
