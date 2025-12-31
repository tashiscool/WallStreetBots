"""
Count-based Data Consolidators.

Consolidate data based on tick count, volume, or value.
"""

from datetime import datetime
from decimal import Decimal
from typing import Optional

from .base import (
    Bar,
    BarConsolidatorBase,
    Tick,
    TickConsolidatorBase,
)


class TickCountConsolidator(TickConsolidatorBase):
    """
    Consolidate every N ticks into a bar.

    Creates time-independent bars based on activity.
    """

    def __init__(
        self,
        symbol: str,
        tick_count: int = 100,
    ):
        """
        Initialize tick count consolidator.

        Args:
            symbol: Symbol to consolidate
            tick_count: Number of ticks per bar
        """
        super().__init__(symbol)
        self.tick_count = tick_count
        self._current_tick_count = 0

    def should_consolidate(self, data: Tick) -> bool:
        """Check if tick count reached."""
        return self._current_tick_count >= self.tick_count

    def update(self, data: Tick) -> Optional[Bar]:
        """
        Process a tick.

        Args:
            data: Input tick

        Returns:
            Bar when tick count reached
        """
        self._input_count += 1
        result = None

        # Check if bar complete
        if self._working_bar is not None and self.should_consolidate(data):
            result = self._working_bar
            self.emit(result)
            self._working_bar = None
            self._current_tick_count = 0

        # Start new bar or update existing
        if self._working_bar is None:
            self._working_bar = self._create_new_bar(data)
            self._current_tick_count = 1
        else:
            self._working_bar = self._update_bar(self._working_bar, data)
            self._current_tick_count += 1

        return result

    def reset(self) -> None:
        """Reset consolidator state."""
        super().reset()
        self._current_tick_count = 0


class VolumeConsolidator(TickConsolidatorBase):
    """
    Consolidate ticks by volume.

    Creates a new bar every N shares/contracts traded.
    """

    def __init__(
        self,
        symbol: str,
        volume_threshold: int = 10000,
    ):
        """
        Initialize volume consolidator.

        Args:
            symbol: Symbol to consolidate
            volume_threshold: Volume per bar
        """
        super().__init__(symbol)
        self.volume_threshold = volume_threshold
        self._current_volume = 0

    def should_consolidate(self, data: Tick) -> bool:
        """Check if volume threshold reached."""
        return self._current_volume >= self.volume_threshold

    def update(self, data: Tick) -> Optional[Bar]:
        """
        Process a tick.

        Args:
            data: Input tick

        Returns:
            Bar when volume threshold reached
        """
        self._input_count += 1
        result = None

        # Check if bar complete
        if self._working_bar is not None and self.should_consolidate(data):
            result = self._working_bar
            self.emit(result)
            self._working_bar = None
            self._current_volume = 0

        # Start new bar or update existing
        if self._working_bar is None:
            self._working_bar = self._create_new_bar(data)
            self._current_volume = data.quantity
        else:
            self._working_bar = self._update_bar(self._working_bar, data)
            self._current_volume += data.quantity

        return result

    def reset(self) -> None:
        """Reset consolidator state."""
        super().reset()
        self._current_volume = 0


class VolumeBarConsolidator(BarConsolidatorBase):
    """
    Consolidate bars by cumulative volume.

    Creates a new bar every N shares/contracts.
    """

    def __init__(
        self,
        symbol: str,
        volume_threshold: int = 100000,
    ):
        """
        Initialize volume bar consolidator.

        Args:
            symbol: Symbol to consolidate
            volume_threshold: Volume per consolidated bar
        """
        super().__init__(symbol)
        self.volume_threshold = volume_threshold
        self._current_volume = 0

    def should_consolidate(self, data: Bar) -> bool:
        """Check if volume threshold reached."""
        return self._current_volume >= self.volume_threshold

    def update(self, data: Bar) -> Optional[Bar]:
        """Process a bar."""
        self._input_count += 1
        result = None

        # Check if bar complete
        if self._working_bar is not None and self.should_consolidate(data):
            result = self._working_bar
            self.emit(result)
            self._working_bar = None
            self._current_volume = 0

        # Start new bar or update existing
        if self._working_bar is None:
            self._working_bar = self._create_new_bar(data)
            self._current_volume = data.volume
        else:
            self._working_bar = self._update_bar(self._working_bar, data)
            self._current_volume += data.volume

        return result


class DollarVolumeConsolidator(TickConsolidatorBase):
    """
    Consolidate by dollar volume (price * quantity).

    Normalizes bars by notional value traded.
    """

    def __init__(
        self,
        symbol: str,
        dollar_threshold: Decimal = Decimal("1000000"),
    ):
        """
        Initialize dollar volume consolidator.

        Args:
            symbol: Symbol to consolidate
            dollar_threshold: Dollar volume per bar
        """
        super().__init__(symbol)
        self.dollar_threshold = dollar_threshold
        self._current_dollar_volume = Decimal("0")

    def should_consolidate(self, data: Tick) -> bool:
        """Check if dollar volume threshold reached."""
        return self._current_dollar_volume >= self.dollar_threshold

    def update(self, data: Tick) -> Optional[Bar]:
        """Process a tick."""
        self._input_count += 1
        result = None

        # Check if bar complete
        if self._working_bar is not None and self.should_consolidate(data):
            result = self._working_bar
            self.emit(result)
            self._working_bar = None
            self._current_dollar_volume = Decimal("0")

        # Calculate dollar volume for this tick
        tick_dollar_volume = data.price * data.quantity

        # Start new bar or update existing
        if self._working_bar is None:
            self._working_bar = self._create_new_bar(data)
            self._current_dollar_volume = tick_dollar_volume
        else:
            self._working_bar = self._update_bar(self._working_bar, data)
            self._current_dollar_volume += tick_dollar_volume

        return result

    def reset(self) -> None:
        """Reset consolidator state."""
        super().reset()
        self._current_dollar_volume = Decimal("0")


class TradeCountBarConsolidator(BarConsolidatorBase):
    """
    Consolidate every N input bars.

    Simple count-based bar aggregation.
    """

    def __init__(
        self,
        symbol: str,
        bar_count: int = 5,
    ):
        """
        Initialize bar count consolidator.

        Args:
            symbol: Symbol to consolidate
            bar_count: Number of input bars per output bar
        """
        super().__init__(symbol)
        self.bar_count = bar_count
        self._current_bar_count = 0

    def should_consolidate(self, data: Bar) -> bool:
        """Check if bar count reached."""
        return self._current_bar_count >= self.bar_count

    def update(self, data: Bar) -> Optional[Bar]:
        """Process a bar."""
        self._input_count += 1
        result = None

        # Check if bar complete
        if self._working_bar is not None and self.should_consolidate(data):
            result = self._working_bar
            self.emit(result)
            self._working_bar = None
            self._current_bar_count = 0

        # Start new bar or update existing
        if self._working_bar is None:
            self._working_bar = self._create_new_bar(data)
            self._current_bar_count = 1
        else:
            self._working_bar = self._update_bar(self._working_bar, data)
            self._current_bar_count += 1

        return result

