"""
Consolidator Manager Module.

Manages multiple consolidators and their relationships with indicators.
"""

import logging
from datetime import timedelta
from decimal import Decimal
from typing import Any, Callable, Dict, List, Optional, Type, Union

from .base import (
    Bar,
    IDataConsolidator,
    Resolution,
    Tick,
)
from .period import (
    DailyConsolidator,
    HourlyConsolidator,
    MarketHoursConsolidator,
    MinuteConsolidator,
    MonthlyConsolidator,
    SessionConsolidator,
    TickTimePeriodConsolidator,
    TimePeriodConsolidator,
    WeeklyConsolidator,
)
from .count import (
    DollarVolumeConsolidator,
    TickCountConsolidator,
    TradeCountBarConsolidator,
    VolumeBarConsolidator,
    VolumeConsolidator,
)
from .renko import (
    ClassicRenkoConsolidator,
    RangeConsolidator,
    RenkoConsolidator,
    TickRangeConsolidator,
)

logger = logging.getLogger(__name__)


class ConsolidatorManager:
    """
    Manages data consolidators for a trading system.

    Handles consolidator lifecycle, indicator registration,
    and data routing.
    """

    def __init__(self):
        """Initialize consolidator manager."""
        self._consolidators: Dict[str, List[IDataConsolidator]] = {}
        self._indicator_registrations: Dict[str, List[Any]] = {}

    def add_consolidator(
        self,
        symbol: str,
        consolidator: IDataConsolidator,
    ) -> IDataConsolidator:
        """
        Add a consolidator for a symbol.

        Args:
            symbol: Symbol to consolidate
            consolidator: Consolidator instance

        Returns:
            The added consolidator
        """
        if symbol not in self._consolidators:
            self._consolidators[symbol] = []

        self._consolidators[symbol].append(consolidator)
        logger.info(
            f"Added {consolidator.__class__.__name__} for {symbol}"
        )
        return consolidator

    def remove_consolidator(
        self,
        symbol: str,
        consolidator: IDataConsolidator,
    ) -> bool:
        """
        Remove a consolidator.

        Args:
            symbol: Symbol
            consolidator: Consolidator to remove

        Returns:
            True if removed
        """
        if symbol in self._consolidators:
            if consolidator in self._consolidators[symbol]:
                self._consolidators[symbol].remove(consolidator)
                return True
        return False

    def get_consolidators(
        self,
        symbol: str,
    ) -> List[IDataConsolidator]:
        """Get all consolidators for a symbol."""
        return self._consolidators.get(symbol, [])

    def register_indicator(
        self,
        symbol: str,
        indicator: Any,
        consolidator: Optional[IDataConsolidator] = None,
    ) -> None:
        """
        Register an indicator to receive consolidated data.

        Args:
            symbol: Symbol for the indicator
            indicator: Indicator instance (must have update method)
            consolidator: Specific consolidator (or None for first)
        """
        if consolidator is None:
            consolidators = self.get_consolidators(symbol)
            if not consolidators:
                raise ValueError(f"No consolidator found for {symbol}")
            consolidator = consolidators[0]

        # Register callback
        def update_indicator(bar: Bar):
            if hasattr(indicator, 'update'):
                indicator.update(bar)

        consolidator.on_data_consolidated(update_indicator)

        # Track registration
        key = f"{symbol}:{id(consolidator)}"
        if key not in self._indicator_registrations:
            self._indicator_registrations[key] = []
        self._indicator_registrations[key].append(indicator)

        logger.debug(
            f"Registered {indicator.__class__.__name__} to "
            f"{consolidator.__class__.__name__} for {symbol}"
        )

    def process_bar(self, symbol: str, bar: Bar) -> None:
        """
        Process a bar through all consolidators for a symbol.

        Args:
            symbol: Symbol of the bar
            bar: Bar data to process
        """
        consolidators = self.get_consolidators(symbol)
        for consolidator in consolidators:
            try:
                consolidator.update(bar)
            except Exception as e:
                logger.error(
                    f"Error in consolidator {consolidator.__class__.__name__}: {e}"
                )

    def process_tick(self, symbol: str, tick: Tick) -> None:
        """
        Process a tick through all consolidators for a symbol.

        Args:
            symbol: Symbol of the tick
            tick: Tick data to process
        """
        consolidators = self.get_consolidators(symbol)
        for consolidator in consolidators:
            try:
                consolidator.update(tick)
            except Exception as e:
                logger.error(
                    f"Error in consolidator {consolidator.__class__.__name__}: {e}"
                )

    def reset_all(self) -> None:
        """Reset all consolidators."""
        for consolidators in self._consolidators.values():
            for consolidator in consolidators:
                consolidator.reset()

    def clear(self) -> None:
        """Clear all consolidators and registrations."""
        self._consolidators.clear()
        self._indicator_registrations.clear()


class ConsolidatorFactory:
    """
    Factory for creating common consolidators.

    Provides convenient methods for standard configurations.
    """

    @staticmethod
    def create_time_consolidator(
        symbol: str,
        resolution: Union[Resolution, timedelta],
        timezone: str = "America/New_York",
    ) -> IDataConsolidator:
        """
        Create a time-based consolidator.

        Args:
            symbol: Symbol to consolidate
            resolution: Time resolution
            timezone: Market timezone

        Returns:
            Appropriate time consolidator
        """
        if isinstance(resolution, Resolution):
            if resolution == Resolution.MINUTE:
                return MinuteConsolidator(symbol, minutes=1, timezone=timezone)
            elif resolution == Resolution.HOUR:
                return HourlyConsolidator(symbol, hours=1, timezone=timezone)
            elif resolution == Resolution.DAILY:
                return DailyConsolidator(symbol, timezone=timezone)
            elif resolution == Resolution.WEEKLY:
                return WeeklyConsolidator(symbol, timezone=timezone)
            elif resolution == Resolution.MONTHLY:
                return MonthlyConsolidator(symbol, timezone=timezone)
            else:
                raise ValueError(f"Unsupported resolution: {resolution}")
        else:
            return TimePeriodConsolidator(
                symbol=symbol,
                period=resolution,
                timezone=timezone,
            )

    @staticmethod
    def create_minute_consolidator(
        symbol: str,
        minutes: int = 1,
        timezone: str = "America/New_York",
    ) -> MinuteConsolidator:
        """Create N-minute consolidator."""
        return MinuteConsolidator(symbol, minutes=minutes, timezone=timezone)

    @staticmethod
    def create_hourly_consolidator(
        symbol: str,
        hours: int = 1,
        timezone: str = "America/New_York",
    ) -> HourlyConsolidator:
        """Create N-hour consolidator."""
        return HourlyConsolidator(symbol, hours=hours, timezone=timezone)

    @staticmethod
    def create_daily_consolidator(
        symbol: str,
        timezone: str = "America/New_York",
    ) -> DailyConsolidator:
        """Create daily consolidator."""
        return DailyConsolidator(symbol, timezone=timezone)

    @staticmethod
    def create_volume_consolidator(
        symbol: str,
        volume_threshold: int = 10000,
    ) -> VolumeConsolidator:
        """Create volume-based consolidator."""
        return VolumeConsolidator(symbol, volume_threshold=volume_threshold)

    @staticmethod
    def create_tick_count_consolidator(
        symbol: str,
        tick_count: int = 100,
    ) -> TickCountConsolidator:
        """Create tick count consolidator."""
        return TickCountConsolidator(symbol, tick_count=tick_count)

    @staticmethod
    def create_renko_consolidator(
        symbol: str,
        brick_size: Decimal,
        classic: bool = True,
    ) -> Union[RenkoConsolidator, ClassicRenkoConsolidator]:
        """Create Renko consolidator."""
        if classic:
            return ClassicRenkoConsolidator(symbol, brick_size=brick_size)
        return RenkoConsolidator(symbol, brick_size=brick_size)

    @staticmethod
    def create_range_consolidator(
        symbol: str,
        range_size: Decimal,
    ) -> RangeConsolidator:
        """Create range bar consolidator."""
        return RangeConsolidator(symbol, range_size=range_size)

    @staticmethod
    def create_market_hours_consolidator(
        symbol: str,
        period: timedelta,
        include_extended: bool = False,
        timezone: str = "America/New_York",
    ) -> MarketHoursConsolidator:
        """Create market hours aware consolidator."""
        return MarketHoursConsolidator(
            symbol=symbol,
            period=period,
            include_extended=include_extended,
            timezone=timezone,
        )


# Preset consolidator configurations
class ConsolidatorPresets:
    """Common consolidator configurations."""

    @staticmethod
    def day_trading_set(symbol: str) -> List[IDataConsolidator]:
        """
        Day trading consolidator set.

        1-min, 5-min, 15-min, 1-hour bars.
        """
        return [
            MinuteConsolidator(symbol, minutes=1),
            MinuteConsolidator(symbol, minutes=5),
            MinuteConsolidator(symbol, minutes=15),
            HourlyConsolidator(symbol, hours=1),
        ]

    @staticmethod
    def swing_trading_set(symbol: str) -> List[IDataConsolidator]:
        """
        Swing trading consolidator set.

        15-min, 1-hour, 4-hour, daily bars.
        """
        return [
            MinuteConsolidator(symbol, minutes=15),
            HourlyConsolidator(symbol, hours=1),
            HourlyConsolidator(symbol, hours=4),
            DailyConsolidator(symbol),
        ]

    @staticmethod
    def scalping_set(symbol: str) -> List[IDataConsolidator]:
        """
        Scalping consolidator set.

        Tick count, 1-min, volume bars.
        """
        return [
            TickCountConsolidator(symbol, tick_count=50),
            MinuteConsolidator(symbol, minutes=1),
            VolumeConsolidator(symbol, volume_threshold=5000),
        ]

    @staticmethod
    def options_trading_set(symbol: str) -> List[IDataConsolidator]:
        """
        Options trading consolidator set.

        5-min, 15-min, 1-hour, daily for options analysis.
        """
        return [
            MinuteConsolidator(symbol, minutes=5),
            MinuteConsolidator(symbol, minutes=15),
            HourlyConsolidator(symbol, hours=1),
            DailyConsolidator(symbol),
        ]

