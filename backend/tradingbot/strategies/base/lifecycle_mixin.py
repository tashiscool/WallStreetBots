"""
Strategy Lifecycle Mixin

Provides lifecycle hooks inspired by Lumibot's strategy lifecycle:
- initialize(), before_market_opens(), on_trading_iteration(),
  after_market_closes(), before_starting_trading(), on_abrupt_closing(),
  trace_stats()

Usage:
    class MyStrategy(BaseStrategy, LifecycleMixin):
        def initialize(self):
            self.sleeptime = "1D"

        def before_market_opens(self):
            self.get_watchlist()

        def on_trading_iteration(self):
            # Main logic
            pass
"""

import logging
from datetime import datetime, time
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class LifecycleMixin:
    """Mixin that adds lifecycle hooks to trading strategies."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._lifecycle_initialized = False
        self._stats_history: List[Dict[str, Any]] = []
        self._lifecycle_hooks: Dict[str, List[Callable]] = {
            "initialize": [],
            "before_market_opens": [],
            "on_trading_iteration": [],
            "after_market_closes": [],
            "before_starting_trading": [],
            "on_abrupt_closing": [],
        }

    def initialize(self) -> None:
        """Called once when the strategy starts.

        Override to set up initial state, watchlists, etc.
        """
        pass

    def before_market_opens(self) -> None:
        """Called before market opens each day.

        Override to refresh data, adjust parameters, etc.
        """
        pass

    def on_trading_iteration(self) -> None:
        """Called on each trading iteration (main strategy logic).

        Override with your core trading logic.
        """
        pass

    def after_market_closes(self) -> None:
        """Called after market closes each day.

        Override to log performance, clean up positions, etc.
        """
        pass

    def before_starting_trading(self) -> None:
        """Called once before the first trading iteration.

        Override for one-time setup that requires market connection.
        """
        pass

    def on_abrupt_closing(self) -> None:
        """Called when strategy is stopped unexpectedly.

        Override to handle cleanup, save state, close positions.
        """
        pass

    def trace_stats(self) -> Dict[str, Any]:
        """Return stats to be tracked each iteration.

        Override to define custom metrics to log.

        Returns:
            Dict of metric name -> value
        """
        return {}

    # --- Hook management ---

    def register_hook(self, event: str, callback: Callable) -> None:
        """Register a callback for a lifecycle event.

        Args:
            event: Lifecycle event name
            callback: Callable to invoke
        """
        if event in self._lifecycle_hooks:
            self._lifecycle_hooks[event].append(callback)
        else:
            logger.warning(f"Unknown lifecycle event: {event}")

    def _run_lifecycle(self, event: str) -> None:
        """Run a lifecycle event and all registered hooks."""
        # Run the method
        method = getattr(self, event, None)
        if method:
            try:
                method()
            except Exception as e:
                logger.error(f"Error in {event}: {e}")

        # Run hooks
        for hook in self._lifecycle_hooks.get(event, []):
            try:
                hook()
            except Exception as e:
                logger.error(f"Error in {event} hook: {e}")

    def run_lifecycle_cycle(self) -> None:
        """Run a full lifecycle cycle (typically called by scheduler)."""
        if not self._lifecycle_initialized:
            self._run_lifecycle("initialize")
            self._run_lifecycle("before_starting_trading")
            self._lifecycle_initialized = True

        self._run_lifecycle("before_market_opens")
        self._run_lifecycle("on_trading_iteration")

        # Track stats
        stats = self.trace_stats()
        if stats:
            stats["timestamp"] = datetime.now().isoformat()
            self._stats_history.append(stats)

        self._run_lifecycle("after_market_closes")

    def get_stats_history(self) -> List[Dict[str, Any]]:
        """Get historical stats from trace_stats()."""
        return self._stats_history
