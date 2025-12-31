"""
Backtesting Progress Monitor.

Ported from QuantConnect LEAN's progress monitoring system.
Provides real-time progress tracking, ETA estimation, and status updates.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Optional
import json
from collections import deque
import statistics

logger = logging.getLogger(__name__)


class BacktestStatus(Enum):
    """Backtest execution status."""
    PENDING = "pending"
    INITIALIZING = "initializing"
    LOADING_DATA = "loading_data"
    RUNNING = "running"
    ANALYZING = "analyzing"
    GENERATING_REPORT = "generating_report"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ProgressEventType(Enum):
    """Types of progress events."""
    STATUS_CHANGE = "status_change"
    PROGRESS_UPDATE = "progress_update"
    TRADE_EXECUTED = "trade_executed"
    ERROR = "error"
    WARNING = "warning"
    LOG = "log"
    METRIC_UPDATE = "metric_update"
    CHECKPOINT = "checkpoint"


@dataclass
class ProgressEvent:
    """A progress event during backtesting."""
    event_type: ProgressEventType
    timestamp: datetime
    message: str
    data: Optional[dict] = None
    progress: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "message": self.message,
            "data": self.data,
            "progress": self.progress,
        }


@dataclass
class BacktestMetrics:
    """Real-time backtest metrics."""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    current_equity: float = 0.0
    max_equity: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: Optional[float] = None
    win_rate: float = 0.0

    def to_dict(self) -> dict:
        return {
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "total_pnl": self.total_pnl,
            "current_equity": self.current_equity,
            "max_equity": self.max_equity,
            "max_drawdown": self.max_drawdown,
            "sharpe_ratio": self.sharpe_ratio,
            "win_rate": self.win_rate,
        }

    def update_win_rate(self) -> None:
        """Update win rate calculation."""
        if self.total_trades > 0:
            self.win_rate = self.winning_trades / self.total_trades * 100


@dataclass
class ProgressState:
    """Current progress state of a backtest."""
    backtest_id: str
    status: BacktestStatus = BacktestStatus.PENDING
    progress: float = 0.0  # 0-100
    current_date: Optional[datetime] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    elapsed_seconds: float = 0.0
    estimated_remaining_seconds: Optional[float] = None
    estimated_completion: Optional[datetime] = None
    metrics: BacktestMetrics = field(default_factory=BacktestMetrics)
    error_message: Optional[str] = None
    events: list[ProgressEvent] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "backtest_id": self.backtest_id,
            "status": self.status.value,
            "progress": self.progress,
            "current_date": self.current_date.isoformat() if self.current_date else None,
            "start_date": self.start_date.isoformat() if self.start_date else None,
            "end_date": self.end_date.isoformat() if self.end_date else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "elapsed_seconds": self.elapsed_seconds,
            "estimated_remaining_seconds": self.estimated_remaining_seconds,
            "estimated_completion": self.estimated_completion.isoformat() if self.estimated_completion else None,
            "metrics": self.metrics.to_dict(),
            "error_message": self.error_message,
            "events": [e.to_dict() for e in self.events[-50:]],  # Last 50 events
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


class ETAEstimator:
    """
    Estimates time to completion based on progress history.

    Uses exponential moving average for smoother estimates.
    """

    def __init__(self, window_size: int = 20, smoothing: float = 0.3):
        self.window_size = window_size
        self.smoothing = smoothing
        self._progress_history: deque = deque(maxlen=window_size)
        self._time_history: deque = deque(maxlen=window_size)
        self._ema_speed: Optional[float] = None

    def update(self, progress: float, timestamp: float) -> None:
        """Update with new progress data point."""
        self._progress_history.append(progress)
        self._time_history.append(timestamp)

        if len(self._progress_history) >= 2:
            # Calculate instantaneous speed (progress per second)
            progress_delta = self._progress_history[-1] - self._progress_history[-2]
            time_delta = self._time_history[-1] - self._time_history[-2]

            if time_delta > 0:
                instant_speed = progress_delta / time_delta

                # Update EMA
                if self._ema_speed is None:
                    self._ema_speed = instant_speed
                else:
                    self._ema_speed = (
                        self.smoothing * instant_speed +
                        (1 - self.smoothing) * self._ema_speed
                    )

    def estimate_remaining(self, current_progress: float) -> Optional[float]:
        """Estimate remaining time in seconds."""
        if self._ema_speed is None or self._ema_speed <= 0:
            return None

        remaining_progress = 100 - current_progress
        if remaining_progress <= 0:
            return 0

        return remaining_progress / self._ema_speed

    def reset(self) -> None:
        """Reset the estimator."""
        self._progress_history.clear()
        self._time_history.clear()
        self._ema_speed = None


class ProgressMonitor:
    """
    Real-time progress monitor for backtesting.

    Features:
    - Progress tracking with percentage completion
    - ETA estimation with adaptive smoothing
    - Real-time metrics updates
    - Event logging and checkpoints
    - Callback support for UI updates
    - WebSocket broadcast integration
    """

    def __init__(
        self,
        backtest_id: str,
        start_date: datetime,
        end_date: datetime,
        initial_capital: float = 100000,
        broadcast_interval: float = 1.0,
    ):
        """Initialize progress monitor."""
        self.state = ProgressState(
            backtest_id=backtest_id,
            start_date=start_date,
            end_date=end_date,
        )
        self.state.metrics.current_equity = initial_capital
        self.state.metrics.max_equity = initial_capital

        self._initial_capital = initial_capital
        self._broadcast_interval = broadcast_interval
        self._last_broadcast = 0.0
        self._eta_estimator = ETAEstimator()
        self._callbacks: list[Callable[[ProgressState], None]] = []
        self._async_callbacks: list[Callable[[ProgressState], Any]] = []
        self._start_time: Optional[float] = None
        self._is_running = False

    def add_callback(self, callback: Callable[[ProgressState], None]) -> None:
        """Add a progress callback."""
        self._callbacks.append(callback)

    def add_async_callback(self, callback: Callable[[ProgressState], Any]) -> None:
        """Add an async progress callback."""
        self._async_callbacks.append(callback)

    def remove_callback(self, callback: Callable) -> None:
        """Remove a callback."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)
        if callback in self._async_callbacks:
            self._async_callbacks.remove(callback)

    def _notify_callbacks(self) -> None:
        """Notify all callbacks of state change."""
        for callback in self._callbacks:
            try:
                callback(self.state)
            except Exception as e:
                logger.error(f"Progress callback error: {e}")

    async def _notify_async_callbacks(self) -> None:
        """Notify all async callbacks."""
        for callback in self._async_callbacks:
            try:
                result = callback(self.state)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error(f"Async progress callback error: {e}")

    def start(self) -> None:
        """Mark backtest as started."""
        self._start_time = time.time()
        self._is_running = True
        self.state.started_at = datetime.now()
        self.state.status = BacktestStatus.INITIALIZING
        self._add_event(
            ProgressEventType.STATUS_CHANGE,
            "Backtest started",
            {"status": self.state.status.value},
        )
        self._notify_callbacks()

    def set_status(self, status: BacktestStatus, message: Optional[str] = None) -> None:
        """Update backtest status."""
        old_status = self.state.status
        self.state.status = status
        self._add_event(
            ProgressEventType.STATUS_CHANGE,
            message or f"Status changed to {status.value}",
            {"old_status": old_status.value, "new_status": status.value},
        )
        self._notify_callbacks()

    def update_progress(
        self,
        current_date: datetime,
        force_broadcast: bool = False,
    ) -> None:
        """Update progress based on current simulation date."""
        if not self._is_running or self.state.start_date is None or self.state.end_date is None:
            return

        self.state.current_date = current_date

        # Calculate progress percentage
        total_days = (self.state.end_date - self.state.start_date).days
        if total_days > 0:
            elapsed_days = (current_date - self.state.start_date).days
            self.state.progress = min(100, max(0, (elapsed_days / total_days) * 100))

        # Update elapsed time
        if self._start_time:
            self.state.elapsed_seconds = time.time() - self._start_time

        # Update ETA
        self._eta_estimator.update(self.state.progress, time.time())
        remaining = self._eta_estimator.estimate_remaining(self.state.progress)
        if remaining is not None:
            self.state.estimated_remaining_seconds = remaining
            self.state.estimated_completion = datetime.now() + timedelta(seconds=remaining)

        # Broadcast if interval has passed
        current_time = time.time()
        if force_broadcast or (current_time - self._last_broadcast >= self._broadcast_interval):
            self._last_broadcast = current_time
            self._add_event(
                ProgressEventType.PROGRESS_UPDATE,
                f"Progress: {self.state.progress:.1f}%",
                {"progress": self.state.progress},
                progress=self.state.progress,
            )
            self._notify_callbacks()

    def record_trade(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        pnl: Optional[float] = None,
    ) -> None:
        """Record a trade execution."""
        self.state.metrics.total_trades += 1

        if pnl is not None:
            self.state.metrics.total_pnl += pnl
            if pnl >= 0:
                self.state.metrics.winning_trades += 1
            else:
                self.state.metrics.losing_trades += 1

        self.state.metrics.update_win_rate()

        self._add_event(
            ProgressEventType.TRADE_EXECUTED,
            f"Trade: {side.upper()} {quantity} {symbol} @ ${price:.2f}",
            {
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "price": price,
                "pnl": pnl,
            },
        )

    def update_equity(self, equity: float) -> None:
        """Update current equity."""
        self.state.metrics.current_equity = equity
        self.state.metrics.max_equity = max(self.state.metrics.max_equity, equity)

        # Update drawdown
        if self.state.metrics.max_equity > 0:
            drawdown = (self.state.metrics.max_equity - equity) / self.state.metrics.max_equity * 100
            self.state.metrics.max_drawdown = max(self.state.metrics.max_drawdown, drawdown)

    def update_metrics(self, metrics: dict) -> None:
        """Update multiple metrics at once."""
        for key, value in metrics.items():
            if hasattr(self.state.metrics, key):
                setattr(self.state.metrics, key, value)

        self._add_event(
            ProgressEventType.METRIC_UPDATE,
            "Metrics updated",
            metrics,
        )

    def add_checkpoint(self, name: str, data: Optional[dict] = None) -> None:
        """Add a progress checkpoint."""
        self._add_event(
            ProgressEventType.CHECKPOINT,
            f"Checkpoint: {name}",
            data or {},
        )
        self._notify_callbacks()

    def log(self, message: str, level: str = "info") -> None:
        """Add a log entry."""
        self._add_event(
            ProgressEventType.LOG,
            message,
            {"level": level},
        )

    def warning(self, message: str, data: Optional[dict] = None) -> None:
        """Add a warning."""
        self._add_event(
            ProgressEventType.WARNING,
            message,
            data,
        )
        logger.warning(f"Backtest {self.state.backtest_id}: {message}")

    def error(self, message: str, data: Optional[dict] = None) -> None:
        """Record an error."""
        self.state.error_message = message
        self._add_event(
            ProgressEventType.ERROR,
            message,
            data,
        )
        logger.error(f"Backtest {self.state.backtest_id}: {message}")
        self._notify_callbacks()

    def complete(self, results: Optional[dict] = None) -> None:
        """Mark backtest as completed."""
        self._is_running = False
        self.state.status = BacktestStatus.COMPLETED
        self.state.progress = 100.0
        self.state.completed_at = datetime.now()

        if self._start_time:
            self.state.elapsed_seconds = time.time() - self._start_time

        self._add_event(
            ProgressEventType.STATUS_CHANGE,
            "Backtest completed",
            {
                "status": "completed",
                "elapsed_seconds": self.state.elapsed_seconds,
                "total_trades": self.state.metrics.total_trades,
                "total_pnl": self.state.metrics.total_pnl,
            },
        )
        self._notify_callbacks()

    def fail(self, error_message: str) -> None:
        """Mark backtest as failed."""
        self._is_running = False
        self.state.status = BacktestStatus.FAILED
        self.state.error_message = error_message
        self.state.completed_at = datetime.now()

        self._add_event(
            ProgressEventType.ERROR,
            f"Backtest failed: {error_message}",
            {"error": error_message},
        )
        self._notify_callbacks()

    def cancel(self) -> None:
        """Mark backtest as cancelled."""
        self._is_running = False
        self.state.status = BacktestStatus.CANCELLED
        self.state.completed_at = datetime.now()

        self._add_event(
            ProgressEventType.STATUS_CHANGE,
            "Backtest cancelled",
            {"status": "cancelled"},
        )
        self._notify_callbacks()

    def _add_event(
        self,
        event_type: ProgressEventType,
        message: str,
        data: Optional[dict] = None,
        progress: Optional[float] = None,
    ) -> None:
        """Add an event to the log."""
        event = ProgressEvent(
            event_type=event_type,
            timestamp=datetime.now(),
            message=message,
            data=data,
            progress=progress or self.state.progress,
        )
        self.state.events.append(event)

        # Keep events list bounded
        if len(self.state.events) > 1000:
            self.state.events = self.state.events[-500:]

    def get_state(self) -> ProgressState:
        """Get current progress state."""
        return self.state

    def get_state_dict(self) -> dict:
        """Get current state as dictionary."""
        return self.state.to_dict()

    def get_formatted_eta(self) -> str:
        """Get formatted ETA string."""
        if self.state.estimated_remaining_seconds is None:
            return "Calculating..."

        remaining = self.state.estimated_remaining_seconds
        if remaining < 60:
            return f"{remaining:.0f}s"
        elif remaining < 3600:
            minutes = remaining / 60
            return f"{minutes:.1f}m"
        else:
            hours = remaining / 3600
            return f"{hours:.1f}h"

    def get_progress_bar(self, width: int = 50) -> str:
        """Get ASCII progress bar."""
        filled = int(self.state.progress / 100 * width)
        bar = "=" * filled + "-" * (width - filled)
        return f"[{bar}] {self.state.progress:.1f}%"


class BacktestProgressManager:
    """
    Manages progress monitors for multiple concurrent backtests.
    """

    def __init__(self):
        self._monitors: dict[str, ProgressMonitor] = {}

    def create_monitor(
        self,
        backtest_id: str,
        start_date: datetime,
        end_date: datetime,
        initial_capital: float = 100000,
    ) -> ProgressMonitor:
        """Create a new progress monitor."""
        monitor = ProgressMonitor(
            backtest_id=backtest_id,
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
        )
        self._monitors[backtest_id] = monitor
        return monitor

    def get_monitor(self, backtest_id: str) -> Optional[ProgressMonitor]:
        """Get a progress monitor by ID."""
        return self._monitors.get(backtest_id)

    def remove_monitor(self, backtest_id: str) -> None:
        """Remove a progress monitor."""
        if backtest_id in self._monitors:
            del self._monitors[backtest_id]

    def get_all_states(self) -> dict[str, dict]:
        """Get states of all monitors."""
        return {
            backtest_id: monitor.get_state_dict()
            for backtest_id, monitor in self._monitors.items()
        }

    def get_active_backtests(self) -> list[str]:
        """Get IDs of active backtests."""
        return [
            backtest_id
            for backtest_id, monitor in self._monitors.items()
            if monitor.state.status in (
                BacktestStatus.INITIALIZING,
                BacktestStatus.LOADING_DATA,
                BacktestStatus.RUNNING,
                BacktestStatus.ANALYZING,
                BacktestStatus.GENERATING_REPORT,
            )
        ]


# Global progress manager
progress_manager = BacktestProgressManager()


def get_progress_manager() -> BacktestProgressManager:
    """Get the global progress manager."""
    return progress_manager


def create_progress_monitor(
    backtest_id: str,
    start_date: datetime,
    end_date: datetime,
    initial_capital: float = 100000,
) -> ProgressMonitor:
    """Factory function to create a progress monitor."""
    return progress_manager.create_monitor(
        backtest_id=backtest_id,
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital,
    )
