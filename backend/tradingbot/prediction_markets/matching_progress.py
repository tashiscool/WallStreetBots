"""
Cross-Platform Market Matching Progress Tracker.

Synthesized from:
- polymarket-arbitrage: Matching progress display
- Real-time progress callbacks

Tracks market matching operations for dashboard display.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
import threading

logger = logging.getLogger(__name__)


class MatchingStatus(Enum):
    """Status of matching operation."""
    IDLE = "idle"
    LOADING_POLYMARKET = "loading_polymarket"
    LOADING_KALSHI = "loading_kalshi"
    MATCHING = "matching"
    COMPLETE = "complete"
    ERROR = "error"


@dataclass
class MatchingProgress:
    """
    Tracks progress of cross-platform market matching.

    From polymarket-arbitrage: Progress tracking with callbacks.
    """
    # Status
    status: MatchingStatus = MatchingStatus.IDLE
    error_message: Optional[str] = None

    # Market counts
    polymarket_total: int = 0
    polymarket_loaded: int = 0
    kalshi_total: int = 0
    kalshi_loaded: int = 0

    # Matching
    pairs_to_match: int = 0
    pairs_matched: int = 0
    matched_pairs: int = 0  # Successful matches

    # Timing
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Callbacks
    _progress_callbacks: List[Callable] = field(default_factory=list)

    @property
    def progress_pct(self) -> float:
        """Overall progress percentage."""
        if self.status == MatchingStatus.IDLE:
            return 0.0
        if self.status == MatchingStatus.COMPLETE:
            return 100.0
        if self.status == MatchingStatus.ERROR:
            return self._calculate_progress()

        return self._calculate_progress()

    def _calculate_progress(self) -> float:
        """Calculate progress based on current phase."""
        # Phase 1: Loading Polymarket (0-30%)
        # Phase 2: Loading Kalshi (30-60%)
        # Phase 3: Matching (60-100%)

        if self.status == MatchingStatus.LOADING_POLYMARKET:
            if self.polymarket_total == 0:
                return 0.0
            return (self.polymarket_loaded / self.polymarket_total) * 30

        elif self.status == MatchingStatus.LOADING_KALSHI:
            poly_progress = 30.0  # Polymarket done
            if self.kalshi_total == 0:
                return poly_progress
            kalshi_progress = (self.kalshi_loaded / self.kalshi_total) * 30
            return poly_progress + kalshi_progress

        elif self.status == MatchingStatus.MATCHING:
            loading_progress = 60.0  # Loading done
            if self.pairs_to_match == 0:
                return loading_progress
            matching_progress = (self.pairs_matched / self.pairs_to_match) * 40
            return loading_progress + matching_progress

        return 0.0

    @property
    def duration_seconds(self) -> float:
        """Duration of matching operation."""
        if self.started_at is None:
            return 0.0
        end = self.completed_at or datetime.now()
        return (end - self.started_at).total_seconds()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "status": self.status.value,
            "error_message": self.error_message,
            "polymarket_total": self.polymarket_total,
            "polymarket_loaded": self.polymarket_loaded,
            "kalshi_total": self.kalshi_total,
            "kalshi_loaded": self.kalshi_loaded,
            "pairs_to_match": self.pairs_to_match,
            "pairs_matched": self.pairs_matched,
            "matched_pairs": self.matched_pairs,
            "progress_pct": self.progress_pct,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_seconds": self.duration_seconds,
        }


class MatchingProgressTracker:
    """
    Manages matching progress with callbacks.

    Usage:
        tracker = MatchingProgressTracker()
        tracker.on_progress(lambda p: print(f"Progress: {p.progress_pct}%"))

        tracker.start_matching()
        tracker.set_polymarket_total(100)
        for i in range(100):
            tracker.increment_polymarket_loaded()
        # ... etc
    """

    def __init__(self):
        self._progress = MatchingProgress()
        self._callbacks: List[Callable[[MatchingProgress], None]] = []
        self._lock = threading.RLock()

    def on_progress(
        self,
        callback: Callable[[MatchingProgress], None],
    ) -> None:
        """Register progress callback."""
        self._callbacks.append(callback)

    def _notify(self) -> None:
        """Notify all callbacks."""
        progress = self._progress
        for callback in self._callbacks:
            try:
                callback(progress)
            except Exception as e:
                logger.error(f"Error in progress callback: {e}")

    def get_progress(self) -> MatchingProgress:
        """Get current progress."""
        with self._lock:
            return self._progress

    def start_matching(self) -> None:
        """Start matching operation."""
        with self._lock:
            self._progress = MatchingProgress(
                status=MatchingStatus.LOADING_POLYMARKET,
                started_at=datetime.now(),
            )
        self._notify()
        logger.info("Matching started")

    def set_polymarket_total(self, total: int) -> None:
        """Set total Polymarket markets to load."""
        with self._lock:
            self._progress.polymarket_total = total
        self._notify()

    def increment_polymarket_loaded(self, count: int = 1) -> None:
        """Increment loaded Polymarket markets."""
        with self._lock:
            self._progress.polymarket_loaded += count
            if self._progress.polymarket_loaded >= self._progress.polymarket_total:
                self._progress.status = MatchingStatus.LOADING_KALSHI
        self._notify()

    def set_kalshi_total(self, total: int) -> None:
        """Set total Kalshi markets to load."""
        with self._lock:
            self._progress.kalshi_total = total
        self._notify()

    def increment_kalshi_loaded(self, count: int = 1) -> None:
        """Increment loaded Kalshi markets."""
        with self._lock:
            self._progress.kalshi_loaded += count
            if self._progress.kalshi_loaded >= self._progress.kalshi_total:
                self._progress.status = MatchingStatus.MATCHING
        self._notify()

    def set_pairs_to_match(self, total: int) -> None:
        """Set total pairs to match."""
        with self._lock:
            self._progress.pairs_to_match = total
            self._progress.status = MatchingStatus.MATCHING
        self._notify()

    def increment_pairs_matched(
        self,
        count: int = 1,
        successful: int = 0,
    ) -> None:
        """
        Increment matched pairs.

        Args:
            count: Total pairs processed
            successful: Number of successful matches found
        """
        with self._lock:
            self._progress.pairs_matched += count
            self._progress.matched_pairs += successful
        self._notify()

    def complete(self) -> None:
        """Mark matching as complete."""
        with self._lock:
            self._progress.status = MatchingStatus.COMPLETE
            self._progress.completed_at = datetime.now()
        self._notify()
        logger.info(
            f"Matching complete: {self._progress.matched_pairs} pairs matched "
            f"in {self._progress.duration_seconds:.2f}s"
        )

    def error(self, message: str) -> None:
        """Mark matching as failed."""
        with self._lock:
            self._progress.status = MatchingStatus.ERROR
            self._progress.error_message = message
            self._progress.completed_at = datetime.now()
        self._notify()
        logger.error(f"Matching failed: {message}")

    def reset(self) -> None:
        """Reset progress tracker."""
        with self._lock:
            self._progress = MatchingProgress()
        self._notify()


class AsyncMatchingProgressTracker(MatchingProgressTracker):
    """
    Async version of progress tracker with queue-based updates.

    Useful for non-blocking progress updates in async code.
    """

    def __init__(self):
        super().__init__()
        self._queue: asyncio.Queue = asyncio.Queue()
        self._running = False
        self._task: Optional[asyncio.Task] = None

    async def start_updates(self) -> None:
        """Start processing updates."""
        self._running = True
        self._task = asyncio.create_task(self._process_updates())

    async def stop_updates(self) -> None:
        """Stop processing updates."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _process_updates(self) -> None:
        """Process update queue."""
        while self._running:
            try:
                update = await asyncio.wait_for(
                    self._queue.get(),
                    timeout=1.0
                )
                self._apply_update(update)
                self._notify()
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

    def _apply_update(self, update: Dict[str, Any]) -> None:
        """Apply update to progress."""
        action = update.get("action")

        if action == "set_polymarket_total":
            self._progress.polymarket_total = update["value"]
        elif action == "increment_polymarket":
            self._progress.polymarket_loaded += update.get("value", 1)
        elif action == "set_kalshi_total":
            self._progress.kalshi_total = update["value"]
        elif action == "increment_kalshi":
            self._progress.kalshi_loaded += update.get("value", 1)
        elif action == "set_pairs_total":
            self._progress.pairs_to_match = update["value"]
        elif action == "increment_pairs":
            self._progress.pairs_matched += update.get("value", 1)
            self._progress.matched_pairs += update.get("successful", 0)
        elif action == "set_status":
            self._progress.status = MatchingStatus(update["value"])

    async def queue_update(self, update: Dict[str, Any]) -> None:
        """Queue an update for processing."""
        await self._queue.put(update)


# =============================================================================
# Progress Display Utilities
# =============================================================================

def format_progress_bar(
    progress: float,
    width: int = 40,
    fill_char: str = "=",
    empty_char: str = " ",
) -> str:
    """
    Create ASCII progress bar.

    Args:
        progress: Progress percentage (0-100)
        width: Bar width in characters
        fill_char: Character for filled portion
        empty_char: Character for empty portion

    Returns:
        Formatted progress bar string
    """
    filled = int(width * progress / 100)
    bar = fill_char * filled + empty_char * (width - filled)
    return f"[{bar}] {progress:.1f}%"


def print_matching_progress(progress: MatchingProgress) -> None:
    """Print matching progress to console."""
    status = progress.status.value.upper()
    pct = progress.progress_pct
    bar = format_progress_bar(pct)

    print(f"\rMatching: {status} {bar}", end="", flush=True)

    if progress.status == MatchingStatus.COMPLETE:
        print(f"\nMatched {progress.matched_pairs} pairs in {progress.duration_seconds:.2f}s")
    elif progress.status == MatchingStatus.ERROR:
        print(f"\nError: {progress.error_message}")


# =============================================================================
# Singleton Instance
# =============================================================================

_progress_tracker: Optional[MatchingProgressTracker] = None


def get_matching_progress_tracker() -> MatchingProgressTracker:
    """Get or create the global progress tracker."""
    global _progress_tracker
    if _progress_tracker is None:
        _progress_tracker = MatchingProgressTracker()
    return _progress_tracker
