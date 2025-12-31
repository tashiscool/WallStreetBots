"""
Opportunity Timing Analytics for Prediction Market Arbitrage.

Synthesized from:
- polymarket-arbitrage: Opportunity duration tracking
- kalshi-polymarket-arbitrage-bot: Execution statistics

Tracks opportunity lifecycle for performance analysis.
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional
from collections import deque
import statistics
import threading

logger = logging.getLogger(__name__)


@dataclass
class OpportunityTiming:
    """
    Tracks individual opportunity lifecycle.

    From polymarket-arbitrage: Complete timing data.
    """
    opportunity_id: str
    market_id: str
    opportunity_type: str  # "bundle_arb", "buy_both", etc.
    edge: Decimal

    # Timestamps
    detected_at: datetime = field(default_factory=datetime.now)
    expired_at: Optional[datetime] = None
    executed_at: Optional[datetime] = None

    # Outcome
    was_executed: bool = False
    execution_success: bool = False

    @property
    def duration_ms(self) -> float:
        """Duration from detection to expiry/execution."""
        end_time = self.executed_at or self.expired_at or datetime.now()
        delta = end_time - self.detected_at
        return delta.total_seconds() * 1000

    @property
    def is_complete(self) -> bool:
        """Whether opportunity lifecycle is complete."""
        return self.expired_at is not None or self.executed_at is not None


@dataclass
class ArbStats:
    """
    Aggregated arbitrage statistics.

    From polymarket-arbitrage: Counter-based statistics.
    """
    # Counters
    bundle_opportunities_detected: int = 0
    mm_opportunities_detected: int = 0
    signals_generated: int = 0

    # Timing
    last_opportunity_time: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "bundle_opportunities_detected": self.bundle_opportunities_detected,
            "mm_opportunities_detected": self.mm_opportunities_detected,
            "signals_generated": self.signals_generated,
            "last_opportunity_time": (
                self.last_opportunity_time.isoformat()
                if self.last_opportunity_time else None
            ),
        }


@dataclass
class ExecutionStats:
    """
    Trade execution statistics.

    From kalshi-polymarket-arbitrage-bot: Execution tracking.
    """
    orders_placed: int = 0
    orders_filled: int = 0
    orders_cancelled: int = 0
    orders_rejected: int = 0

    total_notional: Decimal = Decimal("0")

    signals_processed: int = 0
    signals_rejected: int = 0
    slippage_rejections: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "orders_placed": self.orders_placed,
            "orders_filled": self.orders_filled,
            "orders_cancelled": self.orders_cancelled,
            "orders_rejected": self.orders_rejected,
            "fill_rate": (
                self.orders_filled / self.orders_placed * 100
                if self.orders_placed > 0 else 0
            ),
            "total_notional": float(self.total_notional),
            "signals_processed": self.signals_processed,
            "signals_rejected": self.signals_rejected,
            "slippage_rejections": self.slippage_rejections,
        }


class OpportunityAnalytics:
    """
    Comprehensive opportunity analytics tracker.

    From polymarket-arbitrage: Duration bucketing and statistics.
    """

    # Duration buckets (milliseconds)
    BUCKET_UNDER_100MS = "under_100ms"
    BUCKET_UNDER_500MS = "under_500ms"
    BUCKET_UNDER_1S = "under_1s"
    BUCKET_OVER_1S = "over_1s"

    def __init__(self, max_history: int = 10000):
        """
        Initialize analytics tracker.

        Args:
            max_history: Maximum opportunities to track
        """
        self._opportunities: deque[OpportunityTiming] = deque(maxlen=max_history)
        self._completed: deque[OpportunityTiming] = deque(maxlen=max_history)
        self._arb_stats = ArbStats()
        self._execution_stats = ExecutionStats()
        self._lock = threading.RLock()

        # Duration tracking
        self._durations: List[float] = []
        self._duration_buckets: Dict[str, int] = {
            self.BUCKET_UNDER_100MS: 0,
            self.BUCKET_UNDER_500MS: 0,
            self.BUCKET_UNDER_1S: 0,
            self.BUCKET_OVER_1S: 0,
        }

    def track_opportunity(
        self,
        opportunity_id: str,
        market_id: str,
        opportunity_type: str,
        edge: Decimal,
    ) -> OpportunityTiming:
        """
        Start tracking a new opportunity.

        Returns:
            OpportunityTiming object for updates
        """
        with self._lock:
            timing = OpportunityTiming(
                opportunity_id=opportunity_id,
                market_id=market_id,
                opportunity_type=opportunity_type,
                edge=edge,
            )
            self._opportunities.append(timing)

            # Update stats
            self._arb_stats.last_opportunity_time = datetime.now()
            if "bundle" in opportunity_type.lower():
                self._arb_stats.bundle_opportunities_detected += 1
            else:
                self._arb_stats.mm_opportunities_detected += 1

            logger.debug(
                f"Tracking opportunity {opportunity_id}: {opportunity_type} "
                f"edge={edge}"
            )

            return timing

    def mark_executed(
        self,
        opportunity_id: str,
        success: bool = True,
    ) -> Optional[OpportunityTiming]:
        """Mark opportunity as executed."""
        with self._lock:
            timing = self._find_opportunity(opportunity_id)
            if timing:
                timing.executed_at = datetime.now()
                timing.was_executed = True
                timing.execution_success = success
                self._complete_opportunity(timing)

                # Update execution stats
                if success:
                    self._execution_stats.orders_filled += 1
                else:
                    self._execution_stats.orders_rejected += 1

                return timing
            return None

    def mark_expired(
        self,
        opportunity_id: str,
    ) -> Optional[OpportunityTiming]:
        """Mark opportunity as expired (not executed)."""
        with self._lock:
            timing = self._find_opportunity(opportunity_id)
            if timing:
                timing.expired_at = datetime.now()
                timing.was_executed = False
                self._complete_opportunity(timing)
                return timing
            return None

    def record_order_placed(self, notional: Decimal = Decimal("0")) -> None:
        """Record order placement."""
        with self._lock:
            self._execution_stats.orders_placed += 1
            self._execution_stats.total_notional += notional

    def record_order_cancelled(self) -> None:
        """Record order cancellation."""
        with self._lock:
            self._execution_stats.orders_cancelled += 1

    def record_signal_generated(self) -> None:
        """Record signal generation."""
        with self._lock:
            self._arb_stats.signals_generated += 1
            self._execution_stats.signals_processed += 1

    def record_signal_rejected(self, reason: str = "") -> None:
        """Record rejected signal."""
        with self._lock:
            self._execution_stats.signals_rejected += 1
            if "slippage" in reason.lower():
                self._execution_stats.slippage_rejections += 1

    def _find_opportunity(
        self,
        opportunity_id: str,
    ) -> Optional[OpportunityTiming]:
        """Find opportunity by ID."""
        for timing in self._opportunities:
            if timing.opportunity_id == opportunity_id:
                return timing
        return None

    def _complete_opportunity(self, timing: OpportunityTiming) -> None:
        """Move opportunity to completed and update stats."""
        duration = timing.duration_ms
        self._durations.append(duration)

        # Update buckets
        if duration < 100:
            self._duration_buckets[self.BUCKET_UNDER_100MS] += 1
        elif duration < 500:
            self._duration_buckets[self.BUCKET_UNDER_500MS] += 1
        elif duration < 1000:
            self._duration_buckets[self.BUCKET_UNDER_1S] += 1
        else:
            self._duration_buckets[self.BUCKET_OVER_1S] += 1

        self._completed.append(timing)

        logger.debug(
            f"Completed opportunity {timing.opportunity_id}: "
            f"duration={duration:.2f}ms executed={timing.was_executed}"
        )

    def get_duration_stats(self) -> Dict[str, Any]:
        """
        Get duration statistics.

        Returns:
            Dictionary with min, max, avg, median, percentiles
        """
        with self._lock:
            if not self._durations:
                return {
                    "total_opportunities_tracked": 0,
                    "avg_opportunity_duration_ms": 0.0,
                    "min_opportunity_duration_ms": 0.0,
                    "max_opportunity_duration_ms": 0.0,
                    "median_duration_ms": 0.0,
                    "p90_duration_ms": 0.0,
                    "p99_duration_ms": 0.0,
                    "buckets": self._duration_buckets.copy(),
                }

            sorted_durations = sorted(self._durations)
            n = len(sorted_durations)

            return {
                "total_opportunities_tracked": n,
                "avg_opportunity_duration_ms": statistics.mean(self._durations),
                "min_opportunity_duration_ms": min(self._durations),
                "max_opportunity_duration_ms": max(self._durations),
                "median_duration_ms": statistics.median(self._durations),
                "p90_duration_ms": sorted_durations[int(n * 0.90)] if n > 0 else 0,
                "p99_duration_ms": sorted_durations[int(n * 0.99)] if n > 0 else 0,
                "buckets": self._duration_buckets.copy(),
            }

    def get_execution_rate(self) -> float:
        """Get percentage of opportunities that were executed."""
        with self._lock:
            if not self._completed:
                return 0.0
            executed = sum(1 for t in self._completed if t.was_executed)
            return executed / len(self._completed) * 100

    def get_success_rate(self) -> float:
        """Get percentage of executed opportunities that succeeded."""
        with self._lock:
            executed = [t for t in self._completed if t.was_executed]
            if not executed:
                return 0.0
            succeeded = sum(1 for t in executed if t.execution_success)
            return succeeded / len(executed) * 100

    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get all statistics."""
        with self._lock:
            duration_stats = self.get_duration_stats()

            return {
                "arb_stats": self._arb_stats.to_dict(),
                "execution_stats": self._execution_stats.to_dict(),
                "duration_stats": duration_stats,
                "execution_rate": self.get_execution_rate(),
                "success_rate": self.get_success_rate(),
                "pending_opportunities": len([
                    t for t in self._opportunities if not t.is_complete
                ]),
                "completed_opportunities": len(self._completed),
            }

    def get_recent_opportunities(
        self,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Get recent opportunities with details."""
        with self._lock:
            recent = list(self._completed)[-limit:]
            return [
                {
                    "opportunity_id": t.opportunity_id,
                    "market_id": t.market_id,
                    "type": t.opportunity_type,
                    "edge": float(t.edge),
                    "detected_at": t.detected_at.isoformat(),
                    "duration_ms": t.duration_ms,
                    "was_executed": t.was_executed,
                    "success": t.execution_success,
                }
                for t in recent
            ]

    def reset(self) -> None:
        """Reset all statistics."""
        with self._lock:
            self._opportunities.clear()
            self._completed.clear()
            self._durations.clear()
            self._duration_buckets = {
                self.BUCKET_UNDER_100MS: 0,
                self.BUCKET_UNDER_500MS: 0,
                self.BUCKET_UNDER_1S: 0,
                self.BUCKET_OVER_1S: 0,
            }
            self._arb_stats = ArbStats()
            self._execution_stats = ExecutionStats()
            logger.info("Analytics reset")


# =============================================================================
# Latency Tracker
# =============================================================================

class LatencyTracker:
    """
    Tracks operation latencies.

    Useful for identifying performance bottlenecks.
    """

    def __init__(self, max_samples: int = 1000):
        self._samples: Dict[str, deque] = {}
        self._max_samples = max_samples
        self._lock = threading.RLock()

    def record(self, operation: str, latency_ms: float) -> None:
        """Record latency sample."""
        with self._lock:
            if operation not in self._samples:
                self._samples[operation] = deque(maxlen=self._max_samples)
            self._samples[operation].append(latency_ms)

    def get_stats(self, operation: str) -> Dict[str, float]:
        """Get statistics for an operation."""
        with self._lock:
            samples = self._samples.get(operation, [])
            if not samples:
                return {
                    "count": 0,
                    "avg_ms": 0.0,
                    "min_ms": 0.0,
                    "max_ms": 0.0,
                    "p50_ms": 0.0,
                    "p95_ms": 0.0,
                    "p99_ms": 0.0,
                }

            sorted_samples = sorted(samples)
            n = len(sorted_samples)

            return {
                "count": n,
                "avg_ms": statistics.mean(samples),
                "min_ms": min(samples),
                "max_ms": max(samples),
                "p50_ms": sorted_samples[int(n * 0.50)],
                "p95_ms": sorted_samples[int(n * 0.95)] if n > 1 else sorted_samples[-1],
                "p99_ms": sorted_samples[int(n * 0.99)] if n > 1 else sorted_samples[-1],
            }

    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all operations."""
        with self._lock:
            return {
                operation: self.get_stats(operation)
                for operation in self._samples.keys()
            }


class Timer:
    """Context manager for timing operations."""

    def __init__(
        self,
        tracker: Optional[LatencyTracker] = None,
        operation: str = "unknown",
    ):
        self._tracker = tracker
        self._operation = operation
        self._start: float = 0
        self._elapsed_ms: float = 0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._elapsed_ms = (time.perf_counter() - self._start) * 1000
        if self._tracker:
            self._tracker.record(self._operation, self._elapsed_ms)

    @property
    def elapsed_ms(self) -> float:
        return self._elapsed_ms


# =============================================================================
# Singleton Instances
# =============================================================================

_opportunity_analytics: Optional[OpportunityAnalytics] = None
_latency_tracker: Optional[LatencyTracker] = None


def get_opportunity_analytics() -> OpportunityAnalytics:
    """Get or create the global analytics instance."""
    global _opportunity_analytics
    if _opportunity_analytics is None:
        _opportunity_analytics = OpportunityAnalytics()
    return _opportunity_analytics


def get_latency_tracker() -> LatencyTracker:
    """Get or create the global latency tracker."""
    global _latency_tracker
    if _latency_tracker is None:
        _latency_tracker = LatencyTracker()
    return _latency_tracker
