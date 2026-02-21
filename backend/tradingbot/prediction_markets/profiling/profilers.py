"""
Performance Profiling Scripts.

Synthesized from:
- kalshi-polymarket-arbitrage-bot: live_cprofile_profiler.py, live_yappi_profiler.py
- CPU and wall-clock time profiling

Tools for identifying performance bottlenecks.
"""

import asyncio
import cProfile
import functools
import io
import logging
import pstats
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union
import threading

logger = logging.getLogger(__name__)

# Try to import yappi (optional)
try:
    import yappi
    YAPPI_AVAILABLE = True
except ImportError:
    YAPPI_AVAILABLE = False
    logger.debug("yappi not installed. Wall-clock profiling disabled.")


F = TypeVar("F", bound=Callable[..., Any])


@dataclass
class ProfileResult:
    """Results from a profiling session."""
    duration_seconds: float
    num_calls: int = 0
    total_time: float = 0.0
    top_functions: List[Dict[str, Any]] = field(default_factory=list)
    output_file: Optional[str] = None
    raw_stats: Any = None

    def __str__(self) -> str:
        """Format results as string."""
        lines = [
            "Profile Results:",
            f"  Duration: {self.duration_seconds:.2f}s",
            f"  Total Calls: {self.num_calls}",
            f"  Total Time: {self.total_time:.4f}s",
            "\nTop Functions:",
        ]
        for i, func in enumerate(self.top_functions[:20], 1):
            lines.append(
                f"  {i:2d}. {func['name'][:50]:<50} "
                f"calls={func['calls']:>8} "
                f"time={func['cumtime']:.4f}s"
            )
        return "\n".join(lines)


class CProfiler:
    """
    cProfile-based CPU profiler.

    From kalshi-polymarket-arbitrage-bot: live_cprofile_profiler.py
    """

    def __init__(
        self,
        output_dir: str = "profiles",
        sort_by: str = "cumtime",
    ):
        """
        Initialize cProfile profiler.

        Args:
            output_dir: Directory for profile output files
            sort_by: Sort key for results (cumtime, tottime, calls)
        """
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._sort_by = sort_by
        self._profiler: Optional[cProfile.Profile] = None
        self._start_time: float = 0

    def start(self) -> None:
        """Start profiling."""
        self._profiler = cProfile.Profile()
        self._profiler.enable()
        self._start_time = time.perf_counter()
        logger.info("cProfile profiling started")

    def stop(self) -> ProfileResult:
        """Stop profiling and return results."""
        if self._profiler is None:
            raise RuntimeError("Profiler not started")

        self._profiler.disable()
        duration = time.perf_counter() - self._start_time

        # Get stats
        stream = io.StringIO()
        stats = pstats.Stats(self._profiler, stream=stream)
        stats.sort_stats(self._sort_by)

        # Extract top functions
        top_functions = []
        for key, value in stats.stats.items():
            filename, lineno, funcname = key
            cc, nc, tt, ct, callers = value
            top_functions.append({
                "name": f"{funcname} ({filename}:{lineno})",
                "calls": nc,
                "tottime": tt,
                "cumtime": ct,
            })

        top_functions.sort(key=lambda x: x["cumtime"], reverse=True)

        # Save to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self._output_dir / f"cprofile_{timestamp}.txt"

        stats.print_stats(50)
        with open(output_file, "w") as f:
            f.write(stream.getvalue())

        result = ProfileResult(
            duration_seconds=duration,
            num_calls=sum(f["calls"] for f in top_functions),
            total_time=sum(f["cumtime"] for f in top_functions[:10]),
            top_functions=top_functions[:50],
            output_file=str(output_file),
            raw_stats=stats,
        )

        logger.info(f"cProfile results saved to {output_file}")
        return result

    def __enter__(self) -> "CProfiler":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()


class YappiProfiler:
    """
    yappi-based wall-clock time profiler.

    From kalshi-polymarket-arbitrage-bot: live_yappi_profiler.py
    """

    def __init__(
        self,
        output_dir: str = "profiles",
        clock_type: str = "wall",  # "wall" or "cpu"
    ):
        """
        Initialize yappi profiler.

        Args:
            output_dir: Directory for profile output files
            clock_type: Clock type ("wall" for wall-clock, "cpu" for CPU time)
        """
        if not YAPPI_AVAILABLE:
            raise ImportError(
                "yappi is required for wall-clock profiling. "
                "Install with: pip install yappi"
            )

        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._clock_type = clock_type
        self._start_time: float = 0

    def start(self) -> None:
        """Start profiling."""
        yappi.set_clock_type(self._clock_type)
        yappi.start()
        self._start_time = time.perf_counter()
        logger.info(f"yappi profiling started (clock={self._clock_type})")

    def stop(self) -> ProfileResult:
        """Stop profiling and return results."""
        yappi.stop()
        duration = time.perf_counter() - self._start_time

        # Get function stats
        func_stats = yappi.get_func_stats()

        # Extract top functions
        top_functions = []
        for stat in func_stats:
            top_functions.append({
                "name": stat.name,
                "module": stat.module,
                "calls": stat.ncall,
                "tottime": stat.tsub,
                "cumtime": stat.ttot,
                "avgtime": stat.tavg,
            })

        top_functions.sort(key=lambda x: x["cumtime"], reverse=True)

        # Save to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self._output_dir / f"yappi_{timestamp}.txt"

        with open(output_file, "w") as f:
            f.write(f"yappi Profile Results ({self._clock_type} time)\n")
            f.write(f"Duration: {duration:.2f}s\n")
            f.write("=" * 100 + "\n\n")

            f.write(f"{'Function':<60} {'Calls':>10} {'Tot Time':>12} {'Cum Time':>12}\n")
            f.write("-" * 100 + "\n")

            for func in top_functions[:100]:
                name = func["name"][:58]
                f.write(
                    f"{name:<60} {func['calls']:>10} "
                    f"{func['tottime']:>12.6f} {func['cumtime']:>12.6f}\n"
                )

        # Clear stats for next run
        yappi.clear_stats()

        result = ProfileResult(
            duration_seconds=duration,
            num_calls=sum(f["calls"] for f in top_functions),
            total_time=sum(f["cumtime"] for f in top_functions[:10]),
            top_functions=top_functions[:50],
            output_file=str(output_file),
        )

        logger.info(f"yappi results saved to {output_file}")
        return result

    def __enter__(self) -> "YappiProfiler":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()


async def run_cprofile_session(
    coroutine: asyncio.Future,
    duration_minutes: float = 5.0,
    output_dir: str = "profiles",
) -> ProfileResult:
    """
    Run cProfile session for a coroutine.

    Args:
        coroutine: Async function to profile
        duration_minutes: Profile duration in minutes
        output_dir: Output directory for results

    Returns:
        ProfileResult with profiling data
    """
    profiler = CProfiler(output_dir=output_dir)

    async def run_with_timeout():
        try:
            await asyncio.wait_for(
                coroutine,
                timeout=duration_minutes * 60,
            )
        except asyncio.TimeoutError:
            logger.info(f"Profile session completed after {duration_minutes} minutes")

    profiler.start()
    try:
        await run_with_timeout()
    finally:
        return profiler.stop()


async def run_yappi_session(
    coroutine: asyncio.Future,
    duration_minutes: float = 5.0,
    output_dir: str = "profiles",
    clock_type: str = "wall",
) -> ProfileResult:
    """
    Run yappi session for a coroutine.

    Args:
        coroutine: Async function to profile
        duration_minutes: Profile duration in minutes
        output_dir: Output directory for results
        clock_type: "wall" or "cpu"

    Returns:
        ProfileResult with profiling data
    """
    if not YAPPI_AVAILABLE:
        raise ImportError("yappi required. Install: pip install yappi")

    profiler = YappiProfiler(output_dir=output_dir, clock_type=clock_type)

    async def run_with_timeout():
        try:
            await asyncio.wait_for(
                coroutine,
                timeout=duration_minutes * 60,
            )
        except asyncio.TimeoutError:
            logger.info(f"Profile session completed after {duration_minutes} minutes")

    profiler.start()
    try:
        await run_with_timeout()
    finally:
        return profiler.stop()


def profile_function(
    output_dir: str = "profiles",
    use_yappi: bool = False,
) -> Callable[[F], F]:
    """
    Decorator to profile a function.

    Args:
        output_dir: Directory for output files
        use_yappi: Use yappi instead of cProfile

    Usage:
        @profile_function()
        def my_function():
            ...

        @profile_function(use_yappi=True)
        async def my_async_function():
            ...
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            profiler = CProfiler(output_dir=output_dir)
            profiler.start()
            try:
                return func(*args, **kwargs)
            finally:
                result = profiler.stop()
                logger.info(f"Profile for {func.__name__}:\n{result}")

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            if use_yappi and YAPPI_AVAILABLE:
                profiler = YappiProfiler(output_dir=output_dir)
            else:
                profiler = CProfiler(output_dir=output_dir)

            profiler.start()
            try:
                return await func(*args, **kwargs)
            finally:
                result = profiler.stop()
                logger.info(f"Profile for {func.__name__}:\n{result}")

        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        return sync_wrapper  # type: ignore

    return decorator


class BenchmarkRunner:
    """
    Run multiple profiling trials.

    From kalshi-polymarket-arbitrage-bot: Multiple runs for statistical significance.
    """

    def __init__(
        self,
        output_dir: str = "profiles",
        num_trials: int = 3,
    ):
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._num_trials = num_trials
        self._results: List[ProfileResult] = []

    async def run_trials(
        self,
        coroutine_factory: Callable[[], asyncio.Future],
        duration_minutes: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Run multiple profiling trials.

        Args:
            coroutine_factory: Factory function that creates the coroutine
            duration_minutes: Duration per trial

        Returns:
            Aggregated statistics
        """
        self._results.clear()

        for i in range(self._num_trials):
            logger.info(f"Starting trial {i + 1}/{self._num_trials}")
            result = await run_cprofile_session(
                coroutine_factory(),
                duration_minutes=duration_minutes,
                output_dir=str(self._output_dir),
            )
            self._results.append(result)

        return self._aggregate_results()

    def _aggregate_results(self) -> Dict[str, Any]:
        """Aggregate results from all trials."""
        if not self._results:
            return {}

        durations = [r.duration_seconds for r in self._results]
        total_calls = [r.num_calls for r in self._results]

        # Aggregate function timings
        func_times: Dict[str, List[float]] = {}
        for result in self._results:
            for func in result.top_functions:
                name = func["name"]
                if name not in func_times:
                    func_times[name] = []
                func_times[name].append(func["cumtime"])

        # Calculate averages
        avg_func_times = {
            name: sum(times) / len(times)
            for name, times in func_times.items()
        }

        top_by_avg = sorted(
            avg_func_times.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:20]

        return {
            "num_trials": self._num_trials,
            "avg_duration": sum(durations) / len(durations),
            "min_duration": min(durations),
            "max_duration": max(durations),
            "avg_calls": sum(total_calls) / len(total_calls),
            "top_functions_by_avg_time": [
                {"name": name, "avg_cumtime": time}
                for name, time in top_by_avg
            ],
        }
