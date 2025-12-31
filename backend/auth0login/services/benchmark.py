"""
Benchmark Comparison Service

Provides SPY (and other benchmark) comparison functionality for portfolio performance.
Includes caching to reduce API calls since benchmark data is the same for all users.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional
from functools import lru_cache
import time

logger = logging.getLogger(__name__)

# Try to import yfinance for market data
try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False
    logger.warning("yfinance not available - benchmark service will use mock data")

# Try to import numpy for calculations
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    logger.warning("numpy not available - some calculations may be limited")


@dataclass
class BenchmarkReturn:
    """Return data for a benchmark."""
    ticker: str
    start_price: float
    end_price: float
    period_return_pct: float
    daily_return_pct: float
    start_date: str
    end_date: str


@dataclass
class PortfolioComparison:
    """Comparison between portfolio and benchmark."""
    portfolio_return_pct: float
    benchmark_return_pct: float
    daily_excess: float
    period_excess: float
    hypothetical_benchmark_value: float
    your_value: float
    alpha_generated: float  # Dollar value of outperformance
    information_ratio: Optional[float]
    tracking_error: Optional[float]


@dataclass
class BenchmarkSeries:
    """Time series data for benchmark comparison."""
    dates: list[str]
    portfolio_values: list[float]
    benchmark_values: list[float]
    portfolio_normalized: list[float]  # Normalized to 100 at start
    benchmark_normalized: list[float]  # Normalized to 100 at start


# Cache for benchmark data (TTL of 5 minutes)
_benchmark_cache = {}
_cache_timestamps = {}
CACHE_TTL_SECONDS = 300  # 5 minutes


class BenchmarkService:
    """Service for benchmark comparison calculations."""

    DEFAULT_BENCHMARK = 'SPY'
    SUPPORTED_BENCHMARKS = ['SPY', 'QQQ', 'IWM', 'DIA', 'VTI']

    def __init__(self):
        pass

    def _get_cached(self, key: str):
        """Get cached value if still valid."""
        if key in _benchmark_cache:
            timestamp = _cache_timestamps.get(key, 0)
            if time.time() - timestamp < CACHE_TTL_SECONDS:
                return _benchmark_cache[key]
        return None

    def _set_cached(self, key: str, value):
        """Set cached value with timestamp."""
        _benchmark_cache[key] = value
        _cache_timestamps[key] = time.time()

    def get_benchmark_return(
        self,
        start_date: datetime,
        end_date: datetime,
        benchmark: str = 'SPY',
    ) -> BenchmarkReturn:
        """
        Get benchmark return for a period.

        Args:
            start_date: Period start
            end_date: Period end
            benchmark: Ticker symbol (default SPY)

        Returns:
            BenchmarkReturn with period and daily returns
        """
        cache_key = f"{benchmark}_{start_date.date()}_{end_date.date()}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        if HAS_YFINANCE:
            try:
                ticker = yf.Ticker(benchmark)
                # Get data with a buffer for daily return calculation
                hist = ticker.history(
                    start=start_date - timedelta(days=5),
                    end=end_date + timedelta(days=1),
                )

                if len(hist) < 2:
                    return self._mock_benchmark_return(benchmark, start_date, end_date)

                # Get start and end prices
                start_idx = hist.index.searchsorted(start_date)
                if start_idx >= len(hist):
                    start_idx = 0
                start_price = float(hist['Close'].iloc[start_idx])
                end_price = float(hist['Close'].iloc[-1])

                # Calculate period return
                period_return_pct = ((end_price - start_price) / start_price) * 100

                # Calculate daily return (last day)
                if len(hist) >= 2:
                    prev_close = float(hist['Close'].iloc[-2])
                    daily_return_pct = ((end_price - prev_close) / prev_close) * 100
                else:
                    daily_return_pct = 0.0

                result = BenchmarkReturn(
                    ticker=benchmark,
                    start_price=start_price,
                    end_price=end_price,
                    period_return_pct=round(period_return_pct, 2),
                    daily_return_pct=round(daily_return_pct, 2),
                    start_date=start_date.strftime('%Y-%m-%d'),
                    end_date=end_date.strftime('%Y-%m-%d'),
                )

                self._set_cached(cache_key, result)
                return result

            except Exception as e:
                logger.error(f"Error fetching benchmark data: {e}")
                return self._mock_benchmark_return(benchmark, start_date, end_date)
        else:
            return self._mock_benchmark_return(benchmark, start_date, end_date)

    def _mock_benchmark_return(
        self,
        benchmark: str,
        start_date: datetime,
        end_date: datetime,
    ) -> BenchmarkReturn:
        """Return mock benchmark data when real data unavailable."""
        days = (end_date - start_date).days
        # Approximate average SPY returns
        avg_daily_return = 0.04  # ~10% annually
        period_return = avg_daily_return * days

        return BenchmarkReturn(
            ticker=benchmark,
            start_price=450.00,
            end_price=450.00 * (1 + period_return / 100),
            period_return_pct=round(period_return, 2),
            daily_return_pct=round(avg_daily_return, 2),
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d'),
        )

    def get_benchmark_series(
        self,
        start_date: datetime,
        end_date: datetime,
        benchmark: str = 'SPY',
    ) -> list[dict]:
        """
        Get benchmark price series for charting.

        Returns list of {date, close, return_pct} dicts.
        """
        cache_key = f"series_{benchmark}_{start_date.date()}_{end_date.date()}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        if HAS_YFINANCE:
            try:
                ticker = yf.Ticker(benchmark)
                hist = ticker.history(start=start_date, end=end_date + timedelta(days=1))

                if len(hist) == 0:
                    return self._mock_benchmark_series(start_date, end_date)

                start_price = float(hist['Close'].iloc[0])
                series = []
                for date, row in hist.iterrows():
                    close = float(row['Close'])
                    return_pct = ((close - start_price) / start_price) * 100
                    series.append({
                        'date': date.strftime('%Y-%m-%d'),
                        'close': round(close, 2),
                        'return_pct': round(return_pct, 2),
                        'normalized': round(100 * (close / start_price), 2),
                    })

                self._set_cached(cache_key, series)
                return series

            except Exception as e:
                logger.error(f"Error fetching benchmark series: {e}")
                return self._mock_benchmark_series(start_date, end_date)
        else:
            return self._mock_benchmark_series(start_date, end_date)

    def _mock_benchmark_series(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> list[dict]:
        """Return mock series data."""
        series = []
        current = start_date
        price = 450.0
        while current <= end_date:
            # Skip weekends
            if current.weekday() < 5:
                daily_change = (0.5 - (hash(str(current)) % 100) / 100) * 2  # Random -1% to +1%
                price *= (1 + daily_change / 100)
                return_pct = ((price - 450.0) / 450.0) * 100
                series.append({
                    'date': current.strftime('%Y-%m-%d'),
                    'close': round(price, 2),
                    'return_pct': round(return_pct, 2),
                    'normalized': round(100 * (price / 450.0), 2),
                })
            current += timedelta(days=1)
        return series

    def compare_portfolio_to_benchmark(
        self,
        portfolio_start_value: float,
        portfolio_end_value: float,
        portfolio_daily_pnl: float,
        start_date: datetime,
        end_date: datetime,
        benchmark: str = 'SPY',
    ) -> PortfolioComparison:
        """
        Compare portfolio performance to benchmark.

        Args:
            portfolio_start_value: Portfolio value at start of period
            portfolio_end_value: Portfolio value at end of period
            portfolio_daily_pnl: Today's P&L in dollars
            start_date: Period start
            end_date: Period end
            benchmark: Benchmark ticker

        Returns:
            PortfolioComparison with all comparison metrics
        """
        # Get benchmark returns
        bench_return = self.get_benchmark_return(start_date, end_date, benchmark)

        # Calculate portfolio returns
        portfolio_return_pct = (
            (portfolio_end_value - portfolio_start_value) / portfolio_start_value * 100
            if portfolio_start_value > 0 else 0
        )

        portfolio_daily_pct = (
            portfolio_daily_pnl / (portfolio_end_value - portfolio_daily_pnl) * 100
            if (portfolio_end_value - portfolio_daily_pnl) > 0 else 0
        )

        # Calculate excess returns
        daily_excess = portfolio_daily_pct - bench_return.daily_return_pct
        period_excess = portfolio_return_pct - bench_return.period_return_pct

        # Calculate hypothetical benchmark value
        # "If you had invested in SPY instead"
        hypothetical_benchmark_value = portfolio_start_value * (1 + bench_return.period_return_pct / 100)

        # Alpha generated in dollars
        alpha_generated = portfolio_end_value - hypothetical_benchmark_value

        return PortfolioComparison(
            portfolio_return_pct=round(portfolio_return_pct, 2),
            benchmark_return_pct=bench_return.period_return_pct,
            daily_excess=round(daily_excess, 2),
            period_excess=round(period_excess, 2),
            hypothetical_benchmark_value=round(hypothetical_benchmark_value, 2),
            your_value=round(portfolio_end_value, 2),
            alpha_generated=round(alpha_generated, 2),
            information_ratio=None,  # Requires daily returns series
            tracking_error=None,  # Requires daily returns series
        )

    def calculate_tracking_metrics(
        self,
        portfolio_returns: list[float],
        benchmark_returns: list[float],
    ) -> dict:
        """
        Calculate advanced tracking metrics.

        Args:
            portfolio_returns: List of daily portfolio returns (percentages)
            benchmark_returns: List of daily benchmark returns (percentages)

        Returns:
            Dict with tracking_error, information_ratio, correlation
        """
        if not HAS_NUMPY or len(portfolio_returns) < 5:
            return {
                'tracking_error': None,
                'information_ratio': None,
                'correlation': None,
                'beta': None,
                'alpha': None,
            }

        portfolio_arr = np.array(portfolio_returns)
        benchmark_arr = np.array(benchmark_returns)

        # Calculate excess returns
        excess_returns = portfolio_arr - benchmark_arr

        # Tracking error (annualized std of excess returns)
        tracking_error = np.std(excess_returns) * np.sqrt(252)

        # Information ratio
        if tracking_error > 0:
            information_ratio = (np.mean(excess_returns) * 252) / tracking_error
        else:
            information_ratio = 0

        # Correlation
        correlation = np.corrcoef(portfolio_arr, benchmark_arr)[0, 1]

        # Beta (covariance / variance of benchmark)
        covariance = np.cov(portfolio_arr, benchmark_arr)[0, 1]
        benchmark_variance = np.var(benchmark_arr)
        beta = covariance / benchmark_variance if benchmark_variance > 0 else 1.0

        # Alpha (annualized)
        alpha = (np.mean(portfolio_arr) - beta * np.mean(benchmark_arr)) * 252

        return {
            'tracking_error': round(tracking_error, 4),
            'information_ratio': round(information_ratio, 4),
            'correlation': round(correlation, 4),
            'beta': round(beta, 4),
            'alpha': round(alpha, 4),
        }

    def get_comparison_data(
        self,
        portfolio_values: list[dict],  # List of {date, value}
        start_date: datetime,
        end_date: datetime,
        benchmark: str = 'SPY',
    ) -> dict:
        """
        Get complete comparison data for charting.

        Args:
            portfolio_values: List of {date: str, value: float} dicts
            start_date: Period start
            end_date: Period end
            benchmark: Benchmark ticker

        Returns:
            Dict with all data needed for comparison chart
        """
        # Get benchmark series
        bench_series = self.get_benchmark_series(start_date, end_date, benchmark)

        if not portfolio_values:
            return {
                'status': 'error',
                'message': 'No portfolio data available',
            }

        # Normalize portfolio to start at 100
        start_value = portfolio_values[0]['value'] if portfolio_values else 1
        portfolio_normalized = []
        for pv in portfolio_values:
            portfolio_normalized.append({
                'date': pv['date'],
                'value': pv['value'],
                'normalized': round(100 * (pv['value'] / start_value), 2),
            })

        # Calculate summary metrics
        end_value = portfolio_values[-1]['value'] if portfolio_values else start_value
        portfolio_return = ((end_value - start_value) / start_value * 100) if start_value > 0 else 0

        bench_return = bench_series[-1]['return_pct'] if bench_series else 0

        return {
            'status': 'success',
            'benchmark': benchmark,
            'period': {
                'start': start_date.strftime('%Y-%m-%d'),
                'end': end_date.strftime('%Y-%m-%d'),
            },
            'portfolio': {
                'start_value': round(start_value, 2),
                'end_value': round(end_value, 2),
                'return_pct': round(portfolio_return, 2),
                'series': portfolio_normalized,
            },
            'benchmark_data': {
                'return_pct': bench_return,
                'series': bench_series,
            },
            'comparison': {
                'excess_return': round(portfolio_return - bench_return, 2),
                'outperforming': portfolio_return > bench_return,
            },
        }


# Singleton instance
benchmark_service = BenchmarkService()
