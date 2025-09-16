"""Data quality monitoring for staleness and outlier detection.

This module provides real-time data quality checks to ensure
trading decisions are based on fresh, sane data.
"""
from __future__ import annotations
import time
import logging
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

import numpy as np

log = logging.getLogger("wsb.data_quality")


class DataQualityMonitor:
    """Monitor data quality for staleness and outliers.
    
    Provides hard stops for:
    - Data staleness
    - Outlier price jumps
    - Missing bars
    - Corrupted data
    """

    def __init__(self, max_staleness_sec: int = 60, max_return_z: float = 3.0):
        """Initialize data quality monitor.
        
        Args:
            max_staleness_sec: Maximum allowed data staleness in seconds
            max_return_z: Maximum Z-score for price returns (outlier threshold)
        """
        self.max_staleness_sec = max_staleness_sec
        self.max_return_z = max_return_z
        self.last_tick_ts = time.time()
        self.last_prices: dict[str, float] = {}
        self.price_history: dict[str, list[float]] = {}

    def mark_tick(self) -> None:
        """Mark data tick as fresh (call on every data update)."""
        self.last_tick_ts = time.time()

    def assert_fresh(self) -> None:
        """Assert data is fresh.
        
        Raises:
            RuntimeError: If data is stale
        """
        staleness = time.time() - self.last_tick_ts
        if staleness > self.max_staleness_sec:
            raise RuntimeError(f"DATA_STALE: {staleness:.1f}s > {self.max_staleness_sec}s")

    def assert_sane_returns(self, bars: "pd.DataFrame", symbol: str) -> None:
        """Assert price returns are sane (no outliers).
        
        Args:
            bars: Price bars DataFrame with 'Close' column
            symbol: Symbol for logging
            
        Raises:
            RuntimeError: If outliers detected
        """
        if bars.empty:
            log.warning(f"No data for {symbol} - skipping outlier check")
            return
            
        if "Close" not in bars.columns:
            log.warning(f"No Close column for {symbol} - skipping outlier check")
            return
            
        r = bars["Close"].pct_change().dropna()
        if r.empty:
            return
            
        # Calculate Z-scores
        mean_ret = r.mean()
        std_ret = r.std(ddof=1)
        
        if std_ret == 0 or np.isnan(std_ret):
            log.warning(f"Zero or NaN std for {symbol} - skipping outlier check")
            return
            
        z_scores = (r - mean_ret) / std_ret
        max_z = np.nanmax(np.abs(z_scores))
        
        if max_z > self.max_return_z:
            outlier_idx = z_scores.abs().idxmax()
            outlier_ret = r.loc[outlier_idx]
            raise RuntimeError(
                f"DATA_OUTLIER: {symbol} return {outlier_ret:.4f} "
                f"(Z={max_z:.2f}) exceeds threshold {self.max_return_z}"
            )

    def assert_complete_bars(self, bars: "pd.DataFrame", symbol: str) -> None:
        """Assert bars are complete (no missing data).
        
        Args:
            bars: Price bars DataFrame
            symbol: Symbol for logging
            
        Raises:
            RuntimeError: If incomplete bars detected
        """
        if bars.empty:
            raise RuntimeError(f"DATA_EMPTY: No bars for {symbol}")
            
        # Check for required columns
        required_cols = ["Open", "High", "Low", "Close", "Volume"]
        missing_cols = [col for col in required_cols if col not in bars.columns]
        if missing_cols:
            raise RuntimeError(f"DATA_INCOMPLETE: Missing columns {missing_cols} for {symbol}")
            
        # Check for NaN values in critical columns
        critical_cols = ["Close", "Volume"]
        for col in critical_cols:
            if bars[col].isna().any():
                nan_count = bars[col].isna().sum()
                raise RuntimeError(f"DATA_CORRUPTED: {nan_count} NaN values in {col} for {symbol}")

    def assert_price_continuity(self, bars: "pd.DataFrame", symbol: str) -> None:
        """Assert price continuity (no impossible values).
        
        Args:
            bars: Price bars DataFrame
            symbol: Symbol for logging
            
        Raises:
            RuntimeError: If price continuity issues detected
        """
        if bars.empty:
            return
            
        # Check for negative prices
        if (bars["Close"] <= 0).any():
            raise RuntimeError(f"DATA_CORRUPTED: Non-positive prices for {symbol}")
            
        # Check OHLC relationships
        invalid_ohlc = (
            (bars["High"] < bars["Low"]) |
            (bars["High"] < bars["Open"]) |
            (bars["High"] < bars["Close"]) |
            (bars["Low"] > bars["Open"]) |
            (bars["Low"] > bars["Close"])
        )
        
        if invalid_ohlc.any():
            invalid_count = invalid_ohlc.sum()
            raise RuntimeError(f"DATA_CORRUPTED: {invalid_count} invalid OHLC relationships for {symbol}")

    def check_all(self, bars: "pd.DataFrame", symbol: str) -> None:
        """Run all data quality checks.
        
        Args:
            bars: Price bars DataFrame
            symbol: Symbol for logging
            
        Raises:
            RuntimeError: If any quality check fails
        """
        self.assert_fresh()
        self.assert_complete_bars(bars, symbol)
        self.assert_price_continuity(bars, symbol)
        self.assert_sane_returns(bars, symbol)

    def get_staleness_seconds(self) -> float:
        """Get current data staleness in seconds.
        
        Returns:
            Staleness in seconds
        """
        return time.time() - self.last_tick_ts

    def status(self) -> dict:
        """Get data quality status.
        
        Returns:
            Dictionary with quality metrics
        """
        return {
            "staleness_sec": self.get_staleness_seconds(),
            "max_staleness_sec": self.max_staleness_sec,
            "max_return_z": self.max_return_z,
            "is_fresh": self.get_staleness_seconds() <= self.max_staleness_sec,
        }

    def validate_price(self, symbol: str, price: float) -> dict:
        """Validate price data for quality issues.
        
        Args:
            symbol: Stock symbol
            price: Current price to validate
            
        Returns:
            dict: Validation results with quality metrics
            
        Raises:
            RuntimeError: If price is an outlier or data is stale
        """
        if price is None:
            raise ValueError("Price cannot be None")
        
        self.mark_tick()
        
        # Check for staleness
        staleness_sec = self.get_staleness_seconds()
        is_fresh = staleness_sec <= self.max_staleness_sec
        
        if not is_fresh:
            raise RuntimeError("STALE_DATA")
        
        # Initialize price history for symbol if needed
        if symbol not in self.price_history:
            self.price_history[symbol] = []
        
        # Check for outliers if we have previous price
        is_outlier = False
        if symbol in self.last_prices:
            prev_price = self.last_prices[symbol]
            if prev_price > 0:
                return_pct = (price - prev_price) / prev_price
                # Check if this is an outlier based on max_return_z
                if abs(return_pct) > self.max_return_z / 10:  # Convert to percentage
                    is_outlier = True
                    raise RuntimeError("PRICE_OUTLIER")
        
        # Update last price and history
        self.last_prices[symbol] = price
        self.price_history[symbol].append(price)
        
        # Limit history size for memory management (keep last 150 prices)
        if len(self.price_history[symbol]) > 150:
            self.price_history[symbol] = self.price_history[symbol][-150:]
        
        return {
            "symbol": symbol,
            "price": price,
            "staleness_sec": staleness_sec,
            "is_fresh": is_fresh,
            "is_outlier": is_outlier,
            "validation_passed": is_fresh and not is_outlier
        }

    def get_quality_report(self) -> dict:
        """Get comprehensive quality report.
        
        Returns:
            dict: Complete quality report with all metrics
        """
        staleness_sec = self.get_staleness_seconds()
        is_fresh = staleness_sec <= self.max_staleness_sec
        
        # Calculate additional metrics
        symbols_tracked = len(self.last_prices)
        total_price_points = sum(len(history) for history in self.price_history.values())
        
        # Calculate average prices and ranges
        average_prices = {}
        price_ranges = {}
        for symbol, history in self.price_history.items():
            if history:
                average_prices[symbol] = sum(history) / len(history)
                price_ranges[symbol] = {"min": min(history), "max": max(history)}
        
        return {
            "overall_status": "healthy" if is_fresh else "stale",
            "staleness_seconds": staleness_sec,
            "max_staleness_seconds": self.max_staleness_sec,
            "is_fresh": is_fresh,
            "max_return_z_threshold": self.max_return_z,
            "tracked_symbols": list(self.last_prices.keys()),
            "symbol_count": symbols_tracked,
            "symbols_tracked": symbols_tracked,
            "total_price_points": total_price_points,
            "freshness_status": "fresh" if is_fresh else "stale",
            "outliers_detected": 0,  # Would need to track this separately
            "average_prices": average_prices,
            "price_ranges": price_ranges,
            "update_frequencies": {},  # Would need to track this separately
            "last_tick_timestamp": self.last_tick_ts,
            "current_time": time.time()
        }


class OutlierDetector:
    """Detect outliers in price data using statistical methods."""
    
    def __init__(self, window_size: int = 20, z_threshold: float = 3.0):
        """Initialize outlier detector.
        
        Args:
            window_size: Number of recent prices to consider
            z_threshold: Z-score threshold for outlier detection
        """
        self.window_size = window_size
        self.z_threshold = z_threshold
        self.price_history: dict[str, list[float]] = {}
    
    def add_price(self, symbol: str, price: float) -> None:
        """Add price to history for outlier detection."""
        if symbol not in self.price_history:
            self.price_history[symbol] = []
        
        self.price_history[symbol].append(price)
        
        # Keep only recent prices
        if len(self.price_history[symbol]) > self.window_size:
            self.price_history[symbol] = self.price_history[symbol][-self.window_size:]
    
    def is_outlier(self, symbol: str, price: float) -> bool:
        """Check if price is an outlier.
        
        Args:
            symbol: Stock symbol
            price: Price to check
            
        Returns:
            bool: True if price is an outlier
        """
        try:
            if symbol not in self.price_history or len(self.price_history[symbol]) < 5:
                return False
            
            prices = self.price_history[symbol]
            mean_price = np.mean(prices)
            std_price = np.std(prices)
            
            if std_price == 0:
                return False
            
            z_score = abs((price - mean_price) / std_price)
            return z_score > self.z_threshold
        except (ValueError, TypeError, IndexError) as e:
            print(f"Data access error in is_outlier for {symbol}: {e}")
            return False
        except Exception as e:
            print(f"Unexpected error in is_outlier for {symbol}: {e}")
            return False
    
    def get_statistics(self, symbol: str) -> dict:
        """Get statistics for a symbol's price history.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            dict: Statistics including mean, std, sample size
        """
        if symbol not in self.price_history or len(self.price_history[symbol]) < 2:
            return {"mean_return": 0, "std_return": 0, "sample_size": 0}
        
        prices = self.price_history[symbol]
        returns = []
        for i in range(1, len(prices)):
            returns.append((prices[i] - prices[i-1]) / prices[i-1])
        
        if not returns:
            return {"mean_return": 0, "std_return": 0, "sample_size": 0}
        
        return {
            "mean_return": np.mean(returns),
            "std_return": np.std(returns),
            "sample_size": len(returns)
        }


class QualityCheckResult:
    """Result of a data quality check."""
    
    def __init__(self, symbol: str, price: float, timestamp: float, 
                 is_valid: bool, issues: list = None):
        """Initialize quality check result.
        
        Args:
            symbol: Stock symbol
            price: Price that was checked
            timestamp: When the check was performed
            is_valid: Whether the check passed
            issues: List of issues found
        """
        self.symbol = symbol
        self.price = price
        self.timestamp = timestamp
        self.is_valid = is_valid
        self.issues = issues or []
        self.passed = is_valid  # For backward compatibility
    
    def to_dict(self) -> dict:
        """Convert result to dictionary."""
        return {
            "symbol": self.symbol,
            "price": self.price,
            "timestamp": self.timestamp,
            "is_valid": self.is_valid,
            "issues": self.issues,
            "passed": self.passed
        }
    
    def __str__(self) -> str:
        """String representation of quality check result."""
        status = "VALID" if self.is_valid else "INVALID"
        return f"QualityCheckResult({self.symbol}, {self.price}, {status}, {len(self.issues)} issues)"
    
    def __eq__(self, other) -> bool:
        """Compare two quality check results."""
        if not isinstance(other, QualityCheckResult):
            return False
        return (self.symbol == other.symbol and 
                self.price == other.price and
                self.is_valid == other.is_valid)
