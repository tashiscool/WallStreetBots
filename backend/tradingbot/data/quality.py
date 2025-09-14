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

    def __init__(self, max_staleness_sec: int = 20, max_return_z: float = 6.0):
        """Initialize data quality monitor.
        
        Args:
            max_staleness_sec: Maximum allowed data staleness in seconds
            max_return_z: Maximum Z-score for price returns (outlier threshold)
        """
        self.max_staleness_sec = max_staleness_sec
        self.max_return_z = max_return_z
        self.last_tick_ts = time.time()
        self.last_prices: dict[str, float] = {}

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
