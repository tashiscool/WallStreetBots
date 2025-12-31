"""
Market Monitor Service - VIX monitoring and volatility-based circuit breakers.

Provides real-time VIX monitoring with caching to prevent excessive API calls
and integrates with the circuit breaker system for automated trading halts.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, Dict, Any
from functools import lru_cache
import time

logger = logging.getLogger(__name__)


class VIXLevel(Enum):
    """VIX volatility levels with corresponding thresholds."""
    NORMAL = "normal"       # VIX < 20
    ELEVATED = "elevated"   # VIX 20-25
    HIGH = "high"          # VIX 25-35
    EXTREME = "extreme"    # VIX 35-45
    CRITICAL = "critical"  # VIX > 45


@dataclass
class VIXThresholds:
    """VIX threshold configuration."""
    normal_max: float = 20.0
    elevated_max: float = 25.0
    high_max: float = 35.0
    extreme_max: float = 45.0

    # Position sizing adjustments
    elevated_position_reduction: float = 0.75  # 25% reduction
    high_position_reduction: float = 0.50      # 50% reduction
    extreme_position_reduction: float = 0.25   # 75% reduction
    critical_pause_trading: bool = True        # Halt new positions


@dataclass
class VIXData:
    """Container for VIX data with metadata."""
    value: float
    level: VIXLevel
    percentile: float
    timestamp: datetime
    change_1d: Optional[float] = None
    change_5d: Optional[float] = None
    is_spike: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            'value': round(self.value, 2),
            'level': self.level.value,
            'percentile': round(self.percentile, 1),
            'timestamp': self.timestamp.isoformat(),
            'change_1d': round(self.change_1d, 2) if self.change_1d else None,
            'change_5d': round(self.change_5d, 2) if self.change_5d else None,
            'is_spike': self.is_spike,
        }


class VIXCache:
    """Simple TTL cache for VIX data."""

    def __init__(self, ttl_seconds: int = 300):  # 5 minute default
        self.ttl_seconds = ttl_seconds
        self._cache: Dict[str, tuple] = {}

    def get(self, key: str) -> Optional[Any]:
        """Get cached value if not expired."""
        if key in self._cache:
            value, expiry = self._cache[key]
            if time.time() < expiry:
                return value
            del self._cache[key]
        return None

    def set(self, key: str, value: Any) -> None:
        """Set cached value with TTL."""
        self._cache[key] = (value, time.time() + self.ttl_seconds)

    def clear(self) -> None:
        """Clear all cached values."""
        self._cache.clear()


class MarketMonitorService:
    """
    Service for monitoring market volatility via VIX.

    Provides:
    - Real-time VIX fetching with caching
    - Historical percentile calculation
    - Volatility level classification
    - Position sizing recommendations
    - Circuit breaker integration
    """

    VIX_SYMBOL = "^VIX"
    CACHE_KEY_CURRENT = "vix_current"
    CACHE_KEY_HISTORICAL = "vix_historical"

    def __init__(self, thresholds: Optional[VIXThresholds] = None, cache_ttl: int = 300):
        self.thresholds = thresholds or VIXThresholds()
        self.cache = VIXCache(ttl_seconds=cache_ttl)
        self._last_alert_level: Optional[VIXLevel] = None

    def get_current_vix(self, force_refresh: bool = False) -> Optional[float]:
        """
        Fetch current VIX value with caching.

        Args:
            force_refresh: Bypass cache and fetch fresh data

        Returns:
            Current VIX value or None if fetch fails
        """
        if not force_refresh:
            cached = self.cache.get(self.CACHE_KEY_CURRENT)
            if cached is not None:
                return cached

        try:
            import yfinance as yf

            ticker = yf.Ticker(self.VIX_SYMBOL)
            # Get latest price - try fast_info first, fall back to history
            try:
                vix_value = ticker.fast_info.get('lastPrice')
                if vix_value is None:
                    hist = ticker.history(period="1d")
                    if not hist.empty:
                        vix_value = float(hist['Close'].iloc[-1])
            except Exception:
                hist = ticker.history(period="1d")
                if not hist.empty:
                    vix_value = float(hist['Close'].iloc[-1])
                else:
                    vix_value = None

            if vix_value is not None:
                self.cache.set(self.CACHE_KEY_CURRENT, vix_value)
                logger.info(f"VIX fetched: {vix_value:.2f}")
                return vix_value

        except ImportError:
            logger.warning("yfinance not installed, cannot fetch VIX")
        except Exception as e:
            logger.error(f"Failed to fetch VIX: {e}")

        return None

    def get_vix_history(self, days: int = 252) -> Optional[list]:
        """
        Fetch historical VIX values for percentile calculation.

        Args:
            days: Number of trading days to fetch (252 = ~1 year)

        Returns:
            List of historical VIX closing prices
        """
        cached = self.cache.get(f"{self.CACHE_KEY_HISTORICAL}_{days}")
        if cached is not None:
            return cached

        try:
            import yfinance as yf

            ticker = yf.Ticker(self.VIX_SYMBOL)
            hist = ticker.history(period=f"{days}d")

            if hist.empty:
                return None

            values = hist['Close'].tolist()
            self.cache.set(f"{self.CACHE_KEY_HISTORICAL}_{days}", values)
            return values

        except Exception as e:
            logger.error(f"Failed to fetch VIX history: {e}")
            return None

    def get_vix_percentile(self, current_vix: Optional[float] = None) -> Optional[float]:
        """
        Calculate where current VIX stands vs historical values.

        Args:
            current_vix: VIX value to compare (fetches current if None)

        Returns:
            Percentile (0-100) or None if calculation fails
        """
        if current_vix is None:
            current_vix = self.get_current_vix()

        if current_vix is None:
            return None

        history = self.get_vix_history()
        if not history or len(history) < 20:
            return None

        # Calculate percentile
        below_count = sum(1 for v in history if v < current_vix)
        percentile = (below_count / len(history)) * 100

        return percentile

    def get_vix_level(self, vix_value: Optional[float] = None) -> VIXLevel:
        """
        Classify VIX value into volatility level.

        Args:
            vix_value: VIX to classify (fetches current if None)

        Returns:
            VIXLevel enum value
        """
        if vix_value is None:
            vix_value = self.get_current_vix()

        if vix_value is None:
            return VIXLevel.NORMAL  # Default to normal if can't fetch

        if vix_value > self.thresholds.extreme_max:
            return VIXLevel.CRITICAL
        elif vix_value > self.thresholds.high_max:
            return VIXLevel.EXTREME
        elif vix_value > self.thresholds.elevated_max:
            return VIXLevel.HIGH
        elif vix_value > self.thresholds.normal_max:
            return VIXLevel.ELEVATED
        else:
            return VIXLevel.NORMAL

    def is_vix_elevated(self, vix_value: Optional[float] = None) -> bool:
        """Check if VIX is above elevated threshold (> 25)."""
        if vix_value is None:
            vix_value = self.get_current_vix()
        return vix_value is not None and vix_value > self.thresholds.elevated_max

    def is_vix_extreme(self, vix_value: Optional[float] = None) -> bool:
        """Check if VIX is at extreme levels (> 35)."""
        if vix_value is None:
            vix_value = self.get_current_vix()
        return vix_value is not None and vix_value > self.thresholds.high_max

    def is_vix_critical(self, vix_value: Optional[float] = None) -> bool:
        """Check if VIX is at critical levels (> 45)."""
        if vix_value is None:
            vix_value = self.get_current_vix()
        return vix_value is not None and vix_value > self.thresholds.extreme_max

    def detect_vix_spike(self, current_vix: Optional[float] = None) -> bool:
        """
        Detect if VIX has spiked significantly (> 20% increase in 1 day).

        Returns:
            True if VIX spike detected
        """
        if current_vix is None:
            current_vix = self.get_current_vix()

        if current_vix is None:
            return False

        history = self.get_vix_history(days=5)
        if not history or len(history) < 2:
            return False

        prev_close = history[-2] if len(history) >= 2 else history[0]
        change_pct = ((current_vix - prev_close) / prev_close) * 100

        # Spike = > 20% increase
        return change_pct > 20

    def get_vix_data(self, force_refresh: bool = False) -> Optional[VIXData]:
        """
        Get comprehensive VIX data including level, percentile, and changes.

        Returns:
            VIXData object with all metrics
        """
        current_vix = self.get_current_vix(force_refresh=force_refresh)
        if current_vix is None:
            return None

        level = self.get_vix_level(current_vix)
        percentile = self.get_vix_percentile(current_vix)
        is_spike = self.detect_vix_spike(current_vix)

        # Calculate changes
        history = self.get_vix_history(days=10)
        change_1d = None
        change_5d = None

        if history and len(history) >= 2:
            change_1d = current_vix - history[-2]
        if history and len(history) >= 5:
            change_5d = current_vix - history[-5]

        return VIXData(
            value=current_vix,
            level=level,
            percentile=percentile or 50.0,
            timestamp=datetime.now(),
            change_1d=change_1d,
            change_5d=change_5d,
            is_spike=is_spike,
        )

    def get_position_size_multiplier(self, vix_value: Optional[float] = None) -> float:
        """
        Calculate position size multiplier based on VIX level.

        Returns:
            Multiplier (0.0 to 1.0) to apply to position sizes
        """
        level = self.get_vix_level(vix_value)

        if level == VIXLevel.CRITICAL:
            return 0.0  # No new positions
        elif level == VIXLevel.EXTREME:
            return self.thresholds.extreme_position_reduction
        elif level == VIXLevel.HIGH:
            return self.thresholds.high_position_reduction
        elif level == VIXLevel.ELEVATED:
            return self.thresholds.elevated_position_reduction
        else:
            return 1.0  # Normal sizing

    def should_pause_trading(self, vix_value: Optional[float] = None) -> bool:
        """
        Determine if trading should be paused due to extreme volatility.

        Returns:
            True if trading should be halted
        """
        level = self.get_vix_level(vix_value)
        return level == VIXLevel.CRITICAL and self.thresholds.critical_pause_trading

    def check_alert_threshold(self, vix_data: Optional[VIXData] = None) -> Optional[Dict[str, Any]]:
        """
        Check if VIX level change warrants an alert.

        Returns:
            Alert dict if threshold crossed, None otherwise
        """
        if vix_data is None:
            vix_data = self.get_vix_data()

        if vix_data is None:
            return None

        current_level = vix_data.level

        # Check for level change
        if self._last_alert_level != current_level:
            alert = {
                'type': 'vix_level_change',
                'previous_level': self._last_alert_level.value if self._last_alert_level else 'unknown',
                'current_level': current_level.value,
                'vix_value': vix_data.value,
                'percentile': vix_data.percentile,
                'timestamp': datetime.now().isoformat(),
                'severity': self._get_alert_severity(current_level),
                'message': self._get_alert_message(current_level, vix_data),
            }
            self._last_alert_level = current_level
            return alert

        # Check for spike
        if vix_data.is_spike:
            return {
                'type': 'vix_spike',
                'vix_value': vix_data.value,
                'change_1d': vix_data.change_1d,
                'timestamp': datetime.now().isoformat(),
                'severity': 'high',
                'message': f"VIX spike detected: +{vix_data.change_1d:.1f} ({vix_data.value:.1f})",
            }

        return None

    def _get_alert_severity(self, level: VIXLevel) -> str:
        """Map VIX level to alert severity."""
        severity_map = {
            VIXLevel.NORMAL: 'info',
            VIXLevel.ELEVATED: 'low',
            VIXLevel.HIGH: 'medium',
            VIXLevel.EXTREME: 'high',
            VIXLevel.CRITICAL: 'critical',
        }
        return severity_map.get(level, 'medium')

    def _get_alert_message(self, level: VIXLevel, vix_data: VIXData) -> str:
        """Generate human-readable alert message."""
        messages = {
            VIXLevel.NORMAL: f"VIX returned to normal levels ({vix_data.value:.1f})",
            VIXLevel.ELEVATED: f"VIX elevated ({vix_data.value:.1f}) - Position sizes reduced 25%",
            VIXLevel.HIGH: f"VIX high ({vix_data.value:.1f}) - Position sizes reduced 50%",
            VIXLevel.EXTREME: f"VIX extreme ({vix_data.value:.1f}) - Position sizes reduced 75%",
            VIXLevel.CRITICAL: f"VIX critical ({vix_data.value:.1f}) - TRADING PAUSED",
        }
        return messages.get(level, f"VIX: {vix_data.value:.1f}")

    def get_status(self) -> Dict[str, Any]:
        """
        Get full VIX monitoring status for dashboard display.

        Returns:
            Dict with VIX data, thresholds, and recommendations
        """
        vix_data = self.get_vix_data()

        if vix_data is None:
            return {
                'available': False,
                'error': 'Unable to fetch VIX data',
            }

        return {
            'available': True,
            'vix': vix_data.to_dict(),
            'thresholds': {
                'normal_max': self.thresholds.normal_max,
                'elevated_max': self.thresholds.elevated_max,
                'high_max': self.thresholds.high_max,
                'extreme_max': self.thresholds.extreme_max,
            },
            'position_multiplier': self.get_position_size_multiplier(vix_data.value),
            'trading_paused': self.should_pause_trading(vix_data.value),
            'recommendations': self._get_recommendations(vix_data),
        }

    def _get_recommendations(self, vix_data: VIXData) -> list:
        """Generate trading recommendations based on VIX level."""
        recommendations = []

        if vix_data.level == VIXLevel.CRITICAL:
            recommendations.append("HALT all new position entries")
            recommendations.append("Consider reducing existing exposure")
            recommendations.append("Review stop-loss levels")
        elif vix_data.level == VIXLevel.EXTREME:
            recommendations.append("Reduce position sizes by 75%")
            recommendations.append("Avoid leveraged positions")
            recommendations.append("Focus on defensive sectors")
        elif vix_data.level == VIXLevel.HIGH:
            recommendations.append("Reduce position sizes by 50%")
            recommendations.append("Tighten stop-losses")
            recommendations.append("Consider hedging strategies")
        elif vix_data.level == VIXLevel.ELEVATED:
            recommendations.append("Reduce position sizes by 25%")
            recommendations.append("Increase cash allocation")
            recommendations.append("Monitor for further increases")
        else:
            recommendations.append("Normal trading conditions")
            recommendations.append("Standard position sizing")

        if vix_data.is_spike:
            recommendations.insert(0, "⚠️ VIX spike detected - exercise caution")

        return recommendations


# Global singleton instance
_market_monitor: Optional[MarketMonitorService] = None


def get_market_monitor() -> MarketMonitorService:
    """Get or create the global MarketMonitorService instance."""
    global _market_monitor
    if _market_monitor is None:
        _market_monitor = MarketMonitorService()
    return _market_monitor
