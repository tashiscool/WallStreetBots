"""Advanced Pattern Detection for WSB - style Trading
Replaces oversimplified logic with sophisticated technical analysis.
"""

import logging
import statistics
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class PriceBar:
    """Price bar data structure."""

    timestamp: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: int


@dataclass
class PatternSignal:
    """Enhanced pattern recognition signal."""

    ticker: str
    signal_type: str
    confidence: float
    entry_price: Decimal
    signal_strength: int
    metadata: dict[str, Any]


class TechnicalIndicators:
    """Technical analysis indicators."""

    @staticmethod
    def calculate_rsi(prices: list[Decimal], period: int = 14) -> Decimal | None:
        """Calculate RSI (Relative Strength Index)."""
        if len(prices) < period + 1:
            return None

        try:
            deltas = [float(prices[i]) - float(prices[i - 1]) for i in range(1, len(prices))]

            gains = [max(0, delta) for delta in deltas]
            losses = [max(0, -delta) for delta in deltas]

            # Calculate initial averages
            avg_gain = statistics.mean(gains[:period])
            avg_loss = statistics.mean(losses[:period])

            # Calculate subsequent values using smoothing
            for i in range(period, len(deltas)):
                avg_gain = (avg_gain * (period - 1) + gains[i]) / period
                avg_loss = (avg_loss * (period - 1) + losses[i]) / period

            if avg_loss == 0:
                return Decimal("100")

            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))

            return Decimal(str(round(rsi, 2)))

        except Exception as e:
            logger.error(f"Error calculating RSI: {e}")
            return None

    @staticmethod
    def calculate_sma(prices: list[Decimal], period: int) -> Decimal | None:
        """Calculate Simple Moving Average."""
        if len(prices) < period:
            return None

        try:
            recent_prices = prices[-period:]
            avg = statistics.mean([float(p) for p in recent_prices])
            return Decimal(str(round(avg, 4)))
        except Exception as e:
            logger.error(f"Error calculating SMA: {e}")
            return None

    @staticmethod
    def calculate_bollinger_bands(
        prices: list[Decimal], period: int = 20, std_dev: float = 2.0
    ) -> dict[str, Decimal] | None:
        """Calculate Bollinger Bands."""
        if len(prices) < period:
            return None

        try:
            recent_prices = prices[-period:]
            price_floats = [float(p) for p in recent_prices]

            sma = statistics.mean(price_floats)
            std = statistics.stdev(price_floats)

            upper_band = sma + (std * std_dev)
            lower_band = sma - (std * std_dev)

            return {
                "upper": Decimal(str(round(upper_band, 4))),
                "middle": Decimal(str(round(sma, 4))),
                "lower": Decimal(str(round(lower_band, 4))),
                "position": (float(prices[-1]) - lower_band) / (upper_band - lower_band),
            }

        except Exception as e:
            logger.error(f"Error calculating Bollinger Bands: {e}")
            return None

    @staticmethod
    def calculate_volume_spike(volumes: list[int], period: int = 20) -> float | None:
        """Calculate volume spike ratio."""
        if len(volumes) < period + 1:
            return None

        try:
            recent_volume = volumes[-1]
            avg_volume = statistics.mean(volumes[-period - 1 : -1])  # Exclude current day

            if avg_volume == 0:
                return None

            spike_ratio = recent_volume / avg_volume
            return round(spike_ratio, 2)

        except Exception as e:
            logger.error(f"Error calculating volume spike: {e}")
            return None


class WSBDipDetector:
    """Advanced WSB dip detection with multiple confirmations."""

    def __init__(self):
        # Enhanced thresholds for better signal quality
        self.min_run_percentage = 0.20  # 20% minimum run (more selective)
        self.min_dip_percentage = 0.05  # 5% minimum dip (deeper dips)
        self.max_dip_age_days = 3  # Dip must be recent
        self.volume_spike_threshold = 1.5  # 50% above average volume
        self.rsi_oversold_threshold = 35  # RSI below 35 indicates oversold

        # Advanced pattern requirements
        self.min_run_duration = 1  # Minimum 1 day for run
        self.max_run_duration = 15  # Maximum 15 days for run
        self.volume_confirmation_required = True
        self.technical_confirmation_required = True

    async def detect_wsb_dip_pattern(
        self, ticker: str, price_history: list[PriceBar]
    ) -> PatternSignal | None:
        """Detect WSB - style dip - after - run pattern with multiple confirmations."""
        try:
            if len(price_history) < 30:  # Need at least 30 days of data
                return None

            # Extract price and volume arrays
            closes = [bar.close for bar in price_history]
            highs = [bar.high for bar in price_history]
            volumes = [bar.volume for bar in price_history]

            # Step 1: Identify the "big run"
            run_analysis = self._analyze_recent_run(closes, highs)
            if not run_analysis["valid_run"]:
                return None

            # Step 2: Identify the current dip
            dip_analysis = self._analyze_current_dip(closes, highs)
            if not dip_analysis["valid_dip"]:
                return None

            # Step 3: Volume analysis
            volume_analysis = self._analyze_volume_pattern(volumes)

            # Step 4: Technical indicators
            technical_analysis = self._analyze_technical_indicators(closes, volumes)

            # Step 5: Calculate signal strength
            signal_strength = self._calculate_signal_strength(
                run_analysis, dip_analysis, volume_analysis, technical_analysis
            )

            # Require minimum signal strength
            if signal_strength < 4:
                return None

            # Calculate confidence
            confidence = min(0.95, signal_strength / 10.0)

            return PatternSignal(
                ticker=ticker,
                signal_type="WSB_DIP_AFTER_RUN",
                confidence=confidence,
                entry_price=closes[-1],
                signal_strength=signal_strength,
                metadata={
                    "run_percentage": run_analysis["run_percentage"],
                    "dip_percentage": dip_analysis["dip_percentage"],
                    "volume_spike": volume_analysis.get("volume_spike", 0),
                    "rsi": technical_analysis.get("rsi"),
                    "bb_position": technical_analysis.get("bb_position"),
                    "days_since_high": dip_analysis["days_since_high"],
                    "run_duration": run_analysis["run_duration"],
                },
            )

        except Exception as e:
            logger.error(f"Error detecting WSB dip pattern for {ticker}: {e}")
            return None

    def _analyze_recent_run(self, closes: list[Decimal], highs: list[Decimal]) -> dict[str, Any]:
        """Analyze recent price run to identify 'big run' setup."""
        try:
            # Look for highest high in last 10 days
            recent_high = max(highs[-10:])
            recent_high_idx = len(highs) - 1 - highs[-10:][::-1].index(recent_high)

            # Find the base price before the run (look back 5 - 20 days from high)
            lookback_start = max(0, recent_high_idx - 20)
            lookback_end = max(lookback_start + 1, recent_high_idx - 5)

            if lookback_end >= recent_high_idx:
                return {"valid_run": False}

            base_price = min(closes[lookback_start:lookback_end])
            run_percentage = float((recent_high - base_price) / base_price)
            run_duration = (
                recent_high_idx
                - closes[lookback_start:lookback_end].index(base_price)
                - lookback_start
            )

            # Validate run criteria
            valid_run = (
                run_percentage >= self.min_run_percentage
                and run_duration >= 1  # At least 1 day
                and run_duration <= 15  # No more than 15 days
            )

            return {
                "valid_run": valid_run,
                "run_percentage": run_percentage,
                "run_duration": run_duration,
                "base_price": base_price,
                "recent_high": recent_high,
                "high_date_idx": recent_high_idx,
            }

        except Exception as e:
            logger.error(f"Error analyzing recent run: {e}")
            return {"valid_run": False}

    def _analyze_current_dip(self, closes: list[Decimal], highs: list[Decimal]) -> dict[str, Any]:
        """Analyze current dip from recent high."""
        try:
            # Find recent high
            recent_high = max(highs[-10:])
            recent_high_idx = len(highs) - 1 - highs[-10:][::-1].index(recent_high)

            current_price = closes[-1]
            dip_percentage = float((recent_high - current_price) / recent_high)
            days_since_high = len(closes) - 1 - recent_high_idx

            # Validate dip criteria
            valid_dip = (
                dip_percentage >= self.min_dip_percentage
                and days_since_high <= self.max_dip_age_days
                and current_price < recent_high * Decimal("0.98")  # At least 2% below high
            )

            return {
                "valid_dip": valid_dip,
                "dip_percentage": dip_percentage,
                "days_since_high": days_since_high,
                "recent_high": recent_high,
            }

        except Exception as e:
            logger.error(f"Error analyzing current dip: {e}")
            return {"valid_dip": False}

    def _analyze_volume_pattern(self, volumes: list[int]) -> dict[str, Any]:
        """Analyze volume pattern for confirmation."""
        try:
            if len(volumes) < 20:
                return {}

            # Volume spike analysis
            volume_spike = TechnicalIndicators.calculate_volume_spike(volumes, 20)

            # Volume trend (increasing volume during dip suggests selling exhaustion)
            recent_volume_avg = statistics.mean(volumes[-3:])
            prior_volume_avg = statistics.mean(volumes[-10:-3])
            volume_trend = recent_volume_avg / prior_volume_avg if prior_volume_avg > 0 else 1.0

            return {
                "volume_spike": volume_spike,
                "volume_trend": volume_trend,
                "high_volume": volume_spike and volume_spike >= self.volume_spike_threshold,
            }

        except Exception as e:
            logger.error(f"Error analyzing volume pattern: {e}")
            return {}

    def _analyze_technical_indicators(
        self, closes: list[Decimal], volumes: list[int]
    ) -> dict[str, Any]:
        """Calculate technical indicators for confirmation."""
        try:
            # RSI
            rsi = TechnicalIndicators.calculate_rsi(closes, 14)

            # Bollinger Bands
            bb = TechnicalIndicators.calculate_bollinger_bands(closes, 20, 2.0)
            bb_position = bb["position"] if bb else None

            # Price relative to moving averages
            sma_20 = TechnicalIndicators.calculate_sma(closes, 20)
            sma_50 = TechnicalIndicators.calculate_sma(closes, 50)

            current_price = closes[-1]
            price_vs_sma20 = float(current_price / sma_20) if sma_20 else None
            price_vs_sma50 = float(current_price / sma_50) if sma_50 else None

            return {
                "rsi": rsi,
                "bb_position": bb_position,
                "price_vs_sma20": price_vs_sma20,
                "price_vs_sma50": price_vs_sma50,
                "oversold_rsi": rsi and float(rsi) <= self.rsi_oversold_threshold,
                "below_lower_bb": bb_position and bb_position <= 0.2,
            }

        except Exception as e:
            logger.error(f"Error analyzing technical indicators: {e}")
            return {}

    def _calculate_signal_strength(
        self,
        run_analysis: dict,
        dip_analysis: dict,
        volume_analysis: dict,
        technical_analysis: dict,
    ) -> int:
        """Calculate overall signal strength (0 - 10 scale)."""
        strength = 0

        # Run strength (0 - 3 points)
        run_pct = run_analysis.get("run_percentage", 0)
        if run_pct >= 0.30:  # 30%+ run
            strength += 3
        elif run_pct >= 0.20:  # 20%+ run
            strength += 2
        elif run_pct >= 0.15:  # 15%+ run
            strength += 1

        # Dip strength (0 - 2 points)
        dip_pct = dip_analysis.get("dip_percentage", 0)
        if dip_pct >= 0.08:  # 8%+ dip
            strength += 2
        elif dip_pct >= 0.05:  # 5%+ dip
            strength += 1

        # Volume confirmation (0 - 2 points)
        if volume_analysis.get("high_volume", False):
            strength += 2
        elif volume_analysis.get("volume_spike", 0) >= 1.2:  # 20% above average
            strength += 1

        # Technical indicators (0 - 3 points)
        if technical_analysis.get("oversold_rsi", False):
            strength += 1
        if technical_analysis.get("below_lower_bb", False):
            strength += 1
        if technical_analysis.get("price_vs_sma20", 1.0) < 0.95:  # 5% below 20 - day SMA
            strength += 1

        return min(10, strength)


# Factory function
def create_wsb_dip_detector() -> WSBDipDetector:
    """Create and return a configured WSB dip detector."""
    return WSBDipDetector()
