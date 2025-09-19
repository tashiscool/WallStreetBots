"""
Liquid Anomaly Detection for Fast-Edge Implementation
Detects simple, liquid market anomalies for rapid deployment and profit capture.
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
from collections import defaultdict, deque
import yfinance as yf
import threading


class AnomalyType(Enum):
    VOLUME_SPIKE = "volume_spike"
    PRICE_GAP = "price_gap"
    SPREAD_COMPRESSION = "spread_compression"
    ORDER_BOOK_IMBALANCE = "order_book_imbalance"
    MOMENTUM_REVERSAL = "momentum_reversal"
    EARNINGS_DRIFT = "earnings_drift"
    ETF_ARBITRAGE = "etf_arbitrage"
    OPTIONS_SKEW = "options_skew"


class LiquidityTier(Enum):
    ULTRA_LIQUID = "ultra_liquid"  # SPY, QQQ, AAPL, etc.
    HIGH_LIQUID = "high_liquid"    # S&P 500 components
    MEDIUM_LIQUID = "medium_liquid" # Russell 1000
    LOW_LIQUID = "low_liquid"      # Small caps


@dataclass
class AnomalySignal:
    """Represents a detected market anomaly."""
    symbol: str
    anomaly_type: AnomalyType
    strength: float  # 0-1 confidence score
    timestamp: datetime
    expected_duration_minutes: int
    target_profit_bps: int  # Basis points
    max_position_size: float
    entry_price: float
    stop_loss: float
    take_profit: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    liquidity_tier: LiquidityTier = LiquidityTier.MEDIUM_LIQUID


@dataclass
class MarketState:
    """Current market state for anomaly detection."""
    symbol: str
    timestamp: datetime
    price: float
    volume: int
    bid: float
    ask: float
    spread_bps: float
    volume_ratio_1m: float  # Volume vs 1-month average
    momentum_5m: float      # 5-minute momentum
    momentum_1h: float      # 1-hour momentum
    rsi_14: float          # 14-period RSI
    volatility_20d: float  # 20-day realized volatility


class LiquidityClassifier:
    """Classifies securities by liquidity for appropriate anomaly detection."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.ultra_liquid_symbols = {
            'SPY', 'QQQ', 'IWM', 'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'TSLA',
            'NVDA', 'META', 'BRK.B', 'UNH', 'JNJ', 'V', 'PG', 'HD', 'MA',
            'BAC', 'XOM', 'DIS', 'ADBE', 'CRM', 'NFLX', 'KO', 'PEP'
        }
        self.liquidity_cache: Dict[str, Tuple[LiquidityTier, datetime]] = {}

    def classify_liquidity(self, symbol: str, market_state: MarketState) -> LiquidityTier:
        """Classify symbol liquidity tier."""
        # Check cache first
        if symbol in self.liquidity_cache:
            tier, timestamp = self.liquidity_cache[symbol]
            if datetime.now() - timestamp < timedelta(hours=1):
                return tier

        # Ultra liquid predefined list
        if symbol in self.ultra_liquid_symbols:
            tier = LiquidityTier.ULTRA_LIQUID
        else:
            # Classify based on metrics
            tier = self._classify_by_metrics(symbol, market_state)

        # Cache result
        self.liquidity_cache[symbol] = (tier, datetime.now())
        return tier

    def _classify_by_metrics(self, symbol: str, market_state: MarketState) -> LiquidityTier:
        """Classify liquidity based on market metrics."""
        try:
            # Spread-based classification
            spread_bps = market_state.spread_bps

            # Volume-based classification
            avg_volume = market_state.volume * (1 / max(0.1, market_state.volume_ratio_1m))

            if spread_bps <= 2 and avg_volume >= 1_000_000:
                return LiquidityTier.HIGH_LIQUID
            elif spread_bps <= 5 and avg_volume >= 100_000:
                return LiquidityTier.MEDIUM_LIQUID
            else:
                return LiquidityTier.LOW_LIQUID

        except Exception as e:
            self.logger.warning(f"Error classifying liquidity for {symbol}: {e}")
            return LiquidityTier.LOW_LIQUID


class VolumeAnomalyDetector:
    """Detects unusual volume patterns."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.volume_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))

    def detect_volume_spike(self, market_state: MarketState) -> Optional[AnomalySignal]:
        """Detect volume spikes indicating potential breakouts."""
        symbol = market_state.symbol
        volume_ratio = market_state.volume_ratio_1m

        # Volume spike thresholds by liquidity
        liquidity_classifier = LiquidityClassifier()
        liquidity_tier = liquidity_classifier.classify_liquidity(symbol, market_state)

        thresholds = {
            LiquidityTier.ULTRA_LIQUID: 3.0,   # 3x normal volume
            LiquidityTier.HIGH_LIQUID: 2.5,    # 2.5x normal volume
            LiquidityTier.MEDIUM_LIQUID: 2.0,  # 2x normal volume
            LiquidityTier.LOW_LIQUID: 1.8      # 1.8x normal volume
        }

        threshold = thresholds[liquidity_tier]

        if volume_ratio >= threshold:
            # Calculate strength based on how much above threshold
            strength = min(1.0, (volume_ratio - threshold) / threshold)

            # Determine position sizing based on liquidity and strength
            max_position_sizes = {
                LiquidityTier.ULTRA_LIQUID: 1_000_000,
                LiquidityTier.HIGH_LIQUID: 500_000,
                LiquidityTier.MEDIUM_LIQUID: 100_000,
                LiquidityTier.LOW_LIQUID: 25_000
            }

            max_position = max_position_sizes[liquidity_tier] * strength

            # Set stops and targets based on momentum direction
            momentum = market_state.momentum_5m
            if momentum > 0:  # Upward momentum
                stop_loss = market_state.price * 0.98    # 2% stop
                take_profit = market_state.price * 1.04  # 4% target
            else:  # Downward momentum
                stop_loss = market_state.price * 1.02    # 2% stop
                take_profit = market_state.price * 0.96  # 4% target

            return AnomalySignal(
                symbol=symbol,
                anomaly_type=AnomalyType.VOLUME_SPIKE,
                strength=strength,
                timestamp=market_state.timestamp,
                expected_duration_minutes=30,
                target_profit_bps=200,  # 200 bps target
                max_position_size=max_position,
                entry_price=market_state.price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                liquidity_tier=liquidity_tier,
                metadata={
                    'volume_ratio': volume_ratio,
                    'threshold': threshold,
                    'momentum_5m': momentum
                }
            )

        return None


class SpreadAnomalyDetector:
    """Detects bid-ask spread anomalies."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.spread_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))

    def detect_spread_compression(self, market_state: MarketState) -> Optional[AnomalySignal]:
        """Detect unusual spread compression indicating liquidity influx."""
        symbol = market_state.symbol
        current_spread = market_state.spread_bps

        # Store historical spreads
        self.spread_history[symbol].append((market_state.timestamp, current_spread))

        if len(self.spread_history[symbol]) < 20:
            return None  # Need history

        # Calculate average spread over last 20 periods
        recent_spreads = [spread for _, spread in list(self.spread_history[symbol])[-20:]]
        avg_spread = np.mean(recent_spreads)
        std_spread = np.std(recent_spreads)

        # Detect compression (current spread significantly below average)
        if std_spread > 0 and current_spread < (avg_spread - 2 * std_spread):
            compression_strength = min(1.0, (avg_spread - current_spread) / avg_spread)

            # Spread compression often precedes volatility
            return AnomalySignal(
                symbol=symbol,
                anomaly_type=AnomalyType.SPREAD_COMPRESSION,
                strength=compression_strength,
                timestamp=market_state.timestamp,
                expected_duration_minutes=15,
                target_profit_bps=100,
                max_position_size=50_000,  # Smaller position for volatility plays
                entry_price=market_state.price,
                stop_loss=market_state.price * 0.995,  # Tight stop
                take_profit=market_state.price * 1.01,  # Quick profit
                metadata={
                    'current_spread_bps': current_spread,
                    'avg_spread_bps': avg_spread,
                    'compression_strength': compression_strength
                }
            )

        return None


class MomentumAnomalyDetector:
    """Detects momentum-based anomalies."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def detect_momentum_reversal(self, market_state: MarketState) -> Optional[AnomalySignal]:
        """Detect potential momentum reversals."""
        momentum_5m = market_state.momentum_5m
        momentum_1h = market_state.momentum_1h
        rsi = market_state.rsi_14

        # Look for divergence between short and long momentum
        momentum_divergence = abs(momentum_5m - momentum_1h)

        # RSI extremes suggest reversal potential
        rsi_extreme = rsi <= 30 or rsi >= 70

        if momentum_divergence > 0.02 and rsi_extreme:  # 2% divergence
            # Determine reversal direction
            if rsi <= 30 and momentum_5m > momentum_1h:
                # Oversold with recent uptick
                direction = "long"
                stop_loss = market_state.price * 0.97
                take_profit = market_state.price * 1.05
            elif rsi >= 70 and momentum_5m < momentum_1h:
                # Overbought with recent downtick
                direction = "short"
                stop_loss = market_state.price * 1.03
                take_profit = market_state.price * 0.95
            else:
                return None

            strength = min(1.0, momentum_divergence / 0.05)  # Scale to 5% max

            return AnomalySignal(
                symbol=market_state.symbol,
                anomaly_type=AnomalyType.MOMENTUM_REVERSAL,
                strength=strength,
                timestamp=market_state.timestamp,
                expected_duration_minutes=60,
                target_profit_bps=300,
                max_position_size=100_000,
                entry_price=market_state.price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                metadata={
                    'direction': direction,
                    'momentum_5m': momentum_5m,
                    'momentum_1h': momentum_1h,
                    'rsi': rsi,
                    'divergence': momentum_divergence
                }
            )

        return None


class PriceGapDetector:
    """Detects and analyzes price gaps."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.previous_close: Dict[str, float] = {}

    def detect_gap_fill_opportunity(self, market_state: MarketState) -> Optional[AnomalySignal]:
        """Detect gaps that are likely to fill."""
        symbol = market_state.symbol
        current_price = market_state.price

        if symbol not in self.previous_close:
            self.previous_close[symbol] = current_price
            return None

        prev_close = self.previous_close[symbol]
        gap_percent = (current_price - prev_close) / prev_close

        # Look for significant gaps (>1% for liquid stocks)
        if abs(gap_percent) > 0.01:
            # Gap up or gap down
            gap_direction = "up" if gap_percent > 0 else "down"

            # Gap fill probability based on volume and momentum
            volume_confirmation = market_state.volume_ratio_1m > 1.5
            momentum_alignment = (gap_percent > 0 and market_state.momentum_5m > 0) or \
                               (gap_percent < 0 and market_state.momentum_5m < 0)

            if not momentum_alignment and volume_confirmation:
                # Gap without momentum support - likely to fill
                strength = min(1.0, abs(gap_percent) / 0.05)  # Scale to 5% max gap

                if gap_direction == "up":
                    # Gap up likely to fill down
                    target_price = prev_close
                    stop_loss = current_price * 1.02
                else:
                    # Gap down likely to fill up
                    target_price = prev_close
                    stop_loss = current_price * 0.98

                return AnomalySignal(
                    symbol=symbol,
                    anomaly_type=AnomalyType.PRICE_GAP,
                    strength=strength,
                    timestamp=market_state.timestamp,
                    expected_duration_minutes=120,
                    target_profit_bps=int(abs(gap_percent) * 10000 * 0.8),  # 80% of gap
                    max_position_size=200_000,
                    entry_price=current_price,
                    stop_loss=stop_loss,
                    take_profit=target_price,
                    metadata={
                        'gap_percent': gap_percent,
                        'gap_direction': gap_direction,
                        'previous_close': prev_close,
                        'volume_confirmation': volume_confirmation
                    }
                )

        # Update previous close for next check
        self.previous_close[symbol] = current_price
        return None


class LiquidAnomalyEngine:
    """Main engine for detecting liquid market anomalies."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.detectors = {
            'volume': VolumeAnomalyDetector(),
            'spread': SpreadAnomalyDetector(),
            'momentum': MomentumAnomalyDetector(),
            'gap': PriceGapDetector()
        }
        self.liquidity_classifier = LiquidityClassifier()
        self.detected_signals: List[AnomalySignal] = []
        self.watchlist = self._build_liquid_watchlist()

    def _build_liquid_watchlist(self) -> List[str]:
        """Build watchlist of liquid securities for monitoring."""
        return [
            # Major ETFs
            'SPY', 'QQQ', 'IWM', 'VTI', 'VEA', 'VWO', 'GLD', 'SLV', 'TLT', 'HYG',

            # Mega caps
            'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'TSLA', 'META', 'NVDA', 'BRK.B',

            # High volume stocks
            'BAC', 'JPM', 'WFC', 'GE', 'F', 'AMD', 'INTC', 'XOM', 'CVX',
            'JNJ', 'PFE', 'UNH', 'V', 'MA', 'DIS', 'KO', 'PEP', 'WMT', 'HD'
        ]

    async def scan_for_anomalies(self, market_states: Dict[str, MarketState]) -> List[AnomalySignal]:
        """Scan market states for anomalies across all detectors."""
        new_signals = []

        for symbol, market_state in market_states.items():
            # Only scan liquid securities
            liquidity_tier = self.liquidity_classifier.classify_liquidity(symbol, market_state)
            if liquidity_tier == LiquidityTier.LOW_LIQUID:
                continue

            # Run all detectors
            for detector_name, detector in self.detectors.items():
                try:
                    if detector_name == 'volume':
                        signal = detector.detect_volume_spike(market_state)
                    elif detector_name == 'spread':
                        signal = detector.detect_spread_compression(market_state)
                    elif detector_name == 'momentum':
                        signal = detector.detect_momentum_reversal(market_state)
                    elif detector_name == 'gap':
                        signal = detector.detect_gap_fill_opportunity(market_state)
                    else:
                        continue

                    if signal and self._validate_signal(signal):
                        new_signals.append(signal)
                        self.logger.info(f"Detected {signal.anomaly_type.value} in {symbol}: {signal.strength:.2f}")

                except Exception as e:
                    self.logger.error(f"Detector {detector_name} failed for {symbol}: {e}")

        # Filter and rank signals
        filtered_signals = self._filter_and_rank_signals(new_signals)
        self.detected_signals.extend(filtered_signals)

        return filtered_signals

    def _validate_signal(self, signal: AnomalySignal) -> bool:
        """Validate signal meets minimum criteria."""
        # Minimum strength threshold
        if signal.strength < 0.3:
            return False

        # Minimum profit target
        if signal.target_profit_bps < 50:  # 50 bps minimum
            return False

        # Risk/reward ratio check
        entry_price = signal.entry_price
        stop_distance = abs(entry_price - signal.stop_loss) / entry_price
        profit_distance = abs(signal.take_profit - entry_price) / entry_price

        if profit_distance / stop_distance < 1.5:  # Min 1.5:1 reward/risk
            return False

        return True

    def _filter_and_rank_signals(self, signals: List[AnomalySignal]) -> List[AnomalySignal]:
        """Filter duplicate signals and rank by attractiveness."""
        # Remove duplicates (same symbol, same type within 5 minutes)
        filtered = []
        seen = set()

        for signal in signals:
            key = (signal.symbol, signal.anomaly_type.value)
            if key not in seen:
                filtered.append(signal)
                seen.add(key)

        # Rank by combined score: strength * liquidity_weight * profit_target
        liquidity_weights = {
            LiquidityTier.ULTRA_LIQUID: 1.0,
            LiquidityTier.HIGH_LIQUID: 0.8,
            LiquidityTier.MEDIUM_LIQUID: 0.6,
            LiquidityTier.LOW_LIQUID: 0.3
        }

        for signal in filtered:
            liquidity_weight = liquidity_weights[signal.liquidity_tier]
            signal.metadata['rank_score'] = (
                signal.strength *
                liquidity_weight *
                (signal.target_profit_bps / 100)
            )

        # Sort by rank score descending
        filtered.sort(key=lambda s: s.metadata.get('rank_score', 0), reverse=True)

        return filtered[:10]  # Return top 10 signals

    def get_current_signals(self, max_age_minutes: int = 30) -> List[AnomalySignal]:
        """Get currently active signals."""
        cutoff_time = datetime.now() - timedelta(minutes=max_age_minutes)
        return [s for s in self.detected_signals if s.timestamp >= cutoff_time]

    async def generate_market_states(self) -> Dict[str, MarketState]:
        """Generate market states for watchlist symbols."""
        market_states = {}

        # In production, this would connect to real market data
        # For demo, generate realistic market states
        for symbol in self.watchlist[:10]:  # Limit for demo
            try:
                # Simulate market data
                price = 100 + np.random.normal(0, 5)
                volume = int(np.random.lognormal(13, 1))  # ~300K average
                spread_bps = max(1, np.random.exponential(3))

                market_state = MarketState(
                    symbol=symbol,
                    timestamp=datetime.now(),
                    price=price,
                    volume=volume,
                    bid=price - spread_bps/10000 * price / 2,
                    ask=price + spread_bps/10000 * price / 2,
                    spread_bps=spread_bps,
                    volume_ratio_1m=np.random.lognormal(0, 0.5),
                    momentum_5m=np.random.normal(0, 0.02),
                    momentum_1h=np.random.normal(0, 0.05),
                    rsi_14=np.random.uniform(20, 80),
                    volatility_20d=np.random.uniform(0.15, 0.35)
                )

                market_states[symbol] = market_state

            except Exception as e:
                self.logger.error(f"Failed to generate market state for {symbol}: {e}")

        return market_states

    def get_signal_summary(self) -> Dict[str, Any]:
        """Get summary of detected signals."""
        active_signals = self.get_current_signals()

        by_type = defaultdict(int)
        by_liquidity = defaultdict(int)

        for signal in active_signals:
            by_type[signal.anomaly_type.value] += 1
            by_liquidity[signal.liquidity_tier.value] += 1

        return {
            'total_active_signals': len(active_signals),
            'by_anomaly_type': dict(by_type),
            'by_liquidity_tier': dict(by_liquidity),
            'avg_strength': np.mean([s.strength for s in active_signals]) if active_signals else 0,
            'avg_profit_target_bps': np.mean([s.target_profit_bps for s in active_signals]) if active_signals else 0,
            'total_potential_position_size': sum([s.max_position_size for s in active_signals])
        }


# Example usage and testing
if __name__ == "__main__":
    async def demo_liquid_anomaly_detection():
        print("=== Liquid Anomaly Detection Demo ===")

        engine = LiquidAnomalyEngine()

        # Generate market states
        print("Generating market states...")
        market_states = await engine.generate_market_states()
        print(f"Generated states for {len(market_states)} symbols")

        # Scan for anomalies
        print("\nScanning for anomalies...")
        signals = await engine.scan_for_anomalies(market_states)

        print(f"Detected {len(signals)} anomaly signals:")
        for signal in signals[:5]:  # Show top 5
            print(f"  {signal.symbol}: {signal.anomaly_type.value}")
            print(f"    Strength: {signal.strength:.2f}")
            print(f"    Target: {signal.target_profit_bps} bps")
            print(f"    Max Size: ${signal.max_position_size:,.0f}")
            print(f"    Risk/Reward: 1:{signal.target_profit_bps/200:.1f}")
            print()

        # Get summary
        summary = engine.get_signal_summary()
        print("Signal Summary:")
        print(f"  Active signals: {summary['total_active_signals']}")
        print(f"  Avg strength: {summary['avg_strength']:.2f}")
        print(f"  Avg profit target: {summary['avg_profit_target_bps']:.0f} bps")
        print(f"  Total position size: ${summary['total_potential_position_size']:,.0f}")

    asyncio.run(demo_liquid_anomaly_detection())