#!/usr / bin / env python3
"""Market Regime Adapter
Integrates market regime detection with strategy parameter adaptation.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

# Import the existing market regime system
from ..market_regime import (
    MarketRegime,
    SignalGenerator,
    TechnicalIndicators,
    create_sample_indicators,
)

logger = logging.getLogger(__name__)


class AdaptationLevel(Enum):
    """Strategy adaptation intensity levels."""

    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


@dataclass
class RegimeAdaptationConfig:
    """Configuration for regime-based adaptation."""

    # Position sizing adjustments by regime
    bull_position_multiplier: float = 1.2  # Increase positions in bull market
    bear_position_multiplier: float = 0.3  # Reduce positions in bear market
    sideways_position_multiplier: float = 0.7  # Moderate positions in sideways

    # Risk adjustments by regime
    bull_max_risk: float = 0.05  # 5% max risk per trade in bull
    bear_max_risk: float = 0.02  # 2% max risk per trade in bear
    sideways_max_risk: float = 0.03  # 3% max risk per trade in sideways

    # Strategy selection by regime
    enabled_strategies_by_regime: dict[str, list[str]] = field(
        default_factory=lambda: {
            "bull": ["wsb_dip_bot", "momentum_weeklies", "debit_spreads", "leaps_tracker"],
            "bear": ["spx_credit_spreads", "wheel_strategy"],
            "sideways": ["wheel_strategy", "spx_credit_spreads", "index_baseline"],
            "undefined": ["index_baseline"],
        }
    )

    # Time-based adaptations
    adapt_during_earnings: bool = True
    adapt_during_fomc: bool = True
    adapt_during_opex: bool = True

    # Confidence thresholds
    min_regime_confidence: float = 0.7
    regime_change_cooldown_hours: int = 4


@dataclass
class StrategyAdaptation:
    """Strategy adaptation recommendations."""

    regime: MarketRegime
    confidence: float

    # Position sizing
    position_size_multiplier: float
    max_risk_per_trade: float

    # Strategy selection
    recommended_strategies: list[str]
    disabled_strategies: list[str]

    # Parameters adjustments
    parameter_adjustments: dict[str, Any]

    # Risk management
    stop_loss_adjustment: float  # Multiplier for stop losses
    take_profit_adjustment: float  # Multiplier for take profits

    # Timing adjustments
    entry_delay: int  # Minutes to delay entries
    exit_urgency: float  # Multiplier for exit speed

    # Metadata
    timestamp: datetime
    reason: str
    next_review: datetime


class MarketRegimeAdapter:
    """Market Regime Strategy Adapter.

    Integrates market regime detection with dynamic strategy adaptation:
    - Adjusts position sizes based on market regime
    - Enables / disables strategies based on market conditions
    - Modifies risk parameters dynamically
    - Adapts to calendar events and volatility
    """

    def __init__(self, config: RegimeAdaptationConfig = None):
        """Initialize market regime adapter.

        Args:
            config: Adaptation configuration
        """
        self.config = config or RegimeAdaptationConfig()
        self.signal_generator = SignalGenerator()
        self.logger = logging.getLogger(__name__)

        # State tracking
        self.current_regime = MarketRegime.UNDEFINED
        self.regime_confidence = 0.0
        self.last_regime_change = None
        self.adaptation_history: list[StrategyAdaptation] = []

        # Market data cache
        self.indicator_history: list[TechnicalIndicators] = []
        self.max_history_length = 100

        self.logger.info("Market Regime Adapter initialized")

    async def detect_current_regime(self, market_data: dict[str, Any]) -> MarketRegime:
        """Detect current market regime from market data.

        Args:
            market_data: Market data containing price, volume, indicators

        Returns:
            Current market regime
        """
        try:
            # Extract indicators from market data
            indicators = self._extract_indicators(market_data)

            if not indicators:
                return MarketRegime.UNDEFINED

            # Store in history
            self.indicator_history.append(indicators)
            if len(self.indicator_history) > self.max_history_length:
                self.indicator_history = self.indicator_history[-self.max_history_length :]

            # Need at least 2 data points for comparison
            if len(self.indicator_history) < 2:
                return MarketRegime.UNDEFINED

            # Generate regime signal
            current_indicators = self.indicator_history[-1]
            previous_indicators = self.indicator_history[-2]

            # Check for calendar events
            earnings_risk = self._check_earnings_risk(market_data)
            macro_risk = self._check_macro_risk(market_data)

            # Generate signal (includes regime detection)
            signal = self.signal_generator.generate_signal(
                current_indicators=current_indicators,
                previous_indicators=previous_indicators,
                earnings_risk=earnings_risk,
                macro_risk=macro_risk,
            )

            # Update regime state
            new_regime = signal.regime
            self.regime_confidence = signal.confidence

            # Check for regime change
            if new_regime != self.current_regime:
                self.logger.info(
                    f"Market regime changed: {self.current_regime.value} - >  {new_regime.value}"
                )
                self.last_regime_change = datetime.now()
                self.current_regime = new_regime

            return new_regime

        except Exception as e:
            self.logger.error(f"Error detecting market regime: {e}")
            return MarketRegime.UNDEFINED

    async def generate_strategy_adaptation(
        self, market_data: dict[str, Any], current_positions: dict[str, Any] | None = None
    ) -> StrategyAdaptation:
        """Generate strategy adaptation based on current market regime.

        Args:
            market_data: Current market data
            current_positions: Current portfolio positions

        Returns:
            StrategyAdaptation with recommendations
        """
        try:
            # Detect current regime
            regime = await self.detect_current_regime(market_data)

            # Check if we should adapt (confidence and cooldown)
            if not self._should_adapt(regime):
                # Return current adaptation if no change needed
                if self.adaptation_history:
                    return self.adaptation_history[-1]
                else:
                    # Create default adaptation
                    return self._create_default_adaptation(regime)

            # Calculate position size adjustments
            position_multiplier = self._get_position_multiplier(regime)
            max_risk = self._get_max_risk(regime)

            # Determine strategy selection
            recommended_strategies = self.config.enabled_strategies_by_regime.get(regime.value, [])
            all_strategies = set()
            for strategies in self.config.enabled_strategies_by_regime.values():
                all_strategies.update(strategies)
            disabled_strategies = list(all_strategies - set(recommended_strategies))

            # Calculate parameter adjustments
            parameter_adjustments = self._calculate_parameter_adjustments(regime, market_data)

            # Risk management adjustments
            stop_loss_adj, take_profit_adj = self._calculate_risk_adjustments(regime)

            # Timing adjustments
            entry_delay, exit_urgency = self._calculate_timing_adjustments(regime, market_data)

            # Create adaptation
            adaptation = StrategyAdaptation(
                regime=regime,
                confidence=self.regime_confidence,
                position_size_multiplier=position_multiplier,
                max_risk_per_trade=max_risk,
                recommended_strategies=recommended_strategies,
                disabled_strategies=disabled_strategies,
                parameter_adjustments=parameter_adjustments,
                stop_loss_adjustment=stop_loss_adj,
                take_profit_adjustment=take_profit_adj,
                entry_delay=entry_delay,
                exit_urgency=exit_urgency,
                timestamp=datetime.now(),
                reason=self._generate_adaptation_reason(regime, market_data),
                next_review=datetime.now() + timedelta(hours=1),
            )

            # Store in history
            self.adaptation_history.append(adaptation)
            if len(self.adaptation_history) > 50:  # Keep last 50 adaptations
                self.adaptation_history = self.adaptation_history[-50:]

            self.logger.info(f"Generated strategy adaptation for {regime.value} regime")
            return adaptation

        except Exception as e:
            self.logger.error(f"Error generating strategy adaptation: {e}")
            return self._create_default_adaptation(MarketRegime.UNDEFINED)

    def _extract_indicators(self, market_data: dict[str, Any]) -> TechnicalIndicators | None:
        """Extract technical indicators from market data."""
        try:
            # Handle different market data formats
            if "SPY" in market_data or "spy" in market_data:
                # Use SPY as primary indicator
                spy_data = market_data.get("SPY") or market_data.get("spy")

                if isinstance(spy_data, dict):
                    price = spy_data.get("price", spy_data.get("close", 0))
                    volume = spy_data.get("volume", 1000000)
                    high = spy_data.get("high", price * 1.01)
                    low = spy_data.get("low", price * 0.99)

                    # Use provided indicators or calculate defaults
                    ema_20 = spy_data.get("ema_20", price * 0.98)
                    ema_50 = spy_data.get("ema_50", price * 0.95)
                    ema_200 = spy_data.get("ema_200", price * 0.90)
                    rsi = spy_data.get("rsi", 50.0)

                    return create_sample_indicators(
                        price=float(price),
                        ema_20=float(ema_20),
                        ema_50=float(ema_50),
                        ema_200=float(ema_200),
                        rsi=float(rsi),
                        volume=int(volume),
                        high=float(high),
                        low=float(low),
                    )

            # Fallback: create indicators from any available price data
            for _symbol, data in market_data.items():
                if isinstance(data, dict) and "price" in data:
                    price = float(data["price"])
                    return create_sample_indicators(
                        price=price,
                        ema_20=price * 0.98,
                        ema_50=price * 0.95,
                        ema_200=price * 0.90,
                        rsi=50.0,
                    )

            return None

        except Exception as e:
            self.logger.error(f"Error extracting indicators: {e}")
            return None

    def _check_earnings_risk(self, market_data: dict[str, Any]) -> bool:
        """Check if we're in earnings season or near earnings."""
        try:
            # Check for earnings indicators in market data
            if "earnings_risk" in market_data:
                return bool(market_data["earnings_risk"])

            # Check for upcoming earnings
            if "upcoming_earnings" in market_data:
                upcoming = market_data["upcoming_earnings"]
                if isinstance(upcoming, list) and len(upcoming) > 0:
                    return True

            # Default: assume earnings risk during typical earnings weeks
            now = datetime.now()
            # Simplified: first and last weeks of quarter months
            if now.month in [1, 4, 7, 10]:  # Quarter end months
                if now.day <= 7 or now.day >= 24:  # First or last week
                    return True

            return False

        except Exception as e:
            self.logger.error(f"Error checking earnings risk: {e}")
            return False

    def _check_macro_risk(self, market_data: dict[str, Any]) -> bool:
        """Check for macro economic events."""
        try:
            # Check for macro risk indicators
            if "macro_risk" in market_data:
                return bool(market_data["macro_risk"])

            # Check for FOMC meetings, CPI releases, etc.
            if "economic_events" in market_data:
                events = market_data["economic_events"]
                if isinstance(events, list) and len(events) > 0:
                    return True

            # Default: assume macro risk around FOMC schedule
            now = datetime.now()
            # FOMC meetings typically every 6 - 8 weeks
            # Simplified: assume risk in certain weeks
            if now.isocalendar()[1] % 6 in [0, 1]:  # Every 6th week +/- 1
                return True

            return False

        except Exception as e:
            self.logger.error(f"Error checking macro risk: {e}")
            return False

    def _should_adapt(self, regime: MarketRegime) -> bool:
        """Check if we should adapt strategies."""
        # Check confidence threshold
        if self.regime_confidence < self.config.min_regime_confidence:
            return False

        # Check cooldown period
        if self.last_regime_change:
            time_since_change = datetime.now() - self.last_regime_change
            if time_since_change < timedelta(hours=self.config.regime_change_cooldown_hours):
                return False

        return True

    def _get_position_multiplier(self, regime: MarketRegime) -> float:
        """Get position size multiplier for regime."""
        multipliers = {
            MarketRegime.BULL: self.config.bull_position_multiplier,
            MarketRegime.BEAR: self.config.bear_position_multiplier,
            MarketRegime.SIDEWAYS: self.config.sideways_position_multiplier,
            MarketRegime.UNDEFINED: 0.5,  # Conservative default
        }
        return multipliers.get(regime, 0.5)

    def _get_max_risk(self, regime: MarketRegime) -> float:
        """Get max risk per trade for regime."""
        risks = {
            MarketRegime.BULL: self.config.bull_max_risk,
            MarketRegime.BEAR: self.config.bear_max_risk,
            MarketRegime.SIDEWAYS: self.config.sideways_max_risk,
            MarketRegime.UNDEFINED: 0.01,  # Very conservative
        }
        return risks.get(regime, 0.01)

    def _calculate_parameter_adjustments(
        self, regime: MarketRegime, market_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Calculate strategy parameter adjustments."""
        adjustments = {}

        if regime == MarketRegime.BULL:
            adjustments.update(
                {
                    "profit_target_multiplier": 1.3,  # Higher profit targets
                    "dte_preference": 30,  # Prefer ~30 DTE
                    "delta_preference": 0.30,  # Prefer ~30 delta
                    "iv_rank_min": 20,  # Lower IV rank threshold
                    "momentum_threshold": 0.02,  # 2% momentum threshold
                }
            )
        elif regime == MarketRegime.BEAR:
            adjustments.update(
                {
                    "profit_target_multiplier": 0.8,  # Lower profit targets
                    "dte_preference": 15,  # Shorter duration
                    "delta_preference": 0.15,  # Lower delta
                    "iv_rank_min": 40,  # Higher IV rank needed
                    "credit_spread_width": 10,  # Wider spreads
                }
            )
        elif regime == MarketRegime.SIDEWAYS:
            adjustments.update(
                {
                    "profit_target_multiplier": 1.0,  # Standard targets
                    "dte_preference": 21,  # Standard duration
                    "delta_preference": 0.20,  # Moderate delta
                    "iv_rank_min": 30,  # Standard IV requirements
                    "theta_preference": "high",  # Prefer high theta strategies
                }
            )

        return adjustments

    def _calculate_risk_adjustments(self, regime: MarketRegime) -> tuple[float, float]:
        """Calculate stop loss and take profit adjustments."""
        if regime == MarketRegime.BULL:
            return 1.2, 0.8  # Wider stops, tighter profits
        elif regime == MarketRegime.BEAR:
            return 0.7, 1.5  # Tighter stops, wider profits
        elif regime == MarketRegime.SIDEWAYS:
            return 1.0, 1.0  # Standard adjustments
        else:
            return 0.8, 1.2  # Conservative: tight stops, wider profits

    def _calculate_timing_adjustments(
        self, regime: MarketRegime, market_data: dict[str, Any]
    ) -> tuple[int, float]:
        """Calculate entry delay and exit urgency."""
        # Check volatility
        volatility = market_data.get("volatility", 0.2)

        if regime == MarketRegime.BULL:
            entry_delay = 0  # No delay in bull market
            exit_urgency = 0.8  # Less urgent exits
        elif regime == MarketRegime.BEAR:
            entry_delay = 15  # 15 minute delay to confirm
            exit_urgency = 1.5  # More urgent exits
        elif regime == MarketRegime.SIDEWAYS:
            entry_delay = 5  # Small delay
            exit_urgency = 1.0  # Standard urgency
        else:
            entry_delay = 30  # Long delay for undefined
            exit_urgency = 1.3  # Moderately urgent

        # Adjust for high volatility
        if volatility > 0.3:
            entry_delay += 10
            exit_urgency *= 1.2

        return entry_delay, exit_urgency

    def _generate_adaptation_reason(self, regime: MarketRegime, market_data: dict[str, Any]) -> str:
        """Generate human - readable reason for adaptation."""
        reasons = []

        reasons.append(f"Market regime: {regime.value}")
        reasons.append(f"Confidence: {self.regime_confidence:.1%}")

        # Add specific market conditions
        if market_data.get("volatility", 0) > 0.3:
            reasons.append("High volatility detected")

        if self._check_earnings_risk(market_data):
            reasons.append("Earnings season risk")

        if self._check_macro_risk(market_data):
            reasons.append("Macro event risk")

        return "; ".join(reasons)

    def _create_default_adaptation(self, regime: MarketRegime) -> StrategyAdaptation:
        """Create default adaptation."""
        return StrategyAdaptation(
            regime=regime,
            confidence=0.5,
            position_size_multiplier=0.5,
            max_risk_per_trade=0.01,
            recommended_strategies=["index_baseline"],
            disabled_strategies=[],
            parameter_adjustments={},
            stop_loss_adjustment=1.0,
            take_profit_adjustment=1.0,
            entry_delay=30,
            exit_urgency=1.0,
            timestamp=datetime.now(),
            reason="Default conservative adaptation",
            next_review=datetime.now() + timedelta(hours=1),
        )

    def get_adaptation_summary(self) -> dict[str, Any]:
        """Get current adaptation summary."""
        if not self.adaptation_history:
            return {"status": "no_adaptations", "current_regime": "undefined", "confidence": 0.0}

        current = self.adaptation_history[-1]
        return {
            "status": "active",
            "current_regime": current.regime.value,
            "confidence": current.confidence,
            "position_multiplier": current.position_size_multiplier,
            "max_risk": current.max_risk_per_trade,
            "active_strategies": len(current.recommended_strategies),
            "disabled_strategies": len(current.disabled_strategies),
            "last_update": current.timestamp.isoformat(),
            "next_review": current.next_review.isoformat(),
            "reason": current.reason,
        }


# Convenience function
async def adapt_strategies_to_market(
    market_data: dict[str, Any],
    current_positions: dict[str, Any] | None = None,
    config: RegimeAdaptationConfig | None = None,
) -> StrategyAdaptation:
    """Quick strategy adaptation based on market regime.

    Args:
        market_data: Current market data
        current_positions: Current portfolio positions
        config: Adaptation configuration

    Returns:
        StrategyAdaptation recommendations
    """
    adapter = MarketRegimeAdapter(config)
    return await adapter.generate_strategy_adaptation(market_data, current_positions)
