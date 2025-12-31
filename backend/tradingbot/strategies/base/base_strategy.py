"""Base strategy class and interfaces.

This module defines the core interfaces that all trading strategies
must implement, providing a consistent API across the system.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
import pandas as pd


class StrategyStatus(Enum):
    """Strategy execution status."""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"
    STOPPED = "stopped"


class SignalType(Enum):
    """Trading signal types."""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    CLOSE = "close"


@dataclass
class StrategyConfig:
    """Base configuration for all strategies."""
    name: str
    enabled: bool = True
    max_position_size: float = 10000.0
    max_total_risk: float = 50000.0
    stop_loss_pct: float = 0.05
    take_profit_multiplier: float = 2.0
    risk_free_rate: float = 0.02
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StrategyResult:
    """Result of strategy execution."""
    symbol: str
    signal: SignalType
    confidence: float
    price: float
    quantity: int
    timestamp: datetime
    reasoning: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseStrategy(ABC):
    """Abstract base class for all trading strategies.
    
    All strategies must implement the core methods defined here to ensure
    consistency and interoperability across the trading system.
    """
    
    def __init__(self, config: StrategyConfig):
        """Initialize strategy with configuration."""
        self.config = config
        self.status = StrategyStatus.IDLE
        self.last_update = datetime.now()
        self.performance_metrics = {}
        
    @abstractmethod
    def analyze(self, data: pd.DataFrame, symbol: str) -> Optional[StrategyResult]:
        """Analyze market data and generate trading signal.
        
        Args:
            data: Market data for the symbol
            symbol: Stock symbol to analyze
            
        Returns:
            StrategyResult if signal generated, None otherwise
        """
        pass
    
    @abstractmethod
    def get_required_data(self) -> List[str]:
        """Get list of required data fields for this strategy.
        
        Returns:
            List of required data column names
        """
        pass
    
    @abstractmethod
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate that data meets strategy requirements.
        
        Args:
            data: Market data to validate
            
        Returns:
            True if data is valid, False otherwise
        """
        pass
    
    def start(self) -> None:
        """Start strategy execution."""
        self.status = StrategyStatus.RUNNING
        self.last_update = datetime.now()
        
    def stop(self) -> None:
        """Stop strategy execution."""
        self.status = StrategyStatus.STOPPED
        self.last_update = datetime.now()
        
    def pause(self) -> None:
        """Pause strategy execution."""
        self.status = StrategyStatus.PAUSED
        self.last_update = datetime.now()
        
    def resume(self) -> None:
        """Resume strategy execution."""
        if self.status == StrategyStatus.PAUSED:
            self.status = StrategyStatus.RUNNING
            self.last_update = datetime.now()
    
    def get_status(self) -> Dict[str, Any]:
        """Get current strategy status.
        
        Returns:
            Dictionary containing status information
        """
        return {
            "name": self.config.name,
            "status": self.status.value,
            "last_update": self.last_update.isoformat(),
            "enabled": self.config.enabled,
            "performance_metrics": self.performance_metrics,
        }
    
    def update_config(self, new_config: StrategyConfig) -> None:
        """Update strategy configuration.
        
        Args:
            new_config: New configuration to apply
        """
        self.config = new_config
        self.last_update = datetime.now()
    
    def calculate_position_size(self, price: float, risk_amount: float) -> int:
        """Calculate position size based on risk management rules.

        Args:
            price: Current price of the asset
            risk_amount: Maximum amount to risk

        Returns:
            Number of shares to trade
        """
        if price <= 0:
            return 0

        max_shares = int(self.config.max_position_size / price)
        risk_shares = int(risk_amount / price)

        return min(max_shares, risk_shares)


@dataclass
class SignalSnapshot:
    """Captured signal state at trade time."""
    signals: Dict[str, Dict[str, Any]]
    confidence_score: int
    signals_triggered: int
    signals_checked: int
    explanation: str


class SignalCaptureMixin:
    """Mixin to add signal capture functionality to strategies.

    This mixin provides methods to capture and store signal state
    at the moment of trade execution for transparency and analysis.

    Usage:
        class MyStrategy(BaseStrategy, SignalCaptureMixin):
            def analyze(self, data, symbol):
                # Calculate indicators
                rsi = self.calculate_rsi(data)
                macd = self.calculate_macd(data)

                # Capture signals before trade
                self.capture_signal('rsi', rsi, threshold=30, triggered=rsi<30)
                self.capture_signal('macd', macd, crossover=True, triggered=True)

                # If trade triggered, create snapshot
                if should_trade:
                    snapshot = self.create_signal_snapshot()
                    self.store_signal_snapshot(trade_id, symbol, 'buy', price, qty, snapshot)
    """

    def __init__(self, *args, **kwargs):
        """Initialize signal capture state."""
        super().__init__(*args, **kwargs)
        self._captured_signals: Dict[str, Dict[str, Any]] = {}
        self._signals_triggered: int = 0
        self._signals_checked: int = 0

    def capture_signal(
        self,
        signal_name: str,
        value: Any,
        threshold: Optional[float] = None,
        triggered: bool = False,
        **extra_data
    ) -> None:
        """Capture a signal's state for the current analysis.

        Args:
            signal_name: Name of the signal (e.g., 'rsi', 'macd', 'volume')
            value: Current value of the signal
            threshold: Threshold value that would trigger the signal
            triggered: Whether this signal is triggered/active
            **extra_data: Additional signal-specific data
        """
        self._signals_checked += 1
        if triggered:
            self._signals_triggered += 1

        signal_data = {
            'value': value,
            'triggered': triggered,
        }
        if threshold is not None:
            signal_data['threshold'] = threshold
        signal_data.update(extra_data)

        self._captured_signals[signal_name] = signal_data

    def capture_rsi(
        self,
        value: float,
        threshold: float = 30,
        period: int = 14
    ) -> bool:
        """Capture RSI signal state.

        Args:
            value: Current RSI value
            threshold: Oversold threshold (default 30)
            period: RSI period used

        Returns:
            True if RSI indicates oversold condition
        """
        triggered = value < threshold
        self.capture_signal(
            'rsi',
            value=round(value, 2),
            threshold=threshold,
            triggered=triggered,
            period=period
        )
        return triggered

    def capture_macd(
        self,
        macd_value: float,
        signal_value: float,
        histogram: float,
        crossover: bool = False
    ) -> bool:
        """Capture MACD signal state.

        Args:
            macd_value: MACD line value
            signal_value: Signal line value
            histogram: MACD histogram value
            crossover: Whether a crossover occurred

        Returns:
            True if bullish signal
        """
        triggered = crossover or histogram > 0
        self.capture_signal(
            'macd',
            value=round(macd_value, 4),
            signal=round(signal_value, 4),
            histogram=round(histogram, 4),
            crossover=crossover,
            triggered=triggered
        )
        return triggered

    def capture_volume(
        self,
        current_volume: float,
        average_volume: float,
        ratio_threshold: float = 2.0
    ) -> bool:
        """Capture volume signal state.

        Args:
            current_volume: Current volume
            average_volume: Average volume (e.g., 20-day avg)
            ratio_threshold: Volume ratio to trigger signal

        Returns:
            True if volume is elevated
        """
        ratio = current_volume / average_volume if average_volume > 0 else 1.0
        triggered = ratio >= ratio_threshold
        self.capture_signal(
            'volume',
            value=round(ratio, 2),
            current=int(current_volume),
            average=int(average_volume),
            ratio=round(ratio, 2),
            triggered=triggered
        )
        return triggered

    def capture_price_action(
        self,
        change_pct: float,
        from_sma20: float,
        from_sma50: Optional[float] = None,
        threshold: float = -3.0
    ) -> bool:
        """Capture price action signal state.

        Args:
            change_pct: Percent change from previous close
            from_sma20: Percent distance from 20-day SMA
            from_sma50: Percent distance from 50-day SMA
            threshold: Dip threshold to trigger signal

        Returns:
            True if price dipped significantly
        """
        triggered = change_pct <= threshold
        signal_data = {
            'change_pct': round(change_pct, 2),
            'from_sma20': round(from_sma20, 2),
        }
        if from_sma50 is not None:
            signal_data['from_sma50'] = round(from_sma50, 2)

        self.capture_signal(
            'price_action',
            value=round(change_pct, 2),
            threshold=threshold,
            triggered=triggered,
            **signal_data
        )
        return triggered

    def capture_bollinger(
        self,
        price: float,
        upper: float,
        lower: float,
        middle: float
    ) -> str:
        """Capture Bollinger Bands signal state.

        Args:
            price: Current price
            upper: Upper Bollinger Band
            lower: Lower Bollinger Band
            middle: Middle Bollinger Band (SMA)

        Returns:
            Position relative to bands: 'below_lower', 'above_upper', 'middle'
        """
        if price < lower:
            position = 'below_lower'
            triggered = True
        elif price > upper:
            position = 'above_upper'
            triggered = True
        else:
            position = 'middle'
            triggered = False

        self.capture_signal(
            'bollinger',
            value=round(price, 2),
            upper=round(upper, 2),
            lower=round(lower, 2),
            middle=round(middle, 2),
            position=position,
            triggered=triggered
        )
        return position

    def capture_stochastic(
        self,
        k: float,
        d: float,
        oversold_threshold: float = 20,
        overbought_threshold: float = 80
    ) -> bool:
        """Capture Stochastic signal state.

        Args:
            k: %K value
            d: %D value
            oversold_threshold: Oversold threshold
            overbought_threshold: Overbought threshold

        Returns:
            True if oversold
        """
        oversold = k < oversold_threshold
        overbought = k > overbought_threshold

        self.capture_signal(
            'stochastic',
            value=round(k, 2),
            k=round(k, 2),
            d=round(d, 2),
            oversold=oversold,
            overbought=overbought,
            triggered=oversold or overbought
        )
        return oversold

    def capture_trend(
        self,
        short_term: str,
        medium_term: str,
        long_term: str
    ) -> bool:
        """Capture trend analysis signal state.

        Args:
            short_term: Short-term trend ('bullish', 'bearish', 'neutral')
            medium_term: Medium-term trend
            long_term: Long-term trend

        Returns:
            True if trends are aligned
        """
        aligned = (short_term == medium_term == long_term)

        self.capture_signal(
            'trend',
            value=1 if aligned else 0,
            short_term=short_term,
            medium_term=medium_term,
            long_term=long_term,
            aligned=aligned,
            triggered=aligned
        )
        return aligned

    def calculate_confidence_score(self) -> int:
        """Calculate confidence score based on captured signals.

        Returns:
            Confidence score from 0-100
        """
        if self._signals_checked == 0:
            return 50

        # Base confidence from triggered ratio
        trigger_ratio = self._signals_triggered / self._signals_checked
        base_score = int(trigger_ratio * 60)  # Max 60 from triggers

        # Bonus for specific signal combinations
        bonus = 0
        signals = self._captured_signals

        # RSI + Volume combo
        if 'rsi' in signals and 'volume' in signals:
            if signals['rsi'].get('triggered') and signals['volume'].get('triggered'):
                bonus += 15

        # MACD crossover bonus
        if 'macd' in signals and signals['macd'].get('crossover'):
            bonus += 10

        # Trend alignment bonus
        if 'trend' in signals and signals['trend'].get('aligned'):
            bonus += 15

        return min(100, base_score + bonus)

    def generate_explanation(self, direction: str) -> str:
        """Generate a plain English explanation of the trade.

        Args:
            direction: Trade direction ('buy' or 'sell')

        Returns:
            Human-readable explanation string
        """
        parts = [f"This {direction} was triggered because:"]

        signals = self._captured_signals

        if 'rsi' in signals and signals['rsi'].get('triggered'):
            rsi_val = signals['rsi'].get('value', 0)
            parts.append(f"RSI dropped to {rsi_val:.0f} (oversold)")

        if 'macd' in signals and signals['macd'].get('crossover'):
            parts.append("MACD showed bullish crossover")

        if 'volume' in signals and signals['volume'].get('triggered'):
            ratio = signals['volume'].get('ratio', 1)
            parts.append(f"volume was {ratio:.1f}x average")

        if 'price_action' in signals and signals['price_action'].get('triggered'):
            change = signals['price_action'].get('change_pct', 0)
            parts.append(f"price dropped {abs(change):.1f}%")

        if 'bollinger' in signals and signals['bollinger'].get('position') == 'below_lower':
            parts.append("price broke below Bollinger lower band")

        if 'stochastic' in signals and signals['stochastic'].get('oversold'):
            k = signals['stochastic'].get('k', 0)
            parts.append(f"Stochastic %K at {k:.0f} (oversold)")

        if len(parts) == 1:
            parts.append("multiple technical signals aligned")

        return " ".join(parts[:1]) + " " + ", ".join(parts[1:]) + "."

    def create_signal_snapshot(self) -> SignalSnapshot:
        """Create a snapshot of all captured signals.

        Returns:
            SignalSnapshot with all current signal data
        """
        confidence = self.calculate_confidence_score()
        explanation = self.generate_explanation('trade')

        return SignalSnapshot(
            signals=self._captured_signals.copy(),
            confidence_score=confidence,
            signals_triggered=self._signals_triggered,
            signals_checked=self._signals_checked,
            explanation=explanation,
        )

    def store_signal_snapshot(
        self,
        trade_id: str,
        symbol: str,
        direction: str,
        entry_price: float,
        quantity: float,
        snapshot: Optional[SignalSnapshot] = None
    ) -> Optional[Any]:
        """Store signal snapshot to database.

        Args:
            trade_id: Unique trade identifier
            symbol: Trading symbol
            direction: Trade direction
            entry_price: Entry price
            quantity: Trade quantity
            snapshot: Pre-created snapshot, or None to create one

        Returns:
            TradeSignalSnapshot model instance, or None on error
        """
        if snapshot is None:
            snapshot = self.create_signal_snapshot()

        try:
            from backend.tradingbot.models.models import TradeSignalSnapshot

            # Find similar historical trades
            similar_trades = self._find_similar_historical_trades(snapshot)

            # Generate explanation specific to direction
            explanation = self.generate_explanation(direction)

            trade_snapshot = TradeSignalSnapshot.objects.create(
                trade_id=trade_id,
                strategy_name=getattr(self, 'config', {}).name if hasattr(self, 'config') else 'unknown',
                symbol=symbol,
                direction=direction,
                entry_price=entry_price,
                quantity=quantity,
                signals_at_entry=snapshot.signals,
                confidence_score=snapshot.confidence_score,
                signals_triggered=snapshot.signals_triggered,
                signals_checked=snapshot.signals_checked,
                explanation=explanation,
                similar_historical_trades=similar_trades,
            )

            return trade_snapshot

        except Exception as e:
            import logging
            logging.getLogger(__name__).error(f"Failed to store signal snapshot: {e}")
            return None

    def _find_similar_historical_trades(self, snapshot: SignalSnapshot, limit: int = 5) -> list:
        """Find similar historical trades for comparison.

        Args:
            snapshot: Current signal snapshot
            limit: Maximum similar trades to find

        Returns:
            List of similar trade dictionaries
        """
        try:
            from backend.tradingbot.models.models import TradeSignalSnapshot

            # Get recent historical trades with outcomes
            historical = TradeSignalSnapshot.objects.filter(
                outcome__isnull=False
            ).order_by('-created_at')[:200]

            similar = []
            for hist in historical:
                similarity = self._calculate_signal_similarity(snapshot.signals, hist.signals_at_entry)
                if similarity >= 0.7:
                    similar.append({
                        'trade_id': hist.trade_id,
                        'similarity': round(similarity, 2),
                        'outcome': hist.outcome,
                        'pnl_pct': float(hist.pnl_percent) if hist.pnl_percent else 0,
                    })

            similar.sort(key=lambda x: x['similarity'], reverse=True)
            return similar[:limit]

        except Exception:
            return []

    def _calculate_signal_similarity(self, signals1: dict, signals2: dict) -> float:
        """Calculate similarity between two signal snapshots."""
        if not signals1 or not signals2:
            return 0.0

        score = 0.0
        comparisons = 0

        for signal_name in signals1:
            if signal_name in signals2:
                s1 = signals1[signal_name]
                s2 = signals2[signal_name]
                if isinstance(s1, dict) and isinstance(s2, dict):
                    # Compare triggered state
                    if s1.get('triggered') == s2.get('triggered'):
                        score += 0.5
                    # Compare values if numeric
                    v1 = s1.get('value', 0)
                    v2 = s2.get('value', 0)
                    if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
                        max_val = max(abs(v1), abs(v2), 1)
                        diff_ratio = abs(v1 - v2) / max_val
                        score += 0.5 * (1 - min(diff_ratio, 1))
                    comparisons += 1

        return score / comparisons if comparisons > 0 else 0.0

    def reset_captured_signals(self) -> None:
        """Reset captured signals for next analysis cycle."""
        self._captured_signals = {}
        self._signals_triggered = 0
        self._signals_checked = 0

    # -------------------------------------------------------------------------
    # Reasoning Capture Methods
    # -------------------------------------------------------------------------

    def capture_entry_reasoning(
        self,
        symbol: str,
        custom_summary: str = None,
    ) -> Dict[str, Any]:
        """Capture structured entry reasoning from current signals.

        Call this before executing entry trade to generate structured
        reasoning that explains "why" the trade was triggered.

        Args:
            symbol: Trading symbol
            custom_summary: Optional custom summary (auto-generated if None)

        Returns:
            Entry reasoning dict with summary, signals, confidence, market_context
        """
        from django.utils import timezone

        # Generate summary from triggered signals
        if custom_summary is None:
            custom_summary = self._generate_entry_summary()

        # Format signals with 'met' flag for triggered
        formatted_signals = {}
        for name, data in self._captured_signals.items():
            sig_copy = dict(data) if isinstance(data, dict) else {'value': data}
            # Normalize triggered -> met
            if 'triggered' in sig_copy:
                sig_copy['met'] = sig_copy['triggered']
            formatted_signals[name] = sig_copy

        # Get market context
        market_context = self._get_market_context()

        confidence = self.calculate_confidence_score()

        return {
            'summary': custom_summary,
            'signals': formatted_signals,
            'confidence': confidence,
            'market_context': market_context,
            'timestamp': timezone.now().isoformat(),
        }

    def capture_exit_reasoning(
        self,
        trigger: str,
        entry_timestamp=None,
        custom_summary: str = None,
    ) -> Dict[str, Any]:
        """Capture structured exit reasoning.

        Call this before executing exit trade to document why
        the position was closed.

        Args:
            trigger: Exit trigger type (take_profit, stop_loss, trailing_stop, signal, manual)
            entry_timestamp: When position was opened (for duration calculation)
            custom_summary: Optional custom summary

        Returns:
            Exit reasoning dict with summary, trigger, duration, signals
        """
        from django.utils import timezone

        # Calculate held duration
        if entry_timestamp:
            held_delta = timezone.now() - entry_timestamp
            held_hours = held_delta.total_seconds() / 3600
            held_duration = self._format_held_duration(held_hours)
        else:
            held_hours = 0
            held_duration = "unknown"

        # Generate summary
        if custom_summary is None:
            trigger_display = {
                'take_profit': 'Profit Target Reached',
                'stop_loss': 'Stop Loss Triggered',
                'trailing_stop': 'Trailing Stop Triggered',
                'signal': 'Exit Signal Triggered',
                'time_based': 'Time-Based Exit',
                'manual': 'Manual Exit',
                'expiration': 'Option Expiration',
            }.get(trigger, trigger.replace('_', ' ').title())
            custom_summary = trigger_display

        # Format current signals at exit
        signals_at_exit = {}
        for name, data in self._captured_signals.items():
            sig_copy = dict(data) if isinstance(data, dict) else {'value': data}
            signals_at_exit[name] = sig_copy

        return {
            'summary': custom_summary,
            'trigger': trigger,
            'held_duration': held_duration,
            'held_duration_hours': round(held_hours, 2),
            'signals_at_exit': signals_at_exit,
            'timestamp': timezone.now().isoformat(),
        }

    def _generate_entry_summary(self) -> str:
        """Generate human-readable entry summary from captured signals."""
        triggered = []
        for name, data in self._captured_signals.items():
            if isinstance(data, dict) and (data.get('triggered') or data.get('met')):
                triggered.append(name.upper().replace('_', ' '))

        if not triggered:
            return "Multiple technical factors aligned"

        if len(triggered) == 1:
            return f"{triggered[0]} signal"

        if len(triggered) <= 3:
            return " + ".join(triggered)

        return f"{len(triggered)} signals aligned"

    def _get_market_context(self) -> Dict[str, Any]:
        """Get current market context for reasoning.

        Returns:
            Dict with VIX, SPY trend, and other market context
        """
        context = {}

        try:
            # Try to get VIX from market monitor
            from backend.auth0login.services.market_monitor import get_market_monitor
            monitor = get_market_monitor()

            if hasattr(monitor, 'get_current_vix'):
                vix = monitor.get_current_vix()
                if vix:
                    context['vix'] = vix
                    if vix >= 30:
                        context['vix_level'] = 'high'
                    elif vix >= 20:
                        context['vix_level'] = 'elevated'
                    else:
                        context['vix_level'] = 'normal'

            if hasattr(monitor, 'get_market_overview'):
                overview = monitor.get_market_overview()
                if overview:
                    context['spy_trend'] = overview.get('spy_trend', 'unknown')
                    context['market_sentiment'] = overview.get('sentiment', 'neutral')

        except ImportError:
            pass
        except Exception:
            pass

        return context

    def _format_held_duration(self, hours: float) -> str:
        """Format held duration in hours to human-readable string."""
        if hours < 1:
            return f"{int(hours * 60)} minutes"
        elif hours < 24:
            return f"{hours:.1f} hours"
        elif hours < 48:
            return "1 day"
        else:
            days = int(hours / 24)
            return f"{days} days"

    def store_trade_with_reasoning(
        self,
        trade_id: str,
        symbol: str,
        direction: str,
        entry_price: float,
        quantity: float,
        order=None,
    ) -> Optional[Any]:
        """Store trade snapshot with full reasoning.

        Enhanced version of store_signal_snapshot that includes
        structured entry reasoning.

        Args:
            trade_id: Unique trade identifier
            symbol: Trading symbol
            direction: Trade direction
            entry_price: Entry price
            quantity: Trade quantity
            order: Optional Order model instance

        Returns:
            TradeSignalSnapshot model instance, or None on error
        """
        try:
            from backend.tradingbot.models.models import TradeSignalSnapshot
            from decimal import Decimal

            # Create entry reasoning
            entry_reasoning = self.capture_entry_reasoning(symbol)
            confidence = entry_reasoning['confidence']

            # Find similar historical trades
            snapshot = self.create_signal_snapshot()
            similar_trades = self._find_similar_historical_trades(snapshot)

            # Generate explanation
            explanation = self.generate_explanation(direction)

            trade_snapshot = TradeSignalSnapshot.objects.create(
                trade_id=trade_id,
                order=order,
                strategy_name=getattr(self, 'config', {}).name if hasattr(self, 'config') else 'unknown',
                symbol=symbol,
                direction=direction,
                entry_price=Decimal(str(entry_price)),
                quantity=Decimal(str(quantity)),
                signals_at_entry=snapshot.signals,
                confidence_score=confidence,
                signals_triggered=snapshot.signals_triggered,
                signals_checked=snapshot.signals_checked,
                explanation=explanation,
                entry_reasoning=entry_reasoning,
                similar_historical_trades=similar_trades,
            )

            import logging
            logging.getLogger(__name__).info(
                f"Stored trade with reasoning: {trade_id} {symbol} {direction} "
                f"(confidence={confidence})"
            )

            return trade_snapshot

        except Exception as e:
            import logging
            logging.getLogger(__name__).error(f"Failed to store trade with reasoning: {e}")
            return None

    def record_trade_exit(
        self,
        trade_id: str,
        exit_price: float,
        trigger: str,
        pnl_amount: float = None,
        pnl_percent: float = None,
    ) -> Optional[Any]:
        """Record trade exit with reasoning.

        Args:
            trade_id: Trade identifier to update
            exit_price: Exit price
            trigger: Exit trigger type
            pnl_amount: P&L amount
            pnl_percent: P&L percentage

        Returns:
            Updated TradeSignalSnapshot or None
        """
        try:
            from backend.tradingbot.models.models import TradeSignalSnapshot

            snapshot = TradeSignalSnapshot.objects.get(trade_id=trade_id)

            # Generate exit reasoning
            exit_reasoning = self.capture_exit_reasoning(
                trigger=trigger,
                entry_timestamp=snapshot.created_at,
            )

            # Record exit with reasoning
            snapshot.record_exit(
                exit_price=exit_price,
                pnl_amount=pnl_amount,
                pnl_percent=pnl_percent,
                exit_reasoning=exit_reasoning,
            )

            return snapshot

        except Exception as e:
            import logging
            logging.getLogger(__name__).error(f"Failed to record trade exit: {e}")
            return None



