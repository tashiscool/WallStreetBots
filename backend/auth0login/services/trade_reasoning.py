"""Trade Reasoning Capture Service.

Provides structured reasoning capture for trades, enabling:
- Entry reasoning with signal summary, confidence, and market context
- Exit reasoning with trigger type and duration
- Post-trade outcome analysis with timing evaluation
- Similar trade pattern matching

Usage:
    from backend.auth0login.services.trade_reasoning import (
        TradeReasoningService, get_trade_reasoning_service
    )

    # Capture entry reasoning
    service = get_trade_reasoning_service()
    entry_reasoning = service.capture_entry_reasoning(
        signals={'rsi': {'value': 28, 'threshold': 30, 'met': True}},
        confidence=85,
        summary="RSI oversold signal"
    )

    # Create trade snapshot with reasoning
    snapshot = service.create_trade_with_reasoning(
        trade_id="order_123",
        symbol="AAPL",
        direction="buy",
        entry_price=150.00,
        quantity=100,
        strategy_name="wsb_dip_bot",
        signals=captured_signals,
        confidence=85
    )

    # Record exit with reasoning
    service.record_exit_with_reasoning(
        trade_id="order_123",
        exit_price=155.00,
        trigger="take_profit",
        pnl_percent=3.33
    )
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional

from django.db.models import Avg, Count, Q
from django.utils import timezone

logger = logging.getLogger("wsb.trade_reasoning")


# Exit trigger type constants
EXIT_TRIGGERS = {
    'take_profit': 'Take Profit',
    'stop_loss': 'Stop Loss',
    'trailing_stop': 'Trailing Stop',
    'signal': 'Signal Exit',
    'time_based': 'Time-Based Exit',
    'manual': 'Manual Exit',
    'expiration': 'Option Expiration',
    'assignment': 'Option Assignment',
}


class TradeReasoningService:
    """Service for capturing and managing trade reasoning.

    This service provides methods to:
    - Generate structured entry reasoning from signals
    - Generate exit reasoning with trigger analysis
    - Perform post-trade outcome analysis
    - Find similar historical trades
    """

    def __init__(self):
        """Initialize trade reasoning service."""
        self.logger = logging.getLogger(__name__)

    def _get_snapshot_model(self):
        """Lazy import TradeSignalSnapshot model."""
        from backend.tradingbot.models.models import TradeSignalSnapshot
        return TradeSignalSnapshot

    def capture_entry_reasoning(
        self,
        signals: Dict[str, Dict[str, Any]],
        confidence: int,
        summary: str = None,
        market_context: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """Capture structured entry reasoning.

        Args:
            signals: Dict of signal states {signal_name: {value, threshold, met, ...}}
            confidence: Confidence score 0-100
            summary: Optional custom summary (auto-generated if not provided)
            market_context: Optional market context (VIX, SPY trend, etc.)

        Returns:
            Structured entry reasoning dict
        """
        # Auto-generate summary if not provided
        if summary is None:
            summary = self._generate_entry_summary(signals)

        # Format signals for storage
        formatted_signals = self._format_signals_for_storage(signals)

        return {
            'summary': summary,
            'signals': formatted_signals,
            'confidence': confidence,
            'market_context': market_context or {},
            'timestamp': timezone.now().isoformat(),
        }

    def capture_exit_reasoning(
        self,
        trigger: str,
        entry_timestamp: datetime = None,
        signals_at_exit: Dict[str, Any] = None,
        summary: str = None,
    ) -> Dict[str, Any]:
        """Capture structured exit reasoning.

        Args:
            trigger: Exit trigger type (take_profit, stop_loss, etc.)
            entry_timestamp: When the position was opened
            signals_at_exit: Optional signal states at exit time
            summary: Optional custom summary

        Returns:
            Structured exit reasoning dict
        """
        # Calculate held duration
        if entry_timestamp:
            held_delta = timezone.now() - entry_timestamp
            held_hours = held_delta.total_seconds() / 3600
            held_duration = self._format_duration(held_hours)
        else:
            held_hours = 0
            held_duration = "unknown"

        # Auto-generate summary if not provided
        if summary is None:
            trigger_display = EXIT_TRIGGERS.get(trigger, trigger.replace('_', ' ').title())
            summary = f"{trigger_display} triggered"

        return {
            'summary': summary,
            'trigger': trigger,
            'held_duration': held_duration,
            'held_duration_hours': round(held_hours, 2),
            'signals_at_exit': signals_at_exit or {},
            'timestamp': timezone.now().isoformat(),
        }

    def create_trade_with_reasoning(
        self,
        trade_id: str,
        symbol: str,
        direction: str,
        entry_price: float,
        quantity: float,
        strategy_name: str,
        signals: Dict[str, Dict[str, Any]],
        confidence: int,
        market_context: Dict[str, Any] = None,
        explanation: str = None,
        order=None,
    ) -> Any:
        """Create a TradeSignalSnapshot with full reasoning.

        Args:
            trade_id: Unique trade identifier
            symbol: Trading symbol
            direction: Trade direction (buy, sell, etc.)
            entry_price: Entry price
            quantity: Trade quantity
            strategy_name: Strategy name
            signals: Signal states at entry
            confidence: Confidence score 0-100
            market_context: Optional market context
            explanation: Optional explanation text
            order: Optional Order model instance

        Returns:
            TradeSignalSnapshot instance
        """
        TradeSignalSnapshot = self._get_snapshot_model()

        # Generate entry reasoning
        entry_reasoning = self.capture_entry_reasoning(
            signals=signals,
            confidence=confidence,
            market_context=market_context,
        )

        # Count triggered signals
        triggered_count = sum(1 for s in signals.values() if s.get('met') or s.get('triggered'))

        # Generate explanation if not provided
        if explanation is None:
            explanation = self._generate_explanation(signals, confidence, strategy_name)

        # Find similar historical trades
        similar_trades = self._find_similar_trades(signals, strategy_name)

        snapshot = TradeSignalSnapshot.objects.create(
            trade_id=trade_id,
            order=order,
            symbol=symbol,
            direction=direction,
            entry_price=Decimal(str(entry_price)),
            quantity=Decimal(str(quantity)),
            strategy_name=strategy_name,
            signals_at_entry=signals,
            confidence_score=confidence,
            signals_triggered=triggered_count,
            signals_checked=len(signals),
            explanation=explanation,
            entry_reasoning=entry_reasoning,
            similar_historical_trades=similar_trades,
        )

        self.logger.info(
            f"Created trade snapshot {trade_id}: {symbol} {direction} "
            f"(confidence={confidence}, signals={triggered_count}/{len(signals)})"
        )

        return snapshot

    def record_exit_with_reasoning(
        self,
        trade_id: str,
        exit_price: float,
        trigger: str,
        pnl_amount: float = None,
        pnl_percent: float = None,
        signals_at_exit: Dict[str, Any] = None,
        summary: str = None,
    ) -> Optional[Any]:
        """Record trade exit with structured reasoning.

        Args:
            trade_id: Trade identifier to update
            exit_price: Exit price
            trigger: Exit trigger type
            pnl_amount: P&L amount
            pnl_percent: P&L percentage
            signals_at_exit: Optional signals at exit time
            summary: Optional custom summary

        Returns:
            Updated TradeSignalSnapshot or None if not found
        """
        TradeSignalSnapshot = self._get_snapshot_model()

        try:
            snapshot = TradeSignalSnapshot.objects.get(trade_id=trade_id)
        except TradeSignalSnapshot.DoesNotExist:
            self.logger.warning(f"Trade snapshot not found: {trade_id}")
            return None

        # Generate exit reasoning
        exit_reasoning = self.capture_exit_reasoning(
            trigger=trigger,
            entry_timestamp=snapshot.created_at,
            signals_at_exit=signals_at_exit,
            summary=summary,
        )

        # Record exit
        snapshot.record_exit(
            exit_price=exit_price,
            pnl_amount=pnl_amount,
            pnl_percent=pnl_percent,
            exit_reasoning=exit_reasoning,
        )

        self.logger.info(
            f"Recorded exit for {trade_id}: {trigger}, P&L={pnl_percent}%"
        )

        return snapshot

    def analyze_closed_trade(
        self,
        trade_id: str,
        historical_prices: Dict[str, float] = None,
    ) -> Optional[Dict[str, Any]]:
        """Perform post-trade outcome analysis.

        Args:
            trade_id: Trade identifier to analyze
            historical_prices: Optional dict with price data during hold period

        Returns:
            Outcome analysis dict or None if trade not found/still open
        """
        TradeSignalSnapshot = self._get_snapshot_model()

        try:
            snapshot = TradeSignalSnapshot.objects.get(trade_id=trade_id)
        except TradeSignalSnapshot.DoesNotExist:
            return None

        # Ensure trade is closed
        if not snapshot.exit_price or not snapshot.exit_timestamp:
            self.logger.warning(f"Trade {trade_id} is still open")
            return None

        analysis = {}

        # Basic P&L
        analysis['pnl'] = float(snapshot.pnl_amount) if snapshot.pnl_amount else 0
        analysis['pnl_pct'] = float(snapshot.pnl_percent) if snapshot.pnl_percent else 0

        # Calculate vs_hold (what if just bought and held)
        if historical_prices:
            hold_return = self._calculate_hold_return(
                symbol=snapshot.symbol,
                entry_time=snapshot.created_at,
                exit_time=snapshot.exit_timestamp,
                historical_prices=historical_prices,
            )
            analysis['vs_hold'] = analysis['pnl_pct'] - hold_return
        else:
            analysis['vs_hold'] = None

        # Timing analysis
        timing = self._analyze_timing(snapshot, historical_prices)
        analysis['timing_score'] = timing['score']
        analysis['entry_timing_analysis'] = timing.get('entry', {})
        analysis['exit_timing_analysis'] = timing.get('exit', {})

        # Find similar trades and calculate stats
        similar_stats = self._get_similar_trades_stats(snapshot)
        analysis['similar_trades_avg_pnl'] = similar_stats.get('avg_pnl', 0)
        analysis['similar_trades_win_rate'] = similar_stats.get('win_rate', 0)
        analysis['similar_trades_count'] = similar_stats.get('count', 0)

        # Generate notes
        analysis['notes'] = self._generate_analysis_notes(analysis, timing)

        # Save analysis to snapshot
        snapshot.set_outcome_analysis(analysis)

        self.logger.info(f"Analyzed closed trade {trade_id}: {analysis['timing_score']}")

        return analysis

    def get_trade_with_full_reasoning(self, trade_id: str) -> Optional[Dict[str, Any]]:
        """Get trade with all reasoning data.

        Args:
            trade_id: Trade identifier

        Returns:
            Complete trade data dict or None
        """
        TradeSignalSnapshot = self._get_snapshot_model()

        try:
            snapshot = TradeSignalSnapshot.objects.get(trade_id=trade_id)
            return snapshot.to_dict_with_reasoning()
        except TradeSignalSnapshot.DoesNotExist:
            return None

    def get_trades_by_strategy(
        self,
        strategy_name: str,
        include_open: bool = True,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Get trades for a strategy with reasoning.

        Args:
            strategy_name: Strategy name
            include_open: Include open trades
            limit: Maximum trades to return

        Returns:
            List of trade dicts with reasoning
        """
        TradeSignalSnapshot = self._get_snapshot_model()

        query = TradeSignalSnapshot.objects.filter(strategy_name=strategy_name)

        if not include_open:
            query = query.exclude(outcome='open').exclude(outcome__isnull=True)

        snapshots = query.order_by('-created_at')[:limit]

        return [s.to_dict_with_reasoning() for s in snapshots]

    def get_reasoning_stats(
        self,
        strategy_name: str = None,
        days: int = 30,
    ) -> Dict[str, Any]:
        """Get reasoning statistics.

        Args:
            strategy_name: Optional strategy filter
            days: Number of days to analyze

        Returns:
            Stats dict with confidence distribution, trigger breakdown, etc.
        """
        TradeSignalSnapshot = self._get_snapshot_model()

        cutoff = timezone.now() - timedelta(days=days)
        query = TradeSignalSnapshot.objects.filter(created_at__gte=cutoff)

        if strategy_name:
            query = query.filter(strategy_name=strategy_name)

        # Basic stats
        total = query.count()
        with_entry_reasoning = query.exclude(entry_reasoning__isnull=True).count()
        with_exit_reasoning = query.exclude(exit_reasoning__isnull=True).count()
        with_analysis = query.exclude(outcome_analysis__isnull=True).count()

        # Confidence distribution
        avg_confidence = query.aggregate(avg=Avg('confidence_score'))['avg'] or 0

        # Outcome by confidence level
        high_conf = query.filter(confidence_score__gte=70)
        mid_conf = query.filter(confidence_score__gte=40, confidence_score__lt=70)
        low_conf = query.filter(confidence_score__lt=40)

        def win_rate(qs):
            closed = qs.filter(outcome__in=['profit', 'loss', 'break_even'])
            if closed.count() == 0:
                return 0
            return closed.filter(outcome='profit').count() / closed.count()

        return {
            'total_trades': total,
            'with_entry_reasoning': with_entry_reasoning,
            'with_exit_reasoning': with_exit_reasoning,
            'with_analysis': with_analysis,
            'reasoning_coverage': with_entry_reasoning / total if total > 0 else 0,
            'avg_confidence': round(avg_confidence, 1),
            'confidence_breakdown': {
                'high': {
                    'count': high_conf.count(),
                    'win_rate': round(win_rate(high_conf), 2),
                },
                'medium': {
                    'count': mid_conf.count(),
                    'win_rate': round(win_rate(mid_conf), 2),
                },
                'low': {
                    'count': low_conf.count(),
                    'win_rate': round(win_rate(low_conf), 2),
                },
            },
        }

    # -------------------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------------------

    def _generate_entry_summary(self, signals: Dict[str, Dict[str, Any]]) -> str:
        """Generate human-readable entry summary from signals."""
        triggered = []
        for name, data in signals.items():
            if data.get('met') or data.get('triggered'):
                triggered.append(name.upper().replace('_', ' '))

        if not triggered:
            return "No specific signals triggered"

        if len(triggered) == 1:
            return f"{triggered[0]} signal"

        if len(triggered) <= 3:
            return " + ".join(triggered)

        return f"{len(triggered)} signals aligned"

    def _format_signals_for_storage(self, signals: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Format signals for storage, ensuring consistent structure."""
        formatted = {}
        for name, data in signals.items():
            # Normalize 'triggered' to 'met'
            if 'triggered' in data and 'met' not in data:
                data = dict(data)
                data['met'] = data['triggered']
            formatted[name] = data
        return formatted

    def _generate_explanation(
        self,
        signals: Dict[str, Dict[str, Any]],
        confidence: int,
        strategy_name: str,
    ) -> str:
        """Generate human-readable explanation."""
        parts = [f"Strategy: {strategy_name}"]
        parts.append(f"Confidence: {confidence}%")

        triggered = []
        for name, data in signals.items():
            if data.get('met') or data.get('triggered'):
                if 'value' in data and 'threshold' in data:
                    triggered.append(f"{name}={data['value']} (threshold {data['threshold']})")
                elif 'value' in data:
                    triggered.append(f"{name}={data['value']}")
                else:
                    triggered.append(name)

        if triggered:
            parts.append(f"Triggered signals: {', '.join(triggered)}")

        return " | ".join(parts)

    def _find_similar_trades(
        self,
        signals: Dict[str, Dict[str, Any]],
        strategy_name: str,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Find similar historical trades."""
        TradeSignalSnapshot = self._get_snapshot_model()

        # Get recent closed trades from same strategy
        recent = TradeSignalSnapshot.objects.filter(
            strategy_name=strategy_name,
            outcome__in=['profit', 'loss', 'break_even'],
        ).order_by('-created_at')[:100]

        # Simple similarity based on triggered signals
        triggered_names = {
            name for name, data in signals.items()
            if data.get('met') or data.get('triggered')
        }

        similar = []
        for trade in recent:
            trade_triggered = {
                name for name, data in trade.signals_at_entry.items()
                if isinstance(data, dict) and (data.get('met') or data.get('triggered'))
            }

            if not triggered_names and not trade_triggered:
                similarity = 0.5
            elif not triggered_names or not trade_triggered:
                similarity = 0.2
            else:
                overlap = len(triggered_names & trade_triggered)
                union = len(triggered_names | trade_triggered)
                similarity = overlap / union if union > 0 else 0

            if similarity > 0.3:
                similar.append({
                    'trade_id': trade.trade_id,
                    'similarity': round(similarity, 2),
                    'outcome': trade.outcome,
                    'pnl_pct': float(trade.pnl_percent) if trade.pnl_percent else 0,
                })

        similar.sort(key=lambda x: x['similarity'], reverse=True)
        return similar[:limit]

    def _get_similar_trades_stats(self, snapshot) -> Dict[str, Any]:
        """Get statistics from similar historical trades."""
        if not snapshot.similar_historical_trades:
            return {'count': 0, 'avg_pnl': 0, 'win_rate': 0}

        trades = snapshot.similar_historical_trades
        if not trades:
            return {'count': 0, 'avg_pnl': 0, 'win_rate': 0}

        pnls = [t['pnl_pct'] for t in trades if 'pnl_pct' in t]
        wins = sum(1 for t in trades if t.get('outcome') == 'profit')

        return {
            'count': len(trades),
            'avg_pnl': round(sum(pnls) / len(pnls), 2) if pnls else 0,
            'win_rate': round(wins / len(trades), 2) if trades else 0,
        }

    def _calculate_hold_return(
        self,
        symbol: str,
        entry_time: datetime,
        exit_time: datetime,
        historical_prices: Dict[str, float],
    ) -> float:
        """Calculate what return would have been with simple buy and hold."""
        # This would need actual price data integration
        # For now, return a placeholder
        entry_key = entry_time.strftime('%Y-%m-%d')
        exit_key = exit_time.strftime('%Y-%m-%d')

        entry_price = historical_prices.get(entry_key)
        exit_price = historical_prices.get(exit_key)

        if entry_price and exit_price and entry_price > 0:
            return ((exit_price - entry_price) / entry_price) * 100

        return 0

    def _analyze_timing(
        self,
        snapshot,
        historical_prices: Dict[str, float] = None,
    ) -> Dict[str, Any]:
        """Analyze entry and exit timing quality."""
        result = {'score': 'unknown', 'entry': {}, 'exit': {}}

        if not snapshot.exit_timestamp:
            return result

        # Basic timing score based on outcome
        if snapshot.pnl_percent:
            pnl = float(snapshot.pnl_percent)
            if pnl >= 5:
                result['score'] = 'excellent'
            elif pnl >= 2:
                result['score'] = 'good'
            elif pnl >= 0:
                result['score'] = 'fair'
            elif pnl >= -2:
                result['score'] = 'poor'
            else:
                result['score'] = 'bad'

        # If we have historical prices, do more detailed analysis
        if historical_prices:
            # Entry timing analysis
            result['entry'] = {
                'actual_price': float(snapshot.entry_price),
            }

            # Exit timing analysis
            if snapshot.exit_price:
                result['exit'] = {
                    'actual_price': float(snapshot.exit_price),
                }

        return result

    def _generate_analysis_notes(
        self,
        analysis: Dict[str, Any],
        timing: Dict[str, Any],
    ) -> str:
        """Generate analysis notes based on outcome."""
        notes = []

        # P&L commentary
        pnl_pct = analysis.get('pnl_pct', 0)
        if pnl_pct >= 5:
            notes.append("Strong profit captured")
        elif pnl_pct >= 2:
            notes.append("Solid profitable trade")
        elif pnl_pct >= 0:
            notes.append("Small profit or breakeven")
        elif pnl_pct >= -2:
            notes.append("Small loss within risk parameters")
        else:
            notes.append("Significant loss - review risk management")

        # vs_hold commentary
        vs_hold = analysis.get('vs_hold')
        if vs_hold is not None:
            if vs_hold > 2:
                notes.append("Outperformed buy-and-hold significantly")
            elif vs_hold < -2:
                notes.append("Underperformed buy-and-hold - timing needs review")

        # Similar trades comparison
        similar_avg = analysis.get('similar_trades_avg_pnl', 0)
        if similar_avg and pnl_pct:
            if pnl_pct > similar_avg * 1.5:
                notes.append("Outperformed similar historical trades")
            elif pnl_pct < similar_avg * 0.5:
                notes.append("Underperformed similar setups")

        return ". ".join(notes) if notes else ""

    def _format_duration(self, hours: float) -> str:
        """Format duration in hours to human-readable string."""
        if hours < 1:
            return f"{int(hours * 60)} minutes"
        elif hours < 24:
            return f"{hours:.1f} hours"
        elif hours < 48:
            return "1 day"
        else:
            days = int(hours / 24)
            return f"{days} days"


def get_trade_reasoning_service() -> TradeReasoningService:
    """Get trade reasoning service instance.

    Returns:
        TradeReasoningService instance
    """
    return TradeReasoningService()
