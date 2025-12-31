"""
Trade Explainer Service

Provides human-readable explanations for trades, finds similar historical trades,
and formats signal data for visualization.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class SignalExplanation:
    """Explanation for a single signal."""
    signal_name: str
    triggered: bool
    value: float
    threshold: Optional[float]
    description: str
    impact: str  # 'bullish', 'bearish', 'neutral'


@dataclass
class TradeExplanation:
    """Complete explanation for a trade."""
    trade_id: str
    symbol: str
    direction: str
    strategy_name: str
    entry_price: float
    quantity: float
    confidence_score: int
    summary: str  # Plain English summary
    signal_explanations: list[SignalExplanation]
    key_factors: list[str]
    risk_assessment: str
    similar_trades_summary: str
    timestamp: str


@dataclass
class SimilarTrade:
    """A similar historical trade."""
    trade_id: str
    symbol: str
    similarity_score: float
    outcome: str
    pnl_percent: float
    entry_date: str
    key_signals_matched: list[str]


@dataclass
class VisualizationData:
    """Data formatted for chart visualization."""
    rsi_gauge: Optional[dict]
    macd_chart: Optional[dict]
    volume_bar: Optional[dict]
    price_chart: Optional[dict]
    confidence_meter: dict
    signal_timeline: list[dict]


class TradeExplainerService:
    """Service for generating trade explanations and finding similar trades."""

    # Signal descriptions for human-readable explanations
    SIGNAL_DESCRIPTIONS = {
        'rsi': {
            'name': 'Relative Strength Index (RSI)',
            'oversold': 'RSI dropped to {value:.0f} (below {threshold:.0f}), indicating oversold conditions',
            'overbought': 'RSI rose to {value:.0f} (above {threshold:.0f}), indicating overbought conditions',
            'neutral': 'RSI at {value:.0f}, within normal range',
        },
        'macd': {
            'name': 'MACD',
            'bullish_crossover': 'MACD crossed above signal line, indicating bullish momentum',
            'bearish_crossover': 'MACD crossed below signal line, indicating bearish momentum',
            'positive_histogram': 'MACD histogram is positive ({histogram:.3f}), showing bullish momentum',
            'negative_histogram': 'MACD histogram is negative ({histogram:.3f}), showing bearish momentum',
        },
        'volume': {
            'name': 'Volume',
            'high_volume': 'Volume is {ratio:.1f}x the average ({current:,.0f} vs avg {average:,.0f})',
            'normal_volume': 'Volume is at {ratio:.1f}x average, indicating normal activity',
            'low_volume': 'Volume is only {ratio:.1f}x average, indicating low interest',
        },
        'price_action': {
            'name': 'Price Action',
            'significant_drop': 'Price dropped {change_pct:.1f}% and is {from_sma20:.1f}% below 20-day SMA',
            'significant_rise': 'Price rose {change_pct:.1f}% and is {from_sma20:.1f}% above 20-day SMA',
            'near_sma': 'Price is within {from_sma20:.1f}% of 20-day SMA',
        },
        'bollinger': {
            'name': 'Bollinger Bands',
            'below_lower': 'Price broke below lower Bollinger Band, potential reversal zone',
            'above_upper': 'Price broke above upper Bollinger Band, potentially overextended',
            'middle': 'Price is within Bollinger Bands',
        },
        'stochastic': {
            'name': 'Stochastic',
            'oversold': 'Stochastic %K at {k:.0f} (oversold zone)',
            'overbought': 'Stochastic %K at {k:.0f} (overbought zone)',
            'neutral': 'Stochastic at neutral levels',
        },
        'trend': {
            'name': 'Trend Analysis',
            'aligned_bullish': 'Short, medium, and long-term trends are all bullish',
            'aligned_bearish': 'Short, medium, and long-term trends are all bearish',
            'mixed': 'Mixed trend signals across timeframes',
        },
    }

    def __init__(self):
        pass

    def explain_trade(self, trade_id: str) -> Optional[TradeExplanation]:
        """
        Generate a complete human-readable explanation for a trade.

        Args:
            trade_id: The trade identifier

        Returns:
            TradeExplanation with full details, or None if not found
        """
        from backend.tradingbot.models.models import TradeSignalSnapshot

        try:
            snapshot = TradeSignalSnapshot.objects.get(trade_id=trade_id)
        except TradeSignalSnapshot.DoesNotExist:
            logger.warning(f"Trade snapshot not found for trade_id: {trade_id}")
            return None

        # Generate signal explanations
        signal_explanations = self._generate_signal_explanations(snapshot.signals_at_entry)

        # Generate summary
        summary = self._generate_summary(snapshot, signal_explanations)

        # Extract key factors
        key_factors = self._extract_key_factors(snapshot)

        # Generate risk assessment
        risk_assessment = self._generate_risk_assessment(snapshot)

        # Summarize similar trades
        similar_summary = self._summarize_similar_trades(snapshot.similar_historical_trades)

        return TradeExplanation(
            trade_id=snapshot.trade_id,
            symbol=snapshot.symbol,
            direction=snapshot.direction,
            strategy_name=snapshot.strategy_name,
            entry_price=float(snapshot.entry_price),
            quantity=float(snapshot.quantity),
            confidence_score=snapshot.confidence_score,
            summary=summary,
            signal_explanations=signal_explanations,
            key_factors=key_factors,
            risk_assessment=risk_assessment,
            similar_trades_summary=similar_summary,
            timestamp=snapshot.created_at.isoformat(),
        )

    def _generate_signal_explanations(self, signals: dict) -> list[SignalExplanation]:
        """Generate human-readable explanations for each signal."""
        explanations = []

        for signal_name, signal_data in signals.items():
            if not isinstance(signal_data, dict):
                continue

            explanation = self._explain_single_signal(signal_name, signal_data)
            if explanation:
                explanations.append(explanation)

        return explanations

    def _explain_single_signal(self, signal_name: str, data: dict) -> Optional[SignalExplanation]:
        """Generate explanation for a single signal."""
        triggered = data.get('triggered', False)
        value = data.get('value', 0)
        threshold = data.get('threshold')

        if signal_name == 'rsi':
            if value < 30:
                desc = self.SIGNAL_DESCRIPTIONS['rsi']['oversold'].format(
                    value=value, threshold=threshold or 30
                )
                impact = 'bullish'
            elif value > 70:
                desc = self.SIGNAL_DESCRIPTIONS['rsi']['overbought'].format(
                    value=value, threshold=threshold or 70
                )
                impact = 'bearish'
            else:
                desc = self.SIGNAL_DESCRIPTIONS['rsi']['neutral'].format(value=value)
                impact = 'neutral'

        elif signal_name == 'macd':
            crossover = data.get('crossover', False)
            histogram = data.get('histogram', 0)
            if crossover:
                if histogram > 0:
                    desc = self.SIGNAL_DESCRIPTIONS['macd']['bullish_crossover']
                    impact = 'bullish'
                else:
                    desc = self.SIGNAL_DESCRIPTIONS['macd']['bearish_crossover']
                    impact = 'bearish'
            elif histogram > 0:
                desc = self.SIGNAL_DESCRIPTIONS['macd']['positive_histogram'].format(histogram=histogram)
                impact = 'bullish'
            else:
                desc = self.SIGNAL_DESCRIPTIONS['macd']['negative_histogram'].format(histogram=histogram)
                impact = 'bearish'

        elif signal_name == 'volume':
            ratio = data.get('ratio', 1)
            current = data.get('current', 0)
            average = data.get('average', 1)
            if ratio >= 2.0:
                desc = self.SIGNAL_DESCRIPTIONS['volume']['high_volume'].format(
                    ratio=ratio, current=current, average=average
                )
                impact = 'bullish' if triggered else 'neutral'
            elif ratio < 0.5:
                desc = self.SIGNAL_DESCRIPTIONS['volume']['low_volume'].format(ratio=ratio)
                impact = 'neutral'
            else:
                desc = self.SIGNAL_DESCRIPTIONS['volume']['normal_volume'].format(ratio=ratio)
                impact = 'neutral'

        elif signal_name == 'price_action':
            change_pct = data.get('change_pct', 0)
            from_sma20 = data.get('from_sma20', 0)
            if change_pct < -2:
                desc = self.SIGNAL_DESCRIPTIONS['price_action']['significant_drop'].format(
                    change_pct=change_pct, from_sma20=from_sma20
                )
                impact = 'bearish'
            elif change_pct > 2:
                desc = self.SIGNAL_DESCRIPTIONS['price_action']['significant_rise'].format(
                    change_pct=change_pct, from_sma20=from_sma20
                )
                impact = 'bullish'
            else:
                desc = self.SIGNAL_DESCRIPTIONS['price_action']['near_sma'].format(from_sma20=abs(from_sma20))
                impact = 'neutral'

        elif signal_name == 'bollinger':
            position = data.get('position', 'middle')
            if position == 'below_lower':
                desc = self.SIGNAL_DESCRIPTIONS['bollinger']['below_lower']
                impact = 'bullish'
            elif position == 'above_upper':
                desc = self.SIGNAL_DESCRIPTIONS['bollinger']['above_upper']
                impact = 'bearish'
            else:
                desc = self.SIGNAL_DESCRIPTIONS['bollinger']['middle']
                impact = 'neutral'

        elif signal_name == 'stochastic':
            k = data.get('k', 50)
            if k < 20:
                desc = self.SIGNAL_DESCRIPTIONS['stochastic']['oversold'].format(k=k)
                impact = 'bullish'
            elif k > 80:
                desc = self.SIGNAL_DESCRIPTIONS['stochastic']['overbought'].format(k=k)
                impact = 'bearish'
            else:
                desc = self.SIGNAL_DESCRIPTIONS['stochastic']['neutral']
                impact = 'neutral'

        elif signal_name == 'trend':
            short_term = data.get('short_term', 'neutral')
            medium_term = data.get('medium_term', 'neutral')
            long_term = data.get('long_term', 'neutral')
            if short_term == medium_term == long_term == 'bullish':
                desc = self.SIGNAL_DESCRIPTIONS['trend']['aligned_bullish']
                impact = 'bullish'
            elif short_term == medium_term == long_term == 'bearish':
                desc = self.SIGNAL_DESCRIPTIONS['trend']['aligned_bearish']
                impact = 'bearish'
            else:
                desc = self.SIGNAL_DESCRIPTIONS['trend']['mixed']
                impact = 'neutral'

        else:
            # Generic signal
            desc = f"{signal_name.replace('_', ' ').title()}: value={value}"
            impact = 'bullish' if triggered else 'neutral'

        return SignalExplanation(
            signal_name=signal_name,
            triggered=triggered,
            value=value,
            threshold=threshold,
            description=desc,
            impact=impact,
        )

    def _generate_summary(self, snapshot, explanations: list[SignalExplanation]) -> str:
        """Generate a plain English summary of the trade."""
        direction_text = 'buy' if snapshot.direction in ('buy', 'buy_to_cover') else 'sell'
        triggered_signals = [e for e in explanations if e.triggered]
        bullish_count = sum(1 for e in triggered_signals if e.impact == 'bullish')
        bearish_count = sum(1 for e in triggered_signals if e.impact == 'bearish')

        # Build summary parts
        parts = [f"This {direction_text} trade on {snapshot.symbol} was triggered by the {snapshot.strategy_name} strategy."]

        if triggered_signals:
            signal_names = [e.signal_name.upper() for e in triggered_signals[:3]]
            parts.append(f"Key signals that triggered this trade: {', '.join(signal_names)}.")

        if bullish_count > bearish_count:
            parts.append(f"The overall signal bias was bullish ({bullish_count} bullish vs {bearish_count} bearish signals).")
        elif bearish_count > bullish_count:
            parts.append(f"The overall signal bias was bearish ({bearish_count} bearish vs {bullish_count} bullish signals).")
        else:
            parts.append("The signals showed mixed sentiment.")

        parts.append(f"Confidence score: {snapshot.confidence_score}%.")

        return " ".join(parts)

    def _extract_key_factors(self, snapshot) -> list[str]:
        """Extract the most important factors that drove this trade."""
        factors = []
        signals = snapshot.signals_at_entry

        # Check RSI
        if 'rsi' in signals:
            rsi_val = signals['rsi'].get('value', 50)
            if rsi_val < 30:
                factors.append(f"Oversold RSI ({rsi_val:.0f})")
            elif rsi_val > 70:
                factors.append(f"Overbought RSI ({rsi_val:.0f})")

        # Check volume
        if 'volume' in signals:
            ratio = signals['volume'].get('ratio', 1)
            if ratio >= 2.0:
                factors.append(f"High volume ({ratio:.1f}x average)")

        # Check MACD
        if 'macd' in signals:
            if signals['macd'].get('crossover'):
                factors.append("MACD crossover")

        # Check price action
        if 'price_action' in signals:
            change = signals['price_action'].get('change_pct', 0)
            if abs(change) >= 3:
                factors.append(f"Significant price move ({change:+.1f}%)")

        # Check Bollinger
        if 'bollinger' in signals:
            position = signals['bollinger'].get('position')
            if position == 'below_lower':
                factors.append("Below Bollinger lower band")
            elif position == 'above_upper':
                factors.append("Above Bollinger upper band")

        return factors[:5]  # Return top 5 factors

    def _generate_risk_assessment(self, snapshot) -> str:
        """Generate a risk assessment for the trade."""
        confidence = snapshot.confidence_score

        if confidence >= 80:
            return "High confidence trade with strong signal alignment. Risk is relatively low for this setup."
        elif confidence >= 60:
            return "Moderate confidence trade. Some signals aligned but not all. Standard risk management applies."
        elif confidence >= 40:
            return "Lower confidence trade. Consider reducing position size or tightening stops."
        else:
            return "Low confidence trade. High risk - careful position sizing recommended."

    def _summarize_similar_trades(self, similar_trades: list) -> str:
        """Summarize the outcomes of similar historical trades."""
        if not similar_trades:
            return "No similar historical trades found for comparison."

        total = len(similar_trades)
        profits = sum(1 for t in similar_trades if t.get('outcome') == 'profit')
        losses = sum(1 for t in similar_trades if t.get('outcome') == 'loss')
        avg_pnl = sum(t.get('pnl_pct', 0) for t in similar_trades) / total if total > 0 else 0

        win_rate = (profits / total * 100) if total > 0 else 0

        return (
            f"Found {total} similar historical trades. "
            f"Win rate: {win_rate:.0f}% ({profits} wins, {losses} losses). "
            f"Average P&L: {avg_pnl:+.1f}%."
        )

    def find_similar_trades(
        self,
        trade_id: str,
        limit: int = 10,
        min_similarity: float = 0.7
    ) -> list[SimilarTrade]:
        """
        Find historical trades with similar signal setups.

        Args:
            trade_id: The trade to find similar trades for
            limit: Maximum number of similar trades to return
            min_similarity: Minimum similarity score (0-1)

        Returns:
            List of similar trades sorted by similarity
        """
        from backend.tradingbot.models.models import TradeSignalSnapshot

        try:
            target = TradeSignalSnapshot.objects.get(trade_id=trade_id)
        except TradeSignalSnapshot.DoesNotExist:
            return []

        # Get historical trades for comparison (exclude the target trade)
        historical = TradeSignalSnapshot.objects.filter(
            outcome__isnull=False  # Only completed trades
        ).exclude(
            trade_id=trade_id
        ).order_by('-created_at')[:500]  # Limit for performance

        similar_trades = []
        for hist_trade in historical:
            similarity = target.calculate_similarity(hist_trade)
            if similarity >= min_similarity:
                # Find which signals matched
                matched_signals = self._find_matched_signals(
                    target.signals_at_entry,
                    hist_trade.signals_at_entry
                )
                similar_trades.append(SimilarTrade(
                    trade_id=hist_trade.trade_id,
                    symbol=hist_trade.symbol,
                    similarity_score=round(similarity, 2),
                    outcome=hist_trade.outcome or 'unknown',
                    pnl_percent=float(hist_trade.pnl_percent) if hist_trade.pnl_percent else 0,
                    entry_date=hist_trade.created_at.strftime('%Y-%m-%d'),
                    key_signals_matched=matched_signals[:3],
                ))

        # Sort by similarity and return top matches
        similar_trades.sort(key=lambda x: x.similarity_score, reverse=True)
        return similar_trades[:limit]

    def _find_matched_signals(self, signals1: dict, signals2: dict) -> list[str]:
        """Find which signals matched between two snapshots."""
        matched = []
        for signal_name in signals1:
            if signal_name in signals2:
                s1 = signals1[signal_name]
                s2 = signals2[signal_name]
                if isinstance(s1, dict) and isinstance(s2, dict):
                    if s1.get('triggered') == s2.get('triggered'):
                        matched.append(signal_name)
        return matched

    def get_signal_visualization_data(self, trade_id: str) -> Optional[VisualizationData]:
        """
        Format signal data for chart visualization.

        Args:
            trade_id: The trade identifier

        Returns:
            VisualizationData for chart components, or None if not found
        """
        from backend.tradingbot.models.models import TradeSignalSnapshot

        try:
            snapshot = TradeSignalSnapshot.objects.get(trade_id=trade_id)
        except TradeSignalSnapshot.DoesNotExist:
            return None

        signals = snapshot.signals_at_entry

        # RSI gauge data
        rsi_gauge = None
        if 'rsi' in signals:
            rsi_data = signals['rsi']
            rsi_gauge = {
                'value': rsi_data.get('value', 50),
                'min': 0,
                'max': 100,
                'zones': [
                    {'min': 0, 'max': 30, 'color': '#2dce89', 'label': 'Oversold'},
                    {'min': 30, 'max': 70, 'color': '#6c757d', 'label': 'Neutral'},
                    {'min': 70, 'max': 100, 'color': '#f5365c', 'label': 'Overbought'},
                ],
                'threshold': rsi_data.get('threshold'),
                'triggered': rsi_data.get('triggered', False),
            }

        # MACD chart data
        macd_chart = None
        if 'macd' in signals:
            macd_data = signals['macd']
            macd_chart = {
                'macd_line': macd_data.get('value', 0),
                'signal_line': macd_data.get('signal', 0),
                'histogram': macd_data.get('histogram', 0),
                'crossover': macd_data.get('crossover', False),
                'histogram_color': '#2dce89' if macd_data.get('histogram', 0) > 0 else '#f5365c',
            }

        # Volume bar data
        volume_bar = None
        if 'volume' in signals:
            vol_data = signals['volume']
            volume_bar = {
                'current': vol_data.get('current', 0),
                'average': vol_data.get('average', 1),
                'ratio': vol_data.get('ratio', 1),
                'percentage': min(vol_data.get('ratio', 1) / 3 * 100, 100),  # Cap at 3x = 100%
                'color': '#2dce89' if vol_data.get('ratio', 1) >= 2 else '#6c757d',
            }

        # Price chart data (would need historical prices from elsewhere)
        price_chart = None
        if 'price_action' in signals:
            pa_data = signals['price_action']
            price_chart = {
                'entry_price': float(snapshot.entry_price),
                'change_pct': pa_data.get('change_pct', 0),
                'from_sma20': pa_data.get('from_sma20', 0),
                'from_sma50': pa_data.get('from_sma50', 0),
            }

        # Confidence meter
        confidence_meter = {
            'value': snapshot.confidence_score,
            'min': 0,
            'max': 100,
            'color': self._get_confidence_color(snapshot.confidence_score),
            'label': self._get_confidence_label(snapshot.confidence_score),
        }

        # Signal timeline
        signal_timeline = []
        for signal_name, signal_data in signals.items():
            if isinstance(signal_data, dict):
                signal_timeline.append({
                    'name': signal_name.upper(),
                    'triggered': signal_data.get('triggered', False),
                    'impact': self._determine_impact(signal_name, signal_data),
                })

        return VisualizationData(
            rsi_gauge=rsi_gauge,
            macd_chart=macd_chart,
            volume_bar=volume_bar,
            price_chart=price_chart,
            confidence_meter=confidence_meter,
            signal_timeline=signal_timeline,
        )

    def _get_confidence_color(self, confidence: int) -> str:
        """Get color for confidence score."""
        if confidence >= 70:
            return '#2dce89'  # Green
        elif confidence >= 50:
            return '#fb6340'  # Orange
        else:
            return '#f5365c'  # Red

    def _get_confidence_label(self, confidence: int) -> str:
        """Get label for confidence score."""
        if confidence >= 80:
            return 'Very High'
        elif confidence >= 70:
            return 'High'
        elif confidence >= 60:
            return 'Moderate'
        elif confidence >= 50:
            return 'Fair'
        else:
            return 'Low'

    def _determine_impact(self, signal_name: str, data: dict) -> str:
        """Determine the impact direction of a signal."""
        if signal_name == 'rsi':
            val = data.get('value', 50)
            return 'bullish' if val < 30 else ('bearish' if val > 70 else 'neutral')
        elif signal_name == 'macd':
            hist = data.get('histogram', 0)
            return 'bullish' if hist > 0 else 'bearish'
        elif signal_name == 'volume':
            return 'neutral'
        elif signal_name == 'price_action':
            change = data.get('change_pct', 0)
            return 'bullish' if change > 0 else ('bearish' if change < 0 else 'neutral')
        return 'neutral'


# Singleton instance
trade_explainer_service = TradeExplainerService()
