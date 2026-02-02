"""Integration tests for TradeExplainerService.

These are REAL integration tests that:
- Use actual database operations
- Test real trade explanation generation
- Create real TradeSignalSnapshot records
- Use real signal analysis and calculations

Tests cover:
- Trade explanation generation from real trade data
- Signal analysis using real indicators
- Risk assessment calculation
- Explanation text formatting
- Similar trade finding
- Visualization data generation
"""

import pytest
from datetime import datetime, timedelta
from decimal import Decimal
from django.utils import timezone
import uuid

from backend.auth0login.services.trade_explainer import (
    TradeExplainerService,
    SignalExplanation,
    TradeExplanation,
    SimilarTrade,
    VisualizationData,
)
from backend.tradingbot.models.models import TradeSignalSnapshot


@pytest.fixture
def service():
    """Create a TradeExplainerService instance."""
    return TradeExplainerService()


@pytest.fixture
def create_trade_snapshot(db):
    """Factory fixture to create real TradeSignalSnapshot records."""
    created_snapshots = []

    def _create_snapshot(
        trade_id=None,
        symbol='AAPL',
        direction='buy',
        strategy_name='wsb_dip_bot',
        entry_price=150.00,
        quantity=100,
        confidence_score=75,
        signals_at_entry=None,
        similar_historical_trades=None,
        outcome=None,
        pnl_percent=None,
    ):
        if trade_id is None:
            trade_id = f'TEST-{uuid.uuid4().hex[:8].upper()}'

        if signals_at_entry is None:
            signals_at_entry = {
                'rsi': {'value': 28, 'threshold': 30, 'triggered': True},
                'macd': {'value': -0.5, 'signal': -0.3, 'histogram': -0.2, 'crossover': False},
                'volume': {'current': 1000000, 'average': 500000, 'ratio': 2.0, 'triggered': True}
            }

        if similar_historical_trades is None:
            similar_historical_trades = []

        snapshot = TradeSignalSnapshot.objects.create(
            trade_id=trade_id,
            symbol=symbol,
            direction=direction,
            strategy_name=strategy_name,
            entry_price=Decimal(str(entry_price)),
            quantity=Decimal(str(quantity)),
            confidence_score=confidence_score,
            signals_at_entry=signals_at_entry,
            similar_historical_trades=similar_historical_trades,
            outcome=outcome,
            pnl_percent=Decimal(str(pnl_percent)) if pnl_percent is not None else None,
        )
        created_snapshots.append(snapshot)
        return snapshot

    yield _create_snapshot

    # Cleanup - delete all created snapshots
    for snapshot in created_snapshots:
        try:
            snapshot.delete()
        except Exception:
            pass


class TestTradeExplainerServiceInit:
    """Test TradeExplainerService initialization."""

    def test_service_initialization(self):
        """Test service initializes correctly with signal descriptions."""
        service = TradeExplainerService()

        assert service is not None
        assert hasattr(service, 'SIGNAL_DESCRIPTIONS')
        assert 'rsi' in service.SIGNAL_DESCRIPTIONS
        assert 'macd' in service.SIGNAL_DESCRIPTIONS
        assert 'volume' in service.SIGNAL_DESCRIPTIONS
        assert 'price_action' in service.SIGNAL_DESCRIPTIONS
        assert 'bollinger' in service.SIGNAL_DESCRIPTIONS
        assert 'stochastic' in service.SIGNAL_DESCRIPTIONS
        assert 'trend' in service.SIGNAL_DESCRIPTIONS


@pytest.mark.django_db
class TestExplainTradeIntegration:
    """Integration tests for explain_trade method using real database records."""

    def test_explain_trade_not_found_returns_none(self, service):
        """Test explanation returns None when trade not found in database."""
        result = service.explain_trade('NONEXISTENT-TRADE-ID')
        assert result is None

    def test_explain_trade_with_real_snapshot(self, service, create_trade_snapshot):
        """Test successful trade explanation with real database record."""
        snapshot = create_trade_snapshot(
            trade_id='REAL-TEST-001',
            symbol='AAPL',
            direction='buy',
            strategy_name='wsb_dip_bot',
            entry_price=150.00,
            quantity=100,
            confidence_score=85,
            signals_at_entry={
                'rsi': {'value': 28, 'threshold': 30, 'triggered': True},
                'macd': {'value': -0.5, 'signal': -0.3, 'histogram': -0.2, 'crossover': False},
                'volume': {'current': 1000000, 'average': 500000, 'ratio': 2.0, 'triggered': True}
            }
        )

        result = service.explain_trade('REAL-TEST-001')

        assert result is not None
        assert isinstance(result, TradeExplanation)
        assert result.trade_id == 'REAL-TEST-001'
        assert result.symbol == 'AAPL'
        assert result.direction == 'buy'
        assert result.strategy_name == 'wsb_dip_bot'
        assert result.entry_price == 150.0
        assert result.quantity == 100.0
        assert result.confidence_score == 85
        assert len(result.signal_explanations) > 0
        assert result.summary is not None
        assert len(result.summary) > 0

    def test_explain_trade_with_similar_historical_trades(self, service, create_trade_snapshot):
        """Test explanation includes similar trades summary from real data."""
        snapshot = create_trade_snapshot(
            trade_id='REAL-TEST-002',
            similar_historical_trades=[
                {'trade_id': 'HIST-001', 'outcome': 'profit', 'pnl_pct': 5.0},
                {'trade_id': 'HIST-002', 'outcome': 'profit', 'pnl_pct': 3.0},
                {'trade_id': 'HIST-003', 'outcome': 'loss', 'pnl_pct': -2.0}
            ]
        )

        result = service.explain_trade('REAL-TEST-002')

        assert result is not None
        assert 'similar' in result.similar_trades_summary.lower()
        assert '3' in result.similar_trades_summary  # Found 3 similar trades
        assert '67%' in result.similar_trades_summary or '66%' in result.similar_trades_summary  # Win rate

    def test_explain_trade_generates_correct_summary(self, service, create_trade_snapshot):
        """Test trade summary is properly generated with real data."""
        snapshot = create_trade_snapshot(
            trade_id='REAL-TEST-003',
            symbol='TSLA',
            direction='buy',
            strategy_name='momentum_bot',
            confidence_score=75,
            signals_at_entry={
                'rsi': {'value': 25, 'threshold': 30, 'triggered': True},
                'volume': {'current': 2000000, 'average': 1000000, 'ratio': 2.0, 'triggered': True}
            }
        )

        result = service.explain_trade('REAL-TEST-003')

        assert 'buy' in result.summary.lower()
        assert 'TSLA' in result.summary
        assert 'momentum_bot' in result.summary
        assert '75%' in result.summary  # Confidence score

    def test_explain_trade_sell_direction(self, service, create_trade_snapshot):
        """Test explanation for sell direction trade."""
        snapshot = create_trade_snapshot(
            trade_id='REAL-TEST-004',
            direction='sell',
            signals_at_entry={
                'rsi': {'value': 75, 'threshold': 70, 'triggered': True},
            }
        )

        result = service.explain_trade('REAL-TEST-004')

        assert result.direction == 'sell'
        assert 'sell' in result.summary.lower()


@pytest.mark.django_db
class TestSignalExplanationsIntegration:
    """Integration tests for signal explanation generation."""

    def test_rsi_oversold_explanation(self, service, create_trade_snapshot):
        """Test RSI oversold signal generates correct explanation."""
        snapshot = create_trade_snapshot(
            trade_id='RSI-OVERSOLD-TEST',
            signals_at_entry={
                'rsi': {'value': 22, 'threshold': 30, 'triggered': True}
            }
        )

        result = service.explain_trade('RSI-OVERSOLD-TEST')

        rsi_explanation = next(
            (e for e in result.signal_explanations if e.signal_name == 'rsi'),
            None
        )
        assert rsi_explanation is not None
        assert rsi_explanation.impact == 'bullish'
        assert rsi_explanation.triggered is True
        assert '22' in rsi_explanation.description
        assert 'oversold' in rsi_explanation.description.lower()

    def test_rsi_overbought_explanation(self, service, create_trade_snapshot):
        """Test RSI overbought signal generates correct explanation."""
        snapshot = create_trade_snapshot(
            trade_id='RSI-OVERBOUGHT-TEST',
            signals_at_entry={
                'rsi': {'value': 78, 'threshold': 70, 'triggered': True}
            }
        )

        result = service.explain_trade('RSI-OVERBOUGHT-TEST')

        rsi_explanation = next(
            (e for e in result.signal_explanations if e.signal_name == 'rsi'),
            None
        )
        assert rsi_explanation is not None
        assert rsi_explanation.impact == 'bearish'
        assert '78' in rsi_explanation.description
        assert 'overbought' in rsi_explanation.description.lower()

    def test_rsi_neutral_explanation(self, service, create_trade_snapshot):
        """Test RSI neutral signal generates correct explanation."""
        snapshot = create_trade_snapshot(
            trade_id='RSI-NEUTRAL-TEST',
            signals_at_entry={
                'rsi': {'value': 50, 'threshold': 30, 'triggered': False}
            }
        )

        result = service.explain_trade('RSI-NEUTRAL-TEST')

        rsi_explanation = next(
            (e for e in result.signal_explanations if e.signal_name == 'rsi'),
            None
        )
        assert rsi_explanation is not None
        assert rsi_explanation.impact == 'neutral'
        assert 'normal range' in rsi_explanation.description.lower()

    def test_macd_bullish_crossover_explanation(self, service, create_trade_snapshot):
        """Test MACD bullish crossover signal generates correct explanation."""
        snapshot = create_trade_snapshot(
            trade_id='MACD-BULLISH-TEST',
            signals_at_entry={
                'macd': {'value': 0.5, 'signal': 0.3, 'histogram': 0.2, 'crossover': True}
            }
        )

        result = service.explain_trade('MACD-BULLISH-TEST')

        macd_explanation = next(
            (e for e in result.signal_explanations if e.signal_name == 'macd'),
            None
        )
        assert macd_explanation is not None
        assert macd_explanation.impact == 'bullish'
        # Description says "MACD crossed above signal line" for bullish crossover
        assert 'crossed above signal line' in macd_explanation.description.lower()

    def test_macd_bearish_crossover_explanation(self, service, create_trade_snapshot):
        """Test MACD bearish crossover signal generates correct explanation."""
        snapshot = create_trade_snapshot(
            trade_id='MACD-BEARISH-TEST',
            signals_at_entry={
                'macd': {'value': -0.5, 'signal': -0.3, 'histogram': -0.2, 'crossover': True}
            }
        )

        result = service.explain_trade('MACD-BEARISH-TEST')

        macd_explanation = next(
            (e for e in result.signal_explanations if e.signal_name == 'macd'),
            None
        )
        assert macd_explanation is not None
        assert macd_explanation.impact == 'bearish'

    def test_high_volume_explanation(self, service, create_trade_snapshot):
        """Test high volume signal generates correct explanation."""
        snapshot = create_trade_snapshot(
            trade_id='HIGH-VOLUME-TEST',
            signals_at_entry={
                'volume': {'current': 3000000, 'average': 1000000, 'ratio': 3.0, 'triggered': True}
            }
        )

        result = service.explain_trade('HIGH-VOLUME-TEST')

        volume_explanation = next(
            (e for e in result.signal_explanations if e.signal_name == 'volume'),
            None
        )
        assert volume_explanation is not None
        assert '3.0x' in volume_explanation.description
        assert volume_explanation.impact == 'bullish'

    def test_low_volume_explanation(self, service, create_trade_snapshot):
        """Test low volume signal generates correct explanation."""
        snapshot = create_trade_snapshot(
            trade_id='LOW-VOLUME-TEST',
            signals_at_entry={
                'volume': {'current': 400000, 'average': 1000000, 'ratio': 0.4, 'triggered': False}
            }
        )

        result = service.explain_trade('LOW-VOLUME-TEST')

        volume_explanation = next(
            (e for e in result.signal_explanations if e.signal_name == 'volume'),
            None
        )
        assert volume_explanation is not None
        assert volume_explanation.impact == 'neutral'
        assert 'low' in volume_explanation.description.lower()

    def test_price_action_drop_explanation(self, service, create_trade_snapshot):
        """Test significant price drop generates correct explanation."""
        snapshot = create_trade_snapshot(
            trade_id='PRICE-DROP-TEST',
            signals_at_entry={
                'price_action': {'change_pct': -4.5, 'from_sma20': -6.0}
            }
        )

        result = service.explain_trade('PRICE-DROP-TEST')

        pa_explanation = next(
            (e for e in result.signal_explanations if e.signal_name == 'price_action'),
            None
        )
        assert pa_explanation is not None
        assert pa_explanation.impact == 'bearish'
        assert '-4.5%' in pa_explanation.description

    def test_price_action_rise_explanation(self, service, create_trade_snapshot):
        """Test significant price rise generates correct explanation."""
        snapshot = create_trade_snapshot(
            trade_id='PRICE-RISE-TEST',
            signals_at_entry={
                'price_action': {'change_pct': 5.0, 'from_sma20': 7.0}
            }
        )

        result = service.explain_trade('PRICE-RISE-TEST')

        pa_explanation = next(
            (e for e in result.signal_explanations if e.signal_name == 'price_action'),
            None
        )
        assert pa_explanation is not None
        assert pa_explanation.impact == 'bullish'

    def test_bollinger_below_lower_band_explanation(self, service, create_trade_snapshot):
        """Test Bollinger below lower band generates correct explanation."""
        snapshot = create_trade_snapshot(
            trade_id='BB-BELOW-TEST',
            signals_at_entry={
                'bollinger': {'upper': 160.0, 'lower': 140.0, 'position': 'below_lower'}
            }
        )

        result = service.explain_trade('BB-BELOW-TEST')

        bb_explanation = next(
            (e for e in result.signal_explanations if e.signal_name == 'bollinger'),
            None
        )
        assert bb_explanation is not None
        assert bb_explanation.impact == 'bullish'
        assert 'below' in bb_explanation.description.lower()

    def test_stochastic_oversold_explanation(self, service, create_trade_snapshot):
        """Test stochastic oversold generates correct explanation."""
        snapshot = create_trade_snapshot(
            trade_id='STOCH-OVERSOLD-TEST',
            signals_at_entry={
                'stochastic': {'k': 15, 'd': 18, 'oversold': True}
            }
        )

        result = service.explain_trade('STOCH-OVERSOLD-TEST')

        stoch_explanation = next(
            (e for e in result.signal_explanations if e.signal_name == 'stochastic'),
            None
        )
        assert stoch_explanation is not None
        assert stoch_explanation.impact == 'bullish'
        assert '15' in stoch_explanation.description

    def test_trend_aligned_bullish_explanation(self, service, create_trade_snapshot):
        """Test aligned bullish trends generate correct explanation."""
        snapshot = create_trade_snapshot(
            trade_id='TREND-BULLISH-TEST',
            signals_at_entry={
                'trend': {'short_term': 'bullish', 'medium_term': 'bullish', 'long_term': 'bullish'}
            }
        )

        result = service.explain_trade('TREND-BULLISH-TEST')

        trend_explanation = next(
            (e for e in result.signal_explanations if e.signal_name == 'trend'),
            None
        )
        assert trend_explanation is not None
        assert trend_explanation.impact == 'bullish'
        assert 'bullish' in trend_explanation.description.lower()

    def test_multiple_signals_explanation(self, service, create_trade_snapshot):
        """Test explanation with multiple signals."""
        snapshot = create_trade_snapshot(
            trade_id='MULTI-SIGNAL-TEST',
            signals_at_entry={
                'rsi': {'value': 25, 'threshold': 30, 'triggered': True},
                'macd': {'value': 0.3, 'signal': 0.1, 'histogram': 0.2, 'crossover': True},
                'volume': {'current': 2500000, 'average': 1000000, 'ratio': 2.5, 'triggered': True},
                'bollinger': {'position': 'below_lower'},
                'stochastic': {'k': 18}
            }
        )

        result = service.explain_trade('MULTI-SIGNAL-TEST')

        assert len(result.signal_explanations) == 5
        bullish_count = sum(1 for e in result.signal_explanations if e.impact == 'bullish')
        assert bullish_count >= 3  # RSI, MACD, bollinger, stochastic should be bullish


@pytest.mark.django_db
class TestKeyFactorsIntegration:
    """Integration tests for key factor extraction."""

    def test_extract_oversold_rsi_factor(self, service, create_trade_snapshot):
        """Test oversold RSI is extracted as key factor."""
        snapshot = create_trade_snapshot(
            trade_id='KEY-FACTOR-RSI-TEST',
            signals_at_entry={
                'rsi': {'value': 22, 'triggered': True}
            }
        )

        result = service.explain_trade('KEY-FACTOR-RSI-TEST')

        assert any('Oversold RSI' in f for f in result.key_factors)

    def test_extract_high_volume_factor(self, service, create_trade_snapshot):
        """Test high volume is extracted as key factor."""
        snapshot = create_trade_snapshot(
            trade_id='KEY-FACTOR-VOL-TEST',
            signals_at_entry={
                'volume': {'ratio': 3.0, 'triggered': True}
            }
        )

        result = service.explain_trade('KEY-FACTOR-VOL-TEST')

        assert any('High volume' in f for f in result.key_factors)

    def test_extract_macd_crossover_factor(self, service, create_trade_snapshot):
        """Test MACD crossover is extracted as key factor."""
        snapshot = create_trade_snapshot(
            trade_id='KEY-FACTOR-MACD-TEST',
            signals_at_entry={
                'macd': {'crossover': True}
            }
        )

        result = service.explain_trade('KEY-FACTOR-MACD-TEST')

        assert any('MACD crossover' in f for f in result.key_factors)

    def test_extract_significant_price_move_factor(self, service, create_trade_snapshot):
        """Test significant price move is extracted as key factor."""
        snapshot = create_trade_snapshot(
            trade_id='KEY-FACTOR-PRICE-TEST',
            signals_at_entry={
                'price_action': {'change_pct': -5.0}
            }
        )

        result = service.explain_trade('KEY-FACTOR-PRICE-TEST')

        assert any('price move' in f.lower() for f in result.key_factors)

    def test_extract_bollinger_factor(self, service, create_trade_snapshot):
        """Test Bollinger band break is extracted as key factor."""
        snapshot = create_trade_snapshot(
            trade_id='KEY-FACTOR-BB-TEST',
            signals_at_entry={
                'bollinger': {'position': 'below_lower'}
            }
        )

        result = service.explain_trade('KEY-FACTOR-BB-TEST')

        assert any('Bollinger' in f for f in result.key_factors)

    def test_key_factors_limited_to_five(self, service, create_trade_snapshot):
        """Test that key factors are limited to top 5."""
        snapshot = create_trade_snapshot(
            trade_id='KEY-FACTOR-LIMIT-TEST',
            signals_at_entry={
                'rsi': {'value': 20},  # Oversold
                'volume': {'ratio': 3.5},  # High volume
                'macd': {'crossover': True},  # Crossover
                'price_action': {'change_pct': -6.0},  # Significant move
                'bollinger': {'position': 'below_lower'},  # Below lower
                'stochastic': {'k': 10},  # Would add another factor
            }
        )

        result = service.explain_trade('KEY-FACTOR-LIMIT-TEST')

        assert len(result.key_factors) <= 5


@pytest.mark.django_db
class TestRiskAssessmentIntegration:
    """Integration tests for risk assessment calculation."""

    def test_high_confidence_risk_assessment(self, service, create_trade_snapshot):
        """Test risk assessment for high confidence trade (>=80)."""
        snapshot = create_trade_snapshot(
            trade_id='RISK-HIGH-CONF-TEST',
            confidence_score=88
        )

        result = service.explain_trade('RISK-HIGH-CONF-TEST')

        assert 'high confidence' in result.risk_assessment.lower()
        assert 'low' in result.risk_assessment.lower()  # Low risk

    def test_moderate_confidence_risk_assessment(self, service, create_trade_snapshot):
        """Test risk assessment for moderate confidence trade (60-79)."""
        snapshot = create_trade_snapshot(
            trade_id='RISK-MOD-CONF-TEST',
            confidence_score=68
        )

        result = service.explain_trade('RISK-MOD-CONF-TEST')

        assert 'moderate' in result.risk_assessment.lower()

    def test_lower_confidence_risk_assessment(self, service, create_trade_snapshot):
        """Test risk assessment for lower confidence trade (40-59)."""
        snapshot = create_trade_snapshot(
            trade_id='RISK-LOWER-CONF-TEST',
            confidence_score=48
        )

        result = service.explain_trade('RISK-LOWER-CONF-TEST')

        assert 'lower confidence' in result.risk_assessment.lower()
        assert 'position size' in result.risk_assessment.lower() or 'stop' in result.risk_assessment.lower()

    def test_low_confidence_risk_assessment(self, service, create_trade_snapshot):
        """Test risk assessment for low confidence trade (<40)."""
        snapshot = create_trade_snapshot(
            trade_id='RISK-LOW-CONF-TEST',
            confidence_score=28
        )

        result = service.explain_trade('RISK-LOW-CONF-TEST')

        assert 'low confidence' in result.risk_assessment.lower()
        assert 'high risk' in result.risk_assessment.lower()


@pytest.mark.django_db
class TestSimilarTradesIntegration:
    """Integration tests for similar trade functionality."""

    def test_summarize_no_similar_trades(self, service, create_trade_snapshot):
        """Test summary when no similar trades exist."""
        snapshot = create_trade_snapshot(
            trade_id='NO-SIMILAR-TEST',
            similar_historical_trades=[]
        )

        result = service.explain_trade('NO-SIMILAR-TEST')

        assert 'No similar' in result.similar_trades_summary

    def test_summarize_similar_trades_with_outcomes(self, service, create_trade_snapshot):
        """Test summary correctly calculates win rate and average P&L."""
        snapshot = create_trade_snapshot(
            trade_id='SIMILAR-OUTCOMES-TEST',
            similar_historical_trades=[
                {'trade_id': 'HIST-1', 'outcome': 'profit', 'pnl_pct': 8.0},
                {'trade_id': 'HIST-2', 'outcome': 'profit', 'pnl_pct': 4.0},
                {'trade_id': 'HIST-3', 'outcome': 'profit', 'pnl_pct': 2.0},
                {'trade_id': 'HIST-4', 'outcome': 'loss', 'pnl_pct': -3.0}
            ]
        )

        result = service.explain_trade('SIMILAR-OUTCOMES-TEST')

        assert '4' in result.similar_trades_summary  # 4 similar trades
        assert '75%' in result.similar_trades_summary  # 3/4 win rate
        assert '+' in result.similar_trades_summary  # Positive average P&L

    def test_find_similar_trades_not_found(self, service):
        """Test finding similar trades when target trade doesn't exist."""
        result = service.find_similar_trades('NONEXISTENT-TRADE-ID')
        assert result == []

    def test_find_similar_trades_with_real_data(self, service, create_trade_snapshot):
        """Test finding similar trades using real database records."""
        # Create target trade
        target = create_trade_snapshot(
            trade_id='TARGET-SIMILAR-TEST',
            symbol='AAPL',
            direction='buy',
            signals_at_entry={
                'rsi': {'value': 28, 'triggered': True},
                'volume': {'ratio': 2.0, 'triggered': True},
                'macd': {'crossover': False},
                'price_action': {'change_pct': -3.0}
            }
        )

        # Create historical trades with outcomes
        hist1 = create_trade_snapshot(
            trade_id='HIST-SIM-001',
            symbol='AAPL',
            direction='buy',
            outcome='profit',
            pnl_percent=5.0,
            signals_at_entry={
                'rsi': {'value': 30, 'triggered': True},
                'volume': {'ratio': 2.2, 'triggered': True},
                'macd': {'crossover': False},
                'price_action': {'change_pct': -2.5}
            }
        )

        hist2 = create_trade_snapshot(
            trade_id='HIST-SIM-002',
            symbol='MSFT',
            direction='buy',
            outcome='loss',
            pnl_percent=-2.0,
            signals_at_entry={
                'rsi': {'value': 25, 'triggered': True},
                'volume': {'ratio': 1.8, 'triggered': True},
                'macd': {'crossover': False},
                'price_action': {'change_pct': -4.0}
            }
        )

        # Find similar trades
        result = service.find_similar_trades('TARGET-SIMILAR-TEST', min_similarity=0.5)

        # Should find at least one similar trade
        assert len(result) >= 1
        for trade in result:
            assert isinstance(trade, SimilarTrade)
            assert trade.similarity_score >= 0.5

    def test_find_matched_signals_between_trades(self, service):
        """Test finding matched signals between two signal sets."""
        signals1 = {
            'rsi': {'value': 28, 'triggered': True},
            'macd': {'histogram': 0.5, 'triggered': False},
            'volume': {'ratio': 2.0, 'triggered': True}
        }
        signals2 = {
            'rsi': {'value': 30, 'triggered': True},
            'macd': {'histogram': -0.3, 'triggered': False},
            'volume': {'ratio': 1.5, 'triggered': False}  # Different triggered state
        }

        matched = service._find_matched_signals(signals1, signals2)

        assert 'rsi' in matched  # Both triggered=True
        assert 'macd' in matched  # Both triggered=False
        assert 'volume' not in matched  # Different triggered states


@pytest.mark.django_db
class TestVisualizationDataIntegration:
    """Integration tests for visualization data generation."""

    def test_visualization_data_not_found(self, service):
        """Test visualization returns None when trade not found."""
        result = service.get_signal_visualization_data('NONEXISTENT')
        assert result is None

    def test_visualization_data_complete(self, service, create_trade_snapshot):
        """Test complete visualization data generation."""
        snapshot = create_trade_snapshot(
            trade_id='VIZ-COMPLETE-TEST',
            confidence_score=75,
            signals_at_entry={
                'rsi': {'value': 35, 'threshold': 30, 'triggered': False},
                'macd': {'value': 0.2, 'signal': 0.1, 'histogram': 0.1, 'crossover': False},
                'volume': {'current': 1500000, 'average': 1000000, 'ratio': 1.5},
                'price_action': {'change_pct': 1.5, 'from_sma20': 2.0, 'from_sma50': 3.0}
            }
        )

        result = service.get_signal_visualization_data('VIZ-COMPLETE-TEST')

        assert result is not None
        assert isinstance(result, VisualizationData)

        # Check RSI gauge
        assert result.rsi_gauge is not None
        assert result.rsi_gauge['value'] == 35
        assert len(result.rsi_gauge['zones']) == 3

        # Check MACD chart
        assert result.macd_chart is not None
        assert result.macd_chart['histogram'] == 0.1

        # Check volume bar
        assert result.volume_bar is not None
        assert result.volume_bar['ratio'] == 1.5

        # Check price chart
        assert result.price_chart is not None

        # Check confidence meter
        assert result.confidence_meter is not None
        assert result.confidence_meter['value'] == 75

        # Check signal timeline
        assert len(result.signal_timeline) == 4

    def test_visualization_rsi_gauge_zones(self, service, create_trade_snapshot):
        """Test RSI gauge has correct zone configuration."""
        snapshot = create_trade_snapshot(
            trade_id='VIZ-RSI-ZONES-TEST',
            signals_at_entry={
                'rsi': {'value': 28, 'threshold': 30, 'triggered': True}
            }
        )

        result = service.get_signal_visualization_data('VIZ-RSI-ZONES-TEST')

        zones = result.rsi_gauge['zones']
        assert zones[0]['min'] == 0
        assert zones[0]['max'] == 30
        assert zones[0]['label'] == 'Oversold'
        assert zones[1]['min'] == 30
        assert zones[1]['max'] == 70
        assert zones[1]['label'] == 'Neutral'
        assert zones[2]['min'] == 70
        assert zones[2]['max'] == 100
        assert zones[2]['label'] == 'Overbought'

    def test_visualization_macd_histogram_color(self, service, create_trade_snapshot):
        """Test MACD histogram color based on value."""
        # Positive histogram
        snapshot1 = create_trade_snapshot(
            trade_id='VIZ-MACD-POS-TEST',
            signals_at_entry={
                'macd': {'value': 0.5, 'signal': 0.3, 'histogram': 0.2}
            }
        )

        result1 = service.get_signal_visualization_data('VIZ-MACD-POS-TEST')
        assert result1.macd_chart['histogram_color'] == '#2dce89'  # Green

        # Negative histogram
        snapshot2 = create_trade_snapshot(
            trade_id='VIZ-MACD-NEG-TEST',
            signals_at_entry={
                'macd': {'value': -0.5, 'signal': -0.3, 'histogram': -0.2}
            }
        )

        result2 = service.get_signal_visualization_data('VIZ-MACD-NEG-TEST')
        assert result2.macd_chart['histogram_color'] == '#f5365c'  # Red

    def test_confidence_color_high(self, service):
        """Test confidence color for high confidence."""
        color = service._get_confidence_color(80)
        assert color == '#2dce89'  # Green

    def test_confidence_color_medium(self, service):
        """Test confidence color for medium confidence."""
        color = service._get_confidence_color(55)
        assert color == '#fb6340'  # Orange

    def test_confidence_color_low(self, service):
        """Test confidence color for low confidence."""
        color = service._get_confidence_color(30)
        assert color == '#f5365c'  # Red

    def test_confidence_labels(self, service):
        """Test confidence labels at various levels."""
        assert service._get_confidence_label(90) == 'Very High'
        assert service._get_confidence_label(75) == 'High'
        assert service._get_confidence_label(65) == 'Moderate'
        assert service._get_confidence_label(55) == 'Fair'
        assert service._get_confidence_label(40) == 'Low'


class TestSignalImpactDetermination:
    """Test signal impact determination logic."""

    def test_determine_impact_rsi_bullish(self, service):
        """Test RSI impact when oversold (bullish)."""
        assert service._determine_impact('rsi', {'value': 25}) == 'bullish'

    def test_determine_impact_rsi_bearish(self, service):
        """Test RSI impact when overbought (bearish)."""
        assert service._determine_impact('rsi', {'value': 78}) == 'bearish'

    def test_determine_impact_rsi_neutral(self, service):
        """Test RSI impact when neutral."""
        assert service._determine_impact('rsi', {'value': 50}) == 'neutral'

    def test_determine_impact_macd_bullish(self, service):
        """Test MACD impact when histogram positive."""
        assert service._determine_impact('macd', {'histogram': 0.5}) == 'bullish'

    def test_determine_impact_macd_bearish(self, service):
        """Test MACD impact when histogram negative."""
        assert service._determine_impact('macd', {'histogram': -0.5}) == 'bearish'

    def test_determine_impact_price_action_bullish(self, service):
        """Test price action impact when positive change."""
        assert service._determine_impact('price_action', {'change_pct': 3.0}) == 'bullish'

    def test_determine_impact_price_action_bearish(self, service):
        """Test price action impact when negative change."""
        assert service._determine_impact('price_action', {'change_pct': -3.0}) == 'bearish'

    def test_determine_impact_price_action_neutral(self, service):
        """Test price action impact when no change."""
        assert service._determine_impact('price_action', {'change_pct': 0.0}) == 'neutral'

    def test_determine_impact_volume_neutral(self, service):
        """Test volume is always neutral."""
        assert service._determine_impact('volume', {'ratio': 3.0}) == 'neutral'

    def test_determine_impact_unknown_signal(self, service):
        """Test unknown signals default to neutral."""
        assert service._determine_impact('unknown_signal', {}) == 'neutral'


class TestSignalExplanationHelpers:
    """Test helper methods for signal explanation generation."""

    def test_generate_signal_explanations_empty_dict(self, service):
        """Test with empty signals dictionary."""
        result = service._generate_signal_explanations({})
        assert result == []

    def test_generate_signal_explanations_filters_non_dict(self, service):
        """Test that non-dict signal data is filtered out."""
        signals = {
            'rsi': 'not a dict',  # Invalid
            'macd': None,  # Invalid
            'volume': {'value': 100, 'ratio': 1.5}  # Valid
        }

        result = service._generate_signal_explanations(signals)

        # Should only include the valid volume signal
        assert len(result) == 1
        assert result[0].signal_name == 'volume'

    def test_explain_single_signal_handles_missing_values(self, service):
        """Test single signal explanation with missing optional values."""
        # RSI without threshold
        result = service._explain_single_signal('rsi', {'value': 25})
        assert result is not None
        assert result.threshold is None

        # MACD without optional fields
        result = service._explain_single_signal('macd', {})
        assert result is not None

    def test_explain_generic_unknown_signal(self, service):
        """Test explanation for unknown signal types."""
        data = {'value': 42, 'triggered': True, 'custom_field': 'test'}
        result = service._explain_single_signal('custom_signal', data)

        assert result is not None
        assert result.signal_name == 'custom_signal'
        assert 'value=42' in result.description
        assert result.impact == 'bullish'  # triggered=True means bullish for generic


@pytest.mark.django_db
class TestEdgeCasesIntegration:
    """Test edge cases and error conditions with real database."""

    def test_trade_with_empty_signals(self, service, create_trade_snapshot):
        """Test explanation with empty signals dictionary."""
        snapshot = create_trade_snapshot(
            trade_id='EMPTY-SIGNALS-TEST',
            signals_at_entry={}
        )

        result = service.explain_trade('EMPTY-SIGNALS-TEST')

        assert result is not None
        assert result.signal_explanations == []
        assert result.key_factors == []
        assert result.summary is not None

    def test_trade_with_partial_signal_data(self, service, create_trade_snapshot):
        """Test explanation with partial signal data."""
        snapshot = create_trade_snapshot(
            trade_id='PARTIAL-SIGNALS-TEST',
            signals_at_entry={
                'rsi': {'value': 30},  # Missing threshold, triggered
                'macd': {},  # Empty dict
            }
        )

        result = service.explain_trade('PARTIAL-SIGNALS-TEST')

        assert result is not None
        # Should handle gracefully without crashing

    def test_trade_with_very_high_confidence(self, service, create_trade_snapshot):
        """Test with maximum confidence score."""
        snapshot = create_trade_snapshot(
            trade_id='MAX-CONF-TEST',
            confidence_score=100
        )

        result = service.explain_trade('MAX-CONF-TEST')

        assert result.confidence_score == 100
        assert 'high confidence' in result.risk_assessment.lower()

    def test_trade_with_zero_confidence(self, service, create_trade_snapshot):
        """Test with minimum confidence score."""
        snapshot = create_trade_snapshot(
            trade_id='MIN-CONF-TEST',
            confidence_score=0
        )

        result = service.explain_trade('MIN-CONF-TEST')

        assert result.confidence_score == 0
        assert 'low confidence' in result.risk_assessment.lower()

    def test_trade_with_extreme_rsi_values(self, service, create_trade_snapshot):
        """Test with extreme RSI values."""
        # Very low RSI
        snapshot1 = create_trade_snapshot(
            trade_id='EXTREME-RSI-LOW-TEST',
            signals_at_entry={'rsi': {'value': 5, 'threshold': 30, 'triggered': True}}
        )

        result1 = service.explain_trade('EXTREME-RSI-LOW-TEST')
        rsi1 = next(e for e in result1.signal_explanations if e.signal_name == 'rsi')
        assert rsi1.impact == 'bullish'

        # Very high RSI
        snapshot2 = create_trade_snapshot(
            trade_id='EXTREME-RSI-HIGH-TEST',
            signals_at_entry={'rsi': {'value': 95, 'threshold': 70, 'triggered': True}}
        )

        result2 = service.explain_trade('EXTREME-RSI-HIGH-TEST')
        rsi2 = next(e for e in result2.signal_explanations if e.signal_name == 'rsi')
        assert rsi2.impact == 'bearish'

    def test_trade_with_large_volume_ratio(self, service, create_trade_snapshot):
        """Test with very large volume ratio."""
        snapshot = create_trade_snapshot(
            trade_id='LARGE-VOL-TEST',
            signals_at_entry={
                'volume': {'current': 50000000, 'average': 1000000, 'ratio': 50.0, 'triggered': True}
            }
        )

        result = service.explain_trade('LARGE-VOL-TEST')

        vol_explanation = next(
            e for e in result.signal_explanations if e.signal_name == 'volume'
        )
        assert '50.0x' in vol_explanation.description

    def test_similar_trades_with_no_historical_data(self, service, create_trade_snapshot):
        """Test finding similar trades when no historical trades exist."""
        snapshot = create_trade_snapshot(
            trade_id='NO-HISTORY-TEST',
            signals_at_entry={'rsi': {'value': 28, 'triggered': True}}
        )

        # Delete all other snapshots except the target
        TradeSignalSnapshot.objects.exclude(trade_id='NO-HISTORY-TEST').delete()

        result = service.find_similar_trades('NO-HISTORY-TEST')

        # Should return empty list without errors
        assert result == []
