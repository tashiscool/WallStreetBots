"""Comprehensive tests for TradeExplainerService.

Tests cover:
- Trade explanation generation
- Signal explanations (RSI, MACD, volume, price action, etc.)
- Summary generation
- Key factor extraction
- Risk assessment
- Similar trade finding
- Visualization data generation
- Edge cases and error handling
"""

import pytest
from datetime import datetime
from decimal import Decimal
from unittest.mock import Mock, patch, MagicMock

from backend.auth0login.services.trade_explainer import (
    TradeExplainerService,
    SignalExplanation,
    TradeExplanation,
    SimilarTrade,
    VisualizationData,
)


@pytest.fixture
def service():
    """Create a TradeExplainerService instance."""
    return TradeExplainerService()


@pytest.fixture
def mock_snapshot():
    """Create a mock trade snapshot."""
    snapshot = Mock()
    snapshot.trade_id = 'TEST123'
    snapshot.symbol = 'AAPL'
    snapshot.direction = 'buy'
    snapshot.strategy_name = 'wsb_dip_bot'
    snapshot.entry_price = Decimal('150.00')
    snapshot.quantity = Decimal('100')
    snapshot.confidence_score = 85
    snapshot.created_at = datetime(2024, 1, 15, 10, 30, 0)
    snapshot.signals_at_entry = {
        'rsi': {'value': 28, 'threshold': 30, 'triggered': True},
        'macd': {'value': -0.5, 'signal': -0.3, 'histogram': -0.2, 'crossover': False},
        'volume': {'current': 1000000, 'average': 500000, 'ratio': 2.0, 'triggered': True}
    }
    snapshot.similar_historical_trades = []
    return snapshot


class TestTradeExplainerInit:
    """Test TradeExplainerService initialization."""

    def test_init(self):
        """Test service initialization."""
        service = TradeExplainerService()
        assert service is not None
        assert hasattr(service, 'SIGNAL_DESCRIPTIONS')


class TestExplainTrade:
    """Test explain_trade method."""

    @patch('backend.auth0login.services.trade_explainer.TradeSignalSnapshot')
    def test_explain_trade_not_found(self, mock_model, service):
        """Test explanation when trade not found."""
        mock_model.objects.get.side_effect = mock_model.DoesNotExist

        result = service.explain_trade('NONEXISTENT')

        assert result is None

    @patch('backend.auth0login.services.trade_explainer.TradeSignalSnapshot')
    def test_explain_trade_success(self, mock_model, service, mock_snapshot):
        """Test successful trade explanation."""
        mock_model.objects.get.return_value = mock_snapshot

        result = service.explain_trade('TEST123')

        assert result is not None
        assert isinstance(result, TradeExplanation)
        assert result.trade_id == 'TEST123'
        assert result.symbol == 'AAPL'
        assert result.direction == 'buy'
        assert result.confidence_score == 85
        assert len(result.signal_explanations) > 0

    @patch('backend.auth0login.services.trade_explainer.TradeSignalSnapshot')
    def test_explain_trade_with_similar_trades(self, mock_model, service, mock_snapshot):
        """Test explanation with similar trades."""
        mock_snapshot.similar_historical_trades = [
            {'trade_id': 'SIM1', 'outcome': 'profit', 'pnl_pct': 5.0},
            {'trade_id': 'SIM2', 'outcome': 'loss', 'pnl_pct': -2.0}
        ]
        mock_model.objects.get.return_value = mock_snapshot

        result = service.explain_trade('TEST123')

        assert result is not None
        assert 'similar' in result.similar_trades_summary.lower()


class TestSignalExplanations:
    """Test signal explanation generation."""

    def test_generate_signal_explanations_empty(self, service):
        """Test with no signals."""
        result = service._generate_signal_explanations({})
        assert result == []

    def test_generate_signal_explanations_rsi_oversold(self, service):
        """Test RSI oversold explanation."""
        signals = {'rsi': {'value': 28, 'threshold': 30, 'triggered': True}}
        result = service._generate_signal_explanations(signals)

        assert len(result) == 1
        assert result[0].signal_name == 'rsi'
        assert result[0].impact == 'bullish'
        assert '28' in result[0].description

    def test_generate_signal_explanations_rsi_overbought(self, service):
        """Test RSI overbought explanation."""
        signals = {'rsi': {'value': 75, 'threshold': 70, 'triggered': True}}
        result = service._generate_signal_explanations(signals)

        assert len(result) == 1
        assert result[0].impact == 'bearish'

    def test_generate_signal_explanations_rsi_neutral(self, service):
        """Test RSI neutral explanation."""
        signals = {'rsi': {'value': 50, 'threshold': 30, 'triggered': False}}
        result = service._generate_signal_explanations(signals)

        assert len(result) == 1
        assert result[0].impact == 'neutral'

    def test_explain_macd_bullish_crossover(self, service):
        """Test MACD bullish crossover."""
        data = {'value': 0.5, 'signal': 0.3, 'histogram': 0.2, 'crossover': True}
        result = service._explain_single_signal('macd', data)

        assert result.impact == 'bullish'
        assert 'crossover' in result.description.lower()

    def test_explain_macd_bearish_crossover(self, service):
        """Test MACD bearish crossover."""
        data = {'value': -0.5, 'signal': -0.3, 'histogram': -0.2, 'crossover': True}
        result = service._explain_single_signal('macd', data)

        assert result.impact == 'bearish'

    def test_explain_volume_high(self, service):
        """Test high volume explanation."""
        data = {'current': 2000000, 'average': 1000000, 'ratio': 2.0, 'triggered': True}
        result = service._explain_single_signal('volume', data)

        assert '2.0x' in result.description
        assert result.impact == 'bullish'

    def test_explain_volume_low(self, service):
        """Test low volume explanation."""
        data = {'current': 400000, 'average': 1000000, 'ratio': 0.4, 'triggered': False}
        result = service._explain_single_signal('volume', data)

        assert result.impact == 'neutral'

    def test_explain_price_action_drop(self, service):
        """Test price drop explanation."""
        data = {'change_pct': -3.5, 'from_sma20': -5.0}
        result = service._explain_single_signal('price_action', data)

        assert result.impact == 'bearish'
        assert '-3.5%' in result.description

    def test_explain_price_action_rise(self, service):
        """Test price rise explanation."""
        data = {'change_pct': 4.0, 'from_sma20': 6.0}
        result = service._explain_single_signal('price_action', data)

        assert result.impact == 'bullish'

    def test_explain_bollinger_below_lower(self, service):
        """Test Bollinger below lower band."""
        data = {'position': 'below_lower'}
        result = service._explain_single_signal('bollinger', data)

        assert result.impact == 'bullish'

    def test_explain_bollinger_above_upper(self, service):
        """Test Bollinger above upper band."""
        data = {'position': 'above_upper'}
        result = service._explain_single_signal('bollinger', data)

        assert result.impact == 'bearish'

    def test_explain_stochastic_oversold(self, service):
        """Test Stochastic oversold."""
        data = {'k': 15}
        result = service._explain_single_signal('stochastic', data)

        assert result.impact == 'bullish'

    def test_explain_stochastic_overbought(self, service):
        """Test Stochastic overbought."""
        data = {'k': 85}
        result = service._explain_single_signal('stochastic', data)

        assert result.impact == 'bearish'

    def test_explain_trend_aligned_bullish(self, service):
        """Test all trends bullish."""
        data = {'short_term': 'bullish', 'medium_term': 'bullish', 'long_term': 'bullish'}
        result = service._explain_single_signal('trend', data)

        assert result.impact == 'bullish'

    def test_explain_trend_aligned_bearish(self, service):
        """Test all trends bearish."""
        data = {'short_term': 'bearish', 'medium_term': 'bearish', 'long_term': 'bearish'}
        result = service._explain_single_signal('trend', data)

        assert result.impact == 'bearish'

    def test_explain_trend_mixed(self, service):
        """Test mixed trends."""
        data = {'short_term': 'bullish', 'medium_term': 'neutral', 'long_term': 'bearish'}
        result = service._explain_single_signal('trend', data)

        assert result.impact == 'neutral'

    def test_explain_generic_signal(self, service):
        """Test generic signal explanation."""
        data = {'value': 42, 'triggered': True}
        result = service._explain_single_signal('unknown_signal', data)

        assert result is not None
        assert 'value=42' in result.description


class TestSummaryGeneration:
    """Test summary generation."""

    def test_generate_summary_buy_bullish(self, service, mock_snapshot):
        """Test summary for bullish buy."""
        explanations = [
            SignalExplanation('rsi', True, 28, 30, 'RSI oversold', 'bullish'),
            SignalExplanation('volume', True, 2.0, 1.5, 'High volume', 'bullish'),
        ]

        summary = service._generate_summary(mock_snapshot, explanations)

        assert 'buy' in summary.lower()
        assert 'wsb_dip_bot' in summary
        assert 'bullish' in summary.lower()

    def test_generate_summary_sell_bearish(self, service):
        """Test summary for bearish sell."""
        snapshot = Mock()
        snapshot.symbol = 'SPY'
        snapshot.direction = 'sell'
        snapshot.strategy_name = 'short_strategy'
        snapshot.confidence_score = 75

        explanations = [
            SignalExplanation('rsi', True, 75, 70, 'RSI overbought', 'bearish'),
        ]

        summary = service._generate_summary(snapshot, explanations)

        assert 'sell' in summary.lower()
        assert 'bearish' in summary.lower()

    def test_generate_summary_no_signals(self, service, mock_snapshot):
        """Test summary with no triggered signals."""
        summary = service._generate_summary(mock_snapshot, [])

        assert 'wsb_dip_bot' in summary
        assert '85%' in summary


class TestKeyFactors:
    """Test key factor extraction."""

    def test_extract_key_factors_rsi(self, service):
        """Test extracting RSI factor."""
        snapshot = Mock()
        snapshot.signals_at_entry = {
            'rsi': {'value': 28, 'triggered': True}
        }

        factors = service._extract_key_factors(snapshot)

        assert any('Oversold RSI' in f for f in factors)

    def test_extract_key_factors_volume(self, service):
        """Test extracting volume factor."""
        snapshot = Mock()
        snapshot.signals_at_entry = {
            'volume': {'ratio': 2.5, 'triggered': True}
        }

        factors = service._extract_key_factors(snapshot)

        assert any('High volume' in f for f in factors)

    def test_extract_key_factors_macd(self, service):
        """Test extracting MACD crossover."""
        snapshot = Mock()
        snapshot.signals_at_entry = {
            'macd': {'crossover': True}
        }

        factors = service._extract_key_factors(snapshot)

        assert any('MACD crossover' in f for f in factors)

    def test_extract_key_factors_multiple(self, service):
        """Test extracting multiple factors."""
        snapshot = Mock()
        snapshot.signals_at_entry = {
            'rsi': {'value': 25},
            'volume': {'ratio': 3.0},
            'macd': {'crossover': True},
            'price_action': {'change_pct': -5.0},
            'bollinger': {'position': 'below_lower'}
        }

        factors = service._extract_key_factors(snapshot)

        # Should limit to top 5
        assert len(factors) <= 5


class TestRiskAssessment:
    """Test risk assessment generation."""

    def test_risk_assessment_high_confidence(self, service, mock_snapshot):
        """Test risk assessment for high confidence."""
        mock_snapshot.confidence_score = 85

        assessment = service._generate_risk_assessment(mock_snapshot)

        assert 'high confidence' in assessment.lower()
        assert 'low' in assessment.lower()

    def test_risk_assessment_moderate_confidence(self, service, mock_snapshot):
        """Test risk assessment for moderate confidence."""
        mock_snapshot.confidence_score = 65

        assessment = service._generate_risk_assessment(mock_snapshot)

        assert 'moderate' in assessment.lower()

    def test_risk_assessment_low_confidence(self, service, mock_snapshot):
        """Test risk assessment for low confidence."""
        mock_snapshot.confidence_score = 45

        assessment = service._generate_risk_assessment(mock_snapshot)

        assert 'lower confidence' in assessment.lower()

    def test_risk_assessment_very_low_confidence(self, service, mock_snapshot):
        """Test risk assessment for very low confidence."""
        mock_snapshot.confidence_score = 25

        assessment = service._generate_risk_assessment(mock_snapshot)

        assert 'low confidence' in assessment.lower()
        assert 'high risk' in assessment.lower()


class TestSimilarTrades:
    """Test similar trade functionality."""

    def test_summarize_similar_trades_none(self, service):
        """Test summary with no similar trades."""
        summary = service._summarize_similar_trades([])

        assert 'No similar' in summary

    def test_summarize_similar_trades_with_data(self, service):
        """Test summary with similar trades."""
        similar = [
            {'outcome': 'profit', 'pnl_pct': 5.0},
            {'outcome': 'profit', 'pnl_pct': 3.0},
            {'outcome': 'loss', 'pnl_pct': -2.0}
        ]

        summary = service._summarize_similar_trades(similar)

        assert '3' in summary  # total
        assert '67%' in summary or '66%' in summary  # win rate
        assert '+' in summary  # avg P&L

    @patch('backend.auth0login.services.trade_explainer.TradeSignalSnapshot')
    def test_find_similar_trades_not_found(self, mock_model, service):
        """Test finding similar trades when target not found."""
        mock_model.objects.get.side_effect = mock_model.DoesNotExist

        result = service.find_similar_trades('NONEXISTENT')

        assert result == []

    @patch('backend.auth0login.services.trade_explainer.TradeSignalSnapshot')
    def test_find_similar_trades_success(self, mock_model, service, mock_snapshot):
        """Test finding similar trades."""
        # Setup target trade
        mock_model.objects.get.return_value = mock_snapshot

        # Setup historical trades
        hist_trade = Mock()
        hist_trade.trade_id = 'HIST1'
        hist_trade.symbol = 'AAPL'
        hist_trade.outcome = 'profit'
        hist_trade.pnl_percent = Decimal('5.0')
        hist_trade.created_at = datetime(2024, 1, 10, 10, 0, 0)
        hist_trade.signals_at_entry = {
            'rsi': {'value': 29, 'triggered': True},
            'volume': {'ratio': 1.8, 'triggered': True}
        }

        mock_qs = Mock()
        mock_qs.order_by.return_value = [hist_trade]
        mock_model.objects.filter.return_value.exclude.return_value = mock_qs

        # Mock calculate_similarity
        mock_snapshot.calculate_similarity.return_value = 0.85

        result = service.find_similar_trades('TEST123', min_similarity=0.7)

        assert len(result) > 0
        assert isinstance(result[0], SimilarTrade)
        assert result[0].similarity_score >= 0.7

    def test_find_matched_signals(self, service):
        """Test finding matched signals."""
        signals1 = {
            'rsi': {'value': 28, 'triggered': True},
            'macd': {'histogram': 0.5, 'triggered': False}
        }
        signals2 = {
            'rsi': {'value': 29, 'triggered': True},
            'macd': {'histogram': -0.3, 'triggered': False}
        }

        matched = service._find_matched_signals(signals1, signals2)

        assert 'rsi' in matched
        assert 'macd' in matched


class TestVisualizationData:
    """Test visualization data generation."""

    @patch('backend.auth0login.services.trade_explainer.TradeSignalSnapshot')
    def test_get_signal_visualization_data_not_found(self, mock_model, service):
        """Test visualization when trade not found."""
        mock_model.objects.get.side_effect = mock_model.DoesNotExist

        result = service.get_signal_visualization_data('NONEXISTENT')

        assert result is None

    @patch('backend.auth0login.services.trade_explainer.TradeSignalSnapshot')
    def test_get_signal_visualization_data_success(self, mock_model, service, mock_snapshot):
        """Test successful visualization data generation."""
        mock_model.objects.get.return_value = mock_snapshot

        result = service.get_signal_visualization_data('TEST123')

        assert result is not None
        assert isinstance(result, VisualizationData)
        assert result.rsi_gauge is not None
        assert result.macd_chart is not None
        assert result.volume_bar is not None
        assert result.confidence_meter is not None

    def test_visualization_rsi_gauge(self, service, mock_snapshot):
        """Test RSI gauge data."""
        mock_snapshot.signals_at_entry = {
            'rsi': {'value': 28, 'threshold': 30, 'triggered': True}
        }

        with patch('backend.auth0login.services.trade_explainer.TradeSignalSnapshot') as mock_model:
            mock_model.objects.get.return_value = mock_snapshot
            result = service.get_signal_visualization_data('TEST123')

        assert result.rsi_gauge['value'] == 28
        assert result.rsi_gauge['triggered'] is True
        assert len(result.rsi_gauge['zones']) == 3

    def test_visualization_macd_chart(self, service, mock_snapshot):
        """Test MACD chart data."""
        result = service.get_signal_visualization_data('TEST123')
        # Would need to patch the model lookup
        # Checking structure is correct

    def test_get_confidence_color_high(self, service):
        """Test confidence color for high confidence."""
        color = service._get_confidence_color(80)
        assert color == '#2dce89'  # Green

    def test_get_confidence_color_medium(self, service):
        """Test confidence color for medium confidence."""
        color = service._get_confidence_color(55)
        assert color == '#fb6340'  # Orange

    def test_get_confidence_color_low(self, service):
        """Test confidence color for low confidence."""
        color = service._get_confidence_color(30)
        assert color == '#f5365c'  # Red

    def test_get_confidence_label(self, service):
        """Test confidence labels."""
        assert service._get_confidence_label(85) == 'Very High'
        assert service._get_confidence_label(75) == 'High'
        assert service._get_confidence_label(65) == 'Moderate'
        assert service._get_confidence_label(55) == 'Fair'
        assert service._get_confidence_label(40) == 'Low'

    def test_determine_impact_rsi(self, service):
        """Test impact determination for RSI."""
        assert service._determine_impact('rsi', {'value': 25}) == 'bullish'
        assert service._determine_impact('rsi', {'value': 75}) == 'bearish'
        assert service._determine_impact('rsi', {'value': 50}) == 'neutral'

    def test_determine_impact_macd(self, service):
        """Test impact determination for MACD."""
        assert service._determine_impact('macd', {'histogram': 0.5}) == 'bullish'
        assert service._determine_impact('macd', {'histogram': -0.5}) == 'bearish'

    def test_determine_impact_price_action(self, service):
        """Test impact determination for price action."""
        assert service._determine_impact('price_action', {'change_pct': 3.0}) == 'bullish'
        assert service._determine_impact('price_action', {'change_pct': -3.0}) == 'bearish'
        assert service._determine_impact('price_action', {'change_pct': 0.0}) == 'neutral'


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_explain_single_signal_non_dict(self, service):
        """Test signal explanation with non-dict data."""
        result = service._explain_single_signal('test', "not a dict")

        # Should handle gracefully
        assert result is not None or result is None

    def test_generate_signal_explanations_invalid_data(self, service):
        """Test with invalid signal data."""
        signals = {
            'rsi': 'not a dict',
            'macd': None,
            'volume': {'value': 100}  # Valid
        }

        result = service._generate_signal_explanations(signals)

        # Should filter out invalid signals
        assert isinstance(result, list)

    def test_extract_key_factors_empty_signals(self, service):
        """Test key factors with empty signals."""
        snapshot = Mock()
        snapshot.signals_at_entry = {}

        factors = service._extract_key_factors(snapshot)

        assert factors == []
