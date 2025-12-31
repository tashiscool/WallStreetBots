"""Comprehensive tests for TradeReasoningService.

Tests cover all public methods, edge cases, and error handling.
"""

import pytest
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock, patch, MagicMock

from django.contrib.auth.models import User
from django.utils import timezone

from backend.auth0login.services.trade_reasoning import (
    TradeReasoningService,
    get_trade_reasoning_service,
    EXIT_TRIGGERS
)


@pytest.fixture
def service():
    """Create a TradeReasoningService instance."""
    return TradeReasoningService()


@pytest.fixture
def user():
    """Create a test user."""
    return User.objects.create_user(username='testuser', email='test@example.com')


@pytest.fixture
def sample_signals():
    """Sample signals for testing."""
    return {
        'rsi': {'value': 28, 'threshold': 30, 'met': True},
        'volume': {'value': 2.0, 'threshold': 1.5, 'met': True},
        'macd': {'value': 0.5, 'threshold': 0, 'met': False}
    }


class TestServiceInit:
    """Test service initialization."""

    def test_init(self):
        """Test service initializes correctly."""
        service = TradeReasoningService()
        assert service is not None
        assert service.logger is not None

    def test_get_service_factory(self):
        """Test factory function."""
        service = get_trade_reasoning_service()
        assert isinstance(service, TradeReasoningService)


class TestCaptureEntryReasoning:
    """Test capture_entry_reasoning method."""

    def test_capture_entry_reasoning_basic(self, service, sample_signals):
        """Test basic entry reasoning capture."""
        result = service.capture_entry_reasoning(
            signals=sample_signals,
            confidence=85,
            summary="RSI + Volume signal"
        )

        assert result['summary'] == "RSI + Volume signal"
        assert result['confidence'] == 85
        assert 'signals' in result
        assert 'timestamp' in result
        assert 'market_context' in result

    def test_capture_entry_reasoning_auto_summary(self, service, sample_signals):
        """Test auto-generated summary."""
        result = service.capture_entry_reasoning(
            signals=sample_signals,
            confidence=70
        )

        assert result['summary'] is not None
        assert len(result['summary']) > 0

    def test_capture_entry_reasoning_with_market_context(self, service, sample_signals):
        """Test with market context."""
        market_ctx = {'vix': 15.5, 'spy_trend': 'bullish'}

        result = service.capture_entry_reasoning(
            signals=sample_signals,
            confidence=80,
            market_context=market_ctx
        )

        assert result['market_context'] == market_ctx

    def test_generate_entry_summary_single_signal(self, service):
        """Test summary with single triggered signal."""
        signals = {'rsi': {'met': True}}
        summary = service._generate_entry_summary(signals)

        assert 'RSI' in summary

    def test_generate_entry_summary_multiple_signals(self, service):
        """Test summary with multiple signals."""
        signals = {
            'rsi': {'met': True},
            'volume': {'triggered': True},
            'macd': {'met': True}
        }
        summary = service._generate_entry_summary(signals)

        assert '+' in summary or 'RSI' in summary

    def test_generate_entry_summary_many_signals(self, service):
        """Test summary with many signals."""
        signals = {f'signal_{i}': {'met': True} for i in range(5)}
        summary = service._generate_entry_summary(signals)

        assert 'signals aligned' in summary

    def test_generate_entry_summary_no_signals(self, service):
        """Test summary with no triggered signals."""
        signals = {'rsi': {'met': False}, 'macd': {'met': False}}
        summary = service._generate_entry_summary(signals)

        assert 'No specific signals' in summary


class TestCaptureExitReasoning:
    """Test capture_exit_reasoning method."""

    def test_capture_exit_reasoning_basic(self, service):
        """Test basic exit reasoning capture."""
        entry_time = timezone.now() - timedelta(hours=5)

        result = service.capture_exit_reasoning(
            trigger='take_profit',
            entry_timestamp=entry_time
        )

        assert result['trigger'] == 'take_profit'
        assert 'held_duration' in result
        assert result['held_duration_hours'] > 0

    def test_capture_exit_reasoning_all_triggers(self, service):
        """Test all exit trigger types."""
        for trigger in EXIT_TRIGGERS.keys():
            result = service.capture_exit_reasoning(trigger=trigger)
            assert result['trigger'] == trigger
            assert result['summary'] is not None

    def test_capture_exit_reasoning_custom_summary(self, service):
        """Test with custom summary."""
        result = service.capture_exit_reasoning(
            trigger='stop_loss',
            summary='Custom exit reason'
        )

        assert result['summary'] == 'Custom exit reason'

    def test_capture_exit_reasoning_with_signals(self, service):
        """Test with exit signals."""
        exit_signals = {'rsi': {'value': 75}}

        result = service.capture_exit_reasoning(
            trigger='signal',
            signals_at_exit=exit_signals
        )

        assert result['signals_at_exit'] == exit_signals

    def test_format_duration_minutes(self, service):
        """Test duration formatting for minutes."""
        duration = service._format_duration(0.5)
        assert 'minutes' in duration

    def test_format_duration_hours(self, service):
        """Test duration formatting for hours."""
        duration = service._format_duration(3.5)
        assert 'hours' in duration

    def test_format_duration_one_day(self, service):
        """Test duration formatting for one day."""
        duration = service._format_duration(30)
        assert 'day' in duration

    def test_format_duration_multiple_days(self, service):
        """Test duration formatting for multiple days."""
        duration = service._format_duration(72)
        assert 'days' in duration


class TestCreateTradeWithReasoning:
    """Test create_trade_with_reasoning method."""

    @patch('backend.auth0login.services.trade_reasoning.TradeSignalSnapshot')
    def test_create_trade_basic(self, mock_model, service, sample_signals):
        """Test basic trade creation."""
        mock_snapshot = Mock()
        mock_model.objects.create.return_value = mock_snapshot

        result = service.create_trade_with_reasoning(
            trade_id='TEST123',
            symbol='AAPL',
            direction='buy',
            entry_price=150.00,
            quantity=100,
            strategy_name='wsb_dip_bot',
            signals=sample_signals,
            confidence=85
        )

        assert result == mock_snapshot
        mock_model.objects.create.assert_called_once()

    @patch('backend.auth0login.services.trade_reasoning.TradeSignalSnapshot')
    def test_create_trade_with_all_params(self, mock_model, service, sample_signals):
        """Test trade creation with all parameters."""
        mock_snapshot = Mock()
        mock_model.objects.create.return_value = mock_snapshot
        market_ctx = {'vix': 15.5}
        order = Mock()

        result = service.create_trade_with_reasoning(
            trade_id='TEST123',
            symbol='AAPL',
            direction='buy',
            entry_price=150.00,
            quantity=100,
            strategy_name='wsb_dip_bot',
            signals=sample_signals,
            confidence=85,
            market_context=market_ctx,
            explanation='Test explanation',
            order=order
        )

        call_kwargs = mock_model.objects.create.call_args[1]
        assert call_kwargs['market_context'] is not None or 'market_context' not in call_kwargs


class TestRecordExitWithReasoning:
    """Test record_exit_with_reasoning method."""

    @patch('backend.auth0login.services.trade_reasoning.TradeSignalSnapshot')
    def test_record_exit_not_found(self, mock_model, service):
        """Test recording exit when trade not found."""
        mock_model.objects.get.side_effect = mock_model.DoesNotExist

        result = service.record_exit_with_reasoning(
            trade_id='NONEXISTENT',
            exit_price=155.00,
            trigger='take_profit',
            pnl_percent=3.33
        )

        assert result is None

    @patch('backend.auth0login.services.trade_reasoning.TradeSignalSnapshot')
    def test_record_exit_success(self, mock_model, service):
        """Test successful exit recording."""
        mock_snapshot = Mock()
        mock_snapshot.created_at = timezone.now() - timedelta(hours=2)
        mock_model.objects.get.return_value = mock_snapshot

        result = service.record_exit_with_reasoning(
            trade_id='TEST123',
            exit_price=155.00,
            trigger='take_profit',
            pnl_amount=500.00,
            pnl_percent=3.33
        )

        assert result == mock_snapshot
        mock_snapshot.record_exit.assert_called_once()


class TestAnalyzeClosedTrade:
    """Test analyze_closed_trade method."""

    @patch('backend.auth0login.services.trade_reasoning.TradeSignalSnapshot')
    def test_analyze_closed_trade_not_found(self, mock_model, service):
        """Test analysis when trade not found."""
        mock_model.objects.get.side_effect = mock_model.DoesNotExist

        result = service.analyze_closed_trade('NONEXISTENT')

        assert result is None

    @patch('backend.auth0login.services.trade_reasoning.TradeSignalSnapshot')
    def test_analyze_closed_trade_still_open(self, mock_model, service):
        """Test analysis when trade still open."""
        mock_snapshot = Mock()
        mock_snapshot.exit_price = None
        mock_snapshot.exit_timestamp = None
        mock_model.objects.get.return_value = mock_snapshot

        result = service.analyze_closed_trade('TEST123')

        assert result is None

    @patch('backend.auth0login.services.trade_reasoning.TradeSignalSnapshot')
    def test_analyze_closed_trade_success(self, mock_model, service):
        """Test successful trade analysis."""
        mock_snapshot = Mock()
        mock_snapshot.exit_price = Decimal('155.00')
        mock_snapshot.exit_timestamp = timezone.now()
        mock_snapshot.pnl_amount = Decimal('500.00')
        mock_snapshot.pnl_percent = Decimal('3.33')
        mock_snapshot.similar_historical_trades = []
        mock_model.objects.get.return_value = mock_snapshot

        result = service.analyze_closed_trade('TEST123')

        assert result is not None
        assert 'pnl' in result
        assert 'pnl_pct' in result
        assert 'timing_score' in result
        assert 'notes' in result
        mock_snapshot.set_outcome_analysis.assert_called_once()

    def test_analyze_timing_excellent(self, service):
        """Test timing analysis for excellent trade."""
        snapshot = Mock()
        snapshot.exit_timestamp = timezone.now()
        snapshot.pnl_percent = Decimal('6.0')

        result = service._analyze_timing(snapshot)

        assert result['score'] == 'excellent'

    def test_analyze_timing_poor(self, service):
        """Test timing analysis for poor trade."""
        snapshot = Mock()
        snapshot.exit_timestamp = timezone.now()
        snapshot.pnl_percent = Decimal('-1.5')

        result = service._analyze_timing(snapshot)

        assert result['score'] == 'poor'

    def test_generate_analysis_notes_strong_profit(self, service):
        """Test analysis notes for strong profit."""
        analysis = {'pnl_pct': 6.0, 'vs_hold': None}
        timing = {'score': 'excellent'}

        notes = service._generate_analysis_notes(analysis, timing)

        assert 'Strong profit' in notes

    def test_generate_analysis_notes_significant_loss(self, service):
        """Test analysis notes for significant loss."""
        analysis = {'pnl_pct': -5.0, 'vs_hold': None}
        timing = {'score': 'bad'}

        notes = service._generate_analysis_notes(analysis, timing)

        assert 'Significant loss' in notes


class TestFindSimilarTrades:
    """Test finding similar trades."""

    @patch('backend.auth0login.services.trade_reasoning.TradeSignalSnapshot')
    def test_find_similar_trades(self, mock_model, service, sample_signals):
        """Test finding similar trades."""
        # Mock recent trades
        recent_trade = Mock()
        recent_trade.signals_at_entry = {'rsi': {'met': True}}
        recent_trade.to_dict.return_value = {
            'trade_id': 'SIMILAR1',
            'outcome': 'profit',
            'pnl_percent': 5.0
        }

        mock_qs = Mock()
        mock_qs.order_by.return_value = [recent_trade]
        mock_model.objects.filter.return_value = mock_qs

        result = service._find_similar_trades(sample_signals, 'wsb_dip_bot')

        assert isinstance(result, list)

    def test_get_similar_trades_stats_empty(self, service):
        """Test stats with no similar trades."""
        snapshot = Mock()
        snapshot.similar_historical_trades = []

        result = service._get_similar_trades_stats(snapshot)

        assert result['count'] == 0
        assert result['avg_pnl'] == 0
        assert result['win_rate'] == 0

    def test_get_similar_trades_stats_with_data(self, service):
        """Test stats with similar trades."""
        snapshot = Mock()
        snapshot.similar_historical_trades = [
            {'pnl_pct': 5.0, 'outcome': 'profit'},
            {'pnl_pct': -2.0, 'outcome': 'loss'},
            {'pnl_pct': 3.0, 'outcome': 'profit'}
        ]

        result = service._get_similar_trades_stats(snapshot)

        assert result['count'] == 3
        assert result['win_rate'] > 0
        assert 'avg_pnl' in result


class TestGetTradeWithReasoning:
    """Test getting trades with reasoning."""

    @patch('backend.auth0login.services.trade_reasoning.TradeSignalSnapshot')
    def test_get_trade_with_full_reasoning_found(self, mock_model, service):
        """Test getting trade when found."""
        mock_snapshot = Mock()
        mock_snapshot.to_dict_with_reasoning.return_value = {'trade_id': 'TEST123'}
        mock_model.objects.get.return_value = mock_snapshot

        result = service.get_trade_with_full_reasoning('TEST123')

        assert result is not None
        assert result['trade_id'] == 'TEST123'

    @patch('backend.auth0login.services.trade_reasoning.TradeSignalSnapshot')
    def test_get_trade_with_full_reasoning_not_found(self, mock_model, service):
        """Test getting trade when not found."""
        mock_model.objects.get.side_effect = mock_model.DoesNotExist

        result = service.get_trade_with_full_reasoning('NONEXISTENT')

        assert result is None


class TestGetTradesByStrategy:
    """Test getting trades by strategy."""

    @patch('backend.auth0login.services.trade_reasoning.TradeSignalSnapshot')
    def test_get_trades_by_strategy_include_open(self, mock_model, service):
        """Test getting trades including open."""
        mock_snapshot = Mock()
        mock_snapshot.to_dict_with_reasoning.return_value = {'trade_id': 'TEST123'}

        mock_qs = Mock()
        mock_qs.order_by.return_value = [mock_snapshot]
        mock_model.objects.filter.return_value = mock_qs

        result = service.get_trades_by_strategy('wsb_dip_bot', include_open=True)

        assert isinstance(result, list)
        mock_qs.exclude.assert_not_called()

    @patch('backend.auth0login.services.trade_reasoning.TradeSignalSnapshot')
    def test_get_trades_by_strategy_exclude_open(self, mock_model, service):
        """Test getting trades excluding open."""
        mock_snapshot = Mock()
        mock_snapshot.to_dict_with_reasoning.return_value = {'trade_id': 'TEST123'}

        mock_qs = Mock()
        mock_qs.exclude.return_value.exclude.return_value.order_by.return_value = [mock_snapshot]
        mock_model.objects.filter.return_value = mock_qs

        result = service.get_trades_by_strategy('wsb_dip_bot', include_open=False)

        assert isinstance(result, list)


class TestGetReasoningStats:
    """Test reasoning statistics."""

    @patch('backend.auth0login.services.trade_reasoning.TradeSignalSnapshot')
    def test_get_reasoning_stats(self, mock_model, service):
        """Test getting reasoning stats."""
        mock_qs = Mock()
        mock_qs.filter.return_value = mock_qs
        mock_qs.exclude.return_value = mock_qs
        mock_qs.count.return_value = 10
        mock_qs.aggregate.return_value = {'avg': 75.0}
        mock_model.objects.filter.return_value = mock_qs

        result = service.get_reasoning_stats('wsb_dip_bot', days=30)

        assert 'total_trades' in result
        assert 'reasoning_coverage' in result
        assert 'avg_confidence' in result
        assert 'confidence_breakdown' in result


class TestHelperMethods:
    """Test helper methods."""

    def test_format_signals_for_storage(self, service):
        """Test signal formatting."""
        signals = {
            'rsi': {'triggered': True, 'value': 28},
            'macd': {'met': True, 'histogram': 0.5}
        }

        result = service._format_signals_for_storage(signals)

        assert result['rsi']['met'] is True
        assert 'macd' in result

    def test_generate_explanation(self, service, sample_signals):
        """Test explanation generation."""
        explanation = service._generate_explanation(sample_signals, 85, 'wsb_dip_bot')

        assert 'wsb_dip_bot' in explanation
        assert '85%' in explanation

    def test_calculate_hold_return_with_prices(self, service):
        """Test hold return calculation."""
        historical_prices = {
            '2024-01-15': 150.00,
            '2024-01-20': 155.00
        }

        entry_time = datetime(2024, 1, 15)
        exit_time = datetime(2024, 1, 20)

        result = service._calculate_hold_return(
            'AAPL', entry_time, exit_time, historical_prices
        )

        assert result > 0

    def test_calculate_hold_return_no_prices(self, service):
        """Test hold return with missing prices."""
        result = service._calculate_hold_return(
            'AAPL',
            datetime(2024, 1, 15),
            datetime(2024, 1, 20),
            {}
        )

        assert result == 0
