"""Integration tests for TradeReasoningService.

Tests actual database operations, real reasoning generation, and LLM-style
output format without mocking business logic. Uses @pytest.mark.django_db
for real database transactions.
"""

import pytest
from datetime import datetime, timedelta
from decimal import Decimal

from django.utils import timezone

from backend.auth0login.services.trade_reasoning import (
    TradeReasoningService,
    get_trade_reasoning_service,
    EXIT_TRIGGERS
)
from backend.tradingbot.models.models import TradeSignalSnapshot


@pytest.fixture
def service():
    """Create a TradeReasoningService instance."""
    return TradeReasoningService()


@pytest.fixture
def sample_signals():
    """Sample signals for testing - realistic trading signals."""
    return {
        'rsi': {'value': 28, 'threshold': 30, 'met': True, 'period': 14},
        'volume': {'value': 2.5, 'threshold': 1.5, 'met': True, 'ratio': 2.5},
        'macd': {'value': 0.5, 'threshold': 0, 'met': False, 'histogram': -0.2},
        'price_action': {'change_pct': -3.2, 'from_sma20': -5.1, 'met': True}
    }


@pytest.fixture
def bearish_signals():
    """Bearish market signals for testing."""
    return {
        'rsi': {'value': 75, 'threshold': 70, 'met': True, 'period': 14},
        'macd': {'value': -0.3, 'threshold': 0, 'met': True, 'crossover': False},
        'volume': {'value': 3.0, 'threshold': 1.5, 'met': True, 'ratio': 3.0},
        'bollinger': {'position': 'above_upper', 'met': True}
    }


@pytest.fixture
def market_context():
    """Sample market context data."""
    return {
        'vix': 18.5,
        'spy_trend': 'bullish',
        'sector_performance': 1.2,
        'market_hours': True
    }


class TestServiceInitialization:
    """Test service initialization and factory function."""

    def test_service_initializes_correctly(self):
        """Test that TradeReasoningService initializes with proper attributes."""
        service = TradeReasoningService()
        assert service is not None
        assert service.logger is not None

    def test_get_service_factory_returns_service_instance(self):
        """Test factory function returns a valid service instance."""
        service = get_trade_reasoning_service()
        assert isinstance(service, TradeReasoningService)
        assert hasattr(service, 'capture_entry_reasoning')
        assert hasattr(service, 'capture_exit_reasoning')


class TestCaptureEntryReasoning:
    """Test entry reasoning capture with real logic - no mocks."""

    def test_capture_entry_reasoning_with_custom_summary(self, service, sample_signals):
        """Test entry reasoning capture with custom summary."""
        result = service.capture_entry_reasoning(
            signals=sample_signals,
            confidence=85,
            summary="RSI oversold + Volume surge detected"
        )

        # Verify structure
        assert result['summary'] == "RSI oversold + Volume surge detected"
        assert result['confidence'] == 85
        assert 'signals' in result
        assert 'timestamp' in result
        assert 'market_context' in result

        # Verify signals are properly formatted
        assert 'rsi' in result['signals']
        assert 'volume' in result['signals']
        assert result['signals']['rsi']['value'] == 28

    def test_capture_entry_reasoning_auto_generates_summary(self, service, sample_signals):
        """Test that summary is auto-generated from triggered signals."""
        result = service.capture_entry_reasoning(
            signals=sample_signals,
            confidence=70
        )

        # Summary should be auto-generated and non-empty
        assert result['summary'] is not None
        assert len(result['summary']) > 0

        # Should mention triggered signals (RSI, VOLUME, PRICE ACTION are met)
        summary_upper = result['summary'].upper()
        assert 'RSI' in summary_upper or 'VOLUME' in summary_upper or 'PRICE' in summary_upper

    def test_capture_entry_reasoning_with_market_context(self, service, sample_signals, market_context):
        """Test entry reasoning includes market context."""
        result = service.capture_entry_reasoning(
            signals=sample_signals,
            confidence=80,
            market_context=market_context
        )

        assert result['market_context'] == market_context
        assert result['market_context']['vix'] == 18.5
        assert result['market_context']['spy_trend'] == 'bullish'

    def test_capture_entry_reasoning_timestamp_format(self, service, sample_signals):
        """Test that timestamp is in ISO format."""
        result = service.capture_entry_reasoning(
            signals=sample_signals,
            confidence=75
        )

        # Should be ISO format timestamp
        assert 'T' in result['timestamp']
        # Should be parseable as datetime
        from dateutil.parser import parse
        parsed = parse(result['timestamp'])
        assert isinstance(parsed, datetime)

    def test_generate_entry_summary_single_signal(self, service):
        """Test summary generation with single triggered signal."""
        signals = {'rsi': {'met': True, 'value': 28}}
        summary = service._generate_entry_summary(signals)

        assert 'RSI' in summary
        assert 'signal' in summary.lower()

    def test_generate_entry_summary_two_signals(self, service):
        """Test summary generation with two triggered signals."""
        signals = {
            'rsi': {'met': True, 'value': 28},
            'volume': {'triggered': True, 'ratio': 2.5}
        }
        summary = service._generate_entry_summary(signals)

        # Should contain both signals joined by +
        assert '+' in summary
        assert 'RSI' in summary
        assert 'VOLUME' in summary

    def test_generate_entry_summary_three_signals(self, service):
        """Test summary generation with three triggered signals."""
        signals = {
            'rsi': {'met': True},
            'volume': {'triggered': True},
            'macd': {'met': True}
        }
        summary = service._generate_entry_summary(signals)

        # Should contain all three joined by +
        assert '+' in summary

    def test_generate_entry_summary_many_signals(self, service):
        """Test summary generation with more than 3 signals - should show count."""
        signals = {f'signal_{i}': {'met': True} for i in range(5)}
        summary = service._generate_entry_summary(signals)

        assert 'signals aligned' in summary
        assert '5' in summary

    def test_generate_entry_summary_no_triggered_signals(self, service):
        """Test summary when no signals are triggered."""
        signals = {
            'rsi': {'met': False, 'value': 50},
            'macd': {'met': False, 'value': 0.1}
        }
        summary = service._generate_entry_summary(signals)

        assert 'No specific signals' in summary

    def test_format_signals_normalizes_triggered_to_met(self, service):
        """Test that 'triggered' key is normalized to 'met'."""
        signals = {
            'rsi': {'triggered': True, 'value': 28},
            'volume': {'met': True, 'value': 2.0}
        }
        result = service._format_signals_for_storage(signals)

        assert result['rsi']['met'] is True
        assert result['volume']['met'] is True


class TestCaptureExitReasoning:
    """Test exit reasoning capture with real calculations."""

    def test_capture_exit_reasoning_basic(self, service):
        """Test basic exit reasoning with entry timestamp."""
        entry_time = timezone.now() - timedelta(hours=5)

        result = service.capture_exit_reasoning(
            trigger='take_profit',
            entry_timestamp=entry_time
        )

        assert result['trigger'] == 'take_profit'
        assert 'held_duration' in result
        assert result['held_duration_hours'] > 4.9  # Approximately 5 hours
        assert result['held_duration_hours'] < 5.1
        assert 'timestamp' in result

    def test_capture_exit_reasoning_all_trigger_types(self, service):
        """Test all exit trigger types produce valid reasoning."""
        for trigger in EXIT_TRIGGERS.keys():
            result = service.capture_exit_reasoning(trigger=trigger)

            assert result['trigger'] == trigger
            assert result['summary'] is not None
            # Summary should contain the trigger display name
            expected_display = EXIT_TRIGGERS[trigger]
            assert expected_display in result['summary'] or 'triggered' in result['summary'].lower()

    def test_capture_exit_reasoning_with_custom_summary(self, service):
        """Test custom summary overrides auto-generation."""
        result = service.capture_exit_reasoning(
            trigger='stop_loss',
            summary='Volatility spike caused early exit'
        )

        assert result['summary'] == 'Volatility spike caused early exit'
        assert result['trigger'] == 'stop_loss'

    def test_capture_exit_reasoning_with_exit_signals(self, service):
        """Test exit reasoning captures signals at exit time."""
        exit_signals = {
            'rsi': {'value': 72, 'overbought': True},
            'price_vs_entry': {'pct_gain': 5.2}
        }

        result = service.capture_exit_reasoning(
            trigger='signal',
            signals_at_exit=exit_signals
        )

        assert result['signals_at_exit'] == exit_signals
        assert result['signals_at_exit']['rsi']['value'] == 72

    def test_capture_exit_reasoning_unknown_entry_timestamp(self, service):
        """Test exit reasoning handles missing entry timestamp."""
        result = service.capture_exit_reasoning(
            trigger='manual',
            entry_timestamp=None
        )

        assert result['held_duration'] == 'unknown'
        assert result['held_duration_hours'] == 0

    def test_format_duration_minutes(self, service):
        """Test duration formatting for less than 1 hour."""
        duration = service._format_duration(0.5)
        assert 'minutes' in duration
        assert '30' in duration

    def test_format_duration_hours(self, service):
        """Test duration formatting for 1-24 hours."""
        duration = service._format_duration(3.5)
        assert 'hours' in duration
        assert '3.5' in duration

    def test_format_duration_one_day(self, service):
        """Test duration formatting for 24-48 hours."""
        duration = service._format_duration(30)
        assert 'day' in duration

    def test_format_duration_multiple_days(self, service):
        """Test duration formatting for more than 48 hours."""
        duration = service._format_duration(72)
        assert 'days' in duration
        assert '3' in duration


@pytest.mark.django_db
class TestCreateTradeWithReasoningDatabase:
    """Integration tests for trade creation with real database operations."""

    def test_create_trade_with_reasoning_persists_to_database(self, service, sample_signals, market_context):
        """Test that trade is actually created in the database."""
        snapshot = service.create_trade_with_reasoning(
            trade_id='INT_TEST_001',
            symbol='AAPL',
            direction='buy',
            entry_price=150.00,
            quantity=100,
            strategy_name='wsb_dip_bot',
            signals=sample_signals,
            confidence=85,
            market_context=market_context
        )

        # Verify snapshot was created
        assert snapshot is not None
        assert snapshot.pk is not None  # Has database ID

        # Verify we can fetch it from database
        fetched = TradeSignalSnapshot.objects.get(trade_id='INT_TEST_001')
        assert fetched.symbol == 'AAPL'
        assert fetched.direction == 'buy'
        assert fetched.confidence_score == 85
        assert float(fetched.entry_price) == 150.00
        assert float(fetched.quantity) == 100

    def test_create_trade_generates_entry_reasoning_structure(self, service, sample_signals):
        """Test entry reasoning has correct LLM-style structure."""
        snapshot = service.create_trade_with_reasoning(
            trade_id='INT_TEST_002',
            symbol='TSLA',
            direction='buy',
            entry_price=200.00,
            quantity=50,
            strategy_name='momentum_bot',
            signals=sample_signals,
            confidence=75
        )

        # Verify entry reasoning structure
        entry_reasoning = snapshot.entry_reasoning
        assert entry_reasoning is not None
        assert 'summary' in entry_reasoning
        assert 'signals' in entry_reasoning
        assert 'confidence' in entry_reasoning
        assert 'market_context' in entry_reasoning
        assert 'timestamp' in entry_reasoning

        # Verify confidence matches
        assert entry_reasoning['confidence'] == 75

    def test_create_trade_counts_triggered_signals_correctly(self, service, sample_signals):
        """Test that signals_triggered count is accurate."""
        # sample_signals has 3 met signals: rsi, volume, price_action
        snapshot = service.create_trade_with_reasoning(
            trade_id='INT_TEST_003',
            symbol='MSFT',
            direction='buy',
            entry_price=300.00,
            quantity=25,
            strategy_name='dip_hunter',
            signals=sample_signals,
            confidence=80
        )

        assert snapshot.signals_triggered == 3  # rsi, volume, price_action
        assert snapshot.signals_checked == 4    # total signals

    def test_create_trade_generates_explanation(self, service, sample_signals):
        """Test auto-generated explanation contains key information."""
        snapshot = service.create_trade_with_reasoning(
            trade_id='INT_TEST_004',
            symbol='GOOGL',
            direction='buy',
            entry_price=140.00,
            quantity=75,
            strategy_name='value_dip_bot',
            signals=sample_signals,
            confidence=90
        )

        explanation = snapshot.explanation
        assert explanation is not None
        assert 'value_dip_bot' in explanation  # Strategy name
        assert '90%' in explanation            # Confidence
        assert 'rsi' in explanation.lower() or 'volume' in explanation.lower()

    def test_create_trade_with_custom_explanation(self, service, sample_signals):
        """Test custom explanation is preserved."""
        custom_explanation = "Strong dip detected with high conviction signals"

        snapshot = service.create_trade_with_reasoning(
            trade_id='INT_TEST_005',
            symbol='NVDA',
            direction='buy',
            entry_price=450.00,
            quantity=20,
            strategy_name='gpu_dip_bot',
            signals=sample_signals,
            confidence=95,
            explanation=custom_explanation
        )

        assert snapshot.explanation == custom_explanation

    def test_create_trade_stores_signals_at_entry(self, service, sample_signals):
        """Test signals at entry are stored correctly."""
        snapshot = service.create_trade_with_reasoning(
            trade_id='INT_TEST_006',
            symbol='AMD',
            direction='buy',
            entry_price=120.00,
            quantity=100,
            strategy_name='chip_dip_bot',
            signals=sample_signals,
            confidence=85
        )

        signals_at_entry = snapshot.signals_at_entry
        assert 'rsi' in signals_at_entry
        assert signals_at_entry['rsi']['value'] == 28
        assert signals_at_entry['rsi']['met'] is True


@pytest.mark.django_db
class TestRecordExitWithReasoningDatabase:
    """Integration tests for recording trade exits with real database."""

    def test_record_exit_updates_database(self, service, sample_signals):
        """Test exit recording updates the trade in database."""
        # First create a trade
        snapshot = service.create_trade_with_reasoning(
            trade_id='EXIT_TEST_001',
            symbol='AAPL',
            direction='buy',
            entry_price=150.00,
            quantity=100,
            strategy_name='test_strategy',
            signals=sample_signals,
            confidence=80
        )

        # Record exit
        updated = service.record_exit_with_reasoning(
            trade_id='EXIT_TEST_001',
            exit_price=155.00,
            trigger='take_profit',
            pnl_amount=500.00,
            pnl_percent=3.33
        )

        # Verify update
        assert updated is not None
        assert float(updated.exit_price) == 155.00
        assert float(updated.pnl_amount) == 500.00
        assert float(updated.pnl_percent) == 3.33
        assert updated.outcome == 'profit'

    def test_record_exit_generates_exit_reasoning(self, service, sample_signals):
        """Test exit recording generates proper exit reasoning structure."""
        # Create trade
        service.create_trade_with_reasoning(
            trade_id='EXIT_TEST_002',
            symbol='TSLA',
            direction='buy',
            entry_price=200.00,
            quantity=50,
            strategy_name='test_strategy',
            signals=sample_signals,
            confidence=75
        )

        # Wait a bit for realistic duration
        import time
        time.sleep(0.1)

        # Record exit
        updated = service.record_exit_with_reasoning(
            trade_id='EXIT_TEST_002',
            exit_price=190.00,
            trigger='stop_loss',
            pnl_percent=-5.0
        )

        # Verify exit reasoning structure
        exit_reasoning = updated.exit_reasoning
        assert exit_reasoning is not None
        assert 'summary' in exit_reasoning
        assert 'trigger' in exit_reasoning
        assert exit_reasoning['trigger'] == 'stop_loss'
        assert 'held_duration' in exit_reasoning
        assert 'held_duration_hours' in exit_reasoning

    def test_record_exit_not_found_returns_none(self, service):
        """Test recording exit for non-existent trade returns None."""
        result = service.record_exit_with_reasoning(
            trade_id='NONEXISTENT_TRADE_123',
            exit_price=155.00,
            trigger='take_profit',
            pnl_percent=3.33
        )

        assert result is None

    def test_record_exit_with_exit_signals(self, service, sample_signals):
        """Test exit recording includes signals at exit time."""
        # Create trade
        service.create_trade_with_reasoning(
            trade_id='EXIT_TEST_003',
            symbol='GOOGL',
            direction='buy',
            entry_price=140.00,
            quantity=30,
            strategy_name='test_strategy',
            signals=sample_signals,
            confidence=70
        )

        exit_signals = {
            'rsi': {'value': 75, 'overbought': True},
            'macd': {'crossover': False, 'bearish': True}
        }

        updated = service.record_exit_with_reasoning(
            trade_id='EXIT_TEST_003',
            exit_price=148.00,
            trigger='signal',
            pnl_percent=5.71,
            signals_at_exit=exit_signals
        )

        assert updated.exit_reasoning['signals_at_exit'] == exit_signals

    def test_record_exit_determines_outcome_correctly(self, service, sample_signals):
        """Test that outcome is determined correctly based on P&L."""
        # Create multiple trades and test different outcomes
        test_cases = [
            ('OUTCOME_001', 5.0, 'profit'),
            ('OUTCOME_002', -3.0, 'loss'),
            ('OUTCOME_003', 0.3, 'break_even'),
        ]

        for trade_id, pnl_pct, expected_outcome in test_cases:
            service.create_trade_with_reasoning(
                trade_id=trade_id,
                symbol='TEST',
                direction='buy',
                entry_price=100.00,
                quantity=10,
                strategy_name='test_strategy',
                signals=sample_signals,
                confidence=70
            )

            updated = service.record_exit_with_reasoning(
                trade_id=trade_id,
                exit_price=100.00 + pnl_pct,
                trigger='manual',
                pnl_percent=pnl_pct
            )

            assert updated.outcome == expected_outcome, f"Expected {expected_outcome} for P&L {pnl_pct}%"


@pytest.mark.django_db
class TestAnalyzeClosedTradeDatabase:
    """Integration tests for post-trade analysis with real database."""

    def test_analyze_closed_trade_generates_analysis(self, service, sample_signals):
        """Test that closed trade analysis generates proper structure."""
        # Create and close a trade
        service.create_trade_with_reasoning(
            trade_id='ANALYSIS_001',
            symbol='AAPL',
            direction='buy',
            entry_price=150.00,
            quantity=100,
            strategy_name='analysis_test',
            signals=sample_signals,
            confidence=85
        )

        service.record_exit_with_reasoning(
            trade_id='ANALYSIS_001',
            exit_price=158.00,
            trigger='take_profit',
            pnl_amount=800.00,
            pnl_percent=5.33
        )

        # Analyze
        analysis = service.analyze_closed_trade('ANALYSIS_001')

        assert analysis is not None
        assert 'pnl' in analysis
        assert 'pnl_pct' in analysis
        assert analysis['pnl_pct'] == 5.33
        assert 'timing_score' in analysis
        assert 'notes' in analysis

    def test_analyze_closed_trade_timing_scores(self, service, sample_signals):
        """Test timing score is calculated based on P&L."""
        test_cases = [
            ('TIMING_001', 6.0, 'excellent'),
            ('TIMING_002', 3.0, 'good'),
            ('TIMING_003', 1.0, 'fair'),
            ('TIMING_004', -1.0, 'poor'),
            ('TIMING_005', -5.0, 'bad'),
        ]

        for trade_id, pnl_pct, expected_score in test_cases:
            service.create_trade_with_reasoning(
                trade_id=trade_id,
                symbol='TEST',
                direction='buy',
                entry_price=100.00,
                quantity=10,
                strategy_name='timing_test',
                signals=sample_signals,
                confidence=75
            )

            service.record_exit_with_reasoning(
                trade_id=trade_id,
                exit_price=100.00 + pnl_pct,
                trigger='manual',
                pnl_percent=pnl_pct
            )

            analysis = service.analyze_closed_trade(trade_id)
            assert analysis['timing_score'] == expected_score, f"Expected {expected_score} for P&L {pnl_pct}%"

    def test_analyze_closed_trade_generates_notes(self, service, sample_signals):
        """Test that analysis generates appropriate notes."""
        # Strong profit trade
        service.create_trade_with_reasoning(
            trade_id='NOTES_001',
            symbol='AAPL',
            direction='buy',
            entry_price=150.00,
            quantity=100,
            strategy_name='notes_test',
            signals=sample_signals,
            confidence=90
        )

        service.record_exit_with_reasoning(
            trade_id='NOTES_001',
            exit_price=160.00,
            trigger='take_profit',
            pnl_percent=6.67
        )

        analysis = service.analyze_closed_trade('NOTES_001')
        assert 'Strong profit' in analysis['notes']

    def test_analyze_closed_trade_not_found_returns_none(self, service):
        """Test analyzing non-existent trade returns None."""
        result = service.analyze_closed_trade('NONEXISTENT_TRADE')
        assert result is None

    def test_analyze_open_trade_returns_none(self, service, sample_signals):
        """Test analyzing open trade returns None."""
        service.create_trade_with_reasoning(
            trade_id='OPEN_TRADE_001',
            symbol='AAPL',
            direction='buy',
            entry_price=150.00,
            quantity=100,
            strategy_name='open_test',
            signals=sample_signals,
            confidence=80
        )

        # Don't record exit - trade is still open
        result = service.analyze_closed_trade('OPEN_TRADE_001')
        assert result is None

    def test_analyze_closed_trade_with_historical_prices(self, service, sample_signals):
        """Test analysis with historical prices for vs_hold calculation."""
        service.create_trade_with_reasoning(
            trade_id='HIST_001',
            symbol='AAPL',
            direction='buy',
            entry_price=150.00,
            quantity=100,
            strategy_name='hist_test',
            signals=sample_signals,
            confidence=85
        )

        service.record_exit_with_reasoning(
            trade_id='HIST_001',
            exit_price=155.00,
            trigger='take_profit',
            pnl_percent=3.33
        )

        # Get the trade to find dates
        snapshot = TradeSignalSnapshot.objects.get(trade_id='HIST_001')
        entry_date = snapshot.created_at.strftime('%Y-%m-%d')
        exit_date = snapshot.exit_timestamp.strftime('%Y-%m-%d')

        historical_prices = {
            entry_date: 150.00,
            exit_date: 152.00  # Market went up 1.33%
        }

        analysis = service.analyze_closed_trade('HIST_001', historical_prices)

        # Trade returned 3.33%, market returned 1.33%, so vs_hold = 2.0%
        assert analysis['vs_hold'] is not None


@pytest.mark.django_db
class TestGetTradeWithFullReasoning:
    """Integration tests for retrieving trades with full reasoning."""

    def test_get_trade_with_full_reasoning_returns_complete_data(self, service, sample_signals, market_context):
        """Test retrieving trade includes all reasoning data."""
        service.create_trade_with_reasoning(
            trade_id='FULL_001',
            symbol='AAPL',
            direction='buy',
            entry_price=150.00,
            quantity=100,
            strategy_name='full_test',
            signals=sample_signals,
            confidence=85,
            market_context=market_context
        )

        result = service.get_trade_with_full_reasoning('FULL_001')

        assert result is not None
        assert result['trade_id'] == 'FULL_001'
        assert result['symbol'] == 'AAPL'
        assert result['direction'] == 'buy'
        assert result['strategy_name'] == 'full_test'
        assert result['entry_price'] == 150.00
        assert result['confidence_score'] == 85
        assert result['entry_reasoning'] is not None
        assert result['signals_at_entry'] is not None

    def test_get_trade_with_full_reasoning_after_exit(self, service, sample_signals):
        """Test retrieving closed trade includes exit reasoning."""
        service.create_trade_with_reasoning(
            trade_id='FULL_002',
            symbol='TSLA',
            direction='buy',
            entry_price=200.00,
            quantity=50,
            strategy_name='full_test',
            signals=sample_signals,
            confidence=75
        )

        service.record_exit_with_reasoning(
            trade_id='FULL_002',
            exit_price=210.00,
            trigger='take_profit',
            pnl_percent=5.0
        )

        result = service.get_trade_with_full_reasoning('FULL_002')

        assert result['exit_price'] == 210.00
        assert result['exit_reasoning'] is not None
        assert result['exit_trigger'] == 'take_profit'
        assert result['outcome'] == 'profit'

    def test_get_trade_not_found_returns_none(self, service):
        """Test retrieving non-existent trade returns None."""
        result = service.get_trade_with_full_reasoning('NONEXISTENT_123')
        assert result is None


@pytest.mark.django_db
class TestGetTradesByStrategy:
    """Integration tests for retrieving trades by strategy."""

    def test_get_trades_by_strategy_returns_correct_trades(self, service, sample_signals):
        """Test retrieving trades filters by strategy correctly."""
        # Create trades for different strategies
        for i in range(3):
            service.create_trade_with_reasoning(
                trade_id=f'STRAT_A_{i}',
                symbol='AAPL',
                direction='buy',
                entry_price=150.00,
                quantity=100,
                strategy_name='strategy_a',
                signals=sample_signals,
                confidence=80
            )

        for i in range(2):
            service.create_trade_with_reasoning(
                trade_id=f'STRAT_B_{i}',
                symbol='TSLA',
                direction='buy',
                entry_price=200.00,
                quantity=50,
                strategy_name='strategy_b',
                signals=sample_signals,
                confidence=75
            )

        # Get trades for strategy_a
        trades = service.get_trades_by_strategy('strategy_a', include_open=True)

        assert len(trades) == 3
        for trade in trades:
            assert trade['strategy_name'] == 'strategy_a'

    def test_get_trades_by_strategy_excludes_open(self, service, sample_signals):
        """Test excluding open trades works correctly."""
        # Create open and closed trades
        service.create_trade_with_reasoning(
            trade_id='OPEN_STRAT_001',
            symbol='AAPL',
            direction='buy',
            entry_price=150.00,
            quantity=100,
            strategy_name='mixed_strategy',
            signals=sample_signals,
            confidence=80
        )

        service.create_trade_with_reasoning(
            trade_id='CLOSED_STRAT_001',
            symbol='AAPL',
            direction='buy',
            entry_price=150.00,
            quantity=100,
            strategy_name='mixed_strategy',
            signals=sample_signals,
            confidence=80
        )
        service.record_exit_with_reasoning(
            trade_id='CLOSED_STRAT_001',
            exit_price=155.00,
            trigger='take_profit',
            pnl_percent=3.33
        )

        # Get only closed trades
        trades = service.get_trades_by_strategy('mixed_strategy', include_open=False)

        assert len(trades) == 1
        assert trades[0]['trade_id'] == 'CLOSED_STRAT_001'

    def test_get_trades_by_strategy_respects_limit(self, service, sample_signals):
        """Test that limit parameter works."""
        # Create 5 trades
        for i in range(5):
            service.create_trade_with_reasoning(
                trade_id=f'LIMIT_TEST_{i}',
                symbol='AAPL',
                direction='buy',
                entry_price=150.00,
                quantity=100,
                strategy_name='limit_strategy',
                signals=sample_signals,
                confidence=80
            )

        trades = service.get_trades_by_strategy('limit_strategy', limit=3)
        assert len(trades) == 3


@pytest.mark.django_db
class TestGetReasoningStats:
    """Integration tests for reasoning statistics."""

    def test_get_reasoning_stats_calculates_correctly(self, service, sample_signals):
        """Test that stats are calculated correctly."""
        # Create trades with different confidence levels
        confidence_levels = [80, 85, 45, 35, 90, 60, 75, 25, 70, 55]

        for i, conf in enumerate(confidence_levels):
            service.create_trade_with_reasoning(
                trade_id=f'STATS_{i}',
                symbol='AAPL',
                direction='buy',
                entry_price=150.00,
                quantity=100,
                strategy_name='stats_strategy',
                signals=sample_signals,
                confidence=conf
            )

        stats = service.get_reasoning_stats('stats_strategy', days=30)

        assert stats['total_trades'] == 10
        assert stats['with_entry_reasoning'] == 10  # All have entry reasoning
        assert stats['reasoning_coverage'] == 1.0
        assert stats['avg_confidence'] > 0
        assert 'confidence_breakdown' in stats
        assert 'high' in stats['confidence_breakdown']
        assert 'medium' in stats['confidence_breakdown']
        assert 'low' in stats['confidence_breakdown']

    def test_get_reasoning_stats_confidence_breakdown(self, service, sample_signals):
        """Test confidence breakdown categorization."""
        # Create trades: 2 high (>=70), 2 medium (40-70), 1 low (<40)
        configs = [
            ('HIGH_1', 85),
            ('HIGH_2', 75),
            ('MED_1', 55),
            ('MED_2', 50),
            ('LOW_1', 30),
        ]

        for trade_id, conf in configs:
            service.create_trade_with_reasoning(
                trade_id=f'BREAKDOWN_{trade_id}',
                symbol='AAPL',
                direction='buy',
                entry_price=150.00,
                quantity=100,
                strategy_name='breakdown_strategy',
                signals=sample_signals,
                confidence=conf
            )

        stats = service.get_reasoning_stats('breakdown_strategy')

        assert stats['confidence_breakdown']['high']['count'] == 2
        assert stats['confidence_breakdown']['medium']['count'] == 2
        assert stats['confidence_breakdown']['low']['count'] == 1


@pytest.mark.django_db
class TestSimilarTrades:
    """Integration tests for finding similar historical trades."""

    def test_find_similar_trades_with_matching_signals(self, service, sample_signals):
        """Test finding similar trades based on signal patterns."""
        # Create historical closed trades with similar signals
        for i in range(3):
            service.create_trade_with_reasoning(
                trade_id=f'HISTORICAL_{i}',
                symbol='AAPL',
                direction='buy',
                entry_price=150.00,
                quantity=100,
                strategy_name='similar_strategy',
                signals=sample_signals,
                confidence=80
            )
            service.record_exit_with_reasoning(
                trade_id=f'HISTORICAL_{i}',
                exit_price=155.00 if i % 2 == 0 else 145.00,
                trigger='take_profit' if i % 2 == 0 else 'stop_loss',
                pnl_percent=3.33 if i % 2 == 0 else -3.33
            )

        # Find similar trades for a new trade with same signals
        similar = service._find_similar_trades(sample_signals, 'similar_strategy')

        assert isinstance(similar, list)
        # Should find the closed trades with similar signals
        for trade in similar:
            assert 'trade_id' in trade
            assert 'similarity' in trade
            assert 'outcome' in trade
            assert 'pnl_pct' in trade

    def test_get_similar_trades_stats_calculates_correctly(self, service):
        """Test stats calculation from similar trades."""
        from unittest.mock import Mock

        snapshot = Mock()
        snapshot.similar_historical_trades = [
            {'pnl_pct': 5.0, 'outcome': 'profit'},
            {'pnl_pct': -2.0, 'outcome': 'loss'},
            {'pnl_pct': 3.0, 'outcome': 'profit'}
        ]

        result = service._get_similar_trades_stats(snapshot)

        assert result['count'] == 3
        assert result['win_rate'] == 0.67  # 2/3 wins
        assert result['avg_pnl'] == 2.0    # (5 - 2 + 3) / 3

    def test_get_similar_trades_stats_empty(self, service):
        """Test stats with no similar trades."""
        from unittest.mock import Mock

        snapshot = Mock()
        snapshot.similar_historical_trades = []

        result = service._get_similar_trades_stats(snapshot)

        assert result['count'] == 0
        assert result['avg_pnl'] == 0
        assert result['win_rate'] == 0


class TestGenerateExplanation:
    """Test explanation generation with real logic."""

    def test_generate_explanation_contains_strategy(self, service, sample_signals):
        """Test explanation includes strategy name."""
        explanation = service._generate_explanation(sample_signals, 85, 'wsb_momentum')
        assert 'wsb_momentum' in explanation

    def test_generate_explanation_contains_confidence(self, service, sample_signals):
        """Test explanation includes confidence percentage."""
        explanation = service._generate_explanation(sample_signals, 85, 'test_strategy')
        assert '85%' in explanation

    def test_generate_explanation_contains_triggered_signals(self, service, sample_signals):
        """Test explanation includes triggered signals with values."""
        explanation = service._generate_explanation(sample_signals, 85, 'test_strategy')

        # Should contain signal names and values
        assert 'rsi' in explanation.lower()
        assert '28' in explanation  # RSI value

    def test_generate_explanation_with_no_triggered_signals(self, service):
        """Test explanation with no triggered signals."""
        signals = {
            'rsi': {'met': False, 'value': 50},
            'macd': {'met': False, 'value': 0.1}
        }
        explanation = service._generate_explanation(signals, 60, 'test_strategy')

        assert 'test_strategy' in explanation
        assert '60%' in explanation
        # Should not have triggered signals line or it should be empty


class TestGenerateAnalysisNotes:
    """Test analysis notes generation with real logic."""

    def test_generate_notes_strong_profit(self, service):
        """Test notes for strong profit trade."""
        analysis = {'pnl_pct': 6.0, 'vs_hold': None}
        timing = {'score': 'excellent'}

        notes = service._generate_analysis_notes(analysis, timing)
        assert 'Strong profit' in notes

    def test_generate_notes_solid_profit(self, service):
        """Test notes for solid profit trade."""
        analysis = {'pnl_pct': 3.0, 'vs_hold': None}
        timing = {'score': 'good'}

        notes = service._generate_analysis_notes(analysis, timing)
        assert 'Solid profitable' in notes

    def test_generate_notes_small_profit(self, service):
        """Test notes for small profit/breakeven."""
        analysis = {'pnl_pct': 0.5, 'vs_hold': None}
        timing = {'score': 'fair'}

        notes = service._generate_analysis_notes(analysis, timing)
        assert 'Small profit' in notes or 'breakeven' in notes

    def test_generate_notes_small_loss(self, service):
        """Test notes for small loss within risk parameters."""
        analysis = {'pnl_pct': -1.5, 'vs_hold': None}
        timing = {'score': 'poor'}

        notes = service._generate_analysis_notes(analysis, timing)
        assert 'Small loss' in notes or 'risk parameters' in notes

    def test_generate_notes_significant_loss(self, service):
        """Test notes for significant loss."""
        analysis = {'pnl_pct': -5.0, 'vs_hold': None}
        timing = {'score': 'bad'}

        notes = service._generate_analysis_notes(analysis, timing)
        assert 'Significant loss' in notes

    def test_generate_notes_outperformed_hold(self, service):
        """Test notes when outperforming buy-and-hold."""
        analysis = {'pnl_pct': 5.0, 'vs_hold': 3.0}
        timing = {'score': 'excellent'}

        notes = service._generate_analysis_notes(analysis, timing)
        assert 'Outperformed buy-and-hold' in notes

    def test_generate_notes_underperformed_hold(self, service):
        """Test notes when underperforming buy-and-hold."""
        analysis = {'pnl_pct': 1.0, 'vs_hold': -3.0}
        timing = {'score': 'fair'}

        notes = service._generate_analysis_notes(analysis, timing)
        assert 'Underperformed buy-and-hold' in notes


class TestCalculateHoldReturn:
    """Test hold return calculation with real logic."""

    def test_calculate_hold_return_with_valid_prices(self, service):
        """Test hold return calculation with valid price data."""
        historical_prices = {
            '2024-01-15': 150.00,
            '2024-01-20': 157.50
        }

        entry_time = datetime(2024, 1, 15)
        exit_time = datetime(2024, 1, 20)

        result = service._calculate_hold_return(
            'AAPL', entry_time, exit_time, historical_prices
        )

        # (157.50 - 150) / 150 * 100 = 5%
        assert result == 5.0

    def test_calculate_hold_return_with_missing_prices(self, service):
        """Test hold return with missing price data."""
        result = service._calculate_hold_return(
            'AAPL',
            datetime(2024, 1, 15),
            datetime(2024, 1, 20),
            {}
        )

        assert result == 0

    def test_calculate_hold_return_negative(self, service):
        """Test hold return calculation for negative returns."""
        historical_prices = {
            '2024-01-15': 150.00,
            '2024-01-20': 142.50
        }

        result = service._calculate_hold_return(
            'AAPL',
            datetime(2024, 1, 15),
            datetime(2024, 1, 20),
            historical_prices
        )

        # (142.50 - 150) / 150 * 100 = -5%
        assert result == -5.0


class TestAnalyzeTiming:
    """Test timing analysis with real logic."""

    def test_analyze_timing_excellent(self, service):
        """Test timing analysis for excellent trade."""
        from unittest.mock import Mock

        snapshot = Mock()
        snapshot.exit_timestamp = timezone.now()
        snapshot.pnl_percent = Decimal('6.0')

        result = service._analyze_timing(snapshot)
        assert result['score'] == 'excellent'

    def test_analyze_timing_good(self, service):
        """Test timing analysis for good trade."""
        from unittest.mock import Mock

        snapshot = Mock()
        snapshot.exit_timestamp = timezone.now()
        snapshot.pnl_percent = Decimal('3.0')

        result = service._analyze_timing(snapshot)
        assert result['score'] == 'good'

    def test_analyze_timing_fair(self, service):
        """Test timing analysis for fair trade."""
        from unittest.mock import Mock

        snapshot = Mock()
        snapshot.exit_timestamp = timezone.now()
        snapshot.pnl_percent = Decimal('1.0')

        result = service._analyze_timing(snapshot)
        assert result['score'] == 'fair'

    def test_analyze_timing_poor(self, service):
        """Test timing analysis for poor trade."""
        from unittest.mock import Mock

        snapshot = Mock()
        snapshot.exit_timestamp = timezone.now()
        snapshot.pnl_percent = Decimal('-1.5')

        result = service._analyze_timing(snapshot)
        assert result['score'] == 'poor'

    def test_analyze_timing_bad(self, service):
        """Test timing analysis for bad trade."""
        from unittest.mock import Mock

        snapshot = Mock()
        snapshot.exit_timestamp = timezone.now()
        snapshot.pnl_percent = Decimal('-5.0')

        result = service._analyze_timing(snapshot)
        assert result['score'] == 'bad'

    def test_analyze_timing_open_trade(self, service):
        """Test timing analysis for open trade."""
        from unittest.mock import Mock

        snapshot = Mock()
        snapshot.exit_timestamp = None

        result = service._analyze_timing(snapshot)
        assert result['score'] == 'unknown'


@pytest.mark.django_db
class TestRiskFactorIdentification:
    """Integration tests for risk factor identification in reasoning."""

    def test_high_volatility_signals_identified(self, service):
        """Test that high volatility signals are captured in reasoning."""
        volatile_signals = {
            'rsi': {'value': 15, 'threshold': 30, 'met': True, 'oversold': True},
            'atr': {'value': 5.0, 'percent': 3.5, 'met': True, 'high_volatility': True},
            'volume': {'value': 5.0, 'threshold': 1.5, 'met': True, 'ratio': 5.0}
        }

        snapshot = service.create_trade_with_reasoning(
            trade_id='RISK_001',
            symbol='GME',
            direction='buy',
            entry_price=25.00,
            quantity=100,
            strategy_name='risk_test',
            signals=volatile_signals,
            confidence=60
        )

        # Signals should be stored for risk analysis
        assert snapshot.signals_at_entry['atr']['high_volatility'] is True
        assert snapshot.signals_at_entry['volume']['ratio'] == 5.0

    def test_overbought_conditions_captured(self, service, bearish_signals):
        """Test that overbought conditions are captured in reasoning."""
        snapshot = service.create_trade_with_reasoning(
            trade_id='RISK_002',
            symbol='TSLA',
            direction='sell',
            entry_price=300.00,
            quantity=50,
            strategy_name='overbought_test',
            signals=bearish_signals,
            confidence=70
        )

        # RSI overbought should be in signals
        assert snapshot.signals_at_entry['rsi']['value'] == 75
        assert snapshot.signals_at_entry['rsi']['met'] is True


@pytest.mark.django_db
class TestOutputFormatValidation:
    """Integration tests validating LLM-style output format."""

    def test_entry_reasoning_has_llm_style_format(self, service, sample_signals, market_context):
        """Test that entry reasoning follows expected LLM-style structure."""
        snapshot = service.create_trade_with_reasoning(
            trade_id='FORMAT_001',
            symbol='AAPL',
            direction='buy',
            entry_price=150.00,
            quantity=100,
            strategy_name='format_test',
            signals=sample_signals,
            confidence=85,
            market_context=market_context
        )

        entry_reasoning = snapshot.entry_reasoning

        # Required fields
        assert 'summary' in entry_reasoning
        assert isinstance(entry_reasoning['summary'], str)
        assert len(entry_reasoning['summary']) > 0

        assert 'signals' in entry_reasoning
        assert isinstance(entry_reasoning['signals'], dict)

        assert 'confidence' in entry_reasoning
        assert isinstance(entry_reasoning['confidence'], int)
        assert 0 <= entry_reasoning['confidence'] <= 100

        assert 'market_context' in entry_reasoning
        assert isinstance(entry_reasoning['market_context'], dict)

        assert 'timestamp' in entry_reasoning
        assert isinstance(entry_reasoning['timestamp'], str)

    def test_exit_reasoning_has_llm_style_format(self, service, sample_signals):
        """Test that exit reasoning follows expected LLM-style structure."""
        service.create_trade_with_reasoning(
            trade_id='FORMAT_002',
            symbol='AAPL',
            direction='buy',
            entry_price=150.00,
            quantity=100,
            strategy_name='format_test',
            signals=sample_signals,
            confidence=80
        )

        service.record_exit_with_reasoning(
            trade_id='FORMAT_002',
            exit_price=155.00,
            trigger='take_profit',
            pnl_percent=3.33
        )

        snapshot = TradeSignalSnapshot.objects.get(trade_id='FORMAT_002')
        exit_reasoning = snapshot.exit_reasoning

        # Required fields
        assert 'summary' in exit_reasoning
        assert isinstance(exit_reasoning['summary'], str)

        assert 'trigger' in exit_reasoning
        assert exit_reasoning['trigger'] in EXIT_TRIGGERS

        assert 'held_duration' in exit_reasoning
        assert isinstance(exit_reasoning['held_duration'], str)

        assert 'held_duration_hours' in exit_reasoning
        assert isinstance(exit_reasoning['held_duration_hours'], (int, float))

        assert 'signals_at_exit' in exit_reasoning
        assert isinstance(exit_reasoning['signals_at_exit'], dict)

        assert 'timestamp' in exit_reasoning
        assert isinstance(exit_reasoning['timestamp'], str)

    def test_full_trade_dict_has_complete_structure(self, service, sample_signals, market_context):
        """Test that to_dict_with_reasoning returns complete structure."""
        service.create_trade_with_reasoning(
            trade_id='FORMAT_003',
            symbol='NVDA',
            direction='buy',
            entry_price=450.00,
            quantity=20,
            strategy_name='format_test',
            signals=sample_signals,
            confidence=90,
            market_context=market_context
        )

        service.record_exit_with_reasoning(
            trade_id='FORMAT_003',
            exit_price=470.00,
            trigger='take_profit',
            pnl_percent=4.44
        )

        result = service.get_trade_with_full_reasoning('FORMAT_003')

        # Core trade fields
        assert 'trade_id' in result
        assert 'symbol' in result
        assert 'direction' in result
        assert 'strategy_name' in result
        assert 'entry_price' in result
        assert 'quantity' in result
        assert 'confidence_score' in result

        # Entry fields
        assert 'entry_reasoning' in result
        assert 'signals_at_entry' in result
        assert 'explanation' in result
        assert 'reasoning_summary' in result

        # Exit fields
        assert 'exit_price' in result
        assert 'exit_timestamp' in result
        assert 'exit_reasoning' in result
        assert 'exit_trigger' in result
        assert 'held_duration' in result

        # Outcome fields
        assert 'outcome' in result
        assert 'pnl_amount' in result
        assert 'pnl_percent' in result

        # Similar trades
        assert 'similar_historical_trades' in result

        # Timestamps
        assert 'created_at' in result
        assert 'updated_at' in result
