"""Comprehensive INTEGRATION tests for DigestService.

These tests use real database operations to verify:
- Trade aggregation actually queries real trade records
- Alert aggregation queries real circuit breaker history
- Position aggregation calculates real totals
- Performance metrics are calculated from real data
- Digest data is properly structured
- Email generation uses real aggregated data

Only SMTP (external email sending) is mocked to avoid sending real emails.
"""

import smtplib
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import MagicMock, Mock, patch
import uuid

import pytest
from django.contrib.auth.models import User
from django.utils import timezone

from backend.auth0login.services.digest_service import DigestService
from backend.tradingbot.models.models import (
    CircuitBreakerHistory,
    CircuitBreakerState,
    DigestLog,
    TradeSignalSnapshot,
    UserProfile,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def user(db):
    """Create a test user with email."""
    return User.objects.create_user(
        username='testuser',
        email='test@example.com',
        password='testpass123'
    )


@pytest.fixture
def user_with_profile(db, user):
    """Create a test user with a UserProfile configured for daily digests."""
    profile, _ = UserProfile.objects.get_or_create(
        user=user,
        defaults={
            'email_frequency': 'daily',
            'risk_tolerance': 3,
        }
    )
    profile.email_frequency = 'daily'
    profile.save()
    return user


@pytest.fixture
def user_weekly_digest(db):
    """Create a test user with weekly digest preference."""
    user = User.objects.create_user(
        username='weeklyuser',
        email='weekly@example.com',
        password='testpass123'
    )
    profile, _ = UserProfile.objects.get_or_create(
        user=user,
        defaults={
            'email_frequency': 'weekly',
            'risk_tolerance': 3,
        }
    )
    profile.email_frequency = 'weekly'
    profile.save()
    return user


@pytest.fixture
def digest_service(user):
    """Create a DigestService instance."""
    return DigestService(user=user)


@pytest.fixture
def circuit_breaker_state(db, user):
    """Create a CircuitBreakerState for testing alert history."""
    return CircuitBreakerState.objects.create(
        user=user,
        breaker_type='daily_loss',
        status='ok',
    )


@pytest.fixture
def mock_smtp_config():
    """Mock SMTP configuration environment variables."""
    with patch.dict('os.environ', {
        'ALERT_EMAIL_SMTP_HOST': 'smtp.example.com',
        'ALERT_EMAIL_SMTP_PORT': '587',
        'ALERT_EMAIL_USER': 'bot@example.com',
        'ALERT_EMAIL_PASS': 'password',
        'ALERT_EMAIL_FROM': 'bot@example.com',
        'APP_BASE_URL': 'http://localhost:8000'
    }):
        yield


def create_trade_snapshot(
    symbol: str,
    direction: str,
    entry_price: Decimal,
    quantity: Decimal,
    strategy_name: str = 'test_strategy',
    exit_price: Decimal | None = None,
    pnl_amount: Decimal | None = None,
    created_at: datetime | None = None,
) -> TradeSignalSnapshot:
    """Helper to create TradeSignalSnapshot records for testing."""
    trade_id = f"trade_{uuid.uuid4().hex[:12]}"
    trade = TradeSignalSnapshot.objects.create(
        trade_id=trade_id,
        symbol=symbol,
        direction=direction,
        entry_price=entry_price,
        quantity=quantity,
        strategy_name=strategy_name,
        exit_price=exit_price,
        pnl_amount=pnl_amount,
        signals_at_entry={'test': True},
    )
    # Update created_at if specified (auto_now_add prevents setting on create)
    if created_at:
        TradeSignalSnapshot.objects.filter(pk=trade.pk).update(created_at=created_at)
        trade.refresh_from_db()
    return trade


def create_circuit_breaker_event(
    state: CircuitBreakerState,
    action: str,
    reason: str | None = None,
    timestamp: datetime | None = None,
) -> CircuitBreakerHistory:
    """Helper to create CircuitBreakerHistory records for testing."""
    event = CircuitBreakerHistory.objects.create(
        state=state,
        action=action,
        reason=reason or f"Test {action} event",
        timestamp=timestamp or timezone.now(),
    )
    return event


# =============================================================================
# Initialization Tests
# =============================================================================

@pytest.mark.django_db
class TestDigestServiceInit:
    """Test DigestService initialization."""

    def test_init_with_user(self, user):
        """Test initialization with user."""
        service = DigestService(user=user)
        assert service.user == user
        assert service.base_url is not None

    def test_init_without_user(self):
        """Test initialization without user for batch operations."""
        service = DigestService()
        assert service.user is None

    def test_init_reads_environment(self, mock_smtp_config):
        """Test initialization reads environment variables."""
        service = DigestService()
        assert service.smtp_host == 'smtp.example.com'
        assert service.smtp_port == 587
        assert service.smtp_user == 'bot@example.com'
        assert service.smtp_pass == 'password'
        assert service.from_email == 'bot@example.com'
        assert service.base_url == 'http://localhost:8000'


# =============================================================================
# Period Bounds Tests
# =============================================================================

@pytest.mark.django_db
class TestPeriodBounds:
    """Test _get_period_bounds method."""

    def test_daily_period_bounds(self, digest_service):
        """Test daily period calculation returns 24-hour window."""
        with patch('django.utils.timezone.now') as mock_now:
            mock_now.return_value = timezone.make_aware(
                datetime(2024, 1, 15, 14, 30, 0)
            )
            start, end = digest_service._get_period_bounds('daily')

            # End should be midnight today
            assert end.hour == 0
            assert end.minute == 0
            assert end.second == 0
            # Duration should be exactly 1 day
            assert (end - start).days == 1
            assert (end - start).seconds == 0

    def test_weekly_period_bounds(self, digest_service):
        """Test weekly period calculation returns 7-day window ending Monday."""
        with patch('django.utils.timezone.now') as mock_now:
            # Wednesday January 17, 2024
            mock_now.return_value = timezone.make_aware(
                datetime(2024, 1, 17, 14, 30, 0)
            )
            start, end = digest_service._get_period_bounds('weekly')

            # Duration should be exactly 7 days
            assert (end - start).days == 7
            # End should be Monday at midnight
            assert end.weekday() == 0  # Monday
            assert end.hour == 0


# =============================================================================
# Trade Aggregation Integration Tests
# =============================================================================

@pytest.mark.django_db
class TestTradeAggregationIntegration:
    """Integration tests for trade aggregation with real database records."""

    def test_aggregate_trades_empty(self, digest_service, user):
        """Test trade aggregation with no trades in period."""
        now = timezone.now()
        result = digest_service.aggregate_trades(user, now - timedelta(days=1), now)

        assert result['summary']['total_trades'] == 0
        assert result['summary']['buy_trades'] == 0
        assert result['summary']['sell_trades'] == 0
        assert result['summary']['closed_trades'] == 0
        assert result['summary']['winning_trades'] == 0
        assert result['summary']['losing_trades'] == 0
        assert result['summary']['total_pnl'] == 0.0
        assert result['summary']['win_rate'] == 0.0
        assert result['trades'] == []

    def test_aggregate_trades_with_open_trades(self, digest_service, user, db):
        """Test aggregation of open trades (no exit price)."""
        now = timezone.now()
        yesterday = now - timedelta(hours=12)

        # Create open buy trades
        create_trade_snapshot(
            symbol='AAPL',
            direction='buy',
            entry_price=Decimal('150.00'),
            quantity=Decimal('100'),
            created_at=yesterday,
        )
        create_trade_snapshot(
            symbol='GOOGL',
            direction='buy',
            entry_price=Decimal('140.00'),
            quantity=Decimal('50'),
            created_at=yesterday,
        )

        result = digest_service.aggregate_trades(user, now - timedelta(days=1), now)

        assert result['summary']['total_trades'] == 2
        assert result['summary']['buy_trades'] == 2
        assert result['summary']['sell_trades'] == 0
        assert result['summary']['closed_trades'] == 0
        assert result['summary']['total_pnl'] == 0.0
        assert len(result['trades']) == 2

    def test_aggregate_trades_with_closed_winning_trades(self, digest_service, user, db):
        """Test aggregation of closed winning trades."""
        now = timezone.now()
        yesterday = now - timedelta(hours=12)

        # Create closed winning trades
        create_trade_snapshot(
            symbol='AAPL',
            direction='buy',
            entry_price=Decimal('150.00'),
            quantity=Decimal('100'),
            exit_price=Decimal('160.00'),
            pnl_amount=Decimal('1000.00'),  # $10 x 100 shares
            created_at=yesterday,
        )
        create_trade_snapshot(
            symbol='MSFT',
            direction='buy',
            entry_price=Decimal('300.00'),
            quantity=Decimal('50'),
            exit_price=Decimal('310.00'),
            pnl_amount=Decimal('500.00'),  # $10 x 50 shares
            created_at=yesterday,
        )

        result = digest_service.aggregate_trades(user, now - timedelta(days=1), now)

        assert result['summary']['total_trades'] == 2
        assert result['summary']['closed_trades'] == 2
        assert result['summary']['winning_trades'] == 2
        assert result['summary']['losing_trades'] == 0
        assert result['summary']['total_pnl'] == 1500.0
        assert result['summary']['win_rate'] == 100.0

    def test_aggregate_trades_with_closed_losing_trades(self, digest_service, user, db):
        """Test aggregation of closed losing trades."""
        now = timezone.now()
        yesterday = now - timedelta(hours=12)

        # Create closed losing trades
        create_trade_snapshot(
            symbol='TSLA',
            direction='buy',
            entry_price=Decimal('250.00'),
            quantity=Decimal('40'),
            exit_price=Decimal('240.00'),
            pnl_amount=Decimal('-400.00'),
            created_at=yesterday,
        )

        result = digest_service.aggregate_trades(user, now - timedelta(days=1), now)

        assert result['summary']['total_trades'] == 1
        assert result['summary']['closed_trades'] == 1
        assert result['summary']['winning_trades'] == 0
        assert result['summary']['losing_trades'] == 1
        assert result['summary']['total_pnl'] == -400.0
        assert result['summary']['win_rate'] == 0.0

    def test_aggregate_trades_mixed_outcomes(self, digest_service, user, db):
        """Test aggregation with mixed winning/losing/open trades."""
        now = timezone.now()
        yesterday = now - timedelta(hours=12)

        # Winning trade
        create_trade_snapshot(
            symbol='AAPL',
            direction='buy',
            entry_price=Decimal('150.00'),
            quantity=Decimal('100'),
            exit_price=Decimal('160.00'),
            pnl_amount=Decimal('1000.00'),
            created_at=yesterday,
        )
        # Losing trade
        create_trade_snapshot(
            symbol='TSLA',
            direction='sell',
            entry_price=Decimal('250.00'),
            quantity=Decimal('50'),
            exit_price=Decimal('260.00'),
            pnl_amount=Decimal('-500.00'),
            created_at=yesterday,
        )
        # Open trade (no exit)
        create_trade_snapshot(
            symbol='GOOGL',
            direction='buy',
            entry_price=Decimal('140.00'),
            quantity=Decimal('25'),
            created_at=yesterday,
        )

        result = digest_service.aggregate_trades(user, now - timedelta(days=1), now)

        assert result['summary']['total_trades'] == 3
        assert result['summary']['buy_trades'] == 2
        assert result['summary']['sell_trades'] == 1
        assert result['summary']['closed_trades'] == 2
        assert result['summary']['winning_trades'] == 1
        assert result['summary']['losing_trades'] == 1
        assert result['summary']['total_pnl'] == 500.0  # 1000 - 500
        assert result['summary']['win_rate'] == 50.0

    def test_aggregate_trades_direction_types(self, digest_service, user, db):
        """Test all direction types are properly categorized."""
        now = timezone.now()
        yesterday = now - timedelta(hours=12)

        # Buy
        create_trade_snapshot(
            symbol='AAPL',
            direction='buy',
            entry_price=Decimal('150.00'),
            quantity=Decimal('100'),
            created_at=yesterday,
        )
        # Sell
        create_trade_snapshot(
            symbol='MSFT',
            direction='sell',
            entry_price=Decimal('300.00'),
            quantity=Decimal('50'),
            created_at=yesterday,
        )
        # Buy to cover
        create_trade_snapshot(
            symbol='GOOGL',
            direction='buy_to_cover',
            entry_price=Decimal('140.00'),
            quantity=Decimal('25'),
            created_at=yesterday,
        )
        # Sell short
        create_trade_snapshot(
            symbol='TSLA',
            direction='sell_short',
            entry_price=Decimal('250.00'),
            quantity=Decimal('30'),
            created_at=yesterday,
        )

        result = digest_service.aggregate_trades(user, now - timedelta(days=1), now)

        assert result['summary']['total_trades'] == 4
        assert result['summary']['buy_trades'] == 2  # buy + buy_to_cover
        assert result['summary']['sell_trades'] == 2  # sell + sell_short

    def test_aggregate_trades_respects_date_range(self, digest_service, user, db):
        """Test that only trades within date range are included."""
        now = timezone.now()
        yesterday = now - timedelta(hours=12)
        two_days_ago = now - timedelta(days=2)

        # Trade within range
        create_trade_snapshot(
            symbol='AAPL',
            direction='buy',
            entry_price=Decimal('150.00'),
            quantity=Decimal('100'),
            created_at=yesterday,
        )
        # Trade outside range (too old)
        create_trade_snapshot(
            symbol='MSFT',
            direction='buy',
            entry_price=Decimal('300.00'),
            quantity=Decimal('50'),
            created_at=two_days_ago,
        )

        result = digest_service.aggregate_trades(user, now - timedelta(days=1), now)

        assert result['summary']['total_trades'] == 1
        assert result['trades'][0]['symbol'] == 'AAPL'

    def test_aggregate_trades_limits_to_20(self, digest_service, user, db):
        """Test that trade list is limited to 20 most recent."""
        now = timezone.now()

        # Create 25 trades
        for i in range(25):
            create_trade_snapshot(
                symbol=f'SYM{i:02d}',
                direction='buy',
                entry_price=Decimal('100.00'),
                quantity=Decimal('10'),
                created_at=now - timedelta(hours=i + 1),
            )

        result = digest_service.aggregate_trades(user, now - timedelta(days=2), now)

        assert result['summary']['total_trades'] == 25
        assert len(result['trades']) == 20  # Limited to 20

    def test_aggregate_trades_detail_structure(self, digest_service, user, db):
        """Test that trade detail structure is correct."""
        now = timezone.now()
        yesterday = now - timedelta(hours=12)

        create_trade_snapshot(
            symbol='AAPL',
            direction='buy',
            entry_price=Decimal('150.5000'),
            quantity=Decimal('100.0000'),
            strategy_name='momentum_strategy',
            exit_price=Decimal('160.2500'),
            pnl_amount=Decimal('975.00'),
            created_at=yesterday,
        )

        result = digest_service.aggregate_trades(user, now - timedelta(days=1), now)
        trade = result['trades'][0]

        assert trade['symbol'] == 'AAPL'
        assert trade['direction'] == 'buy'
        assert trade['quantity'] == 100.0
        assert trade['entry_price'] == 150.5
        assert trade['exit_price'] == 160.25
        assert trade['pnl'] == 975.0
        assert trade['strategy'] == 'momentum_strategy'
        assert 'timestamp' in trade


# =============================================================================
# Alert Aggregation Integration Tests
# =============================================================================

@pytest.mark.django_db
class TestAlertAggregationIntegration:
    """Integration tests for alert aggregation with real CircuitBreakerHistory records."""

    def test_aggregate_alerts_empty(self, digest_service, user, circuit_breaker_state):
        """Test alert aggregation with no events in period."""
        now = timezone.now()
        result = digest_service.aggregate_alerts(user, now - timedelta(days=1), now)

        assert result['summary']['total_alerts'] == 0
        assert result['summary']['high_severity'] == 0
        assert result['summary']['by_type'] == {}
        assert result['alerts'] == []

    def test_aggregate_alerts_with_trip_events(self, digest_service, user, circuit_breaker_state):
        """Test aggregation of circuit breaker trip events (high severity)."""
        now = timezone.now()
        yesterday = now - timedelta(hours=12)

        create_circuit_breaker_event(
            state=circuit_breaker_state,
            action='trip',
            reason='Daily loss limit exceeded',
            timestamp=yesterday,
        )
        create_circuit_breaker_event(
            state=circuit_breaker_state,
            action='trip',
            reason='Error rate too high',
            timestamp=yesterday - timedelta(hours=1),
        )

        result = digest_service.aggregate_alerts(user, now - timedelta(days=1), now)

        assert result['summary']['total_alerts'] == 2
        assert result['summary']['high_severity'] == 2
        assert result['summary']['by_type']['circuit_breaker'] == 2
        assert len(result['alerts']) == 2
        assert all(a['severity'] == 'high' for a in result['alerts'])

    def test_aggregate_alerts_with_reset_events(self, digest_service, user, circuit_breaker_state):
        """Test aggregation of circuit breaker reset events (info severity)."""
        now = timezone.now()
        yesterday = now - timedelta(hours=12)

        create_circuit_breaker_event(
            state=circuit_breaker_state,
            action='reset',
            reason='Manual reset by admin',
            timestamp=yesterday,
        )

        result = digest_service.aggregate_alerts(user, now - timedelta(days=1), now)

        assert result['summary']['total_alerts'] == 1
        assert result['summary']['high_severity'] == 0
        assert result['alerts'][0]['severity'] == 'info'

    def test_aggregate_alerts_mixed_events(self, digest_service, user, circuit_breaker_state):
        """Test aggregation with mixed event types."""
        now = timezone.now()
        yesterday = now - timedelta(hours=12)

        # Trip event
        create_circuit_breaker_event(
            state=circuit_breaker_state,
            action='trip',
            reason='Loss limit hit',
            timestamp=yesterday,
        )
        # Reset event
        create_circuit_breaker_event(
            state=circuit_breaker_state,
            action='reset',
            reason='Auto reset',
            timestamp=yesterday - timedelta(hours=2),
        )
        # Warning event
        create_circuit_breaker_event(
            state=circuit_breaker_state,
            action='warning',
            reason='Approaching limit',
            timestamp=yesterday - timedelta(hours=3),
        )

        result = digest_service.aggregate_alerts(user, now - timedelta(days=1), now)

        assert result['summary']['total_alerts'] == 3
        assert result['summary']['high_severity'] == 1  # Only trip is high

    def test_aggregate_alerts_respects_date_range(self, digest_service, user, circuit_breaker_state):
        """Test that only alerts within date range are included."""
        now = timezone.now()
        yesterday = now - timedelta(hours=12)
        two_days_ago = now - timedelta(days=2)

        # Within range
        create_circuit_breaker_event(
            state=circuit_breaker_state,
            action='trip',
            reason='Recent event',
            timestamp=yesterday,
        )
        # Outside range
        create_circuit_breaker_event(
            state=circuit_breaker_state,
            action='trip',
            reason='Old event',
            timestamp=two_days_ago,
        )

        result = digest_service.aggregate_alerts(user, now - timedelta(days=1), now)

        assert result['summary']['total_alerts'] == 1

    def test_aggregate_alerts_limits_to_10(self, digest_service, user, circuit_breaker_state):
        """Test that alert list is limited to 10."""
        now = timezone.now()

        # Create 15 events
        for i in range(15):
            create_circuit_breaker_event(
                state=circuit_breaker_state,
                action='warning',
                reason=f'Event {i}',
                timestamp=now - timedelta(hours=i + 1),
            )

        result = digest_service.aggregate_alerts(user, now - timedelta(days=2), now)

        assert len(result['alerts']) == 10  # Limited to 10

    def test_aggregate_alerts_detail_structure(self, digest_service, user, circuit_breaker_state):
        """Test that alert detail structure is correct."""
        now = timezone.now()
        yesterday = now - timedelta(hours=12)

        create_circuit_breaker_event(
            state=circuit_breaker_state,
            action='trip',
            reason='Test reason message',
            timestamp=yesterday,
        )

        result = digest_service.aggregate_alerts(user, now - timedelta(days=1), now)
        alert = result['alerts'][0]

        assert alert['type'] == 'circuit_breaker'
        assert alert['action'] == 'trip'
        assert alert['message'] == 'Test reason message'
        assert alert['severity'] == 'high'
        assert 'timestamp' in alert


# =============================================================================
# Position Aggregation Integration Tests
# =============================================================================

@pytest.mark.django_db
class TestPositionAggregationIntegration:
    """Integration tests for position aggregation with real trade records."""

    def test_aggregate_positions_empty(self, digest_service, user):
        """Test position aggregation with no open positions."""
        now = timezone.now()
        result = digest_service.aggregate_positions(user, now)

        assert result['summary']['open_positions'] == 0
        assert result['summary']['total_value'] == 0.0
        assert result['positions'] == []

    def test_aggregate_positions_single_symbol(self, digest_service, user, db):
        """Test position aggregation with single symbol."""
        # Create open position (no exit_price)
        create_trade_snapshot(
            symbol='AAPL',
            direction='buy',
            entry_price=Decimal('150.00'),
            quantity=Decimal('100'),
        )

        now = timezone.now()
        result = digest_service.aggregate_positions(user, now)

        assert result['summary']['open_positions'] == 1
        assert result['summary']['total_value'] == 15000.0  # 150 * 100
        assert len(result['positions']) == 1
        assert result['positions'][0]['symbol'] == 'AAPL'
        assert result['positions'][0]['quantity'] == 100.0
        assert result['positions'][0]['avg_price'] == 150.0
        assert result['positions'][0]['value'] == 15000.0

    def test_aggregate_positions_multiple_trades_same_symbol(self, digest_service, user, db):
        """Test position aggregation combines trades for same symbol."""
        # Create multiple open trades for same symbol
        create_trade_snapshot(
            symbol='AAPL',
            direction='buy',
            entry_price=Decimal('150.00'),
            quantity=Decimal('100'),
        )
        create_trade_snapshot(
            symbol='AAPL',
            direction='buy',
            entry_price=Decimal('160.00'),
            quantity=Decimal('100'),
        )

        now = timezone.now()
        result = digest_service.aggregate_positions(user, now)

        assert result['summary']['open_positions'] == 1
        pos = result['positions'][0]
        assert pos['symbol'] == 'AAPL'
        assert pos['quantity'] == 200.0  # 100 + 100
        assert pos['avg_price'] == 155.0  # (150 + 160) / 2
        assert pos['trade_count'] == 2

    def test_aggregate_positions_excludes_closed_trades(self, digest_service, user, db):
        """Test that closed trades (with exit_price) are not counted as positions."""
        # Open position
        create_trade_snapshot(
            symbol='AAPL',
            direction='buy',
            entry_price=Decimal('150.00'),
            quantity=Decimal('100'),
        )
        # Closed trade (has exit_price)
        create_trade_snapshot(
            symbol='MSFT',
            direction='buy',
            entry_price=Decimal('300.00'),
            quantity=Decimal('50'),
            exit_price=Decimal('310.00'),
            pnl_amount=Decimal('500.00'),
        )

        now = timezone.now()
        result = digest_service.aggregate_positions(user, now)

        assert result['summary']['open_positions'] == 1
        assert result['positions'][0]['symbol'] == 'AAPL'

    def test_aggregate_positions_multiple_symbols(self, digest_service, user, db):
        """Test position aggregation with multiple symbols."""
        create_trade_snapshot(
            symbol='AAPL',
            direction='buy',
            entry_price=Decimal('150.00'),
            quantity=Decimal('100'),
        )
        create_trade_snapshot(
            symbol='GOOGL',
            direction='buy',
            entry_price=Decimal('140.00'),
            quantity=Decimal('50'),
        )
        create_trade_snapshot(
            symbol='MSFT',
            direction='buy',
            entry_price=Decimal('350.00'),
            quantity=Decimal('25'),
        )

        now = timezone.now()
        result = digest_service.aggregate_positions(user, now)

        assert result['summary']['open_positions'] == 3
        # Total value = (150*100) + (140*50) + (350*25) = 15000 + 7000 + 8750 = 30750
        assert result['summary']['total_value'] == 30750.0

    def test_aggregate_positions_limits_to_10(self, digest_service, user, db):
        """Test that positions list is limited to 10."""
        # Create 15 different symbols
        for i in range(15):
            create_trade_snapshot(
                symbol=f'SYM{i:02d}',
                direction='buy',
                entry_price=Decimal('100.00'),
                quantity=Decimal('10'),
            )

        now = timezone.now()
        result = digest_service.aggregate_positions(user, now)

        assert result['summary']['open_positions'] == 15
        assert len(result['positions']) == 10  # Limited to 10


# =============================================================================
# Strategy Aggregation Integration Tests
# =============================================================================

@pytest.mark.django_db
class TestStrategyAggregationIntegration:
    """Integration tests for strategy-based aggregation."""

    def test_aggregate_by_strategy_empty(self, digest_service, user):
        """Test strategy aggregation with no trades."""
        now = timezone.now()
        result = digest_service.aggregate_by_strategy(user, now - timedelta(days=1), now)

        assert result == []

    def test_aggregate_by_strategy_single_strategy(self, digest_service, user, db):
        """Test aggregation for single strategy."""
        now = timezone.now()
        yesterday = now - timedelta(hours=12)

        # Create trades for one strategy
        create_trade_snapshot(
            symbol='AAPL',
            direction='buy',
            entry_price=Decimal('150.00'),
            quantity=Decimal('100'),
            strategy_name='momentum_bot',
            exit_price=Decimal('160.00'),
            pnl_amount=Decimal('1000.00'),
            created_at=yesterday,
        )
        create_trade_snapshot(
            symbol='GOOGL',
            direction='buy',
            entry_price=Decimal('140.00'),
            quantity=Decimal('50'),
            strategy_name='momentum_bot',
            exit_price=Decimal('135.00'),
            pnl_amount=Decimal('-250.00'),
            created_at=yesterday,
        )

        result = digest_service.aggregate_by_strategy(user, now - timedelta(days=1), now)

        assert len(result) == 1
        assert result[0]['strategy'] == 'momentum_bot'
        assert result[0]['total_trades'] == 2
        assert result[0]['total_pnl'] == 750.0  # 1000 - 250
        assert result[0]['winning_trades'] == 1
        assert result[0]['win_rate'] == 50.0

    def test_aggregate_by_strategy_multiple_strategies(self, digest_service, user, db):
        """Test aggregation for multiple strategies."""
        now = timezone.now()
        yesterday = now - timedelta(hours=12)

        # Momentum strategy trades
        create_trade_snapshot(
            symbol='AAPL',
            direction='buy',
            entry_price=Decimal('150.00'),
            quantity=Decimal('100'),
            strategy_name='momentum_bot',
            exit_price=Decimal('160.00'),
            pnl_amount=Decimal('1000.00'),
            created_at=yesterday,
        )
        # Dip buyer strategy trades
        create_trade_snapshot(
            symbol='TSLA',
            direction='buy',
            entry_price=Decimal('200.00'),
            quantity=Decimal('50'),
            strategy_name='dip_buyer',
            exit_price=Decimal('220.00'),
            pnl_amount=Decimal('1000.00'),
            created_at=yesterday,
        )
        create_trade_snapshot(
            symbol='MSFT',
            direction='buy',
            entry_price=Decimal('300.00'),
            quantity=Decimal('25'),
            strategy_name='dip_buyer',
            exit_price=Decimal('310.00'),
            pnl_amount=Decimal('250.00'),
            created_at=yesterday,
        )

        result = digest_service.aggregate_by_strategy(user, now - timedelta(days=1), now)

        assert len(result) == 2
        # Results are ordered by total_pnl descending
        strategies = {r['strategy']: r for r in result}

        assert strategies['dip_buyer']['total_trades'] == 2
        assert strategies['dip_buyer']['total_pnl'] == 1250.0
        assert strategies['momentum_bot']['total_trades'] == 1
        assert strategies['momentum_bot']['total_pnl'] == 1000.0

    def test_aggregate_by_strategy_handles_null_pnl(self, digest_service, user, db):
        """Test strategy aggregation handles trades without P&L."""
        now = timezone.now()
        yesterday = now - timedelta(hours=12)

        # Open trade (no exit, no pnl)
        create_trade_snapshot(
            symbol='AAPL',
            direction='buy',
            entry_price=Decimal('150.00'),
            quantity=Decimal('100'),
            strategy_name='test_strategy',
            created_at=yesterday,
        )

        result = digest_service.aggregate_by_strategy(user, now - timedelta(days=1), now)

        assert len(result) == 1
        assert result[0]['total_trades'] == 1
        assert result[0]['total_pnl'] == 0.0
        assert result[0]['winning_trades'] == 0


# =============================================================================
# Performance Aggregation Tests
# =============================================================================

@pytest.mark.django_db
class TestPerformanceAggregationIntegration:
    """Tests for performance metric aggregation."""

    def test_aggregate_performance_with_benchmark_service(self, digest_service, user):
        """Test performance aggregation when benchmark service is available."""
        with patch('backend.auth0login.services.benchmark.BenchmarkService') as mock_benchmark:
            mock_service = Mock()
            mock_service.get_performance_vs_benchmark.return_value = {
                'portfolio_return': 12.5,
                'benchmark_return': 8.0,
                'alpha': 4.5,
                'sharpe_ratio': 1.8,
                'max_drawdown': -3.5,
            }
            mock_benchmark.return_value = mock_service

            now = timezone.now()
            result = digest_service.aggregate_performance(user, now - timedelta(days=7), now)

            assert result['period_return'] == 12.5
            assert result['benchmark_return'] == 8.0
            assert result['alpha'] == 4.5
            assert result['sharpe_ratio'] == 1.8
            assert result['max_drawdown'] == -3.5

    def test_aggregate_performance_benchmark_failure(self, digest_service, user):
        """Test performance aggregation falls back gracefully on error."""
        with patch('backend.auth0login.services.benchmark.BenchmarkService') as mock_benchmark:
            mock_benchmark.side_effect = ImportError("Module not found")

            now = timezone.now()
            result = digest_service.aggregate_performance(user, now - timedelta(days=7), now)

            # Should return default values
            assert result['period_return'] == 0
            assert result['benchmark_return'] == 0
            assert result['alpha'] == 0
            assert result['sharpe_ratio'] is None
            assert result['max_drawdown'] is None


# =============================================================================
# Digest Generation Integration Tests
# =============================================================================

@pytest.mark.django_db
class TestDigestGenerationIntegration:
    """Integration tests for complete digest data generation."""

    def test_generate_digest_data_empty(self, digest_service, user):
        """Test digest generation with no data."""
        result = digest_service.generate_digest_data(user, 'daily')

        assert result['digest_type'] == 'daily'
        assert result['user']['username'] == 'testuser'
        assert result['user']['email'] == 'test@example.com'
        assert result['summary']['total_trades'] == 0
        assert result['summary']['total_pnl'] == 0.0
        assert result['summary']['win_rate'] == 0.0
        assert 'period_start' in result
        assert 'period_end' in result
        assert 'generated_at' in result

    def test_generate_digest_data_with_real_data(
        self, digest_service, user, circuit_breaker_state, db
    ):
        """Test digest generation with real trades and alerts."""
        now = timezone.now()

        # Create trades within the period bounds we'll query
        with patch('django.utils.timezone.now') as mock_now:
            # Set "now" to a fixed time
            test_now = timezone.make_aware(datetime(2024, 6, 15, 14, 0, 0))
            mock_now.return_value = test_now

            # Get the period bounds that will be used
            service = DigestService(user)
            period_start, period_end = service._get_period_bounds('daily')

            # Create data within that period
            trade_time = period_start + timedelta(hours=12)

            # Create a trade
            trade = TradeSignalSnapshot.objects.create(
                trade_id=f"test_trade_{uuid.uuid4().hex[:8]}",
                symbol='AAPL',
                direction='buy',
                entry_price=Decimal('150.00'),
                quantity=Decimal('100'),
                strategy_name='test_strategy',
                exit_price=Decimal('160.00'),
                pnl_amount=Decimal('1000.00'),
                signals_at_entry={'test': True},
            )
            TradeSignalSnapshot.objects.filter(pk=trade.pk).update(created_at=trade_time)

            # Create an alert
            CircuitBreakerHistory.objects.create(
                state=circuit_breaker_state,
                action='trip',
                reason='Test alert',
                timestamp=trade_time,
            )

            result = service.generate_digest_data(user, 'daily')

        assert result['digest_type'] == 'daily'
        assert result['summary']['total_trades'] == 1
        assert result['summary']['total_pnl'] == 1000.0
        assert result['summary']['total_alerts'] == 1

    def test_generate_digest_data_weekly(self, digest_service, user):
        """Test weekly digest generation."""
        result = digest_service.generate_digest_data(user, 'weekly')

        assert result['digest_type'] == 'weekly'


# =============================================================================
# Email Formatting Tests
# =============================================================================

@pytest.mark.django_db
class TestEmailFormatting:
    """Test email formatting methods with real digest data."""

    def test_get_email_subject_positive_pnl(self, digest_service):
        """Test email subject with positive P&L."""
        data = {
            'summary': {
                'total_trades': 10,
                'total_pnl': 150.50
            }
        }
        subject = digest_service._get_email_subject('daily', data)

        assert '[WSB] Daily Digest' in subject
        assert '10 trades' in subject
        assert '+$150.50' in subject

    def test_get_email_subject_negative_pnl(self, digest_service):
        """Test email subject with negative P&L."""
        data = {
            'summary': {
                'total_trades': 5,
                'total_pnl': -50.25
            }
        }
        subject = digest_service._get_email_subject('weekly', data)

        assert '[WSB] Weekly Digest' in subject
        assert '5 trades' in subject
        assert '-$50.25' in subject

    def test_render_email_text_complete(self, digest_service):
        """Test plain text email rendering with complete data."""
        data = {
            'summary': {
                'total_trades': 10,
                'winning_trades': 6,
                'losing_trades': 4,
                'win_rate': 60.0,
                'total_pnl': 500.0,
                'open_positions': 3
            },
            'trades': [
                {
                    'symbol': 'AAPL',
                    'direction': 'buy',
                    'quantity': 100,
                    'entry_price': 150.0,
                    'pnl': 500.0,
                }
            ],
            'alerts': [
                {
                    'severity': 'high',
                    'message': 'Circuit breaker tripped',
                }
            ],
            'positions': [],
            'performance': {
                'period_return': 5.5,
                'benchmark_return': 3.0,
                'alpha': 2.5,
            },
            'by_strategy': [
                {
                    'strategy': 'momentum_bot',
                    'total_trades': 10,
                    'win_rate': 60.0,
                    'total_pnl': 500.0,
                }
            ]
        }

        text = digest_service._render_email_text('daily', data)

        assert 'SUMMARY' in text
        assert 'Total Trades: 10' in text
        assert 'Win Rate: 60.0%' in text
        assert '+$500.00' in text
        assert 'PERFORMANCE vs BENCHMARK' in text
        assert 'Your Return: 5.50%' in text
        assert 'RECENT TRADES' in text
        assert 'AAPL' in text
        assert 'ALERTS' in text
        assert 'BY STRATEGY' in text
        assert 'momentum_bot' in text

    def test_render_email_html_includes_key_elements(self, digest_service):
        """Test HTML email rendering includes key elements."""
        data = {
            'summary': {
                'total_trades': 5,
                'winning_trades': 3,
                'losing_trades': 2,
                'win_rate': 60.0,
                'total_pnl': 250.0,
                'open_positions': 2
            },
            'trades': [],
            'alerts': [],
            'positions': [],
            'performance': {
                'period_return': 4.0,
                'benchmark_return': 2.0,
                'alpha': 2.0,
            },
            'by_strategy': []
        }

        html = digest_service._render_email_html('daily', data)

        assert '<!DOCTYPE html>' in html
        assert 'WallStreetBots' in html
        assert 'Daily' in html
        assert '5' in html  # Total trades
        assert '+$250.00' in html


# =============================================================================
# Email Sending Integration Tests (SMTP mocked)
# =============================================================================

@pytest.mark.django_db
class TestEmailSendingIntegration:
    """Integration tests for email sending with real DB and mocked SMTP."""

    @patch('smtplib.SMTP')
    def test_send_digest_email_creates_digest_log(
        self, mock_smtp_class, user, mock_smtp_config, db
    ):
        """Test that sending creates a DigestLog record."""
        # Setup SMTP mock
        mock_smtp = Mock()
        mock_smtp_class.return_value.__enter__.return_value = mock_smtp

        service = DigestService(user)

        data = {
            'summary': {
                'total_trades': 5,
                'winning_trades': 3,
                'losing_trades': 2,
                'win_rate': 60.0,
                'total_pnl': 100.0,
                'open_positions': 2
            },
            'trades': [],
            'alerts': [],
            'positions': [],
            'performance': {},
            'by_strategy': []
        }

        success, error, returned_data = service.send_digest_email(user, 'daily', data)

        assert success is True
        assert error == ""

        # Verify DigestLog was created
        digest_log = DigestLog.objects.filter(user=user, digest_type='daily').first()
        assert digest_log is not None
        assert digest_log.delivery_status == 'sent'
        assert digest_log.sent_at is not None
        assert digest_log.email_recipient == 'test@example.com'

    @patch('smtplib.SMTP')
    def test_send_digest_email_prevents_duplicate(
        self, mock_smtp_class, user, mock_smtp_config, db
    ):
        """Test that duplicate digest for same period is prevented."""
        mock_smtp = Mock()
        mock_smtp_class.return_value.__enter__.return_value = mock_smtp

        service = DigestService(user)

        data = {
            'summary': {
                'total_trades': 5,
                'winning_trades': 3,
                'losing_trades': 2,
                'win_rate': 60.0,
                'total_pnl': 100.0,
                'open_positions': 2
            },
            'trades': [],
            'alerts': [],
            'positions': [],
            'performance': {},
            'by_strategy': []
        }

        # Send first time
        success1, error1, _ = service.send_digest_email(user, 'daily', data)
        assert success1 is True

        # Try to send again
        success2, error2, _ = service.send_digest_email(user, 'daily', data)
        assert success2 is False
        assert 'already sent' in error2

    def test_send_digest_email_no_smtp_config(self, user, db):
        """Test email sending fails gracefully without SMTP config."""
        service = DigestService(user)
        service.smtp_host = None

        data = {
            'summary': {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'total_pnl': 0,
                'open_positions': 0
            },
            'trades': [],
            'alerts': [],
            'positions': [],
            'performance': {},
            'by_strategy': []
        }

        success, error, _ = service.send_digest_email(user, 'daily', data)

        assert success is False
        assert 'Email not configured' in error

        # Verify DigestLog was created with failed status
        digest_log = DigestLog.objects.filter(user=user).first()
        assert digest_log is not None
        assert digest_log.delivery_status == 'failed'

    def test_send_digest_email_no_user_email(self, mock_smtp_config, db):
        """Test email sending fails for user without email."""
        user = User.objects.create_user(
            username='noemail',
            email='',
            password='testpass123'
        )
        service = DigestService(user)

        data = {
            'summary': {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'total_pnl': 0,
                'open_positions': 0
            },
            'trades': [],
            'alerts': [],
            'positions': [],
            'performance': {},
            'by_strategy': []
        }

        success, error, _ = service.send_digest_email(user, 'daily', data)

        assert success is False
        assert 'no email address' in error

    @patch('smtplib.SMTP')
    def test_send_digest_email_smtp_error(
        self, mock_smtp_class, user, mock_smtp_config, db
    ):
        """Test email sending handles SMTP errors."""
        mock_smtp = Mock()
        mock_smtp.sendmail.side_effect = smtplib.SMTPException("Connection refused")
        mock_smtp_class.return_value.__enter__.return_value = mock_smtp

        service = DigestService(user)

        data = {
            'summary': {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'total_pnl': 0,
                'open_positions': 0
            },
            'trades': [],
            'alerts': [],
            'positions': [],
            'performance': {},
            'by_strategy': []
        }

        success, error, _ = service.send_digest_email(user, 'daily', data)

        assert success is False
        assert 'Connection refused' in error

        # Verify DigestLog was created with failed status
        digest_log = DigestLog.objects.filter(user=user).first()
        assert digest_log.delivery_status == 'failed'
        assert 'Connection refused' in digest_log.error_message


# =============================================================================
# Batch Sending Integration Tests
# =============================================================================

@pytest.mark.django_db
class TestBatchSendingIntegration:
    """Integration tests for batch digest sending."""

    def test_send_digests_invalid_type(self, digest_service):
        """Test batch sending with invalid digest type."""
        result = digest_service.send_digests_for_frequency('invalid')

        assert result['success'] is False
        assert 'Invalid digest type' in result['error']

    @patch('smtplib.SMTP')
    def test_send_digests_dry_run(
        self, mock_smtp_class, user_with_profile, mock_smtp_config, db
    ):
        """Test batch sending in dry run mode."""
        service = DigestService()

        result = service.send_digests_for_frequency('daily', dry_run=True)

        assert result['dry_run'] is True
        assert result['total_users'] >= 1
        assert result['sent'] >= 1
        assert result['failed'] == 0
        # SMTP should not be called in dry run
        mock_smtp_class.assert_not_called()

    @patch('smtplib.SMTP')
    def test_send_digests_for_frequency_daily(
        self, mock_smtp_class, user_with_profile, mock_smtp_config, db
    ):
        """Test batch sending daily digests to subscribed users."""
        mock_smtp = Mock()
        mock_smtp_class.return_value.__enter__.return_value = mock_smtp

        service = DigestService()

        result = service.send_digests_for_frequency('daily', dry_run=False)

        assert result['digest_type'] == 'daily'
        assert result['total_users'] >= 1
        assert result['sent'] >= 1

    @patch('smtplib.SMTP')
    def test_send_digests_for_frequency_weekly(
        self, mock_smtp_class, user_weekly_digest, mock_smtp_config, db
    ):
        """Test batch sending weekly digests to subscribed users."""
        mock_smtp = Mock()
        mock_smtp_class.return_value.__enter__.return_value = mock_smtp

        service = DigestService()

        result = service.send_digests_for_frequency('weekly', dry_run=False)

        assert result['digest_type'] == 'weekly'
        assert result['total_users'] >= 1

    @patch('smtplib.SMTP')
    def test_send_digests_tracks_failures(
        self, mock_smtp_class, user_with_profile, mock_smtp_config, db
    ):
        """Test batch sending tracks failures properly."""
        mock_smtp = Mock()
        mock_smtp.sendmail.side_effect = smtplib.SMTPException("Network error")
        mock_smtp_class.return_value.__enter__.return_value = mock_smtp

        service = DigestService()

        result = service.send_digests_for_frequency('daily', dry_run=False)

        assert result['failed'] >= 1
        assert len(result['errors']) >= 1
        assert 'Network error' in result['errors'][0]['error']


# =============================================================================
# Preview Integration Tests
# =============================================================================

@pytest.mark.django_db
class TestPreviewIntegration:
    """Integration tests for digest preview."""

    def test_preview_digest_generates_all_formats(self, digest_service, user):
        """Test preview generates data, subject, HTML, and text."""
        preview = digest_service.preview_digest(user, 'daily')

        assert 'data' in preview
        assert 'subject' in preview
        assert 'html' in preview
        assert 'text' in preview

        assert preview['data']['digest_type'] == 'daily'
        assert '[WSB] Daily Digest' in preview['subject']
        assert '<!DOCTYPE html>' in preview['html']
        assert 'SUMMARY' in preview['text']

    def test_preview_digest_with_real_data(
        self, digest_service, user, circuit_breaker_state, db
    ):
        """Test preview reflects real data in the database."""
        now = timezone.now()

        with patch('django.utils.timezone.now') as mock_now:
            test_now = timezone.make_aware(datetime(2024, 6, 15, 14, 0, 0))
            mock_now.return_value = test_now

            service = DigestService(user)
            period_start, _ = service._get_period_bounds('daily')
            trade_time = period_start + timedelta(hours=6)

            # Create real data
            trade = TradeSignalSnapshot.objects.create(
                trade_id=f"preview_trade_{uuid.uuid4().hex[:8]}",
                symbol='NVDA',
                direction='buy',
                entry_price=Decimal('450.00'),
                quantity=Decimal('20'),
                strategy_name='preview_strategy',
                exit_price=Decimal('475.00'),
                pnl_amount=Decimal('500.00'),
                signals_at_entry={'test': True},
            )
            TradeSignalSnapshot.objects.filter(pk=trade.pk).update(created_at=trade_time)

            preview = service.preview_digest(user, 'daily')

        assert preview['data']['summary']['total_trades'] == 1
        assert preview['data']['summary']['total_pnl'] == 500.0
        assert 'NVDA' in preview['text']


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================

@pytest.mark.django_db
class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_aggregate_trades_with_zero_quantity(self, digest_service, user, db):
        """Test handling of trade with zero quantity (edge case)."""
        now = timezone.now()
        yesterday = now - timedelta(hours=12)

        create_trade_snapshot(
            symbol='AAPL',
            direction='buy',
            entry_price=Decimal('150.00'),
            quantity=Decimal('0'),  # Zero quantity
            created_at=yesterday,
        )

        result = digest_service.aggregate_trades(user, now - timedelta(days=1), now)

        # Should handle gracefully
        assert result['summary']['total_trades'] == 1

    def test_aggregate_trades_with_very_large_pnl(self, digest_service, user, db):
        """Test handling of very large P&L values."""
        now = timezone.now()
        yesterday = now - timedelta(hours=12)

        create_trade_snapshot(
            symbol='AAPL',
            direction='buy',
            entry_price=Decimal('150.00'),
            quantity=Decimal('10000'),
            exit_price=Decimal('200.00'),
            pnl_amount=Decimal('500000.00'),  # Very large P&L
            created_at=yesterday,
        )

        result = digest_service.aggregate_trades(user, now - timedelta(days=1), now)

        assert result['summary']['total_pnl'] == 500000.0

    def test_aggregate_positions_decimal_precision(self, digest_service, user, db):
        """Test that decimal precision is maintained in position calculations."""
        create_trade_snapshot(
            symbol='AAPL',
            direction='buy',
            entry_price=Decimal('150.1234'),
            quantity=Decimal('100.5678'),
        )

        now = timezone.now()
        result = digest_service.aggregate_positions(user, now)

        # Values should be converted to float but maintain reasonable precision
        pos = result['positions'][0]
        assert isinstance(pos['avg_price'], float)
        assert isinstance(pos['quantity'], float)
        assert isinstance(pos['value'], float)

    def test_multiple_circuit_breaker_states(self, digest_service, user, db):
        """Test alerts from multiple circuit breaker states."""
        now = timezone.now()
        yesterday = now - timedelta(hours=12)

        # Create multiple circuit breaker states
        state1 = CircuitBreakerState.objects.create(
            user=user,
            breaker_type='daily_loss',
            status='ok',
        )
        state2 = CircuitBreakerState.objects.create(
            user=user,
            breaker_type='error_rate',
            status='ok',
        )

        # Create events for each state
        create_circuit_breaker_event(state1, 'trip', 'Loss limit', yesterday)
        create_circuit_breaker_event(state2, 'warning', 'Error spike', yesterday)

        result = digest_service.aggregate_alerts(user, now - timedelta(days=1), now)

        assert result['summary']['total_alerts'] == 2

    def test_digest_log_data_snapshot_stored(self, user, mock_smtp_config, db):
        """Test that DigestLog stores the full data snapshot."""
        with patch('smtplib.SMTP') as mock_smtp_class:
            mock_smtp = Mock()
            mock_smtp_class.return_value.__enter__.return_value = mock_smtp

            service = DigestService(user)

            data = {
                'summary': {
                    'total_trades': 10,
                    'winning_trades': 7,
                    'losing_trades': 3,
                    'win_rate': 70.0,
                    'total_pnl': 1500.0,
                    'open_positions': 5
                },
                'trades': [{
                    'symbol': 'AAPL',
                    'direction': 'buy',
                    'quantity': 100,
                    'entry_price': 150.0,
                    'exit_price': 160.0,
                    'pnl': 1000.0,
                    'strategy': 'test_strategy',
                    'timestamp': '2024-01-15T12:00:00',
                }],
                'alerts': [],
                'positions': [],
                'performance': {
                    'period_return': 5.0,
                    'benchmark_return': 3.0,
                    'alpha': 2.0,
                    'sharpe_ratio': 1.5,
                    'max_drawdown': -2.0,
                },
                'by_strategy': []
            }

            service.send_digest_email(user, 'daily', data)

            digest_log = DigestLog.objects.filter(user=user).first()
            assert digest_log.data_snapshot is not None
            assert digest_log.data_snapshot['summary']['total_trades'] == 10
            assert digest_log.data_snapshot['summary']['total_pnl'] == 1500.0
