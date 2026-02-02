"""Comprehensive integration tests for CircuitBreakerPersistence.

These tests use actual database operations to verify real persistence behavior.
"""

import pytest
from datetime import timedelta
from decimal import Decimal
from unittest.mock import Mock, patch

from django.contrib.auth.models import User
from django.utils import timezone

from backend.auth0login.services.circuit_breaker_persistence import (
    CircuitBreakerPersistence,
    get_circuit_breaker_persistence,
    restore_circuit_breaker_on_startup,
    sync_circuit_breaker_to_db,
    perform_daily_reset,
    BREAKER_TYPES
)
from backend.tradingbot.models.models import CircuitBreakerState, CircuitBreakerHistory


@pytest.fixture
def user(db):
    """Create a test user."""
    return User.objects.create_user(
        username='testuser',
        email='test@example.com',
        password='testpass123'
    )


@pytest.fixture
def another_user(db):
    """Create another test user for isolation tests."""
    return User.objects.create_user(
        username='anotheruser',
        email='another@example.com',
        password='testpass456'
    )


@pytest.fixture
def service(user):
    """Create a persistence service for the test user."""
    return CircuitBreakerPersistence(user=user)


@pytest.fixture
def global_service():
    """Create a global persistence service (no user)."""
    return CircuitBreakerPersistence(user=None)


@pytest.mark.django_db
class TestInit:
    """Test initialization."""

    def test_init_with_user(self, user):
        service = CircuitBreakerPersistence(user=user)
        assert service.user == user

    def test_init_global(self):
        service = CircuitBreakerPersistence(user=None)
        assert service.user is None


@pytest.mark.django_db
class TestGetOrCreateState:
    """Test state creation and retrieval."""

    def test_creates_new_state(self, service, user):
        """Test that a new state is created in the database."""
        state = service.get_or_create_state('daily_loss')

        assert state is not None
        assert state.breaker_type == 'daily_loss'
        assert state.user == user
        assert state.status == 'ok'

        # Verify it's actually in the database
        db_state = CircuitBreakerState.objects.get(
            user=user, breaker_type='daily_loss'
        )
        assert db_state.id == state.id

    def test_returns_existing_state(self, service, user):
        """Test that existing state is returned, not duplicated."""
        state1 = service.get_or_create_state('daily_loss')
        state2 = service.get_or_create_state('daily_loss')

        assert state1.id == state2.id
        assert CircuitBreakerState.objects.filter(
            user=user, breaker_type='daily_loss'
        ).count() == 1

    def test_creates_with_defaults(self, service, user):
        """Test state creation with custom defaults."""
        defaults = {
            'status': 'warning',
            'threshold': Decimal('5.0'),
        }
        state = service.get_or_create_state('daily_loss', defaults=defaults)

        assert state.status == 'warning'
        assert state.threshold == Decimal('5.0')

    def test_creates_different_breaker_types(self, service, user):
        """Test creating states for different breaker types."""
        state1 = service.get_or_create_state('daily_loss')
        state2 = service.get_or_create_state('vix')
        state3 = service.get_or_create_state('error_rate')

        assert state1.id != state2.id != state3.id
        assert state1.breaker_type == 'daily_loss'
        assert state2.breaker_type == 'vix'
        assert state3.breaker_type == 'error_rate'

    def test_user_isolation(self, service, another_user):
        """Test that states are isolated per user."""
        state1 = service.get_or_create_state('daily_loss')

        other_service = CircuitBreakerPersistence(user=another_user)
        state2 = other_service.get_or_create_state('daily_loss')

        assert state1.id != state2.id
        assert state1.user != state2.user


@pytest.mark.django_db
class TestGetState:
    """Test getting existing state."""

    def test_get_state_found(self, service, user):
        """Test retrieving an existing state."""
        created_state = service.get_or_create_state('daily_loss')
        retrieved_state = service.get_state('daily_loss')

        assert retrieved_state is not None
        assert retrieved_state.id == created_state.id

    def test_get_state_not_found(self, service):
        """Test retrieving non-existent state returns None."""
        result = service.get_state('nonexistent')
        assert result is None

    def test_get_state_user_specific(self, service, another_user):
        """Test that get_state respects user context."""
        service.get_or_create_state('daily_loss')

        other_service = CircuitBreakerPersistence(user=another_user)
        result = other_service.get_state('daily_loss')

        # Should not find the state because it belongs to different user
        assert result is None


@pytest.mark.django_db
class TestGetAllStates:
    """Test getting all states for a user."""

    def test_get_all_states_empty(self, service):
        """Test getting all states when none exist."""
        result = service.get_all_states()
        assert result == []

    def test_get_all_states(self, service):
        """Test getting all states for a user."""
        service.get_or_create_state('daily_loss')
        service.get_or_create_state('vix')
        service.get_or_create_state('error_rate')

        result = service.get_all_states()

        assert len(result) == 3
        breaker_types = {s.breaker_type for s in result}
        assert breaker_types == {'daily_loss', 'vix', 'error_rate'}

    def test_get_all_states_user_isolation(self, service, another_user):
        """Test that get_all_states only returns user's states."""
        service.get_or_create_state('daily_loss')
        service.get_or_create_state('vix')

        other_service = CircuitBreakerPersistence(user=another_user)
        other_service.get_or_create_state('error_rate')

        result = service.get_all_states()
        assert len(result) == 2

        other_result = other_service.get_all_states()
        assert len(other_result) == 1


@pytest.mark.django_db
class TestGetTrippedStates:
    """Test getting tripped states."""

    def test_get_tripped_states_none_tripped(self, service):
        """Test when no states are tripped."""
        service.get_or_create_state('daily_loss')
        service.get_or_create_state('vix')

        result = service.get_tripped_states()
        assert result == []

    def test_get_tripped_states(self, service, user):
        """Test getting only tripped states."""
        state1 = service.get_or_create_state('daily_loss')
        state2 = service.get_or_create_state('vix')
        state3 = service.get_or_create_state('error_rate')

        # Trip two states
        state1.status = 'triggered'
        state1.save()
        state3.status = 'critical'
        state3.save()

        result = service.get_tripped_states()

        assert len(result) == 2
        tripped_types = {s.breaker_type for s in result}
        assert tripped_types == {'daily_loss', 'error_rate'}


@pytest.mark.django_db
class TestTripBreaker:
    """Test tripping a circuit breaker."""

    def test_trip_breaker_creates_state_if_not_exists(self, service, user):
        """Test that trip_breaker creates state if it doesn't exist."""
        result = service.trip_breaker(
            breaker_type='daily_loss',
            reason='Loss limit exceeded',
            value=5.5,
            threshold=5.0,
        )

        assert result is not None
        assert result.status == 'triggered'
        assert result.trip_reason == 'Loss limit exceeded'

    def test_trip_breaker_updates_state(self, service, user):
        """Test that trip_breaker updates the state correctly."""
        # Create initial state
        initial_state = service.get_or_create_state('daily_loss')
        assert initial_state.status == 'ok'

        # Trip the breaker
        result = service.trip_breaker(
            breaker_type='daily_loss',
            reason='Loss limit exceeded',
            value=5.5,
            threshold=5.0,
            cooldown_seconds=1800,
        )

        # Verify state was updated
        assert result.status == 'triggered'
        assert result.trip_reason == 'Loss limit exceeded'
        assert result.current_value == Decimal('5.5')
        assert result.threshold == Decimal('5.0')
        assert result.tripped_at is not None
        assert result.cooldown_until is not None

        # Verify changes persisted to database
        db_state = CircuitBreakerState.objects.get(id=result.id)
        assert db_state.status == 'triggered'

    def test_trip_breaker_creates_history(self, service, user):
        """Test that trip_breaker creates a history entry."""
        result = service.trip_breaker(
            breaker_type='daily_loss',
            reason='Loss limit exceeded',
            value=5.5,
            threshold=5.0,
        )

        # Verify history was created
        history_entries = CircuitBreakerHistory.objects.filter(state=result)
        assert history_entries.count() == 1

        history = history_entries.first()
        assert history.action == 'trip'
        assert history.reason == 'Loss limit exceeded'
        assert history.value_at_action == Decimal('5.5')
        assert history.threshold_at_action == Decimal('5.0')
        assert history.new_status == 'triggered'

    def test_trip_breaker_with_metadata(self, service, user):
        """Test that trip_breaker stores metadata in history."""
        metadata = {'symbol': 'AAPL', 'position_size': 100}
        result = service.trip_breaker(
            breaker_type='daily_loss',
            reason='Loss limit exceeded',
            value=5.5,
            threshold=5.0,
            metadata=metadata,
        )

        history = CircuitBreakerHistory.objects.filter(state=result).first()
        assert history.metadata == metadata


@pytest.mark.django_db
class TestResetBreaker:
    """Test resetting a circuit breaker."""

    def test_reset_breaker_not_found(self, service):
        """Test resetting non-existent breaker."""
        result = service.reset_breaker('nonexistent')

        assert result['success'] is False
        assert result['error'] == 'State not found'

    def test_reset_breaker_success(self, service, user):
        """Test successful reset of a tripped breaker."""
        # Trip the breaker first
        state = service.trip_breaker(
            breaker_type='daily_loss',
            reason='Loss limit exceeded',
            cooldown_seconds=0,  # No cooldown for testing
        )
        assert state.status == 'triggered'

        # Reset the breaker
        result = service.reset_breaker('daily_loss', reason='Manual reset')

        assert result['success'] is True
        assert result['previous_status'] == 'triggered'
        assert result['new_status'] == 'ok'

        # Verify state was updated in database
        db_state = CircuitBreakerState.objects.get(id=state.id)
        assert db_state.status == 'ok'
        assert db_state.tripped_at is None
        assert db_state.cooldown_until is None

    def test_reset_breaker_in_cooldown(self, service, user):
        """Test that reset fails when in cooldown."""
        # Trip with long cooldown
        state = service.trip_breaker(
            breaker_type='daily_loss',
            reason='Loss limit exceeded',
            cooldown_seconds=3600,  # 1 hour cooldown
        )

        # Attempt to reset without force
        result = service.reset_breaker('daily_loss', force=False)

        assert result['success'] is False
        assert result['error'] == 'Still in cooldown period'
        assert 'cooldown_remaining_seconds' in result

        # Verify state unchanged
        db_state = CircuitBreakerState.objects.get(id=state.id)
        assert db_state.status == 'triggered'

    def test_reset_breaker_force(self, service, user):
        """Test force reset bypasses cooldown."""
        # Trip with long cooldown
        state = service.trip_breaker(
            breaker_type='daily_loss',
            reason='Loss limit exceeded',
            cooldown_seconds=3600,
        )

        # Force reset
        result = service.reset_breaker('daily_loss', force=True, reason='Admin override')

        assert result['success'] is True

        # Verify state was reset
        db_state = CircuitBreakerState.objects.get(id=state.id)
        assert db_state.status == 'ok'

    def test_reset_breaker_creates_history(self, service, user):
        """Test that reset creates a history entry."""
        state = service.trip_breaker(
            breaker_type='daily_loss',
            reason='Loss limit exceeded',
            cooldown_seconds=0,
        )

        service.reset_breaker('daily_loss', reason='Manual reset')

        # Get reset history entry
        history_entries = CircuitBreakerHistory.objects.filter(
            state=state, action='reset'
        )
        assert history_entries.count() == 1

        history = history_entries.first()
        assert history.reason == 'Manual reset'
        assert history.previous_status == 'triggered'
        assert history.new_status == 'ok'


@pytest.mark.django_db
class TestUpdateEquity:
    """Test equity tracking updates."""

    def test_update_equity(self, service, user):
        """Test updating equity and calculating drawdown."""
        result = service.update_equity(
            breaker_type='daily_loss',
            current_equity=9500.00,
            start_of_day_equity=10000.00,
        )

        assert result['breaker_type'] == 'daily_loss'
        assert result['current_equity'] == 9500.00
        assert result['start_of_day_equity'] == 10000.00
        # Drawdown = 1 - (9500/10000) = 0.05 (5%)
        assert result['daily_drawdown'] == pytest.approx(0.05, rel=1e-4)

        # Verify persisted to database
        state = service.get_state('daily_loss')
        assert float(state.current_equity) == 9500.00
        assert float(state.start_of_day_equity) == 10000.00
        assert float(state.daily_drawdown) == pytest.approx(0.05, rel=1e-4)

    def test_update_equity_subsequent_updates(self, service, user):
        """Test subsequent equity updates."""
        # First update sets start of day
        service.update_equity(
            breaker_type='daily_loss',
            current_equity=10000.00,
            start_of_day_equity=10000.00,
        )

        # Second update only changes current equity
        result = service.update_equity(
            breaker_type='daily_loss',
            current_equity=9000.00,
        )

        assert result['daily_drawdown'] == pytest.approx(0.10, rel=1e-4)  # 10% drawdown


@pytest.mark.django_db
class TestUpdateVIX:
    """Test VIX tracking."""

    def test_update_vix_normal(self, service, user):
        """Test VIX update below elevated threshold."""
        result = service.update_vix(20.0)

        assert result['vix_value'] == 20.0
        assert result['vix_level'] == 'none'
        assert result['changed'] is False  # Default is 'none'

    def test_update_vix_elevated(self, service, user):
        """Test VIX update to elevated level."""
        # First update to set baseline
        service.update_vix(20.0)

        # Update to elevated
        result = service.update_vix(28.0)

        assert result['vix_level'] == 'elevated'
        assert result['changed'] is True
        assert result['position_multiplier'] == 0.5

        # Verify in database
        state = service.get_state('vix')
        assert state.vix_level == 'elevated'
        assert float(state.current_vix) == 28.0

    def test_update_vix_extreme(self, service, user):
        """Test VIX update to extreme level."""
        result = service.update_vix(38.0)

        assert result['vix_level'] == 'extreme'
        assert result['position_multiplier'] == 0.25

    def test_update_vix_critical(self, service, user):
        """Test VIX update to critical level."""
        result = service.update_vix(50.0)

        assert result['vix_level'] == 'critical'
        assert result['position_multiplier'] == 0.0

    def test_update_vix_creates_history_on_change(self, service, user):
        """Test that VIX level changes create history entries."""
        service.update_vix(20.0)  # none
        service.update_vix(30.0)  # elevated

        state = service.get_state('vix')
        history_entries = CircuitBreakerHistory.objects.filter(
            state=state, action='vix_change'
        )
        assert history_entries.count() == 1

        history = history_entries.first()
        assert 'VIX level changed' in history.reason


@pytest.mark.django_db
class TestMarkError:
    """Test error rate tracking."""

    def test_mark_error_increments_count(self, service, user):
        """Test that mark_error increments error count."""
        result = service.mark_error()

        assert result['errors_last_minute'] == 1
        assert result['tripped'] is False
        assert result['status'] == 'ok'

        # Verify in database
        state = service.get_state('error_rate')
        assert state.errors_last_minute == 1

    def test_mark_error_trips_on_threshold(self, service, user):
        """Test that reaching error threshold trips the breaker."""
        # Mark errors up to threshold
        for i in range(4):
            result = service.mark_error(max_per_minute=5)
            assert result['tripped'] is False

        # This one should trip
        result = service.mark_error(max_per_minute=5)
        assert result['tripped'] is True
        assert result['status'] == 'triggered'

        # Verify in database
        state = service.get_state('error_rate')
        assert state.status == 'triggered'

    def test_mark_error_creates_history_on_trip(self, service, user):
        """Test that error spike creates history entry."""
        for _ in range(5):
            service.mark_error(max_per_minute=5)

        state = service.get_state('error_rate')
        history_entries = CircuitBreakerHistory.objects.filter(
            state=state, action='error_spike'
        )
        assert history_entries.count() == 1


@pytest.mark.django_db
class TestMarkDataFresh:
    """Test data freshness marking."""

    def test_mark_data_fresh(self, service, user):
        """Test marking data as fresh."""
        state = service.get_or_create_state('stale_data')
        assert state.last_data_timestamp is None

        service.mark_data_fresh('stale_data')

        # Refresh from database
        state.refresh_from_db()
        assert state.last_data_timestamp is not None


@pytest.mark.django_db
class TestDailyResetAll:
    """Test daily reset functionality."""

    def test_daily_reset_all_resets_counters(self, service, user):
        """Test that daily reset clears daily counters."""
        # Setup states with data
        state1 = service.get_or_create_state('daily_loss')
        state1.daily_drawdown = Decimal('0.05')
        state1.errors_last_minute = 3
        state1.error_window_start = timezone.now()
        state1.save()

        state2 = service.get_or_create_state('error_rate')
        state2.errors_last_minute = 2
        state2.error_window_start = timezone.now()
        state2.save()

        # Perform daily reset
        results = service.daily_reset_all(new_equity=10000.00)

        assert len(results) == 2

        # Verify states were reset
        state1.refresh_from_db()
        assert float(state1.daily_drawdown) == 0
        assert state1.errors_last_minute == 0
        assert state1.error_window_start is None
        assert float(state1.start_of_day_equity) == 10000.00

        state2.refresh_from_db()
        assert state2.errors_last_minute == 0
        assert state2.error_window_start is None

    def test_daily_reset_all_creates_history(self, service, user):
        """Test that daily reset creates history entries."""
        service.get_or_create_state('daily_loss')
        service.get_or_create_state('vix')

        service.daily_reset_all(new_equity=10000.00)

        history_entries = CircuitBreakerHistory.objects.filter(action='daily_reset')
        assert history_entries.count() == 2

    def test_daily_reset_does_not_clear_trips(self, service, user):
        """Test that daily reset does not auto-clear tripped breakers."""
        # Trip a breaker
        state = service.trip_breaker(
            breaker_type='daily_loss',
            reason='Loss limit exceeded',
        )
        assert state.status == 'triggered'

        # Daily reset
        service.daily_reset_all(new_equity=10000.00)

        # Verify trip is preserved
        state.refresh_from_db()
        assert state.status == 'triggered'

    def test_daily_reset_only_once_per_day(self, service, user):
        """Test that daily reset only happens once per day."""
        state = service.get_or_create_state('daily_loss')
        state.errors_last_minute = 5
        state.save()

        # First reset
        service.daily_reset_all(new_equity=10000.00)
        state.refresh_from_db()
        assert state.errors_last_minute == 0
        assert state.last_daily_reset == timezone.now().date()

        # Add more errors
        state.errors_last_minute = 3
        state.save()

        # Second reset same day - should not reset
        service.daily_reset_all(new_equity=11000.00)
        state.refresh_from_db()
        assert state.errors_last_minute == 3  # Unchanged


@pytest.mark.django_db
class TestSyncFromMemory:
    """Test syncing from in-memory CircuitBreaker to database."""

    def test_sync_from_memory_not_tripped(self, service, user):
        """Test syncing a non-tripped breaker."""
        mock_breaker = Mock()
        mock_breaker.tripped_at = None
        mock_breaker.errors_last_min = 2
        mock_breaker.errors_window_start = None
        mock_breaker.current_vix = 22.5
        mock_breaker.vix_level = Mock(value='elevated')
        mock_breaker.last_vix_check = None
        mock_breaker.start_equity = 10000.0
        mock_breaker.last_data_ts = None

        result = service.sync_from_memory(mock_breaker)

        assert result.status == 'ok'
        assert result.errors_last_minute == 2
        assert float(result.current_vix) == 22.5
        assert result.vix_level == 'elevated'
        assert float(result.start_of_day_equity) == 10000.0

    def test_sync_from_memory_tripped(self, service, user):
        """Test syncing a tripped breaker."""
        import time

        trip_time = time.time()
        mock_breaker = Mock()
        mock_breaker.tripped_at = trip_time
        mock_breaker.reason = 'Test trip reason'
        mock_breaker.lim = Mock(cooldown_sec=1800)
        mock_breaker.errors_last_min = 5
        mock_breaker.errors_window_start = None
        mock_breaker.current_vix = None
        mock_breaker.vix_level = None
        mock_breaker.last_vix_check = None
        mock_breaker.start_equity = None
        mock_breaker.last_data_ts = None

        result = service.sync_from_memory(mock_breaker)

        assert result.status == 'triggered'
        assert result.trip_reason == 'Test trip reason'
        assert result.tripped_at is not None
        assert result.cooldown_until is not None


@pytest.mark.django_db
class TestRestoreToMemory:
    """Test restoring database state to in-memory CircuitBreaker."""

    def test_restore_to_memory_no_state(self, service, user):
        """Test restore when no state exists."""
        mock_breaker = Mock()
        result = service.restore_to_memory(mock_breaker)

        assert result is False

    def test_restore_to_memory_tripped_state(self, service, user):
        """Test restoring a tripped state."""
        # Create tripped state in database
        state = service.get_or_create_state('global')
        state.status = 'triggered'
        state.tripped_at = timezone.now()
        state.trip_reason = 'Persisted trip'
        state.errors_last_minute = 3
        state.current_vix = Decimal('25.5')
        state.vix_level = 'elevated'
        state.start_of_day_equity = Decimal('10000.00')
        state.last_data_timestamp = timezone.now()
        state.save()

        mock_breaker = Mock()
        mock_breaker.vix_level = None

        with patch('backend.auth0login.services.circuit_breaker_persistence.VIXTripLevel') as mock_vix:
            mock_vix.return_value = Mock()
            result = service.restore_to_memory(mock_breaker)

        assert result is True
        assert mock_breaker.tripped_at is not None
        assert mock_breaker.reason == 'Persisted trip'
        assert mock_breaker.errors_last_min == 3
        assert mock_breaker.current_vix == 25.5
        assert mock_breaker.start_equity == 10000.0

    def test_restore_to_memory_ok_state(self, service, user):
        """Test restoring a non-tripped state."""
        # Create OK state in database
        state = service.get_or_create_state('global')
        state.status = 'ok'
        state.current_vix = Decimal('20.0')
        state.vix_level = 'none'
        state.last_data_timestamp = timezone.now()
        state.save()

        mock_breaker = Mock()
        mock_breaker.is_tripped = False

        with patch('backend.auth0login.services.circuit_breaker_persistence.VIXTripLevel') as mock_vix:
            mock_vix.return_value = Mock()
            result = service.restore_to_memory(mock_breaker)

        assert result is False  # Not tripped
        assert mock_breaker.current_vix == 20.0


@pytest.mark.django_db
class TestGetHistory:
    """Test history retrieval."""

    def test_get_history_empty(self, service, user):
        """Test getting history when none exists."""
        result = service.get_history()
        assert result == []

    def test_get_history(self, service, user):
        """Test getting history entries."""
        # Create some history by tripping and resetting
        service.trip_breaker(
            breaker_type='daily_loss',
            reason='Loss limit exceeded',
            value=5.5,
            threshold=5.0,
            cooldown_seconds=0,
        )
        service.reset_breaker('daily_loss', reason='Manual reset')

        result = service.get_history()

        assert len(result) == 2
        actions = {entry['action'] for entry in result}
        assert actions == {'trip', 'reset'}

    def test_get_history_filtered_by_breaker_type(self, service, user):
        """Test filtering history by breaker type."""
        service.trip_breaker(breaker_type='daily_loss', reason='Test', cooldown_seconds=0)
        service.trip_breaker(breaker_type='vix', reason='Test', cooldown_seconds=0)

        result = service.get_history(breaker_type='daily_loss')

        assert len(result) == 1
        assert result[0]['breaker_type'] == 'daily_loss'

    def test_get_history_filtered_by_actions(self, service, user):
        """Test filtering history by action type."""
        service.trip_breaker(breaker_type='daily_loss', reason='Test', cooldown_seconds=0)
        service.reset_breaker('daily_loss')

        result = service.get_history(actions=['trip'])

        assert len(result) == 1
        assert result[0]['action'] == 'trip'

    def test_get_history_respects_limit(self, service, user):
        """Test history limit parameter."""
        for _ in range(5):
            service.trip_breaker(breaker_type='daily_loss', reason='Test', cooldown_seconds=0)
            service.reset_breaker('daily_loss', force=True)

        result = service.get_history(limit=3)
        assert len(result) == 3


@pytest.mark.django_db
class TestGetStatusSummary:
    """Test status summary generation."""

    def test_get_status_summary_empty(self, service):
        """Test status summary with no states."""
        result = service.get_status_summary()

        assert result['overall_status'] == 'ok'
        assert result['can_trade'] is True
        assert result['position_size_multiplier'] == 1.0
        assert result['tripped_count'] == 0

    def test_get_status_summary_ok(self, service, user):
        """Test status summary when all OK."""
        service.get_or_create_state('daily_loss')
        service.get_or_create_state('vix')

        result = service.get_status_summary()

        assert result['overall_status'] == 'ok'
        assert result['can_trade'] is True
        assert len(result['breakers']) == 2

    def test_get_status_summary_triggered(self, service, user):
        """Test status summary with triggered breaker."""
        service.get_or_create_state('vix')
        service.trip_breaker(
            breaker_type='daily_loss',
            reason='Test',
            cooldown_seconds=3600,
        )

        result = service.get_status_summary()

        assert result['overall_status'] == 'triggered'
        assert result['can_trade'] is False
        assert result['tripped_count'] == 1
        assert 'daily_loss' in result['tripped_types']

    def test_get_status_summary_critical(self, service, user):
        """Test status summary with critical breaker."""
        state = service.get_or_create_state('daily_loss')
        state.status = 'critical'
        state.save()

        result = service.get_status_summary()

        assert result['overall_status'] == 'critical'

    def test_get_status_summary_position_multiplier(self, service, user):
        """Test that status summary returns minimum position multiplier."""
        state1 = service.get_or_create_state('daily_loss')
        state1.vix_level = 'none'  # 1.0 multiplier
        state1.save()

        state2 = service.get_or_create_state('vix')
        state2.vix_level = 'elevated'  # 0.5 multiplier
        state2.save()

        result = service.get_status_summary()

        assert result['position_size_multiplier'] == 0.5


@pytest.mark.django_db
class TestInitializeDefaultStates:
    """Test default state initialization."""

    def test_initialize_default_states(self, service, user):
        """Test that all default breaker types are initialized."""
        created = service.initialize_default_states()

        assert len(created) == len(BREAKER_TYPES)

        # Verify all types exist
        all_states = service.get_all_states()
        breaker_types = {s.breaker_type for s in all_states}
        assert breaker_types == set(BREAKER_TYPES)

    def test_initialize_default_states_idempotent(self, service, user):
        """Test that initialization is idempotent."""
        created1 = service.initialize_default_states()
        created2 = service.initialize_default_states()

        assert len(created1) == len(BREAKER_TYPES)
        assert len(created2) == 0  # No new states created

        # Still only have the default number of states
        all_states = service.get_all_states()
        assert len(all_states) == len(BREAKER_TYPES)


@pytest.mark.django_db
class TestFactoryFunctions:
    """Test factory functions and convenience utilities."""

    def test_get_circuit_breaker_persistence_with_user(self, user):
        """Test factory function with user."""
        service = get_circuit_breaker_persistence(user)

        assert isinstance(service, CircuitBreakerPersistence)
        assert service.user == user

    def test_get_circuit_breaker_persistence_global(self):
        """Test factory function for global service."""
        service = get_circuit_breaker_persistence(None)

        assert isinstance(service, CircuitBreakerPersistence)
        assert service.user is None

    def test_restore_circuit_breaker_on_startup(self, user):
        """Test restore on startup function."""
        # Create tripped state
        service = get_circuit_breaker_persistence(user)
        state = service.get_or_create_state('global')
        state.status = 'triggered'
        state.tripped_at = timezone.now()
        state.trip_reason = 'Startup test'
        state.save()

        mock_breaker = Mock()
        mock_breaker.vix_level = None

        with patch('backend.auth0login.services.circuit_breaker_persistence.VIXTripLevel'):
            result = restore_circuit_breaker_on_startup(mock_breaker, user)

        assert result is True
        assert mock_breaker.reason == 'Startup test'

    def test_sync_circuit_breaker_to_db(self, user):
        """Test sync to database function."""
        mock_breaker = Mock()
        mock_breaker.tripped_at = None
        mock_breaker.errors_last_min = 0
        mock_breaker.errors_window_start = None
        mock_breaker.current_vix = None
        mock_breaker.vix_level = None
        mock_breaker.last_vix_check = None
        mock_breaker.start_equity = None
        mock_breaker.last_data_ts = None

        result = sync_circuit_breaker_to_db(mock_breaker, user)

        assert result is not None
        assert isinstance(result, CircuitBreakerState)
        assert result.user == user

    def test_perform_daily_reset(self, user):
        """Test daily reset utility function."""
        service = get_circuit_breaker_persistence(user)
        service.get_or_create_state('daily_loss')
        service.get_or_create_state('vix')

        results = perform_daily_reset(user, new_equity=10000.0)

        assert len(results) == 2


@pytest.mark.django_db
class TestGlobalVsUserStates:
    """Test isolation between global and user-specific states."""

    def test_global_state_creation(self, global_service):
        """Test creating global state (no user)."""
        state = global_service.get_or_create_state('daily_loss')

        assert state.user is None
        assert state.breaker_type == 'daily_loss'

    def test_global_and_user_states_isolated(self, service, global_service, user):
        """Test that global and user states are separate."""
        user_state = service.get_or_create_state('daily_loss')
        global_state = global_service.get_or_create_state('daily_loss')

        assert user_state.id != global_state.id
        assert user_state.user == user
        assert global_state.user is None

    def test_get_all_states_respects_user_context(self, service, global_service, user):
        """Test that get_all_states returns correct states based on context."""
        service.get_or_create_state('daily_loss')
        service.get_or_create_state('vix')
        global_service.get_or_create_state('error_rate')

        user_states = service.get_all_states()
        global_states = global_service.get_all_states()

        assert len(user_states) == 2
        assert len(global_states) == 1
        assert all(s.user == user for s in user_states)
        assert all(s.user is None for s in global_states)


@pytest.mark.django_db
class TestConcurrentOperations:
    """Test concurrent/transactional behavior."""

    def test_trip_breaker_atomic(self, service, user):
        """Test that trip_breaker is atomic."""
        # Trip the breaker
        result = service.trip_breaker(
            breaker_type='daily_loss',
            reason='Test',
            value=5.5,
            threshold=5.0,
        )

        # Both state and history should exist
        state = CircuitBreakerState.objects.get(id=result.id)
        history = CircuitBreakerHistory.objects.filter(state=state, action='trip')

        assert state.status == 'triggered'
        assert history.count() == 1

    def test_reset_breaker_atomic(self, service, user):
        """Test that reset_breaker is atomic."""
        state = service.trip_breaker(
            breaker_type='daily_loss',
            reason='Test',
            cooldown_seconds=0,
        )

        service.reset_breaker('daily_loss')

        # Both state update and history should exist
        state.refresh_from_db()
        history = CircuitBreakerHistory.objects.filter(state=state, action='reset')

        assert state.status == 'ok'
        assert history.count() == 1
