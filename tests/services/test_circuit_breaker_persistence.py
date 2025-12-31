"""Comprehensive tests for CircuitBreakerPersistence."""

import pytest
from datetime import datetime, timedelta
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


@pytest.fixture
def user():
    return User.objects.create_user(username='testuser', email='test@example.com')


@pytest.fixture
def service(user):
    return CircuitBreakerPersistence(user=user)


@pytest.fixture
def global_service():
    return CircuitBreakerPersistence(user=None)


class TestInit:
    """Test initialization."""

    def test_init_with_user(self, user):
        service = CircuitBreakerPersistence(user=user)
        assert service.user == user

    def test_init_global(self):
        service = CircuitBreakerPersistence(user=None)
        assert service.user is None


class TestGetOrCreateState:
    """Test state creation."""

    @patch('backend.auth0login.services.circuit_breaker_persistence.CircuitBreakerState')
    def test_get_or_create_state_creates(self, mock_model, service, user):
        mock_state = Mock()
        mock_model.objects.get_or_create.return_value = (mock_state, True)

        result = service.get_or_create_state('daily_loss')

        assert result == mock_state

    @patch('backend.auth0login.services.circuit_breaker_persistence.CircuitBreakerState')
    def test_get_or_create_state_gets(self, mock_model, service, user):
        mock_state = Mock()
        mock_model.objects.get_or_create.return_value = (mock_state, False)

        result = service.get_or_create_state('daily_loss')

        assert result == mock_state


class TestGetState:
    """Test getting state."""

    @patch('backend.auth0login.services.circuit_breaker_persistence.CircuitBreakerState')
    def test_get_state_found(self, mock_model, service, user):
        mock_state = Mock()
        mock_model.objects.get.return_value = mock_state

        result = service.get_state('daily_loss')

        assert result == mock_state

    @patch('backend.auth0login.services.circuit_breaker_persistence.CircuitBreakerState')
    def test_get_state_not_found(self, mock_model, service, user):
        mock_model.objects.get.side_effect = mock_model.DoesNotExist

        result = service.get_state('nonexistent')

        assert result is None


class TestGetAllStates:
    """Test getting all states."""

    @patch('backend.auth0login.services.circuit_breaker_persistence.CircuitBreakerState')
    def test_get_all_states(self, mock_model, service, user):
        mock_qs = Mock()
        mock_qs.__iter__ = Mock(return_value=iter([Mock(), Mock()]))
        mock_model.objects.filter.return_value = mock_qs

        result = service.get_all_states()

        assert isinstance(result, list)


class TestTripBreaker:
    """Test tripping breaker."""

    @patch('backend.auth0login.services.circuit_breaker_persistence.CircuitBreakerHistory')
    @patch('backend.auth0login.services.circuit_breaker_persistence.CircuitBreakerState')
    def test_trip_breaker(self, mock_state_model, mock_history_model, service, user):
        mock_state = Mock()
        mock_state.status = 'ok'
        mock_state_model.objects.get_or_create.return_value = (mock_state, False)
        mock_history_model.log_action = Mock()

        result = service.trip_breaker(
            breaker_type='daily_loss',
            reason='Loss limit exceeded',
            value=5.5,
            threshold=5.0
        )

        assert result == mock_state
        assert mock_state.status == 'triggered'
        mock_history_model.log_action.assert_called_once()


class TestResetBreaker:
    """Test resetting breaker."""

    @patch('backend.auth0login.services.circuit_breaker_persistence.CircuitBreakerHistory')
    @patch('backend.auth0login.services.circuit_breaker_persistence.CircuitBreakerState')
    def test_reset_breaker_not_found(self, mock_state_model, mock_history_model, service, user):
        mock_state_model.objects.get.side_effect = mock_state_model.DoesNotExist

        result = service.reset_breaker('daily_loss')

        assert result['success'] is False

    @patch('backend.auth0login.services.circuit_breaker_persistence.CircuitBreakerHistory')
    @patch('backend.auth0login.services.circuit_breaker_persistence.CircuitBreakerState')
    def test_reset_breaker_success(self, mock_state_model, mock_history_model, service, user):
        mock_state = Mock()
        mock_state.status = 'triggered'
        mock_state.is_in_cooldown = False
        mock_state.reset.return_value = True
        mock_state_model.objects.get.return_value = mock_state
        mock_history_model.log_action = Mock()

        result = service.reset_breaker('daily_loss', force=False)

        assert result['success'] is True


class TestUpdateEquity:
    """Test equity updates."""

    @patch('backend.auth0login.services.circuit_breaker_persistence.CircuitBreakerState')
    def test_update_equity(self, mock_state_model, service, user):
        mock_state = Mock()
        mock_state.start_of_day_equity = Decimal('10000')
        mock_state.update_equity.return_value = Decimal('2.5')
        mock_state_model.objects.get_or_create.return_value = (mock_state, False)

        result = service.update_equity('daily_loss', 9750.00, 10000.00)

        assert result['daily_drawdown'] == Decimal('2.5')


class TestUpdateVIX:
    """Test VIX updates."""

    @patch('backend.auth0login.services.circuit_breaker_persistence.CircuitBreakerHistory')
    @patch('backend.auth0login.services.circuit_breaker_persistence.CircuitBreakerState')
    def test_update_vix(self, mock_state_model, mock_history_model, service, user):
        mock_state = Mock()
        mock_state.vix_level = 'normal'
        mock_state.update_vix = Mock()
        mock_state_model.objects.get_or_create.return_value = (mock_state, False)
        mock_history_model.log_action = Mock()

        result = service.update_vix(25.5)

        assert 'vix_value' in result


class TestMarkError:
    """Test error marking."""

    @patch('backend.auth0login.services.circuit_breaker_persistence.CircuitBreakerHistory')
    @patch('backend.auth0login.services.circuit_breaker_persistence.CircuitBreakerState')
    def test_mark_error(self, mock_state_model, mock_history_model, service, user):
        mock_state = Mock()
        mock_state.errors_last_minute = 3
        mock_state.status = 'ok'
        mock_state.mark_error.return_value = False
        mock_state_model.objects.get_or_create.return_value = (mock_state, False)

        result = service.mark_error()

        assert 'errors_last_minute' in result


class TestDailyResetAll:
    """Test daily reset."""

    @patch('backend.auth0login.services.circuit_breaker_persistence.CircuitBreakerHistory')
    @patch.object(CircuitBreakerPersistence, 'get_all_states')
    def test_daily_reset_all(self, mock_get_states, mock_history_model, service, user):
        mock_state = Mock()
        mock_state.breaker_type = 'daily_loss'
        mock_state.daily_drawdown = Decimal('2.5')
        mock_state.daily_reset = Mock()
        mock_get_states.return_value = [mock_state]
        mock_history_model.log_action = Mock()

        results = service.daily_reset_all(new_equity=10000.00)

        assert len(results) > 0
        mock_state.daily_reset.assert_called_once()


class TestSyncFromMemory:
    """Test syncing from memory."""

    @patch('backend.auth0login.services.circuit_breaker_persistence.CircuitBreakerState')
    def test_sync_from_memory_not_tripped(self, mock_state_model, service, user):
        mock_state = Mock()
        mock_state_model.objects.get_or_create.return_value = (mock_state, False)

        mock_breaker = Mock()
        mock_breaker.tripped_at = None
        mock_breaker.errors_last_min = 0
        mock_breaker.current_vix = None
        mock_breaker.vix_level = None
        mock_breaker.start_equity = None
        mock_breaker.last_data_ts = None
        mock_breaker.errors_window_start = None
        mock_breaker.last_vix_check = None

        result = service.sync_from_memory(mock_breaker)

        assert result == mock_state


class TestRestoreToMemory:
    """Test restoring to memory."""

    @patch('backend.auth0login.services.circuit_breaker_persistence.CircuitBreakerState')
    def test_restore_to_memory_no_state(self, mock_state_model, service, user):
        mock_state_model.objects.get.side_effect = mock_state_model.DoesNotExist

        mock_breaker = Mock()
        result = service.restore_to_memory(mock_breaker)

        assert result is False

    @patch('backend.auth0login.services.circuit_breaker_persistence.CircuitBreakerState')
    def test_restore_to_memory_tripped(self, mock_state_model, service, user):
        mock_state = Mock()
        mock_state.is_tripped = True
        mock_state.tripped_at = timezone.now()
        mock_state.trip_reason = 'Test reason'
        mock_state.errors_last_minute = 0
        mock_state.current_vix = None
        mock_state.vix_level = None
        mock_state.start_of_day_equity = None
        mock_state.last_data_timestamp = None
        mock_state_model.objects.get.return_value = mock_state

        mock_breaker = Mock()
        result = service.restore_to_memory(mock_breaker)

        assert result is True


class TestGetStatusSummary:
    """Test status summary."""

    @patch.object(CircuitBreakerPersistence, 'get_all_states')
    @patch.object(CircuitBreakerPersistence, 'get_tripped_states')
    def test_get_status_summary(self, mock_tripped, mock_all, service):
        mock_state = Mock()
        mock_state.status = 'ok'
        mock_state.can_trade = True
        mock_state.position_size_multiplier = 1.0
        mock_state.is_tripped = False
        mock_state.breaker_type = 'daily_loss'
        mock_state.to_dict.return_value = {}

        mock_all.return_value = [mock_state]
        mock_tripped.return_value = []

        result = service.get_status_summary()

        assert result['overall_status'] == 'ok'
        assert result['can_trade'] is True


class TestFactoryFunctions:
    """Test factory functions."""

    def test_get_circuit_breaker_persistence(self, user):
        service = get_circuit_breaker_persistence(user)
        assert isinstance(service, CircuitBreakerPersistence)
        assert service.user == user

    @patch.object(CircuitBreakerPersistence, 'restore_to_memory')
    def test_restore_circuit_breaker_on_startup(self, mock_restore, user):
        mock_restore.return_value = True
        mock_breaker = Mock()

        result = restore_circuit_breaker_on_startup(mock_breaker, user)

        assert result is True

    @patch.object(CircuitBreakerPersistence, 'sync_from_memory')
    def test_sync_circuit_breaker_to_db(self, mock_sync, user):
        mock_state = Mock()
        mock_sync.return_value = mock_state
        mock_breaker = Mock()

        result = sync_circuit_breaker_to_db(mock_breaker, user)

        assert result == mock_state

    @patch.object(CircuitBreakerPersistence, 'daily_reset_all')
    def test_perform_daily_reset(self, mock_reset, user):
        mock_reset.return_value = []

        result = perform_daily_reset(user, 10000.0)

        assert isinstance(result, list)
