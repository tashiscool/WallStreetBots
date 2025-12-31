"""Comprehensive tests for RecoveryManagerService."""

import pytest
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock, patch
from django.contrib.auth.models import User
from django.utils import timezone

from backend.auth0login.services.recovery_manager import (
    RecoveryManagerService,
    RecoveryMode,
    RecoveryStrategy,
    RecoveryStage,
    RecoverySchedule,
    RecoveryStatus,
    get_recovery_manager,
    DEFAULT_RECOVERY_SCHEDULES
)


@pytest.fixture
def user():
    return User.objects.create_user(username='testuser', email='test@example.com')


@pytest.fixture
def service(user):
    return RecoveryManagerService(user=user)


class TestRecoveryStage:
    """Test RecoveryStage dataclass."""

    def test_to_dict(self):
        stage = RecoveryStage(
            mode=RecoveryMode.RESTRICTED,
            hours_after_trigger=4.0,
            min_profitable_trades=2,
            description='Test stage'
        )
        result = stage.to_dict()

        assert result['mode'] == 'restricted'
        assert result['hours_after_trigger'] == 4.0

    def test_from_dict(self):
        data = {
            'mode': 'cautious',
            'hours_after_trigger': 12.0,
            'min_profitable_trades': 3
        }
        stage = RecoveryStage.from_dict(data)

        assert stage.mode == RecoveryMode.CAUTIOUS
        assert stage.hours_after_trigger == 12.0


class TestRecoverySchedule:
    """Test RecoverySchedule dataclass."""

    def test_to_list(self):
        schedule = RecoverySchedule(stages=[
            RecoveryStage(RecoveryMode.PAUSED, 0),
            RecoveryStage(RecoveryMode.RESTRICTED, 4)
        ])
        result = schedule.to_list()

        assert len(result) == 2
        assert result[0]['mode'] == 'paused'

    def test_from_list(self):
        data = [
            {'mode': 'paused', 'hours_after_trigger': 0},
            {'mode': 'restricted', 'hours_after_trigger': 4}
        ]
        schedule = RecoverySchedule.from_list(data)

        assert len(schedule.stages) == 2


class TestInit:
    """Test service initialization."""

    def test_init_with_user(self, user):
        service = RecoveryManagerService(user=user)
        assert service.user == user


class TestGetRecoverySchedule:
    """Test getting recovery schedules."""

    def test_get_recovery_schedule_daily_loss(self, service):
        schedule = service.get_recovery_schedule('daily_loss')
        assert isinstance(schedule, list)
        assert len(schedule) > 0

    def test_get_recovery_schedule_unknown(self, service):
        schedule = service.get_recovery_schedule('unknown_type')
        assert isinstance(schedule, list)


class TestCreateBreakerEvent:
    """Test creating breaker events."""

    @patch('backend.auth0login.services.recovery_manager.CircuitBreakerEvent')
    def test_create_breaker_event(self, mock_model, service, user):
        mock_event = Mock()
        mock_model.objects.create.return_value = mock_event

        result = service.create_breaker_event(
            breaker_type='daily_loss',
            trigger_value=5.5,
            trigger_threshold=5.0,
            context={'equity': 10000},
            notes='Test event'
        )

        assert result == mock_event
        mock_model.objects.create.assert_called_once()


class TestGetActiveEvents:
    """Test getting active events."""

    @patch('backend.auth0login.services.recovery_manager.CircuitBreakerEvent')
    def test_get_active_events(self, mock_model, service, user):
        mock_qs = Mock()
        mock_qs.order_by.return_value = []
        mock_model.objects.filter.return_value = mock_qs

        result = service.get_active_events()

        assert isinstance(result, list)


class TestGetCurrentMode:
    """Test getting current recovery mode."""

    @patch('backend.auth0login.services.recovery_manager.CircuitBreakerEvent')
    def test_get_current_mode_no_events(self, mock_model, service, user):
        mock_qs = Mock()
        mock_qs.order_by.return_value = []
        mock_model.objects.filter.return_value = mock_qs

        result = service.get_current_mode()

        assert result == RecoveryMode.NORMAL

    @patch('backend.auth0login.services.recovery_manager.CircuitBreakerEvent')
    def test_get_current_mode_with_events(self, mock_model, service, user):
        mock_event = Mock()
        mock_event.current_recovery_mode = 'paused'

        mock_qs = Mock()
        mock_qs.order_by.return_value = [mock_event]
        mock_model.objects.filter.return_value = mock_qs

        result = service.get_current_mode()

        assert result == RecoveryMode.PAUSED


class TestPositionSizeMultiplier:
    """Test position size multiplier."""

    @patch.object(RecoveryManagerService, 'get_current_mode')
    def test_get_position_size_multiplier_paused(self, mock_mode, service):
        mock_mode.return_value = RecoveryMode.PAUSED
        assert service.get_position_size_multiplier() == 0.0

    @patch.object(RecoveryManagerService, 'get_current_mode')
    def test_get_position_size_multiplier_restricted(self, mock_mode, service):
        mock_mode.return_value = RecoveryMode.RESTRICTED
        assert service.get_position_size_multiplier() == 0.25

    @patch.object(RecoveryManagerService, 'get_current_mode')
    def test_get_position_size_multiplier_normal(self, mock_mode, service):
        mock_mode.return_value = RecoveryMode.NORMAL
        assert service.get_position_size_multiplier() == 1.0


class TestCanTrade:
    """Test trading permissions."""

    @patch.object(RecoveryManagerService, 'get_current_mode')
    def test_can_trade_paused(self, mock_mode, service):
        mock_mode.return_value = RecoveryMode.PAUSED
        assert service.can_trade() is False

    @patch.object(RecoveryManagerService, 'get_current_mode')
    def test_can_trade_normal(self, mock_mode, service):
        mock_mode.return_value = RecoveryMode.NORMAL
        assert service.can_trade() is True


class TestRecoveryStatus:
    """Test recovery status."""

    @patch.object(RecoveryManagerService, 'get_active_events')
    def test_get_recovery_status_no_events(self, mock_events, service):
        mock_events.return_value = []

        result = service.get_recovery_status()

        assert result.is_in_recovery is False
        assert result.current_mode == RecoveryMode.NORMAL

    @patch.object(RecoveryManagerService, 'get_active_events')
    @patch.object(RecoveryManagerService, 'get_current_mode')
    @patch.object(RecoveryManagerService, 'get_position_size_multiplier')
    def test_get_recovery_status_with_events(
        self, mock_multiplier, mock_mode, mock_events, service
    ):
        mock_event = Mock()
        mock_event.can_advance_recovery = True
        mock_event.recovery_schedule = []
        mock_event.recovery_stage = 0
        mock_event.to_dict.return_value = {}

        mock_events.return_value = [mock_event]
        mock_mode.return_value = RecoveryMode.RESTRICTED
        mock_multiplier.return_value = 0.25

        result = service.get_recovery_status()

        assert result.is_in_recovery is True
        assert result.position_multiplier == 0.25


class TestAutoRecovery:
    """Test automatic recovery advancement."""

    @patch.object(RecoveryManagerService, 'get_active_events')
    def test_check_auto_recovery(self, mock_events, service):
        mock_event = Mock()
        mock_event.id = 1
        mock_event.breaker_type = 'daily_loss'
        mock_event.current_recovery_mode = 'restricted'
        mock_event.can_advance_recovery = True
        mock_event.advance_recovery_stage.return_value = True
        mock_event.resolved_at = None

        mock_events.return_value = [mock_event]

        results = service.check_auto_recovery()

        assert len(results) > 0
        mock_event.advance_recovery_stage.assert_called_once()


class TestAdvanceRecovery:
    """Test manual recovery advancement."""

    @patch('backend.auth0login.services.recovery_manager.CircuitBreakerEvent')
    def test_advance_recovery_not_found(self, mock_model, service, user):
        mock_model.objects.get.side_effect = mock_model.DoesNotExist

        result = service.advance_recovery(999)

        assert result['success'] is False

    @patch('backend.auth0login.services.recovery_manager.CircuitBreakerEvent')
    def test_advance_recovery_success(self, mock_model, service, user):
        mock_event = Mock()
        mock_event.id = 1
        mock_event.current_recovery_mode = 'paused'
        mock_event.advance_recovery_stage.return_value = True
        mock_event.resolved_at = None
        mock_model.objects.get.return_value = mock_event

        result = service.advance_recovery(1)

        assert result['success'] is True


class TestResetBreaker:
    """Test breaker reset."""

    @patch('backend.auth0login.services.recovery_manager.CircuitBreakerEvent')
    def test_reset_breaker_no_confirmation(self, mock_model, service, user):
        result = service.reset_breaker(1, confirmation='WRONG')

        assert result['success'] is False

    @patch('backend.auth0login.services.recovery_manager.CircuitBreakerEvent')
    def test_reset_breaker_success(self, mock_model, service, user):
        mock_event = Mock()
        mock_event.id = 1
        mock_event.resolve = Mock()
        mock_model.objects.get.return_value = mock_event

        result = service.reset_breaker(1, confirmation='CONFIRM')

        assert result['success'] is True
        mock_event.resolve.assert_called_once()


class TestRecordTrade:
    """Test recording trades during recovery."""

    @patch.object(RecoveryManagerService, 'get_active_events')
    def test_record_trade(self, mock_events, service):
        mock_event = Mock()
        mock_event.record_recovery_trade = Mock()
        mock_events.return_value = [mock_event]

        service.record_trade(is_profitable=True, pnl=100.0)

        mock_event.record_recovery_trade.assert_called_once_with(True, 100.0)


class TestGetRecoveryTimeline:
    """Test recovery timeline."""

    @patch('backend.auth0login.services.recovery_manager.CircuitBreakerEvent')
    def test_get_recovery_timeline_no_events(self, mock_model, service, user):
        mock_qs = Mock()
        mock_qs.order_by.return_value = []
        mock_model.objects.filter.return_value = mock_qs

        result = service.get_recovery_timeline()

        assert result['has_active_events'] is False


class TestFactoryFunction:
    """Test factory function."""

    def test_get_recovery_manager(self, user):
        service = get_recovery_manager(user)
        assert isinstance(service, RecoveryManagerService)
        assert service.user == user
