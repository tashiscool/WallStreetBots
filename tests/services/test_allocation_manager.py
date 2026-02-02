"""
Comprehensive tests for AllocationManagerService.

Tests allocation management, reservation, enforcement, reconciliation,
and all edge cases.
Target: 80%+ coverage.
"""
import unittest
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock, patch, MagicMock

from django.contrib.auth.models import User
from django.core.exceptions import ObjectDoesNotExist
from django.test import TestCase
from django.utils import timezone

from backend.auth0login.services.allocation_manager import (
    AllocationManagerService,
    AllocationExceededError,
    AllocationInfo,
    RebalanceRecommendation,
    get_allocation_manager,
)


# Create a proper DoesNotExist exception for mocking
class MockDoesNotExist(ObjectDoesNotExist):
    """Mock DoesNotExist exception for testing."""
    pass


class TestAllocationManagerService(TestCase):
    """Test suite for AllocationManagerService."""

    def setUp(self):
        """Set up test fixtures."""
        self.user = Mock(spec=User)
        self.user.id = 1
        self.user.username = "testuser"
        self.service = AllocationManagerService()

    def test_initialization(self):
        """Test service initialization."""
        service = AllocationManagerService()
        self.assertIsNotNone(service.logger)

    def test_default_allocations_exist(self):
        """Test that default allocations are defined."""
        self.assertIn('conservative', AllocationManagerService.DEFAULT_ALLOCATIONS)
        self.assertIn('moderate', AllocationManagerService.DEFAULT_ALLOCATIONS)
        self.assertIn('aggressive', AllocationManagerService.DEFAULT_ALLOCATIONS)

    def test_default_allocations_sum_to_100(self):
        """Test that default allocations sum to 100%."""
        for profile, allocations in AllocationManagerService.DEFAULT_ALLOCATIONS.items():
            total = sum(allocations.values())
            self.assertEqual(total, 100, f"{profile} allocations should sum to 100%")

    @patch('backend.tradingbot.models.models.StrategyAllocationLimit')
    def test_get_strategy_allocation_exists(self, mock_model):
        """Test get_strategy_allocation when allocation exists."""
        mock_model.DoesNotExist = MockDoesNotExist
        mock_allocation = Mock()
        mock_allocation.strategy_name = 'wsb-dip-bot'
        mock_allocation.allocated_pct = Decimal('20.0')
        mock_allocation.allocated_amount = Decimal('20000.0')
        mock_allocation.current_exposure = Decimal('15000.0')
        mock_allocation.reserved_amount = Decimal('2000.0')
        mock_allocation.available_capital = Decimal('3000.0')
        mock_allocation.utilization_pct = 75.0
        mock_allocation.utilization_level = 'high'
        mock_allocation.is_maxed_out = False
        mock_allocation.is_enabled = True

        mock_model.objects.get.return_value = mock_allocation

        result = self.service.get_strategy_allocation(self.user, 'wsb-dip-bot')

        self.assertIsInstance(result, AllocationInfo)
        self.assertEqual(result.strategy_name, 'wsb-dip-bot')
        self.assertEqual(result.allocated_pct, 20.0)
        self.assertEqual(result.available_capital, 3000.0)

    @patch('backend.tradingbot.models.models.StrategyAllocationLimit')
    def test_get_strategy_allocation_not_exists(self, mock_model):
        """Test get_strategy_allocation when allocation doesn't exist."""
        mock_model.objects.get.side_effect = MockDoesNotExist

        result = self.service.get_strategy_allocation(self.user, 'unknown-strategy')

        self.assertIsNone(result)

    @patch('backend.tradingbot.models.models.StrategyAllocationLimit')
    def test_check_allocation_available_no_limit(self, mock_model):
        """Test check_allocation_available when no limit is configured."""
        mock_model.objects.get.side_effect = MockDoesNotExist

        is_available, message = self.service.check_allocation_available(
            self.user, 'strategy1', 5000.0
        )

        self.assertTrue(is_available)
        self.assertIn('No allocation limit', message)

    @patch('backend.tradingbot.models.models.StrategyAllocationLimit')
    def test_check_allocation_available_disabled(self, mock_model):
        """Test check_allocation_available when limits are disabled."""
        mock_allocation = Mock()
        mock_allocation.is_enabled = False
        mock_model.objects.get.return_value = mock_allocation

        is_available, message = self.service.check_allocation_available(
            self.user, 'strategy1', 5000.0
        )

        self.assertTrue(is_available)
        self.assertIn('disabled', message)

    @patch('backend.tradingbot.models.models.StrategyAllocationLimit')
    def test_check_allocation_available_sufficient(self, mock_model):
        """Test check_allocation_available when sufficient capital available."""
        mock_allocation = Mock()
        mock_allocation.is_enabled = True
        mock_allocation.available_capital = Decimal('10000.0')
        mock_model.objects.get.return_value = mock_allocation

        is_available, message = self.service.check_allocation_available(
            self.user, 'strategy1', 5000.0
        )

        self.assertTrue(is_available)
        self.assertIn('available', message.lower())

    @patch('backend.tradingbot.models.models.StrategyAllocationLimit')
    def test_check_allocation_available_insufficient(self, mock_model):
        """Test check_allocation_available when insufficient capital."""
        mock_allocation = Mock()
        mock_allocation.is_enabled = True
        mock_allocation.available_capital = Decimal('3000.0')
        mock_model.objects.get.return_value = mock_allocation

        is_available, message = self.service.check_allocation_available(
            self.user, 'strategy1', 5000.0
        )

        self.assertFalse(is_available)
        self.assertIn('Insufficient', message)

    @patch('backend.auth0login.services.allocation_manager.AllocationReservation')
    @patch('backend.tradingbot.models.models.StrategyAllocationLimit')
    def test_reserve_allocation_success(self, mock_limit_model, mock_reservation_model):
        """Test successful reservation."""
        mock_allocation = Mock()
        mock_allocation.is_enabled = True
        mock_allocation.reserve.return_value = True

        mock_queryset = MagicMock()
        mock_limit_model.objects.select_for_update.return_value = mock_queryset
        mock_queryset.get.return_value = mock_allocation

        result = self.service.reserve_allocation(
            self.user, 'strategy1', 5000.0, 'order123', 'AAPL'
        )

        self.assertTrue(result)
        mock_allocation.reserve.assert_called_once_with(5000.0)
        mock_reservation_model.objects.create.assert_called_once()

    @patch('backend.tradingbot.models.models.StrategyAllocationLimit')
    def test_reserve_allocation_disabled(self, mock_limit_model):
        """Test reservation when limits are disabled."""
        mock_allocation = Mock()
        mock_allocation.is_enabled = False

        mock_queryset = MagicMock()
        mock_limit_model.objects.select_for_update.return_value = mock_queryset
        mock_queryset.get.return_value = mock_allocation

        result = self.service.reserve_allocation(
            self.user, 'strategy1', 5000.0, 'order123', 'AAPL'
        )

        self.assertTrue(result)

    @patch('backend.tradingbot.models.models.StrategyAllocationLimit')
    def test_reserve_allocation_insufficient(self, mock_limit_model):
        """Test reservation when insufficient capital."""
        mock_allocation = Mock()
        mock_allocation.is_enabled = True
        mock_allocation.reserve.return_value = False
        mock_allocation.available_capital = Decimal('1000.0')

        mock_queryset = MagicMock()
        mock_limit_model.objects.select_for_update.return_value = mock_queryset
        mock_queryset.get.return_value = mock_allocation

        result = self.service.reserve_allocation(
            self.user, 'strategy1', 5000.0, 'order123', 'AAPL'
        )

        self.assertFalse(result)

    @patch('backend.tradingbot.models.models.StrategyAllocationLimit')
    def test_reserve_allocation_no_config(self, mock_limit_model):
        """Test reservation when no allocation configured."""
        mock_queryset = MagicMock()
        mock_limit_model.objects.select_for_update.return_value = mock_queryset
        mock_queryset.get.side_effect = mock_limit_model.DoesNotExist

        result = self.service.reserve_allocation(
            self.user, 'strategy1', 5000.0, 'order123', 'AAPL'
        )

        self.assertTrue(result)  # Allows order when no config

    @patch('backend.auth0login.services.allocation_manager.AllocationReservation')
    @patch('backend.tradingbot.models.models.StrategyAllocationLimit')
    def test_release_allocation(self, mock_limit_model, mock_reservation_model):
        """Test releasing allocation."""
        mock_allocation = Mock()
        mock_queryset = MagicMock()
        mock_limit_model.objects.select_for_update.return_value = mock_queryset
        mock_queryset.get.return_value = mock_allocation

        mock_reservation = Mock()
        mock_reservation_model.objects.get.return_value = mock_reservation

        self.service.release_allocation(self.user, 'strategy1', 5000.0, 'order123')

        mock_allocation.release_reservation.assert_called_once_with(5000.0)
        mock_reservation.resolve.assert_called_once_with('cancelled')

    @patch('backend.auth0login.services.allocation_manager.AllocationReservation')
    @patch('backend.tradingbot.models.models.StrategyAllocationLimit')
    def test_release_allocation_no_order_id(self, mock_limit_model, mock_reservation_model):
        """Test releasing allocation without order ID."""
        mock_allocation = Mock()
        mock_queryset = MagicMock()
        mock_limit_model.objects.select_for_update.return_value = mock_queryset
        mock_queryset.get.return_value = mock_allocation

        self.service.release_allocation(self.user, 'strategy1', 5000.0)

        mock_allocation.release_reservation.assert_called_once_with(5000.0)

    @patch('backend.auth0login.services.allocation_manager.AllocationReservation')
    @patch('backend.tradingbot.models.models.StrategyAllocationLimit')
    def test_confirm_allocation(self, mock_limit_model, mock_reservation_model):
        """Test confirming allocation."""
        mock_allocation = Mock()
        mock_queryset = MagicMock()
        mock_limit_model.objects.select_for_update.return_value = mock_queryset
        mock_queryset.get.return_value = mock_allocation

        mock_reservation = Mock()
        mock_reservation_model.objects.get.return_value = mock_reservation

        self.service.confirm_allocation(self.user, 'strategy1', 5000.0, 'order123')

        mock_allocation.add_exposure.assert_called_once_with(5000.0)
        mock_reservation.resolve.assert_called_once_with('filled')

    @patch('backend.tradingbot.models.models.StrategyAllocationLimit')
    def test_reduce_exposure(self, mock_limit_model):
        """Test reducing exposure."""
        mock_allocation = Mock()
        mock_queryset = MagicMock()
        mock_limit_model.objects.select_for_update.return_value = mock_queryset
        mock_queryset.get.return_value = mock_allocation

        self.service.reduce_exposure(self.user, 'strategy1', 3000.0)

        mock_allocation.remove_exposure.assert_called_once_with(3000.0)

    @patch('backend.tradingbot.models.models.StrategyAllocationLimit')
    def test_recalculate_all_allocations(self, mock_limit_model):
        """Test recalculating all allocations."""
        mock_allocation1 = Mock()
        mock_allocation2 = Mock()

        mock_queryset = MagicMock()
        mock_limit_model.objects.filter.return_value = [mock_allocation1, mock_allocation2]

        self.service.recalculate_all_allocations(self.user, 150000.0)

        mock_allocation1.recalculate_allocated_amount.assert_called_once_with(150000.0)
        mock_allocation2.recalculate_allocated_amount.assert_called_once_with(150000.0)

    @patch('backend.tradingbot.models.models.StrategyAllocationLimit')
    def test_get_allocation_summary_not_configured(self, mock_limit_model):
        """Test get_allocation_summary when not configured."""
        mock_queryset = MagicMock()
        mock_limit_model.objects.filter.return_value = mock_queryset
        mock_queryset.exists.return_value = False

        result = self.service.get_allocation_summary(self.user)

        self.assertFalse(result['configured'])
        self.assertEqual(result['total_allocated_pct'], 0)

    @patch('backend.tradingbot.models.models.StrategyAllocationLimit')
    def test_get_allocation_summary_configured(self, mock_limit_model):
        """Test get_allocation_summary with configured allocations."""
        mock_allocation = Mock()
        mock_allocation.allocated_pct = Decimal('20.0')
        mock_allocation.current_exposure = Decimal('18000.0')
        mock_allocation.available_capital = Decimal('2000.0')
        mock_allocation.utilization_level = 'normal'
        mock_allocation.to_dict.return_value = {
            'strategy_name': 'strategy1',
            'utilization_pct': 90.0
        }

        mock_queryset = MagicMock()
        mock_limit_model.objects.filter.return_value = [mock_allocation]
        mock_queryset.exists.return_value = True

        result = self.service.get_allocation_summary(self.user)

        self.assertTrue(result['configured'])
        self.assertGreater(result['total_allocated_pct'], 0)
        self.assertIn('strategies', result)

    def test_get_allocation_warnings(self):
        """Test _get_allocation_warnings."""
        mock_allocation1 = Mock()
        mock_allocation1.strategy_name = 'strategy1'
        mock_allocation1.utilization_level = 'maxed_out'
        mock_allocation1.utilization_pct = 100.0

        mock_allocation2 = Mock()
        mock_allocation2.strategy_name = 'strategy2'
        mock_allocation2.utilization_level = 'critical'
        mock_allocation2.utilization_pct = 95.0

        mock_allocation3 = Mock()
        mock_allocation3.strategy_name = 'strategy3'
        mock_allocation3.utilization_level = 'normal'
        mock_allocation3.utilization_pct = 50.0

        allocations = [mock_allocation1, mock_allocation2, mock_allocation3]
        warnings = self.service._get_allocation_warnings(allocations)

        self.assertEqual(len(warnings), 2)
        self.assertTrue(any('maxed out' in w for w in warnings))
        self.assertTrue(any('near limit' in w for w in warnings))

    @patch('backend.tradingbot.models.models.StrategyAllocationLimit')
    def test_reconcile_allocations(self, mock_limit_model):
        """Test reconciling allocations with actual positions."""
        mock_allocation = Mock()
        mock_allocation.strategy_name = 'strategy1'
        mock_allocation.current_exposure = Decimal('10000.0')
        mock_allocation.last_reconciled = None

        mock_queryset = MagicMock()
        mock_limit_model.objects.filter.return_value = [mock_allocation]

        positions = [
            {'strategy': 'strategy1', 'symbol': 'AAPL', 'market_value': 11000.0},
            {'strategy': 'strategy1', 'symbol': 'MSFT', 'market_value': 2000.0},
        ]

        with patch('backend.auth0login.services.allocation_manager.timezone') as mock_tz:
            mock_tz.now.return_value = datetime.now()

            report = self.service.reconcile_allocations(self.user, positions)

            self.assertIn('adjustments', report)
            self.assertGreater(len(report['adjustments']), 0)

    @patch('backend.tradingbot.models.models.StrategyAllocationLimit')
    def test_initialize_allocations_conservative(self, mock_limit_model):
        """Test initializing allocations with conservative profile."""
        mock_limit_model.objects.update_or_create.return_value = (Mock(), True)

        self.service.initialize_allocations(self.user, 'conservative', 100000.0)

        # Should have created allocations for conservative profile
        call_count = mock_limit_model.objects.update_or_create.call_count
        self.assertGreater(call_count, 0)

    @patch('backend.tradingbot.models.models.StrategyAllocationLimit')
    def test_initialize_allocations_with_enabled_filter(self, mock_limit_model):
        """Test initializing allocations with enabled strategies filter."""
        mock_limit_model.objects.update_or_create.return_value = (Mock(), True)

        enabled_strategies = ['index-baseline', 'wheel']
        self.service.initialize_allocations(
            self.user, 'conservative', 100000.0, enabled_strategies=enabled_strategies
        )

        # Should only create for enabled strategies
        call_count = mock_limit_model.objects.update_or_create.call_count
        self.assertLessEqual(call_count, len(enabled_strategies))

    @patch('backend.tradingbot.models.models.StrategyAllocationLimit')
    def test_update_allocation(self, mock_limit_model):
        """Test updating allocation percentage."""
        mock_allocation = Mock()
        mock_limit_model.objects.get_or_create.return_value = (mock_allocation, False)

        self.service.update_allocation(
            self.user, 'strategy1', 25.0, portfolio_value=100000.0
        )

        self.assertEqual(mock_allocation.allocated_pct, Decimal('25.0'))
        self.assertEqual(mock_allocation.allocated_amount, Decimal('25000.0'))
        mock_allocation.save.assert_called_once()

    @patch('backend.tradingbot.models.models.StrategyAllocationLimit')
    def test_get_rebalance_recommendations(self, mock_limit_model):
        """Test generating rebalance recommendations."""
        mock_allocation = Mock()
        mock_allocation.strategy_name = 'strategy1'
        mock_allocation.current_exposure = Decimal('15000.0')
        mock_allocation.allocated_pct = Decimal('20.0')

        mock_queryset = MagicMock()
        mock_limit_model.objects.filter.return_value = [mock_allocation]

        recommendations = self.service.get_rebalance_recommendations(
            self.user, 100000.0
        )

        self.assertIsInstance(recommendations, list)
        if len(recommendations) > 0:
            self.assertIsInstance(recommendations[0], RebalanceRecommendation)

    @patch('backend.tradingbot.models.models.StrategyAllocationLimit')
    def test_get_rebalance_recommendations_with_target_profile(self, mock_limit_model):
        """Test rebalance recommendations with target profile."""
        mock_allocation = Mock()
        mock_allocation.strategy_name = 'index-baseline'
        mock_allocation.current_exposure = Decimal('10000.0')
        mock_allocation.allocated_pct = Decimal('10.0')

        mock_queryset = MagicMock()
        mock_limit_model.objects.filter.return_value = [mock_allocation]

        recommendations = self.service.get_rebalance_recommendations(
            self.user, 100000.0, target_profile='aggressive'
        )

        self.assertIsInstance(recommendations, list)

    @patch('backend.tradingbot.models.models.StrategyAllocationLimit')
    def test_enforce_allocation_no_config(self, mock_limit_model):
        """Test enforce_allocation when no config exists."""
        mock_limit_model.objects.get.side_effect = mock_limit_model.DoesNotExist

        result = self.service.enforce_allocation(
            self.user, 'strategy1', 5000.0
        )

        self.assertTrue(result['allowed'])
        self.assertIn('No allocation limit', result['message'])

    @patch('backend.tradingbot.models.models.StrategyAllocationLimit')
    def test_enforce_allocation_disabled(self, mock_limit_model):
        """Test enforce_allocation when limits disabled."""
        mock_allocation = Mock()
        mock_allocation.is_enabled = False
        mock_limit_model.objects.get.return_value = mock_allocation

        result = self.service.enforce_allocation(
            self.user, 'strategy1', 5000.0
        )

        self.assertTrue(result['allowed'])
        self.assertIn('disabled', result['message'])

    @patch('backend.tradingbot.models.models.StrategyAllocationLimit')
    def test_enforce_allocation_exceeded(self, mock_limit_model):
        """Test enforce_allocation when limit exceeded."""
        mock_allocation = Mock()
        mock_allocation.is_enabled = True
        mock_allocation.available_capital = Decimal('3000.0')
        mock_limit_model.objects.get.return_value = mock_allocation

        with self.assertRaises(AllocationExceededError) as context:
            self.service.enforce_allocation(
                self.user, 'strategy1', 5000.0
            )

        self.assertEqual(context.exception.strategy_name, 'strategy1')
        self.assertEqual(context.exception.available, 3000.0)
        self.assertEqual(context.exception.requested, 5000.0)

    @patch('backend.tradingbot.models.models.StrategyAllocationLimit')
    def test_enforce_allocation_with_reservation(self, mock_limit_model):
        """Test enforce_allocation with reservation."""
        mock_allocation = Mock()
        mock_allocation.is_enabled = True
        mock_allocation.available_capital = Decimal('10000.0')
        mock_limit_model.objects.get.return_value = mock_allocation

        with patch.object(self.service, 'reserve_allocation', return_value=True):
            result = self.service.enforce_allocation(
                self.user, 'strategy1', 5000.0, order_id='order123', symbol='AAPL'
            )

            self.assertTrue(result['allowed'])

    @patch('backend.tradingbot.models.models.StrategyAllocationLimit')
    def test_enforce_allocation_reservation_failed(self, mock_limit_model):
        """Test enforce_allocation when reservation fails."""
        mock_allocation = Mock()
        mock_allocation.is_enabled = True
        mock_allocation.available_capital = Decimal('10000.0')
        mock_limit_model.objects.get.return_value = mock_allocation

        with patch.object(self.service, 'reserve_allocation', return_value=False):
            with self.assertRaises(AllocationExceededError):
                self.service.enforce_allocation(
                    self.user, 'strategy1', 5000.0, order_id='order123', symbol='AAPL'
                )

    def test_get_allocation_manager_singleton(self):
        """Test get_allocation_manager returns singleton."""
        manager1 = get_allocation_manager()
        manager2 = get_allocation_manager()

        self.assertIs(manager1, manager2)
        self.assertIsInstance(manager1, AllocationManagerService)

    def test_allocation_exceeded_error(self):
        """Test AllocationExceededError exception."""
        error = AllocationExceededError('strategy1', 3000.0, 5000.0)

        self.assertEqual(error.strategy_name, 'strategy1')
        self.assertEqual(error.available, 3000.0)
        self.assertEqual(error.requested, 5000.0)
        self.assertIn('strategy1', str(error))
        self.assertIn('3,000', str(error))

    def test_allocation_exceeded_error_custom_message(self):
        """Test AllocationExceededError with custom message."""
        custom_msg = "Custom error message"
        error = AllocationExceededError('strategy1', 3000.0, 5000.0, message=custom_msg)

        self.assertEqual(error.message, custom_msg)


if __name__ == '__main__':
    unittest.main()
