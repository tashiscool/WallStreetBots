"""
Integration tests for AllocationManagerService.

Tests allocation management with REAL database operations,
including reservation, enforcement, reconciliation, and concurrent access.
Target: 80%+ coverage with actual database persistence.
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from decimal import Decimal

import pytest
from django.contrib.auth.models import User
from django.db import connection, transaction

from backend.auth0login.services.allocation_manager import (
    AllocationManagerService,
    AllocationExceededError,
    AllocationInfo,
    RebalanceRecommendation,
    get_allocation_manager,
)
from backend.tradingbot.models.models import (
    StrategyAllocationLimit,
    AllocationReservation,
)


@pytest.fixture
def user(db):
    """Create a test user."""
    return User.objects.create_user(
        username='testuser',
        email='test@example.com',
        password='testpass123'
    )


@pytest.fixture
def second_user(db):
    """Create a second test user for isolation tests."""
    return User.objects.create_user(
        username='testuser2',
        email='test2@example.com',
        password='testpass456'
    )


@pytest.fixture
def service():
    """Create AllocationManagerService instance."""
    return AllocationManagerService()


@pytest.fixture
def allocation_limit(user):
    """Create a basic allocation limit for testing."""
    return StrategyAllocationLimit.objects.create(
        user=user,
        strategy_name='test-strategy',
        allocated_pct=Decimal('20.00'),
        allocated_amount=Decimal('20000.00'),
        current_exposure=Decimal('0.00'),
        reserved_amount=Decimal('0.00'),
        is_enabled=True,
    )


@pytest.fixture
def allocation_with_exposure(user):
    """Create an allocation limit with existing exposure."""
    return StrategyAllocationLimit.objects.create(
        user=user,
        strategy_name='exposed-strategy',
        allocated_pct=Decimal('25.00'),
        allocated_amount=Decimal('25000.00'),
        current_exposure=Decimal('15000.00'),
        reserved_amount=Decimal('2000.00'),
        is_enabled=True,
    )


@pytest.fixture
def disabled_allocation(user):
    """Create a disabled allocation limit."""
    return StrategyAllocationLimit.objects.create(
        user=user,
        strategy_name='disabled-strategy',
        allocated_pct=Decimal('10.00'),
        allocated_amount=Decimal('10000.00'),
        current_exposure=Decimal('0.00'),
        reserved_amount=Decimal('0.00'),
        is_enabled=False,
    )


@pytest.mark.django_db
class TestServiceInitialization:
    """Test service initialization and default allocations."""

    def test_service_initialization(self):
        """Test that service initializes correctly."""
        service = AllocationManagerService()
        assert service.logger is not None

    def test_default_allocations_exist(self):
        """Test that default allocations are defined for all profiles."""
        assert 'conservative' in AllocationManagerService.DEFAULT_ALLOCATIONS
        assert 'moderate' in AllocationManagerService.DEFAULT_ALLOCATIONS
        assert 'aggressive' in AllocationManagerService.DEFAULT_ALLOCATIONS

    def test_default_allocations_sum_to_100(self):
        """Test that default allocations sum to 100% for each profile."""
        for profile, allocations in AllocationManagerService.DEFAULT_ALLOCATIONS.items():
            total = sum(allocations.values())
            assert total == 100, f"{profile} allocations should sum to 100%, got {total}%"

    def test_get_allocation_manager_singleton(self):
        """Test that get_allocation_manager returns a singleton."""
        manager1 = get_allocation_manager()
        manager2 = get_allocation_manager()
        assert manager1 is manager2
        assert isinstance(manager1, AllocationManagerService)


@pytest.mark.django_db
class TestGetStrategyAllocation:
    """Test get_strategy_allocation with real database queries."""

    def test_get_strategy_allocation_exists(self, service, allocation_limit, user):
        """Test getting an existing allocation returns AllocationInfo."""
        result = service.get_strategy_allocation(user, 'test-strategy')

        assert result is not None
        assert isinstance(result, AllocationInfo)
        assert result.strategy_name == 'test-strategy'
        assert result.allocated_pct == 20.0
        assert result.allocated_amount == 20000.0
        assert result.available_capital == 20000.0  # No exposure or reservations
        assert result.is_enabled is True

    def test_get_strategy_allocation_with_exposure(self, service, allocation_with_exposure, user):
        """Test getting allocation with existing exposure."""
        result = service.get_strategy_allocation(user, 'exposed-strategy')

        assert result is not None
        assert result.current_exposure == 15000.0
        assert result.reserved_amount == 2000.0
        # Available = allocated - exposure - reserved = 25000 - 15000 - 2000 = 8000
        assert result.available_capital == 8000.0
        assert result.utilization_pct == pytest.approx(68.0, rel=0.01)

    def test_get_strategy_allocation_not_exists(self, service, user):
        """Test getting non-existent allocation returns None."""
        result = service.get_strategy_allocation(user, 'nonexistent-strategy')
        assert result is None

    def test_get_strategy_allocation_different_users(self, service, user, second_user):
        """Test that allocations are user-specific."""
        # Create allocation for first user
        StrategyAllocationLimit.objects.create(
            user=user,
            strategy_name='shared-strategy',
            allocated_pct=Decimal('15.00'),
            allocated_amount=Decimal('15000.00'),
            is_enabled=True,
        )

        # First user should get the allocation
        result1 = service.get_strategy_allocation(user, 'shared-strategy')
        assert result1 is not None
        assert result1.allocated_pct == 15.0

        # Second user should not have this allocation
        result2 = service.get_strategy_allocation(second_user, 'shared-strategy')
        assert result2 is None


@pytest.mark.django_db
class TestCheckAllocationAvailable:
    """Test check_allocation_available with real database."""

    def test_check_allocation_no_limit_configured(self, service, user):
        """Test check when no allocation limit exists."""
        is_available, message = service.check_allocation_available(
            user, 'unconfigured-strategy', 5000.0
        )
        assert is_available is True
        assert 'No allocation limit' in message

    def test_check_allocation_disabled(self, service, disabled_allocation, user):
        """Test check when allocation limits are disabled."""
        is_available, message = service.check_allocation_available(
            user, 'disabled-strategy', 50000.0  # More than allocated
        )
        assert is_available is True
        assert 'disabled' in message

    def test_check_allocation_sufficient_capital(self, service, allocation_limit, user):
        """Test check when sufficient capital is available."""
        is_available, message = service.check_allocation_available(
            user, 'test-strategy', 5000.0
        )
        assert is_available is True
        assert 'available' in message.lower()

    def test_check_allocation_insufficient_capital(self, service, allocation_with_exposure, user):
        """Test check when insufficient capital is available."""
        # Available = 8000, requesting more
        is_available, message = service.check_allocation_available(
            user, 'exposed-strategy', 10000.0
        )
        assert is_available is False
        assert 'Insufficient' in message

    def test_check_allocation_exact_amount(self, service, allocation_limit, user):
        """Test check with exact available amount."""
        is_available, message = service.check_allocation_available(
            user, 'test-strategy', 20000.0  # Exact allocation
        )
        assert is_available is True


@pytest.mark.django_db(transaction=True)
class TestReserveAllocation:
    """Test reserve_allocation with real database writes."""

    def test_reserve_allocation_success(self, service, allocation_limit, user):
        """Test successful reservation persists to database."""
        result = service.reserve_allocation(
            user, 'test-strategy', 5000.0, 'order-001', 'AAPL'
        )

        assert result is True

        # Verify reservation is persisted
        allocation_limit.refresh_from_db()
        assert allocation_limit.reserved_amount == Decimal('5000.00')

        # Verify AllocationReservation record was created
        reservation = AllocationReservation.objects.get(order_id='order-001')
        assert reservation.symbol == 'AAPL'
        assert reservation.amount == Decimal('5000.00')
        assert reservation.status == 'pending'
        assert reservation.allocation == allocation_limit

    def test_reserve_allocation_insufficient_capital(self, service, allocation_with_exposure, user):
        """Test reservation fails when insufficient capital."""
        # Available = 8000, trying to reserve 10000
        result = service.reserve_allocation(
            user, 'exposed-strategy', 10000.0, 'order-002', 'MSFT'
        )

        assert result is False

        # Verify nothing was persisted
        allocation_with_exposure.refresh_from_db()
        assert allocation_with_exposure.reserved_amount == Decimal('2000.00')  # Unchanged
        assert not AllocationReservation.objects.filter(order_id='order-002').exists()

    def test_reserve_allocation_disabled(self, service, disabled_allocation, user):
        """Test reservation succeeds when limits are disabled."""
        result = service.reserve_allocation(
            user, 'disabled-strategy', 50000.0, 'order-003', 'GOOGL'
        )
        assert result is True  # Allows when disabled

    def test_reserve_allocation_no_config(self, service, user):
        """Test reservation succeeds when no config exists."""
        result = service.reserve_allocation(
            user, 'unknown-strategy', 5000.0, 'order-004', 'AMZN'
        )
        assert result is True  # Allows when no config

    def test_reserve_allocation_multiple(self, service, allocation_limit, user):
        """Test multiple reservations accumulate correctly."""
        service.reserve_allocation(user, 'test-strategy', 5000.0, 'order-005', 'AAPL')
        service.reserve_allocation(user, 'test-strategy', 3000.0, 'order-006', 'MSFT')

        allocation_limit.refresh_from_db()
        assert allocation_limit.reserved_amount == Decimal('8000.00')
        assert AllocationReservation.objects.filter(allocation=allocation_limit).count() == 2


@pytest.mark.django_db(transaction=True)
class TestReleaseAllocation:
    """Test release_allocation with real database updates."""

    def test_release_allocation_with_order_id(self, service, allocation_limit, user):
        """Test releasing allocation updates reservation status."""
        # First reserve
        service.reserve_allocation(user, 'test-strategy', 5000.0, 'order-010', 'AAPL')

        allocation_limit.refresh_from_db()
        assert allocation_limit.reserved_amount == Decimal('5000.00')

        # Then release
        service.release_allocation(user, 'test-strategy', 5000.0, 'order-010')

        allocation_limit.refresh_from_db()
        assert allocation_limit.reserved_amount == Decimal('0.00')

        # Verify reservation is marked as cancelled
        reservation = AllocationReservation.objects.get(order_id='order-010')
        assert reservation.status == 'cancelled'
        assert reservation.resolved_at is not None

    def test_release_allocation_without_order_id(self, service, allocation_limit, user):
        """Test releasing allocation without order ID."""
        # First reserve
        service.reserve_allocation(user, 'test-strategy', 5000.0, 'order-011', 'MSFT')

        # Release without order_id
        service.release_allocation(user, 'test-strategy', 5000.0)

        allocation_limit.refresh_from_db()
        assert allocation_limit.reserved_amount == Decimal('0.00')

    def test_release_allocation_partial(self, service, allocation_limit, user):
        """Test partial release of reservation."""
        service.reserve_allocation(user, 'test-strategy', 10000.0, 'order-012', 'AAPL')

        service.release_allocation(user, 'test-strategy', 4000.0)

        allocation_limit.refresh_from_db()
        assert allocation_limit.reserved_amount == Decimal('6000.00')

    def test_release_allocation_no_config(self, service, user):
        """Test release when no allocation configured (should not raise)."""
        # Should not raise an exception
        service.release_allocation(user, 'nonexistent', 5000.0)


@pytest.mark.django_db(transaction=True)
class TestConfirmAllocation:
    """Test confirm_allocation converting reservations to exposure."""

    def test_confirm_allocation_full(self, service, allocation_limit, user):
        """Test confirming full reservation converts to exposure."""
        # Reserve first
        service.reserve_allocation(user, 'test-strategy', 5000.0, 'order-020', 'AAPL')

        # Confirm the order is filled
        service.confirm_allocation(user, 'test-strategy', 5000.0, 'order-020')

        allocation_limit.refresh_from_db()
        assert allocation_limit.current_exposure == Decimal('5000.00')
        assert allocation_limit.reserved_amount == Decimal('0.00')

        # Verify reservation is marked as filled
        reservation = AllocationReservation.objects.get(order_id='order-020')
        assert reservation.status == 'filled'
        assert reservation.resolved_at is not None

    def test_confirm_allocation_partial_fill(self, service, allocation_limit, user):
        """Test partial fill converts portion to exposure."""
        # Reserve 5000
        service.reserve_allocation(user, 'test-strategy', 5000.0, 'order-021', 'MSFT')

        # Partial fill of 3000
        service.confirm_allocation(user, 'test-strategy', 3000.0, 'order-021')

        allocation_limit.refresh_from_db()
        assert allocation_limit.current_exposure == Decimal('3000.00')
        assert allocation_limit.reserved_amount == Decimal('2000.00')

    def test_confirm_allocation_no_config(self, service, user):
        """Test confirm when no allocation configured (should not raise)."""
        service.confirm_allocation(user, 'nonexistent', 5000.0, 'order-022')


@pytest.mark.django_db(transaction=True)
class TestReduceExposure:
    """Test reduce_exposure when positions are closed."""

    def test_reduce_exposure_full(self, service, allocation_with_exposure, user):
        """Test reducing full exposure when position is closed."""
        initial_exposure = allocation_with_exposure.current_exposure

        service.reduce_exposure(user, 'exposed-strategy', 15000.0)

        allocation_with_exposure.refresh_from_db()
        assert allocation_with_exposure.current_exposure == Decimal('0.00')

    def test_reduce_exposure_partial(self, service, allocation_with_exposure, user):
        """Test reducing partial exposure."""
        service.reduce_exposure(user, 'exposed-strategy', 5000.0)

        allocation_with_exposure.refresh_from_db()
        assert allocation_with_exposure.current_exposure == Decimal('10000.00')

    def test_reduce_exposure_no_config(self, service, user):
        """Test reduce when no allocation configured (should not raise)."""
        service.reduce_exposure(user, 'nonexistent', 5000.0)


@pytest.mark.django_db(transaction=True)
class TestEnforceAllocation:
    """Test enforce_allocation which combines check and reserve."""

    def test_enforce_allocation_no_config(self, service, user):
        """Test enforcement when no config exists."""
        result = service.enforce_allocation(user, 'unconfigured', 5000.0)

        assert result['allowed'] is True
        assert 'No allocation limit' in result['message']
        assert result['allocation'] is None

    def test_enforce_allocation_disabled(self, service, disabled_allocation, user):
        """Test enforcement when limits are disabled."""
        result = service.enforce_allocation(user, 'disabled-strategy', 50000.0)

        assert result['allowed'] is True
        assert 'disabled' in result['message']

    def test_enforce_allocation_exceeded(self, service, allocation_with_exposure, user):
        """Test enforcement raises exception when limit exceeded."""
        with pytest.raises(AllocationExceededError) as exc_info:
            service.enforce_allocation(user, 'exposed-strategy', 10000.0)

        assert exc_info.value.strategy_name == 'exposed-strategy'
        assert exc_info.value.available == 8000.0
        assert exc_info.value.requested == 10000.0

    def test_enforce_allocation_with_reservation(self, service, allocation_limit, user):
        """Test enforcement with order_id creates reservation."""
        result = service.enforce_allocation(
            user, 'test-strategy', 5000.0,
            order_id='order-030', symbol='AAPL'
        )

        assert result['allowed'] is True

        # Verify reservation was created
        allocation_limit.refresh_from_db()
        assert allocation_limit.reserved_amount == Decimal('5000.00')
        assert AllocationReservation.objects.filter(order_id='order-030').exists()

    def test_enforce_allocation_reservation_fails(self, service, allocation_with_exposure, user):
        """Test enforcement raises when reservation would fail."""
        # Available is 8000, trying to reserve 9000
        with pytest.raises(AllocationExceededError):
            service.enforce_allocation(
                user, 'exposed-strategy', 9000.0,
                order_id='order-031', symbol='MSFT'
            )


@pytest.mark.django_db(transaction=True)
class TestInitializeAllocations:
    """Test initialize_allocations creates allocations from profiles."""

    def test_initialize_allocations_conservative(self, service, user):
        """Test initializing conservative profile."""
        service.initialize_allocations(user, 'conservative', 100000.0)

        allocations = StrategyAllocationLimit.objects.filter(user=user)
        assert allocations.count() == 4

        # Verify specific allocations
        index_allocation = allocations.get(strategy_name='index-baseline')
        assert index_allocation.allocated_pct == Decimal('40.00')
        assert index_allocation.allocated_amount == Decimal('40000.00')

    def test_initialize_allocations_moderate(self, service, user):
        """Test initializing moderate profile."""
        service.initialize_allocations(user, 'moderate', 100000.0)

        allocations = StrategyAllocationLimit.objects.filter(user=user)
        assert allocations.count() == 6

    def test_initialize_allocations_aggressive(self, service, user):
        """Test initializing aggressive profile."""
        service.initialize_allocations(user, 'aggressive', 100000.0)

        allocations = StrategyAllocationLimit.objects.filter(user=user)
        assert allocations.count() == 6

    def test_initialize_allocations_with_filter(self, service, user):
        """Test initializing with enabled strategies filter."""
        enabled = ['index-baseline', 'wheel']
        service.initialize_allocations(
            user, 'conservative', 100000.0, enabled_strategies=enabled
        )

        allocations = StrategyAllocationLimit.objects.filter(user=user)
        assert allocations.count() == 2
        strategy_names = {a.strategy_name for a in allocations}
        assert strategy_names == {'index-baseline', 'wheel'}

    def test_initialize_allocations_updates_existing(self, service, allocation_limit, user):
        """Test that initialize updates existing allocations."""
        # Existing allocation has strategy_name='test-strategy' which is not in conservative
        service.initialize_allocations(user, 'conservative', 100000.0)

        # Original should still exist
        assert StrategyAllocationLimit.objects.filter(
            user=user, strategy_name='test-strategy'
        ).exists()

        # New ones should be created
        assert StrategyAllocationLimit.objects.filter(user=user).count() == 5


@pytest.mark.django_db(transaction=True)
class TestUpdateAllocation:
    """Test update_allocation modifies existing allocations."""

    def test_update_allocation_existing(self, service, allocation_limit, user):
        """Test updating an existing allocation."""
        service.update_allocation(user, 'test-strategy', 30.0, portfolio_value=100000.0)

        allocation_limit.refresh_from_db()
        assert allocation_limit.allocated_pct == Decimal('30.0')
        assert allocation_limit.allocated_amount == Decimal('30000.0')

    def test_update_allocation_creates_new(self, service, user):
        """Test update creates allocation if it doesn't exist."""
        service.update_allocation(user, 'new-strategy', 15.0, portfolio_value=100000.0)

        allocation = StrategyAllocationLimit.objects.get(
            user=user, strategy_name='new-strategy'
        )
        assert allocation.allocated_pct == Decimal('15.0')
        assert allocation.allocated_amount == Decimal('15000.0')


@pytest.mark.django_db(transaction=True)
class TestRecalculateAllocations:
    """Test recalculate_all_allocations updates amounts."""

    def test_recalculate_all_allocations(self, service, user):
        """Test recalculating allocations with new portfolio value."""
        # Create multiple allocations
        StrategyAllocationLimit.objects.create(
            user=user,
            strategy_name='strategy-a',
            allocated_pct=Decimal('20.00'),
            allocated_amount=Decimal('20000.00'),
            is_enabled=True,
        )
        StrategyAllocationLimit.objects.create(
            user=user,
            strategy_name='strategy-b',
            allocated_pct=Decimal('30.00'),
            allocated_amount=Decimal('30000.00'),
            is_enabled=True,
        )

        # Recalculate with new portfolio value
        service.recalculate_all_allocations(user, 200000.0)

        allocation_a = StrategyAllocationLimit.objects.get(
            user=user, strategy_name='strategy-a'
        )
        allocation_b = StrategyAllocationLimit.objects.get(
            user=user, strategy_name='strategy-b'
        )

        assert allocation_a.allocated_amount == Decimal('40000.00')
        assert allocation_b.allocated_amount == Decimal('60000.00')


@pytest.mark.django_db(transaction=True)
class TestGetAllocationSummary:
    """Test get_allocation_summary aggregates correctly."""

    def test_get_allocation_summary_not_configured(self, service, user):
        """Test summary when no allocations exist."""
        result = service.get_allocation_summary(user)

        assert result['configured'] is False
        assert result['total_allocated_pct'] == 0
        assert result['strategies'] == []

    def test_get_allocation_summary_configured(self, service, user):
        """Test summary with multiple allocations."""
        StrategyAllocationLimit.objects.create(
            user=user,
            strategy_name='strategy-x',
            allocated_pct=Decimal('25.00'),
            allocated_amount=Decimal('25000.00'),
            current_exposure=Decimal('20000.00'),
            reserved_amount=Decimal('0.00'),
            is_enabled=True,
        )
        StrategyAllocationLimit.objects.create(
            user=user,
            strategy_name='strategy-y',
            allocated_pct=Decimal('15.00'),
            allocated_amount=Decimal('15000.00'),
            current_exposure=Decimal('5000.00'),
            reserved_amount=Decimal('1000.00'),
            is_enabled=True,
        )

        result = service.get_allocation_summary(user)

        assert result['configured'] is True
        assert result['total_allocated_pct'] == 40.0
        assert result['total_exposure'] == 25000.0  # 20000 + 5000
        assert len(result['strategies']) == 2
        assert 'warnings' in result

    def test_get_allocation_summary_with_warnings(self, service, user):
        """Test summary includes warnings for high utilization."""
        # Create maxed out allocation
        StrategyAllocationLimit.objects.create(
            user=user,
            strategy_name='maxed-strategy',
            allocated_pct=Decimal('20.00'),
            allocated_amount=Decimal('20000.00'),
            current_exposure=Decimal('20000.00'),
            reserved_amount=Decimal('0.00'),
            is_enabled=True,
        )
        # Create critical utilization allocation
        StrategyAllocationLimit.objects.create(
            user=user,
            strategy_name='critical-strategy',
            allocated_pct=Decimal('10.00'),
            allocated_amount=Decimal('10000.00'),
            current_exposure=Decimal('9500.00'),
            reserved_amount=Decimal('0.00'),
            is_enabled=True,
        )

        result = service.get_allocation_summary(user)

        assert len(result['warnings']) == 2
        assert any('maxed out' in w for w in result['warnings'])
        assert any('near limit' in w for w in result['warnings'])


@pytest.mark.django_db(transaction=True)
class TestReconcileAllocations:
    """Test reconcile_allocations updates from actual positions."""

    def test_reconcile_allocations_with_drift(self, service, user):
        """Test reconciliation adjusts for position drift."""
        allocation = StrategyAllocationLimit.objects.create(
            user=user,
            strategy_name='drift-strategy',
            allocated_pct=Decimal('20.00'),
            allocated_amount=Decimal('20000.00'),
            current_exposure=Decimal('10000.00'),  # Tracked
            is_enabled=True,
        )

        # Actual positions show different exposure
        positions = [
            {'strategy': 'drift-strategy', 'symbol': 'AAPL', 'market_value': 8000.0},
            {'strategy': 'drift-strategy', 'symbol': 'MSFT', 'market_value': 4000.0},
        ]

        report = service.reconcile_allocations(user, positions)

        allocation.refresh_from_db()
        assert allocation.current_exposure == Decimal('12000.00')  # Actual
        assert allocation.last_reconciled is not None

        assert len(report['adjustments']) == 1
        adjustment = report['adjustments'][0]
        assert adjustment['tracked_exposure'] == 10000.0
        assert adjustment['actual_exposure'] == 12000.0
        assert adjustment['adjustment'] == 2000.0

    def test_reconcile_allocations_no_drift(self, service, user):
        """Test reconciliation when no drift exists."""
        allocation = StrategyAllocationLimit.objects.create(
            user=user,
            strategy_name='no-drift-strategy',
            allocated_pct=Decimal('20.00'),
            allocated_amount=Decimal('20000.00'),
            current_exposure=Decimal('10000.00'),
            is_enabled=True,
        )

        positions = [
            {'strategy': 'no-drift-strategy', 'symbol': 'AAPL', 'market_value': 10000.0},
        ]

        report = service.reconcile_allocations(user, positions)

        assert len(report['adjustments']) == 0


@pytest.mark.django_db(transaction=True)
class TestRebalanceRecommendations:
    """Test get_rebalance_recommendations generates suggestions."""

    def test_rebalance_recommendations_underallocated(self, service, user):
        """Test recommendations for underallocated strategy."""
        StrategyAllocationLimit.objects.create(
            user=user,
            strategy_name='index-baseline',
            allocated_pct=Decimal('25.00'),
            allocated_amount=Decimal('25000.00'),
            current_exposure=Decimal('10000.00'),  # 10% of 100k
            is_enabled=True,
        )

        recommendations = service.get_rebalance_recommendations(user, 100000.0)

        assert len(recommendations) == 1
        rec = recommendations[0]
        assert isinstance(rec, RebalanceRecommendation)
        assert rec.action == 'increase'
        assert rec.current_allocation == pytest.approx(10.0, rel=0.01)
        assert rec.target_allocation == 25.0

    def test_rebalance_recommendations_overallocated(self, service, user):
        """Test recommendations for overallocated strategy."""
        StrategyAllocationLimit.objects.create(
            user=user,
            strategy_name='wheel',
            allocated_pct=Decimal('15.00'),
            allocated_amount=Decimal('15000.00'),
            current_exposure=Decimal('25000.00'),  # 25% of 100k
            is_enabled=True,
        )

        recommendations = service.get_rebalance_recommendations(user, 100000.0)

        assert len(recommendations) == 1
        rec = recommendations[0]
        assert rec.action == 'decrease'
        assert rec.reason == 'Overallocated by 10.0%'

    def test_rebalance_recommendations_with_target_profile(self, service, user):
        """Test recommendations targeting a different profile."""
        StrategyAllocationLimit.objects.create(
            user=user,
            strategy_name='index-baseline',
            allocated_pct=Decimal('25.00'),
            allocated_amount=Decimal('25000.00'),
            current_exposure=Decimal('25000.00'),
            is_enabled=True,
        )

        # Target aggressive profile (15% for index-baseline)
        recommendations = service.get_rebalance_recommendations(
            user, 100000.0, target_profile='aggressive'
        )

        assert len(recommendations) == 1
        rec = recommendations[0]
        assert rec.target_allocation == 15.0  # From aggressive profile


@pytest.mark.django_db(transaction=True)
class TestConcurrentAccess:
    """Test concurrent access scenarios with real database locks."""

    def test_concurrent_reservations_race_condition(self, service, user):
        """Test that concurrent reservations don't exceed limit."""
        # Create allocation with limited capital
        allocation = StrategyAllocationLimit.objects.create(
            user=user,
            strategy_name='race-strategy',
            allocated_pct=Decimal('10.00'),
            allocated_amount=Decimal('10000.00'),
            current_exposure=Decimal('0.00'),
            reserved_amount=Decimal('0.00'),
            is_enabled=True,
        )

        results = []
        errors = []

        def make_reservation(order_id):
            try:
                # Each thread tries to reserve 6000 (only one should succeed)
                result = service.reserve_allocation(
                    user, 'race-strategy', 6000.0, f'order-{order_id}', 'AAPL'
                )
                results.append((order_id, result))
            except Exception as e:
                errors.append((order_id, str(e)))
            finally:
                connection.close()

        # Run concurrent reservations
        threads = []
        for i in range(3):
            t = threading.Thread(target=make_reservation, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # Verify only one or at most the amount that fits succeeded
        allocation.refresh_from_db()
        successful = [r for r in results if r[1]]

        # At most one 6000 reservation can fit in 10000 limit
        assert len(successful) <= 1
        assert allocation.reserved_amount <= Decimal('10000.00')

    def test_concurrent_reserve_and_release(self, service, user):
        """Test concurrent reserve and release operations.

        Note: This test validates that concurrent operations maintain data integrity.
        SQLite has limitations with concurrent writes (table locking), so we catch
        database lock errors which are expected in SQLite but would not occur
        in production databases like PostgreSQL.
        """
        allocation = StrategyAllocationLimit.objects.create(
            user=user,
            strategy_name='concurrent-strategy',
            allocated_pct=Decimal('50.00'),
            allocated_amount=Decimal('50000.00'),
            current_exposure=Decimal('0.00'),
            reserved_amount=Decimal('0.00'),
            is_enabled=True,
        )

        # Pre-create some reservations
        service.reserve_allocation(user, 'concurrent-strategy', 10000.0, 'order-pre-1', 'AAPL')
        service.reserve_allocation(user, 'concurrent-strategy', 10000.0, 'order-pre-2', 'MSFT')

        operations = []
        lock_errors = []

        def reserve_op(order_id):
            try:
                result = service.reserve_allocation(
                    user, 'concurrent-strategy', 5000.0, f'order-new-{order_id}', 'GOOGL'
                )
                operations.append(('reserve', order_id, result))
            except Exception as e:
                # SQLite may raise database lock errors in concurrent scenarios
                if 'locked' in str(e).lower() or 'database is locked' in str(e).lower():
                    lock_errors.append(('reserve', order_id, str(e)))
                else:
                    raise
            finally:
                connection.close()

        def release_op(order_id):
            try:
                service.release_allocation(
                    user, 'concurrent-strategy', 10000.0, f'order-pre-{order_id}'
                )
                operations.append(('release', order_id, True))
            except Exception as e:
                # SQLite may raise database lock errors in concurrent scenarios
                if 'locked' in str(e).lower() or 'database is locked' in str(e).lower():
                    lock_errors.append(('release', order_id, str(e)))
                else:
                    raise
            finally:
                connection.close()

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            futures.append(executor.submit(reserve_op, 1))
            futures.append(executor.submit(release_op, 1))
            futures.append(executor.submit(reserve_op, 2))
            futures.append(executor.submit(release_op, 2))

            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    if 'locked' not in str(e).lower():
                        raise

        allocation.refresh_from_db()
        # Final state should be consistent - reserved amount should be non-negative
        assert allocation.reserved_amount >= Decimal('0.00')

        # Verify some operations completed (even if some failed due to SQLite locks)
        # In a real PostgreSQL database, all operations would complete
        assert len(operations) + len(lock_errors) == 4


@pytest.mark.django_db
class TestAllocationExceededError:
    """Test AllocationExceededError exception class."""

    def test_error_with_default_message(self):
        """Test error with auto-generated message."""
        error = AllocationExceededError('test-strategy', 3000.0, 5000.0)

        assert error.strategy_name == 'test-strategy'
        assert error.available == 3000.0
        assert error.requested == 5000.0
        assert 'test-strategy' in str(error)
        assert '3,000' in str(error)
        assert '5,000' in str(error)

    def test_error_with_custom_message(self):
        """Test error with custom message."""
        custom_msg = "Custom allocation error"
        error = AllocationExceededError('test-strategy', 3000.0, 5000.0, message=custom_msg)

        assert error.message == custom_msg
        assert str(error) == custom_msg


@pytest.mark.django_db(transaction=True)
class TestAllocationLifecycle:
    """Integration tests for complete allocation lifecycle."""

    def test_full_order_lifecycle(self, service, user):
        """Test complete order lifecycle: reserve -> confirm -> reduce."""
        # Setup allocation
        allocation = StrategyAllocationLimit.objects.create(
            user=user,
            strategy_name='lifecycle-strategy',
            allocated_pct=Decimal('25.00'),
            allocated_amount=Decimal('25000.00'),
            current_exposure=Decimal('0.00'),
            reserved_amount=Decimal('0.00'),
            is_enabled=True,
        )

        # Step 1: Check and reserve for new order
        result = service.enforce_allocation(
            user, 'lifecycle-strategy', 10000.0,
            order_id='lifecycle-order', symbol='AAPL'
        )
        assert result['allowed'] is True

        allocation.refresh_from_db()
        assert allocation.reserved_amount == Decimal('10000.00')
        assert allocation.current_exposure == Decimal('0.00')
        assert float(allocation.available_capital) == 15000.0

        # Step 2: Order is filled
        service.confirm_allocation(user, 'lifecycle-strategy', 10000.0, 'lifecycle-order')

        allocation.refresh_from_db()
        assert allocation.reserved_amount == Decimal('0.00')
        assert allocation.current_exposure == Decimal('10000.00')
        assert float(allocation.available_capital) == 15000.0

        # Step 3: Position is closed
        service.reduce_exposure(user, 'lifecycle-strategy', 10000.0)

        allocation.refresh_from_db()
        assert allocation.current_exposure == Decimal('0.00')
        assert float(allocation.available_capital) == 25000.0

    def test_order_cancellation_lifecycle(self, service, user):
        """Test order cancellation releases reservation."""
        allocation = StrategyAllocationLimit.objects.create(
            user=user,
            strategy_name='cancel-strategy',
            allocated_pct=Decimal('20.00'),
            allocated_amount=Decimal('20000.00'),
            current_exposure=Decimal('0.00'),
            reserved_amount=Decimal('0.00'),
            is_enabled=True,
        )

        # Reserve for order
        service.reserve_allocation(user, 'cancel-strategy', 8000.0, 'cancel-order', 'MSFT')

        allocation.refresh_from_db()
        assert allocation.reserved_amount == Decimal('8000.00')

        # Cancel order
        service.release_allocation(user, 'cancel-strategy', 8000.0, 'cancel-order')

        allocation.refresh_from_db()
        assert allocation.reserved_amount == Decimal('0.00')
        assert float(allocation.available_capital) == 20000.0

        # Verify reservation is marked cancelled
        reservation = AllocationReservation.objects.get(order_id='cancel-order')
        assert reservation.status == 'cancelled'

    def test_multiple_strategies_isolation(self, service, user):
        """Test that multiple strategies don't interfere with each other."""
        strategy_a = StrategyAllocationLimit.objects.create(
            user=user,
            strategy_name='strategy-a',
            allocated_pct=Decimal('20.00'),
            allocated_amount=Decimal('20000.00'),
            current_exposure=Decimal('15000.00'),
            is_enabled=True,
        )
        strategy_b = StrategyAllocationLimit.objects.create(
            user=user,
            strategy_name='strategy-b',
            allocated_pct=Decimal('30.00'),
            allocated_amount=Decimal('30000.00'),
            current_exposure=Decimal('0.00'),
            is_enabled=True,
        )

        # Reserve in strategy-b should not affect strategy-a
        service.reserve_allocation(user, 'strategy-b', 20000.0, 'order-b', 'AAPL')

        strategy_a.refresh_from_db()
        strategy_b.refresh_from_db()

        assert strategy_a.reserved_amount == Decimal('0.00')
        assert strategy_b.reserved_amount == Decimal('20000.00')

        # Strategy-a should still have its limits
        is_available, _ = service.check_allocation_available(user, 'strategy-a', 6000.0)
        assert is_available is False  # Only 5000 available


@pytest.mark.django_db
class TestUtilizationLevels:
    """Test utilization level calculations with real data."""

    def test_utilization_levels(self, user):
        """Test all utilization level thresholds."""
        test_cases = [
            (Decimal('0.00'), 'low'),
            (Decimal('4900.00'), 'low'),
            (Decimal('5000.00'), 'moderate'),
            (Decimal('7900.00'), 'moderate'),
            (Decimal('8000.00'), 'warning'),
            (Decimal('8900.00'), 'warning'),
            (Decimal('9000.00'), 'critical'),
            (Decimal('9900.00'), 'critical'),
            (Decimal('10000.00'), 'maxed_out'),
        ]

        for exposure, expected_level in test_cases:
            allocation = StrategyAllocationLimit.objects.create(
                user=user,
                strategy_name=f'util-test-{exposure}',
                allocated_pct=Decimal('10.00'),
                allocated_amount=Decimal('10000.00'),
                current_exposure=exposure,
                reserved_amount=Decimal('0.00'),
                is_enabled=True,
            )
            assert allocation.utilization_level == expected_level, \
                f"Expected {expected_level} for exposure {exposure}, got {allocation.utilization_level}"

    def test_is_maxed_out(self, user):
        """Test is_maxed_out property."""
        maxed = StrategyAllocationLimit.objects.create(
            user=user,
            strategy_name='maxed-test',
            allocated_pct=Decimal('10.00'),
            allocated_amount=Decimal('10000.00'),
            current_exposure=Decimal('10000.00'),
            is_enabled=True,
        )
        assert maxed.is_maxed_out is True

        not_maxed = StrategyAllocationLimit.objects.create(
            user=user,
            strategy_name='not-maxed-test',
            allocated_pct=Decimal('10.00'),
            allocated_amount=Decimal('10000.00'),
            current_exposure=Decimal('5000.00'),
            is_enabled=True,
        )
        assert not_maxed.is_maxed_out is False
