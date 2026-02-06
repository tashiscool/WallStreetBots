"""
Integration tests for CopyTradingService.

Tests all public methods with real database operations.
Covers provider management, subscription lifecycle, risk validation,
signal processing, proportional sizing, delay rejection, and replication.
Target: 80%+ coverage.
"""
import pytest
from datetime import timedelta
from decimal import Decimal
from unittest.mock import MagicMock, patch

from django.contrib.auth.models import User
from django.utils import timezone

from backend.auth0login.services.copy_trading_service import CopyTradingService
from backend.tradingbot.models.models import (
    ReplicatedTrade,
    SignalProvider,
    SignalSubscription,
    UserProfile,
)


@pytest.mark.django_db
class TestCopyTradingServiceProviders:
    """Tests for signal provider management."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test data."""
        self.user = User.objects.create_user(
            username="provider_owner",
            email="owner@example.com",
            password="testpass123",
        )
        # Ensure profile exists (may be auto-created by signal)
        if not hasattr(self.user, 'profile'):
            UserProfile.objects.create(user=self.user, risk_tolerance=3)
        else:
            self.user.profile.risk_tolerance = 3
            self.user.profile.investable_capital = Decimal('100000')
            self.user.profile.save()

        self.service = CopyTradingService()

    def test_create_provider(self):
        """Provider creation persists all fields correctly."""
        provider = self.service.create_provider(
            user=self.user,
            strategy_name="test_strategy",
            display_name="Test Strategy Provider",
            description="A test provider",
            fee_type="flat",
            fee_amount=Decimal('9.99'),
            min_risk_tolerance=3,
            is_public=True,
            max_subscribers=50,
        )

        assert provider.id is not None
        assert provider.owner == self.user
        assert provider.strategy_name == "test_strategy"
        assert provider.display_name == "Test Strategy Provider"
        assert provider.description == "A test provider"
        assert provider.fee_type == "flat"
        assert provider.fee_amount == Decimal('9.99')
        assert provider.min_risk_tolerance == 3
        assert provider.is_public is True
        assert provider.max_subscribers == 50
        assert provider.status == "active"
        assert provider.subscribers_count == 0
        assert provider.total_signals_sent == 0

    def test_create_provider_defaults(self):
        """Provider creation uses sensible defaults."""
        provider = self.service.create_provider(
            user=self.user,
            strategy_name="default_strategy",
            display_name="Default Provider",
        )

        assert provider.fee_type == "free"
        assert provider.fee_amount == Decimal('0')
        assert provider.min_risk_tolerance == 1
        assert provider.is_public is True
        assert provider.max_subscribers == 100

    def test_create_provider_unique_constraint(self):
        """Same owner cannot create two providers with same strategy name."""
        self.service.create_provider(
            user=self.user,
            strategy_name="unique_strategy",
            display_name="Provider 1",
        )

        with pytest.raises(Exception):
            self.service.create_provider(
                user=self.user,
                strategy_name="unique_strategy",
                display_name="Provider 2",
            )

    def test_get_providers_active_public(self):
        """get_providers returns only active, public providers by default."""
        self.service.create_provider(
            user=self.user,
            strategy_name="active_public",
            display_name="Active Public",
            is_public=True,
        )
        private = self.service.create_provider(
            user=self.user,
            strategy_name="active_private",
            display_name="Active Private",
            is_public=False,
        )
        paused = self.service.create_provider(
            user=self.user,
            strategy_name="paused_public",
            display_name="Paused Public",
            is_public=True,
        )
        paused.status = "paused"
        paused.save()

        providers = self.service.get_providers()
        names = [p.strategy_name for p in providers]
        assert "active_public" in names
        assert "active_private" not in names
        assert "paused_public" not in names

    def test_get_providers_include_private(self):
        """get_providers with is_public=None returns all active providers."""
        self.service.create_provider(
            user=self.user,
            strategy_name="pub",
            display_name="Public",
            is_public=True,
        )
        self.service.create_provider(
            user=self.user,
            strategy_name="priv",
            display_name="Private",
            is_public=False,
        )

        providers = self.service.get_providers(is_public=None)
        names = [p.strategy_name for p in providers]
        assert "pub" in names
        assert "priv" in names

    def test_get_provider_by_id(self):
        """get_provider returns the correct provider."""
        provider = self.service.create_provider(
            user=self.user,
            strategy_name="findme",
            display_name="Find Me",
        )

        found = self.service.get_provider(provider.id)
        assert found.id == provider.id
        assert found.display_name == "Find Me"

    def test_get_provider_not_found(self):
        """get_provider raises DoesNotExist for missing ID."""
        with pytest.raises(SignalProvider.DoesNotExist):
            self.service.get_provider(99999)


@pytest.mark.django_db
class TestCopyTradingServiceSubscriptions:
    """Tests for subscription management."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test data with provider and subscriber."""
        self.owner = User.objects.create_user(
            username="signal_owner",
            email="owner@test.com",
            password="testpass123",
        )
        self.subscriber = User.objects.create_user(
            username="signal_subscriber",
            email="sub@test.com",
            password="testpass123",
        )
        # Ensure profiles exist
        for user in [self.owner, self.subscriber]:
            try:
                profile = user.profile
            except UserProfile.DoesNotExist:
                profile = UserProfile.objects.create(user=user)
            profile.risk_tolerance = 3
            profile.investable_capital = Decimal('100000')
            profile.save()

        self.service = CopyTradingService()
        self.provider = self.service.create_provider(
            user=self.owner,
            strategy_name="sub_test_strategy",
            display_name="Subscription Test",
            min_risk_tolerance=2,
        )

    def test_subscribe_success(self):
        """Successful subscription creates record and increments count."""
        sub = self.service.subscribe(
            user=self.subscriber,
            provider_id=self.provider.id,
            auto_replicate=True,
            max_allocation_pct=Decimal('10.00'),
        )

        assert sub.id is not None
        assert sub.subscriber == self.subscriber
        assert sub.provider == self.provider
        assert sub.status == "active"
        assert sub.auto_replicate is True
        assert sub.max_allocation_pct == Decimal('10.00')

        # Check subscriber count incremented
        self.provider.refresh_from_db()
        assert self.provider.subscribers_count == 1

    def test_subscribe_defaults(self):
        """Subscription uses correct defaults."""
        sub = self.service.subscribe(
            user=self.subscriber,
            provider_id=self.provider.id,
        )

        assert sub.auto_replicate is False
        assert sub.max_allocation_pct == Decimal('5.00')
        assert sub.proportional_sizing is True
        assert sub.max_replication_delay_seconds == 300
        assert sub.notify_on_signal is True

    def test_subscribe_self_prevention(self):
        """Cannot subscribe to own signal provider."""
        with pytest.raises(ValueError, match="Cannot subscribe to your own"):
            self.service.subscribe(
                user=self.owner,
                provider_id=self.provider.id,
            )

    def test_subscribe_risk_tolerance_check(self):
        """Conservative user cannot follow aggressive provider."""
        # Set subscriber risk tolerance to 1 (very conservative)
        profile = self.subscriber.profile
        profile.risk_tolerance = 1
        profile.save()

        # Provider requires minimum risk tolerance of 2
        with pytest.raises(ValueError, match="Insufficient risk tolerance"):
            self.service.subscribe(
                user=self.subscriber,
                provider_id=self.provider.id,
            )

    def test_subscribe_risk_tolerance_equal_passes(self):
        """User with exact minimum risk tolerance can subscribe."""
        profile = self.subscriber.profile
        profile.risk_tolerance = 2  # Matches provider.min_risk_tolerance
        profile.save()

        sub = self.service.subscribe(
            user=self.subscriber,
            provider_id=self.provider.id,
        )
        assert sub.id is not None

    def test_subscribe_max_subscribers_check(self):
        """Cannot exceed max subscriber capacity."""
        self.provider.max_subscribers = 1
        self.provider.subscribers_count = 1
        self.provider.save()

        with pytest.raises(ValueError, match="maximum subscriber capacity"):
            self.service.subscribe(
                user=self.subscriber,
                provider_id=self.provider.id,
            )

    def test_subscribe_inactive_provider(self):
        """Cannot subscribe to a paused/closed provider."""
        self.provider.status = "paused"
        self.provider.save()

        with pytest.raises(ValueError, match="provider status is 'paused'"):
            self.service.subscribe(
                user=self.subscriber,
                provider_id=self.provider.id,
            )

    def test_subscribe_nonexistent_provider(self):
        """Subscribing to nonexistent provider raises ValueError."""
        with pytest.raises(ValueError, match="not found"):
            self.service.subscribe(
                user=self.subscriber,
                provider_id=99999,
            )

    def test_subscribe_duplicate_prevention(self):
        """Cannot create duplicate active subscription."""
        self.service.subscribe(
            user=self.subscriber,
            provider_id=self.provider.id,
        )

        with pytest.raises(ValueError, match="Already subscribed"):
            self.service.subscribe(
                user=self.subscriber,
                provider_id=self.provider.id,
            )

    def test_unsubscribe_success(self):
        """Unsubscribe sets status to cancelled and decrements count."""
        self.service.subscribe(
            user=self.subscriber,
            provider_id=self.provider.id,
        )

        self.provider.refresh_from_db()
        assert self.provider.subscribers_count == 1

        result = self.service.unsubscribe(
            user=self.subscriber,
            provider_id=self.provider.id,
        )

        assert result.status == "cancelled"

        self.provider.refresh_from_db()
        assert self.provider.subscribers_count == 0

    def test_unsubscribe_no_active_subscription(self):
        """Unsubscribing without active subscription raises ValueError."""
        with pytest.raises(ValueError, match="No active subscription"):
            self.service.unsubscribe(
                user=self.subscriber,
                provider_id=self.provider.id,
            )

    def test_get_subscriptions(self):
        """get_subscriptions returns only active/paused subscriptions."""
        sub = self.service.subscribe(
            user=self.subscriber,
            provider_id=self.provider.id,
        )

        subs = self.service.get_subscriptions(self.subscriber)
        assert subs.count() == 1
        assert subs.first().id == sub.id

    def test_get_subscriptions_excludes_cancelled(self):
        """Cancelled subscriptions are excluded from get_subscriptions."""
        self.service.subscribe(
            user=self.subscriber,
            provider_id=self.provider.id,
        )
        self.service.unsubscribe(
            user=self.subscriber,
            provider_id=self.provider.id,
        )

        subs = self.service.get_subscriptions(self.subscriber)
        assert subs.count() == 0

    def test_resubscribe_after_cancel(self):
        """Can re-subscribe after cancelling previous subscription."""
        self.service.subscribe(
            user=self.subscriber,
            provider_id=self.provider.id,
        )
        self.service.unsubscribe(
            user=self.subscriber,
            provider_id=self.provider.id,
        )

        # Should be able to subscribe again
        new_sub = self.service.subscribe(
            user=self.subscriber,
            provider_id=self.provider.id,
        )
        assert new_sub.status == "active"


@pytest.mark.django_db
class TestCopyTradingServiceSignalProcessing:
    """Tests for signal processing and trade replication."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up provider, subscriber, and active subscription."""
        self.owner = User.objects.create_user(
            username="sig_owner",
            email="sigowner@test.com",
            password="testpass123",
        )
        self.subscriber = User.objects.create_user(
            username="sig_subscriber",
            email="sigsub@test.com",
            password="testpass123",
        )
        # Ensure profiles
        for user in [self.owner, self.subscriber]:
            try:
                profile = user.profile
            except UserProfile.DoesNotExist:
                profile = UserProfile.objects.create(user=user)
            profile.risk_tolerance = 4
            profile.investable_capital = Decimal('100000')
            profile.save()

        self.service = CopyTradingService()
        self.provider = self.service.create_provider(
            user=self.owner,
            strategy_name="sig_strategy",
            display_name="Signal Strategy",
        )
        self.subscription = self.service.subscribe(
            user=self.subscriber,
            provider_id=self.provider.id,
            auto_replicate=True,
            max_allocation_pct=Decimal('5.00'),
            proportional_sizing=True,
        )

        self.trade_data = {
            'trade_id': 'test_trade_001',
            'symbol': 'AAPL',
            'side': 'buy',
            'qty': 100,
            'price': 150.00,
            'timestamp': timezone.now().isoformat(),
        }

    @patch.object(CopyTradingService, 'broadcaster', new_callable=lambda: MagicMock)
    @patch.object(CopyTradingService, 'execution_client')
    def test_process_signal_basic(self, mock_exec, mock_broadcaster):
        """Signal processing increments total_signals_sent and returns results."""
        mock_client = MagicMock()
        mock_client.place_order.return_value = MagicMock(
            accepted=True,
            broker_order_id='broker_123',
        )
        # Patch the property to return our mock client
        type(self.service).execution_client = property(lambda self: mock_client)
        type(self.service).broadcaster = property(lambda self: mock_broadcaster)

        result = self.service.process_signal(self.provider.id, self.trade_data)

        assert result['provider_id'] == self.provider.id
        assert result['subscribers_notified'] == 1

        self.provider.refresh_from_db()
        assert self.provider.total_signals_sent == 1

    @patch.object(CopyTradingService, 'broadcaster', new_callable=lambda: MagicMock)
    def test_process_signal_notification_only(self, mock_broadcaster):
        """Signal with auto_replicate=False only notifies, does not replicate."""
        # Disable auto-replicate
        self.subscription.auto_replicate = False
        self.subscription.save()

        type(self.service).broadcaster = property(lambda self: mock_broadcaster)

        result = self.service.process_signal(self.provider.id, self.trade_data)

        assert result['replications'][0]['status'] == 'notified_only'
        assert result['replications'][0]['auto_replicate'] is False
        assert ReplicatedTrade.objects.count() == 0

    def test_process_signal_nonexistent_provider(self):
        """Processing signal for nonexistent provider returns error."""
        result = self.service.process_signal(99999, self.trade_data)
        assert 'error' in result

    def test_proportional_sizing_calculation(self):
        """Proportional sizing uses subscriber capital and allocation percentage."""
        # subscriber capital = 100000, max_allocation_pct = 5%
        # allocation amount = 100000 * 5/100 = 5000
        # price = 150, so replicated qty = 5000 / 150 = 33.3333
        mock_client = MagicMock()
        mock_client.place_order.return_value = MagicMock(
            accepted=True,
            broker_order_id='broker_456',
        )
        self.service._execution_client = mock_client
        self.service._broadcaster = MagicMock()

        result = self.service._replicate_for_subscriber(
            self.subscription, self.trade_data
        )

        assert result['status'] == 'executed'

        # Check the order was placed with proportional quantity
        call_args = mock_client.place_order.call_args
        order_req = call_args[0][0]
        # 100000 * 0.05 / 150 = 33.3333
        assert abs(order_req.qty - 33.3333) < 0.01

    def test_delay_rejection(self):
        """Trade older than max_replication_delay_seconds is rejected."""
        self.subscription.max_replication_delay_seconds = 60
        self.subscription.save()

        # Set trade timestamp to 2 minutes ago
        old_trade_data = dict(self.trade_data)
        old_trade_data['timestamp'] = (
            timezone.now() - timedelta(seconds=120)
        ).isoformat()

        self.service._broadcaster = MagicMock()

        result = self.service._replicate_for_subscriber(
            self.subscription, old_trade_data
        )

        assert result['status'] == 'rejected_delay'

        # Verify ReplicatedTrade record was created with rejected status
        trade = ReplicatedTrade.objects.get(id=result['trade_id'])
        assert trade.status == 'rejected_delay'
        assert 'exceeds maximum' in trade.failure_reason

    def test_execution_failure_handling(self):
        """Failed order execution is recorded correctly."""
        mock_client = MagicMock()
        mock_client.place_order.return_value = MagicMock(
            accepted=False,
            broker_order_id=None,
            reason='Insufficient buying power',
        )
        self.service._execution_client = mock_client
        self.service._broadcaster = MagicMock()

        result = self.service._replicate_for_subscriber(
            self.subscription, self.trade_data
        )

        assert result['status'] == 'failed'
        trade = ReplicatedTrade.objects.get(id=result['trade_id'])
        assert trade.status == 'failed'
        assert 'Insufficient buying power' in trade.failure_reason

    def test_execution_exception_handling(self):
        """Exceptions during execution are caught and recorded."""
        mock_client = MagicMock()
        mock_client.place_order.side_effect = Exception("Network error")
        self.service._execution_client = mock_client
        self.service._broadcaster = MagicMock()

        result = self.service._replicate_for_subscriber(
            self.subscription, self.trade_data
        )

        assert result['status'] == 'failed'
        trade = ReplicatedTrade.objects.get(id=result['trade_id'])
        assert trade.status == 'failed'
        assert 'Network error' in trade.failure_reason

    def test_no_execution_client(self):
        """Missing execution client causes failure status."""
        self.service._execution_client = None
        # Force the property to return None
        self.service._broadcaster = MagicMock()

        # We need to make the execution_client property return None
        original_prop = type(self.service).execution_client
        type(self.service).execution_client = property(lambda self: None)

        try:
            result = self.service._replicate_for_subscriber(
                self.subscription, self.trade_data
            )
            assert result['status'] == 'failed'
            assert result['reason'] == 'no_execution_client'
        finally:
            type(self.service).execution_client = original_prop

    def test_zero_quantity_rejection(self):
        """Trade with zero calculated quantity is rejected."""
        # Set very low capital to produce zero quantity
        profile = self.subscriber.profile
        profile.investable_capital = Decimal('1')
        profile.save()

        # capital=1, allocation=5% => amount=0.05
        # price=99999999 (fits max_digits=12, decimal_places=4)
        # qty = 0.05 / 99999999 = ~0.0000 (rounds to zero at 4dp)
        high_price_trade = dict(self.trade_data)
        high_price_trade['price'] = 99999999

        self.service._broadcaster = MagicMock()
        mock_client = MagicMock()
        mock_client.place_order.return_value = MagicMock(
            accepted=True,
            broker_order_id='broker_789',
        )
        self.service._execution_client = mock_client

        result = self.service._replicate_for_subscriber(
            self.subscription, high_price_trade
        )

        assert result['status'] == 'rejected_risk'
        assert result['reason'] == 'zero_quantity'

    def test_subscription_stats_updated_on_success(self):
        """Successful replication increments trades_replicated on subscription."""
        mock_client = MagicMock()
        mock_client.place_order.return_value = MagicMock(
            accepted=True,
            broker_order_id='broker_stats',
        )
        self.service._execution_client = mock_client
        self.service._broadcaster = MagicMock()

        initial_count = self.subscription.trades_replicated

        self.service._replicate_for_subscriber(self.subscription, self.trade_data)

        self.subscription.refresh_from_db()
        assert self.subscription.trades_replicated == initial_count + 1


@pytest.mark.django_db
class TestCopyTradingServiceStrategyHook:
    """Tests for the on_strategy_trade class method hook."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up provider linked to a strategy."""
        self.owner = User.objects.create_user(
            username="hook_owner",
            email="hook@test.com",
            password="testpass123",
        )
        try:
            profile = self.owner.profile
        except UserProfile.DoesNotExist:
            profile = UserProfile.objects.create(user=self.owner)
        profile.risk_tolerance = 3
        profile.save()

        self.provider = SignalProvider.objects.create(
            owner=self.owner,
            strategy_name="hooked_strategy",
            display_name="Hooked Strategy",
            status="active",
        )

    @patch.object(CopyTradingService, 'process_signal')
    def test_on_strategy_trade_calls_process_signal(self, mock_process):
        """on_strategy_trade dispatches to process_signal for matching providers."""
        trade_data = {
            'trade_id': 'hook_001',
            'symbol': 'TSLA',
            'side': 'buy',
            'qty': 50,
            'price': 200.00,
            'timestamp': timezone.now().isoformat(),
        }

        CopyTradingService.on_strategy_trade("hooked_strategy", trade_data)

        mock_process.assert_called_once_with(self.provider.id, trade_data)

    @patch.object(CopyTradingService, 'process_signal')
    def test_on_strategy_trade_no_matching_provider(self, mock_process):
        """on_strategy_trade does nothing when no provider matches strategy."""
        CopyTradingService.on_strategy_trade("nonexistent_strategy", {})
        mock_process.assert_not_called()


@pytest.mark.django_db
class TestCopyTradingServiceProviderStats:
    """Tests for provider statistics."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up provider with some replicated trade data."""
        self.owner = User.objects.create_user(
            username="stats_owner",
            email="stats@test.com",
            password="testpass123",
        )
        self.subscriber = User.objects.create_user(
            username="stats_sub",
            email="statssub@test.com",
            password="testpass123",
        )
        for user in [self.owner, self.subscriber]:
            try:
                profile = user.profile
            except UserProfile.DoesNotExist:
                profile = UserProfile.objects.create(user=user)
            profile.risk_tolerance = 4
            profile.investable_capital = Decimal('100000')
            profile.save()

        self.service = CopyTradingService()
        self.provider = self.service.create_provider(
            user=self.owner,
            strategy_name="stats_strategy",
            display_name="Stats Provider",
        )
        self.provider.total_signals_sent = 10
        self.provider.win_rate = Decimal('65.00')
        self.provider.total_return_pct = Decimal('12.50')
        self.provider.save()

        self.subscription = self.service.subscribe(
            user=self.subscriber,
            provider_id=self.provider.id,
            auto_replicate=True,
        )

        # Create some replicated trades
        now = timezone.now()
        ReplicatedTrade.objects.create(
            subscription=self.subscription,
            original_trade_id='t1',
            original_symbol='AAPL',
            original_side='buy',
            original_qty=Decimal('100'),
            original_price=Decimal('150.00'),
            original_timestamp=now,
            status='executed',
            slippage_pct=Decimal('0.05'),
        )
        ReplicatedTrade.objects.create(
            subscription=self.subscription,
            original_trade_id='t2',
            original_symbol='GOOG',
            original_side='sell',
            original_qty=Decimal('50'),
            original_price=Decimal('2800.00'),
            original_timestamp=now,
            status='failed',
            failure_reason='Insufficient funds',
        )
        ReplicatedTrade.objects.create(
            subscription=self.subscription,
            original_trade_id='t3',
            original_symbol='MSFT',
            original_side='buy',
            original_qty=Decimal('200'),
            original_price=Decimal('300.00'),
            original_timestamp=now,
            status='rejected_delay',
            failure_reason='Delay exceeded',
        )

    def test_get_provider_stats(self):
        """Provider stats include all expected fields and counts."""
        stats = self.service.get_provider_stats(self.provider.id)

        assert stats['provider_id'] == self.provider.id
        assert stats['display_name'] == "Stats Provider"
        assert stats['subscribers']['active'] == 1
        assert stats['signals']['total_sent'] == 10
        assert stats['performance']['win_rate'] == 65.00
        assert stats['performance']['total_return_pct'] == 12.50
        assert stats['replication']['total_replicated'] == 3
        assert stats['replication']['executed'] == 1
        assert stats['replication']['failed'] == 1
        assert stats['replication']['rejected_delay'] == 1

    def test_get_provider_stats_not_found(self):
        """Stats for nonexistent provider returns error."""
        stats = self.service.get_provider_stats(99999)
        assert 'error' in stats

    def test_get_provider_stats_execution_rate(self):
        """Execution rate is calculated correctly."""
        stats = self.service.get_provider_stats(self.provider.id)
        # 1 executed out of 3 total = 33.3%
        assert abs(stats['replication']['execution_rate'] - 33.3) < 0.1


@pytest.mark.django_db
class TestCopyTradingServiceManualReplication:
    """Tests for manual trade replication."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up provider and subscriber for manual replication."""
        self.owner = User.objects.create_user(
            username="manual_owner",
            email="manual@test.com",
            password="testpass123",
        )
        self.subscriber = User.objects.create_user(
            username="manual_sub",
            email="manualsub@test.com",
            password="testpass123",
        )
        for user in [self.owner, self.subscriber]:
            try:
                profile = user.profile
            except UserProfile.DoesNotExist:
                profile = UserProfile.objects.create(user=user)
            profile.risk_tolerance = 4
            profile.investable_capital = Decimal('100000')
            profile.save()

        self.service = CopyTradingService()
        self.provider = self.service.create_provider(
            user=self.owner,
            strategy_name="manual_strategy",
            display_name="Manual Strategy",
        )
        self.service.subscribe(
            user=self.subscriber,
            provider_id=self.provider.id,
            auto_replicate=False,  # Manual mode
        )

    def test_manual_replicate_success(self):
        """Manual replication works for active subscriber."""
        mock_client = MagicMock()
        mock_client.place_order.return_value = MagicMock(
            accepted=True,
            broker_order_id='manual_broker_1',
        )
        self.service._execution_client = mock_client
        self.service._broadcaster = MagicMock()

        trade_data = {
            'trade_id': 'manual_001',
            'symbol': 'NVDA',
            'side': 'buy',
            'qty': 50,
            'price': 500.00,
            'timestamp': timezone.now().isoformat(),
        }

        result = self.service.manual_replicate(
            user=self.subscriber,
            provider_id=self.provider.id,
            trade_data=trade_data,
        )

        assert result['status'] == 'executed'

    def test_manual_replicate_no_subscription(self):
        """Manual replication fails without active subscription."""
        other_user = User.objects.create_user(
            username="no_sub_user",
            email="nosub@test.com",
            password="testpass123",
        )

        with pytest.raises(ValueError, match="No active subscription"):
            self.service.manual_replicate(
                user=other_user,
                provider_id=self.provider.id,
                trade_data={'symbol': 'AAPL', 'side': 'buy'},
            )
