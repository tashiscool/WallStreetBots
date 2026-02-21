"""
Copy/Social Trading Service.

Provides signal provider management, subscription handling,
signal broadcasting, and automatic trade replication for copy trading.
"""
import logging
import uuid
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional

from django.contrib.auth.models import User
from django.db import transaction
from django.db.models import QuerySet
from django.utils import timezone

logger = logging.getLogger(__name__)


class CopyTradingService:
    """
    Service for managing copy/social trading operations.

    Handles signal providers, subscriptions, signal broadcasting,
    and automatic trade replication with proportional sizing and risk checks.
    """

    def __init__(self):
        """Initialize the copy trading service."""
        self._broadcaster = None
        self._execution_client = None

    @property
    def broadcaster(self):
        """Lazy-load the sync WebSocket broadcaster."""
        if self._broadcaster is None:
            try:
                from backend.tradingbot.rpc.websocket import SyncWebSocketBroadcaster
                self._broadcaster = SyncWebSocketBroadcaster()
            except Exception as e:
                logger.warning(f"Could not initialize WebSocket broadcaster: {e}")
        return self._broadcaster

    @property
    def execution_client(self):
        """Lazy-load the execution client."""
        if self._execution_client is None:
            try:
                from backend.tradingbot.execution.interfaces import InMemoryExecutionClient
                self._execution_client = InMemoryExecutionClient()
            except Exception as e:
                logger.warning(f"Could not initialize execution client: {e}")
        return self._execution_client

    # ========================================================================
    # Provider Management
    # ========================================================================

    def get_providers(
        self,
        is_public: bool = True,
        status: str = 'active',
    ) -> QuerySet:
        """
        Get available signal providers.

        Args:
            is_public: Filter by public visibility.
            status: Filter by provider status.

        Returns:
            QuerySet of SignalProvider objects.
        """
        from backend.tradingbot.models.models import SignalProvider

        filters = {'status': status}
        if is_public is not None:
            filters['is_public'] = is_public

        return SignalProvider.objects.filter(**filters).select_related('owner')

    def get_provider(self, provider_id: int):
        """
        Get a single signal provider by ID.

        Args:
            provider_id: The provider's primary key.

        Returns:
            SignalProvider instance.

        Raises:
            SignalProvider.DoesNotExist: If provider not found.
        """
        from backend.tradingbot.models.models import SignalProvider

        return SignalProvider.objects.select_related('owner').get(id=provider_id)

    def create_provider(
        self,
        user: User,
        strategy_name: str,
        display_name: str,
        description: str = '',
        fee_type: str = 'free',
        fee_amount: Decimal = Decimal('0'),
        min_risk_tolerance: int = 1,
        is_public: bool = True,
        max_subscribers: int = 100,
    ):
        """
        Create a new signal provider.

        Args:
            user: The owner User instance.
            strategy_name: Internal strategy identifier.
            display_name: Human-readable provider name.
            description: Provider description.
            fee_type: Fee structure type ('free', 'flat', 'performance').
            fee_amount: Fee amount (if applicable).
            min_risk_tolerance: Minimum risk tolerance required (1-5).
            is_public: Whether the provider is publicly visible.
            max_subscribers: Maximum allowed subscribers.

        Returns:
            Created SignalProvider instance.
        """
        from backend.tradingbot.models.models import SignalProvider

        provider = SignalProvider.objects.create(
            owner=user,
            strategy_name=strategy_name,
            display_name=display_name,
            description=description,
            fee_type=fee_type,
            fee_amount=fee_amount,
            min_risk_tolerance=min_risk_tolerance,
            is_public=is_public,
            max_subscribers=max_subscribers,
        )

        logger.info(
            f"Signal provider created: {display_name} by {user.username} "
            f"(strategy={strategy_name})"
        )
        return provider

    # ========================================================================
    # Subscription Management
    # ========================================================================

    def subscribe(
        self,
        user: User,
        provider_id: int,
        auto_replicate: bool = False,
        max_allocation_pct: Decimal = Decimal('5.00'),
        proportional_sizing: bool = True,
    ):
        """
        Subscribe a user to a signal provider.

        Args:
            user: The subscribing user.
            provider_id: The provider to subscribe to.
            auto_replicate: Whether to automatically replicate trades.
            max_allocation_pct: Maximum allocation percentage per trade.
            proportional_sizing: Use proportional position sizing.

        Returns:
            Created SignalSubscription instance.

        Raises:
            ValueError: If validation fails (risk tolerance, capacity, self-subscribe).
        """
        from backend.tradingbot.models.models import SignalProvider, SignalSubscription

        # Get provider
        try:
            provider = SignalProvider.objects.get(id=provider_id)
        except SignalProvider.DoesNotExist as e:
            raise ValueError(f"Signal provider {provider_id} not found") from e

        # Validate: provider must be active
        if provider.status != 'active':
            raise ValueError(
                f"Cannot subscribe to provider '{provider.display_name}': "
                f"provider status is '{provider.status}'"
            )

        # Validate: cannot subscribe to own provider
        if provider.owner_id == user.id:
            raise ValueError("Cannot subscribe to your own signal provider")

        # Validate: risk tolerance check
        user_risk_tolerance = self._get_user_risk_tolerance(user)
        if user_risk_tolerance is not None and user_risk_tolerance < provider.min_risk_tolerance:
            raise ValueError(
                f"Insufficient risk tolerance: your level is {user_risk_tolerance}, "
                f"provider requires minimum {provider.min_risk_tolerance}"
            )

        # Validate: max subscribers not exceeded
        if provider.subscribers_count >= provider.max_subscribers:
            raise ValueError(
                f"Provider '{provider.display_name}' has reached maximum "
                f"subscriber capacity ({provider.max_subscribers})"
            )

        # Check for existing subscription
        existing = SignalSubscription.objects.filter(
            subscriber=user,
            provider=provider,
        ).first()

        if existing and existing.status in ('active', 'paused'):
            raise ValueError(
                f"Already subscribed to provider '{provider.display_name}' "
                f"(status: {existing.status})"
            )

        # Create or re-activate subscription and update subscriber count atomically
        with transaction.atomic():
            if existing and existing.status == 'cancelled':
                # Re-activate cancelled subscription
                existing.status = 'active'
                existing.auto_replicate = auto_replicate
                existing.max_allocation_pct = max_allocation_pct
                existing.proportional_sizing = proportional_sizing
                existing.save(update_fields=[
                    'status', 'auto_replicate', 'max_allocation_pct',
                    'proportional_sizing', 'updated_at',
                ])
                subscription = existing
            else:
                subscription = SignalSubscription.objects.create(
                    subscriber=user,
                    provider=provider,
                    auto_replicate=auto_replicate,
                    max_allocation_pct=max_allocation_pct,
                    proportional_sizing=proportional_sizing,
                )
            provider.subscribers_count += 1
            provider.save(update_fields=['subscribers_count'])

        logger.info(
            f"User {user.username} subscribed to provider "
            f"'{provider.display_name}' (auto_replicate={auto_replicate})"
        )
        return subscription

    def unsubscribe(self, user: User, provider_id: int):
        """
        Unsubscribe a user from a signal provider.

        Args:
            user: The user to unsubscribe.
            provider_id: The provider to unsubscribe from.

        Returns:
            The updated SignalSubscription instance.

        Raises:
            ValueError: If no active subscription found.
        """
        from backend.tradingbot.models.models import SignalSubscription

        subscription = SignalSubscription.objects.filter(
            subscriber=user,
            provider_id=provider_id,
            status__in=['active', 'paused'],
        ).first()

        if not subscription:
            raise ValueError(
                f"No active subscription found for provider {provider_id}"
            )

        with transaction.atomic():
            subscription.status = 'cancelled'
            subscription.save(update_fields=['status', 'updated_at'])

            provider = subscription.provider
            if provider.subscribers_count > 0:
                provider.subscribers_count -= 1
                provider.save(update_fields=['subscribers_count'])

        logger.info(
            f"User {user.username} unsubscribed from provider "
            f"'{provider.display_name}'"
        )
        return subscription

    def get_subscriptions(self, user: User) -> QuerySet:
        """
        Get a user's active subscriptions.

        Args:
            user: The user whose subscriptions to retrieve.

        Returns:
            QuerySet of active SignalSubscription objects.
        """
        from backend.tradingbot.models.models import SignalSubscription

        return SignalSubscription.objects.filter(
            subscriber=user,
            status__in=['active', 'paused'],
        ).select_related('provider', 'provider__owner')

    # ========================================================================
    # Signal Processing & Replication
    # ========================================================================

    def process_signal(self, provider_id: int, trade_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a trading signal from a provider and distribute to subscribers.

        Args:
            provider_id: The signal provider's ID.
            trade_data: Dictionary containing trade details:
                - trade_id: Unique trade identifier
                - symbol: Trading symbol
                - side: 'buy' or 'sell'
                - qty: Quantity
                - price: Execution price
                - timestamp: ISO format timestamp of the original trade

        Returns:
            Summary of signal processing results.
        """
        from backend.tradingbot.models.models import SignalProvider, SignalSubscription

        try:
            provider = SignalProvider.objects.get(id=provider_id)
        except SignalProvider.DoesNotExist:
            logger.error(f"Signal provider {provider_id} not found")
            return {'error': f'Provider {provider_id} not found', 'replications': []}

        # Increment total signals sent
        provider.total_signals_sent += 1
        provider.save(update_fields=['total_signals_sent'])

        # Get all active subscriptions
        subscriptions = SignalSubscription.objects.filter(
            provider=provider,
            status='active',
        ).select_related('subscriber')

        results = []
        for subscription in subscriptions:
            # Send WebSocket notification if subscriber wants it
            if subscription.notify_on_signal and self.broadcaster:
                try:
                    self.broadcaster.broadcast_copy_signal(
                        provider_name=provider.display_name,
                        symbol=trade_data.get('symbol', ''),
                        side=trade_data.get('side', ''),
                        price=float(trade_data.get('price', 0)),
                        subscriber_id=subscription.subscriber_id,
                        trade_id=trade_data.get('trade_id', ''),
                    )
                except Exception as e:
                    logger.warning(f"Failed to send WebSocket notification: {e}")

            # Auto-replicate if enabled
            if subscription.auto_replicate:
                replication_result = self._replicate_for_subscriber(
                    subscription, trade_data
                )
                results.append(replication_result)
            else:
                results.append({
                    'subscriber_id': subscription.subscriber_id,
                    'status': 'notified_only',
                    'auto_replicate': False,
                })

        return {
            'provider_id': provider_id,
            'provider_name': provider.display_name,
            'signal_number': provider.total_signals_sent,
            'subscribers_notified': len(subscriptions),
            'replications': results,
        }

    def _replicate_for_subscriber(
        self,
        subscription,
        trade_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Replicate a trade for a specific subscriber.

        Performs delay check, proportional sizing, creates a ReplicatedTrade record,
        and optionally executes via ExecutionClient.

        Args:
            subscription: The SignalSubscription instance.
            trade_data: Dictionary with trade details.

        Returns:
            Dictionary with replication result details.
        """
        from backend.tradingbot.models.models import ReplicatedTrade
        from backend.tradingbot.execution.interfaces import OrderRequest

        subscriber = subscription.subscriber
        now = timezone.now()

        # Parse original timestamp
        original_timestamp = trade_data.get('timestamp')
        if isinstance(original_timestamp, str):
            original_timestamp = datetime.fromisoformat(
                original_timestamp.replace('Z', '+00:00')
            )
            if original_timestamp.tzinfo is None:
                original_timestamp = timezone.make_aware(original_timestamp)
        elif original_timestamp is None:
            original_timestamp = now

        # Check replication delay
        delay_seconds = (now - original_timestamp).total_seconds()
        delay_ms = int(delay_seconds * 1000)

        if delay_seconds > subscription.max_replication_delay_seconds:
            trade = ReplicatedTrade.objects.create(
                subscription=subscription,
                original_trade_id=trade_data.get('trade_id', str(uuid.uuid4())[:8]),
                original_symbol=trade_data.get('symbol', ''),
                original_side=trade_data.get('side', ''),
                original_qty=Decimal(str(trade_data.get('qty', 0))),
                original_price=Decimal(str(trade_data.get('price', 0))),
                original_timestamp=original_timestamp,
                replication_delay_ms=delay_ms,
                status='rejected_delay',
                failure_reason=(
                    f"Replication delay {delay_seconds:.1f}s exceeds maximum "
                    f"{subscription.max_replication_delay_seconds}s"
                ),
            )
            logger.info(
                f"Rejected replication for {subscriber.username}: "
                f"delay {delay_seconds:.1f}s > max {subscription.max_replication_delay_seconds}s"
            )
            return {
                'subscriber_id': subscriber.id,
                'status': 'rejected_delay',
                'trade_id': trade.id,
                'delay_seconds': delay_seconds,
            }

        # Calculate proportional sizing
        original_qty = Decimal(str(trade_data.get('qty', 0)))
        original_price = Decimal(str(trade_data.get('price', 0)))

        if subscription.proportional_sizing:
            subscriber_capital = self._get_subscriber_capital(subscriber)
            allocation_amount = subscriber_capital * (subscription.max_allocation_pct / Decimal('100'))
            if original_price > 0:
                replicated_qty = (allocation_amount / original_price).quantize(Decimal('0.0001'))
            else:
                replicated_qty = Decimal('0')
        else:
            replicated_qty = original_qty

        # Ensure replicated qty is not zero
        if replicated_qty <= 0:
            trade = ReplicatedTrade.objects.create(
                subscription=subscription,
                original_trade_id=trade_data.get('trade_id', str(uuid.uuid4())[:8]),
                original_symbol=trade_data.get('symbol', ''),
                original_side=trade_data.get('side', ''),
                original_qty=original_qty,
                original_price=original_price,
                original_timestamp=original_timestamp,
                replication_delay_ms=delay_ms,
                status='rejected_risk',
                failure_reason='Calculated replicated quantity is zero or negative',
            )
            return {
                'subscriber_id': subscriber.id,
                'status': 'rejected_risk',
                'trade_id': trade.id,
                'reason': 'zero_quantity',
            }

        # Create replicated trade record
        trade = ReplicatedTrade.objects.create(
            subscription=subscription,
            original_trade_id=trade_data.get('trade_id', str(uuid.uuid4())[:8]),
            original_symbol=trade_data.get('symbol', ''),
            original_side=trade_data.get('side', ''),
            original_qty=original_qty,
            original_price=original_price,
            original_timestamp=original_timestamp,
            replicated_qty=replicated_qty,
            replication_delay_ms=delay_ms,
            status='pending',
        )

        # Execute the order
        try:
            client = self.execution_client
            if client is None:
                trade.status = 'failed'
                trade.failure_reason = 'Execution client not available'
                trade.save(update_fields=['status', 'failure_reason'])
                return {
                    'subscriber_id': subscriber.id,
                    'status': 'failed',
                    'trade_id': trade.id,
                    'reason': 'no_execution_client',
                }

            order_req = OrderRequest(
                client_order_id=f"copy_{trade.id}_{uuid.uuid4().hex[:8]}",
                symbol=trade_data.get('symbol', ''),
                qty=float(replicated_qty),
                side=trade_data.get('side', 'buy'),
                type='market',
                time_in_force='day',
            )

            ack = client.place_order(order_req)

            if ack.accepted:
                trade.status = 'executed'
                trade.replicated_price = original_price  # Will be updated with actual fill
                trade.replicated_timestamp = timezone.now()

                # Calculate slippage (using original price as estimate until fill)
                if original_price > 0:
                    trade.slippage_pct = Decimal('0')  # Updated on fill reconciliation

                trade.save(update_fields=[
                    'status', 'replicated_price', 'replicated_timestamp', 'slippage_pct',
                ])

                # Update subscription stats
                subscription.trades_replicated += 1
                subscription.save(update_fields=['trades_replicated'])

                # Broadcast execution
                if self.broadcaster:
                    try:
                        self.broadcaster.broadcast_copy_trade_executed(
                            provider_name=subscription.provider.display_name,
                            symbol=trade_data.get('symbol', ''),
                            side=trade_data.get('side', ''),
                            qty=float(replicated_qty),
                            price=float(original_price),
                            subscriber_id=subscriber.id,
                        )
                    except Exception as e:
                        logger.warning(f"Failed to broadcast trade execution: {e}")

                logger.info(
                    f"Replicated trade for {subscriber.username}: "
                    f"{trade_data.get('side', '')} {replicated_qty} "
                    f"{trade_data.get('symbol', '')} @ {original_price}"
                )

                return {
                    'subscriber_id': subscriber.id,
                    'status': 'executed',
                    'trade_id': trade.id,
                    'replicated_qty': float(replicated_qty),
                    'broker_order_id': ack.broker_order_id,
                }
            else:
                trade.status = 'failed'
                trade.failure_reason = ack.reason or 'Order rejected by broker'
                trade.save(update_fields=['status', 'failure_reason'])
                return {
                    'subscriber_id': subscriber.id,
                    'status': 'failed',
                    'trade_id': trade.id,
                    'reason': ack.reason,
                }

        except Exception as e:
            logger.error(
                f"Error replicating trade for {subscriber.username}: {e}",
                exc_info=True,
            )
            trade.status = 'failed'
            trade.failure_reason = str(e)
            trade.save(update_fields=['status', 'failure_reason'])
            return {
                'subscriber_id': subscriber.id,
                'status': 'failed',
                'trade_id': trade.id,
                'reason': str(e),
            }

    # ========================================================================
    # Strategy Hook
    # ========================================================================

    @classmethod
    def on_strategy_trade(cls, strategy_name: str, trade_data: Dict[str, Any]) -> None:
        """
        Hook called after a strategy fill to distribute signals to subscribers.

        Should be called from strategy execution code after a trade is filled.

        Args:
            strategy_name: The strategy that generated the trade.
            trade_data: Dictionary with trade details (trade_id, symbol, side, qty, price, timestamp).
        """
        from backend.tradingbot.models.models import SignalProvider

        providers = SignalProvider.objects.filter(
            strategy_name=strategy_name,
            status='active',
        )

        service = cls()
        for provider in providers:
            try:
                service.process_signal(provider.id, trade_data)
            except Exception as e:
                logger.error(
                    f"Error processing signal for provider {provider.id} "
                    f"(strategy={strategy_name}): {e}",
                    exc_info=True,
                )

    # ========================================================================
    # Provider Statistics
    # ========================================================================

    def get_provider_stats(self, provider_id: int) -> Dict[str, Any]:
        """
        Get detailed statistics for a signal provider.

        Args:
            provider_id: The provider's ID.

        Returns:
            Dictionary with provider statistics.
        """
        from backend.tradingbot.models.models import (
            ReplicatedTrade, SignalProvider, SignalSubscription,
        )

        try:
            provider = SignalProvider.objects.get(id=provider_id)
        except SignalProvider.DoesNotExist:
            return {'error': f'Provider {provider_id} not found'}

        # Get subscription stats
        active_subs = SignalSubscription.objects.filter(
            provider=provider,
            status='active',
        ).count()

        total_subs_ever = SignalSubscription.objects.filter(
            provider=provider,
        ).count()

        # Get replicated trade stats across all subscriptions
        replicated_trades = ReplicatedTrade.objects.filter(
            subscription__provider=provider,
        )
        total_replicated = replicated_trades.count()
        executed_trades = replicated_trades.filter(status='executed').count()
        failed_trades = replicated_trades.filter(status='failed').count()
        rejected_delay = replicated_trades.filter(status='rejected_delay').count()
        rejected_risk = replicated_trades.filter(status='rejected_risk').count()

        # Calculate average slippage
        executed_with_slippage = replicated_trades.filter(
            status='executed',
            slippage_pct__isnull=False,
        )
        if executed_with_slippage.exists():
            from django.db.models import Avg
            avg_slippage = executed_with_slippage.aggregate(
                avg=Avg('slippage_pct')
            )['avg']
        else:
            avg_slippage = None

        return {
            'provider_id': provider.id,
            'display_name': provider.display_name,
            'strategy_name': provider.strategy_name,
            'status': provider.status,
            'owner': provider.owner.username,
            'created_at': provider.created_at.isoformat(),
            'subscribers': {
                'active': active_subs,
                'total_ever': total_subs_ever,
                'max_allowed': provider.max_subscribers,
            },
            'signals': {
                'total_sent': provider.total_signals_sent,
            },
            'performance': {
                'win_rate': float(provider.win_rate) if provider.win_rate else None,
                'total_return_pct': float(provider.total_return_pct) if provider.total_return_pct else None,
            },
            'replication': {
                'total_replicated': total_replicated,
                'executed': executed_trades,
                'failed': failed_trades,
                'rejected_delay': rejected_delay,
                'rejected_risk': rejected_risk,
                'execution_rate': (
                    round(executed_trades / total_replicated * 100, 1)
                    if total_replicated > 0 else 0
                ),
                'avg_slippage_pct': float(avg_slippage) if avg_slippage else None,
            },
        }

    # ========================================================================
    # Manual Replication
    # ========================================================================

    def manual_replicate(
        self,
        user: User,
        provider_id: int,
        trade_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Manually replicate a specific signal for a subscriber.

        Args:
            user: The subscriber user requesting manual replication.
            provider_id: The provider whose signal to replicate.
            trade_data: Trade data to replicate.

        Returns:
            Replication result dictionary.

        Raises:
            ValueError: If user has no active subscription to the provider.
        """
        from backend.tradingbot.models.models import SignalSubscription

        subscription = SignalSubscription.objects.filter(
            subscriber=user,
            provider_id=provider_id,
            status='active',
        ).first()

        if not subscription:
            raise ValueError(
                f"No active subscription found for provider {provider_id}"
            )

        return self._replicate_for_subscriber(subscription, trade_data)

    # ========================================================================
    # Helper Methods
    # ========================================================================

    def _get_user_risk_tolerance(self, user: User) -> Optional[int]:
        """Get the user's risk tolerance from their profile."""
        try:
            profile = user.profile
            return profile.risk_tolerance
        except Exception:
            return None

    def _get_subscriber_capital(self, user: User) -> Decimal:
        """Get the subscriber's investable capital from their profile."""
        try:
            profile = user.profile
            if profile.investable_capital:
                return Decimal(str(profile.investable_capital))
        except Exception:
            pass
        # Default capital if profile not available
        return Decimal('100000')
