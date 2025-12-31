"""Circuit Breaker Persistence Service.

Provides database persistence for circuit breaker state, enabling:
- State restoration after application restarts
- Per-user breaker tracking
- Historical logging of all state changes
- Daily reset handling
- Sync between in-memory CircuitBreaker and database

Usage:
    from backend.auth0login.services.circuit_breaker_persistence import (
        get_circuit_breaker_persistence, CircuitBreakerPersistence
    )

    # Get persistence service for a user (or globally)
    persistence = get_circuit_breaker_persistence(user=request.user)

    # Get or create state for a breaker type
    state = persistence.get_or_create_state('daily_loss')

    # Sync in-memory breaker with DB
    persistence.sync_from_memory(circuit_breaker_instance)

    # Restore state to in-memory breaker on startup
    persistence.restore_to_memory(circuit_breaker_instance)
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from django.contrib.auth.models import User
from django.db import transaction
from django.utils import timezone

if TYPE_CHECKING:
    from backend.tradingbot.risk.monitoring.circuit_breaker import CircuitBreaker

logger = logging.getLogger("wsb.circuit_breaker_persistence")


# Standard breaker types
BREAKER_TYPES = [
    'daily_loss',
    'vix',
    'error_rate',
    'stale_data',
    'global',  # Global breaker covering all conditions
]


class CircuitBreakerPersistence:
    """Service for persisting circuit breaker state to database.

    This service manages the lifecycle of CircuitBreakerState records,
    syncing between in-memory CircuitBreaker instances and the database.

    Features:
    - State persistence across restarts
    - Per-user or global breaker states
    - Automatic history logging
    - Daily reset handling
    - VIX level tracking
    """

    def __init__(self, user: Optional[User] = None):
        """Initialize persistence service.

        Args:
            user: Django User instance (None for global breakers)
        """
        self.user = user

    def _get_state_model(self):
        """Lazy import of CircuitBreakerState model."""
        from backend.tradingbot.models.models import CircuitBreakerState
        return CircuitBreakerState

    def _get_history_model(self):
        """Lazy import of CircuitBreakerHistory model."""
        from backend.tradingbot.models.models import CircuitBreakerHistory
        return CircuitBreakerHistory

    def get_or_create_state(
        self,
        breaker_type: str,
        defaults: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Get or create a circuit breaker state.

        Args:
            breaker_type: Type of circuit breaker
            defaults: Default values for new state

        Returns:
            CircuitBreakerState instance
        """
        CircuitBreakerState = self._get_state_model()

        state, created = CircuitBreakerState.objects.get_or_create(
            user=self.user,
            breaker_type=breaker_type,
            defaults=defaults or {},
        )

        if created:
            logger.info(
                f"Created new circuit breaker state: {breaker_type} "
                f"(user={self.user.username if self.user else 'global'})"
            )

        return state

    def get_state(self, breaker_type: str) -> Optional[Any]:
        """Get circuit breaker state if it exists.

        Args:
            breaker_type: Type of circuit breaker

        Returns:
            CircuitBreakerState instance or None
        """
        CircuitBreakerState = self._get_state_model()

        try:
            return CircuitBreakerState.objects.get(
                user=self.user,
                breaker_type=breaker_type,
            )
        except CircuitBreakerState.DoesNotExist:
            return None

    def get_all_states(self) -> List[Any]:
        """Get all circuit breaker states for the user.

        Returns:
            List of CircuitBreakerState instances
        """
        CircuitBreakerState = self._get_state_model()

        return list(CircuitBreakerState.objects.filter(user=self.user))

    def get_tripped_states(self) -> List[Any]:
        """Get all currently tripped circuit breaker states.

        Returns:
            List of tripped CircuitBreakerState instances
        """
        CircuitBreakerState = self._get_state_model()

        return list(CircuitBreakerState.objects.filter(
            user=self.user,
            status__in=['triggered', 'critical'],
        ))

    @transaction.atomic
    def trip_breaker(
        self,
        breaker_type: str,
        reason: str,
        value: float = None,
        threshold: float = None,
        cooldown_seconds: int = 1800,
        metadata: Dict = None,
    ) -> Any:
        """Trip a circuit breaker and log the action.

        Args:
            breaker_type: Type of circuit breaker
            reason: Reason for tripping
            value: Value that caused the trip
            threshold: Threshold that was exceeded
            cooldown_seconds: Cooldown period
            metadata: Additional context

        Returns:
            CircuitBreakerState instance
        """
        state = self.get_or_create_state(breaker_type)
        CircuitBreakerHistory = self._get_history_model()

        previous_status = state.status

        # Update state
        state.status = 'triggered'
        state.tripped_at = timezone.now()
        state.trip_reason = reason
        state.cooldown_until = timezone.now() + timedelta(seconds=cooldown_seconds)

        if value is not None:
            state.current_value = Decimal(str(value))
        if threshold is not None:
            state.threshold = Decimal(str(threshold))

        state.save()

        # Log history
        CircuitBreakerHistory.log_action(
            state=state,
            action='trip',
            reason=reason,
            value=value,
            threshold=threshold,
            previous_status=previous_status,
            new_status='triggered',
            metadata=metadata or {},
        )

        logger.critical(
            f"CIRCUIT_BREAKER_TRIPPED: {breaker_type} - {reason} "
            f"(user={self.user.username if self.user else 'global'})"
        )

        return state

    @transaction.atomic
    def reset_breaker(
        self,
        breaker_type: str,
        force: bool = False,
        reason: str = "",
    ) -> Dict[str, Any]:
        """Reset a circuit breaker.

        Args:
            breaker_type: Type of circuit breaker
            force: Force reset even if in cooldown
            reason: Reason for reset

        Returns:
            Dictionary with result
        """
        state = self.get_state(breaker_type)

        if not state:
            return {
                'success': False,
                'error': 'State not found',
            }

        if not force and state.is_in_cooldown:
            return {
                'success': False,
                'error': 'Still in cooldown period',
                'cooldown_remaining_seconds': state.cooldown_remaining_seconds,
            }

        CircuitBreakerHistory = self._get_history_model()
        previous_status = state.status

        # Reset state
        reset_success = state.reset(force=force)

        if reset_success:
            # Log history
            CircuitBreakerHistory.log_action(
                state=state,
                action='reset',
                reason=reason or 'Manual reset',
                previous_status=previous_status,
                new_status='ok',
            )

            logger.info(
                f"Circuit breaker reset: {breaker_type} "
                f"(user={self.user.username if self.user else 'global'})"
            )

            return {
                'success': True,
                'breaker_type': breaker_type,
                'previous_status': previous_status,
                'new_status': 'ok',
            }

        return {
            'success': False,
            'error': 'Reset failed',
        }

    @transaction.atomic
    def update_equity(
        self,
        breaker_type: str,
        current_equity: float,
        start_of_day_equity: float = None,
    ) -> Dict[str, Any]:
        """Update equity tracking for drawdown calculation.

        Args:
            breaker_type: Type of circuit breaker
            current_equity: Current portfolio equity
            start_of_day_equity: Starting equity (if new day)

        Returns:
            Dictionary with drawdown info
        """
        state = self.get_or_create_state(breaker_type)

        if start_of_day_equity:
            state.start_of_day_equity = Decimal(str(start_of_day_equity))

        drawdown = state.update_equity(current_equity)

        return {
            'breaker_type': breaker_type,
            'current_equity': current_equity,
            'start_of_day_equity': float(state.start_of_day_equity) if state.start_of_day_equity else None,
            'daily_drawdown': drawdown,
        }

    @transaction.atomic
    def update_vix(self, vix_value: float, breaker_type: str = 'vix') -> Dict[str, Any]:
        """Update VIX tracking.

        Args:
            vix_value: Current VIX value
            breaker_type: Breaker type to update

        Returns:
            Dictionary with VIX info
        """
        state = self.get_or_create_state(breaker_type)
        CircuitBreakerHistory = self._get_history_model()

        previous_vix_level = state.vix_level
        state.update_vix(vix_value)

        # Log VIX level changes
        if previous_vix_level != state.vix_level:
            CircuitBreakerHistory.log_action(
                state=state,
                action='vix_change',
                reason=f"VIX level changed: {previous_vix_level} -> {state.vix_level}",
                value=vix_value,
                previous_status=previous_vix_level,
                new_status=state.vix_level,
                metadata={'vix_value': vix_value},
            )

            logger.info(
                f"VIX level changed: {previous_vix_level} -> {state.vix_level} "
                f"(VIX={vix_value})"
            )

        return {
            'vix_value': vix_value,
            'vix_level': state.vix_level,
            'previous_level': previous_vix_level,
            'changed': previous_vix_level != state.vix_level,
            'position_multiplier': state.position_size_multiplier,
        }

    @transaction.atomic
    def mark_error(
        self,
        breaker_type: str = 'error_rate',
        max_per_minute: int = 5,
    ) -> Dict[str, Any]:
        """Mark an error occurrence.

        Args:
            breaker_type: Breaker type for error tracking
            max_per_minute: Maximum errors before tripping

        Returns:
            Dictionary with error info
        """
        state = self.get_or_create_state(breaker_type)

        tripped = state.mark_error(max_per_minute=max_per_minute)

        if tripped:
            CircuitBreakerHistory = self._get_history_model()
            CircuitBreakerHistory.log_action(
                state=state,
                action='error_spike',
                reason=f"Error rate exceeded: {state.errors_last_minute}/{max_per_minute}",
                value=state.errors_last_minute,
                threshold=max_per_minute,
            )

        return {
            'errors_last_minute': state.errors_last_minute,
            'max_per_minute': max_per_minute,
            'tripped': tripped,
            'status': state.status,
        }

    def mark_data_fresh(self, breaker_type: str = 'stale_data') -> None:
        """Mark data as fresh.

        Args:
            breaker_type: Breaker type for data freshness tracking
        """
        state = self.get_or_create_state(breaker_type)
        state.mark_data_fresh()

    @transaction.atomic
    def daily_reset_all(self, new_equity: float = None) -> List[Dict[str, Any]]:
        """Perform daily reset on all breaker states.

        Args:
            new_equity: Starting equity for the new day

        Returns:
            List of reset results
        """
        CircuitBreakerHistory = self._get_history_model()
        results = []

        for state in self.get_all_states():
            old_drawdown = float(state.daily_drawdown) if state.daily_drawdown else None

            state.daily_reset(new_equity)

            # Log daily reset
            CircuitBreakerHistory.log_action(
                state=state,
                action='daily_reset',
                reason='Daily counter reset',
                metadata={
                    'previous_drawdown': old_drawdown,
                    'new_equity': new_equity,
                },
            )

            results.append({
                'breaker_type': state.breaker_type,
                'previous_drawdown': old_drawdown,
                'new_start_equity': new_equity,
            })

        logger.info(
            f"Daily reset completed for {len(results)} breakers "
            f"(user={self.user.username if self.user else 'global'})"
        )

        return results

    def sync_from_memory(self, breaker: 'CircuitBreaker') -> Any:
        """Sync in-memory CircuitBreaker state to database.

        Args:
            breaker: CircuitBreaker instance to sync from

        Returns:
            CircuitBreakerState instance
        """
        state = self.get_or_create_state('global')

        # Sync tripped state
        if breaker.tripped_at is not None:
            state.status = 'triggered'
            # Convert Unix timestamp to datetime
            state.tripped_at = datetime.fromtimestamp(
                breaker.tripped_at,
                tz=timezone.get_current_timezone()
            )
            state.trip_reason = breaker.reason or ''
            state.cooldown_until = state.tripped_at + timedelta(
                seconds=breaker.lim.cooldown_sec
            )
        else:
            state.status = 'ok'
            state.tripped_at = None
            state.trip_reason = ''
            state.cooldown_until = None

        # Sync error tracking
        state.errors_last_minute = breaker.errors_last_min
        if breaker.errors_window_start:
            state.error_window_start = datetime.fromtimestamp(
                breaker.errors_window_start,
                tz=timezone.get_current_timezone()
            )

        # Sync VIX state
        if breaker.current_vix is not None:
            state.current_vix = Decimal(str(breaker.current_vix))
        state.vix_level = breaker.vix_level.value if breaker.vix_level else 'none'

        if breaker.last_vix_check:
            state.last_vix_check = datetime.fromtimestamp(
                breaker.last_vix_check,
                tz=timezone.get_current_timezone()
            )

        # Sync equity
        if breaker.start_equity:
            state.start_of_day_equity = Decimal(str(breaker.start_equity))

        # Sync data freshness
        if breaker.last_data_ts:
            state.last_data_timestamp = datetime.fromtimestamp(
                breaker.last_data_ts,
                tz=timezone.get_current_timezone()
            )

        state.save()

        logger.debug(
            f"Synced circuit breaker to DB: status={state.status}, "
            f"vix_level={state.vix_level}"
        )

        return state

    def restore_to_memory(self, breaker: 'CircuitBreaker') -> bool:
        """Restore database state to in-memory CircuitBreaker.

        This should be called on application startup to restore
        any triggered breakers.

        Args:
            breaker: CircuitBreaker instance to restore to

        Returns:
            True if a tripped state was restored
        """
        import time

        state = self.get_state('global')

        if not state:
            logger.debug("No persisted circuit breaker state found")
            return False

        restored_trip = False

        # Restore tripped state
        if state.is_tripped and state.tripped_at:
            breaker.tripped_at = state.tripped_at.timestamp()
            breaker.reason = state.trip_reason
            restored_trip = True
            logger.warning(
                f"Restored tripped circuit breaker from DB: {state.trip_reason}"
            )

        # Restore error tracking
        if state.errors_last_minute:
            breaker.errors_last_min = state.errors_last_minute
            if state.error_window_start:
                breaker.errors_window_start = state.error_window_start.timestamp()

        # Restore VIX state
        if state.current_vix is not None:
            breaker.current_vix = float(state.current_vix)

        if state.vix_level:
            from backend.tradingbot.risk.monitoring.circuit_breaker import VIXTripLevel
            try:
                breaker.vix_level = VIXTripLevel(state.vix_level)
            except ValueError:
                breaker.vix_level = VIXTripLevel.NONE

        if state.last_vix_check:
            breaker.last_vix_check = state.last_vix_check.timestamp()

        # Restore equity
        if state.start_of_day_equity:
            breaker.start_equity = float(state.start_of_day_equity)

        # Restore data freshness
        if state.last_data_timestamp:
            breaker.last_data_ts = state.last_data_timestamp.timestamp()
        else:
            breaker.last_data_ts = time.time()  # Mark as fresh

        logger.info(
            f"Circuit breaker state restored from DB: "
            f"status={state.status}, vix_level={state.vix_level}"
        )

        return restored_trip

    def get_history(
        self,
        breaker_type: str = None,
        days: int = 7,
        limit: int = 100,
        actions: List[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get circuit breaker history.

        Args:
            breaker_type: Filter by breaker type (None for all)
            days: Number of days to look back
            limit: Maximum entries to return
            actions: Filter by specific actions

        Returns:
            List of history entry dictionaries
        """
        CircuitBreakerHistory = self._get_history_model()
        CircuitBreakerState = self._get_state_model()

        cutoff = timezone.now() - timedelta(days=days)

        # Build query
        query = CircuitBreakerHistory.objects.filter(
            timestamp__gte=cutoff,
        ).select_related('state')

        if self.user:
            query = query.filter(state__user=self.user)
        else:
            query = query.filter(state__user__isnull=True)

        if breaker_type:
            query = query.filter(state__breaker_type=breaker_type)

        if actions:
            query = query.filter(action__in=actions)

        entries = query.order_by('-timestamp')[:limit]

        return [entry.to_dict() for entry in entries]

    def get_status_summary(self) -> Dict[str, Any]:
        """Get summary of all circuit breaker states.

        Returns:
            Dictionary with status summary
        """
        states = self.get_all_states()
        tripped_states = self.get_tripped_states()

        # Determine overall status
        if any(s.status == 'critical' for s in states):
            overall_status = 'critical'
        elif any(s.is_tripped for s in states):
            overall_status = 'triggered'
        elif any(s.status == 'warning' for s in states):
            overall_status = 'warning'
        else:
            overall_status = 'ok'

        # Calculate overall trading permission
        can_trade = all(s.can_trade for s in states) if states else True

        # Get minimum position multiplier
        if states:
            min_multiplier = min(s.position_size_multiplier for s in states)
        else:
            min_multiplier = 1.0

        return {
            'overall_status': overall_status,
            'can_trade': can_trade,
            'position_size_multiplier': min_multiplier,
            'breakers': {
                s.breaker_type: s.to_dict() for s in states
            },
            'tripped_count': len(tripped_states),
            'tripped_types': [s.breaker_type for s in tripped_states],
        }

    def initialize_default_states(self) -> List[Any]:
        """Initialize default breaker states if they don't exist.

        Returns:
            List of created CircuitBreakerState instances
        """
        created = []

        for breaker_type in BREAKER_TYPES:
            state = self.get_state(breaker_type)
            if not state:
                state = self.get_or_create_state(breaker_type)
                created.append(state)

        if created:
            logger.info(
                f"Initialized {len(created)} default circuit breaker states"
            )

        return created


def get_circuit_breaker_persistence(user: Optional[User] = None) -> CircuitBreakerPersistence:
    """Get circuit breaker persistence service.

    Args:
        user: Django User instance (None for global)

    Returns:
        CircuitBreakerPersistence instance
    """
    return CircuitBreakerPersistence(user=user)


def restore_circuit_breaker_on_startup(breaker: 'CircuitBreaker', user: Optional[User] = None) -> bool:
    """Restore circuit breaker state from database on startup.

    Args:
        breaker: CircuitBreaker instance to restore
        user: Django User for the breaker

    Returns:
        True if a tripped state was restored
    """
    persistence = get_circuit_breaker_persistence(user)
    return persistence.restore_to_memory(breaker)


def sync_circuit_breaker_to_db(breaker: 'CircuitBreaker', user: Optional[User] = None) -> Any:
    """Sync in-memory circuit breaker state to database.

    Args:
        breaker: CircuitBreaker instance to sync
        user: Django User for the breaker

    Returns:
        CircuitBreakerState instance
    """
    persistence = get_circuit_breaker_persistence(user)
    return persistence.sync_from_memory(breaker)


def perform_daily_reset(user: Optional[User] = None, new_equity: float = None) -> List[Dict[str, Any]]:
    """Perform daily reset on all circuit breaker states.

    Args:
        user: Django User (None for global)
        new_equity: Starting equity for the new day

    Returns:
        List of reset results
    """
    persistence = get_circuit_breaker_persistence(user)
    return persistence.daily_reset_all(new_equity)
