"""Recovery Manager Service for graduated circuit breaker recovery.

This module provides graduated recovery after circuit breakers trigger,
allowing users to safely resume trading with position size limits that
gradually increase as confidence is restored.

Recovery Modes:
- PAUSED: No trading allowed (0% position sizing)
- RESTRICTED: 25% position sizes, no new strategies
- CAUTIOUS: 50% position sizes
- NORMAL: Full trading (100% position sizing)

Recovery can be:
- Time-based: Auto-advance after X hours
- Performance-based: Advance if paper trades profitable
- Manual: Require user confirmation at each stage
- Hybrid: Time minimum + performance check
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any

from django.contrib.auth.models import User
from django.db.models import Q
from django.utils import timezone

logger = logging.getLogger("wsb.recovery_manager")


class RecoveryMode(Enum):
    """Recovery modes with progressive trading restrictions."""
    PAUSED = "paused"           # No trading
    RESTRICTED = "restricted"   # 25% position sizes, no new strategies
    CAUTIOUS = "cautious"       # 50% position sizes
    NORMAL = "normal"           # Full trading


class RecoveryStrategy(Enum):
    """How recovery advances through stages."""
    TIME_BASED = "time_based"           # Auto after X hours
    PERFORMANCE_BASED = "performance"    # After N profitable trades
    MANUAL = "manual"                    # User must approve each stage
    HYBRID = "hybrid"                    # Time minimum + performance check


@dataclass
class RecoveryStage:
    """A single stage in the recovery schedule."""
    mode: RecoveryMode
    hours_after_trigger: float = 0
    min_profitable_trades: int = 0
    min_win_rate: float = 0.0
    requires_approval: bool = False
    description: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON storage."""
        return {
            'mode': self.mode.value,
            'hours_after_trigger': self.hours_after_trigger,
            'min_profitable_trades': self.min_profitable_trades,
            'min_win_rate': self.min_win_rate,
            'requires_approval': self.requires_approval,
            'description': self.description,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RecoveryStage":
        """Create from dictionary."""
        return cls(
            mode=RecoveryMode(data['mode']),
            hours_after_trigger=data.get('hours_after_trigger', 0),
            min_profitable_trades=data.get('min_profitable_trades', 0),
            min_win_rate=data.get('min_win_rate', 0.0),
            requires_approval=data.get('requires_approval', False),
            description=data.get('description', ''),
        )


@dataclass
class RecoverySchedule:
    """Complete recovery schedule with multiple stages."""
    stages: list[RecoveryStage] = field(default_factory=list)
    strategy: RecoveryStrategy = RecoveryStrategy.HYBRID

    def to_list(self) -> list[dict]:
        """Convert to list of dicts for JSON storage."""
        return [stage.to_dict() for stage in self.stages]

    @classmethod
    def from_list(cls, data: list[dict], strategy: RecoveryStrategy = RecoveryStrategy.HYBRID) -> "RecoverySchedule":
        """Create from list of dicts."""
        return cls(
            stages=[RecoveryStage.from_dict(d) for d in data],
            strategy=strategy,
        )


@dataclass
class RecoveryStatus:
    """Current recovery status for a user."""
    is_in_recovery: bool = False
    current_mode: RecoveryMode = RecoveryMode.NORMAL
    position_multiplier: float = 1.0
    active_events: list[dict] = field(default_factory=list)
    hours_until_next_stage: float | None = None
    trades_until_next_stage: int | None = None
    can_advance: bool = False
    can_trade: bool = True
    can_activate_new_strategies: bool = True
    message: str = ""


# Default recovery schedules by breaker type
DEFAULT_RECOVERY_SCHEDULES = {
    'daily_loss': [
        RecoveryStage(RecoveryMode.PAUSED, hours_after_trigger=0, description="Trading paused"),
        RecoveryStage(RecoveryMode.RESTRICTED, hours_after_trigger=4, min_profitable_trades=0, description="25% sizing, no new strategies"),
        RecoveryStage(RecoveryMode.CAUTIOUS, hours_after_trigger=12, min_profitable_trades=2, description="50% sizing"),
        RecoveryStage(RecoveryMode.NORMAL, hours_after_trigger=24, min_profitable_trades=3, min_win_rate=0.5, description="Full trading"),
    ],
    'vix_critical': [
        RecoveryStage(RecoveryMode.PAUSED, hours_after_trigger=0, description="Trading paused - VIX critical"),
        RecoveryStage(RecoveryMode.RESTRICTED, hours_after_trigger=2, description="25% sizing, monitor VIX"),
        RecoveryStage(RecoveryMode.CAUTIOUS, hours_after_trigger=6, description="50% sizing"),
        RecoveryStage(RecoveryMode.NORMAL, hours_after_trigger=12, description="Full trading if VIX subsides"),
    ],
    'vix_extreme': [
        RecoveryStage(RecoveryMode.RESTRICTED, hours_after_trigger=0, description="25% sizing - VIX extreme"),
        RecoveryStage(RecoveryMode.CAUTIOUS, hours_after_trigger=2, description="50% sizing"),
        RecoveryStage(RecoveryMode.NORMAL, hours_after_trigger=6, description="Full trading"),
    ],
    'error_rate': [
        RecoveryStage(RecoveryMode.PAUSED, hours_after_trigger=0, description="Trading paused - error spike"),
        RecoveryStage(RecoveryMode.RESTRICTED, hours_after_trigger=1, description="25% sizing, monitor errors"),
        RecoveryStage(RecoveryMode.NORMAL, hours_after_trigger=2, description="Full trading if errors resolved"),
    ],
    'stale_data': [
        RecoveryStage(RecoveryMode.PAUSED, hours_after_trigger=0, description="Trading paused - stale data"),
        RecoveryStage(RecoveryMode.NORMAL, hours_after_trigger=0.5, description="Full trading if data fresh"),
    ],
    'consecutive_loss': [
        RecoveryStage(RecoveryMode.PAUSED, hours_after_trigger=0, description="Trading paused - consecutive losses"),
        RecoveryStage(RecoveryMode.RESTRICTED, hours_after_trigger=2, min_profitable_trades=0, description="25% sizing"),
        RecoveryStage(RecoveryMode.CAUTIOUS, hours_after_trigger=8, min_profitable_trades=2, description="50% sizing"),
        RecoveryStage(RecoveryMode.NORMAL, hours_after_trigger=24, min_profitable_trades=4, min_win_rate=0.5, description="Full trading"),
    ],
    'position_limit': [
        RecoveryStage(RecoveryMode.RESTRICTED, hours_after_trigger=0, description="Position limit hit - reduce exposure"),
        RecoveryStage(RecoveryMode.NORMAL, hours_after_trigger=0, description="Full trading when positions closed"),
    ],
    'margin_call': [
        RecoveryStage(RecoveryMode.PAUSED, hours_after_trigger=0, description="Trading paused - margin risk"),
        RecoveryStage(RecoveryMode.RESTRICTED, hours_after_trigger=4, description="25% sizing, reduce exposure"),
        RecoveryStage(RecoveryMode.CAUTIOUS, hours_after_trigger=24, description="50% sizing"),
        RecoveryStage(RecoveryMode.NORMAL, hours_after_trigger=48, description="Full trading"),
    ],
    'manual': [
        RecoveryStage(RecoveryMode.PAUSED, hours_after_trigger=0, requires_approval=True, description="Trading paused"),
        RecoveryStage(RecoveryMode.NORMAL, hours_after_trigger=0, requires_approval=True, description="Full trading on approval"),
    ],
}


class RecoveryManagerService:
    """Manages graduated recovery after circuit breakers trigger.

    This service:
    - Creates recovery events when breakers trip
    - Tracks recovery progress through stages
    - Provides position size multipliers
    - Handles auto-advancement based on time/performance
    - Supports manual approval workflows
    """

    def __init__(self, user: User):
        """Initialize recovery manager for a user.

        Args:
            user: Django User instance
        """
        self.user = user

    def _get_model(self):
        """Lazy import of CircuitBreakerEvent model."""
        from backend.tradingbot.models import CircuitBreakerEvent
        return CircuitBreakerEvent

    def get_recovery_schedule(self, breaker_type: str) -> list[dict]:
        """Get recovery schedule for a breaker type.

        Args:
            breaker_type: Type of circuit breaker

        Returns:
            List of recovery stage dictionaries
        """
        stages = DEFAULT_RECOVERY_SCHEDULES.get(breaker_type, DEFAULT_RECOVERY_SCHEDULES['daily_loss'])
        return [stage.to_dict() for stage in stages]

    def create_breaker_event(
        self,
        breaker_type: str,
        trigger_value: float,
        trigger_threshold: float,
        context: dict = None,
        notes: str = "",
    ) -> Any:
        """Create a new circuit breaker event with recovery schedule.

        Args:
            breaker_type: Type of circuit breaker
            trigger_value: Value that caused the trigger
            trigger_threshold: Threshold that was exceeded
            context: Additional context (positions, market conditions)
            notes: Admin notes

        Returns:
            CircuitBreakerEvent instance
        """
        CircuitBreakerEvent = self._get_model()

        # Get recovery schedule
        schedule = self.get_recovery_schedule(breaker_type)

        # Calculate when full trading should resume (from last stage)
        total_hours = max(stage.get('hours_after_trigger', 0) for stage in schedule)
        recovery_until = timezone.now() + timedelta(hours=total_hours)

        # Create the event
        event = CircuitBreakerEvent.objects.create(
            user=self.user,
            breaker_type=breaker_type,
            trigger_value=Decimal(str(trigger_value)),
            trigger_threshold=Decimal(str(trigger_threshold)),
            trigger_context=context or {},
            current_recovery_mode='paused',
            recovery_mode_until=recovery_until,
            recovery_schedule=schedule,
            recovery_stage=0,
            notes=notes,
        )

        logger.warning(
            f"Circuit breaker triggered for user {self.user.id}: "
            f"{breaker_type} (value={trigger_value}, threshold={trigger_threshold})"
        )

        return event

    def get_active_events(self) -> list[Any]:
        """Get all active circuit breaker events for the user.

        Returns:
            QuerySet of active CircuitBreakerEvent instances
        """
        CircuitBreakerEvent = self._get_model()
        return list(CircuitBreakerEvent.objects.filter(
            user=self.user,
            resolved_at__isnull=True,
        ).order_by('-triggered_at'))

    def get_event_history(self, days: int = 30, limit: int = 50) -> list[Any]:
        """Get circuit breaker event history.

        Args:
            days: Number of days to look back
            limit: Maximum number of events

        Returns:
            List of CircuitBreakerEvent instances
        """
        CircuitBreakerEvent = self._get_model()
        cutoff = timezone.now() - timedelta(days=days)

        return list(CircuitBreakerEvent.objects.filter(
            user=self.user,
            triggered_at__gte=cutoff,
        ).order_by('-triggered_at')[:limit])

    def get_current_mode(self) -> RecoveryMode:
        """Get current recovery mode (most restrictive of all active events).

        Returns:
            Most restrictive RecoveryMode
        """
        active_events = self.get_active_events()

        if not active_events:
            return RecoveryMode.NORMAL

        # Find most restrictive mode
        mode_priority = {
            'paused': 0,
            'restricted': 1,
            'cautious': 2,
            'normal': 3,
        }

        most_restrictive = min(
            active_events,
            key=lambda e: mode_priority.get(e.current_recovery_mode, 3)
        )

        return RecoveryMode(most_restrictive.current_recovery_mode)

    def get_position_size_multiplier(self) -> float:
        """Get position size multiplier based on current recovery mode.

        Returns:
            Multiplier (0.0 to 1.0)
        """
        mode = self.get_current_mode()

        multipliers = {
            RecoveryMode.PAUSED: 0.0,
            RecoveryMode.RESTRICTED: 0.25,
            RecoveryMode.CAUTIOUS: 0.5,
            RecoveryMode.NORMAL: 1.0,
        }

        return multipliers.get(mode, 1.0)

    def can_trade(self) -> bool:
        """Check if trading is allowed.

        Returns:
            True if trading is allowed
        """
        return self.get_current_mode() != RecoveryMode.PAUSED

    def can_activate_new_strategies(self) -> bool:
        """Check if new strategies can be activated.

        Returns:
            True if new strategies can be activated
        """
        mode = self.get_current_mode()
        # Only allow new strategies in CAUTIOUS or NORMAL mode
        return mode in (RecoveryMode.CAUTIOUS, RecoveryMode.NORMAL)

    def get_recovery_status(self) -> RecoveryStatus:
        """Get comprehensive recovery status.

        Returns:
            RecoveryStatus dataclass
        """
        active_events = self.get_active_events()

        if not active_events:
            return RecoveryStatus(
                is_in_recovery=False,
                current_mode=RecoveryMode.NORMAL,
                position_multiplier=1.0,
                message="Trading normally",
            )

        current_mode = self.get_current_mode()
        multiplier = self.get_position_size_multiplier()

        # Calculate time/trades until next stage
        hours_until = None
        trades_until = None
        can_advance = False

        for event in active_events:
            if event.can_advance_recovery:
                can_advance = True

            if event.recovery_schedule and event.recovery_stage < len(event.recovery_schedule) - 1:
                next_stage = event.recovery_schedule[event.recovery_stage + 1]

                # Time remaining
                if 'hours_after_trigger' in next_stage:
                    required_hours = next_stage['hours_after_trigger']
                    elapsed = event.duration_hours
                    remaining = max(0, required_hours - elapsed)
                    if hours_until is None or remaining < hours_until:
                        hours_until = remaining

                # Trades remaining
                if 'min_profitable_trades' in next_stage:
                    required = next_stage['min_profitable_trades']
                    have = event.recovery_profitable_trades
                    remaining = max(0, required - have)
                    if trades_until is None or remaining < trades_until:
                        trades_until = remaining

        # Build message
        mode_messages = {
            RecoveryMode.PAUSED: "Trading paused - circuit breaker active",
            RecoveryMode.RESTRICTED: f"Trading restricted - position sizes at {int(multiplier * 100)}%",
            RecoveryMode.CAUTIOUS: f"Trading cautiously - position sizes at {int(multiplier * 100)}%",
            RecoveryMode.NORMAL: "Trading normally",
        }

        return RecoveryStatus(
            is_in_recovery=True,
            current_mode=current_mode,
            position_multiplier=multiplier,
            active_events=[e.to_dict() for e in active_events],
            hours_until_next_stage=hours_until,
            trades_until_next_stage=trades_until,
            can_advance=can_advance,
            can_trade=self.can_trade(),
            can_activate_new_strategies=self.can_activate_new_strategies(),
            message=mode_messages.get(current_mode, ""),
        )

    def check_auto_recovery(self) -> list[dict]:
        """Check and advance recovery for all active events.

        Returns:
            List of advancement results
        """
        results = []
        active_events = self.get_active_events()

        for event in active_events:
            if event.can_advance_recovery:
                old_mode = event.current_recovery_mode
                success = event.advance_recovery_stage(notes="Auto-advanced by recovery check")

                if success:
                    results.append({
                        'event_id': event.id,
                        'breaker_type': event.breaker_type,
                        'old_mode': old_mode,
                        'new_mode': event.current_recovery_mode,
                        'resolved': event.resolved_at is not None,
                    })

                    logger.info(
                        f"Recovery advanced for user {self.user.id}: "
                        f"{event.breaker_type} {old_mode} -> {event.current_recovery_mode}"
                    )

        return results

    def advance_recovery(self, event_id: int, force: bool = False, notes: str = "") -> dict:
        """Manually advance recovery for a specific event.

        Args:
            event_id: ID of the CircuitBreakerEvent
            force: Force advance even if conditions not met
            notes: Notes for the advancement

        Returns:
            Result dictionary
        """
        CircuitBreakerEvent = self._get_model()

        try:
            event = CircuitBreakerEvent.objects.get(
                id=event_id,
                user=self.user,
                resolved_at__isnull=True,
            )
        except CircuitBreakerEvent.DoesNotExist:
            return {
                'success': False,
                'error': 'Event not found or already resolved',
            }

        old_mode = event.current_recovery_mode
        success = event.advance_recovery_stage(force=force, notes=notes)

        if success:
            logger.info(
                f"Recovery manually advanced for user {self.user.id}: "
                f"{event.breaker_type} {old_mode} -> {event.current_recovery_mode}"
            )

            return {
                'success': True,
                'event_id': event.id,
                'old_mode': old_mode,
                'new_mode': event.current_recovery_mode,
                'resolved': event.resolved_at is not None,
            }
        else:
            return {
                'success': False,
                'error': 'Cannot advance recovery - conditions not met',
                'can_advance': event.can_advance_recovery,
            }

    def reset_breaker(self, event_id: int, confirmation: str = "", notes: str = "") -> dict:
        """Fully reset a circuit breaker event.

        Args:
            event_id: ID of the CircuitBreakerEvent
            confirmation: Confirmation string (must be "CONFIRM")
            notes: Notes for the reset

        Returns:
            Result dictionary
        """
        if confirmation != "CONFIRM":
            return {
                'success': False,
                'error': 'Must provide confirmation string "CONFIRM"',
            }

        CircuitBreakerEvent = self._get_model()

        try:
            event = CircuitBreakerEvent.objects.get(
                id=event_id,
                user=self.user,
                resolved_at__isnull=True,
            )
        except CircuitBreakerEvent.DoesNotExist:
            return {
                'success': False,
                'error': 'Event not found or already resolved',
            }

        event.resolve(method='manual', notes=notes or "Manual reset by user")

        logger.warning(
            f"Circuit breaker manually reset for user {self.user.id}: "
            f"{event.breaker_type} (event_id={event.id})"
        )

        return {
            'success': True,
            'event_id': event.id,
            'resolved': True,
            'message': 'Circuit breaker reset - trading resumed at full capacity',
        }

    def record_trade(self, is_profitable: bool, pnl: float = 0) -> None:
        """Record a trade during recovery period.

        This updates all active events with the trade result.

        Args:
            is_profitable: Whether the trade was profitable
            pnl: P&L of the trade
        """
        active_events = self.get_active_events()

        for event in active_events:
            event.record_recovery_trade(is_profitable, pnl)

    def request_early_recovery(self, event_id: int, justification: str) -> dict:
        """Request early recovery with justification.

        Args:
            event_id: ID of the CircuitBreakerEvent
            justification: User's justification

        Returns:
            Result dictionary
        """
        if not justification or len(justification) < 20:
            return {
                'success': False,
                'error': 'Justification must be at least 20 characters',
            }

        CircuitBreakerEvent = self._get_model()

        try:
            event = CircuitBreakerEvent.objects.get(
                id=event_id,
                user=self.user,
                resolved_at__isnull=True,
            )
        except CircuitBreakerEvent.DoesNotExist:
            return {
                'success': False,
                'error': 'Event not found or already resolved',
            }

        result = event.request_early_recovery(justification)

        logger.info(
            f"Early recovery requested for user {self.user.id}: "
            f"{event.breaker_type} (event_id={event.id})"
        )

        return {
            'success': True,
            **result,
        }

    def get_recovery_timeline(self, event_id: int = None) -> dict:
        """Get recovery timeline for display.

        Args:
            event_id: Specific event ID (or None for all active)

        Returns:
            Timeline data for UI display
        """
        CircuitBreakerEvent = self._get_model()

        if event_id:
            try:
                events = [CircuitBreakerEvent.objects.get(
                    id=event_id,
                    user=self.user,
                )]
            except CircuitBreakerEvent.DoesNotExist:
                return {'error': 'Event not found'}
        else:
            events = self.get_active_events()

        if not events:
            return {
                'has_active_events': False,
                'current_mode': 'normal',
                'message': 'No active circuit breakers',
            }

        # Build timeline for most restrictive event
        mode_priority = {
            'paused': 0,
            'restricted': 1,
            'cautious': 2,
            'normal': 3,
        }

        primary_event = min(
            events,
            key=lambda e: mode_priority.get(e.current_recovery_mode, 3)
        )

        timeline_stages = []
        now = timezone.now()

        for i, stage in enumerate(primary_event.recovery_schedule):
            hours = stage.get('hours_after_trigger', 0)
            stage_time = primary_event.triggered_at + timedelta(hours=hours)

            status = 'completed' if i < primary_event.recovery_stage else (
                'current' if i == primary_event.recovery_stage else 'upcoming'
            )

            timeline_stages.append({
                'index': i,
                'mode': stage['mode'],
                'description': stage.get('description', ''),
                'target_time': stage_time.isoformat(),
                'hours_from_trigger': hours,
                'status': status,
                'is_current': i == primary_event.recovery_stage,
            })

        # Calculate progress
        total_stages = len(primary_event.recovery_schedule)
        current_stage = primary_event.recovery_stage
        progress_pct = (current_stage / max(1, total_stages - 1)) * 100 if total_stages > 1 else 100

        # Time to full recovery
        if primary_event.recovery_mode_until:
            remaining = (primary_event.recovery_mode_until - now).total_seconds() / 3600
            hours_remaining = max(0, remaining)
        else:
            hours_remaining = 0

        return {
            'has_active_events': True,
            'primary_event_id': primary_event.id,
            'breaker_type': primary_event.breaker_type,
            'breaker_type_display': primary_event.get_breaker_type_display(),
            'current_mode': primary_event.current_recovery_mode,
            'current_mode_display': primary_event.get_current_recovery_mode_display(),
            'position_multiplier': primary_event.position_size_multiplier,
            'triggered_at': primary_event.triggered_at.isoformat(),
            'recovery_mode_until': primary_event.recovery_mode_until.isoformat() if primary_event.recovery_mode_until else None,
            'hours_remaining': round(hours_remaining, 1),
            'progress_pct': round(progress_pct, 1),
            'current_stage': current_stage,
            'total_stages': total_stages,
            'timeline': timeline_stages,
            'can_advance': primary_event.can_advance_recovery,
            'recovery_stats': {
                'trades_count': primary_event.recovery_trades_count,
                'profitable_trades': primary_event.recovery_profitable_trades,
                'win_rate': primary_event.recovery_win_rate,
                'pnl': float(primary_event.recovery_pnl),
            },
        }


def get_recovery_manager(user: User) -> RecoveryManagerService:
    """Get recovery manager instance for a user.

    Args:
        user: Django User instance

    Returns:
        RecoveryManagerService instance
    """
    return RecoveryManagerService(user)
