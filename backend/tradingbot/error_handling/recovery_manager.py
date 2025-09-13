"""Trading Error Recovery Manager.

Centralized error handling with smart recovery mechanisms for production trading operations.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

from .error_types import (
    BrokerConnectionError,
    DataProviderError,
    InsufficientFundsError,
    PositionReconciliationError,
    TradingError,
)


class RecoveryAction(Enum):
    """Recovery actions that can be taken."""

    RETRY_WITH_BACKUP = "retry_with_backup"
    PAUSE_AND_RETRY = "pause_and_retry"
    CONTINUE_WITH_REDUCED_SIZE = "continue_with_reduced_size"
    EMERGENCY_HALT = "emergency_halt"
    LOG_AND_CONTINUE = "log_and_continue"
    SWITCH_TO_PAPER_TRADING = "switch_to_paper_trading"
    REDUCE_POSITION_SIZES = "reduce_position_sizes"


@dataclass
class RecoveryContext:
    """Context information for error recovery."""

    error: TradingError
    timestamp: datetime
    retry_count: int = 0
    max_retries: int = 3
    recovery_actions_taken: list = None
    system_state: dict[str, Any] = None

    def __post_init__(self):
        if self.recovery_actions_taken is None:
            self.recovery_actions_taken = []
        if self.system_state is None:
            self.system_state = {}


class TradingErrorRecoveryManager:
    """Centralized error handling with smart recovery.

    Handles different types of trading errors with appropriate recovery actions:
    - Data provider failures: Switch to backup sources
    - Broker connection issues: Pause and retry
    - Insufficient funds: Reduce position sizes
    - Position reconciliation: Emergency halt
    - Unknown errors: Log and continue with caution
    """

    def __init__(
        self,
        trading_system=None,
        alert_system=None,
        config: dict[str, Any] | None = None,
    ):
        self.trading_system = trading_system
        self.alert_system = alert_system
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Recovery configuration
        self.max_retry_attempts = config.get("max_retry_attempts", 3)
        self.retry_delay_seconds = config.get("retry_delay_seconds", 5)
        self.emergency_halt_threshold = config.get(
            "emergency_halt_threshold", 5
        )  # Max critical errors before halt

        # Error tracking
        self.error_counts = {}
        self.last_error_times = {}
        self.recovery_history = []

        self.logger.info("TradingErrorRecoveryManager initialized")

    async def handle_trading_error(
        self, error: TradingError, context: dict[str, Any] | None = None
    ) -> RecoveryAction:
        """Centralized error handling with smart recovery.

        Args:
            error: The trading error that occurred
            context: Additional context about the error

        Returns:
            RecoveryAction: The recommended recovery action
        """
        try:
            # Set timestamp on error
            error.timestamp = datetime.now()

            # Create recovery context
            recovery_context = RecoveryContext(
                error=error,
                timestamp=error.timestamp,
                retry_count=self.error_counts.get(type(error).__name__, 0),
                system_state=context or {},
            )

            self.logger.error(
                f"Handling trading error: {type(error).__name__}: {error.message}"
            )

            # Determine recovery action based on error type
            recovery_action = await self._determine_recovery_action(
                error, recovery_context
            )

            # Execute recovery action
            await self._execute_recovery_action(recovery_action, recovery_context)

            # Track error and recovery
            self._track_error_and_recovery(error, recovery_action, recovery_context)

            # Send alert if critical
            if recovery_action in [
                RecoveryAction.EMERGENCY_HALT,
                RecoveryAction.SWITCH_TO_PAPER_TRADING,
            ]:
                await self._send_critical_alert(error, recovery_action)

            return recovery_action

        except Exception as e:
            self.logger.critical(f"Error in error recovery manager: {e}")
            return RecoveryAction.EMERGENCY_HALT

    async def _determine_recovery_action(
        self, error: TradingError, context: RecoveryContext
    ) -> RecoveryAction:
        """Determine appropriate recovery action based on error type."""
        if isinstance(error, DataProviderError):
            # Data feed issue-switch to backup source
            if context.retry_count < self.max_retry_attempts:
                return RecoveryAction.RETRY_WITH_BACKUP
            else:
                return RecoveryAction.SWITCH_TO_PAPER_TRADING

        elif isinstance(error, BrokerConnectionError):
            # Broker API issue-pause trading temporarily
            if context.retry_count < self.max_retry_attempts:
                return RecoveryAction.PAUSE_AND_RETRY
            else:
                return RecoveryAction.SWITCH_TO_PAPER_TRADING

        elif isinstance(error, InsufficientFundsError):
            # Account issue-reduce position sizes
            return RecoveryAction.CONTINUE_WITH_REDUCED_SIZE

        elif isinstance(error, PositionReconciliationError):
            # Critical - halt all trading
            return RecoveryAction.EMERGENCY_HALT

        # Unknown error - log and continue with caution
        elif context.retry_count < self.max_retry_attempts:
            return RecoveryAction.LOG_AND_CONTINUE
        else:
            return RecoveryAction.SWITCH_TO_PAPER_TRADING

    async def _execute_recovery_action(
        self, action: RecoveryAction, context: RecoveryContext
    ):
        """Execute the determined recovery action."""
        try:
            if action == RecoveryAction.RETRY_WITH_BACKUP:
                await self._switch_to_backup_data_source()

            elif action == RecoveryAction.PAUSE_AND_RETRY:
                await self._pause_trading_temporarily(duration_minutes=5)

            elif action == RecoveryAction.CONTINUE_WITH_REDUCED_SIZE:
                await self._reduce_position_sizes(reduction_factor=0.5)

            elif action == RecoveryAction.EMERGENCY_HALT:
                await self._emergency_halt("Position reconciliation failed")

            elif action == RecoveryAction.SWITCH_TO_PAPER_TRADING:
                await self._switch_to_paper_trading()

            elif action == RecoveryAction.REDUCE_POSITION_SIZES:
                await self._reduce_position_sizes(reduction_factor=0.25)

            elif action == RecoveryAction.LOG_AND_CONTINUE:
                await self._log_unknown_error(context.error, context.system_state)

            self.logger.info(f"Recovery action executed: {action.value}")

        except Exception as e:
            self.logger.error(f"Failed to execute recovery action {action.value}: {e}")

    async def _switch_to_backup_data_source(self):
        """Switch to backup data source."""
        if self.trading_system and hasattr(self.trading_system, "data_provider"):
            await self.trading_system.data_provider.switch_to_backup()
        self.logger.info("Switched to backup data source")

    async def _pause_trading_temporarily(self, duration_minutes: int = 5):
        """Pause trading temporarily."""
        if self.trading_system:
            await self.trading_system.pause_trading(duration_minutes)
        self.logger.info(f"Trading paused for {duration_minutes} minutes")

    async def _reduce_position_sizes(self, reduction_factor: float = 0.5):
        """Reduce position sizes."""
        if self.trading_system:
            await self.trading_system.reduce_position_sizes(reduction_factor)
        self.logger.info(f"Position sizes reduced by {reduction_factor: .1%}")

    async def _emergency_halt(self, reason: str):
        """Emergency halt all trading."""
        if self.trading_system:
            await self.trading_system.emergency_halt(reason)
        self.logger.critical(f"EMERGENCY HALT: {reason}")

    async def _switch_to_paper_trading(self):
        """Switch to paper trading mode."""
        if self.trading_system:
            await self.trading_system.switch_to_paper_trading()
        self.logger.warning("Switched to paper trading mode")

    async def _log_unknown_error(self, error: TradingError, context: dict[str, Any]):
        """Log unknown error and continue with caution."""
        self.logger.warning(
            f"Unknown error occurred: {error.message}, context: {context}"
        )
        # Could implement additional logging to external systems here

    async def _send_critical_alert(self, error: TradingError, action: RecoveryAction):
        """Send critical alert for serious errors."""
        if self.alert_system:
            await self.alert_system.send_critical_alert(
                f"Critical trading error: {type(error).__name__}",
                f"Error: {error.message}\nAction taken: {action.value}",
                priority="CRITICAL",
            )

    def _track_error_and_recovery(
        self, error: TradingError, action: RecoveryAction, context: RecoveryContext
    ):
        """Track error occurrence and recovery actions."""
        error_type = type(error).__name__

        # Update error counts
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        self.last_error_times[error_type] = context.timestamp

        # Add to recovery history
        self.recovery_history.append(
            {
                "timestamp": context.timestamp,
                "error_type": error_type,
                "error_message": error.message,
                "recovery_action": action.value,
                "retry_count": context.retry_count,
            }
        )

        # Keep only last 100 recovery actions
        if len(self.recovery_history) > 100:
            self.recovery_history = self.recovery_history[-100:]

    def get_error_statistics(self) -> dict[str, Any]:
        """Get error statistics and recovery history."""
        return {
            "error_counts": self.error_counts,
            "last_error_times": self.last_error_times,
            "recovery_history": self.recovery_history[-20:],  # Last 20 actions
            "total_errors": sum(self.error_counts.values()),
            "unique_error_types": len(self.error_counts),
        }

    def is_system_healthy(self) -> bool:
        """Check if system is healthy based on recent error patterns."""
        now = datetime.now()
        recent_critical_errors = 0

        # Count critical errors in last hour
        for error_type, last_time in self.last_error_times.items():
            if now - last_time < timedelta(hours=1):
                if error_type in [
                    "PositionReconciliationError",
                    "BrokerConnectionError",
                ]:
                    recent_critical_errors += 1

        return recent_critical_errors < self.emergency_halt_threshold
