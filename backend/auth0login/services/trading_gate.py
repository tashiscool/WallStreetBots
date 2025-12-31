"""
Trading Gate Service

This service manages the paper-to-live trading transition,
enforcing safety requirements before users can trade with real money.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Optional

from django.contrib.auth.models import User
from django.db import transaction
from django.utils import timezone

from ..models import TradingGate

logger = logging.getLogger(__name__)


@dataclass
class GateRequirement:
    """A single requirement for live trading approval."""
    name: str
    description: str
    met: bool
    current_value: str
    required_value: str


@dataclass
class GateStatus:
    """Complete status of a user's trading gate."""
    user_id: int
    username: str

    # Current state
    is_paper_trading: bool
    live_trading_approved: bool
    live_trading_requested: bool

    # Progress
    days_in_paper: int
    days_required: int
    days_remaining: int

    # Performance (from paper trading)
    total_trades: int
    total_pnl: float
    total_pnl_pct: float
    win_rate: float
    sharpe_ratio: Optional[float]

    # Requirements
    requirements: list[GateRequirement]
    all_requirements_met: bool

    # Timestamps
    paper_started_at: Optional[datetime]
    requested_at: Optional[datetime]
    approved_at: Optional[datetime]
    approval_method: Optional[str]

    # If denied
    denial_reason: Optional[str]


class TradingGateService:
    """Service for managing paper-to-live trading transitions."""

    # Default requirements
    MIN_PAPER_TRADING_DAYS = 14
    MIN_TRADES_REQUIRED = 10
    MAX_LOSS_PERCENT = 20.0  # Can't lose more than 20% in paper

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def get_or_create_gate(self, user: User) -> TradingGate:
        """Get or create a TradingGate for a user."""
        gate, created = TradingGate.objects.get_or_create(
            user=user,
            defaults={
                'paper_trading_days_required': self.MIN_PAPER_TRADING_DAYS,
            }
        )
        if created:
            self.logger.info(f"Created TradingGate for user {user.username}")
        return gate

    def start_paper_trading(self, user: User) -> TradingGate:
        """Start paper trading period for a user."""
        gate = self.get_or_create_gate(user)
        if not gate.paper_trading_started_at:
            gate.paper_trading_started_at = timezone.now()
            gate.save()
            self.logger.info(f"Paper trading started for {user.username}")
        return gate

    def check_paper_trading_complete(self, gate: TradingGate) -> tuple[bool, str]:
        """Check if minimum paper trading duration has elapsed.

        Returns:
            Tuple of (is_complete, message)
        """
        if not gate.paper_trading_started_at:
            return False, "Paper trading has not been started yet."

        days_completed = gate.days_in_paper_trading
        days_required = gate.paper_trading_days_required

        if days_completed >= days_required:
            return True, f"Paper trading duration met: {days_completed} days completed."
        else:
            remaining = days_required - days_completed
            return False, f"Paper trading incomplete: {remaining} more days required."

    def check_minimum_trades(
        self, user: User, min_trades: int = None
    ) -> tuple[bool, int, str]:
        """Check if user has executed minimum number of paper trades.

        Returns:
            Tuple of (is_met, trade_count, message)
        """
        min_trades = min_trades or self.MIN_TRADES_REQUIRED

        # Get paper trade count from database
        trade_count = self._get_paper_trade_count(user)

        if trade_count >= min_trades:
            return True, trade_count, f"Minimum trades met: {trade_count} trades executed."
        else:
            remaining = min_trades - trade_count
            return False, trade_count, f"Need {remaining} more trades (have {trade_count}/{min_trades})."

    def check_no_catastrophic_loss(
        self, user: User, max_loss_pct: float = None
    ) -> tuple[bool, float, str]:
        """Check if user hasn't suffered catastrophic loss in paper trading.

        Returns:
            Tuple of (is_passed, current_pnl_pct, message)
        """
        max_loss_pct = max_loss_pct or self.MAX_LOSS_PERCENT

        # Get paper trading performance
        pnl_pct = self._get_paper_trading_pnl_pct(user)

        if pnl_pct >= -max_loss_pct:
            if pnl_pct >= 0:
                return True, pnl_pct, f"Paper trading profitable: +{pnl_pct:.2f}%"
            else:
                return True, pnl_pct, f"Paper trading loss within limits: {pnl_pct:.2f}%"
        else:
            return False, pnl_pct, f"Paper trading loss exceeds limit: {pnl_pct:.2f}% (max allowed: -{max_loss_pct}%)"

    def get_requirements(self, user: User) -> list[GateRequirement]:
        """Get all requirements and their current status."""
        gate = self.get_or_create_gate(user)

        requirements = []

        # 1. Paper trading duration
        duration_met, duration_msg = self.check_paper_trading_complete(gate)
        requirements.append(GateRequirement(
            name="paper_trading_duration",
            description="Complete minimum paper trading period",
            met=duration_met,
            current_value=f"{gate.days_in_paper_trading} days",
            required_value=f"{gate.paper_trading_days_required} days",
        ))

        # 2. Minimum trades
        trades_met, trade_count, trades_msg = self.check_minimum_trades(user)
        requirements.append(GateRequirement(
            name="minimum_trades",
            description="Execute minimum number of paper trades",
            met=trades_met,
            current_value=f"{trade_count} trades",
            required_value=f"{self.MIN_TRADES_REQUIRED} trades",
        ))

        # 3. No catastrophic loss
        loss_ok, pnl_pct, loss_msg = self.check_no_catastrophic_loss(user)
        requirements.append(GateRequirement(
            name="no_catastrophic_loss",
            description="Avoid excessive losses in paper trading",
            met=loss_ok,
            current_value=f"{pnl_pct:+.2f}%",
            required_value=f">{-self.MAX_LOSS_PERCENT:.0f}%",
        ))

        return requirements

    def get_gate_status(self, user: User) -> GateStatus:
        """Get complete gate status for a user."""
        gate = self.get_or_create_gate(user)
        requirements = self.get_requirements(user)

        # Get performance metrics
        performance = self._get_paper_trading_performance(user)

        return GateStatus(
            user_id=user.id,
            username=user.username,
            is_paper_trading=not gate.live_trading_approved,
            live_trading_approved=gate.live_trading_approved,
            live_trading_requested=gate.live_trading_requested_at is not None,
            days_in_paper=gate.days_in_paper_trading,
            days_required=gate.paper_trading_days_required,
            days_remaining=gate.days_remaining,
            total_trades=performance.get('total_trades', 0),
            total_pnl=performance.get('total_pnl', 0.0),
            total_pnl_pct=performance.get('total_pnl_pct', 0.0),
            win_rate=performance.get('win_rate', 0.0),
            sharpe_ratio=performance.get('sharpe_ratio'),
            requirements=requirements,
            all_requirements_met=all(r.met for r in requirements),
            paper_started_at=gate.paper_trading_started_at,
            requested_at=gate.live_trading_requested_at,
            approved_at=gate.live_trading_approved_at,
            approval_method=gate.approval_method,
            denial_reason=gate.denial_reason,
        )

    def request_live_trading(self, user: User) -> dict:
        """User requests transition to live trading.

        Returns:
            Dict with status, approved flag, and message
        """
        gate = self.get_or_create_gate(user)

        # Check if already approved
        if gate.live_trading_approved:
            return {
                'status': 'already_approved',
                'approved': True,
                'message': 'Live trading is already approved for this account.',
            }

        # Gather performance snapshot
        performance = self._get_paper_trading_performance(user)

        # Record the request
        gate.request_live_trading(performance_snapshot=performance)

        # Attempt auto-approval
        return self.approve_live_trading(user)

    @transaction.atomic
    def approve_live_trading(self, user: User, force: bool = False) -> dict:
        """Attempt to approve live trading for a user.

        Args:
            user: The user requesting approval
            force: If True, approve even if requirements aren't met (admin override)

        Returns:
            Dict with status, approved flag, and message
        """
        gate = self.get_or_create_gate(user)

        if gate.live_trading_approved:
            return {
                'status': 'already_approved',
                'approved': True,
                'message': 'Live trading already approved.',
            }

        if force:
            # Admin override
            gate.approve_live_trading(method='override')
            self.logger.warning(f"Live trading OVERRIDE approved for {user.username}")
            return {
                'status': 'approved',
                'approved': True,
                'message': 'Live trading approved via admin override.',
                'method': 'override',
            }

        # Check all requirements
        requirements = self.get_requirements(user)
        failed_requirements = [r for r in requirements if not r.met]

        if failed_requirements:
            # Deny with reasons
            reasons = [f"- {r.description}: {r.current_value} (need {r.required_value})"
                       for r in failed_requirements]
            denial_message = "Requirements not met:\n" + "\n".join(reasons)
            gate.deny_live_trading(denial_message)

            self.logger.info(f"Live trading denied for {user.username}: {len(failed_requirements)} requirements not met")
            return {
                'status': 'denied',
                'approved': False,
                'message': denial_message,
                'failed_requirements': [r.name for r in failed_requirements],
            }

        # All requirements met - auto-approve
        gate.approve_live_trading(method='auto')
        self.logger.info(f"Live trading AUTO-APPROVED for {user.username}")

        return {
            'status': 'approved',
            'approved': True,
            'message': 'Congratulations! All requirements met. Live trading has been approved.',
            'method': 'auto',
        }

    def is_live_trading_allowed(self, user: User) -> tuple[bool, str]:
        """Check if user is allowed to place live trades.

        Returns:
            Tuple of (is_allowed, reason)
        """
        try:
            gate = TradingGate.objects.get(user=user)
            if gate.live_trading_approved:
                return True, "Live trading approved"
            else:
                return False, "Live trading not yet approved. Complete paper trading requirements first."
        except TradingGate.DoesNotExist:
            return False, "No trading gate found. Please complete the setup wizard."

    def revoke_live_trading(self, user: User, reason: str) -> dict:
        """Revoke live trading approval (admin action).

        Args:
            user: User whose approval is being revoked
            reason: Reason for revocation

        Returns:
            Dict with status and message
        """
        gate = self.get_or_create_gate(user)

        if not gate.live_trading_approved:
            return {
                'status': 'not_approved',
                'message': 'User does not have live trading approval.',
            }

        gate.revoke_live_trading(reason)
        self.logger.warning(f"Live trading REVOKED for {user.username}: {reason}")

        return {
            'status': 'revoked',
            'message': f'Live trading has been revoked. Reason: {reason}',
        }

    # Private helper methods

    def _get_paper_trade_count(self, user: User) -> int:
        """Get count of paper trades for a user.

        In a real implementation, this would query the trades table
        filtered by paper_trading=True.
        """
        # TODO: Implement actual query once paper trades are tracked
        # For now, return a simulated count based on time in paper trading
        try:
            gate = TradingGate.objects.get(user=user)
            if gate.paper_trading_started_at:
                # Simulate ~1 trade per day for demo purposes
                days = gate.days_in_paper_trading
                return min(days * 1, 50)  # Cap at 50
            return 0
        except TradingGate.DoesNotExist:
            return 0

    def _get_paper_trading_pnl_pct(self, user: User) -> float:
        """Get paper trading P&L percentage.

        In a real implementation, this would calculate actual P&L
        from the paper trades table.
        """
        # TODO: Implement actual P&L calculation
        # For now, return a simulated value
        try:
            gate = TradingGate.objects.get(user=user)
            if gate.paper_performance_snapshot:
                return gate.paper_performance_snapshot.get('total_pnl_pct', 0.0)
            # Default to slightly positive for demo
            return 5.0
        except TradingGate.DoesNotExist:
            return 0.0

    def _get_paper_trading_performance(self, user: User) -> dict:
        """Get comprehensive paper trading performance metrics.

        In a real implementation, this would calculate from actual trades.
        """
        # TODO: Implement actual performance calculation
        trade_count = self._get_paper_trade_count(user)
        pnl_pct = self._get_paper_trading_pnl_pct(user)

        return {
            'total_trades': trade_count,
            'total_pnl': pnl_pct * 1000,  # Assuming $100k starting
            'total_pnl_pct': pnl_pct,
            'win_rate': 0.55 if trade_count > 5 else 0.0,  # Demo value
            'sharpe_ratio': 1.2 if trade_count > 10 else None,  # Demo value
            'max_drawdown_pct': -5.0,
            'avg_trade_pnl': pnl_pct * 1000 / trade_count if trade_count > 0 else 0,
            'calculated_at': timezone.now().isoformat(),
        }


# Singleton instance
trading_gate_service = TradingGateService()
