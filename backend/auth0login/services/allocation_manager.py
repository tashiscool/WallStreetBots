"""
Allocation Manager Service - Per-Strategy Capital Allocation Enforcement.

This service manages and enforces per-strategy capital allocation limits,
preventing any single strategy from consuming more capital than allocated.

Features:
- Get/check strategy allocations
- Reserve/release capital for pending orders
- Recalculate allocations based on portfolio value changes
- Reconcile actual positions with tracked exposure
- Generate rebalancing suggestions
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Optional, Dict, List, Any

from django.db import transaction
from django.db.models import Sum
from django.utils import timezone

logger = logging.getLogger(__name__)


class AllocationExceededError(Exception):
    """Raised when a trade would exceed strategy allocation limits."""

    def __init__(self, strategy_name: str, available: float, requested: float, message: str = None):
        self.strategy_name = strategy_name
        self.available = available
        self.requested = requested
        self.message = message or (
            f"{strategy_name} allocation exceeded. "
            f"Available: ${available:,.2f}, Requested: ${requested:,.2f}"
        )
        super().__init__(self.message)


@dataclass
class AllocationInfo:
    """Information about a strategy's allocation."""
    strategy_name: str
    allocated_pct: float
    allocated_amount: float
    current_exposure: float
    reserved_amount: float
    available_capital: float
    utilization_pct: float
    utilization_level: str
    is_maxed_out: bool
    is_enabled: bool


@dataclass
class RebalanceRecommendation:
    """Recommendation for rebalancing a strategy."""
    strategy_name: str
    current_allocation: float
    target_allocation: float
    current_amount: float
    target_amount: float
    action: str  # 'increase', 'decrease', 'maintain'
    adjustment_amount: float
    priority: str  # 'high', 'medium', 'low'
    reason: str


class AllocationManagerService:
    """
    Service for managing and enforcing per-strategy capital allocations.

    Provides:
    - Allocation tracking per strategy
    - Capital reservation for pending orders
    - Enforcement before order submission
    - Reconciliation with actual positions
    - Rebalancing recommendations
    """

    # Default allocation percentages by risk profile
    DEFAULT_ALLOCATIONS = {
        'conservative': {
            'index-baseline': 40,
            'wheel': 30,
            'leaps-tracker': 20,
            'spx-credit-spreads': 10,
        },
        'moderate': {
            'index-baseline': 25,
            'wheel': 20,
            'spx-credit-spreads': 20,
            'momentum-weeklies': 15,
            'wsb-dip-bot': 10,
            'swing-trading': 10,
        },
        'aggressive': {
            'wsb-dip-bot': 20,
            'momentum-weeklies': 20,
            'lotto-scanner': 15,
            'swing-trading': 15,
            'index-baseline': 15,
            'debit-spreads': 15,
        },
    }

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def get_strategy_allocation(self, user, strategy_name: str) -> Optional[AllocationInfo]:
        """
        Get allocation information for a specific strategy.

        Args:
            user: Django user object
            strategy_name: Strategy identifier

        Returns:
            AllocationInfo or None if not found
        """
        from backend.tradingbot.models.models import StrategyAllocationLimit

        try:
            allocation = StrategyAllocationLimit.objects.get(
                user=user,
                strategy_name=strategy_name
            )

            return AllocationInfo(
                strategy_name=allocation.strategy_name,
                allocated_pct=float(allocation.allocated_pct),
                allocated_amount=float(allocation.allocated_amount),
                current_exposure=float(allocation.current_exposure),
                reserved_amount=float(allocation.reserved_amount),
                available_capital=float(allocation.available_capital),
                utilization_pct=allocation.utilization_pct,
                utilization_level=allocation.utilization_level,
                is_maxed_out=allocation.is_maxed_out,
                is_enabled=allocation.is_enabled,
            )

        except StrategyAllocationLimit.DoesNotExist:
            return None

    def check_allocation_available(
        self,
        user,
        strategy_name: str,
        proposed_amount: float
    ) -> tuple[bool, str]:
        """
        Check if the proposed amount is within allocation limits.

        Args:
            user: Django user object
            strategy_name: Strategy identifier
            proposed_amount: Amount to allocate

        Returns:
            Tuple of (is_available: bool, message: str)
        """
        allocation = self.get_strategy_allocation(user, strategy_name)

        if allocation is None:
            return True, "No allocation limit configured for this strategy"

        if not allocation.is_enabled:
            return True, "Allocation limits are disabled for this strategy"

        if proposed_amount <= allocation.available_capital:
            return True, f"Allocation available: ${allocation.available_capital:,.2f}"

        return False, (
            f"Insufficient allocation. Available: ${allocation.available_capital:,.2f}, "
            f"Requested: ${proposed_amount:,.2f}"
        )

    @transaction.atomic
    def reserve_allocation(
        self,
        user,
        strategy_name: str,
        amount: float,
        order_id: str,
        symbol: str
    ) -> bool:
        """
        Reserve capital for a pending order.

        Args:
            user: Django user object
            strategy_name: Strategy identifier
            amount: Amount to reserve
            order_id: Order ID for tracking
            symbol: Trading symbol

        Returns:
            True if reservation successful, False otherwise
        """
        from backend.tradingbot.models.models import (
            StrategyAllocationLimit,
            AllocationReservation
        )

        try:
            allocation = StrategyAllocationLimit.objects.select_for_update().get(
                user=user,
                strategy_name=strategy_name
            )

            if not allocation.is_enabled:
                return True  # No enforcement

            if not allocation.reserve(amount):
                self.logger.warning(
                    f"Failed to reserve ${amount:,.2f} for {strategy_name}. "
                    f"Available: ${float(allocation.available_capital):,.2f}"
                )
                return False

            # Create reservation record
            AllocationReservation.objects.create(
                allocation=allocation,
                order_id=order_id,
                symbol=symbol,
                amount=Decimal(str(amount)),
                status='pending'
            )

            self.logger.info(
                f"Reserved ${amount:,.2f} for {strategy_name} (order {order_id})"
            )
            return True

        except StrategyAllocationLimit.DoesNotExist:
            # No allocation configured - allow order
            return True

    @transaction.atomic
    def release_allocation(
        self,
        user,
        strategy_name: str,
        amount: float,
        order_id: str = None
    ):
        """
        Release a reservation (order cancelled or failed).

        Args:
            user: Django user object
            strategy_name: Strategy identifier
            amount: Amount to release
            order_id: Optional order ID to resolve
        """
        from backend.tradingbot.models.models import (
            StrategyAllocationLimit,
            AllocationReservation
        )

        try:
            allocation = StrategyAllocationLimit.objects.select_for_update().get(
                user=user,
                strategy_name=strategy_name
            )

            allocation.release_reservation(amount)

            # Resolve reservation record if order_id provided
            if order_id:
                try:
                    reservation = AllocationReservation.objects.get(order_id=order_id)
                    reservation.resolve('cancelled')
                except AllocationReservation.DoesNotExist:
                    pass

            self.logger.info(
                f"Released ${amount:,.2f} reservation for {strategy_name}"
            )

        except StrategyAllocationLimit.DoesNotExist:
            pass

    @transaction.atomic
    def confirm_allocation(
        self,
        user,
        strategy_name: str,
        amount: float,
        order_id: str = None
    ):
        """
        Confirm allocation when order is filled (convert reservation to exposure).

        Args:
            user: Django user object
            strategy_name: Strategy identifier
            amount: Amount filled
            order_id: Optional order ID to resolve
        """
        from backend.tradingbot.models.models import (
            StrategyAllocationLimit,
            AllocationReservation
        )

        try:
            allocation = StrategyAllocationLimit.objects.select_for_update().get(
                user=user,
                strategy_name=strategy_name
            )

            allocation.add_exposure(amount)

            # Resolve reservation record if order_id provided
            if order_id:
                try:
                    reservation = AllocationReservation.objects.get(order_id=order_id)
                    reservation.resolve('filled')
                except AllocationReservation.DoesNotExist:
                    pass

            self.logger.info(
                f"Confirmed ${amount:,.2f} exposure for {strategy_name}"
            )

        except StrategyAllocationLimit.DoesNotExist:
            pass

    @transaction.atomic
    def reduce_exposure(
        self,
        user,
        strategy_name: str,
        amount: float
    ):
        """
        Reduce exposure when a position is closed.

        Args:
            user: Django user object
            strategy_name: Strategy identifier
            amount: Amount to reduce
        """
        from backend.tradingbot.models.models import StrategyAllocationLimit

        try:
            allocation = StrategyAllocationLimit.objects.select_for_update().get(
                user=user,
                strategy_name=strategy_name
            )

            allocation.remove_exposure(amount)

            self.logger.info(
                f"Reduced ${amount:,.2f} exposure for {strategy_name}"
            )

        except StrategyAllocationLimit.DoesNotExist:
            pass

    def recalculate_all_allocations(self, user, portfolio_value: float):
        """
        Recalculate all allocation amounts based on current portfolio value.

        Args:
            user: Django user object
            portfolio_value: Current total portfolio value
        """
        from backend.tradingbot.models.models import StrategyAllocationLimit

        allocations = StrategyAllocationLimit.objects.filter(user=user)

        for allocation in allocations:
            allocation.recalculate_allocated_amount(portfolio_value)

        self.logger.info(
            f"Recalculated allocations for portfolio value ${portfolio_value:,.2f}"
        )

    def get_allocation_summary(self, user) -> Dict[str, Any]:
        """
        Get summary of all strategy allocations.

        Args:
            user: Django user object

        Returns:
            Dictionary with allocation summary
        """
        from backend.tradingbot.models.models import StrategyAllocationLimit

        allocations = StrategyAllocationLimit.objects.filter(user=user)

        if not allocations.exists():
            return {
                'configured': False,
                'total_allocated_pct': 0,
                'total_exposure': 0,
                'total_available': 0,
                'strategies': [],
            }

        strategy_data = []
        total_allocated_pct = Decimal('0')
        total_exposure = Decimal('0')
        total_available = Decimal('0')

        for allocation in allocations:
            strategy_data.append(allocation.to_dict())
            total_allocated_pct += allocation.allocated_pct
            total_exposure += allocation.current_exposure
            total_available += allocation.available_capital

        # Sort by utilization (highest first)
        strategy_data.sort(key=lambda x: x['utilization_pct'], reverse=True)

        return {
            'configured': True,
            'total_allocated_pct': float(total_allocated_pct),
            'total_exposure': float(total_exposure),
            'total_available': float(total_available),
            'strategies': strategy_data,
            'warnings': self._get_allocation_warnings(allocations),
        }

    def _get_allocation_warnings(self, allocations) -> List[str]:
        """Generate warnings based on allocation status."""
        warnings = []

        for allocation in allocations:
            if allocation.utilization_level == 'maxed_out':
                warnings.append(
                    f"{allocation.strategy_name} is maxed out (100% utilized)"
                )
            elif allocation.utilization_level == 'critical':
                warnings.append(
                    f"{allocation.strategy_name} is near limit "
                    f"({allocation.utilization_pct:.0f}% utilized)"
                )

        return warnings

    @transaction.atomic
    def reconcile_allocations(self, user, positions: List[Dict]) -> Dict[str, Any]:
        """
        Reconcile tracked allocations with actual positions.

        This should be run periodically to catch any drift between
        tracked exposure and actual position values.

        Args:
            user: Django user object
            positions: List of position dictionaries with 'symbol', 'market_value', 'strategy'

        Returns:
            Reconciliation report
        """
        from backend.tradingbot.models.models import StrategyAllocationLimit

        report = {
            'timestamp': timezone.now().isoformat(),
            'adjustments': [],
            'errors': [],
        }

        # Group positions by strategy
        strategy_exposure = {}
        for position in positions:
            strategy = position.get('strategy', 'unknown')
            market_value = float(position.get('market_value', 0))

            if strategy not in strategy_exposure:
                strategy_exposure[strategy] = 0
            strategy_exposure[strategy] += market_value

        # Update each allocation
        allocations = StrategyAllocationLimit.objects.filter(user=user)

        for allocation in allocations:
            actual_exposure = strategy_exposure.get(allocation.strategy_name, 0)
            tracked_exposure = float(allocation.current_exposure)

            if abs(actual_exposure - tracked_exposure) > 0.01:  # Allow for rounding
                report['adjustments'].append({
                    'strategy': allocation.strategy_name,
                    'tracked_exposure': tracked_exposure,
                    'actual_exposure': actual_exposure,
                    'adjustment': actual_exposure - tracked_exposure,
                })

                # Update to actual
                allocation.current_exposure = Decimal(str(actual_exposure))
                allocation.last_reconciled = timezone.now()
                allocation.save()

        return report

    def initialize_allocations(
        self,
        user,
        profile: str,
        portfolio_value: float,
        enabled_strategies: List[str] = None
    ):
        """
        Initialize allocation limits based on risk profile.

        Args:
            user: Django user object
            profile: 'conservative', 'moderate', or 'aggressive'
            portfolio_value: Current portfolio value
            enabled_strategies: Optional list of strategies to enable
        """
        from backend.tradingbot.models.models import StrategyAllocationLimit

        allocations = self.DEFAULT_ALLOCATIONS.get(profile, self.DEFAULT_ALLOCATIONS['moderate'])

        for strategy_name, pct in allocations.items():
            if enabled_strategies and strategy_name not in enabled_strategies:
                continue

            allocation, created = StrategyAllocationLimit.objects.update_or_create(
                user=user,
                strategy_name=strategy_name,
                defaults={
                    'allocated_pct': Decimal(str(pct)),
                    'allocated_amount': Decimal(str(portfolio_value * pct / 100)),
                    'is_enabled': True,
                }
            )

            if created:
                self.logger.info(f"Created allocation for {strategy_name}: {pct}%")

    def update_allocation(
        self,
        user,
        strategy_name: str,
        allocated_pct: float,
        portfolio_value: float = None
    ):
        """
        Update allocation percentage for a strategy.

        Args:
            user: Django user object
            strategy_name: Strategy identifier
            allocated_pct: New allocation percentage
            portfolio_value: Optional portfolio value for amount calculation
        """
        from backend.tradingbot.models.models import StrategyAllocationLimit

        allocation, created = StrategyAllocationLimit.objects.get_or_create(
            user=user,
            strategy_name=strategy_name,
            defaults={
                'allocated_pct': Decimal(str(allocated_pct)),
            }
        )

        allocation.allocated_pct = Decimal(str(allocated_pct))

        if portfolio_value:
            allocation.allocated_amount = Decimal(str(portfolio_value * allocated_pct / 100))

        allocation.save()

    def get_rebalance_recommendations(
        self,
        user,
        portfolio_value: float,
        target_profile: str = None
    ) -> List[RebalanceRecommendation]:
        """
        Generate rebalancing recommendations.

        Args:
            user: Django user object
            portfolio_value: Current portfolio value
            target_profile: Optional target profile to rebalance towards

        Returns:
            List of RebalanceRecommendation objects
        """
        from backend.tradingbot.models.models import StrategyAllocationLimit

        recommendations = []

        allocations = StrategyAllocationLimit.objects.filter(user=user)

        if target_profile:
            target_allocations = self.DEFAULT_ALLOCATIONS.get(
                target_profile,
                self.DEFAULT_ALLOCATIONS['moderate']
            )
        else:
            # Use current allocations as targets
            target_allocations = {
                a.strategy_name: float(a.allocated_pct)
                for a in allocations
            }

        for allocation in allocations:
            strategy = allocation.strategy_name
            current_pct = float(allocation.current_exposure / Decimal(str(portfolio_value)) * 100) if portfolio_value > 0 else 0
            target_pct = target_allocations.get(strategy, float(allocation.allocated_pct))

            current_amount = float(allocation.current_exposure)
            target_amount = portfolio_value * target_pct / 100

            diff = target_pct - current_pct
            adjustment = target_amount - current_amount

            if abs(diff) < 1:  # Less than 1% difference
                action = 'maintain'
                priority = 'low'
                reason = 'Within acceptable range'
            elif diff > 0:
                action = 'increase'
                priority = 'high' if diff > 5 else 'medium'
                reason = f'Underallocated by {diff:.1f}%'
            else:
                action = 'decrease'
                priority = 'high' if abs(diff) > 5 else 'medium'
                reason = f'Overallocated by {abs(diff):.1f}%'

            recommendations.append(RebalanceRecommendation(
                strategy_name=strategy,
                current_allocation=current_pct,
                target_allocation=target_pct,
                current_amount=current_amount,
                target_amount=target_amount,
                action=action,
                adjustment_amount=adjustment,
                priority=priority,
                reason=reason,
            ))

        # Sort by priority (high first) then by adjustment magnitude
        priority_order = {'high': 0, 'medium': 1, 'low': 2}
        recommendations.sort(
            key=lambda x: (priority_order[x.priority], -abs(x.adjustment_amount))
        )

        return recommendations

    def enforce_allocation(
        self,
        user,
        strategy_name: str,
        proposed_amount: float,
        order_id: str = None,
        symbol: str = None
    ) -> Dict[str, Any]:
        """
        Enforce allocation limit before order submission.

        This is the main enforcement method to call before submitting an order.

        Args:
            user: Django user object
            strategy_name: Strategy identifier
            proposed_amount: Proposed order amount
            order_id: Optional order ID for reservation tracking
            symbol: Trading symbol

        Returns:
            Dict with 'allowed', 'message', and optionally 'allocation' info

        Raises:
            AllocationExceededError if allocation would be exceeded
        """
        allocation = self.get_strategy_allocation(user, strategy_name)

        if allocation is None:
            return {
                'allowed': True,
                'message': 'No allocation limit configured',
                'allocation': None,
            }

        if not allocation.is_enabled:
            return {
                'allowed': True,
                'message': 'Allocation limits disabled for this strategy',
                'allocation': allocation,
            }

        if proposed_amount > allocation.available_capital:
            raise AllocationExceededError(
                strategy_name=strategy_name,
                available=allocation.available_capital,
                requested=proposed_amount,
            )

        # Reserve if order_id provided
        if order_id and symbol:
            success = self.reserve_allocation(
                user, strategy_name, proposed_amount, order_id, symbol
            )
            if not success:
                raise AllocationExceededError(
                    strategy_name=strategy_name,
                    available=allocation.available_capital,
                    requested=proposed_amount,
                    message="Failed to reserve allocation"
                )

        return {
            'allowed': True,
            'message': f'Allocation available. Remaining: ${allocation.available_capital - proposed_amount:,.2f}',
            'allocation': allocation,
        }


# Global singleton instance
_allocation_manager: Optional[AllocationManagerService] = None


def get_allocation_manager() -> AllocationManagerService:
    """Get or create the global AllocationManagerService instance."""
    global _allocation_manager
    if _allocation_manager is None:
        _allocation_manager = AllocationManagerService()
    return _allocation_manager


# Alias for backward compatibility with api_views.py
AllocationManager = AllocationManagerService
