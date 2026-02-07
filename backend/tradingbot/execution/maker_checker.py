"""
Maker-Checker (Four-Eyes) Approval Workflow for Trade Execution.

Orders above configurable thresholds require a second approver before
submission to the broker. Supports auto-approval for small orders.
"""

import logging
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from .interfaces import OrderRequest

logger = logging.getLogger(__name__)


class ApprovalStatus(Enum):
    """Status of an approval request."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"
    AUTO_APPROVED = "auto_approved"


@dataclass
class ApprovalThresholds:
    """Thresholds that determine when maker-checker is required."""

    # Orders below these thresholds are auto-approved
    auto_approve_notional: float = 10_000.0  # Auto-approve under $10K
    auto_approve_shares: int = 1_000

    # Orders above these thresholds require approval
    require_approval_notional: float = 10_000.0
    require_approval_shares: int = 1_000

    # Time limits
    approval_timeout_minutes: int = 30
    max_pending_approvals: int = 50

    # Strategy/parameter changes always require approval
    require_approval_for_new_symbols: bool = True
    require_approval_for_market_orders_above: float = 50_000.0


@dataclass
class ApprovalRequest:
    """A pending approval request for an order."""

    request_id: str
    order: OrderRequest
    maker_id: str  # Who submitted the order
    status: ApprovalStatus = ApprovalStatus.PENDING
    notional: float = 0.0
    reason: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    reviewed_at: Optional[datetime] = None
    checker_id: Optional[str] = None  # Who approved/rejected
    checker_comment: str = ""


class MakerCheckerService:
    """
    Four-eyes approval workflow for trade orders.

    Orders below threshold → auto-approved.
    Orders above threshold → held pending checker approval.
    Expired approvals are automatically rejected.
    """

    def __init__(
        self,
        thresholds: Optional[ApprovalThresholds] = None,
        on_approved: Optional[Callable[[ApprovalRequest], None]] = None,
        on_rejected: Optional[Callable[[ApprovalRequest], None]] = None,
    ) -> None:
        self.thresholds = thresholds or ApprovalThresholds()
        self._on_approved = on_approved
        self._on_rejected = on_rejected
        self._lock = threading.Lock()
        self._pending: Dict[str, ApprovalRequest] = {}
        self._history: List[ApprovalRequest] = []
        self._approved_symbols: set = set()

    # ------------------------------------------------------------------
    # Maker: submit order for approval
    # ------------------------------------------------------------------

    def submit(
        self,
        order: OrderRequest,
        maker_id: str,
        current_price: float = 0.0,
        reason: str = "",
    ) -> ApprovalRequest:
        """Submit an order for approval.

        Auto-approves if below thresholds. Otherwise queues for checker.

        Args:
            order: The order to approve.
            maker_id: Identity of the person/system submitting.
            current_price: Current market price for notional calculation.
            reason: Why the order is being placed.

        Returns:
            ``ApprovalRequest`` with status.
        """
        with self._lock:
            self._expire_stale()

            notional = order.qty * current_price
            request_id = str(uuid.uuid4())[:12]

            request = ApprovalRequest(
                request_id=request_id,
                order=order,
                maker_id=maker_id,
                notional=notional,
                reason=reason,
            )

            # Check if auto-approval applies
            if self._can_auto_approve(order, notional):
                request.status = ApprovalStatus.AUTO_APPROVED
                request.reviewed_at = datetime.utcnow()
                request.checker_id = "system"
                request.checker_comment = "Auto-approved: below thresholds"
                self._history.append(request)
                logger.info(
                    "Order auto-approved: %s %s %s ($%.0f)",
                    order.side, order.qty, order.symbol, notional,
                )
                if self._on_approved:
                    self._on_approved(request)
                return request

            # Requires checker approval
            if len(self._pending) >= self.thresholds.max_pending_approvals:
                request.status = ApprovalStatus.REJECTED
                request.reviewed_at = datetime.utcnow()
                request.checker_comment = "Too many pending approvals"
                self._history.append(request)
                return request

            self._pending[request_id] = request
            logger.warning(
                "Order HELD for approval: %s %s %s ($%.0f) [%s]",
                order.side, order.qty, order.symbol, notional, request_id,
            )
            return request

    # ------------------------------------------------------------------
    # Checker: approve or reject
    # ------------------------------------------------------------------

    def approve(
        self,
        request_id: str,
        checker_id: str,
        comment: str = "",
    ) -> Optional[ApprovalRequest]:
        """Approve a pending order.

        Args:
            request_id: ID of the approval request.
            checker_id: Identity of the checker.
            comment: Optional approval comment.

        Returns:
            Updated ``ApprovalRequest`` or None if not found.
        """
        with self._lock:
            request = self._pending.pop(request_id, None)
            if request is None:
                return None

            # Maker cannot approve their own order
            if request.maker_id == checker_id:
                request.status = ApprovalStatus.REJECTED
                request.checker_comment = "Maker cannot approve own order"
                request.checker_id = checker_id
                request.reviewed_at = datetime.utcnow()
                self._pending[request_id] = request
                logger.warning("Self-approval blocked for %s", request_id)
                return request

            request.status = ApprovalStatus.APPROVED
            request.checker_id = checker_id
            request.checker_comment = comment
            request.reviewed_at = datetime.utcnow()
            self._history.append(request)

            logger.info(
                "Order APPROVED by %s: %s %s %s [%s]",
                checker_id, request.order.side, request.order.qty,
                request.order.symbol, request_id,
            )

            if self._on_approved:
                self._on_approved(request)

            return request

    def reject(
        self,
        request_id: str,
        checker_id: str,
        comment: str = "",
    ) -> Optional[ApprovalRequest]:
        """Reject a pending order.

        Args:
            request_id: ID of the approval request.
            checker_id: Identity of the checker.
            comment: Rejection reason.

        Returns:
            Updated ``ApprovalRequest`` or None if not found.
        """
        with self._lock:
            request = self._pending.pop(request_id, None)
            if request is None:
                return None

            request.status = ApprovalStatus.REJECTED
            request.checker_id = checker_id
            request.checker_comment = comment
            request.reviewed_at = datetime.utcnow()
            self._history.append(request)

            logger.info(
                "Order REJECTED by %s: %s %s %s — %s [%s]",
                checker_id, request.order.side, request.order.qty,
                request.order.symbol, comment, request_id,
            )

            if self._on_rejected:
                self._on_rejected(request)

            return request

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_pending(self) -> List[ApprovalRequest]:
        """Get all pending approval requests."""
        with self._lock:
            self._expire_stale()
            return list(self._pending.values())

    def get_request(self, request_id: str) -> Optional[ApprovalRequest]:
        """Look up a specific request (pending or historical)."""
        with self._lock:
            if request_id in self._pending:
                return self._pending[request_id]
            for r in reversed(self._history):
                if r.request_id == request_id:
                    return r
            return None

    def get_history(self, last_n: int = 100) -> List[ApprovalRequest]:
        """Get recent approval history."""
        return self._history[-last_n:]

    def get_stats(self) -> Dict[str, Any]:
        """Get approval statistics."""
        total = len(self._history)
        by_status = {}
        for r in self._history:
            by_status[r.status.value] = by_status.get(r.status.value, 0) + 1
        return {
            "pending": len(self._pending),
            "total_processed": total,
            "by_status": by_status,
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _can_auto_approve(self, order: OrderRequest, notional: float) -> bool:
        if notional > self.thresholds.auto_approve_notional:
            return False
        if order.qty > self.thresholds.auto_approve_shares:
            return False
        if (
            order.type == "market"
            and notional > self.thresholds.require_approval_for_market_orders_above
        ):
            return False
        if (
            self.thresholds.require_approval_for_new_symbols
            and order.symbol not in self._approved_symbols
        ):
            self._approved_symbols.add(order.symbol)
            return False  # First order for a new symbol requires approval
        return True

    def _expire_stale(self) -> None:
        now = datetime.utcnow()
        cutoff = now - timedelta(minutes=self.thresholds.approval_timeout_minutes)
        expired_ids = [
            rid for rid, req in self._pending.items()
            if req.created_at < cutoff
        ]
        for rid in expired_ids:
            req = self._pending.pop(rid)
            req.status = ApprovalStatus.EXPIRED
            req.reviewed_at = now
            req.checker_comment = "Approval timeout expired"
            self._history.append(req)
            logger.warning("Approval expired: %s", rid)
            if self._on_rejected:
                self._on_rejected(req)
