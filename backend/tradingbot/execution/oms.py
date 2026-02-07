"""
Order Management System (OMS) — Central order routing and lifecycle management.

Ties together:
    PreTradeCompliance → MakerChecker → ExecutionClient → OrderStateMachine

All orders flow through the OMS. No direct broker access from strategy code.
"""

import logging
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from .interfaces import ExecutionClient, OrderAck, OrderFill, OrderRequest
from .order_state_machine import OrderStatus
from .pre_trade_compliance import ComplianceResult, PreTradeComplianceService
from .maker_checker import ApprovalRequest, ApprovalStatus, MakerCheckerService

logger = logging.getLogger(__name__)


@dataclass
class ManagedOrder:
    """An order tracked by the OMS with full lifecycle state."""

    order_id: str
    order: OrderRequest
    status: OrderStatus = OrderStatus.INITIALIZED
    compliance_result: Optional[ComplianceResult] = None
    approval: Optional[ApprovalRequest] = None
    broker_ack: Optional[OrderAck] = None
    fill: Optional[OrderFill] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    submitted_by: str = ""
    error: str = ""


class OrderManagementSystem:
    """
    Central OMS that enforces the order pipeline:

        Strategy → OMS.submit() → PreTradeCompliance → MakerChecker → Broker

    All order state is tracked here. Strategies never touch the broker directly.
    """

    def __init__(
        self,
        execution_client: Optional[ExecutionClient] = None,
        compliance: Optional[PreTradeComplianceService] = None,
        maker_checker: Optional[MakerCheckerService] = None,
    ) -> None:
        self.execution_client = execution_client or ExecutionClient()
        self.compliance = compliance or PreTradeComplianceService()
        self.maker_checker = maker_checker
        self._lock = threading.Lock()
        self._orders: Dict[str, ManagedOrder] = {}
        self._on_fill: Optional[Callable[[ManagedOrder], None]] = None
        self._on_reject: Optional[Callable[[ManagedOrder], None]] = None

    # ------------------------------------------------------------------
    # Order submission
    # ------------------------------------------------------------------

    def submit_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        order_type: str = "limit",
        limit_price: Optional[float] = None,
        time_in_force: str = "day",
        submitted_by: str = "system",
        current_price: Optional[float] = None,
    ) -> ManagedOrder:
        """Submit a new order through the full OMS pipeline.

        Args:
            symbol: Ticker symbol.
            qty: Order quantity.
            side: 'buy' or 'sell'.
            order_type: 'market' or 'limit'.
            limit_price: Limit price (required for limit orders).
            time_in_force: 'day', 'gtc', 'ioc', 'fok'.
            submitted_by: Identity of submitter.
            current_price: Current market price for compliance checks.

        Returns:
            ``ManagedOrder`` with current status.
        """
        client_order_id = f"OMS_{uuid.uuid4().hex[:12]}"
        request = OrderRequest(
            client_order_id=client_order_id,
            symbol=symbol,
            qty=qty,
            side=side,
            type=order_type,
            time_in_force=time_in_force,
            limit_price=limit_price,
        )

        managed = ManagedOrder(
            order_id=client_order_id,
            order=request,
            submitted_by=submitted_by,
        )

        with self._lock:
            self._orders[client_order_id] = managed

        price = current_price or limit_price or 0.0

        # Step 1: Pre-trade compliance
        compliance_result = self.compliance.check(request, current_price=price)
        managed.compliance_result = compliance_result

        if not compliance_result.approved:
            managed.status = OrderStatus.DENIED
            managed.error = "; ".join(compliance_result.violations)
            managed.updated_at = datetime.utcnow()
            logger.warning("Order DENIED by compliance: %s — %s", client_order_id, managed.error)
            if self._on_reject:
                self._on_reject(managed)
            return managed

        # Step 2: Maker-checker (if configured)
        if self.maker_checker is not None:
            approval = self.maker_checker.submit(
                request, maker_id=submitted_by, current_price=price,
            )
            managed.approval = approval

            if approval.status == ApprovalStatus.PENDING:
                # Order held — will be routed when approved
                managed.status = OrderStatus.INITIALIZED
                managed.updated_at = datetime.utcnow()
                logger.info("Order held for approval: %s", client_order_id)
                return managed
            elif approval.status in (ApprovalStatus.REJECTED, ApprovalStatus.EXPIRED):
                managed.status = OrderStatus.DENIED
                managed.error = f"Approval {approval.status.value}: {approval.checker_comment}"
                managed.updated_at = datetime.utcnow()
                if self._on_reject:
                    self._on_reject(managed)
                return managed
            # AUTO_APPROVED or APPROVED — proceed

        # Step 3: Route to broker
        return self._route_to_broker(managed)

    # ------------------------------------------------------------------
    # Approval callback (for maker-checker integration)
    # ------------------------------------------------------------------

    def on_approval(self, request_id: str) -> Optional[ManagedOrder]:
        """Called when a pending order is approved by checker.

        Looks up the held order and routes it to the broker.
        """
        with self._lock:
            for managed in self._orders.values():
                if (
                    managed.approval is not None
                    and managed.approval.request_id == request_id
                    and managed.approval.status == ApprovalStatus.APPROVED
                ):
                    return self._route_to_broker(managed)
        return None

    # ------------------------------------------------------------------
    # Order management
    # ------------------------------------------------------------------

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order."""
        with self._lock:
            managed = self._orders.get(order_id)
            if managed is None:
                return False

            if managed.broker_ack and managed.broker_ack.broker_order_id:
                success = self.execution_client.cancel_order(
                    managed.broker_ack.broker_order_id
                )
                if success:
                    managed.status = OrderStatus.CANCELED
                    managed.updated_at = datetime.utcnow()
                return success

            # Not yet submitted — just cancel locally
            managed.status = OrderStatus.CANCELED
            managed.updated_at = datetime.utcnow()
            return True

    def get_order(self, order_id: str) -> Optional[ManagedOrder]:
        """Get order by OMS order ID."""
        return self._orders.get(order_id)

    def get_open_orders(self) -> List[ManagedOrder]:
        """Get all open (non-terminal) orders."""
        terminal = {
            OrderStatus.FILLED, OrderStatus.CANCELED,
            OrderStatus.REJECTED, OrderStatus.EXPIRED,
            OrderStatus.DENIED, OrderStatus.ERROR,
        }
        return [
            o for o in self._orders.values()
            if o.status not in terminal
        ]

    def get_all_orders(self) -> List[ManagedOrder]:
        """Get all orders."""
        return list(self._orders.values())

    def get_order_count(self) -> Dict[str, int]:
        """Get order counts by status."""
        counts: Dict[str, int] = {}
        for o in self._orders.values():
            key = o.status.value
            counts[key] = counts.get(key, 0) + 1
        return counts

    # ------------------------------------------------------------------
    # Fill processing
    # ------------------------------------------------------------------

    def process_fill(self, broker_order_id: str, fill: OrderFill) -> Optional[ManagedOrder]:
        """Process an execution report / fill from the broker."""
        with self._lock:
            for managed in self._orders.values():
                if (
                    managed.broker_ack
                    and managed.broker_ack.broker_order_id == broker_order_id
                ):
                    managed.fill = fill
                    if fill.status == "filled":
                        managed.status = OrderStatus.FILLED
                    elif fill.status == "partially_filled":
                        managed.status = OrderStatus.PARTIALLY_FILLED
                    elif fill.status == "canceled":
                        managed.status = OrderStatus.CANCELED
                    elif fill.status == "rejected":
                        managed.status = OrderStatus.REJECTED
                    managed.updated_at = datetime.utcnow()

                    # Update compliance position tracking
                    if fill.status in ("filled", "partially_filled"):
                        self.compliance.update_position(
                            fill.symbol,
                            self.compliance._positions.get(fill.symbol, 0)
                            + (fill.filled_qty if managed.order.side == "buy" else -fill.filled_qty),
                        )

                    if self._on_fill and fill.status in ("filled", "partially_filled"):
                        self._on_fill(managed)

                    return managed
        return None

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def set_on_fill(self, callback: Callable[[ManagedOrder], None]) -> None:
        """Register callback for order fills."""
        self._on_fill = callback

    def set_on_reject(self, callback: Callable[[ManagedOrder], None]) -> None:
        """Register callback for order rejections."""
        self._on_reject = callback

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _route_to_broker(self, managed: ManagedOrder) -> ManagedOrder:
        """Send order to execution client."""
        try:
            ack = self.execution_client.place_order(managed.order)
            managed.broker_ack = ack
            if ack.accepted:
                managed.status = OrderStatus.SUBMITTED
                logger.info(
                    "Order routed: %s → broker_id=%s",
                    managed.order_id, ack.broker_order_id,
                )
            else:
                managed.status = OrderStatus.REJECTED
                managed.error = ack.reason or "Broker rejected"
                if self._on_reject:
                    self._on_reject(managed)
        except Exception as exc:
            managed.status = OrderStatus.ERROR
            managed.error = str(exc)
            logger.error("Order routing failed: %s — %s", managed.order_id, exc)
            if self._on_reject:
                self._on_reject(managed)

        managed.updated_at = datetime.utcnow()
        return managed
