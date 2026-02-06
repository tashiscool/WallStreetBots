"""Shadow execution client for canary testing.

This module provides a shadow execution client that logs orders
without actually submitting them to the broker.
"""
from __future__ import annotations
import logging
from typing import Dict, Any, List
from .interfaces import ExecutionClient, OrderRequest, OrderAck

log = logging.getLogger("wsb.shadow_execution")


class ShadowExecutionClient(ExecutionClient):
    """Shadow execution client that logs but never submits orders.
    
    Useful for:
    - Canary strategy testing
    - First week of new changes
    - Dry-run validation
    - Performance testing without real orders
    """

    def __init__(self, real: ExecutionClient, log_all_orders: bool = True):
        """Initialize shadow client.

        Args:
            real: Real execution client to wrap
            log_all_orders: Whether to log all order attempts
        """
        super().__init__()
        self.real = real
        self.log_all_orders = log_all_orders
        self.order_log: List[Dict[str, Any]] = []

    def validate_connection(self) -> bool:
        """Validate connection (delegates to real client).
        
        Returns:
            True if connection is valid
        """
        return self.real.validate_connection()

    def place_order(self, req: OrderRequest) -> OrderAck:
        """Place order (shadow mode - logs but doesn't submit).
        
        Args:
            req: Order request
            
        Returns:
            Mock acknowledgment
        """
        # Log the order attempt
        order_log_entry = {
            "timestamp": self._get_timestamp(),
            "action": "shadow_order_attempt",
            "client_order_id": req.client_order_id,
            "symbol": req.symbol,
            "side": req.side,
            "quantity": req.qty,
            "order_type": req.type,
            "price": req.limit_price,
            "time_in_force": req.time_in_force,
        }
        
        if self.log_all_orders:
            log.info(f"SHADOW ORDER: {order_log_entry}")
        
        self.order_log.append(order_log_entry)
        
        # Return mock acknowledgment
        return OrderAck(
            client_order_id=req.client_order_id,
            broker_order_id=None,
            accepted=True,
            reason="shadow_mode"
        )

    def get_order(self, broker_order_id: str) -> Dict[str, Any]:
        """Get order status (returns empty dict in shadow mode).
        
        Args:
            broker_order_id: Broker order ID
            
        Returns:
            Empty dictionary
        """
        log.debug(f"SHADOW: get_order called for {broker_order_id}")
        return {}

    def list_open_orders(self) -> List[Dict[str, Any]]:
        """List open orders (returns empty list in shadow mode).
        
        Returns:
            Empty list
        """
        log.debug("SHADOW: list_open_orders called")
        return []

    def cancel_order(self, broker_order_id: str) -> bool:
        """Cancel order (returns True in shadow mode).
        
        Args:
            broker_order_id: Broker order ID
            
        Returns:
            Always True
        """
        log.info(f"SHADOW: cancel_order called for {broker_order_id}")
        return True

    def reconcile(self, client_order_id: str) -> Dict[str, Any] | None:
        """Reconcile order (returns None in shadow mode).
        
        Args:
            client_order_id: Client order ID
            
        Returns:
            None
        """
        log.debug(f"SHADOW: reconcile called for {client_order_id}")
        return None

    def get_order_log(self) -> List[Dict[str, Any]]:
        """Get log of all shadow orders.
        
        Returns:
            List of order log entries
        """
        return self.order_log.copy()

    def clear_order_log(self) -> None:
        """Clear the order log."""
        self.order_log.clear()
        log.info("SHADOW: Order log cleared")

    def get_order_count(self) -> int:
        """Get count of shadow orders.
        
        Returns:
            Number of orders logged
        """
        return len(self.order_log)

    def _get_timestamp(self) -> str:
        """Get current timestamp.
        
        Returns:
            ISO timestamp string
        """
        from datetime import datetime, timezone
        return datetime.now(timezone.utc).isoformat()

    def status(self) -> Dict[str, Any]:
        """Get shadow client status.
        
        Returns:
            Status information
        """
        return {
            "mode": "shadow",
            "real_client_connected": self.validate_connection(),
            "orders_logged": len(self.order_log),
            "log_all_orders": self.log_all_orders,
        }


class CanaryExecutionClient(ShadowExecutionClient):
    """Canary execution client with allocation limits.
    
    Extends shadow client with canary-specific features:
    - Allocation limits
    - Gradual rollout
    - Performance monitoring
    """

    def __init__(
        self, 
        real: ExecutionClient, 
        canary_allocation_pct: float = 0.1,
        max_daily_orders: int = 10
    ):
        """Initialize canary client.
        
        Args:
            real: Real execution client
            canary_allocation_pct: Percentage of allocation for canary
            max_daily_orders: Maximum orders per day
        """
        super().__init__(real, log_all_orders=True)
        self.canary_allocation_pct = canary_allocation_pct
        self.max_daily_orders = max_daily_orders
        self.daily_order_count = 0
        self.last_reset_date = self._get_current_date()

    def place_order(self, req: OrderRequest) -> OrderAck:
        """Place order with canary limits.
        
        Args:
            req: Order request
            
        Returns:
            Order acknowledgment
        """
        # Reset daily counter if new day
        current_date = self._get_current_date()
        if current_date != self.last_reset_date:
            self.daily_order_count = 0
            self.last_reset_date = current_date

        # Check daily limit
        if self.daily_order_count >= self.max_daily_orders:
            log.warning(f"CANARY: Daily order limit reached ({self.max_daily_orders})")
            return OrderAck(
                client_order_id=req.client_order_id,
                broker_order_id=None,
                accepted=False,
                reason="canary_daily_limit_exceeded"
            )

        # Increment counter
        self.daily_order_count += 1

        # Call parent shadow implementation
        return super().place_order(req)

    def _get_current_date(self) -> str:
        """Get current date string.
        
        Returns:
            Date string in YYYY-MM-DD format
        """
        from datetime import datetime, timezone
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def status(self) -> Dict[str, Any]:
        """Get canary client status.
        
        Returns:
            Status information
        """
        base_status = super().status()
        base_status.update({
            "canary_allocation_pct": self.canary_allocation_pct,
            "max_daily_orders": self.max_daily_orders,
            "daily_order_count": self.daily_order_count,
            "last_reset_date": self.last_reset_date,
        })
        return base_status
