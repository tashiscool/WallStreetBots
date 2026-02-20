# backend/tradingbot/execution/interfaces.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Literal, Dict

OrderSide = Literal["buy", "sell"]
OrderType = Literal["market", "limit"]
TimeInForce = Literal["day", "gtc", "ioc", "fok"]


@dataclass(frozen=True)
class OrderRequest:
    client_order_id: str
    symbol: str
    qty: float
    side: OrderSide
    type: OrderType
    time_in_force: TimeInForce = "day"
    limit_price: Optional[float] = None


@dataclass(frozen=True)
class OrderAck:
    client_order_id: str
    broker_order_id: Optional[str]
    accepted: bool
    reason: Optional[str] = None


@dataclass(frozen=True)
class OrderFill:
    broker_order_id: str
    symbol: str
    avg_price: float
    filled_qty: float
    status: Literal["partially_filled", "filled", "canceled", "rejected"]


class ExecutionClient:
    """Abstract broker execution contract.

    Subclass this to implement broker-specific execution logic.
    Default implementations provide stub behavior for testing.
    """

    def __init__(self):
        """Initialize execution client."""
        self._orders: Dict[str, Dict] = {}  # broker_order_id -> order data
        self._client_to_broker: Dict[str, str] = {}  # client_order_id -> broker_order_id
        self._order_counter: int = 0

    def _raise_if_base_client(self, method_name: str) -> None:
        """Enforce abstract contract for direct ExecutionClient usage."""
        if self.__class__ is ExecutionClient:
            raise NotImplementedError(
                f"ExecutionClient.{method_name}() must be implemented by a concrete client"
            )

    def validate_connection(self) -> bool:
        """Validate broker connection.

        Returns:
            True if connection is valid, False otherwise.
            Default implementation returns True (stub mode).
        """
        self._raise_if_base_client("validate_connection")
        return True

    def place_order(self, req: OrderRequest) -> OrderAck:
        """Place an order with the broker.

        Args:
            req: Order request with symbol, quantity, side, etc.

        Returns:
            OrderAck with acceptance status and broker order ID.
            Default implementation accepts all orders (stub mode).
        """
        self._raise_if_base_client("place_order")
        self._order_counter += 1
        broker_order_id = f"stub_{self._order_counter}"

        self._orders[broker_order_id] = {
            "client_order_id": req.client_order_id,
            "broker_order_id": broker_order_id,
            "symbol": req.symbol,
            "qty": req.qty,
            "side": req.side,
            "type": req.type,
            "time_in_force": req.time_in_force,
            "limit_price": req.limit_price,
            "status": "accepted",
            "filled_qty": 0.0,
            "avg_price": None,
        }
        self._client_to_broker[req.client_order_id] = broker_order_id

        return OrderAck(
            client_order_id=req.client_order_id,
            broker_order_id=broker_order_id,
            accepted=True,
            reason=None,
        )

    def get_order(self, broker_order_id: str) -> Dict:
        """Get order details by broker order ID.

        Args:
            broker_order_id: The broker-assigned order ID.

        Returns:
            Order data dictionary, or empty dict if not found.
        """
        self._raise_if_base_client("get_order")
        return self._orders.get(broker_order_id, {})

    def list_open_orders(self) -> Dict[str, Dict]:
        """List all open orders.

        Returns:
            Dictionary mapping broker order IDs to order data.
        """
        self._raise_if_base_client("list_open_orders")
        return {
            oid: data for oid, data in self._orders.items()
            if data.get("status") in ("accepted", "pending", "partially_filled")
        }

    def cancel_order(self, broker_order_id: str) -> bool:
        """Cancel an order.

        Args:
            broker_order_id: The broker-assigned order ID.

        Returns:
            True if cancelled successfully, False otherwise.
        """
        self._raise_if_base_client("cancel_order")
        if broker_order_id in self._orders:
            self._orders[broker_order_id]["status"] = "canceled"
            return True
        return False

    def reconcile(self, client_order_id: str) -> Optional[OrderFill]:
        """Reconcile final state for a client order id.

        Args:
            client_order_id: The client-assigned order ID.

        Returns:
            OrderFill with fill details, or None if order not found.
        """
        self._raise_if_base_client("reconcile")
        broker_order_id = self._client_to_broker.get(client_order_id)
        if not broker_order_id:
            return None

        order_data = self._orders.get(broker_order_id)
        if not order_data:
            return None

        status = order_data.get("status", "pending")
        filled_qty = order_data.get("filled_qty", 0.0)
        avg_price = order_data.get("avg_price")

        # Map internal status to OrderFill status
        if status == "canceled":
            fill_status = "canceled"
        elif status == "rejected":
            fill_status = "rejected"
        elif filled_qty >= order_data.get("qty", 0):
            fill_status = "filled"
        elif filled_qty > 0:
            fill_status = "partially_filled"
        else:
            return None  # Not filled yet

        return OrderFill(
            broker_order_id=broker_order_id,
            symbol=order_data.get("symbol", ""),
            avg_price=avg_price or 0.0,
            filled_qty=filled_qty,
            status=fill_status,
        )


class InMemoryExecutionClient(ExecutionClient):
    """Concrete in-memory execution client for tests and non-broker environments."""

    pass
