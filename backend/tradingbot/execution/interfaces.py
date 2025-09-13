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
    """Abstract broker execution contract."""

    def validate_connection(self) -> bool:
        raise NotImplementedError

    def place_order(self, req: OrderRequest) -> OrderAck:
        raise NotImplementedError

    def get_order(self, broker_order_id: str) -> Dict:
        raise NotImplementedError

    def list_open_orders(self) -> Dict[str, Dict]:
        raise NotImplementedError

    def cancel_order(self, broker_order_id: str) -> bool:
        raise NotImplementedError

    def reconcile(self, client_order_id: str) -> Optional[OrderFill]:
        """Reconcile final state for a client order id."""
        raise NotImplementedError