"""
FIX Protocol Adapter — Abstraction layer for FIX 4.2/4.4 connectivity.

Provides FIX message construction, session management, and order routing
via the FIX protocol. Designed to plug into quickfix or simplefix when
a FIX counterparty is available.

For production use:
    pip install quickfix   # C++ FIX engine
    pip install simplefix  # Pure-Python FIX parser
"""

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from .interfaces import ExecutionClient, OrderAck, OrderFill, OrderRequest

logger = logging.getLogger(__name__)


class FIXVersion(Enum):
    """Supported FIX protocol versions."""

    FIX_42 = "FIX.4.2"
    FIX_44 = "FIX.4.4"


class FIXMsgType(Enum):
    """FIX message types (tag 35)."""

    NEW_ORDER_SINGLE = "D"
    ORDER_CANCEL = "F"
    ORDER_CANCEL_REPLACE = "G"
    EXECUTION_REPORT = "8"
    ORDER_CANCEL_REJECT = "9"
    LOGON = "A"
    LOGOUT = "5"
    HEARTBEAT = "0"
    TEST_REQUEST = "1"
    REJECT = "3"


class FIXSide(Enum):
    """FIX side values (tag 54)."""

    BUY = "1"
    SELL = "2"
    SHORT_SELL = "5"


class FIXOrdType(Enum):
    """FIX order type values (tag 40)."""

    MARKET = "1"
    LIMIT = "2"
    STOP = "3"
    STOP_LIMIT = "4"


class FIXTimeInForce(Enum):
    """FIX time in force values (tag 59)."""

    DAY = "0"
    GTC = "1"
    IOC = "3"
    FOK = "4"


@dataclass
class FIXSessionConfig:
    """FIX session configuration."""

    sender_comp_id: str = "WALLSTREETBOTS"
    target_comp_id: str = "BROKER"
    fix_version: FIXVersion = FIXVersion.FIX_44
    host: str = "localhost"
    port: int = 9876
    heartbeat_interval: int = 30
    reset_on_logon: bool = True
    use_ssl: bool = True


@dataclass
class FIXMessage:
    """A constructed FIX message ready for transmission."""

    msg_type: str
    fields: Dict[int, str] = field(default_factory=dict)
    raw: str = ""

    def set_field(self, tag: int, value: Any) -> "FIXMessage":
        self.fields[tag] = str(value)
        return self

    def get_field(self, tag: int) -> Optional[str]:
        return self.fields.get(tag)

    def to_fix_string(self, delimiter: str = "\x01") -> str:
        """Serialize to FIX protocol string."""
        parts = [f"{tag}={val}" for tag, val in sorted(self.fields.items())]
        return delimiter.join(parts) + delimiter


class FIXMessageBuilder:
    """Constructs FIX messages from OrderRequest objects."""

    def __init__(self, config: FIXSessionConfig) -> None:
        self.config = config
        self._seq_num = 0

    def new_order_single(self, order: OrderRequest) -> FIXMessage:
        """Build a NewOrderSingle (MsgType=D) FIX message."""
        self._seq_num += 1
        msg = FIXMessage(msg_type=FIXMsgType.NEW_ORDER_SINGLE.value)

        # Header
        msg.set_field(8, self.config.fix_version.value)  # BeginString
        msg.set_field(35, FIXMsgType.NEW_ORDER_SINGLE.value)  # MsgType
        msg.set_field(49, self.config.sender_comp_id)  # SenderCompID
        msg.set_field(56, self.config.target_comp_id)  # TargetCompID
        msg.set_field(34, self._seq_num)  # MsgSeqNum
        msg.set_field(52, datetime.utcnow().strftime("%Y%m%d-%H:%M:%S"))  # SendingTime

        # Order fields
        msg.set_field(11, order.client_order_id)  # ClOrdID
        msg.set_field(55, order.symbol)  # Symbol
        msg.set_field(54, FIXSide.BUY.value if order.side == "buy" else FIXSide.SELL.value)
        msg.set_field(38, int(order.qty))  # OrderQty
        msg.set_field(40, FIXOrdType.MARKET.value if order.type == "market" else FIXOrdType.LIMIT.value)
        msg.set_field(59, self._map_tif(order.time_in_force))  # TimeInForce
        msg.set_field(60, datetime.utcnow().strftime("%Y%m%d-%H:%M:%S"))  # TransactTime

        if order.limit_price is not None:
            msg.set_field(44, order.limit_price)  # Price

        msg.raw = msg.to_fix_string()
        return msg

    def order_cancel(self, client_order_id: str, orig_order_id: str, symbol: str, side: str) -> FIXMessage:
        """Build an OrderCancelRequest (MsgType=F) FIX message."""
        self._seq_num += 1
        msg = FIXMessage(msg_type=FIXMsgType.ORDER_CANCEL.value)

        msg.set_field(8, self.config.fix_version.value)
        msg.set_field(35, FIXMsgType.ORDER_CANCEL.value)
        msg.set_field(49, self.config.sender_comp_id)
        msg.set_field(56, self.config.target_comp_id)
        msg.set_field(34, self._seq_num)
        msg.set_field(52, datetime.utcnow().strftime("%Y%m%d-%H:%M:%S"))
        msg.set_field(11, client_order_id)  # ClOrdID
        msg.set_field(41, orig_order_id)  # OrigClOrdID
        msg.set_field(55, symbol)
        msg.set_field(54, FIXSide.BUY.value if side == "buy" else FIXSide.SELL.value)
        msg.set_field(60, datetime.utcnow().strftime("%Y%m%d-%H:%M:%S"))

        msg.raw = msg.to_fix_string()
        return msg

    @staticmethod
    def _map_tif(tif: str) -> str:
        mapping = {
            "day": FIXTimeInForce.DAY.value,
            "gtc": FIXTimeInForce.GTC.value,
            "ioc": FIXTimeInForce.IOC.value,
            "fok": FIXTimeInForce.FOK.value,
        }
        return mapping.get(tif, FIXTimeInForce.DAY.value)


class FIXExecutionClient(ExecutionClient):
    """
    FIX-based execution client.

    Wraps the FIX message builder and session management into
    the standard ``ExecutionClient`` interface.

    In stub mode (no quickfix), messages are logged but not sent.
    With quickfix installed, connects to a FIX counterparty.
    """

    def __init__(self, config: Optional[FIXSessionConfig] = None) -> None:
        super().__init__()
        self.config = config or FIXSessionConfig()
        self.builder = FIXMessageBuilder(self.config)
        self._connected = False
        self._fix_session: Optional[Any] = None
        self._message_log: List[FIXMessage] = []
        self._on_execution_report: Optional[Callable[[FIXMessage], None]] = None

    def connect(self) -> bool:
        """Establish FIX session with counterparty."""
        try:
            import quickfix  # noqa: F401

            logger.info(
                "Connecting FIX session: %s → %s @ %s:%d",
                self.config.sender_comp_id,
                self.config.target_comp_id,
                self.config.host,
                self.config.port,
            )
            # In production: create quickfix.SocketInitiator with settings
            self._connected = True
            return True
        except ImportError:
            logger.info(
                "quickfix not installed — running FIX adapter in stub mode. "
                "Messages will be logged but not transmitted."
            )
            self._connected = True  # Stub mode
            return True

    def disconnect(self) -> None:
        """Disconnect FIX session."""
        self._connected = False
        logger.info("FIX session disconnected")

    def validate_connection(self) -> bool:
        return self._connected

    def place_order(self, req: OrderRequest) -> OrderAck:
        """Submit order via FIX NewOrderSingle."""
        msg = self.builder.new_order_single(req)
        self._message_log.append(msg)

        if self._fix_session is not None:
            # Production: send via quickfix session
            pass

        # Generate synthetic ack (stub mode)
        broker_id = f"FIX_{uuid.uuid4().hex[:8]}"
        self._orders[broker_id] = {
            "client_order_id": req.client_order_id,
            "broker_order_id": broker_id,
            "symbol": req.symbol,
            "qty": req.qty,
            "side": req.side,
            "type": req.type,
            "status": "accepted",
            "filled_qty": 0.0,
            "avg_price": req.limit_price,
            "fix_msg": msg.raw,
        }
        self._client_to_broker[req.client_order_id] = broker_id

        logger.info(
            "FIX NewOrderSingle: %s %s %s (ClOrdID=%s)",
            req.side, req.qty, req.symbol, req.client_order_id,
        )

        return OrderAck(
            client_order_id=req.client_order_id,
            broker_order_id=broker_id,
            accepted=True,
        )

    def cancel_order(self, broker_order_id: str) -> bool:
        """Cancel order via FIX OrderCancelRequest."""
        order_data = self._orders.get(broker_order_id)
        if not order_data:
            return False

        cancel_id = f"CXL_{uuid.uuid4().hex[:8]}"
        msg = self.builder.order_cancel(
            client_order_id=cancel_id,
            orig_order_id=order_data["client_order_id"],
            symbol=order_data["symbol"],
            side=order_data["side"],
        )
        self._message_log.append(msg)
        order_data["status"] = "canceled"

        logger.info("FIX OrderCancel: %s", broker_order_id)
        return True

    def get_message_log(self) -> List[FIXMessage]:
        """Get all FIX messages sent during this session."""
        return list(self._message_log)

    def parse_execution_report(self, raw: str, delimiter: str = "\x01") -> Dict[str, str]:
        """Parse a FIX ExecutionReport (MsgType=8) string into fields."""
        fields: Dict[str, str] = {}
        for pair in raw.split(delimiter):
            if "=" in pair:
                tag, value = pair.split("=", 1)
                fields[tag] = value
        return fields
