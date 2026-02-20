"""RPC Message Types"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional


class RPCMessageType(Enum):
    """Types of RPC messages."""
    # Status
    STARTUP = "startup"
    SHUTDOWN = "shutdown"
    STATUS = "status"

    # Trading
    ENTRY = "entry"
    ENTRY_FILL = "entry_fill"
    EXIT = "exit"
    EXIT_FILL = "exit_fill"

    # Manual actions
    FORCE_ENTRY = "force_entry"
    FORCE_EXIT = "force_exit"

    # Alerts
    PROTECTION_TRIGGER = "protection_trigger"
    RISK_ALERT = "risk_alert"
    ERROR = "error"
    WARNING = "warning"

    # Performance
    DAILY_REPORT = "daily_report"
    WEEKLY_REPORT = "weekly_report"
    MONTHLY_REPORT = "monthly_report"

    # System
    CONFIG_RELOAD = "config_reload"
    BALANCE_UPDATE = "balance_update"
    STRATEGY_UPDATE = "strategy_update"


@dataclass
class RPCMessage:
    """RPC Message structure."""
    msg_type: RPCMessageType
    message: str
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    priority: int = 0  # Higher = more important
    require_ack: bool = False  # Require user acknowledgment

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'type': self.msg_type.value,
            'message': self.message,
            'data': self.data,
            'timestamp': self.timestamp.isoformat(),
            'priority': self.priority,
        }

    @classmethod
    def trade_entry(
        cls,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        strategy: str,
    ) -> 'RPCMessage':
        """Create trade entry message."""
        return cls(
            msg_type=RPCMessageType.ENTRY,
            message=f"{'BUY' if side == 'buy' else 'SELL'} {symbol}: {quantity} @ ${price:.2f}",
            data={
                'symbol': symbol,
                'side': side,
                'quantity': quantity,
                'price': price,
                'strategy': strategy,
            },
            priority=3,
        )

    @classmethod
    def trade_exit(
        cls,
        symbol: str,
        quantity: float,
        entry_price: float,
        exit_price: float,
        pnl: float,
        pnl_pct: float,
    ) -> 'RPCMessage':
        """Create trade exit message."""
        emoji = ""
        return cls(
            msg_type=RPCMessageType.EXIT,
            message=f"{emoji} Closed {symbol}: PnL ${pnl:.2f} ({pnl_pct:.1f}%)",
            data={
                'symbol': symbol,
                'quantity': quantity,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pnl': pnl,
                'pnl_pct': pnl_pct,
            },
            priority=4,
        )

    @classmethod
    def risk_alert(cls, message: str, severity: str = "warning") -> 'RPCMessage':
        """Create risk alert message."""
        return cls(
            msg_type=RPCMessageType.RISK_ALERT,
            message=f"RISK ALERT: {message}",
            data={'severity': severity},
            priority=5,
            require_ack=severity == "critical",
        )

    @classmethod
    def error(cls, message: str, exception: Optional[str] = None) -> 'RPCMessage':
        """Create error message."""
        return cls(
            msg_type=RPCMessageType.ERROR,
            message=f"ERROR: {message}",
            data={'exception': exception},
            priority=5,
        )
