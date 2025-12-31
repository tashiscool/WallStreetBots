"""
Live Trading Reconciliation System.

Ported from Nautilus Trader's reconciliation module.
Validates orders, fills, positions, and balances against broker reports.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import logging

logger = logging.getLogger(__name__)


class ReconciliationType(Enum):
    """Type of reconciliation."""
    ORDER = "order"
    FILL = "fill"
    POSITION = "position"
    BALANCE = "balance"
    FULL = "full"


class DiscrepancyType(Enum):
    """Type of discrepancy found."""
    MISSING_ORDER = "missing_order"
    EXTRA_ORDER = "extra_order"
    ORDER_STATUS_MISMATCH = "order_status_mismatch"
    ORDER_QUANTITY_MISMATCH = "order_quantity_mismatch"
    MISSING_FILL = "missing_fill"
    EXTRA_FILL = "extra_fill"
    FILL_PRICE_MISMATCH = "fill_price_mismatch"
    FILL_QUANTITY_MISMATCH = "fill_quantity_mismatch"
    POSITION_QUANTITY_MISMATCH = "position_quantity_mismatch"
    POSITION_COST_MISMATCH = "position_cost_mismatch"
    MISSING_POSITION = "missing_position"
    EXTRA_POSITION = "extra_position"
    BALANCE_MISMATCH = "balance_mismatch"
    MARGIN_MISMATCH = "margin_mismatch"


class DiscrepancySeverity(Enum):
    """Severity of discrepancy."""
    LOW = "low"        # Minor difference within tolerance
    MEDIUM = "medium"  # Significant but not critical
    HIGH = "high"      # Critical, may affect trading
    CRITICAL = "critical"  # Requires immediate attention


class ReconciliationAction(Enum):
    """Action to take for discrepancy."""
    IGNORE = "ignore"
    LOG_ONLY = "log_only"
    SYNC_FROM_BROKER = "sync_from_broker"
    SYNC_TO_BROKER = "sync_to_broker"
    ALERT = "alert"
    HALT_TRADING = "halt_trading"


@dataclass
class Order:
    """Order for reconciliation."""
    order_id: str
    symbol: str
    quantity: int
    side: str
    order_type: str
    status: str
    limit_price: Optional[Decimal] = None
    filled_quantity: int = 0
    average_fill_price: Optional[Decimal] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


@dataclass
class Fill:
    """Fill/execution for reconciliation."""
    fill_id: str
    order_id: str
    symbol: str
    quantity: int
    price: Decimal
    side: str
    timestamp: datetime
    commission: Decimal = Decimal("0")


@dataclass
class Position:
    """Position for reconciliation."""
    symbol: str
    quantity: int
    average_price: Decimal
    market_value: Decimal
    unrealized_pnl: Decimal = Decimal("0")
    cost_basis: Decimal = Decimal("0")


@dataclass
class AccountBalance:
    """Account balance for reconciliation."""
    cash: Decimal
    buying_power: Decimal
    equity: Decimal
    margin_used: Decimal = Decimal("0")
    maintenance_margin: Decimal = Decimal("0")


@dataclass
class Discrepancy:
    """Represents a reconciliation discrepancy."""
    discrepancy_type: DiscrepancyType
    severity: DiscrepancySeverity
    description: str
    local_value: Any
    broker_value: Any
    symbol: Optional[str] = None
    order_id: Optional[str] = None
    recommended_action: ReconciliationAction = ReconciliationAction.LOG_ONLY
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "type": self.discrepancy_type.value,
            "severity": self.severity.value,
            "description": self.description,
            "local_value": str(self.local_value),
            "broker_value": str(self.broker_value),
            "symbol": self.symbol,
            "order_id": self.order_id,
            "action": self.recommended_action.value,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class ReconciliationResult:
    """Result of reconciliation."""
    reconciliation_type: ReconciliationType
    is_reconciled: bool
    discrepancies: List[Discrepancy] = field(default_factory=list)
    items_checked: int = 0
    items_matched: int = 0
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def match_rate(self) -> float:
        """Percentage of items matched."""
        if self.items_checked == 0:
            return 1.0
        return self.items_matched / self.items_checked

    @property
    def critical_discrepancies(self) -> List[Discrepancy]:
        """Get critical discrepancies."""
        return [
            d for d in self.discrepancies
            if d.severity == DiscrepancySeverity.CRITICAL
        ]


class ReconciliationConfig:
    """Configuration for reconciliation tolerances."""

    def __init__(
        self,
        price_tolerance: Decimal = Decimal("0.01"),
        quantity_tolerance: int = 0,
        balance_tolerance: Decimal = Decimal("0.01"),
        auto_sync: bool = False,
        halt_on_critical: bool = True,
    ):
        """
        Initialize reconciliation config.

        Args:
            price_tolerance: Max price difference to ignore
            quantity_tolerance: Max quantity difference to ignore
            balance_tolerance: Max balance difference to ignore (%)
            auto_sync: Automatically sync discrepancies
            halt_on_critical: Halt trading on critical discrepancies
        """
        self.price_tolerance = price_tolerance
        self.quantity_tolerance = quantity_tolerance
        self.balance_tolerance = balance_tolerance
        self.auto_sync = auto_sync
        self.halt_on_critical = halt_on_critical


class ReconciliationEngine:
    """
    Reconciles local state with broker state.

    Validates orders, fills, positions, and balances.
    """

    def __init__(
        self,
        config: Optional[ReconciliationConfig] = None,
    ):
        """
        Initialize reconciliation engine.

        Args:
            config: Reconciliation configuration
        """
        self.config = config or ReconciliationConfig()
        self._callbacks: List[Callable[[List[Discrepancy]], None]] = []
        self._history: List[ReconciliationResult] = []

    def on_discrepancy(
        self,
        callback: Callable[[List[Discrepancy]], None],
    ) -> None:
        """Register callback for discrepancies."""
        self._callbacks.append(callback)

    def _notify_callbacks(self, discrepancies: List[Discrepancy]) -> None:
        """Notify callbacks of discrepancies."""
        for callback in self._callbacks:
            try:
                callback(discrepancies)
            except Exception as e:
                logger.error(f"Reconciliation callback error: {e}")

    def reconcile_orders(
        self,
        local_orders: Dict[str, Order],
        broker_orders: Dict[str, Order],
    ) -> ReconciliationResult:
        """
        Reconcile local orders with broker orders.

        Args:
            local_orders: Local order state
            broker_orders: Broker-reported orders

        Returns:
            ReconciliationResult
        """
        discrepancies = []
        items_checked = 0
        items_matched = 0

        all_order_ids = set(local_orders.keys()) | set(broker_orders.keys())

        for order_id in all_order_ids:
            items_checked += 1
            local = local_orders.get(order_id)
            broker = broker_orders.get(order_id)

            if local is None and broker is not None:
                # Missing locally
                discrepancies.append(Discrepancy(
                    discrepancy_type=DiscrepancyType.MISSING_ORDER,
                    severity=DiscrepancySeverity.HIGH,
                    description=f"Order {order_id} exists at broker but not locally",
                    local_value=None,
                    broker_value=broker,
                    order_id=order_id,
                    symbol=broker.symbol,
                    recommended_action=ReconciliationAction.SYNC_FROM_BROKER,
                ))
                continue

            if broker is None and local is not None:
                # Extra locally
                discrepancies.append(Discrepancy(
                    discrepancy_type=DiscrepancyType.EXTRA_ORDER,
                    severity=DiscrepancySeverity.MEDIUM,
                    description=f"Order {order_id} exists locally but not at broker",
                    local_value=local,
                    broker_value=None,
                    order_id=order_id,
                    symbol=local.symbol,
                    recommended_action=ReconciliationAction.LOG_ONLY,
                ))
                continue

            # Both exist - compare
            if local.status != broker.status:
                discrepancies.append(Discrepancy(
                    discrepancy_type=DiscrepancyType.ORDER_STATUS_MISMATCH,
                    severity=DiscrepancySeverity.HIGH,
                    description=f"Order {order_id} status mismatch",
                    local_value=local.status,
                    broker_value=broker.status,
                    order_id=order_id,
                    symbol=local.symbol,
                    recommended_action=ReconciliationAction.SYNC_FROM_BROKER,
                ))
            elif local.filled_quantity != broker.filled_quantity:
                discrepancies.append(Discrepancy(
                    discrepancy_type=DiscrepancyType.ORDER_QUANTITY_MISMATCH,
                    severity=DiscrepancySeverity.HIGH,
                    description=f"Order {order_id} filled quantity mismatch",
                    local_value=local.filled_quantity,
                    broker_value=broker.filled_quantity,
                    order_id=order_id,
                    symbol=local.symbol,
                    recommended_action=ReconciliationAction.SYNC_FROM_BROKER,
                ))
            else:
                items_matched += 1

        result = ReconciliationResult(
            reconciliation_type=ReconciliationType.ORDER,
            is_reconciled=len(discrepancies) == 0,
            discrepancies=discrepancies,
            items_checked=items_checked,
            items_matched=items_matched,
        )

        self._history.append(result)

        if discrepancies:
            self._notify_callbacks(discrepancies)
            for d in discrepancies:
                logger.warning(f"Order discrepancy: {d.description}")

        return result

    def reconcile_positions(
        self,
        local_positions: Dict[str, Position],
        broker_positions: Dict[str, Position],
    ) -> ReconciliationResult:
        """
        Reconcile local positions with broker positions.

        Args:
            local_positions: Local position state
            broker_positions: Broker-reported positions

        Returns:
            ReconciliationResult
        """
        discrepancies = []
        items_checked = 0
        items_matched = 0

        all_symbols = set(local_positions.keys()) | set(broker_positions.keys())

        for symbol in all_symbols:
            items_checked += 1
            local = local_positions.get(symbol)
            broker = broker_positions.get(symbol)

            if local is None and broker is not None:
                if broker.quantity != 0:
                    discrepancies.append(Discrepancy(
                        discrepancy_type=DiscrepancyType.MISSING_POSITION,
                        severity=DiscrepancySeverity.CRITICAL,
                        description=f"Position {symbol} exists at broker but not locally",
                        local_value=0,
                        broker_value=broker.quantity,
                        symbol=symbol,
                        recommended_action=ReconciliationAction.SYNC_FROM_BROKER,
                    ))
                continue

            if broker is None and local is not None:
                if local.quantity != 0:
                    discrepancies.append(Discrepancy(
                        discrepancy_type=DiscrepancyType.EXTRA_POSITION,
                        severity=DiscrepancySeverity.CRITICAL,
                        description=f"Position {symbol} exists locally but not at broker",
                        local_value=local.quantity,
                        broker_value=0,
                        symbol=symbol,
                        recommended_action=ReconciliationAction.SYNC_FROM_BROKER,
                    ))
                continue

            # Both exist - compare
            qty_diff = abs(local.quantity - broker.quantity)

            if qty_diff > self.config.quantity_tolerance:
                discrepancies.append(Discrepancy(
                    discrepancy_type=DiscrepancyType.POSITION_QUANTITY_MISMATCH,
                    severity=DiscrepancySeverity.CRITICAL,
                    description=f"Position {symbol} quantity mismatch",
                    local_value=local.quantity,
                    broker_value=broker.quantity,
                    symbol=symbol,
                    recommended_action=ReconciliationAction.SYNC_FROM_BROKER,
                ))
            elif abs(local.average_price - broker.average_price) > self.config.price_tolerance:
                discrepancies.append(Discrepancy(
                    discrepancy_type=DiscrepancyType.POSITION_COST_MISMATCH,
                    severity=DiscrepancySeverity.MEDIUM,
                    description=f"Position {symbol} cost basis mismatch",
                    local_value=local.average_price,
                    broker_value=broker.average_price,
                    symbol=symbol,
                    recommended_action=ReconciliationAction.SYNC_FROM_BROKER,
                ))
            else:
                items_matched += 1

        result = ReconciliationResult(
            reconciliation_type=ReconciliationType.POSITION,
            is_reconciled=len(discrepancies) == 0,
            discrepancies=discrepancies,
            items_checked=items_checked,
            items_matched=items_matched,
        )

        self._history.append(result)

        if discrepancies:
            self._notify_callbacks(discrepancies)
            for d in discrepancies:
                level = logging.CRITICAL if d.severity == DiscrepancySeverity.CRITICAL else logging.WARNING
                logger.log(level, f"Position discrepancy: {d.description}")

        return result

    def reconcile_balance(
        self,
        local_balance: AccountBalance,
        broker_balance: AccountBalance,
    ) -> ReconciliationResult:
        """
        Reconcile account balances.

        Args:
            local_balance: Local balance state
            broker_balance: Broker-reported balance

        Returns:
            ReconciliationResult
        """
        discrepancies = []
        items_checked = 4  # cash, buying power, equity, margin
        items_matched = 0

        # Check cash
        cash_diff_pct = abs(local_balance.cash - broker_balance.cash) / max(broker_balance.cash, Decimal("1"))
        if cash_diff_pct > self.config.balance_tolerance:
            discrepancies.append(Discrepancy(
                discrepancy_type=DiscrepancyType.BALANCE_MISMATCH,
                severity=DiscrepancySeverity.HIGH,
                description=f"Cash balance mismatch ({cash_diff_pct:.2%})",
                local_value=local_balance.cash,
                broker_value=broker_balance.cash,
                recommended_action=ReconciliationAction.SYNC_FROM_BROKER,
            ))
        else:
            items_matched += 1

        # Check buying power
        bp_diff_pct = abs(local_balance.buying_power - broker_balance.buying_power) / max(broker_balance.buying_power, Decimal("1"))
        if bp_diff_pct > self.config.balance_tolerance:
            discrepancies.append(Discrepancy(
                discrepancy_type=DiscrepancyType.BALANCE_MISMATCH,
                severity=DiscrepancySeverity.MEDIUM,
                description=f"Buying power mismatch ({bp_diff_pct:.2%})",
                local_value=local_balance.buying_power,
                broker_value=broker_balance.buying_power,
                recommended_action=ReconciliationAction.SYNC_FROM_BROKER,
            ))
        else:
            items_matched += 1

        # Check equity
        equity_diff_pct = abs(local_balance.equity - broker_balance.equity) / max(broker_balance.equity, Decimal("1"))
        if equity_diff_pct > self.config.balance_tolerance:
            discrepancies.append(Discrepancy(
                discrepancy_type=DiscrepancyType.BALANCE_MISMATCH,
                severity=DiscrepancySeverity.HIGH,
                description=f"Equity mismatch ({equity_diff_pct:.2%})",
                local_value=local_balance.equity,
                broker_value=broker_balance.equity,
                recommended_action=ReconciliationAction.SYNC_FROM_BROKER,
            ))
        else:
            items_matched += 1

        # Check margin
        if broker_balance.margin_used > 0:
            margin_diff_pct = abs(local_balance.margin_used - broker_balance.margin_used) / broker_balance.margin_used
            if margin_diff_pct > self.config.balance_tolerance:
                discrepancies.append(Discrepancy(
                    discrepancy_type=DiscrepancyType.MARGIN_MISMATCH,
                    severity=DiscrepancySeverity.HIGH,
                    description=f"Margin used mismatch ({margin_diff_pct:.2%})",
                    local_value=local_balance.margin_used,
                    broker_value=broker_balance.margin_used,
                    recommended_action=ReconciliationAction.SYNC_FROM_BROKER,
                ))
            else:
                items_matched += 1
        else:
            items_matched += 1

        result = ReconciliationResult(
            reconciliation_type=ReconciliationType.BALANCE,
            is_reconciled=len(discrepancies) == 0,
            discrepancies=discrepancies,
            items_checked=items_checked,
            items_matched=items_matched,
        )

        self._history.append(result)

        if discrepancies:
            self._notify_callbacks(discrepancies)
            for d in discrepancies:
                logger.warning(f"Balance discrepancy: {d.description}")

        return result

    def full_reconciliation(
        self,
        local_orders: Dict[str, Order],
        broker_orders: Dict[str, Order],
        local_positions: Dict[str, Position],
        broker_positions: Dict[str, Position],
        local_balance: AccountBalance,
        broker_balance: AccountBalance,
    ) -> ReconciliationResult:
        """
        Perform full reconciliation.

        Args:
            local_orders: Local orders
            broker_orders: Broker orders
            local_positions: Local positions
            broker_positions: Broker positions
            local_balance: Local balance
            broker_balance: Broker balance

        Returns:
            Combined ReconciliationResult
        """
        all_discrepancies = []

        # Reconcile orders
        order_result = self.reconcile_orders(local_orders, broker_orders)
        all_discrepancies.extend(order_result.discrepancies)

        # Reconcile positions
        position_result = self.reconcile_positions(local_positions, broker_positions)
        all_discrepancies.extend(position_result.discrepancies)

        # Reconcile balance
        balance_result = self.reconcile_balance(local_balance, broker_balance)
        all_discrepancies.extend(balance_result.discrepancies)

        result = ReconciliationResult(
            reconciliation_type=ReconciliationType.FULL,
            is_reconciled=len(all_discrepancies) == 0,
            discrepancies=all_discrepancies,
            items_checked=(
                order_result.items_checked +
                position_result.items_checked +
                balance_result.items_checked
            ),
            items_matched=(
                order_result.items_matched +
                position_result.items_matched +
                balance_result.items_matched
            ),
        )

        # Check for critical discrepancies
        if result.critical_discrepancies and self.config.halt_on_critical:
            logger.critical(
                f"Critical reconciliation discrepancies found: "
                f"{len(result.critical_discrepancies)}"
            )

        return result

    def get_history(
        self,
        limit: int = 100,
    ) -> List[ReconciliationResult]:
        """Get reconciliation history."""
        return self._history[-limit:]

    def get_status(self) -> Dict[str, Any]:
        """Get reconciliation status."""
        recent = self._history[-10:] if self._history else []

        return {
            "total_reconciliations": len(self._history),
            "recent_results": [
                {
                    "type": r.reconciliation_type.value,
                    "is_reconciled": r.is_reconciled,
                    "match_rate": r.match_rate,
                    "discrepancy_count": len(r.discrepancies),
                    "timestamp": r.timestamp.isoformat(),
                }
                for r in recent
            ],
        }

