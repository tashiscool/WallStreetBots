"""Daily broker reconciliation system.

Ensures positions, cash, and PnL match between internal ledger and broker
statements within tolerance limits.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import asyncio

logger = logging.getLogger(__name__)


class ReconciliationType(Enum):
    """Types of reconciliation checks."""
    POSITIONS = "positions"
    CASH = "cash"
    PNL = "pnl"
    TRADES = "trades"
    CORPORATE_ACTIONS = "corporate_actions"


@dataclass
class Position:
    """Position record for reconciliation."""
    symbol: str
    quantity: int
    market_value: float
    avg_cost: float
    unrealized_pnl: float
    account_id: str
    as_of_date: datetime


@dataclass
class ReconciliationItem:
    """Individual reconciliation discrepancy."""
    item_type: ReconciliationType
    symbol: Optional[str]
    internal_value: float
    broker_value: float
    difference: float
    difference_pct: float
    tolerance_exceeded: bool
    description: str


@dataclass
class ReconciliationResult:
    """Results of daily reconciliation."""
    date: datetime
    account_id: str
    passed: bool
    total_discrepancies: int
    critical_discrepancies: int
    total_difference_abs: float
    items: List[ReconciliationItem]
    summary: Dict[str, Any]


class BrokerInterface:
    """Interface for broker API calls."""

    def __init__(self, broker_name: str, api_config: Dict[str, Any]):
        self.broker_name = broker_name
        self.api_config = api_config
        self.connected = False

    async def connect(self) -> bool:
        """Connect to broker API."""
        try:
            # In production, this would establish actual API connection
            logger.info(f"Connecting to {self.broker_name}")
            self.connected = True
            return True
        except Exception as e:
            logger.error(f"Failed to connect to {self.broker_name}: {e}")
            return False

    async def get_positions(self, account_id: str) -> List[Position]:
        """Get current positions from broker."""
        if not self.connected:
            raise RuntimeError("Not connected to broker")

        # In production, this would make actual API calls
        # For simulation, return sample data
        sample_positions = [
            Position(
                symbol="SPY",
                quantity=100,
                market_value=45000.0,
                avg_cost=440.0,
                unrealized_pnl=1000.0,
                account_id=account_id,
                as_of_date=datetime.now()
            ),
            Position(
                symbol="QQQ",
                quantity=50,
                market_value=17500.0,
                avg_cost=340.0,
                unrealized_pnl=500.0,
                account_id=account_id,
                as_of_date=datetime.now()
            )
        ]

        logger.info(f"Retrieved {len(sample_positions)} positions from {self.broker_name}")
        return sample_positions

    async def get_cash_balance(self, account_id: str) -> Dict[str, float]:
        """Get cash balances from broker."""
        if not self.connected:
            raise RuntimeError("Not connected to broker")

        # Sample cash balance data
        return {
            'total_cash': 25000.0,
            'buying_power': 50000.0,
            'settled_cash': 24000.0,
            'unsettled_cash': 1000.0
        }

    async def get_account_summary(self, account_id: str) -> Dict[str, float]:
        """Get account summary from broker."""
        if not self.connected:
            raise RuntimeError("Not connected to broker")

        return {
            'net_liquidation_value': 87500.0,
            'total_cash_value': 25000.0,
            'stock_market_value': 62500.0,
            'unrealized_pnl': 1500.0,
            'realized_pnl': 2500.0
        }

    async def get_trade_history(self, account_id: str, date: datetime) -> List[Dict[str, Any]]:
        """Get trade history for a specific date."""
        if not self.connected:
            raise RuntimeError("Not connected to broker")

        # Sample trade data
        return [
            {
                'symbol': 'SPY',
                'side': 'buy',
                'quantity': 10,
                'price': 450.0,
                'timestamp': date.replace(hour=10, minute=30),
                'commission': 0.5,
                'trade_id': 'T123456'
            }
        ]


class InternalLedger:
    """Internal position and PnL tracking system."""

    def __init__(self):
        self.positions = {}  # symbol -> Position
        self.cash_balance = 0.0
        self.trade_history = []
        self.pnl_history = []

    def get_positions(self, account_id: str) -> List[Position]:
        """Get current positions from internal ledger."""
        return list(self.positions.values())

    def get_cash_balance(self, account_id: str) -> Dict[str, float]:
        """Get cash balance from internal ledger."""
        return {
            'total_cash': self.cash_balance,
            'buying_power': self.cash_balance * 2,  # 2:1 margin
            'settled_cash': self.cash_balance * 0.95,
            'unsettled_cash': self.cash_balance * 0.05
        }

    def get_account_summary(self, account_id: str) -> Dict[str, float]:
        """Get account summary from internal ledger."""
        total_stock_value = sum(pos.market_value for pos in self.positions.values())
        total_unrealized = sum(pos.unrealized_pnl for pos in self.positions.values())

        return {
            'net_liquidation_value': self.cash_balance + total_stock_value,
            'total_cash_value': self.cash_balance,
            'stock_market_value': total_stock_value,
            'unrealized_pnl': total_unrealized,
            'realized_pnl': 2400.0  # Sample value
        }

    def update_position(self, symbol: str, quantity: int, price: float):
        """Update position in internal ledger."""
        if symbol not in self.positions:
            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=0,
                market_value=0.0,
                avg_cost=0.0,
                unrealized_pnl=0.0,
                account_id="internal",
                as_of_date=datetime.now()
            )

        position = self.positions[symbol]
        old_quantity = position.quantity

        # Update quantity
        position.quantity += quantity

        # Update average cost (simplified)
        if position.quantity != 0:
            total_cost = (old_quantity * position.avg_cost) + (quantity * price)
            position.avg_cost = total_cost / position.quantity

        # Update market value and unrealized PnL
        current_price = price  # In production, would get current market price
        position.market_value = position.quantity * current_price
        position.unrealized_pnl = (current_price - position.avg_cost) * position.quantity
        position.as_of_date = datetime.now()

    def initialize_sample_data(self):
        """Initialize with sample data for testing."""
        self.cash_balance = 24500.0  # Slightly different from broker

        # Add sample positions (slightly different from broker)
        self.update_position("SPY", 100, 441.0)  # Different avg cost
        self.update_position("QQQ", 50, 339.0)   # Different avg cost


class DailyReconciler:
    """Daily reconciliation engine."""

    def __init__(self, broker_interface: BrokerInterface, internal_ledger: InternalLedger):
        self.broker = broker_interface
        self.ledger = internal_ledger
        self.tolerances = {
            'position_quantity': 0,  # Zero tolerance for position quantity
            'position_value_pct': 0.001,  # 0.1% tolerance for position values
            'cash_pct': 0.002,  # 0.2% tolerance for cash
            'pnl_pct': 0.005,  # 0.5% tolerance for PnL
            'total_value_pct': 0.001  # 0.1% tolerance for total account value
        }

    async def run_daily_reconciliation(self, account_id: str) -> ReconciliationResult:
        """Run complete daily reconciliation."""
        logger.info(f"Starting daily reconciliation for account {account_id}")

        reconciliation_items = []

        try:
            # Ensure broker connection
            if not self.broker.connected:
                await self.broker.connect()

            # Reconcile positions
            position_items = await self._reconcile_positions(account_id)
            reconciliation_items.extend(position_items)

            # Reconcile cash
            cash_items = await self._reconcile_cash(account_id)
            reconciliation_items.extend(cash_items)

            # Reconcile account summary
            summary_items = await self._reconcile_account_summary(account_id)
            reconciliation_items.extend(summary_items)

            # Check tolerance violations
            critical_discrepancies = [item for item in reconciliation_items if item.tolerance_exceeded]

            # Determine overall pass/fail
            passed = len(critical_discrepancies) == 0

            # Create summary
            summary = self._create_summary(reconciliation_items)

            result = ReconciliationResult(
                date=datetime.now(),
                account_id=account_id,
                passed=passed,
                total_discrepancies=len(reconciliation_items),
                critical_discrepancies=len(critical_discrepancies),
                total_difference_abs=sum(abs(item.difference) for item in reconciliation_items),
                items=reconciliation_items,
                summary=summary
            )

            logger.info(f"Reconciliation completed: {result.passed}, "
                       f"{result.total_discrepancies} discrepancies, "
                       f"{result.critical_discrepancies} critical")

            return result

        except Exception as e:
            logger.error(f"Reconciliation failed: {e}")
            # Return failed result
            return ReconciliationResult(
                date=datetime.now(),
                account_id=account_id,
                passed=False,
                total_discrepancies=0,
                critical_discrepancies=1,
                total_difference_abs=0,
                items=[],
                summary={'error': str(e)}
            )

    async def _reconcile_positions(self, account_id: str) -> List[ReconciliationItem]:
        """Reconcile position holdings."""
        items = []

        # Get positions from both sources
        broker_positions = await self.broker.get_positions(account_id)
        internal_positions = self.ledger.get_positions(account_id)

        # Create lookup dictionaries
        broker_pos_dict = {pos.symbol: pos for pos in broker_positions}
        internal_pos_dict = {pos.symbol: pos for pos in internal_positions}

        # Get all symbols
        all_symbols = set(broker_pos_dict.keys()) | set(internal_pos_dict.keys())

        for symbol in all_symbols:
            broker_pos = broker_pos_dict.get(symbol)
            internal_pos = internal_pos_dict.get(symbol)

            # Check quantity differences
            broker_qty = broker_pos.quantity if broker_pos else 0
            internal_qty = internal_pos.quantity if internal_pos else 0

            if broker_qty != internal_qty:
                items.append(ReconciliationItem(
                    item_type=ReconciliationType.POSITIONS,
                    symbol=symbol,
                    internal_value=internal_qty,
                    broker_value=broker_qty,
                    difference=internal_qty - broker_qty,
                    difference_pct=0,  # Quantity differences are absolute
                    tolerance_exceeded=abs(internal_qty - broker_qty) > self.tolerances['position_quantity'],
                    description=f"Position quantity mismatch for {symbol}"
                ))

            # Check market value differences (if both positions exist)
            if broker_pos and internal_pos:
                broker_value = broker_pos.market_value
                internal_value = internal_pos.market_value

                if broker_value != 0:
                    value_diff_pct = abs(internal_value - broker_value) / abs(broker_value)
                else:
                    value_diff_pct = 1.0 if internal_value != 0 else 0.0

                if value_diff_pct > self.tolerances['position_value_pct']:
                    items.append(ReconciliationItem(
                        item_type=ReconciliationType.POSITIONS,
                        symbol=symbol,
                        internal_value=internal_value,
                        broker_value=broker_value,
                        difference=internal_value - broker_value,
                        difference_pct=value_diff_pct * 100,
                        tolerance_exceeded=True,
                        description=f"Position value mismatch for {symbol}"
                    ))

        return items

    async def _reconcile_cash(self, account_id: str) -> List[ReconciliationItem]:
        """Reconcile cash balances."""
        items = []

        broker_cash = await self.broker.get_cash_balance(account_id)
        internal_cash = self.ledger.get_cash_balance(account_id)

        # Compare total cash
        broker_total = broker_cash['total_cash']
        internal_total = internal_cash['total_cash']

        if broker_total != 0:
            cash_diff_pct = abs(internal_total - broker_total) / abs(broker_total)
        else:
            cash_diff_pct = 1.0 if internal_total != 0 else 0.0

        if cash_diff_pct > self.tolerances['cash_pct']:
            items.append(ReconciliationItem(
                item_type=ReconciliationType.CASH,
                symbol=None,
                internal_value=internal_total,
                broker_value=broker_total,
                difference=internal_total - broker_total,
                difference_pct=cash_diff_pct * 100,
                tolerance_exceeded=True,
                description="Total cash balance mismatch"
            ))

        # Compare buying power
        broker_bp = broker_cash.get('buying_power', 0)
        internal_bp = internal_cash.get('buying_power', 0)

        if broker_bp != 0:
            bp_diff_pct = abs(internal_bp - broker_bp) / abs(broker_bp)
            if bp_diff_pct > self.tolerances['cash_pct'] * 2:  # More lenient for buying power
                items.append(ReconciliationItem(
                    item_type=ReconciliationType.CASH,
                    symbol=None,
                    internal_value=internal_bp,
                    broker_value=broker_bp,
                    difference=internal_bp - broker_bp,
                    difference_pct=bp_diff_pct * 100,
                    tolerance_exceeded=True,
                    description="Buying power mismatch"
                ))

        return items

    async def _reconcile_account_summary(self, account_id: str) -> List[ReconciliationItem]:
        """Reconcile account-level summary values."""
        items = []

        broker_summary = await self.broker.get_account_summary(account_id)
        internal_summary = self.ledger.get_account_summary(account_id)

        # Key metrics to reconcile
        metrics_to_check = [
            ('net_liquidation_value', 'total_value_pct', "Net liquidation value mismatch"),
            ('stock_market_value', 'position_value_pct', "Total stock value mismatch"),
            ('unrealized_pnl', 'pnl_pct', "Unrealized PnL mismatch"),
            ('realized_pnl', 'pnl_pct', "Realized PnL mismatch")
        ]

        for metric, tolerance_key, description in metrics_to_check:
            broker_value = broker_summary.get(metric, 0)
            internal_value = internal_summary.get(metric, 0)

            if broker_value != 0:
                diff_pct = abs(internal_value - broker_value) / abs(broker_value)
            else:
                diff_pct = 1.0 if internal_value != 0 else 0.0

            if diff_pct > self.tolerances[tolerance_key]:
                items.append(ReconciliationItem(
                    item_type=ReconciliationType.PNL,
                    symbol=None,
                    internal_value=internal_value,
                    broker_value=broker_value,
                    difference=internal_value - broker_value,
                    difference_pct=diff_pct * 100,
                    tolerance_exceeded=True,
                    description=description
                ))

        return items

    def _create_summary(self, items: List[ReconciliationItem]) -> Dict[str, Any]:
        """Create reconciliation summary."""
        summary = {
            'total_items': len(items),
            'by_type': {},
            'critical_items': [],
            'largest_differences': []
        }

        # Group by type
        for item in items:
            item_type = item.item_type.value
            if item_type not in summary['by_type']:
                summary['by_type'][item_type] = {
                    'count': 0,
                    'total_difference': 0,
                    'critical_count': 0
                }

            summary['by_type'][item_type]['count'] += 1
            summary['by_type'][item_type]['total_difference'] += abs(item.difference)
            if item.tolerance_exceeded:
                summary['by_type'][item_type]['critical_count'] += 1

        # Critical items
        summary['critical_items'] = [
            {
                'type': item.item_type.value,
                'symbol': item.symbol,
                'difference': item.difference,
                'difference_pct': item.difference_pct,
                'description': item.description
            }
            for item in items if item.tolerance_exceeded
        ]

        # Largest differences
        sorted_items = sorted(items, key=lambda x: abs(x.difference), reverse=True)
        summary['largest_differences'] = [
            {
                'type': item.item_type.value,
                'symbol': item.symbol,
                'difference': item.difference,
                'description': item.description
            }
            for item in sorted_items[:5]
        ]

        return summary

    def update_tolerances(self, new_tolerances: Dict[str, float]):
        """Update reconciliation tolerances."""
        self.tolerances.update(new_tolerances)
        logger.info(f"Updated reconciliation tolerances: {self.tolerances}")


class ReconciliationReporter:
    """Generates reconciliation reports and alerts."""

    def __init__(self):
        self.report_history = []

    def generate_report(self, result: ReconciliationResult) -> str:
        """Generate human-readable reconciliation report."""
        report = f"""
=== DAILY RECONCILIATION REPORT ===
Date: {result.date.strftime('%Y-%m-%d %H:%M:%S')}
Account: {result.account_id}
Status: {'PASSED' if result.passed else 'FAILED'}

SUMMARY:
- Total Discrepancies: {result.total_discrepancies}
- Critical Discrepancies: {result.critical_discrepancies}
- Total Absolute Difference: ${result.total_difference_abs:,.2f}

"""

        if result.critical_discrepancies > 0:
            report += "🚨 CRITICAL DISCREPANCIES:\n"
            for item in result.items:
                if item.tolerance_exceeded:
                    symbol_str = f" ({item.symbol})" if item.symbol else ""
                    report += f"- {item.item_type.value.upper()}{symbol_str}: "
                    report += f"Internal=${item.internal_value:,.2f}, "
                    report += f"Broker=${item.broker_value:,.2f}, "
                    report += f"Diff=${item.difference:,.2f}"
                    if item.difference_pct > 0:
                        report += f" ({item.difference_pct:.2f}%)"
                    report += f"\n  {item.description}\n"

        if result.summary and 'by_type' in result.summary:
            report += "\nBREAKDOWN BY TYPE:\n"
            for item_type, type_summary in result.summary['by_type'].items():
                report += f"- {item_type.upper()}: {type_summary['count']} items, "
                report += f"{type_summary['critical_count']} critical, "
                report += f"${type_summary['total_difference']:,.2f} total difference\n"

        if not result.passed:
            report += "\n⚠️  RECOMMENDED ACTIONS:\n"
            report += "1. Halt automated trading until discrepancies resolved\n"
            report += "2. Review trade execution logs for missing/duplicate trades\n"
            report += "3. Check for pending corporate actions or settlements\n"
            report += "4. Contact broker support if discrepancies persist\n"

        return report

    def should_halt_trading(self, result: ReconciliationResult) -> bool:
        """Determine if trading should be halted based on reconciliation results."""
        if not result.passed:
            return True

        # Additional halt conditions
        if result.total_difference_abs > 10000:  # $10k total difference
            return True

        # Check for position quantity mismatches (always critical)
        for item in result.items:
            if (item.item_type == ReconciliationType.POSITIONS and
                item.symbol and
                abs(item.difference) > 0 and
                item.description.lower().contains('quantity')):
                return True

        return False

    def create_alert(self, result: ReconciliationResult) -> Dict[str, Any]:
        """Create alert for failed reconciliation."""
        severity = "critical" if self.should_halt_trading(result) else "warning"

        return {
            'timestamp': result.date,
            'severity': severity,
            'title': f"Reconciliation {'FAILED' if not result.passed else 'WARNING'}",
            'message': f"Account {result.account_id} has {result.critical_discrepancies} critical discrepancies",
            'should_halt_trading': self.should_halt_trading(result),
            'account_id': result.account_id,
            'discrepancy_count': result.critical_discrepancies,
            'total_difference': result.total_difference_abs
        }

    def store_result(self, result: ReconciliationResult):
        """Store reconciliation result for historical analysis."""
        self.report_history.append(result)

        # Keep only last 30 days of results
        cutoff_date = datetime.now() - timedelta(days=30)
        self.report_history = [
            r for r in self.report_history
            if r.date >= cutoff_date
        ]

    def get_reconciliation_trends(self, days: int = 7) -> Dict[str, Any]:
        """Get reconciliation trends over specified period."""
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_results = [
            r for r in self.report_history
            if r.date >= cutoff_date
        ]

        if not recent_results:
            return {'error': 'No recent reconciliation data'}

        trends = {
            'period_days': days,
            'total_reconciliations': len(recent_results),
            'passed_rate': sum(1 for r in recent_results if r.passed) / len(recent_results),
            'avg_discrepancies': np.mean([r.total_discrepancies for r in recent_results]),
            'avg_critical_discrepancies': np.mean([r.critical_discrepancies for r in recent_results]),
            'avg_difference_abs': np.mean([r.total_difference_abs for r in recent_results]),
            'trend_direction': 'improving'  # Would calculate actual trend
        }

        return trends