"""End-of-day broker reconciliation.

This module reconciles local orders with broker fills and positions,
detecting breaks that require manual intervention.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Dict, List, Optional
import logging

log = logging.getLogger("wsb.eod_recon")


@dataclass
class LocalOrder:
    """Local order record for reconciliation."""
    client_order_id: str
    broker_order_id: Optional[str]
    symbol: str
    side: str
    qty: float
    final_state: Optional[str]


@dataclass
class BrokerFill:
    """Broker fill record for reconciliation."""
    broker_order_id: str
    symbol: str
    filled_qty: float
    avg_price: float
    status: str


@dataclass
class ReconciliationBreaks:
    """Container for reconciliation breaks."""
    missing_fill: List[str]
    unknown_broker_fill: List[str]
    state_mismatch: List[str]
    qty_mismatch: List[str]
    price_mismatch: List[str]


def reconcile(
    local_orders: Iterable[LocalOrder], 
    broker_fills: Iterable[BrokerFill]
) -> ReconciliationBreaks:
    """Reconcile local orders with broker fills.
    
    Args:
        local_orders: Local order records
        broker_fills: Broker fill records
        
    Returns:
        ReconciliationBreaks with detected issues
    """
    breaks = ReconciliationBreaks(
        missing_fill=[],
        unknown_broker_fill=[],
        state_mismatch=[],
        qty_mismatch=[],
        price_mismatch=[]
    )
    
    fills_by_id = {f.broker_order_id: f for f in broker_fills}
    local_broker_ids = set()
    
    # Check each local order
    for lo in local_orders:
        local_broker_ids.add(lo.broker_order_id)
        
        if not lo.broker_order_id:
            breaks.missing_fill.append(lo.client_order_id)
            continue
            
        bf = fills_by_id.get(lo.broker_order_id)
        if not bf:
            breaks.missing_fill.append(lo.client_order_id)
            continue
            
        # Check symbol match
        if lo.symbol != bf.symbol:
            breaks.state_mismatch.append(f"{lo.client_order_id}: symbol mismatch {lo.symbol} vs {bf.symbol}")
            
        # Check quantity match
        if abs(lo.qty - bf.filled_qty) > 0.01:  # Allow small rounding differences
            breaks.qty_mismatch.append(f"{lo.client_order_id}: qty mismatch {lo.qty} vs {bf.filled_qty}")
            
        # Check status alignment
        if lo.final_state and lo.final_state.lower() not in (
            bf.status.lower(), 
            "filled" if bf.filled_qty == lo.qty else "partially_filled"
        ):
            breaks.state_mismatch.append(f"{lo.client_order_id}: status mismatch {lo.final_state} vs {bf.status}")
    
    # Check for unknown broker fills
    for bid in fills_by_id.keys() - local_broker_ids:
        if bid:  # Skip None broker_order_ids
            breaks.unknown_broker_fill.append(bid)
    
    return breaks


def log_reconciliation_results(breaks: ReconciliationBreaks) -> None:
    """Log reconciliation results.
    
    Args:
        breaks: Reconciliation breaks to log
    """
    total_breaks = (
        len(breaks.missing_fill) + 
        len(breaks.unknown_broker_fill) + 
        len(breaks.state_mismatch) + 
        len(breaks.qty_mismatch) + 
        len(breaks.price_mismatch)
    )
    
    if total_breaks == 0:
        log.info("EOD reconciliation: No breaks detected")
        return
    
    log.warning(f"EOD reconciliation: {total_breaks} breaks detected")
    
    if breaks.missing_fill:
        log.warning(f"Missing fills: {breaks.missing_fill}")
    if breaks.unknown_broker_fill:
        log.warning(f"Unknown broker fills: {breaks.unknown_broker_fill}")
    if breaks.state_mismatch:
        log.warning(f"State mismatches: {breaks.state_mismatch}")
    if breaks.qty_mismatch:
        log.warning(f"Quantity mismatches: {breaks.qty_mismatch}")
    if breaks.price_mismatch:
        log.warning(f"Price mismatches: {breaks.price_mismatch}")


def should_disable_next_day(breaks: ReconciliationBreaks) -> bool:
    """Determine if next trading day should be disabled.
    
    Args:
        breaks: Reconciliation breaks
        
    Returns:
        True if next day should be disabled
    """
    # Disable if critical breaks exist
    critical_breaks = (
        len(breaks.missing_fill) > 0 or
        len(breaks.unknown_broker_fill) > 0 or
        len(breaks.state_mismatch) > 0
    )
    
    if critical_breaks:
        log.critical("CRITICAL: Next trading day should be DISABLED due to reconciliation breaks")
        return True
    
    return False


class EODReconciler:
    """End-of-day reconciliation manager."""
    
    def __init__(self, breaks_file: str = "./.state/eod_breaks.json"):
        """Initialize EOD reconciler.
        
        Args:
            breaks_file: Path to store breaks for next day check
        """
        self.breaks_file = breaks_file
        
    def run_daily_reconciliation(
        self, 
        local_orders: Iterable[LocalOrder], 
        broker_fills: Iterable[BrokerFill]
    ) -> ReconciliationBreaks:
        """Run daily reconciliation.
        
        Args:
            local_orders: Local order records
            broker_fills: Broker fill records
            
        Returns:
            Reconciliation breaks
        """
        breaks = reconcile(local_orders, broker_fills)
        log_reconciliation_results(breaks)
        
        # Store breaks for next day check
        self._store_breaks(breaks)
        
        return breaks
    
    def _store_breaks(self, breaks: ReconciliationBreaks) -> None:
        """Store breaks for next day check."""
        import json
        import pathlib
        
        breaks_data = {
            "missing_fill": breaks.missing_fill,
            "unknown_broker_fill": breaks.unknown_broker_fill,
            "state_mismatch": breaks.state_mismatch,
            "qty_mismatch": breaks.qty_mismatch,
            "price_mismatch": breaks.price_mismatch,
        }
        
        p = pathlib.Path(self.breaks_file)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(breaks_data, indent=2))
    
    def check_pending_breaks(self) -> bool:
        """Check if there are pending breaks from previous day.
        
        Returns:
            True if there are pending breaks
        """
        import json
        import pathlib
        
        p = pathlib.Path(self.breaks_file)
        if not p.exists():
            return False
            
        try:
            breaks_data = json.loads(p.read_text())
            total_breaks = sum(len(v) for v in breaks_data.values())
            return total_breaks > 0
        except Exception as e:
            log.error(f"Failed to check pending breaks: {e}")
            return True  # Err on side of caution

