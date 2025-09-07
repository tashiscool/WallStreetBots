"""
Position Reconciliation System
Validates consistency between internal position tracking and broker positions
"""

import asyncio
import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json

logger = logging.getLogger(__name__)


class DiscrepancyType(Enum):
    """Types of position discrepancies"""
    MISSING_AT_BROKER = "missing_at_broker"
    MISSING_IN_DB = "missing_in_db"
    QUANTITY_MISMATCH = "quantity_mismatch"
    PRICE_MISMATCH = "price_mismatch"
    STATUS_MISMATCH = "status_mismatch"
    STALE_POSITION = "stale_position"


class DiscrepancySeverity(Enum):
    """Severity levels for discrepancies"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class PositionSnapshot:
    """Position data from internal database"""
    id: str
    ticker: str
    quantity: int
    avg_cost: Decimal
    current_price: Decimal
    market_value: Decimal
    unrealized_pnl: Decimal
    status: str  # 'open', 'closed', 'pending'
    last_updated: datetime
    position_type: str  # 'stock', 'option'
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BrokerPosition:
    """Position data from broker API"""
    symbol: str
    quantity: int
    avg_cost: Decimal
    current_price: Decimal
    market_value: Decimal
    unrealized_pnl: Decimal
    side: str  # 'long', 'short'
    asset_class: str  # 'us_equity', 'option', etc.
    last_updated: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PositionDiscrepancy:
    """Represents a position discrepancy"""
    discrepancy_type: DiscrepancyType
    severity: DiscrepancySeverity
    ticker: str
    db_position: Optional[PositionSnapshot]
    broker_position: Optional[BrokerPosition]
    description: str
    suggested_action: str
    financial_impact: Decimal
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class ReconciliationReport:
    """Complete reconciliation report"""
    timestamp: datetime
    total_positions_db: int
    total_positions_broker: int
    discrepancies: List[PositionDiscrepancy]
    total_discrepancies: int
    critical_discrepancies: int
    high_priority_discrepancies: int
    total_financial_impact: Decimal
    reconciliation_status: str  # 'CLEAN', 'WARNINGS', 'CRITICAL'
    next_reconciliation: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_clean(self) -> bool:
        """Check if reconciliation is clean"""
        return self.total_discrepancies == 0
    
    @property
    def requires_intervention(self) -> bool:
        """Check if manual intervention is required"""
        return self.critical_discrepancies > 0


class PositionReconciler:
    """
    Position Reconciliation System
    
    Performs comprehensive validation between internal position tracking
    and broker positions to ensure data consistency and detect potential issues.
    
    Features:
    - Real-time position comparison
    - Automatic discrepancy detection and classification
    - Financial impact assessment
    - Suggested remediation actions
    - Emergency halt capabilities
    - Historical reconciliation tracking
    """
    
    def __init__(self, alpaca_manager=None, database_manager=None):
        self.alpaca_manager = alpaca_manager
        self.database_manager = database_manager
        self.logger = logging.getLogger(__name__)
        
        # Reconciliation settings
        self.quantity_tolerance = 0  # Zero tolerance for quantity differences
        self.price_tolerance = Decimal('0.01')  # $0.01 tolerance for price differences
        self.value_tolerance = Decimal('1.00')  # $1.00 tolerance for market value
        self.stale_position_hours = 24  # Consider positions stale after 24 hours
        
        # Emergency thresholds
        self.max_critical_discrepancies = 1  # Halt if any critical discrepancy
        self.max_total_financial_impact = Decimal('1000')  # Halt if impact > $1000
        self.max_missing_positions = 3  # Halt if more than 3 positions missing
        
        # Reconciliation history
        self.reconciliation_history: List[ReconciliationReport] = []
        self.last_reconciliation: Optional[datetime] = None
        self.consecutive_failures = 0
        
        # State tracking
        self.is_emergency_halted = False
        self.halt_reason = None
        
        self.logger.info("PositionReconciler initialized")
    
    async def reconcile_all_positions(self, auto_halt: bool = True) -> ReconciliationReport:
        """
        Perform comprehensive position reconciliation
        
        Args:
            auto_halt: Whether to automatically halt trading on critical discrepancies
            
        Returns:
            ReconciliationReport with all discrepancies and recommendations
        """
        try:
            self.logger.info("Starting comprehensive position reconciliation")
            start_time = datetime.now()
            
            # Get positions from both sources
            db_positions = await self._get_database_positions()
            broker_positions = await self._get_broker_positions()
            
            self.logger.info(f"Retrieved {len(db_positions)} DB positions, {len(broker_positions)} broker positions")
            
            # Perform reconciliation analysis
            discrepancies = await self._analyze_positions(db_positions, broker_positions)
            
            # Calculate financial impact
            total_financial_impact = sum(d.financial_impact for d in discrepancies)
            
            # Classify severity
            critical_count = sum(1 for d in discrepancies if d.severity == DiscrepancySeverity.CRITICAL)
            high_count = sum(1 for d in discrepancies if d.severity == DiscrepancySeverity.HIGH)
            
            # Determine overall status
            if critical_count > 0:
                status = "CRITICAL"
            elif high_count > 0:
                status = "WARNINGS" 
            else:
                status = "CLEAN"
            
            # Create reconciliation report
            report = ReconciliationReport(
                timestamp=start_time,
                total_positions_db=len(db_positions),
                total_positions_broker=len(broker_positions),
                discrepancies=discrepancies,
                total_discrepancies=len(discrepancies),
                critical_discrepancies=critical_count,
                high_priority_discrepancies=high_count,
                total_financial_impact=total_financial_impact,
                reconciliation_status=status,
                next_reconciliation=start_time + timedelta(minutes=15),  # Schedule next reconciliation
                metadata={
                    'reconciliation_duration': (datetime.now() - start_time).total_seconds(),
                    'auto_halt_enabled': auto_halt
                }
            )
            
            # Store in history
            self.reconciliation_history.append(report)
            self.last_reconciliation = start_time
            
            # Keep only last 100 reconciliation reports
            if len(self.reconciliation_history) > 100:
                self.reconciliation_history = self.reconciliation_history[-100:]
            
            # Log results
            self._log_reconciliation_results(report)
            
            # Check if emergency halt is needed
            if auto_halt and self._should_emergency_halt(report):
                await self._trigger_emergency_halt(report)
            
            self.consecutive_failures = 0  # Reset failure count on success
            
            return report
            
        except Exception as e:
            self.consecutive_failures += 1
            self.logger.error(f"Position reconciliation failed: {e}")
            
            # Create error report
            error_report = ReconciliationReport(
                timestamp=datetime.now(),
                total_positions_db=0,
                total_positions_broker=0,
                discrepancies=[],
                total_discrepancies=0,
                critical_discrepancies=0,
                high_priority_discrepancies=0,
                total_financial_impact=Decimal('0'),
                reconciliation_status="ERROR",
                next_reconciliation=datetime.now() + timedelta(minutes=30),
                metadata={'error': str(e), 'consecutive_failures': self.consecutive_failures}
            )
            
            # Emergency halt on repeated failures
            if auto_halt and self.consecutive_failures >= 3:
                await self._trigger_emergency_halt(error_report, f"Reconciliation failed {self.consecutive_failures} times: {e}")
            
            return error_report
    
    async def _get_database_positions(self) -> List[PositionSnapshot]:
        """Get positions from internal database"""
        try:
            # This would typically query the database for open positions
            # For now, return mock data structure
            
            if not self.database_manager:
                self.logger.warning("Database manager not configured, returning empty positions")
                return []
            
            # Mock implementation - replace with actual database query
            positions = []
            
            # In a real implementation, this would be something like:
            # positions_data = await self.database_manager.get_open_positions()
            # for pos_data in positions_data:
            #     positions.append(PositionSnapshot(...))
            
            return positions
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve database positions: {e}")
            raise
    
    async def _get_broker_positions(self) -> List[BrokerPosition]:
        """Get positions from broker API"""
        try:
            if not self.alpaca_manager:
                self.logger.warning("Alpaca manager not configured, returning empty positions")
                return []
            
            # Get positions from Alpaca
            alpaca_positions = await self.alpaca_manager.get_positions()
            broker_positions = []
            
            for pos in alpaca_positions:
                broker_position = BrokerPosition(
                    symbol=pos.symbol,
                    quantity=int(pos.qty),
                    avg_cost=Decimal(str(pos.avg_entry_price)),
                    current_price=Decimal(str(pos.current_price)),
                    market_value=Decimal(str(pos.market_value)),
                    unrealized_pnl=Decimal(str(pos.unrealized_pnl)),
                    side='long' if int(pos.qty) > 0 else 'short',
                    asset_class=pos.asset_class,
                    last_updated=datetime.now(),
                    metadata={'position_id': pos.asset_id}
                )
                broker_positions.append(broker_position)
            
            return broker_positions
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve broker positions: {e}")
            raise
    
    async def _analyze_positions(self, db_positions: List[PositionSnapshot], 
                               broker_positions: List[BrokerPosition]) -> List[PositionDiscrepancy]:
        """Analyze positions for discrepancies"""
        discrepancies = []
        
        # Create lookup dictionaries
        db_by_ticker = {pos.ticker: pos for pos in db_positions}
        broker_by_symbol = {pos.symbol: pos for pos in broker_positions}
        
        # Check for positions in DB but not at broker
        for ticker, db_pos in db_by_ticker.items():
            if ticker not in broker_by_symbol:
                discrepancy = PositionDiscrepancy(
                    discrepancy_type=DiscrepancyType.MISSING_AT_BROKER,
                    severity=DiscrepancySeverity.CRITICAL,
                    ticker=ticker,
                    db_position=db_pos,
                    broker_position=None,
                    description=f"Position {ticker} exists in database but not at broker",
                    suggested_action="Investigate database record, may need to close position manually",
                    financial_impact=abs(db_pos.market_value),
                    timestamp=datetime.now(),
                    metadata={'db_quantity': db_pos.quantity, 'db_value': float(db_pos.market_value)}
                )\n                discrepancies.append(discrepancy)\n        
        # Check for positions at broker but not in DB
        for symbol, broker_pos in broker_by_symbol.items():
            if symbol not in db_by_ticker:
                discrepancy = PositionDiscrepancy(
                    discrepancy_type=DiscrepancyType.MISSING_IN_DB,
                    severity=DiscrepancySeverity.HIGH,
                    ticker=symbol,
                    db_position=None,
                    broker_position=broker_pos,
                    description=f"Position {symbol} exists at broker but not in database",
                    suggested_action="Add position to database or investigate unauthorized trade",
                    financial_impact=abs(broker_pos.market_value),
                    timestamp=datetime.now(),
                    metadata={'broker_quantity': broker_pos.quantity, 'broker_value': float(broker_pos.market_value)}
                )
                discrepancies.append(discrepancy)
        
        # Check for quantity/value mismatches on matching positions
        for ticker in set(db_by_ticker.keys()) & set(broker_by_symbol.keys()):
            db_pos = db_by_ticker[ticker]
            broker_pos = broker_by_symbol[ticker]
            
            # Quantity mismatch
            if abs(db_pos.quantity - broker_pos.quantity) > self.quantity_tolerance:
                severity = DiscrepancySeverity.CRITICAL if abs(db_pos.quantity - broker_pos.quantity) > 10 else DiscrepancySeverity.HIGH
                
                discrepancy = PositionDiscrepancy(
                    discrepancy_type=DiscrepancyType.QUANTITY_MISMATCH,
                    severity=severity,
                    ticker=ticker,
                    db_position=db_pos,
                    broker_position=broker_pos,
                    description=f"Quantity mismatch for {ticker}: DB={db_pos.quantity}, Broker={broker_pos.quantity}",
                    suggested_action="Investigate recent trades, update database or investigate execution issues",
                    financial_impact=abs((db_pos.quantity - broker_pos.quantity) * db_pos.current_price),
                    timestamp=datetime.now(),
                    metadata={
                        'quantity_difference': db_pos.quantity - broker_pos.quantity,
                        'db_quantity': db_pos.quantity,
                        'broker_quantity': broker_pos.quantity
                    }
                )
                discrepancies.append(discrepancy)
            
            # Price/value mismatch (if quantities match)
            elif abs(db_pos.market_value - broker_pos.market_value) > self.value_tolerance:
                discrepancy = PositionDiscrepancy(
                    discrepancy_type=DiscrepancyType.PRICE_MISMATCH,
                    severity=DiscrepancySeverity.MEDIUM,
                    ticker=ticker,
                    db_position=db_pos,
                    broker_position=broker_pos,
                    description=f"Market value mismatch for {ticker}: DB=${db_pos.market_value}, Broker=${broker_pos.market_value}",
                    suggested_action="Update current prices in database or investigate pricing feed",
                    financial_impact=abs(db_pos.market_value - broker_pos.market_value),
                    timestamp=datetime.now(),
                    metadata={
                        'value_difference': float(db_pos.market_value - broker_pos.market_value),
                        'db_value': float(db_pos.market_value),
                        'broker_value': float(broker_pos.market_value)
                    }
                )
                discrepancies.append(discrepancy)
            
            # Check for stale positions
            if datetime.now() - db_pos.last_updated > timedelta(hours=self.stale_position_hours):
                discrepancy = PositionDiscrepancy(
                    discrepancy_type=DiscrepancyType.STALE_POSITION,
                    severity=DiscrepancySeverity.LOW,
                    ticker=ticker,
                    db_position=db_pos,
                    broker_position=broker_pos,
                    description=f"Stale position data for {ticker}: last updated {db_pos.last_updated}",
                    suggested_action="Refresh position data from market feeds",
                    financial_impact=Decimal('0'),
                    timestamp=datetime.now(),
                    metadata={'hours_stale': (datetime.now() - db_pos.last_updated).total_seconds() / 3600}
                )
                discrepancies.append(discrepancy)
        
        return discrepancies
    
    def _should_emergency_halt(self, report: ReconciliationReport) -> bool:
        """Determine if emergency halt should be triggered"""
        # Critical discrepancies
        if report.critical_discrepancies >= self.max_critical_discrepancies:
            return True
        
        # Too many missing positions
        missing_positions = sum(1 for d in report.discrepancies 
                               if d.discrepancy_type in [DiscrepancyType.MISSING_AT_BROKER, DiscrepancyType.MISSING_IN_DB])
        if missing_positions >= self.max_missing_positions:
            return True
        
        # Financial impact too high
        if report.total_financial_impact >= self.max_total_financial_impact:
            return True
        
        return False
    
    async def _trigger_emergency_halt(self, report: ReconciliationReport, custom_reason: str = None):
        """Trigger emergency halt of trading system"""
        try:
            halt_reason = custom_reason or f"Position reconciliation found {report.critical_discrepancies} critical discrepancies"
            
            self.is_emergency_halted = True
            self.halt_reason = halt_reason
            
            self.logger.critical(f"EMERGENCY HALT TRIGGERED: {halt_reason}")
            
            # Log detailed halt information
            halt_details = {
                'timestamp': datetime.now().isoformat(),
                'reason': halt_reason,
                'total_discrepancies': report.total_discrepancies,
                'critical_discrepancies': report.critical_discrepancies,
                'financial_impact': float(report.total_financial_impact),
                'discrepancy_summary': [
                    {
                        'type': d.discrepancy_type.value,
                        'ticker': d.ticker,
                        'severity': d.severity.value,
                        'impact': float(d.financial_impact)
                    } for d in report.discrepancies if d.severity == DiscrepancySeverity.CRITICAL
                ]
            }
            
            self.logger.critical(f"Emergency halt details: {json.dumps(halt_details, indent=2)}")
            
            # In a real implementation, this would:
            # 1. Cancel all pending orders
            # 2. Send alerts to administrators
            # 3. Set trading system status to halted
            # 4. Notify risk management systems
            # 5. Log to audit trail
            
        except Exception as e:
            self.logger.error(f"Failed to trigger emergency halt: {e}")
    
    def _log_reconciliation_results(self, report: ReconciliationReport):
        """Log reconciliation results"""
        if report.is_clean:
            self.logger.info(f"Position reconciliation CLEAN: {report.total_positions_db} positions verified")
        else:
            self.logger.warning(f"Position reconciliation found {report.total_discrepancies} discrepancies "
                              f"({report.critical_discrepancies} critical, {report.high_priority_discrepancies} high)")
            
            # Log each critical discrepancy
            for discrepancy in report.discrepancies:
                if discrepancy.severity == DiscrepancySeverity.CRITICAL:
                    self.logger.error(f"CRITICAL DISCREPANCY: {discrepancy.description} "
                                    f"(Impact: ${discrepancy.financial_impact})")
    
    async def get_reconciliation_summary(self) -> Dict[str, Any]:
        """Get summary of reconciliation status"""
        try:
            if not self.reconciliation_history:
                return {
                    'status': 'NO_DATA',
                    'message': 'No reconciliation performed yet',
                    'last_reconciliation': None
                }
            
            latest_report = self.reconciliation_history[-1]
            
            return {
                'status': latest_report.reconciliation_status,
                'last_reconciliation': latest_report.timestamp.isoformat(),
                'total_positions': latest_report.total_positions_broker,
                'discrepancies': latest_report.total_discrepancies,
                'critical_discrepancies': latest_report.critical_discrepancies,
                'financial_impact': float(latest_report.total_financial_impact),
                'is_emergency_halted': self.is_emergency_halted,
                'halt_reason': self.halt_reason,
                'consecutive_failures': self.consecutive_failures,
                'next_reconciliation': latest_report.next_reconciliation.isoformat() if latest_report.next_reconciliation else None
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get reconciliation summary: {e}")
            return {'status': 'ERROR', 'error': str(e)}
    
    async def force_reconciliation(self) -> ReconciliationReport:
        """Force immediate reconciliation (bypassing schedule)"""
        self.logger.info("Forcing immediate position reconciliation")
        return await self.reconcile_all_positions(auto_halt=True)
    
    async def clear_emergency_halt(self, reason: str = "Manual override"):
        """Clear emergency halt status"""
        if self.is_emergency_halted:
            self.is_emergency_halted = False
            old_reason = self.halt_reason
            self.halt_reason = None
            
            self.logger.info(f"Emergency halt cleared: {reason} (Previous reason: {old_reason})")
        else:
            self.logger.info("No emergency halt to clear")
    
    async def get_position_details(self, ticker: str) -> Dict[str, Any]:
        """Get detailed reconciliation info for specific position"""
        try:
            db_positions = await self._get_database_positions()
            broker_positions = await self._get_broker_positions()
            
            db_pos = next((p for p in db_positions if p.ticker == ticker), None)
            broker_pos = next((p for p in broker_positions if p.symbol == ticker), None)
            
            return {
                'ticker': ticker,
                'database_position': {
                    'exists': db_pos is not None,
                    'quantity': db_pos.quantity if db_pos else 0,
                    'market_value': float(db_pos.market_value) if db_pos else 0,
                    'last_updated': db_pos.last_updated.isoformat() if db_pos else None
                } if db_pos else {'exists': False},
                'broker_position': {
                    'exists': broker_pos is not None,
                    'quantity': broker_pos.quantity if broker_pos else 0,
                    'market_value': float(broker_pos.market_value) if broker_pos else 0,
                    'last_updated': broker_pos.last_updated.isoformat() if broker_pos else None
                } if broker_pos else {'exists': False},
                'reconciled': db_pos is not None and broker_pos is not None and 
                            abs(db_pos.quantity - broker_pos.quantity) <= self.quantity_tolerance
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get position details for {ticker}: {e}")
            return {'ticker': ticker, 'error': str(e)}


def create_position_reconciler(alpaca_manager=None, database_manager=None) -> PositionReconciler:
    """Factory function to create position reconciler"""
    return PositionReconciler(alpaca_manager, database_manager)