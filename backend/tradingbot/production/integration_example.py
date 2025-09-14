"""Integration example showing how to wire production modules together.

This example demonstrates the minimal wiring order for production-grade
trading system with all safety mechanisms in place.
"""
from __future__ import annotations
import logging
import time
from typing import Dict, Any

from ..risk.circuit_breaker import CircuitBreaker, BreakerLimits
from ..execution.replay_guard import ReplayGuard
from ..ops.eod_recon import EODReconciler, LocalOrder, BrokerFill
from ..data.quality import DataQualityMonitor
from ..infra.build_info import BuildStamper
from ..risk.greek_exposure_limits import GreekExposureLimiter, PositionGreeks
from ..execution.shadow_client import ShadowExecutionClient

log = logging.getLogger("wsb.integration")


class ProductionTradingSystem:
    """Production trading system with all safety mechanisms."""
    
    def __init__(self, start_equity: float = 100000.0):
        """Initialize production trading system.
        
        Args:
            start_equity: Starting equity for risk calculations
        """
        # Core safety mechanisms
        self.circuit_breaker = CircuitBreaker(start_equity)
        self.replay_guard = ReplayGuard()
        self.data_quality = DataQualityMonitor()
        self.greek_limiter = GreekExposureLimiter()
        self.eod_reconciler = EODReconciler()
        
        # Build stamping
        self.build_stamper = BuildStamper("trading_system")
        
        # Execution client (would be real in production)
        self.execution_client = None  # Set to real client
        
        log.info("Production trading system initialized")

    def validate_trading_conditions(self) -> None:
        """Validate all trading conditions before any order.
        
        Raises:
            RuntimeError: If any condition fails
        """
        # 1. Data quality check
        self.data_quality.assert_fresh()
        
        # 2. Circuit breaker check
        self.circuit_breaker.require_ok()
        
        # 3. Greek exposure limits
        self.greek_limiter.require_ok()
        
        # 4. EOD reconciliation check
        if self.eod_reconciler.check_pending_breaks():
            raise RuntimeError("EOD reconciliation breaks pending - trading disabled")

    def place_order(self, order_request: Dict[str, Any]) -> Dict[str, Any]:
        """Place order with full validation pipeline.
        
        Args:
            order_request: Order request dictionary
            
        Returns:
            Order response with build stamp
        """
        client_order_id = order_request.get("client_order_id")
        
        # Check replay guard
        if self.replay_guard.seen(client_order_id):
            raise RuntimeError(f"Order {client_order_id} already processed")
        
        # Validate trading conditions
        self.validate_trading_conditions()
        
        # Stamp order with build info
        stamped_order = self.build_stamper.stamp(order_request.copy())
        
        try:
            # Place order (would use real execution client)
            if self.execution_client:
                response = self.execution_client.place_order(stamped_order)
            else:
                # Mock response for example
                response = {
                    "client_order_id": client_order_id,
                    "broker_order_id": f"broker_{int(time.time())}",
                    "status": "acknowledged"
                }
            
            # Record in replay guard
            self.replay_guard.record(client_order_id, "acknowledged")
            
            # Update data freshness
            self.data_quality.mark_tick()
            
            log.info(f"Order placed successfully: {client_order_id}")
            return response
            
        except Exception as e:
            # Mark error in circuit breaker
            self.circuit_breaker.mark_error()
            log.error(f"Order placement failed: {e}")
            raise

    def update_portfolio_greeks(self, positions: Dict[str, PositionGreeks]) -> None:
        """Update portfolio Greeks.
        
        Args:
            positions: Dictionary of position Greeks
        """
        for position in positions.values():
            self.greek_limiter.add_position(position)
        
        log.info(f"Updated Greeks for {len(positions)} positions")

    def check_portfolio_health(self) -> Dict[str, Any]:
        """Check overall portfolio health.
        
        Returns:
            Health status dictionary
        """
        return {
            "circuit_breaker": self.circuit_breaker.status(),
            "data_quality": self.data_quality.status(),
            "greek_exposure": self.greek_limiter.status(),
            "replay_guard": self.replay_guard.status(),
        }

    def run_eod_reconciliation(self, local_orders: list, broker_fills: list) -> Dict[str, Any]:
        """Run end-of-day reconciliation.
        
        Args:
            local_orders: List of local order records
            broker_fills: List of broker fill records
            
        Returns:
            Reconciliation results
        """
        breaks = self.eod_reconciler.run_daily_reconciliation(local_orders, broker_fills)
        
        # Check if next day should be disabled
        should_disable = len(breaks.missing_fill) > 0 or len(breaks.unknown_broker_fill) > 0
        
        return {
            "breaks": breaks,
            "should_disable_next_day": should_disable,
        }

    def start_trading_session(self) -> None:
        """Start a new trading session."""
        log.info("Starting trading session")
        
        # Reset circuit breaker if needed
        if not self.circuit_breaker.can_trade():
            log.warning("Circuit breaker is open - manual reset required")
            return
        
        # Mark data as fresh
        self.data_quality.mark_tick()
        
        log.info("Trading session started successfully")

    def stop_trading_session(self) -> None:
        """Stop trading session."""
        log.info("Stopping trading session")
        
        # Final data quality check
        try:
            self.data_quality.assert_fresh()
        except RuntimeError as e:
            log.warning(f"Data quality issue at session end: {e}")
        
        log.info("Trading session stopped")


def example_usage():
    """Example usage of production trading system."""
    
    # Initialize system
    system = ProductionTradingSystem(start_equity=1000000.0)
    
    # Start trading session
    system.start_trading_session()
    
    # Update portfolio Greeks
    positions = {
        "AAPL": PositionGreeks("AAPL", 1000.0, 50.0, -10.0, 200.0, 1.2, 100000.0),
        "MSFT": PositionGreeks("MSFT", 500.0, 25.0, -5.0, 100.0, 0.8, 50000.0),
    }
    system.update_portfolio_greeks(positions)
    
    # Check health
    health = system.check_portfolio_health()
    log.info(f"Portfolio health: {health}")
    
    # Place an order
    order_request = {
        "client_order_id": "order_123",
        "symbol": "AAPL",
        "side": "buy",
        "quantity": 100,
        "order_type": "market"
    }
    
    try:
        response = system.place_order(order_request)
        log.info(f"Order response: {response}")
    except RuntimeError as e:
        log.error(f"Order failed: {e}")
    
    # Stop trading session
    system.stop_trading_session()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    example_usage()

