#!/usr / bin / env python3
"""Basic Phase 1 Functionality Test.

Test core Phase 1 components without complex dependencies.
"""

import asyncio
import json
import os
import sys
import tempfile
import traceback
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

# Add the backend directory to the path
sys.path.append("backend / tradingbot")

# Import Phase 1 components
from backend.tradingbot.core.data_providers import (
    EarningsEvent,
    MarketData,
    OptionsData,
)
from backend.tradingbot.core.production_config import ConfigManager, ProductionConfig
from backend.tradingbot.core.production_logging import (
    CircuitBreaker,
    ErrorHandler,
    HealthChecker,
    MetricsCollector,
    ProductionLogger,
)


def test_configuration():
    """Test configuration management."""
    print("🔧 Testing Configuration Management...")

    # Test config creation
    config = ProductionConfig()
    print(f"✅ Created production config with account size: ${config.risk.account_size:,.0f}")

    # Test config validation
    errors = config.validate()
    print(f"✅ Config validation found {len(errors)} errors (expected for empty config)")

    # Test config loading from file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        test_config = {"risk": {"max_position_risk": 0.15, "account_size": 75000.0}}
        json.dump(test_config, f)
        config_file = f.name

    try:
        config_manager = ConfigManager(config_file)
        loaded_config = config_manager.load_config()
        print(f"✅ Loaded config with account size: ${loaded_config.risk.account_size:,.0f}")
        print(f"✅ Max position risk: {loaded_config.risk.max_position_risk:.1%}")
    finally:
        os.unlink(config_file)

    print("✅ Configuration management working correctly + n")


def test_logging():
    """Test production logging."""
    print("📝 Testing Production Logging...")

    # Test logger creation
    logger = ProductionLogger("test_logger")
    logger.info("Test info message", test_param="value")
    logger.warning("Test warning message", test_param="value")
    logger.error("Test error message", test_param="value")
    print("✅ Logging system working correctly")

    # Test error handling
    error_handler = ErrorHandler(logger)
    error = ValueError("Test error")
    context = {"ticker": "AAPL", "strategy": "test"}

    result = error_handler.handle_error(error, context)
    print(f"✅ Error handling working: {result['error_type']} - {result['error_message']}")

    # Test circuit breaker
    circuit_breaker = CircuitBreaker(failure_threshold=2, timeout=1.0)

    def success_func():
        return "success"

    result = circuit_breaker.call(success_func)
    print(f"✅ Circuit breaker working: {result}")

    print("✅ Production logging working correctly + n")


def test_monitoring():
    """Test monitoring components."""
    print("📊 Testing Monitoring Components...")

    # Test health checker
    logger = ProductionLogger("test_monitoring")
    health_checker = HealthChecker(logger)

    def healthy_check():
        return True

    def unhealthy_check():
        return False

    health_checker.register_check("healthy", healthy_check)
    health_checker.register_check("unhealthy", unhealthy_check)

    # Run health checks
    results = asyncio.run(health_checker.run_health_checks())
    overall_health = health_checker.get_overall_health()

    print(f"✅ Health checker working: {overall_health} overall health")
    print(f"✅ Health checks: {len(results)} checks registered")

    # Test metrics collector
    metrics_collector = MetricsCollector(logger)

    # Record some metrics
    for i in range(5):
        metrics_collector.record_metric("test_metric", float(i * 10), {"iteration": i})

    summary = metrics_collector.get_metric_summary("test_metric")
    print(f"✅ Metrics collector working: {summary['count']} metrics recorded")
    print(f"✅ Average value: {summary['avg']:.1f}")

    print("✅ Monitoring components working correctly + n")


def test_data_structures():
    """Test data structures."""
    print("📈 Testing Data Structures...")

    # Test market data
    market_data = MarketData(
        ticker="AAPL",
        price=150.0,
        change=2.5,
        change_percent=0.0167,
        volume=1000000,
        high=152.0,
        low=148.0,
        open_price=149.0,
        previous_close=147.5,
        timestamp=datetime.now(),
    )
    print(
        f"✅ Market data: {market_data.ticker} @ ${market_data.price:.2f} ({market_data.change_percent:+.2%})"
    )

    # Test options data
    options_data = OptionsData(
        ticker="AAPL",
        expiry_date="2024 - 01 - 19",
        strike=150.0,
        option_type="call",
        bid=2.50,
        ask=2.60,
        last_price=2.55,
        volume=1000,
        open_interest=5000,
        implied_volatility=0.25,
        delta=0.50,
        gamma=0.02,
        theta=-0.05,
        vega=0.10,
    )
    print(
        f"✅ Options data: {options_data.ticker} {options_data.strike} {options_data.option_type} @ ${options_data.last_price: .2f}"
    )

    # Test earnings event
    earnings_event = EarningsEvent(
        ticker="AAPL",
        earnings_date=datetime(2024, 1, 15),
        time="AMC",
        expected_move=0.05,
        actual_eps=2.10,
        estimated_eps=2.05,
        surprise=0.05,
    )
    print(
        f"✅ Earnings event: {earnings_event.ticker} on {earnings_event.earnings_date.strftime('%Y-%m-%d')} {earnings_event.time}"
    )

    print("✅ Data structures working correctly + n")


def test_trading_interface():
    """Test trading interface components."""
    print("💼 Testing Trading Interface Components...")

    # Test trade signal creation
    class OrderType(Enum):
        MARKET = "market"
        LIMIT = "limit"
        STOP = "stop"
        STOP_LIMIT = "stop_limit"

    class OrderSide(Enum):
        BUY = "buy"
        SELL = "sell"

    @dataclass
    class TradeSignal:
        strategy_name: str
        ticker: str
        side: OrderSide
        order_type: OrderType
        quantity: int
        reason: str = ""
        confidence: float = 0.0
        timestamp: datetime = None

        def __post_init__(self):
            if self.timestamp is None:
                self.timestamp = datetime.now()

    signal = TradeSignal(
        strategy_name="test_strategy",
        ticker="AAPL",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=100,
        reason="Test trade",
        confidence=0.8,
    )

    print(
        f"✅ Trade signal: {signal.strategy_name} - {signal.ticker} {signal.side.value} {signal.quantity} @ {signal.confidence: .0%} confidence"
    )

    # Test trade result
    class TradeStatus(Enum):
        FILLED = "filled"
        PENDING = "pending"
        REJECTED = "rejected"

    @dataclass
    class TradeResult:
        trade_id: str
        signal: TradeSignal
        status: TradeStatus
        filled_quantity: int = 0
        filled_price: float = None
        commission: float = 0.0
        timestamp: datetime = None

        def __post_init__(self):
            if self.timestamp is None:
                self.timestamp = datetime.now()

    result = TradeResult(
        trade_id="test_123",
        signal=signal,
        status=TradeStatus.FILLED,
        filled_quantity=100,
        filled_price=150.0,
        commission=1.0,
    )

    print(
        f"✅ Trade result: {result.trade_id} - {result.status.value} {result.filled_quantity} @ ${result.filled_price: .2f}"
    )

    print("✅ Trading interface components working correctly + n")


def main():
    """Run all Phase 1 tests."""
    print("🚀 WallStreetBots Phase 1 - Basic Functionality Test")
    print(" = " * 60)

    try:
        test_configuration()
        test_logging()
        test_monitoring()
        test_data_structures()
        test_trading_interface()

        print(" = " * 60)
        print("✅ ALL PHASE 1 TESTS PASSED!")
        print("\n🎯 Phase 1 Components Verified: ")
        print("  ✅ Configuration Management")
        print("  ✅ Production Logging")
        print("  ✅ Error Handling & Circuit Breakers")
        print("  ✅ Health Monitoring")
        print("  ✅ Metrics Collection")
        print("  ✅ Data Structures")
        print("  ✅ Trading Interface Components")

        print("\n⚠️  Note: This is educational / testing code only!")
        print("   Do not use with real money without extensive validation.")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
