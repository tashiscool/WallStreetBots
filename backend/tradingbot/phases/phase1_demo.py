"""Phase 1 Demo Script
Demonstrate the unified trading interface and production components.
"""

import asyncio
import logging
from datetime import datetime

# Import with try / except for standalone execution
try:
    from .data_providers import create_data_provider
    from .production_config import ConfigManager, create_config_manager
    from .production_logging import (
        CircuitBreaker,
        ErrorHandler,
        HealthChecker,
        MetricsCollector,
        ProductionLogger,
    )
    from .production_models import Configuration, Position, Strategy, Trade
    from .trading_interface import (
        OrderSide,
        OrderType,
        TradeSignal,
        TradingInterface,
        create_trading_interface,
    )
except ImportError:
    # For standalone execution
    import os
    import sys

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from data_providers import create_data_provider
    from production_config import create_config_manager
    from production_logging import (
        CircuitBreaker,
        ErrorHandler,
        HealthChecker,
        MetricsCollector,
        ProductionLogger,
    )
    from trading_interface import OrderSide, OrderType, TradeSignal, create_trading_interface


class Phase1Demo:
    """Demonstrate Phase 1 implementation."""

    def __init__(self, config_file: str = "config / production.json"):
        self.config_file = config_file
        self.logger = ProductionLogger("phase1_demo")
        self.error_handler = ErrorHandler(self.logger)
        self.metrics_collector = MetricsCollector(self.logger)
        self.health_checker = HealthChecker(self.logger)

        # Initialize components
        self.config_manager = None
        self.config = None
        self.data_provider = None
        self.trading_interface = None

    async def initialize(self):
        """Initialize all Phase 1 components."""
        self.logger.info("Initializing Phase 1 components")

        try:
            # 1. Load configuration
            self.config_manager = create_config_manager(self.config_file)
            self.config = self.config_manager.load_config()

            # Validate configuration
            errors = self.config.validate()
            if errors:
                self.logger.warning(f"Configuration validation errors: {errors}")

            # 2. Create data provider
            self.data_provider = create_data_provider(self.config.data_providers.__dict__)

            # 3. Create trading interface
            self.trading_interface = create_trading_interface(self.config.to_dict())

            # 4. Setup health checks
            await self.setup_health_checks()

            self.logger.info("Phase 1 components initialized successfully")

        except Exception as e:
            self.error_handler.handle_error(e, {"component": "initialization"})
            raise

    async def setup_health_checks(self):
        """Setup system health checks."""

        # Data provider health check
        async def data_provider_check():
            try:
                # Try to fetch data for a test ticker
                data = await self.data_provider.get_market_data("AAPL")
                return data.price > 0
            except Exception:
                return False

        # Trading interface health check
        async def trading_interface_check():
            try:
                # Check if broker connection is working
                account_info = await self.trading_interface.get_account_info()
                return account_info.get("equity", 0) >= 0
            except Exception:
                return False

        # Register health checks
        self.health_checker.register_check("data_provider", data_provider_check)
        self.health_checker.register_check("trading_interface", trading_interface_check)

    async def demonstrate_data_integration(self):
        """Demonstrate real data integration."""
        self.logger.info("Demonstrating data integration")

        try:
            # Test market data fetching
            tickers = ["AAPL", "MSFT", "GOOGL"]

            for ticker in tickers:
                try:
                    data = await self.data_provider.get_market_data(ticker)
                    self.logger.info(
                        f"Market data for {ticker}",
                        ticker=ticker,
                        price=data.price,
                        change=data.change,
                        volume=data.volume,
                    )

                    # Record metrics
                    self.metrics_collector.record_metric(
                        "market_data_price", data.price, {"ticker": ticker}
                    )

                except Exception as e:
                    self.error_handler.handle_error(
                        e, {"ticker": ticker, "operation": "market_data"}
                    )

            # Test earnings data
            try:
                earnings_events = await self.data_provider.get_earnings_data("AAPL", days_ahead=7)
                self.logger.info(
                    f"Found {len(earnings_events)} upcoming earnings events",
                    count=len(earnings_events),
                )

                for event in earnings_events:
                    self.logger.info(
                        f"Earnings event: {event.ticker}",
                        ticker=event.ticker,
                        date=event.earnings_date.isoformat(),
                        time=event.time,
                    )

            except Exception as e:
                self.error_handler.handle_error(e, {"operation": "earnings_data"})

            # Test sentiment analysis
            try:
                sentiment = await self.data_provider.get_sentiment_data("AAPL")
                self.logger.info(
                    "Sentiment analysis for AAPL",
                    score=sentiment.get("score", 0),
                    confidence=sentiment.get("confidence", 0),
                )

                self.metrics_collector.record_metric(
                    "sentiment_score", sentiment.get("score", 0), {"ticker": "AAPL"}
                )

            except Exception as e:
                self.error_handler.handle_error(e, {"operation": "sentiment_analysis"})

        except Exception as e:
            self.error_handler.handle_error(e, {"component": "data_integration"})

    async def demonstrate_trading_interface(self):
        """Demonstrate trading interface functionality."""
        self.logger.info("Demonstrating trading interface")

        try:
            # Create a test trade signal
            signal = TradeSignal(
                strategy_name="demo_strategy",
                ticker="AAPL",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=10,
                reason="Demo trade for Phase 1",
                confidence=0.8,
            )

            self.logger.info(
                "Created trade signal",
                strategy=signal.strategy_name,
                ticker=signal.ticker,
                side=signal.side.value,
                quantity=signal.quantity,
            )

            # Validate signal
            validation_result = await self.trading_interface.validate_signal(signal)
            self.logger.info(
                "Signal validation result",
                valid=validation_result["valid"],
                reason=validation_result["reason"],
            )

            # Check risk limits
            risk_result = await self.trading_interface.check_risk_limits(signal)
            self.logger.info(
                "Risk check result", allowed=risk_result["allowed"], reason=risk_result["reason"]
            )

            # Record metrics
            self.metrics_collector.record_metric(
                "signal_validation",
                1 if validation_result["valid"] else 0,
                {"strategy": signal.strategy_name},
            )

            self.metrics_collector.record_metric(
                "risk_check_passed",
                1 if risk_result["allowed"] else 0,
                {"strategy": signal.strategy_name},
            )

            # If validation and risk checks pass, demonstrate execution
            if validation_result["valid"] and risk_result["allowed"]:
                self.logger.info("Signal passed validation and risk checks - would execute trade")

                # In a real implementation, this would execute the trade
                # trade_result=await self.trading_interface.execute_trade(signal)

            else:
                self.logger.warning(
                    "Signal failed validation or risk checks - trade would be rejected"
                )

        except Exception as e:
            self.error_handler.handle_error(e, {"component": "trading_interface"})

    async def demonstrate_error_handling(self):
        """Demonstrate error handling and resilience."""
        self.logger.info("Demonstrating error handling")

        try:
            # Test circuit breaker
            circuit_breaker = CircuitBreaker(failure_threshold=3, timeout=5.0)

            def flaky_function():
                import random

                if random.random() < 0.7:  # 70% failure rate
                    raise Exception("Simulated failure")
                return "success"

            # Test circuit breaker behavior
            for i in range(5):
                try:
                    result = circuit_breaker.call(flaky_function)
                    self.logger.info(f"Circuit breaker call {i + 1}: {result}")
                except Exception as e:
                    self.logger.warning(f"Circuit breaker call {i + 1} failed: {e}")

            # Test retry mechanism
            call_count = 0

            def retry_function():
                nonlocal call_count
                call_count += 1
                if call_count < 3:
                    raise Exception("Temporary failure")
                return "success after retries"

            try:
                result = retry_function()
                self.logger.info(f"Retry function result: {result}")
                self.logger.info(f"Total attempts: {call_count}")
            except Exception as e:
                self.logger.error(f"Retry function failed: {e}")

        except Exception as e:
            self.error_handler.handle_error(e, {"component": "error_handling"})

    async def demonstrate_monitoring(self):
        """Demonstrate monitoring and metrics."""
        self.logger.info("Demonstrating monitoring and metrics")

        try:
            # Run health checks
            health_results = await self.health_checker.run_health_checks()
            overall_health = self.health_checker.get_overall_health()

            self.logger.info(
                "Health check results", overall_health=overall_health, results=health_results
            )

            # Generate metrics summary
            metrics_summary = {}
            for metric_name in self.metrics_collector.metrics:
                summary = self.metrics_collector.get_metric_summary(metric_name)
                if summary:
                    metrics_summary[metric_name] = summary

            self.logger.info("Metrics summary", summary=metrics_summary)

            # Export metrics
            metrics_file = f"demo_metrics_{datetime.now().strftime('%Y % m % d_ % H % M % S')}.json"
            self.metrics_collector.export_metrics(metrics_file)
            self.logger.info(f"Metrics exported to {metrics_file}")

        except Exception as e:
            self.error_handler.handle_error(e, {"component": "monitoring"})

    async def run_demo(self):
        """Run complete Phase 1 demonstration."""
        self.logger.info("Starting Phase 1 demonstration")

        try:
            # Initialize components
            await self.initialize()

            # Demonstrate each component
            await self.demonstrate_data_integration()
            await self.demonstrate_trading_interface()
            await self.demonstrate_error_handling()
            await self.demonstrate_monitoring()

            self.logger.info("Phase 1 demonstration completed successfully")

        except Exception as e:
            self.error_handler.handle_error(e, {"component": "demo"})
            self.logger.error("Phase 1 demonstration failed")
            raise


async def main():
    """Main demo function."""
    demo = Phase1Demo()
    await demo.run_demo()


if __name__ == "__main__":  # Setup logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Run demo
    asyncio.run(main())
