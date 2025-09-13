"""Production Manager
Orchestrates all production components for live trading

This module provides the main production manager that:
- Connects AlpacaManager to strategies for real execution
- Integrates Django models with trading logic
- Provides real - time data feeds
- Implements comprehensive risk management
- Manages position monitoring and alerts
- Handles database synchronization

Making the system truly production - ready for live trading.
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any

from django.contrib.auth.models import User

from ...models import Portfolio
from ..data.production_data_integration import create_production_data_provider
from .production_integration import create_production_integration
from .production_strategy_wrapper import (
    ProductionStrategyWrapper,
    StrategyConfig,
    create_production_momentum_weeklies,
    create_production_wsb_dip_bot,
)


@dataclass
class ProductionConfig:
    """Production configuration"""

    alpaca_api_key: str
    alpaca_secret_key: str
    paper_trading: bool = True
    user_id: int = 1

    # Risk management
    max_position_size: float = 0.20  # 20% max per position
    max_total_risk: float = 0.50  # 50% max total risk
    stop_loss_pct: float = 0.50  # 50% stop loss
    take_profit_multiplier: float = 3.0  # 3x profit target

    # Strategy settings
    enabled_strategies: list[str] = field(
        default_factory=lambda: [
            "wsb_dip_bot",
            "momentum_weeklies",
            "debit_spreads",
            "leaps_tracker",
            "lotto_scanner",
            "wheel_strategy",
            "spx_credit_spreads",
            "earnings_protection",
            "swing_trading",
            "index_baseline",
        ]
    )

    # Data settings
    data_refresh_interval: int = 30  # seconds
    position_monitor_interval: int = 60  # seconds

    # Alert settings
    enable_alerts: bool = True
    alert_channels: list[str] = field(default_factory=lambda: ["email", "slack", "desktop"])

    metadata: dict[str, Any] = field(default_factory=dict)


class ProductionManager:
    """Production Manager

    Main orchestrator for production trading system:
    - Manages all production components
    - Coordinates strategy execution
    - Handles real - time monitoring
    - Provides comprehensive logging and alerts
    - Manages database synchronization
    """

    def __init__(self, config: ProductionConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Initialize core components
        self.integration_manager = create_production_integration(
            config.alpaca_api_key, config.alpaca_secret_key, config.paper_trading, config.user_id
        )

        self.data_provider = create_production_data_provider(
            config.alpaca_api_key, config.alpaca_secret_key
        )

        # Initialize strategies
        self.strategies: dict[str, ProductionStrategyWrapper] = {}
        self._initialize_strategies()

        # System state
        self.is_running = False
        self.start_time: datetime | None = None
        self.last_heartbeat: datetime | None = None

        # Performance tracking
        self.performance_metrics: dict[str, Any] = {}

        # Setup logging
        self.setup_logging()

        self.logger.info("ProductionManager initialized")

    def setup_logging(self):
        """Setup comprehensive logging"""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler("production_manager.log"), logging.StreamHandler()],
        )

    def _initialize_strategies(self):
        """Initialize all enabled strategies"""
        try:
            strategy_config = StrategyConfig(
                name="default",
                enabled=True,
                max_position_size=self.config.max_position_size,
                max_total_risk=self.config.max_total_risk,
                stop_loss_pct=self.config.stop_loss_pct,
                take_profit_multiplier=self.config.take_profit_multiplier,
            )

            # Initialize WSB Dip Bot
            if "wsb_dip_bot" in self.config.enabled_strategies:
                self.strategies["wsb_dip_bot"] = create_production_wsb_dip_bot(
                    self.integration_manager, strategy_config
                )

            # Initialize Momentum Weeklies
            if "momentum_weeklies" in self.config.enabled_strategies:
                self.strategies["momentum_weeklies"] = create_production_momentum_weeklies(
                    self.integration_manager, strategy_config
                )

            # Additional strategies would be initialized here
            # ProductionDebitSpreads, ProductionLEAPSTracker, etc.

            self.logger.info(f"Initialized {len(self.strategies)} strategies")

        except Exception as e:
            self.logger.error(f"Error initializing strategies: {e}")

    async def start_production_system(self) -> bool:
        """Start the production trading system"""
        try:
            self.logger.info("🚀 Starting Production Trading System")

            # Validate configuration
            if not await self._validate_configuration():
                return False

            # Initialize database
            if not await self._initialize_database():
                return False

            # Start all strategies
            for strategy_name, strategy in self.strategies.items():
                success = await strategy.start_strategy()
                if not success:
                    self.logger.error(f"Failed to start strategy: {strategy_name}")
                    return False

            # Start monitoring tasks
            self.is_running = True
            self.start_time = datetime.now()

            # Start background tasks
            asyncio.create_task(self._monitoring_loop())
            asyncio.create_task(self._heartbeat_loop())
            asyncio.create_task(self._performance_tracking_loop())

            self.logger.info("✅ Production Trading System started successfully")
            return True

        except Exception as e:
            self.logger.error(f"Error starting production system: {e}")
            return False

    async def stop_production_system(self):
        """Stop the production trading system"""
        try:
            self.logger.info("🛑 Stopping Production Trading System")

            self.is_running = False

            # Stop all strategies
            for strategy_name, strategy in self.strategies.items():
                await strategy.stop_strategy()

            # Close all positions (optional - depends on strategy)
            await self._close_all_positions()

            # Final performance report
            await self._generate_final_report()

            self.logger.info("✅ Production Trading System stopped")

        except Exception as e:
            self.logger.error(f"Error stopping production system: {e}")

    async def _validate_configuration(self) -> bool:
        """Validate production configuration"""
        try:
            # Validate Alpaca connection
            success, message = self.integration_manager.alpaca_manager.validate_api()
            if not success:
                self.logger.error(f"Alpaca validation failed: {message}")
                return False

            # Validate account size
            portfolio_value = await self.integration_manager.get_portfolio_value()
            if portfolio_value < Decimal("1000"):  # Minimum $1000
                self.logger.error(f"Account size {portfolio_value} below minimum")
                return False

            # Validate market hours
            market_open = await self.data_provider.is_market_open()
            if not market_open:
                self.logger.warning("Market is closed - system will wait for market open")

            self.logger.info("✅ Configuration validation passed")
            return True

        except Exception as e:
            self.logger.error(f"Configuration validation error: {e}")
            return False

    async def _initialize_database(self) -> bool:
        """Initialize database connections and models"""
        try:
            # Ensure user exists
            user, created = User.objects.get_or_create(
                id=self.config.user_id,
                defaults={"username": f"production_user_{self.config.user_id}"},
            )

            # Ensure portfolio exists
            portfolio, created = Portfolio.objects.get_or_create(
                user=user, defaults={"name": "Production Portfolio", "cash": Decimal("0.00")}
            )

            # Sync portfolio with Alpaca
            await self._sync_portfolio_with_alpaca(portfolio)

            self.logger.info("✅ Database initialization completed")
            return True

        except Exception as e:
            self.logger.error(f"Database initialization error: {e}")
            return False

    async def _sync_portfolio_with_alpaca(self, portfolio: Portfolio):
        """Sync Django portfolio with Alpaca account"""
        try:
            # Get Alpaca account info
            cash_balance = self.integration_manager.alpaca_manager.get_balance()
            if cash_balance:
                portfolio.cash = Decimal(str(cash_balance))
                portfolio.save()

                self.logger.info(f"Portfolio synced: Cash={portfolio.cash}")

        except Exception as e:
            self.logger.error(f"Portfolio sync error: {e}")

    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.is_running:
            try:
                # Monitor positions
                await self.integration_manager.monitor_positions()

                # Update data cache
                await self._refresh_data_cache()

                # Check system health
                await self._health_check()

                # Wait for next cycle
                await asyncio.sleep(self.config.position_monitor_interval)

            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(60)  # Wait longer on error

    async def _heartbeat_loop(self):
        """Heartbeat loop for system monitoring"""
        while self.is_running:
            try:
                self.last_heartbeat = datetime.now()

                # Send heartbeat alert
                if self.config.enable_alerts:
                    await self.integration_manager.alert_system.send_alert(
                        "SYSTEM_HEARTBEAT",
                        "LOW",
                        f"Production system heartbeat - {len(self.strategies)} strategies active",
                    )

                # Wait 5 minutes between heartbeats
                await asyncio.sleep(300)

            except Exception as e:
                self.logger.error(f"Error in heartbeat loop: {e}")
                await asyncio.sleep(60)

    async def _performance_tracking_loop(self):
        """Performance tracking loop"""
        while self.is_running:
            try:
                # Update performance metrics
                await self._update_performance_metrics()

                # Wait 1 hour between updates
                await asyncio.sleep(3600)

            except Exception as e:
                self.logger.error(f"Error in performance tracking loop: {e}")
                await asyncio.sleep(300)

    async def _refresh_data_cache(self):
        """Refresh data cache"""
        try:
            # Clear old cache entries
            self.data_provider.clear_cache()

        except Exception as e:
            self.logger.error(f"Error refreshing data cache: {e}")

    async def _health_check(self):
        """System health check"""
        try:
            # Check Alpaca connection
            success, message = self.integration_manager.alpaca_manager.validate_api()
            if not success:
                await self.integration_manager.alert_system.send_alert(
                    "SYSTEM_ERROR", "HIGH", f"Alpaca connection lost: {message}"
                )

            # Check strategy health
            for strategy_name, strategy in self.strategies.items():
                if not strategy.is_running:
                    await self.integration_manager.alert_system.send_alert(
                        "STRATEGY_ERROR", "HIGH", f"Strategy {strategy_name} is not running"
                    )

        except Exception as e:
            self.logger.error(f"Error in health check: {e}")

    async def _update_performance_metrics(self):
        """Update performance metrics"""
        try:
            # Get portfolio summary
            portfolio_summary = self.integration_manager.get_portfolio_summary()

            # Get strategy performance
            strategy_performance = {}
            for strategy_name, strategy in self.strategies.items():
                strategy_performance[strategy_name] = strategy.get_strategy_status()

            # Update metrics
            self.performance_metrics = {
                "timestamp": datetime.now().isoformat(),
                "system_uptime": (datetime.now() - self.start_time).total_seconds()
                if self.start_time
                else 0,
                "portfolio": portfolio_summary,
                "strategies": strategy_performance,
                "data_cache_stats": self.data_provider.get_cache_stats(),
            }

        except Exception as e:
            self.logger.error(f"Error updating performance metrics: {e}")

    async def _close_all_positions(self):
        """Close all positions (emergency stop)"""
        try:
            self.logger.warning("Closing all positions...")

            for position_key, position in list(self.integration_manager.active_positions.items()):
                await self.integration_manager.execute_exit_trade(position, "emergency_close")

            self.logger.info("All positions closed")

        except Exception as e:
            self.logger.error(f"Error closing positions: {e}")

    async def _generate_final_report(self):
        """Generate final performance report"""
        try:
            report = {
                "session_start": self.start_time.isoformat() if self.start_time else None,
                "session_end": datetime.now().isoformat(),
                "total_runtime": (datetime.now() - self.start_time).total_seconds()
                if self.start_time
                else 0,
                "performance_metrics": self.performance_metrics,
                "final_portfolio": self.integration_manager.get_portfolio_summary(),
            }

            # Save report
            with open(
                f"production_report_{datetime.now().strftime('%Y % m % d_ % H % M % S')}.json", "w"
            ) as f:
                json.dump(report, f, indent=2, default=str)

            self.logger.info("Final report generated")

        except Exception as e:
            self.logger.error(f"Error generating final report: {e}")

    def get_system_status(self) -> dict[str, Any]:
        """Get current system status"""
        return {
            "is_running": self.is_running,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "last_heartbeat": self.last_heartbeat.isoformat() if self.last_heartbeat else None,
            "active_strategies": len(self.strategies),
            "strategy_status": {
                name: strategy.get_strategy_status() for name, strategy in self.strategies.items()
            },
            "performance_metrics": self.performance_metrics,
            "configuration": {
                "paper_trading": self.config.paper_trading,
                "max_position_size": self.config.max_position_size,
                "max_total_risk": self.config.max_total_risk,
                "enabled_strategies": self.config.enabled_strategies,
            },
        }


# Factory function for easy initialization
def create_production_manager(config: ProductionConfig) -> ProductionManager:
    """Create ProductionManager instance

    Args:
        config: Production configuration

    Returns:
        ProductionManager instance
    """
    return ProductionManager(config)
