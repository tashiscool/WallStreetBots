"""Risk - Integrated Production Strategy Manager
Integrates sophisticated risk management with WallStreetBots trading strategies.

This module provides:
- Real - time risk assessment during trading
- Automated risk controls and position sizing
- Cross - strategy risk coordination
- Risk alerts and monitoring integration
- Portfolio - level risk management

Month 3 - 4: Integration with WallStreetBots
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from ...production.core.production_strategy_manager import (
    ProductionStrategyManager,
    ProductionStrategyManagerConfig,
)
from ..risk_aware_strategy_wrapper import RiskAwareStrategy, create_risk_aware_strategy
from .risk_integration_manager import RiskIntegrationManager, RiskLimits


@dataclass
class RiskIntegratedConfig(ProductionStrategyManagerConfig):
    """Configuration for risk - integrated production manager."""

    # Risk management settings
    risk_limits: RiskLimits = field(default_factory=RiskLimits)
    enable_ml_risk: bool = True
    enable_stress_testing: bool = True
    enable_risk_dashboard: bool = True

    # Risk monitoring settings
    risk_calculation_interval: int = 30  # seconds
    risk_alert_threshold: float = 0.8  # 80% of limits

    # Risk integration settings
    auto_position_sizing: bool = True
    auto_risk_controls: bool = True
    cross_strategy_coordination: bool = True


class RiskIntegratedProductionManager:
    """Production strategy manager with integrated sophisticated risk management.

    This manager combines all WallStreetBots trading strategies with:
    - Real - time VaR / CVaR calculations
    - FCA - compliant stress testing
    - Machine learning risk prediction
    - Automated risk controls
    - Cross - strategy risk coordination
    """

    def __init__(self, config: RiskIntegratedConfig):
        """Initialize risk - integrated production manager.

        Args:
            config: Configuration including risk management settings
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Initialize base production manager
        self.base_manager = ProductionStrategyManager(config)

        # Initialize risk management
        self.risk_manager = RiskIntegrationManager(
            risk_limits=config.risk_limits,
            enable_ml=config.enable_ml_risk,
            enable_stress_testing=config.enable_stress_testing,
            enable_dashboard=config.enable_risk_dashboard,
        )

        # Risk - aware strategies
        self.risk_aware_strategies: dict[str, RiskAwareStrategy] = {}

        # Risk monitoring state
        self.risk_monitoring_active = False
        self.last_risk_calculation = None
        self.risk_alerts_sent = set()

        # Performance tracking
        self.risk_calculations_count = 0
        self.trades_blocked_by_risk = 0
        self.risk_adjustments_made = 0

        self.logger.info("Risk - Integrated Production Manager initialized")

    async def start_all_strategies(self) -> bool:
        """Start all strategies with integrated risk management.

        Returns:
            bool: True if all strategies started successfully
        """
        try:
            # Start base strategies
            base_success = await self.base_manager.start_all_strategies()
            if not base_success:
                self.logger.error("Failed to start base strategies")
                return False

            # Create risk - aware strategy wrappers
            await self._create_risk_aware_strategies()

            # Start risk monitoring
            await self._start_risk_monitoring()

            self.logger.info("All strategies started with risk management integration")
            return True

        except Exception as e:
            self.logger.error(f"Error starting strategies with risk management: {e}")
            return False

    async def _create_risk_aware_strategies(self):
        """Create risk - aware wrappers for all strategies."""
        try:
            for (
                strategy_name,
                strategy_instance,
            ) in self.base_manager.strategies.items():
                risk_aware_strategy = create_risk_aware_strategy(
                    strategy_instance, self.risk_manager, strategy_name
                )
                self.risk_aware_strategies[strategy_name] = risk_aware_strategy

                self.logger.info(f"Created risk - aware wrapper for {strategy_name}")

        except Exception as e:
            self.logger.error(f"Error creating risk - aware strategies: {e}")

    async def _start_risk_monitoring(self):
        """Start continuous risk monitoring."""
        try:
            self.risk_monitoring_active = True

            # Start risk monitoring task
            task = asyncio.create_task(self._risk_monitoring_loop())
            self.tasks.append(task)

            self.logger.info("Risk monitoring started")

        except Exception as e:
            self.logger.error(f"Error starting risk monitoring: {e}")

    async def _risk_monitoring_loop(self):
        """Continuous risk monitoring loop."""
        while self.risk_monitoring_active:
            try:
                # Calculate portfolio risk
                await self._calculate_and_update_risk()

                # Check for risk alerts
                await self._check_risk_alerts()

                # Wait for next calculation
                await asyncio.sleep(self.config.risk_calculation_interval)

            except Exception as e:
                self.logger.error(f"Error in risk monitoring loop: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retrying

    async def _calculate_and_update_risk(self):
        """Calculate and update portfolio risk metrics."""
        try:
            # Get current positions
            positions = await self._get_current_positions()

            # Get market data
            market_data = await self._get_market_data()

            # Get portfolio value
            portfolio_value = await self._get_portfolio_value()

            # Calculate risk metrics
            risk_metrics = await self.risk_manager.calculate_portfolio_risk(
                positions, market_data, portfolio_value
            )

            # Update tracking
            self.risk_calculations_count += 1
            self.last_risk_calculation = datetime.now()

            self.logger.debug(
                f"Risk calculated: VaR={risk_metrics.portfolio_var:.2%}, "
                f"CVaR={risk_metrics.portfolio_cvar: .2%}, "
                f"Within limits: {risk_metrics.within_limits}"
            )

        except Exception as e:
            self.logger.error(f"Error calculating risk: {e}")

    async def _check_risk_alerts(self):
        """Check for risk alerts and send notifications."""
        try:
            if not self.risk_manager.current_metrics.within_limits:
                # Send risk limit breach alert
                await self._send_risk_alert(
                    "Risk limits breached", self.risk_manager.current_metrics.alerts
                )

            # Check utilization thresholds
            utilization = self._calculate_risk_utilization()

            for metric, util in utilization.items():
                if util > self.config.risk_alert_threshold:
                    alert_key = f"{metric}_high_utilization"
                    if alert_key not in self.risk_alerts_sent:
                        await self._send_risk_alert(
                            f"High {metric} utilization: {util:.1%}"
                        )
                        self.risk_alerts_sent.add(alert_key)

        except Exception as e:
            self.logger.error(f"Error checking risk alerts: {e}")

    def _calculate_risk_utilization(self) -> dict[str, float]:
        """Calculate risk utilization percentages."""
        metrics = self.risk_manager.current_metrics
        limits = self.risk_manager.risk_limits

        return {
            "var": metrics.portfolio_var / limits.max_total_var
            if limits.max_total_var > 0
            else 0,
            "cvar": metrics.portfolio_cvar / limits.max_total_cvar
            if limits.max_total_cvar > 0
            else 0,
            "concentration": metrics.concentration_risk / limits.max_concentration
            if limits.max_concentration > 0
            else 0,
            "greeks": metrics.greeks_risk / limits.max_greeks_risk
            if limits.max_greeks_risk > 0
            else 0,
        }

    async def _send_risk_alert(self, message: str, details: list[str] | None = None):
        """Send risk alert notification."""
        try:
            alert_message = f"🚨 RISK ALERT: {message}"
            if details:
                alert_message += f"\nDetails: {', '.join(details)}"

            self.logger.warning(alert_message)

            # Here you would integrate with your alerting system
            # For now, we'll just log it

        except Exception as e:
            self.logger.error(f"Error sending risk alert: {e}")

    async def _get_current_positions(self) -> dict[str, dict]:
        """Get current portfolio positions."""
        try:
            # This would get actual positions from the broker
            # For now, return simulated positions
            return {
                "AAPL": {
                    "qty": 100,
                    "value": 15000,
                    "delta": 0.6,
                    "gamma": 0.01,
                    "vega": 0.5,
                },
                "SPY": {
                    "qty": 50,
                    "value": 20000,
                    "delta": 0.5,
                    "gamma": 0.005,
                    "vega": 0.3,
                },
                "TSLA": {
                    "qty": 25,
                    "value": 5000,
                    "delta": 0.8,
                    "gamma": 0.02,
                    "vega": 0.8,
                },
            }
        except Exception as e:
            self.logger.error(f"Error getting current positions: {e}")
            return {}

    async def _get_market_data(self) -> dict[str, Any]:
        """Get current market data."""
        try:
            # This would get actual market data
            # For now, return simulated data
            import numpy as np
            import pandas as pd

            dates = pd.date_range(end=datetime.now(), periods=252, freq="D")
            returns = np.random.normal(0, 0.02, 252)
            prices = 100 * np.cumprod(1 + returns)

            return {
                "AAPL": pd.DataFrame(
                    {
                        "Open": prices * 0.99,
                        "High": prices * 1.02,
                        "Low": prices * 0.98,
                        "Close": prices,
                        "Volume": np.random.randint(1000000, 5000000, 252),
                    },
                    index=dates,
                ),
                "SPY": pd.DataFrame(
                    {
                        "Open": prices * 0.99,
                        "High": prices * 1.01,
                        "Low": prices * 0.99,
                        "Close": prices,
                        "Volume": np.random.randint(5000000, 10000000, 252),
                    },
                    index=dates,
                ),
                "TSLA": pd.DataFrame(
                    {
                        "Open": prices * 0.95,
                        "High": prices * 1.05,
                        "Low": prices * 0.95,
                        "Close": prices,
                        "Volume": np.random.randint(2000000, 8000000, 252),
                    },
                    index=dates,
                ),
            }
        except Exception as e:
            self.logger.error(f"Error getting market data: {e}")
            return {}

    async def _get_portfolio_value(self) -> float:
        """Get current portfolio value."""
        try:
            # This would get actual portfolio value from the broker
            # For now, return simulated value
            return 100000.0  # $100k simulated portfolio
        except Exception as e:
            self.logger.error(f"Error getting portfolio value: {e}")
            return 100000.0

    async def execute_strategy_trade(
        self,
        strategy_name: str,
        symbol: str,
        action: str,
        quantity: float,
        price: float | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Execute a trade through a specific strategy with risk management.

        Args:
            strategy_name: Name of the strategy
            symbol: Symbol to trade
            action: Trade action
            quantity: Quantity to trade
            price: Price (for limit orders)
            **kwargs: Additional parameters

        Returns:
            Dict: Trade execution result
        """
        try:
            if strategy_name not in self.risk_aware_strategies:
                return {
                    "success": False,
                    "reason": f"Strategy {strategy_name} not found",
                    "strategy": strategy_name,
                }

            # Execute trade through risk - aware strategy
            risk_aware_strategy = self.risk_aware_strategies[strategy_name]
            result = await risk_aware_strategy.execute_trade(
                symbol, action, quantity, price, **kwargs
            )

            # Update tracking
            if not result.get("success", False):
                if result.get("blocked_by_risk", False):
                    self.trades_blocked_by_risk += 1
                if result.get("risk_adjusted", False):
                    self.risk_adjustments_made += 1

            return result

        except Exception as e:
            self.logger.error(f"Error executing strategy trade: {e}")
            return {
                "success": False,
                "reason": f"Error: {e}",
                "strategy": strategy_name,
            }

    async def get_risk_summary(self) -> dict[str, Any]:
        """Get comprehensive risk summary."""
        try:
            risk_summary = await self.risk_manager.get_risk_summary()

            # Add manager - specific metrics
            risk_summary["manager_metrics"] = {
                "risk_calculations_count": self.risk_calculations_count,
                "trades_blocked_by_risk": self.trades_blocked_by_risk,
                "risk_adjustments_made": self.risk_adjustments_made,
                "last_risk_calculation": self.last_risk_calculation,
                "risk_monitoring_active": self.risk_monitoring_active,
            }

            # Add strategy - specific risk metrics
            strategy_risks = {}
            for name, strategy in self.risk_aware_strategies.items():
                strategy_risks[name] = await strategy.get_risk_status()

            risk_summary["strategy_risks"] = strategy_risks

            return risk_summary

        except Exception as e:
            self.logger.error(f"Error getting risk summary: {e}")
            return {"error": str(e)}

    async def get_system_status(self) -> dict[str, Any]:
        """Get overall system status including risk management."""
        try:
            # Get base system status
            base_status = self.base_manager.get_system_status()

            # Get risk status
            risk_status = await self.get_risk_summary()

            # Combine statuses
            return {
                **base_status,
                "risk_management": {
                    "enabled": True,
                    "monitoring_active": self.risk_monitoring_active,
                    "current_metrics": risk_status.get("metrics", {}),
                    "within_limits": risk_status.get("metrics", {}).get(
                        "within_limits", True
                    ),
                    "alerts": risk_status.get("metrics", {}).get("alerts", []),
                },
                "risk_performance": risk_status.get("manager_metrics", {}),
            }

        except Exception as e:
            self.logger.error(f"Error getting system status: {e}")
            return {"error": str(e)}

    async def stop_all_strategies(self):
        """Stop all strategies and risk monitoring."""
        try:
            # Stop risk monitoring
            self.risk_monitoring_active = False

            # Stop base strategies
            await self.base_manager.stop_all_strategies()

            self.logger.info("All strategies and risk monitoring stopped")

        except Exception as e:
            self.logger.error(f"Error stopping strategies: {e}")

    def update_risk_limits(self, new_limits: RiskLimits):
        """Update risk limits."""
        try:
            self.risk_manager.risk_limits = new_limits
            self.config.risk_limits = new_limits

            self.logger.info(f"Risk limits updated: {new_limits}")

        except Exception as e:
            self.logger.error(f"Error updating risk limits: {e}")

    def enable_risk_management(self, strategy_name: str | None = None):
        """Enable risk management for specific strategy or all strategies."""
        try:
            if strategy_name:
                if strategy_name in self.risk_aware_strategies:
                    self.risk_aware_strategies[strategy_name].enable_risk_management()
                    self.logger.info(f"Risk management enabled for {strategy_name}")
                else:
                    self.logger.warning(f"Strategy {strategy_name} not found")
            else:
                for strategy in self.risk_aware_strategies.values():
                    strategy.enable_risk_management()
                self.logger.info("Risk management enabled for all strategies")

        except Exception as e:
            self.logger.error(f"Error enabling risk management: {e}")

    def disable_risk_management(self, strategy_name: str | None = None):
        """Disable risk management for specific strategy or all strategies."""
        try:
            if strategy_name:
                if strategy_name in self.risk_aware_strategies:
                    self.risk_aware_strategies[strategy_name].disable_risk_management()
                    self.logger.info(f"Risk management disabled for {strategy_name}")
                else:
                    self.logger.warning(f"Strategy {strategy_name} not found")
            else:
                for strategy in self.risk_aware_strategies.values():
                    strategy.disable_risk_management()
                self.logger.info("Risk management disabled for all strategies")

        except Exception as e:
            self.logger.error(f"Error disabling risk management: {e}")
