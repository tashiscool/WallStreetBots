"""Risk - Aware Strategy Wrapper
Wraps existing strategies with sophisticated risk management.

This module provides risk - aware versions of all trading strategies:
- Real - time risk assessment before trades
- Automated position sizing based on risk limits
- Risk alerts and monitoring integration
- Cross - strategy risk coordination

Month 3 - 4: Integration with WallStreetBots
"""

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

from .risk_integration_manager import RiskIntegrationManager


class RiskAwareStrategy(ABC):
    """Abstract base class for risk - aware strategies.

    All strategies should inherit from this to get risk management integration
    """

    def __init__(
        self, strategy_instance: Any, risk_manager: RiskIntegrationManager, strategy_name: str
    ):
        """Initialize risk - aware strategy wrapper.

        Args:
            strategy_instance: The actual strategy instance
            risk_manager: Risk integration manager
            strategy_name: Name of the strategy
        """
        self.strategy = strategy_instance
        self.risk_manager = risk_manager
        self.strategy_name = strategy_name
        self.logger = logging.getLogger(f"{__name__}.{strategy_name}")

        # Risk state
        self.risk_enabled = True
        self.last_risk_check = None
        self.risk_violations = 0
        self.total_trades_blocked = 0

        self.logger.info(f"Risk - aware wrapper initialized for {strategy_name}")

    async def execute_trade(
        self,
        symbol: str,
        action: str,  # 'buy', 'sell', 'hold'
        quantity: float,
        price: float | None = None,
        order_type: str = "market",
        **kwargs,
    ) -> dict[str, Any]:
        """Execute a trade with risk management.

        Args:
            symbol: Symbol to trade
            action: Trade action
            quantity: Quantity to trade
            price: Price (for limit orders)
            order_type: Type of order
            **kwargs: Additional strategy - specific parameters

        Returns:
            Dict: Trade execution result
        """
        try:
            # Skip risk checks for hold actions
            if action == "hold":
                return await self._execute_hold_action(symbol, **kwargs)

            # Get current portfolio value
            portfolio_value = await self._get_portfolio_value()

            # Calculate trade value
            trade_value = quantity * (price or await self._get_current_price(symbol))

            # Check if trade is allowed
            allowed, reason = await self.risk_manager.should_allow_trade(
                self.strategy_name, symbol, trade_value, portfolio_value
            )

            if not allowed:
                self.total_trades_blocked += 1
                self.logger.warning(f"Trade blocked for {symbol}: {reason}")
                return {
                    "success": False,
                    "reason": f"Risk management: {reason}",
                    "action": action,
                    "symbol": symbol,
                    "quantity": quantity,
                    "blocked_by_risk": True,
                }

            # Get risk - adjusted position size
            if self.risk_enabled:
                risk_adjusted_quantity = await self.risk_manager.get_risk_adjusted_position_size(
                    self.strategy_name, symbol, quantity, portfolio_value
                )

                # Update quantity if risk - adjusted
                if risk_adjusted_quantity != quantity:
                    self.logger.info(
                        f"Risk - adjusted quantity for {symbol}: {quantity} - >  {risk_adjusted_quantity}"
                    )
                    quantity = risk_adjusted_quantity
                    trade_value = quantity * (price or await self._get_current_price(symbol))

            # Execute the actual trade
            trade_result = await self._execute_actual_trade(
                symbol, action, quantity, price, order_type, **kwargs
            )

            # Update risk metrics after trade
            if trade_result.get("success", False):
                await self._update_risk_after_trade(symbol, action, quantity, trade_value)

            return trade_result

        except Exception as e:
            self.logger.error(f"Error executing risk - aware trade: {e}")
            return {
                "success": False,
                "reason": f"Error: {e}",
                "action": action,
                "symbol": symbol,
                "quantity": quantity,
                "error": True,
            }

    async def _execute_hold_action(self, symbol: str, **kwargs) -> dict[str, Any]:
        """Execute hold action (no risk checks needed)."""
        return {
            "success": True,
            "action": "hold",
            "symbol": symbol,
            "quantity": 0,
            "reason": "Hold action - no risk impact",
        }

    async def _execute_actual_trade(
        self, symbol: str, action: str, quantity: float, price: float, order_type: str, **kwargs
    ) -> dict[str, Any]:
        """Execute the actual trade through the underlying strategy."""
        try:
            # This would call the actual strategy's trade execution method
            # For now, we'll simulate the trade execution
            return {
                "success": True,
                "action": action,
                "symbol": symbol,
                "quantity": quantity,
                "price": price,
                "order_type": order_type,
                "timestamp": datetime.now(),
                "strategy": self.strategy_name,
            }
        except Exception as e:
            self.logger.error(f"Error executing actual trade: {e}")
            return {
                "success": False,
                "reason": f"Trade execution error: {e}",
                "action": action,
                "symbol": symbol,
                "quantity": quantity,
            }

    async def _get_portfolio_value(self) -> float:
        """Get current portfolio value."""
        try:
            # This would get the actual portfolio value from the broker
            # For now, return a simulated value
            return 100000.0  # $100k simulated portfolio
        except Exception as e:
            self.logger.error(f"Error getting portfolio value: {e}")
            return 100000.0

    async def _get_current_price(self, symbol: str) -> float:
        """Get current price for symbol."""
        try:
            # This would get the actual current price
            # For now, return a simulated price
            return 100.0  # $100 simulated price
        except Exception as e:
            self.logger.error(f"Error getting current price for {symbol}: {e}")
            return 100.0

    async def _update_risk_after_trade(
        self, symbol: str, action: str, quantity: float, trade_value: float
    ):
        """Update risk metrics after successful trade."""
        try:
            # Update portfolio positions
            current_positions = self.risk_manager.portfolio_positions.copy()

            if symbol not in current_positions:
                current_positions[symbol] = {"value": 0, "qty": 0}

            # Update position based on action
            if action == "buy":
                current_positions[symbol]["qty"] += quantity
                current_positions[symbol]["value"] += trade_value
            elif action == "sell":
                current_positions[symbol]["qty"] -= quantity
                current_positions[symbol]["value"] -= trade_value

            # Remove zero positions
            if current_positions[symbol]["qty"] == 0:
                del current_positions[symbol]

            # Update risk manager
            self.risk_manager.portfolio_positions = current_positions

            self.logger.info(f"Updated risk metrics after {action} {quantity} {symbol}")

        except Exception as e:
            self.logger.error(f"Error updating risk after trade: {e}")

    async def get_risk_status(self) -> dict[str, Any]:
        """Get current risk status for this strategy."""
        return {
            "strategy_name": self.strategy_name,
            "risk_enabled": self.risk_enabled,
            "last_risk_check": self.last_risk_check,
            "risk_violations": self.risk_violations,
            "total_trades_blocked": self.total_trades_blocked,
            "current_risk_metrics": self.risk_manager.current_metrics,
        }

    def enable_risk_management(self):
        """Enable risk management for this strategy."""
        self.risk_enabled = True
        self.logger.info(f"Risk management enabled for {self.strategy_name}")

    def disable_risk_management(self):
        """Disable risk management for this strategy."""
        self.risk_enabled = False
        self.logger.warning(f"Risk management disabled for {self.strategy_name}")

    @abstractmethod
    async def analyze_market(self, symbol: str, market_data: dict[str, Any]) -> dict[str, Any]:
        """Analyze market conditions (to be implemented by specific strategies)."""
        pass

    @abstractmethod
    async def generate_signals(
        self, symbol: str, market_data: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Generate trading signals (to be implemented by specific strategies)."""
        pass


class RiskAwareWSBDipBot(RiskAwareStrategy):
    """Risk - aware version of WSB Dip Bot strategy."""

    def __init__(self, strategy_instance: Any, risk_manager: RiskIntegrationManager):
        super().__init__(strategy_instance, risk_manager, "wsb_dip_bot")

    async def analyze_market(self, symbol: str, market_data: dict[str, Any]) -> dict[str, Any]:
        """Analyze market for dip opportunities with risk assessment."""
        try:
            # Get base analysis from underlying strategy
            base_analysis = await self.strategy.analyze_market(symbol, market_data)

            # Add risk assessment
            risk_assessment = await self._assess_dip_risk(symbol, market_data)

            return {**base_analysis, "risk_assessment": risk_assessment, "risk_adjusted": True}

        except Exception as e:
            self.logger.error(f"Error in risk - aware market analysis: {e}")
            return {"error": str(e), "risk_adjusted": True}

    async def generate_signals(
        self, symbol: str, market_data: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Generate dip signals with risk management."""
        try:
            # Get base signals from underlying strategy
            base_signals = await self.strategy.generate_signals(symbol, market_data)

            # Filter signals based on risk
            risk_filtered_signals = []
            for signal in base_signals:
                if await self._is_signal_risk_acceptable(signal, symbol, market_data):
                    risk_filtered_signals.append(signal)
                else:
                    self.logger.info(f"Signal filtered out due to risk: {signal}")

            return risk_filtered_signals

        except Exception as e:
            self.logger.error(f"Error generating risk - aware signals: {e}")
            return []

    async def _assess_dip_risk(self, symbol: str, market_data: dict[str, Any]) -> dict[str, Any]:
        """Assess risk specific to dip trading."""
        try:
            # Get current risk metrics
            risk_metrics = self.risk_manager.current_metrics

            # Assess dip - specific risks
            dip_risk_score = 0.0
            risk_factors = []

            # High volatility risk
            if risk_metrics.ml_risk_score > 0.7:
                dip_risk_score += 0.3
                risk_factors.append("High ML risk score")

            # Concentration risk
            if risk_metrics.concentration_risk > 0.2:
                dip_risk_score += 0.2
                risk_factors.append("High concentration risk")

            # Portfolio VaR risk
            if risk_metrics.portfolio_var > 0.03:
                dip_risk_score += 0.3
                risk_factors.append("High portfolio VaR")

            # Stress test risk
            if risk_metrics.stress_test_score > 0.05:
                dip_risk_score += 0.2
                risk_factors.append("High stress test score")

            return {
                "dip_risk_score": dip_risk_score,
                "risk_factors": risk_factors,
                "acceptable_for_dip": dip_risk_score < 0.5,
            }

        except Exception as e:
            self.logger.error(f"Error assessing dip risk: {e}")
            return {"dip_risk_score": 1.0, "risk_factors": ["Error"], "acceptable_for_dip": False}

    async def _is_signal_risk_acceptable(
        self, signal: dict[str, Any], symbol: str, market_data: dict[str, Any]
    ) -> bool:
        """Check if a signal is acceptable from a risk perspective."""
        try:
            # Get signal details
            action = signal.get("action", "hold")
            quantity = signal.get("quantity", 0)

            if action == "hold":
                return True

            # Get current portfolio value
            portfolio_value = await self._get_portfolio_value()
            trade_value = quantity * await self._get_current_price(symbol)

            # Check if trade is allowed
            allowed, reason = await self.risk_manager.should_allow_trade(
                self.strategy_name, symbol, trade_value, portfolio_value
            )

            return allowed

        except Exception as e:
            self.logger.error(f"Error checking signal risk acceptability: {e}")
            return False


class RiskAwareEarningsProtection(RiskAwareStrategy):
    """Risk - aware version of Earnings Protection strategy."""

    def __init__(self, strategy_instance: Any, risk_manager: RiskIntegrationManager):
        super().__init__(strategy_instance, risk_manager, "earnings_protection")

    async def analyze_market(self, symbol: str, market_data: dict[str, Any]) -> dict[str, Any]:
        """Analyze market for earnings opportunities with risk assessment."""
        try:
            # Get base analysis from underlying strategy
            base_analysis = await self.strategy.analyze_market(symbol, market_data)

            # Add earnings - specific risk assessment
            earnings_risk = await self._assess_earnings_risk(symbol, market_data)

            return {**base_analysis, "earnings_risk": earnings_risk, "risk_adjusted": True}

        except Exception as e:
            self.logger.error(f"Error in risk - aware earnings analysis: {e}")
            return {"error": str(e), "risk_adjusted": True}

    async def generate_signals(
        self, symbol: str, market_data: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Generate earnings protection signals with risk management."""
        try:
            # Get base signals from underlying strategy
            base_signals = await self.strategy.generate_signals(symbol, market_data)

            # Filter signals based on earnings risk
            risk_filtered_signals = []
            for signal in base_signals:
                if await self._is_earnings_signal_risk_acceptable(signal, symbol, market_data):
                    risk_filtered_signals.append(signal)
                else:
                    self.logger.info(f"Earnings signal filtered out due to risk: {signal}")

            return risk_filtered_signals

        except Exception as e:
            self.logger.error(f"Error generating risk - aware earnings signals: {e}")
            return []

    async def _assess_earnings_risk(
        self, symbol: str, market_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Assess risk specific to earnings trading."""
        try:
            # Get current risk metrics
            risk_metrics = self.risk_manager.current_metrics

            # Assess earnings - specific risks
            earnings_risk_score = 0.0
            risk_factors = []

            # IV risk (high IV can be dangerous around earnings)
            if "iv_percentile" in market_data:
                iv_percentile = market_data["iv_percentile"]
                if iv_percentile > 80:
                    earnings_risk_score += 0.4
                    risk_factors.append("Very high IV percentile")
                elif iv_percentile > 60:
                    earnings_risk_score += 0.2
                    risk_factors.append("High IV percentile")

            # Portfolio risk
            if risk_metrics.portfolio_var > 0.04:
                earnings_risk_score += 0.3
                risk_factors.append("High portfolio VaR")

            # ML risk
            if risk_metrics.ml_risk_score > 0.6:
                earnings_risk_score += 0.3
                risk_factors.append("High ML risk score")

            return {
                "earnings_risk_score": earnings_risk_score,
                "risk_factors": risk_factors,
                "acceptable_for_earnings": earnings_risk_score < 0.6,
            }

        except Exception as e:
            self.logger.error(f"Error assessing earnings risk: {e}")
            return {
                "earnings_risk_score": 1.0,
                "risk_factors": ["Error"],
                "acceptable_for_earnings": False,
            }

    async def _is_earnings_signal_risk_acceptable(
        self, signal: dict[str, Any], symbol: str, market_data: dict[str, Any]
    ) -> bool:
        """Check if an earnings signal is acceptable from a risk perspective."""
        try:
            # Get signal details
            action = signal.get("action", "hold")
            quantity = signal.get("quantity", 0)

            if action == "hold":
                return True

            # Get current portfolio value
            portfolio_value = await self._get_portfolio_value()
            trade_value = quantity * await self._get_current_price(symbol)

            # Check if trade is allowed
            allowed, reason = await self.risk_manager.should_allow_trade(
                self.strategy_name, symbol, trade_value, portfolio_value
            )

            return allowed

        except Exception as e:
            self.logger.error(f"Error checking earnings signal risk acceptability: {e}")
            return False


def create_risk_aware_strategy(
    strategy_instance: Any, risk_manager: RiskIntegrationManager, strategy_name: str
) -> RiskAwareStrategy:
    """Factory function to create risk - aware strategy wrappers.

    Args:
        strategy_instance: The actual strategy instance
        risk_manager: Risk integration manager
        strategy_name: Name of the strategy

    Returns:
        RiskAwareStrategy: Risk - aware strategy wrapper
    """
    if strategy_name == "wsb_dip_bot":
        return RiskAwareWSBDipBot(strategy_instance, risk_manager)
    elif strategy_name == "earnings_protection":
        return RiskAwareEarningsProtection(strategy_instance, risk_manager)
    else:
        # Generic risk - aware wrapper for other strategies
        return RiskAwareStrategy(strategy_instance, risk_manager, strategy_name)
