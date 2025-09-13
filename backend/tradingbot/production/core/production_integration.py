"""Production Integration Layer
Connects existing infrastructure (AlpacaManager, Django models, strategies) for real trading.

This module bridges the gap between:
- AlpacaManager (broker integration)
- Django models (database persistence)
- Trading strategies (business logic)
- Risk management (position sizing, stop losses)

Making the system production - ready for live trading.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any

from asgiref.sync import sync_to_async

from ...alert_system import AlertPriority, AlertType, TradingAlertSystem
from ...apimanagers import AlpacaManager
from ...core.trading_interface import OrderSide, OrderType, TradeResult, TradeSignal, TradeStatus
from ...models import Company, Order, Stock
from ...risk_management import RiskManager, RiskParameters


@dataclass
class ProductionTradeSignal(TradeSignal):
    """Extended TradeSignal for production use."""

    price: float = 0.0
    trade_type: str = "stock"
    risk_amount: Decimal = Decimal("0.00")
    expected_return: Decimal = Decimal("0.00")
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProductionTrade:
    """Production trade record with full integration."""

    id: str | None = None
    strategy_name: str = ""
    ticker: str = ""
    trade_type: str = ""  # 'stock', 'option', 'spread'
    action: str = ""  # 'buy', 'sell', 'open', 'close'
    quantity: int = 0
    entry_price: Decimal = Decimal("0.00")
    exit_price: Decimal | None = None
    pnl: Decimal | None = None
    commission: Decimal = Decimal("0.00")
    slippage: Decimal = Decimal("0.00")
    alpaca_order_id: str = ""
    django_order_id: int | None = None
    fill_timestamp: datetime | None = None
    exit_timestamp: datetime | None = None
    risk_amount: Decimal = Decimal("0.00")
    expected_return: Decimal = Decimal("0.00")
    actual_return: Decimal | None = None
    win: bool | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class ProductionPosition:
    """Production position with real - time tracking."""

    id: str | None = None
    ticker: str = ""
    strategy_name: str = ""
    position_type: str = ""  # 'long', 'short', 'spread'
    quantity: int = 0
    entry_price: Decimal = Decimal("0.00")
    current_price: Decimal = Decimal("0.00")
    unrealized_pnl: Decimal = Decimal("0.00")
    realized_pnl: Decimal = Decimal("0.00")
    risk_amount: Decimal = Decimal("0.00")
    stop_loss: Decimal | None = None
    take_profit: Decimal | None = None
    alpaca_position_id: str = ""
    django_stock_instance_id: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


class ProductionIntegrationManager:
    """Production Integration Manager.

    Connects all components for real trading:
    - AlpacaManager for broker execution
    - Django models for database persistence
    - Risk management for position sizing
    - Alert system for notifications
    """

    def __init__(
        self,
        alpaca_api_key: str,
        alpaca_secret_key: str,
        paper_trading: bool = True,
        user_id: int = 1,
    ):
        self.alpaca_manager = AlpacaManager(alpaca_api_key, alpaca_secret_key, paper_trading)
        self.risk_manager = RiskManager(RiskParameters())
        self.alert_system = TradingAlertSystem()
        self.user_id = user_id
        self.logger = logging.getLogger(__name__)

        # Active trades and positions
        self.active_trades: dict[str, ProductionTrade] = {}
        self.active_positions: dict[str, ProductionPosition] = {}

        # Setup logging
        self.setup_logging()

        self.logger.info("ProductionIntegrationManager initialized")

    def setup_logging(self):
        """Setup structured logging."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler("production_integration.log"), logging.StreamHandler()],
        )

    async def execute_trade(self, signal: ProductionTradeSignal) -> TradeResult:
        """Execute trade with full production integration.

        Steps:
        1. Validate signal and risk limits
        2. Execute via AlpacaManager
        3. Create Django Order record
        4. Update position tracking
        5. Send alerts
        """
        trade_id = f"{signal.strategy_name}_{signal.ticker}_{datetime.now().strftime('%Y % m % d_ % H % M % S')}"

        try:
            self.logger.info(
                f"Executing production trade {trade_id}",
                extra={
                    "trade_id": trade_id,
                    "strategy": signal.strategy_name,
                    "ticker": signal.ticker,
                    "side": signal.side.value,
                    "quantity": signal.quantity,
                },
            )

            # 1. Risk validation
            risk_check = await self.validate_risk_limits(signal)
            if not risk_check["allowed"]:
                await self.alert_system.send_alert(
                    AlertType.RISK_ALERT,
                    AlertPriority.HIGH,
                    f"Risk limit exceeded for {signal.ticker}: {risk_check['reason']}",
                )
                return TradeResult(
                    trade_id=trade_id,
                    signal=signal,
                    status=TradeStatus.REJECTED,
                    error_message=f"Risk limit exceeded: {risk_check['reason']}",
                )

            # 2. Execute via AlpacaManager
            alpaca_result = await self.execute_alpaca_order(signal)
            if not alpaca_result["success"]:
                return TradeResult(
                    trade_id=trade_id,
                    signal=signal,
                    status=TradeStatus.REJECTED,
                    error_message=f"Alpaca execution failed: {alpaca_result['error']}",
                )

            # 3. Create Django Order record
            django_order = await self.create_django_order(signal, alpaca_result["order_id"])

            # 4. Create ProductionTrade record
            production_trade = ProductionTrade(
                id=trade_id,
                strategy_name=signal.strategy_name,
                ticker=signal.ticker,
                trade_type=signal.trade_type,
                action=signal.side.value,
                quantity=signal.quantity,
                entry_price=Decimal(str(alpaca_result["fill_price"])),
                alpaca_order_id=alpaca_result["order_id"],
                django_order_id=django_order.id if django_order else None,
                fill_timestamp=datetime.now(),
                risk_amount=Decimal(str(signal.risk_amount)),
                expected_return=Decimal(str(signal.expected_return)),
                metadata=signal.metadata,
            )

            # 5. Update position tracking
            await self.update_position_tracking(production_trade)

            # 6. Store trade
            self.active_trades[trade_id] = production_trade

            # 7. Send success alert
            await self.alert_system.send_alert(
                AlertType.ENTRY_SIGNAL,
                AlertPriority.MEDIUM,
                f"Trade executed: {signal.ticker} {signal.side.value} {signal.quantity} @ {alpaca_result['fill_price']}",
            )

            return TradeResult(
                trade_id=trade_id,
                signal=signal,
                status=TradeStatus.FILLED,
                filled_price=alpaca_result["fill_price"],
                commission=alpaca_result.get("commission", 0.0),
            )

        except Exception as e:
            self.logger.error(f"Error executing trade {trade_id}: {e}")
            await self.alert_system.send_alert(
                AlertType.SYSTEM_ERROR, AlertPriority.HIGH, f"Trade execution error: {e}"
            )
            return TradeResult(
                trade_id=trade_id, signal=signal, status=TradeStatus.REJECTED, error_message=str(e)
            )

    async def validate_risk_limits(self, signal: ProductionTradeSignal) -> dict[str, Any]:
        """Validate risk limits before execution."""
        try:
            # Get current portfolio value
            portfolio_value = await self.get_portfolio_value()

            # Calculate position risk
            float(signal.risk_amount) / float(portfolio_value) if portfolio_value > 0 else 0

            # Check individual position limit
            current_position_value = await self.get_position_value(signal.ticker)
            # Allow higher limits for index ETFs (SPY, VTI, QQQ, etc.) and lower - risk strategies
            if signal.ticker in ["SPY", "VTI", "QQQ", "IWM", "DIA", "VOO", "VTI", "VTEB"]:
                max_position_value = float(portfolio_value) * 0.80  # 80% for broad market ETFs
            else:
                max_position_value = float(portfolio_value) * 0.20  # 20% for individual stocks

            if float(current_position_value) + float(signal.risk_amount) > max_position_value:
                return {"allowed": False, "reason": f"Position limit exceeded for {signal.ticker}"}

            # Check total risk limit (max 50% of portfolio)
            total_risk = await self.get_total_risk()
            if float(total_risk) + float(signal.risk_amount) > float(portfolio_value) * 0.50:
                return {"allowed": False, "reason": "Total risk limit exceeded"}

            return {"allowed": True, "reason": "Risk limits OK"}

        except Exception as e:
            self.logger.error(f"Risk validation error: {e}")
            return {"allowed": False, "reason": f"Risk validation error: {e}"}

    async def execute_alpaca_order(self, signal: ProductionTradeSignal) -> dict[str, Any]:
        """Execute order via AlpacaManager."""
        try:
            if signal.side == OrderSide.BUY:
                result = self.alpaca_manager.market_buy(
                    symbol=signal.ticker, qty=signal.quantity, time_in_force="day"
                )
            else:
                result = self.alpaca_manager.market_sell(
                    symbol=signal.ticker, qty=signal.quantity, time_in_force="day"
                )

            if result and "id" in result:
                return {
                    "success": True,
                    "order_id": result["id"],
                    "fill_price": result.get("filled_avg_price", signal.price),
                    "commission": result.get("commission", 0.0),
                }
            else:
                return {"success": False, "error": "No order ID returned from Alpaca"}

        except Exception as e:
            self.logger.error(f"Alpaca execution error: {e}")
            return {"success": False, "error": str(e)}

    async def create_django_order(
        self, signal: ProductionTradeSignal, alpaca_order_id: str
    ) -> Order | None:
        """Create Django Order record."""
        try:
            # Get or create Company and Stock
            company, created = await sync_to_async(Company.objects.get_or_create)(
                ticker=signal.ticker, defaults={"name": signal.ticker}
            )

            stock, _created = await sync_to_async(Stock.objects.get_or_create)(
                company=company, defaults={}
            )

            # Create Order
            order = await sync_to_async(Order.objects.create)(
                client_order_id=alpaca_order_id,
                user_id=self.user_id,
                stock=stock,
                order_type="M",  # Market order
                quantity=signal.quantity,
                transaction_type="B" if signal.side == OrderSide.BUY else "S",
                status="F",  # Filled
                filled_avg_price=signal.price,
                filled_timestamp=datetime.now(),
                filled_quantity=signal.quantity,
            )

            self.logger.info(f"Created Django Order {order.id} for {signal.ticker}")
            return order

        except Exception as e:
            self.logger.error(f"Error creating Django Order: {e}")
            return None

    async def update_position_tracking(self, trade: ProductionTrade):
        """Update position tracking after trade execution."""
        try:
            position_key = f"{trade.ticker}_{trade.strategy_name}"

            if position_key in self.active_positions:
                position = self.active_positions[position_key]

                if trade.action == "buy":  # Add to position
                    total_cost = (
                        position.quantity * position.entry_price
                        + trade.quantity * trade.entry_price
                    )
                    total_quantity = position.quantity + trade.quantity
                    position.entry_price = total_cost / total_quantity
                    position.quantity = total_quantity
                else:
                    # Reduce position
                    position.quantity -= trade.quantity
                    if position.quantity <= 0:
                        # Position closed
                        position.realized_pnl += trade.pnl or Decimal("0.00")
                        del self.active_positions[position_key]

                position.updated_at = datetime.now()
            else:
                # Create new position
                position = ProductionPosition(
                    ticker=trade.ticker,
                    strategy_name=trade.strategy_name,
                    position_type="long" if trade.action == "buy" else "short",
                    quantity=trade.quantity,
                    entry_price=trade.entry_price,
                    risk_amount=trade.risk_amount,
                    alpaca_position_id=trade.alpaca_order_id,
                    django_stock_instance_id=trade.django_order_id,
                )
                self.active_positions[position_key] = position

            self.logger.info(f"Updated position tracking for {trade.ticker}")

        except Exception as e:
            self.logger.error(f"Error updating position tracking: {e}")

    async def get_portfolio_value(self) -> Decimal:
        """Get current portfolio value from Alpaca."""
        try:
            account_value = self.alpaca_manager.get_account_value()
            if account_value:
                return Decimal(str(account_value))
            return Decimal("0.00")
        except Exception as e:
            self.logger.error(f"Error getting portfolio value: {e}")
            return Decimal("0.00")

    async def get_position_value(self, ticker: str) -> Decimal:
        """Get current position value for ticker."""
        try:
            positions = self.alpaca_manager.get_positions()
            for position in positions:
                if position.get("symbol") == ticker:
                    return Decimal(str(position.get("market_value", 0)))
            return Decimal("0.00")
        except Exception as e:
            self.logger.error(f"Error getting position value for {ticker}: {e}")
            return Decimal("0.00")

    async def get_total_risk(self) -> Decimal:
        """Get total risk across all positions."""
        try:
            total_risk = Decimal("0.00")
            for position in self.active_positions.values():
                total_risk += position.risk_amount
            return total_risk
        except Exception as e:
            self.logger.error(f"Error calculating total risk: {e}")
            return Decimal("0.00")

    async def monitor_positions(self):
        """Monitor active positions for exit signals."""
        try:
            for _position_key, position in list(self.active_positions.items()):
                # Get current price
                current_price = await self.get_current_price(position.ticker)
                position.current_price = current_price

                # Calculate unrealized P & L
                if position.position_type == "long":
                    position.unrealized_pnl = (
                        current_price - position.entry_price
                    ) * position.quantity
                else:
                    position.unrealized_pnl = (
                        position.entry_price - current_price
                    ) * position.quantity

                # Check stop loss
                if position.stop_loss and (
                    (position.position_type == "long" and current_price <= position.stop_loss)
                    or (position.position_type == "short" and current_price >= position.stop_loss)
                ):
                    await self.execute_exit_trade(position, "stop_loss")

                # Check take profit
                if position.take_profit and (
                    (position.position_type == "long" and current_price >= position.take_profit)
                    or (position.position_type == "short" and current_price <= position.take_profit)
                ):
                    await self.execute_exit_trade(position, "take_profit")

                position.updated_at = datetime.now()

        except Exception as e:
            self.logger.error(f"Error monitoring positions: {e}")

    async def get_current_price(self, ticker: str) -> Decimal:
        """Get current price for ticker."""
        try:
            latest_trade = self.alpaca_manager.get_latest_trade(ticker)
            if latest_trade and "price" in latest_trade:
                return Decimal(str(latest_trade["price"]))
            return Decimal("0.00")
        except Exception as e:
            self.logger.error(f"Error getting current price for {ticker}: {e}")
            return Decimal("0.00")

    async def execute_exit_trade(self, position: ProductionPosition, reason: str):
        """Execute exit trade for position."""
        try:
            # Create exit signal
            exit_signal = ProductionTradeSignal(
                strategy_name=position.strategy_name,
                ticker=position.ticker,
                side=OrderSide.SELL if position.position_type == "long" else OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=position.quantity,
                price=float(position.current_price),
                trade_type="stock",
                risk_amount=Decimal("0.00"),
                expected_return=position.unrealized_pnl,
                metadata={"exit_reason": reason, "position_id": position.id},
            )

            # Execute exit trade
            result = await self.execute_trade(exit_signal)

            if result.status == TradeStatus.FILLED:
                # Update position with realized P & L
                position.realized_pnl += position.unrealized_pnl
                position.unrealized_pnl = Decimal("0.00")

                # Send exit alert
            await self.alert_system.send_alert(
                AlertType.PROFIT_TARGET,
                AlertPriority.MEDIUM,
                f"Position closed: {position.ticker} {reason} P & L: {position.realized_pnl}",
            )

            self.logger.info(f"Exit trade executed for {position.ticker}: {reason}")

        except Exception as e:
            self.logger.error(f"Error executing exit trade: {e}")

    def get_portfolio_summary(self) -> dict[str, Any]:
        """Get portfolio summary."""
        try:
            total_positions = len(self.active_positions)
            total_trades = len(self.active_trades)

            total_unrealized_pnl = sum(pos.unrealized_pnl for pos in self.active_positions.values())
            total_realized_pnl = sum(pos.realized_pnl for pos in self.active_positions.values())

            return {
                "total_positions": total_positions,
                "total_trades": total_trades,
                "total_unrealized_pnl": float(total_unrealized_pnl),
                "total_realized_pnl": float(total_realized_pnl),
                "active_positions": [
                    {
                        "ticker": pos.ticker,
                        "strategy": pos.strategy_name,
                        "quantity": pos.quantity,
                        "entry_price": float(pos.entry_price),
                        "current_price": float(pos.current_price),
                        "unrealized_pnl": float(pos.unrealized_pnl),
                        "risk_amount": float(pos.risk_amount),
                    }
                    for pos in self.active_positions.values()
                ],
            }
        except Exception as e:
            self.logger.error(f"Error getting portfolio summary: {e}")
            return {}


# Factory function for easy initialization
def create_production_integration(
    alpaca_api_key: str, alpaca_secret_key: str, paper_trading: bool = True, user_id: int = 1
) -> ProductionIntegrationManager:
    """Create ProductionIntegrationManager instance.

    Args:
        alpaca_api_key: Alpaca API key
        alpaca_secret_key: Alpaca secret key
        paper_trading: True for paper trading, False for live
        user_id: Django user ID for database records

    Returns:
        ProductionIntegrationManager instance
    """
    return ProductionIntegrationManager(alpaca_api_key, alpaca_secret_key, paper_trading, user_id)
