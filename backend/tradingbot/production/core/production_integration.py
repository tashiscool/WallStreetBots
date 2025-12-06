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
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any

from asgiref.sync import sync_to_async

from ...alert_system import AlertPriority, AlertType, TradingAlertSystem
from ...apimanagers import AlpacaManager
from ...core.trading_interface import (
    OrderSide,
    OrderType,
    TradeResult,
    TradeSignal,
    TradeStatus,
)
# Django models imported lazily when needed
def get_django_models():
    """Lazy import Django models."""
    from ...models.models import Company, Order, Stock
    return Company, Order, Stock
from ...risk_management import RiskManager, RiskParameters


@dataclass
class ProductionTradeSignal(TradeSignal):
    """Extended TradeSignal for production use."""

    price: float = 0.0
    trade_type: str = "stock"
    risk_amount: Decimal = Decimal("0.00")
    expected_return: Decimal = Decimal("0.00")
    metadata: dict[str, Any] = field(default_factory=dict)

    def __init__(self, symbol=None, action=None, **kwargs):
        """Initialize with backward compatibility for symbol/action."""
        from ...core.trading_interface import OrderSide
        
        # Handle backward compatibility
        if symbol is not None:
            kwargs['ticker'] = symbol
        if action is not None:
            kwargs['side'] = OrderSide.BUY if action.lower() == 'buy' else OrderSide.SELL
        
        # Set defaults for required fields
        kwargs.setdefault('strategy_name', 'manual_trade')
        kwargs.setdefault('ticker', 'UNKNOWN')
        kwargs.setdefault('side', OrderSide.BUY)
        kwargs.setdefault('order_type', OrderType.MARKET)
        kwargs.setdefault('quantity', 0)
        
        # Extract ProductionTradeSignal-specific parameters
        price = kwargs.pop('price', 0.0)
        trade_type = kwargs.pop('trade_type', 'stock')
        risk_amount = kwargs.pop('risk_amount', Decimal('0.00'))
        expected_return = kwargs.pop('expected_return', Decimal('0.00'))
        metadata = kwargs.pop('metadata', {})
        
        # Initialize base class
        super().__init__(**kwargs)
        
        # Set ProductionTradeSignal-specific attributes
        self.price = price
        self.trade_type = trade_type
        self.risk_amount = risk_amount
        self.expected_return = expected_return
        self.metadata = metadata

    @property
    def symbol(self) -> str:
        """Alias for ticker for backward compatibility."""
        return self.ticker

    @property
    def action(self) -> str:
        """Alias for side for backward compatibility."""
        return self.side.value.lower()


@dataclass
class ProductionTrade:
    """Production trade record with full integration."""

    id: str | None = None
    strategy_name: str = ""
    ticker: str = ""
    trade_type: str = "pending"  # 'stock', 'option', 'spread'
    action: str = "buy"  # 'buy', 'sell', 'open', 'close'
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


    @property
    def symbol(self) -> str:
        """Alias for ticker for backward compatibility."""
        return self.ticker

    @property
    def side(self) -> str:
        """Alias for action for backward compatibility."""
        return self.action

    @property
    def price(self) -> float:
        """Alias for entry_price for backward compatibility."""
        return float(self.entry_price)

    @property
    def status(self) -> str:
        """Alias for trade_type for backward compatibility."""
        return self.trade_type

    @property
    def timestamp(self) -> datetime:
        """Alias for fill_timestamp for backward compatibility."""
        return self.fill_timestamp or self.created_at
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
        self.alpaca_manager = AlpacaManager(
            alpaca_api_key, alpaca_secret_key, paper_trading
        )
        self.risk_manager = RiskManager(RiskParameters())
        self.alert_system = TradingAlertSystem()
        self.user_id = user_id
        self.logger = logging.getLogger(__name__)

        # Active trades and positions
        self.active_trades: dict[str, ProductionTrade] = {}
        self.active_positions: dict[str, ProductionPosition] = {}
        
        # Legacy attributes for backward compatibility
        self.trades: list = []
        self.positions: dict = {}

        # Setup logging
        self.setup_logging()

        self.logger.info("ProductionIntegrationManager initialized")

    def setup_logging(self):
        """Setup structured logging."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler("production_integration.log"),
                logging.StreamHandler(),
            ],
        )

    async def execute_trade(self, signal: ProductionTradeSignal, validation_result: dict[str, Any] | None = None) -> TradeResult:
        """Execute trade with full production integration.

        Steps:
        1. Validate signal quality (if validation result provided)
        2. Validate signal and risk limits
        3. Calculate expected slippage
        4. Execute via AlpacaManager
        5. Calculate actual slippage and costs
        6. Create Django Order record
        7. Update position tracking with costs
        8. Send alerts
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

            # 0. Signal validation check (if validation result provided)
            if validation_result:
                recommended_action = validation_result.get('recommended_action', 'execute')
                strength_score = validation_result.get('strength_score', 0)
                min_strength_threshold = validation_result.get('min_strength_threshold', 50.0)
                
                if recommended_action != 'execute':
                    self.logger.warning(
                        f"Signal validation rejected trade: {recommended_action}, "
                        f"strength_score: {strength_score:.1f}"
                    )
                    return TradeResult(
                        trade_id=trade_id,
                        signal=signal,
                        status=TradeStatus.REJECTED,
                        error_message=f"Signal validation failed: {recommended_action}",
                    )
                
                if strength_score < min_strength_threshold:
                    self.logger.warning(
                        f"Signal strength too low: {strength_score:.1f} < {min_strength_threshold:.1f}"
                    )
                    return TradeResult(
                        trade_id=trade_id,
                        signal=signal,
                        status=TradeStatus.REJECTED,
                        error_message=f"Signal strength below threshold: {strength_score:.1f}",
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

            # 2. Calculate expected slippage before execution
            expected_slippage = await self._calculate_expected_slippage(signal)
            expected_price = Decimal(str(signal.price)) + expected_slippage if signal.side.value == "buy" else Decimal(str(signal.price)) - expected_slippage

            # 3. Execute via AlpacaManager
            alpaca_result = await self.execute_alpaca_order(signal)
            if not alpaca_result["success"]:
                return TradeResult(
                    trade_id=trade_id,
                    signal=signal,
                    status=TradeStatus.REJECTED,
                    error_message=f"Alpaca execution failed: {alpaca_result['error']}",
                )

            # 4. Calculate actual slippage and total costs
            fill_price = Decimal(str(alpaca_result["fill_price"]))
            actual_slippage = abs(fill_price - Decimal(str(signal.price))) * Decimal(str(signal.quantity))
            commission = Decimal(str(alpaca_result.get("commission", 0.0)))
            total_cost = (fill_price * Decimal(str(signal.quantity))) + commission + actual_slippage
            
            # Calculate net expected return after costs
            net_expected_return = signal.expected_return - commission - actual_slippage

            # 5. Create Django Order record
            django_order = await self.create_django_order(
                signal, alpaca_result["order_id"]
            )

            # 6. Create ProductionTrade record with full cost tracking
            production_trade = ProductionTrade(
                id=trade_id,
                strategy_name=signal.strategy_name,
                ticker=signal.ticker,
                trade_type=signal.trade_type,
                action=signal.side.value,
                quantity=signal.quantity,
                entry_price=fill_price,
                alpaca_order_id=alpaca_result["order_id"],
                django_order_id=django_order.id if django_order else None,
                fill_timestamp=datetime.now(),
                risk_amount=Decimal(str(signal.risk_amount)),
                expected_return=net_expected_return,  # Net return after costs
                commission=commission,
                slippage=actual_slippage,
                metadata={
                    **signal.metadata,
                    "expected_slippage": float(expected_slippage),
                    "actual_slippage": float(actual_slippage),
                    "total_cost": float(total_cost),
                    "validation_result": validation_result if validation_result else None,
                },
            )

            # 7. Update position tracking
            await self.update_position_tracking(production_trade)

            # 6. Store trade
            self.active_trades[trade_id] = production_trade
            self.trades.append(production_trade)

            # 9. Send success alert with cost information
            await self.alert_system.send_alert(
                AlertType.ENTRY_SIGNAL,
                AlertPriority.MEDIUM,
                f"Trade executed: {signal.ticker} {signal.side.value} {signal.quantity} @ {fill_price} "
                f"(Costs: ${commission:.2f} commission, ${actual_slippage:.2f} slippage)",
            )

            return TradeResult(
                trade_id=trade_id,
                signal=signal,
                status=TradeStatus.FILLED,
                filled_price=float(fill_price),
                commission=float(commission),
            )

        except Exception as e:
            self.logger.error(f"Error executing trade {trade_id}: {e}")
            await self.alert_system.send_alert(
                AlertType.SYSTEM_ERROR,
                AlertPriority.HIGH,
                f"Trade execution error: {e}",
            )
            return TradeResult(
                trade_id=trade_id,
                signal=signal,
                status=TradeStatus.REJECTED,
                error_message=str(e),
            )

    async def validate_risk_limits(
        self, signal: ProductionTradeSignal
    ) -> dict[str, Any]:
        """Validate risk limits before execution."""
        try:
            # Get current portfolio value
            portfolio_value = await self.get_portfolio_value()

            # Calculate position risk
            float(signal.risk_amount) / float(
                portfolio_value
            ) if portfolio_value > 0 else 0

            # Check individual position limit
            current_position_value = await self.get_position_value(signal.ticker)
            # Allow higher limits for index ETFs (SPY, VTI, QQQ, etc.) and lower - risk strategies
            if signal.ticker in [
                "SPY",
                "VTI",
                "QQQ",
                "IWM",
                "DIA",
                "VOO",
                "VTI",
                "VTEB",
            ]:
                max_position_value = (
                    float(portfolio_value) * 0.80
                )  # 80% for broad market ETFs
            else:
                max_position_value = (
                    float(portfolio_value) * 0.20
                )  # 20% for individual stocks

            if (
                float(current_position_value) + float(signal.risk_amount)
                > max_position_value
            ):
                return {
                    "allowed": False,
                    "reason": f"Position limit exceeded for {signal.ticker}",
                }

            # Check total risk limit (max 50% of portfolio)
            total_risk = await self.get_total_risk()
            if (
                float(total_risk) + float(signal.risk_amount)
                > float(portfolio_value) * 0.50
            ):
                return {"allowed": False, "reason": "Total risk limit exceeded"}

            return {"allowed": True, "reason": "Risk limits OK"}

        except Exception as e:
            self.logger.error(f"Risk validation error: {e}")
            return {"allowed": False, "reason": f"Risk validation error: {e}"}

    async def execute_alpaca_order(
        self, signal: ProductionTradeSignal
    ) -> dict[str, Any]:
        """Execute order via AlpacaManager."""
        try:
            if signal.side == OrderSide.BUY:
                result = self.alpaca_manager.market_buy(
                    symbol=signal.ticker, quantity=signal.quantity
                )
            else:
                result = self.alpaca_manager.market_sell(
                    symbol=signal.ticker, quantity=signal.quantity
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
    ):
        """Create Django Order record."""
        try:
            # Get Django models lazily
            Company, Order, Stock = get_django_models()

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

    async def get_all_positions(self) -> dict[str, Any]:
        """Get all current positions as a dictionary."""
        try:
            positions_dict = {}
            # First check active_positions
            for key, position in self.active_positions.items():
                ticker = position.ticker
                current_price = await self.get_current_price(ticker)
                positions_dict[ticker] = {
                    "quantity": float(position.quantity),
                    "entry_price": float(position.entry_price),
                    "current_price": float(current_price),
                    "unrealized_pnl": float(position.unrealized_pnl),
                    "strategy_name": position.strategy_name,
                    "position_type": position.position_type,
                }
            # Also check Alpaca positions for completeness
            alpaca_positions = self.alpaca_manager.get_positions()
            for position in alpaca_positions:
                ticker = position.get("symbol")
                if ticker and ticker not in positions_dict:
                    positions_dict[ticker] = {
                        "quantity": float(position.get("qty", 0)),
                        "entry_price": float(position.get("avg_entry_price", 0)),
                        "current_price": 0.0,  # Will be updated if needed
                        "unrealized_pnl": float(position.get("unrealized_pl", 0)),
                        "strategy_name": "unknown",
                        "position_type": position.get("side", "long"),
                    }
            return positions_dict
        except Exception as e:
            self.logger.error(f"Error getting all positions: {e}")
            return {}

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
                    (
                        position.position_type == "long"
                        and current_price <= position.stop_loss
                    )
                    or (
                        position.position_type == "short"
                        and current_price >= position.stop_loss
                    )
                ):
                    await self.execute_exit_trade(position, "stop_loss")

                # Check take profit
                if position.take_profit and (
                    (
                        position.position_type == "long"
                        and current_price >= position.take_profit
                    )
                    or (
                        position.position_type == "short"
                        and current_price <= position.take_profit
                    )
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

    async def get_portfolio_history(self, days: int = 180) -> list[dict[str, Any]]:
        """Get portfolio value history for analytics.
        
        Returns a list of dictionaries with 'timestamp' and 'value' keys.
        For now, returns current value as a single entry since we don't have historical tracking.
        In production, this would query a database or Alpaca's portfolio history API.
        """
        try:
            current_value = await self.get_portfolio_value()
            return [
                {
                    "timestamp": datetime.now() - timedelta(days=days - i),
                    "value": float(current_value)
                }
                for i in range(min(days, 30))  # Return up to 30 days of synthetic data
            ]
        except Exception as e:
            self.logger.error(f"Error getting portfolio history: {e}")
            return []

    async def _calculate_expected_slippage(self, signal: ProductionTradeSignal) -> Decimal:
        """Calculate expected slippage for a trade signal.
        
        Uses simple model based on order size, volatility, and liquidity.
        In production, would use SlippagePredictor for more accurate estimates.
        """
        try:
            from backend.validation.execution_reality.slippage_calibration import SlippagePredictor
            
            # Try to use advanced slippage model if available
            if not hasattr(self, '_advanced_slippage_model'):
                try:
                    from ..execution.advanced_slippage_model import AdvancedSlippageModel
                    self._advanced_slippage_model = AdvancedSlippageModel(model_type='ensemble')
                except ImportError:
                    self._advanced_slippage_model = None
            
            # Use advanced model if available and trained
            if self._advanced_slippage_model and self._advanced_slippage_model.is_trained:
                try:
                    from ..execution.advanced_slippage_model import MarketMicrostructureFeatures
                    
                    # Try to get microstructure features (simplified for now)
                    microstructure = MarketMicrostructureFeatures(
                        bid_ask_spread=0.001,  # Would get from order book
                        order_book_imbalance=0.0,
                        volume_profile=0.5,
                        volatility=market_conditions.get('volatility', 0.20),
                        time_of_day=0.5,  # Would calculate from current time
                        day_of_week=datetime.now().weekday(),
                        recent_volume=market_conditions.get('volume', 1000000),
                        price_momentum=0.0,
                        liquidity_score=0.5
                    )
                    
                    prediction = self._advanced_slippage_model.predict_slippage(
                        symbol=signal.ticker,
                        side=signal.side.value,
                        quantity=signal.quantity,
                        market_conditions=market_conditions,
                        microstructure_features=microstructure
                    )
                    
                    # Convert from basis points to dollar amount
                    slippage_bps = prediction.expected_slippage_bps
                    slippage_pct = slippage_bps / 10000.0
                    expected_slippage = Decimal(str(signal.price)) * Decimal(str(slippage_pct))
                    
                    return expected_slippage
                except Exception as e:
                    self.logger.warning(f"Error using advanced slippage model: {e}")
            
            # Fallback to original slippage predictor
            if not hasattr(self, '_slippage_predictor'):
                self._slippage_predictor = SlippagePredictor()
            
            # Get current market conditions
            current_price = await self.get_current_price(signal.ticker)
            market_conditions = {
                'price': float(current_price),
                'volume': 1000000,  # Default volume estimate
                'volatility': 0.20,  # Default 20% volatility
            }
            
            # Predict slippage
            slippage_pred = self._slippage_predictor.predict_slippage(
                symbol=signal.ticker,
                side=signal.side.value,
                quantity=signal.quantity,
                market_conditions=market_conditions
            )
            
            # Convert from basis points to dollar amount
            slippage_bps = slippage_pred.get('expected_slippage_bps', 5.0)  # Default 5 bps
            slippage_pct = slippage_bps / 10000.0
            expected_slippage = Decimal(str(signal.price)) * Decimal(str(slippage_pct))
            
            return expected_slippage
            
        except Exception as e:
            self.logger.warning(f"Error calculating slippage, using default: {e}")
            # Default slippage: 0.05% (5 basis points) for stocks, 0.1% for options
            default_slippage_pct = 0.0005 if signal.trade_type == "stock" else 0.001
            return Decimal(str(signal.price)) * Decimal(str(default_slippage_pct))

    async def execute_exit_trade(self, position: ProductionPosition, reason: str):
        """Execute exit trade for position with proper PnL calculation including costs."""
        try:
            # Create exit signal
            exit_signal = ProductionTradeSignal(
                strategy_name=position.strategy_name,
                ticker=position.ticker,
                side=OrderSide.SELL
                if position.position_type == "long"
                else OrderSide.BUY,
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
                # Find the entry trade(s) to calculate realized PnL with costs
                entry_trades = [
                    trade for trade in self.trades
                    if (trade.ticker == position.ticker and 
                        trade.strategy_name == position.strategy_name and
                        trade.action == "buy" and
                        trade.exit_price is None)
                ]
                
                if entry_trades:
                    # Calculate total entry cost (including all costs)
                    total_entry_cost = Decimal("0.00")
                    total_entry_quantity = 0
                    for trade in entry_trades:
                        entry_cost = (trade.entry_price * Decimal(str(trade.quantity)) + 
                                     trade.commission + trade.slippage)
                        total_entry_cost += entry_cost
                        total_entry_quantity += trade.quantity
                    
                    # Calculate exit proceeds (after costs)
                    exit_price = Decimal(str(result.filled_price))
                    exit_slippage = abs(exit_price - Decimal(str(exit_signal.price))) * Decimal(str(position.quantity))
                    exit_commission = Decimal(str(result.commission))
                    exit_proceeds = (exit_price * Decimal(str(position.quantity)) - 
                                    exit_commission - exit_slippage)
                    
                    # Calculate realized PnL
                    avg_entry_cost = total_entry_cost / Decimal(str(total_entry_quantity)) if total_entry_quantity > 0 else Decimal("0.00")
                    realized_pnl = exit_proceeds - (avg_entry_cost * Decimal(str(position.quantity)))
                    
                    # Update entry trades with exit information
                    for trade in entry_trades:
                        trade.exit_price = exit_price
                        trade.exit_timestamp = datetime.now()
                        if trade.pnl is None:
                            # Allocate PnL proportionally
                            trade_pnl = realized_pnl * (Decimal(str(trade.quantity)) / Decimal(str(position.quantity)))
                            trade.pnl = trade_pnl
                            trade.win = trade_pnl > 0
                            trade.actual_return = trade_pnl
                    
                    # Update position
                    position.realized_pnl += realized_pnl
                    position.unrealized_pnl = Decimal("0.00")
                    
                    # Remove from active positions
                    position_key = f"{position.ticker}_{position.strategy_name}"
                    if position_key in self.active_positions:
                        del self.active_positions[position_key]
                    
                    # Send exit alert with detailed PnL
                    await self.alert_system.send_alert(
                        AlertType.PROFIT_TARGET if realized_pnl > 0 else AlertType.STOP_LOSS,
                        AlertPriority.MEDIUM,
                        f"Position closed: {position.ticker} {reason} - "
                        f"Realized PnL: ${realized_pnl:.2f} "
                        f"(Entry: ${avg_entry_cost:.2f}, Exit: ${exit_price:.2f}, "
                        f"Total Costs: ${exit_commission + exit_slippage:.2f})",
                    )
                    
                    self.logger.info(
                        f"Exit trade executed for {position.ticker}: {reason} - "
                        f"Realized PnL: ${realized_pnl:.2f} (after all costs)"
                    )
                else:
                    # Fallback: use unrealized PnL
                    position.realized_pnl += position.unrealized_pnl
                    position.unrealized_pnl = Decimal("0.00")
                    self.logger.warning(f"Could not find entry trades for {position.ticker}, using unrealized PnL")

        except Exception as e:
            self.logger.error(f"Error executing exit trade: {e}")

    def get_portfolio_summary(self) -> dict[str, Any]:
        """Get portfolio summary."""
        try:
            total_positions = len(self.active_positions)
            total_trades = len(self.active_trades)

            total_unrealized_pnl = sum(
                pos.unrealized_pnl for pos in self.active_positions.values()
            )
            total_realized_pnl = sum(
                pos.realized_pnl for pos in self.active_positions.values()
            )

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
    alpaca_api_key: str,
    alpaca_secret_key: str,
    paper_trading: bool = True,
    user_id: int = 1,
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
    return ProductionIntegrationManager(
        alpaca_api_key, alpaca_secret_key, paper_trading, user_id
    )
