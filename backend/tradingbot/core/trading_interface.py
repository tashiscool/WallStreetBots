"""Unified Trading Interface
Connects the disconnected Strategy and Broker systems for production use.
"""

from __future__ import annotations

import logging
from dataclasses import InitVar, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from ..alert_system import AlertPriority, AlertType, TradingAlertSystem
from ..apimanagers import AlpacaManager
from ..risk_management import RiskManager, RiskParameters


class TradeStatus(Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


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
    """Signal from a trading strategy.

    Supports canonical fields plus legacy aliases used by older strategy code.
    """

    strategy_name: str = "unknown_strategy"
    ticker: str = "UNKNOWN"
    side: OrderSide = OrderSide.BUY
    order_type: OrderType = OrderType.MARKET
    quantity: int = 0
    limit_price: float | None = None
    stop_price: float | None = None
    time_in_force: str = "gtc"
    reason: str = ""
    confidence: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)

    # Legacy alias inputs (accepted at init, not stored as dataclass fields)
    symbol: InitVar[str | None] = None
    action: InitVar[str | None] = None
    strategy: InitVar[str | None] = None
    price: InitVar[float | None] = None
    signal_type: InitVar[str | None] = None

    def __post_init__(
        self,
        symbol: str | None,
        action: str | None,
        strategy: str | None,
        price: float | None,
        signal_type: str | None,
    ):
        if symbol and self.ticker == "UNKNOWN":
            self.ticker = symbol
        if strategy and self.strategy_name == "unknown_strategy":
            self.strategy_name = strategy

        if isinstance(self.side, str):
            self.side = OrderSide(self.side.lower())
        if isinstance(self.order_type, str):
            self.order_type = OrderType(self.order_type.lower())

        if action:
            action_map = {
                "buy": OrderSide.BUY,
                "sell": OrderSide.SELL,
                "buy_to_close": OrderSide.BUY,
                "sell_to_open": OrderSide.SELL,
                "sell_to_close": OrderSide.SELL,
                "buy_to_open": OrderSide.BUY,
            }
            self.side = action_map.get(str(action).lower(), self.side)
            self.metadata.setdefault("legacy_action", action)

        if price is not None:
            self.metadata.setdefault("price", price)
        if signal_type is not None:
            self.metadata.setdefault("signal_type", signal_type)

    @property
<<<<<<< ours
    def action(self) -> str:  # noqa: F811
        return self.metadata.get("legacy_action", self.side.value)

    @property
    def strategy(self) -> str:  # noqa: F811
=======
    def action(self) -> str:
        return self.metadata.get("legacy_action", self.side.value)

    @property
    def strategy(self) -> str:
>>>>>>> theirs
        return self.strategy_name

    @classmethod
    def from_legacy(cls, **kwargs) -> "TradeSignal":
<<<<<<< ours
        """Create TradeSignal from legacy field names.

        Maps common alternative field names:
        - symbol -> ticker
        - action -> side (mapped to OrderSide)
        - strategy -> strategy_name
        - price -> limit_price
        """
        # Map field names
        if "symbol" in kwargs and "ticker" not in kwargs:
            kwargs["ticker"] = kwargs.pop("symbol")
        if "strategy" in kwargs and "strategy_name" not in kwargs:
            kwargs["strategy_name"] = kwargs.pop("strategy")
        if "action" in kwargs and "side" not in kwargs:
            action = kwargs.pop("action")
            action_map = {
                "BUY": OrderSide.BUY,
                "SELL": OrderSide.SELL,
                "BUY_TO_OPEN": OrderSide.BUY,
                "BUY_TO_CLOSE": OrderSide.BUY,
                "SELL_TO_OPEN": OrderSide.SELL,
                "SELL_TO_CLOSE": OrderSide.SELL,
            }
            kwargs["side"] = action_map.get(action.upper(), OrderSide.BUY)
        if "price" in kwargs and "limit_price" not in kwargs:
            kwargs["limit_price"] = kwargs.pop("price")

        # Remove unknown fields
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in kwargs.items() if k in known_fields}

        return cls(**filtered)
=======
        """Construct from legacy kwargs while keeping the main dataclass init clean."""
        return cls(**kwargs)


>>>>>>> theirs


@dataclass
class TradeResult:
    """Result of trade execution."""

    trade_id: str
    signal: TradeSignal
    status: TradeStatus
    filled_quantity: int = 0
    filled_price: float | None = None
    commission: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    error_message: str | None = None


@dataclass
class PositionUpdate:
    """Position update from broker."""

    ticker: str
    quantity: int
    avg_price: float
    current_price: float
    unrealized_pnl: float
    market_value: float
    timestamp: datetime = field(default_factory=datetime.now)


class TradingInterface:
    """Unified interface connecting strategies to broker execution."""

    def __init__(
        self,
        broker_manager: AlpacaManager,
        risk_manager: RiskManager,
        alert_system: TradingAlertSystem,
        config: dict[str, Any],
    ):
        self.broker = broker_manager
        self.risk = risk_manager
        self.alerts = alert_system
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.active_trades: dict[str, TradeResult] = {}
        self.positions: dict[str, PositionUpdate] = {}

        # Setup logging
        self.setup_logging()

    def setup_logging(self):
        """Setup structured logging."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler("trading_interface.log"),
                logging.StreamHandler(),
            ],
        )

    async def execute_trade(self, signal: TradeSignal) -> TradeResult:
        """Execute trade with comprehensive risk controls and error handling."""
        trade_id = f"{signal.strategy_name}_{signal.ticker}_{datetime.now().strftime('%Y % m % d_ % H % M % S')}"

        try:
            self.logger.info(
                f"Executing trade {trade_id}",
                extra={
                    "trade_id": trade_id,
                    "strategy": signal.strategy_name,
                    "ticker": signal.ticker,
                    "side": signal.side.value,
                    "quantity": signal.quantity,
                },
            )

            # 1. Validate signal
            validation_result = await self.validate_signal(signal)
            if not validation_result["valid"]:
                return TradeResult(
                    trade_id=trade_id,
                    signal=signal,
                    status=TradeStatus.REJECTED,
                    error_message=validation_result["reason"],
                )

            # 2. Check risk limits
            risk_check = await self.check_risk_limits(signal)
            if not risk_check["allowed"]:
                await self.alerts.send_alert(
                    AlertType.RISK_LIMIT_EXCEEDED,
                    AlertPriority.HIGH,
                    f"Risk limit exceeded for {signal.ticker}: {risk_check['reason']}",
                )
                return TradeResult(
                    trade_id=trade_id,
                    signal=signal,
                    status=TradeStatus.REJECTED,
                    error_message=f"Risk limit exceeded: {risk_check['reason']}",
                )

            # 3. Execute via broker
            broker_result = await self.execute_broker_order(signal)

            # 4. Create trade result
            trade_result = TradeResult(
                trade_id=trade_id,
                signal=signal,
                status=TradeStatus.FILLED
                if broker_result["success"]
                else TradeStatus.REJECTED,
                filled_quantity=broker_result.get("filled_quantity", 0),
                filled_price=broker_result.get("filled_price"),
                commission=broker_result.get("commission", 0.0),
                error_message=broker_result.get("error"),
            )

            # 5. Update tracking
            self.active_trades[trade_id] = trade_result

            # 6. Send alerts
            await self.alerts.send_alert(
                AlertType.TRADE_EXECUTED,
                AlertPriority.MEDIUM,
                f"Trade executed: {signal.ticker} {signal.side.value} {signal.quantity} @ {trade_result.filled_price}",
            )

            self.logger.info(
                f"Trade {trade_id} executed successfully",
                extra={
                    "trade_id": trade_id,
                    "filled_quantity": trade_result.filled_quantity,
                    "filled_price": trade_result.filled_price,
                },
            )

            return trade_result

        except Exception as e:
            self.logger.error(
                f"Error executing trade {trade_id}: {e}",
                extra={"trade_id": trade_id, "error": str(e)},
            )

            return TradeResult(
                trade_id=trade_id,
                signal=signal,
                status=TradeStatus.REJECTED,
                error_message=str(e),
            )

    async def validate_signal(self, signal: TradeSignal) -> dict[str, Any]:
        """Validate trading signal."""
        # Check required fields
        if not signal.ticker or not signal.quantity or signal.quantity <= 0:
            return {"valid": False, "reason": "Invalid ticker or quantity"}

        if signal.order_type == OrderType.LIMIT and not signal.limit_price:
            return {"valid": False, "reason": "Limit price required for limit orders"}

        if signal.order_type == OrderType.STOP and not signal.stop_price:
            return {"valid": False, "reason": "Stop price required for stop orders"}

        # Check market hours (simplified)
        if not await self.is_market_open():
            return {"valid": False, "reason": "Market is closed"}

        return {"valid": True, "reason": "Signal validated"}

    async def check_risk_limits(self, signal: TradeSignal) -> dict[str, Any]:
        """Check risk limits before execution."""
        try:
            # Get current account value
            account = await self.get_account_info()
            account_value = float(account.get("equity", 0))

            # Calculate position risk
            # Options tickers are longer (e.g., AAPL240315C00150000) - multiply by 100 for contract size
            is_option = len(signal.ticker) > 10
            contract_multiplier = 100 if is_option else 1

            if signal.limit_price:
                position_value = signal.quantity * signal.limit_price * contract_multiplier
            else:
                # Use current market price as estimate
                current_price = await self.get_current_price(signal.ticker)
                position_value = signal.quantity * current_price * contract_multiplier

            position_risk_pct = position_value / account_value

            # Check position risk limit
            max_position_risk = self.config.get("max_position_risk", 0.10)
            if position_risk_pct > max_position_risk:
                return {
                    "allowed": False,
                    "reason": f"Position risk {position_risk_pct:.2%} exceeds limit {max_position_risk:.2%}",
                }

            # Check total portfolio risk
            total_risk = await self.calculate_total_portfolio_risk()
            max_total_risk = self.config.get("max_total_risk", 0.30)
            if total_risk > max_total_risk:
                return {
                    "allowed": False,
                    "reason": f"Total portfolio risk {total_risk:.2%} exceeds limit {max_total_risk: .2%}",
                }

            return {"allowed": True, "reason": "Risk limits OK"}

        except Exception as e:
            self.logger.error(f"Error checking risk limits: {e}")
            return {"allowed": False, "reason": f"Risk check failed: {e}"}

    async def execute_broker_order(self, signal: TradeSignal) -> dict[str, Any]:
        """Execute order via broker."""
        try:
            if signal.side == OrderSide.BUY:
                if signal.order_type == OrderType.MARKET:
                    result = self.broker.market_buy(signal.ticker, signal.quantity)
                else:
                    # For limit orders, we'd need to implement limit_buy
                    result = self.broker.market_buy(signal.ticker, signal.quantity)
            elif signal.order_type == OrderType.MARKET:
                result = self.broker.market_sell(signal.ticker, signal.quantity)
            else:
                # For limit orders, we'd need to implement limit_sell
                result = self.broker.market_sell(signal.ticker, signal.quantity)

            if result:
                return {
                    "success": True,
                    "filled_quantity": signal.quantity,
                    "filled_price": signal.limit_price
                    or await self.get_current_price(signal.ticker),
                    "commission": self.config.get("default_commission", 1.0),
                }
            else:
                return {"success": False, "error": "Broker order failed"}

        except Exception as e:
            self.logger.error(f"Broker execution error: {e}")
            return {"success": False, "error": str(e)}

    async def get_account_info(self) -> dict[str, Any]:
        """Get account information from broker."""
        try:
            account = self.broker.get_account()
            return {
                "equity": account.equity,
                "buying_power": account.buying_power,
                "cash": account.cash,
                "portfolio_value": account.portfolio_value,
            }
        except Exception as e:
            self.logger.error(f"Error getting account info: {e}")
            return {"equity": 0, "buying_power": 0, "cash": 0, "portfolio_value": 0}

    async def get_current_price(self, ticker: str) -> float:
        """Get current price for ticker."""
        try:
            success, price = self.broker.get_price(ticker)
            if success:
                return float(price)
            else:
                self.logger.warning(f"Could not get price for {ticker}: {price}")
                return 0.0
        except Exception as e:
            self.logger.error(f"Error getting price for {ticker}: {e}")
            return 0.0

    async def is_market_open(self) -> bool:
        """Check if market is open."""
        try:
            return self.broker.market_close() is True
        except Exception as e:
            self.logger.error(f"Error checking market status: {e}")
            return False

    async def calculate_total_portfolio_risk(self) -> float:
        """Calculate total portfolio risk percentage."""
        try:
            positions = self.broker.get_positions()
            if isinstance(positions, str):  # Error case
                return 0.0

            total_risk = 0.0
            for position in positions:
                # Simplified risk calculation
                position_risk = (
                    abs(float(position.unrealized_pl)) / float(position.market_value)
                    if position.market_value > 0
                    else 0
                )
                total_risk += position_risk

            return min(total_risk, 1.0)  # Cap at 100%

        except Exception as e:
            self.logger.error(f"Error calculating portfolio risk: {e}")
            return 0.0

    async def update_positions(self) -> list[PositionUpdate]:
        """Update position information from broker."""
        try:
            positions = self.broker.get_positions()
            if isinstance(positions, str):  # Error case
                return []

            updates = []
            for position in positions:
                update = PositionUpdate(
                    ticker=position.symbol,
                    quantity=int(position.qty),
                    avg_price=float(position.avg_entry_price),
                    current_price=float(position.current_price),
                    unrealized_pnl=float(position.unrealized_pl),
                    market_value=float(position.market_value),
                )
                updates.append(update)
                self.positions[position.symbol] = update

            return updates

        except Exception as e:
            self.logger.error(f"Error updating positions: {e}")
            return []

    async def get_trade_history(self, limit: int = 100) -> list[TradeResult]:
        """Get recent trade history."""
        return list(self.active_trades.values())[-limit:]

    async def cancel_trade(self, trade_id: str) -> bool:
        """Cancel a pending trade."""
        try:
            if trade_id in self.active_trades:
                trade = self.active_trades[trade_id]
                if trade.status == TradeStatus.PENDING:
                    # Implement broker cancel logic here
                    trade.status = TradeStatus.CANCELLED
                    return True
            return False
        except Exception as e:
            self.logger.error(f"Error cancelling trade {trade_id}: {e}")
            return False


# Factory function for easy initialization
def create_trading_interface(config: dict[str, Any]) -> TradingInterface:
    """Create trading interface with default components."""
    # Initialize broker manager with test keys if not provided
    api_key = config.get("alpaca_api_key", "test_key")
    secret_key = config.get("alpaca_secret_key", "test_secret")

    broker = AlpacaManager(api_key, secret_key)

    # Initialize risk manager
    risk_params = RiskParameters()
    risk_manager = RiskManager(risk_params)

    # Initialize alert system
    alert_system = TradingAlertSystem()

    return TradingInterface(broker, risk_manager, alert_system, config)
