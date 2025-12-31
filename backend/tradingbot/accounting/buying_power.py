"""
Buying Power Models.

Ported from QuantConnect/LEAN's buying power framework.
Handles margin requirements, PDT rules, and buying power calculations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class AccountType(Enum):
    """Type of trading account."""
    CASH = "cash"
    MARGIN = "margin"
    PORTFOLIO_MARGIN = "portfolio_margin"


class OrderDirection(Enum):
    """Direction of an order."""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


class SecurityType(Enum):
    """Type of security."""
    EQUITY = "equity"
    OPTION = "option"
    FUTURE = "future"
    FOREX = "forex"
    CRYPTO = "crypto"


@dataclass
class Position:
    """Represents a position in a security."""
    symbol: str
    quantity: int
    average_price: Decimal
    security_type: SecurityType = SecurityType.EQUITY
    current_price: Optional[Decimal] = None

    @property
    def market_value(self) -> Decimal:
        """Current market value of position."""
        price = self.current_price or self.average_price
        return abs(self.quantity) * price

    @property
    def is_long(self) -> bool:
        """True if long position."""
        return self.quantity > 0

    @property
    def is_short(self) -> bool:
        """True if short position."""
        return self.quantity < 0


@dataclass
class Order:
    """Order for buying power calculation."""
    symbol: str
    quantity: int
    direction: OrderDirection
    security_type: SecurityType = SecurityType.EQUITY
    limit_price: Optional[Decimal] = None
    estimated_price: Optional[Decimal] = None


@dataclass
class BuyingPowerResult:
    """Result of buying power calculation."""
    is_sufficient: bool
    reason: str = ""
    buying_power: Decimal = Decimal("0")
    required_buying_power: Decimal = Decimal("0")
    current_margin_used: Decimal = Decimal("0")
    order_margin_required: Decimal = Decimal("0")


@dataclass
class DayTradeInfo:
    """Information about day trades."""
    date: datetime
    symbol: str
    buy_quantity: int
    sell_quantity: int


class IBuyingPowerModel(ABC):
    """
    Abstract interface for buying power models.

    Determines if there's sufficient buying power for an order.
    """

    @abstractmethod
    def has_sufficient_buying_power(
        self,
        order: Order,
        positions: Dict[str, Position],
        cash: Decimal,
        account_value: Decimal,
    ) -> BuyingPowerResult:
        """
        Check if there's sufficient buying power for an order.

        Args:
            order: The order to check
            positions: Current positions
            cash: Available cash
            account_value: Total account value

        Returns:
            BuyingPowerResult with sufficiency status
        """
        pass

    @abstractmethod
    def get_buying_power(
        self,
        positions: Dict[str, Position],
        cash: Decimal,
        account_value: Decimal,
        direction: OrderDirection,
    ) -> Decimal:
        """
        Get available buying power.

        Args:
            positions: Current positions
            cash: Available cash
            account_value: Total account value
            direction: Direction of potential trade

        Returns:
            Available buying power
        """
        pass

    @abstractmethod
    def get_maintenance_margin(
        self,
        position: Position,
    ) -> Decimal:
        """
        Get maintenance margin requirement for a position.

        Args:
            position: The position

        Returns:
            Required maintenance margin
        """
        pass

    @abstractmethod
    def get_initial_margin(
        self,
        order: Order,
        price: Decimal,
    ) -> Decimal:
        """
        Get initial margin requirement for an order.

        Args:
            order: The order
            price: Estimated fill price

        Returns:
            Required initial margin
        """
        pass


class CashBuyingPowerModel(IBuyingPowerModel):
    """
    Cash account buying power model.

    No margin, no shorting. Full cash required for purchases.
    """

    def has_sufficient_buying_power(
        self,
        order: Order,
        positions: Dict[str, Position],
        cash: Decimal,
        account_value: Decimal,
    ) -> BuyingPowerResult:
        """Check buying power for cash account."""
        price = order.limit_price or order.estimated_price or Decimal("0")
        order_value = abs(order.quantity) * price

        if order.direction == OrderDirection.BUY:
            # Need full cash for purchases
            if cash >= order_value:
                return BuyingPowerResult(
                    is_sufficient=True,
                    buying_power=cash,
                    required_buying_power=order_value,
                )
            else:
                return BuyingPowerResult(
                    is_sufficient=False,
                    reason=f"Insufficient cash: have ${cash}, need ${order_value}",
                    buying_power=cash,
                    required_buying_power=order_value,
                )

        elif order.direction == OrderDirection.SELL:
            # Can only sell what we own (no shorting)
            position = positions.get(order.symbol)
            if position is None or position.quantity < abs(order.quantity):
                owned = position.quantity if position else 0
                return BuyingPowerResult(
                    is_sufficient=False,
                    reason=f"Cannot short in cash account. Own {owned}, trying to sell {order.quantity}",
                    buying_power=Decimal("0"),
                    required_buying_power=order_value,
                )
            return BuyingPowerResult(
                is_sufficient=True,
                buying_power=cash,
            )

        return BuyingPowerResult(is_sufficient=True, buying_power=cash)

    def get_buying_power(
        self,
        positions: Dict[str, Position],
        cash: Decimal,
        account_value: Decimal,
        direction: OrderDirection,
    ) -> Decimal:
        """Get buying power for cash account."""
        if direction == OrderDirection.BUY:
            return cash
        return Decimal("0")  # Can't sell short

    def get_maintenance_margin(self, position: Position) -> Decimal:
        """No margin in cash account."""
        return Decimal("0")

    def get_initial_margin(self, order: Order, price: Decimal) -> Decimal:
        """Full value required for cash account."""
        if order.direction == OrderDirection.BUY:
            return abs(order.quantity) * price
        return Decimal("0")


class MarginBuyingPowerModel(IBuyingPowerModel):
    """
    Standard margin account buying power model.

    Reg T margin: 50% initial, 25% maintenance for equities.
    """

    def __init__(
        self,
        initial_margin_ratio: Decimal = Decimal("0.5"),  # 50%
        maintenance_margin_ratio: Decimal = Decimal("0.25"),  # 25%
        leverage: Decimal = Decimal("2.0"),
    ):
        """
        Initialize margin model.

        Args:
            initial_margin_ratio: Initial margin requirement (0.5 = 50%)
            maintenance_margin_ratio: Maintenance margin (0.25 = 25%)
            leverage: Maximum leverage allowed
        """
        self.initial_margin_ratio = initial_margin_ratio
        self.maintenance_margin_ratio = maintenance_margin_ratio
        self.leverage = leverage

    def has_sufficient_buying_power(
        self,
        order: Order,
        positions: Dict[str, Position],
        cash: Decimal,
        account_value: Decimal,
    ) -> BuyingPowerResult:
        """Check buying power for margin account."""
        price = order.limit_price or order.estimated_price or Decimal("0")
        order_value = abs(order.quantity) * price

        # Calculate current margin usage
        current_margin = sum(
            self.get_maintenance_margin(pos)
            for pos in positions.values()
        )

        # Calculate buying power
        buying_power = self.get_buying_power(
            positions, cash, account_value, order.direction
        )

        # Calculate required margin for this order
        required_margin = self.get_initial_margin(order, price)

        if buying_power >= required_margin:
            return BuyingPowerResult(
                is_sufficient=True,
                buying_power=buying_power,
                required_buying_power=required_margin,
                current_margin_used=current_margin,
                order_margin_required=required_margin,
            )
        else:
            return BuyingPowerResult(
                is_sufficient=False,
                reason=f"Insufficient margin: have ${buying_power}, need ${required_margin}",
                buying_power=buying_power,
                required_buying_power=required_margin,
                current_margin_used=current_margin,
                order_margin_required=required_margin,
            )

    def get_buying_power(
        self,
        positions: Dict[str, Position],
        cash: Decimal,
        account_value: Decimal,
        direction: OrderDirection,
    ) -> Decimal:
        """Get available buying power with margin."""
        # Calculate current margin usage
        current_margin = sum(
            self.get_maintenance_margin(pos)
            for pos in positions.values()
        )

        # Excess equity = account value - margin used
        excess_equity = account_value - current_margin

        # Buying power = excess equity * leverage
        return max(Decimal("0"), excess_equity * self.leverage)

    def get_maintenance_margin(self, position: Position) -> Decimal:
        """Get maintenance margin for position."""
        return position.market_value * self.maintenance_margin_ratio

    def get_initial_margin(self, order: Order, price: Decimal) -> Decimal:
        """Get initial margin for order."""
        order_value = abs(order.quantity) * price
        return order_value * self.initial_margin_ratio


class PatternDayTradingMarginModel(MarginBuyingPowerModel):
    """
    Pattern Day Trader (PDT) margin model.

    Implements PDT rules:
    - $25,000 minimum equity requirement
    - 4x intraday buying power
    - 3 day trades max in 5 days for accounts under $25k
    - Overnight margin reverts to 2x
    """

    PDT_MINIMUM_EQUITY = Decimal("25000")
    INTRADAY_LEVERAGE = Decimal("4.0")
    OVERNIGHT_LEVERAGE = Decimal("2.0")
    MAX_DAY_TRADES = 3
    DAY_TRADE_WINDOW = 5  # days

    def __init__(
        self,
        initial_margin_ratio: Decimal = Decimal("0.25"),  # 25% for day trades
        maintenance_margin_ratio: Decimal = Decimal("0.25"),
    ):
        """Initialize PDT margin model."""
        super().__init__(
            initial_margin_ratio=initial_margin_ratio,
            maintenance_margin_ratio=maintenance_margin_ratio,
            leverage=self.INTRADAY_LEVERAGE,
        )
        self._day_trades: List[DayTradeInfo] = []
        self._is_pdt_account: bool = False

    def set_pdt_status(self, is_pdt: bool) -> None:
        """Set PDT account status."""
        self._is_pdt_account = is_pdt

    def record_day_trade(
        self,
        symbol: str,
        buy_quantity: int,
        sell_quantity: int,
    ) -> None:
        """Record a day trade for PDT tracking."""
        self._day_trades.append(DayTradeInfo(
            date=datetime.now(),
            symbol=symbol,
            buy_quantity=buy_quantity,
            sell_quantity=sell_quantity,
        ))

        # Clean up old trades
        cutoff = datetime.now() - timedelta(days=self.DAY_TRADE_WINDOW)
        self._day_trades = [
            t for t in self._day_trades
            if t.date >= cutoff
        ]

    def get_day_trade_count(self) -> int:
        """Get number of day trades in rolling 5-day window."""
        cutoff = datetime.now() - timedelta(days=self.DAY_TRADE_WINDOW)
        return len([t for t in self._day_trades if t.date >= cutoff])

    def can_day_trade(self, account_value: Decimal) -> Tuple[bool, str]:
        """
        Check if day trading is allowed.

        Returns:
            Tuple of (allowed, reason)
        """
        # PDT accounts need $25k minimum
        if self._is_pdt_account:
            if account_value < self.PDT_MINIMUM_EQUITY:
                return False, f"PDT account below ${self.PDT_MINIMUM_EQUITY} minimum"
            return True, ""

        # Non-PDT accounts limited to 3 day trades
        day_trade_count = self.get_day_trade_count()
        if day_trade_count >= self.MAX_DAY_TRADES:
            return False, f"Day trade limit reached ({day_trade_count}/{self.MAX_DAY_TRADES})"

        return True, ""

    def has_sufficient_buying_power(
        self,
        order: Order,
        positions: Dict[str, Position],
        cash: Decimal,
        account_value: Decimal,
        is_day_trade: bool = False,
    ) -> BuyingPowerResult:
        """Check buying power with PDT rules."""
        # Check PDT restrictions for day trades
        if is_day_trade:
            can_trade, reason = self.can_day_trade(account_value)
            if not can_trade:
                return BuyingPowerResult(
                    is_sufficient=False,
                    reason=f"PDT restriction: {reason}",
                    buying_power=Decimal("0"),
                )

            # Use intraday leverage for day trades with PDT accounts
            if self._is_pdt_account and account_value >= self.PDT_MINIMUM_EQUITY:
                self.leverage = self.INTRADAY_LEVERAGE
            else:
                self.leverage = self.OVERNIGHT_LEVERAGE
        else:
            # Overnight positions use standard margin
            self.leverage = self.OVERNIGHT_LEVERAGE

        return super().has_sufficient_buying_power(
            order, positions, cash, account_value
        )

    def get_intraday_buying_power(
        self,
        positions: Dict[str, Position],
        cash: Decimal,
        account_value: Decimal,
    ) -> Decimal:
        """Get intraday buying power (4x for PDT accounts)."""
        if not self._is_pdt_account:
            return self.get_buying_power(
                positions, cash, account_value, OrderDirection.BUY
            )

        if account_value < self.PDT_MINIMUM_EQUITY:
            return Decimal("0")  # PDT account below minimum

        # Calculate intraday buying power
        current_margin = sum(
            self.get_maintenance_margin(pos)
            for pos in positions.values()
        )

        excess_equity = account_value - current_margin
        return max(Decimal("0"), excess_equity * self.INTRADAY_LEVERAGE)

    def get_overnight_buying_power(
        self,
        positions: Dict[str, Position],
        cash: Decimal,
        account_value: Decimal,
    ) -> Decimal:
        """Get overnight buying power (2x standard margin)."""
        current_margin = sum(
            self.get_maintenance_margin(pos)
            for pos in positions.values()
        )

        excess_equity = account_value - current_margin
        return max(Decimal("0"), excess_equity * self.OVERNIGHT_LEVERAGE)


class OptionBuyingPowerModel(MarginBuyingPowerModel):
    """
    Options buying power model.

    Handles different margin requirements for various option positions.
    """

    # Option-specific margin ratios
    NAKED_CALL_MARGIN = Decimal("0.20")  # 20% of underlying
    NAKED_PUT_MARGIN = Decimal("0.20")
    COVERED_CALL_MARGIN = Decimal("0.0")  # No additional margin
    SPREAD_MARGIN = Decimal("1.0")  # Max loss = margin

    def __init__(self):
        """Initialize options buying power model."""
        super().__init__(
            initial_margin_ratio=Decimal("0.5"),
            maintenance_margin_ratio=Decimal("0.25"),
            leverage=Decimal("1.0"),
        )

    def get_option_margin(
        self,
        position: Position,
        underlying_price: Decimal,
        is_covered: bool = False,
    ) -> Decimal:
        """
        Calculate margin for an option position.

        Args:
            position: Option position
            underlying_price: Current price of underlying
            is_covered: True if covered by stock/other options

        Returns:
            Required margin
        """
        # Contract multiplier
        multiplier = Decimal("100")

        if is_covered:
            return Decimal("0")

        # Naked option margin calculation (simplified)
        # Full formula: Max(20% underlying - OTM amount, 10% underlying)
        base_margin = underlying_price * self.NAKED_CALL_MARGIN * multiplier

        return base_margin * abs(position.quantity)

    def get_spread_margin(
        self,
        max_loss: Decimal,
    ) -> Decimal:
        """
        Get margin for a spread position.

        For defined-risk spreads, margin = max loss.

        Args:
            max_loss: Maximum possible loss on the spread

        Returns:
            Required margin
        """
        return max_loss

    def has_sufficient_buying_power(
        self,
        order: Order,
        positions: Dict[str, Position],
        cash: Decimal,
        account_value: Decimal,
    ) -> BuyingPowerResult:
        """Check buying power for options order."""
        if order.security_type != SecurityType.OPTION:
            return super().has_sufficient_buying_power(
                order, positions, cash, account_value
            )

        price = order.limit_price or order.estimated_price or Decimal("0")

        if order.direction == OrderDirection.BUY:
            # Long options require full premium
            required = abs(order.quantity) * price * Decimal("100")

            if cash >= required:
                return BuyingPowerResult(
                    is_sufficient=True,
                    buying_power=cash,
                    required_buying_power=required,
                )
            else:
                return BuyingPowerResult(
                    is_sufficient=False,
                    reason=f"Insufficient cash for option premium: ${required}",
                    buying_power=cash,
                    required_buying_power=required,
                )

        elif order.direction == OrderDirection.SELL:
            # Selling options requires margin
            # Simplified: use 20% of $100 notional per contract
            estimated_underlying = price * Decimal("100")  # Rough estimate
            required_margin = (
                estimated_underlying * self.NAKED_CALL_MARGIN *
                abs(order.quantity) * Decimal("100")
            )

            buying_power = self.get_buying_power(
                positions, cash, account_value, order.direction
            )

            if buying_power >= required_margin:
                return BuyingPowerResult(
                    is_sufficient=True,
                    buying_power=buying_power,
                    required_buying_power=required_margin,
                )
            else:
                return BuyingPowerResult(
                    is_sufficient=False,
                    reason=f"Insufficient margin for naked option: ${required_margin}",
                    buying_power=buying_power,
                    required_buying_power=required_margin,
                )

        return BuyingPowerResult(is_sufficient=True, buying_power=cash)


class BuyingPowerModelFactory:
    """Factory for creating buying power models."""

    @staticmethod
    def create(
        account_type: AccountType,
        is_pdt: bool = False,
    ) -> IBuyingPowerModel:
        """
        Create appropriate buying power model.

        Args:
            account_type: Type of account
            is_pdt: Is pattern day trader

        Returns:
            Appropriate buying power model
        """
        if account_type == AccountType.CASH:
            return CashBuyingPowerModel()
        elif account_type == AccountType.MARGIN:
            if is_pdt:
                model = PatternDayTradingMarginModel()
                model.set_pdt_status(True)
                return model
            return MarginBuyingPowerModel()
        elif account_type == AccountType.PORTFOLIO_MARGIN:
            # Portfolio margin has lower requirements
            return MarginBuyingPowerModel(
                initial_margin_ratio=Decimal("0.15"),
                maintenance_margin_ratio=Decimal("0.12"),
                leverage=Decimal("6.0"),
            )
        else:
            return CashBuyingPowerModel()

