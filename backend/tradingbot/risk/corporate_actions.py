"""
Corporate Actions Handler.

Handles stock splits, dividends, and other corporate actions.
Essential for accurate backtesting and position tracking.
"""

from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from decimal import Decimal, ROUND_DOWN
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class CorporateActionType(Enum):
    """Type of corporate action."""
    STOCK_SPLIT = "stock_split"
    REVERSE_SPLIT = "reverse_split"
    CASH_DIVIDEND = "cash_dividend"
    STOCK_DIVIDEND = "stock_dividend"
    SPECIAL_DIVIDEND = "special_dividend"
    SPIN_OFF = "spin_off"
    MERGER = "merger"
    ACQUISITION = "acquisition"
    NAME_CHANGE = "name_change"
    SYMBOL_CHANGE = "symbol_change"
    DELISTING = "delisting"
    RIGHTS_OFFERING = "rights_offering"


class DividendType(Enum):
    """Type of dividend."""
    REGULAR = "regular"
    SPECIAL = "special"
    QUALIFIED = "qualified"
    RETURN_OF_CAPITAL = "return_of_capital"


@dataclass
class CorporateAction:
    """Represents a corporate action event."""
    symbol: str
    action_type: CorporateActionType
    ex_date: date  # Date on which adjustment applies
    record_date: Optional[date] = None
    payment_date: Optional[date] = None
    announcement_date: Optional[date] = None

    # Split-specific
    split_ratio: Optional[Tuple[int, int]] = None  # (new, old) e.g., (4, 1) for 4:1

    # Dividend-specific
    dividend_amount: Optional[Decimal] = None
    dividend_type: Optional[DividendType] = None

    # Symbol change
    new_symbol: Optional[str] = None

    # Merger/acquisition
    cash_component: Optional[Decimal] = None
    stock_component: Optional[Decimal] = None  # shares of acquirer per share
    acquirer_symbol: Optional[str] = None

    # Metadata
    description: str = ""
    is_processed: bool = False
    processed_at: Optional[datetime] = None

    @property
    def split_factor(self) -> Optional[Decimal]:
        """Get split adjustment factor."""
        if self.split_ratio:
            new, old = self.split_ratio
            return Decimal(new) / Decimal(old)
        return None


@dataclass
class Position:
    """Position to be adjusted."""
    symbol: str
    quantity: int
    average_price: Decimal
    cost_basis: Decimal

    @property
    def market_value(self) -> Decimal:
        """Position cost basis (for adjustment tracking)."""
        return self.cost_basis


@dataclass
class AdjustmentResult:
    """Result of applying corporate action."""
    action: CorporateAction
    original_position: Position
    adjusted_position: Position
    cash_generated: Decimal = Decimal("0")
    fractional_shares: Decimal = Decimal("0")
    adjustment_details: Dict[str, Any] = field(default_factory=dict)


class CorporateActionsHandler:
    """
    Handles corporate actions for positions and historical data.

    Adjusts positions for splits, dividends, mergers, etc.
    """

    def __init__(
        self,
        handle_fractional: str = "cash",  # 'cash' or 'round_down'
        reinvest_dividends: bool = False,
    ):
        """
        Initialize corporate actions handler.

        Args:
            handle_fractional: How to handle fractional shares
            reinvest_dividends: Auto-reinvest dividends (DRIP)
        """
        self.handle_fractional = handle_fractional
        self.reinvest_dividends = reinvest_dividends

        self._pending_actions: List[CorporateAction] = []
        self._processed_actions: List[CorporateAction] = []
        self._callbacks: List[Callable[[CorporateAction, AdjustmentResult], None]] = []

    def on_adjustment(
        self,
        callback: Callable[[CorporateAction, AdjustmentResult], None],
    ) -> None:
        """Register callback for adjustments."""
        self._callbacks.append(callback)

    def _notify_callbacks(
        self,
        action: CorporateAction,
        result: AdjustmentResult,
    ) -> None:
        """Notify callbacks of adjustment."""
        for callback in self._callbacks:
            try:
                callback(action, result)
            except Exception as e:
                logger.error(f"Corporate action callback error: {e}")

    def add_action(self, action: CorporateAction) -> None:
        """Add a corporate action to pending queue."""
        self._pending_actions.append(action)
        self._pending_actions.sort(key=lambda a: a.ex_date)
        logger.info(
            f"Added corporate action: {action.symbol} - "
            f"{action.action_type.value} on {action.ex_date}"
        )

    def get_pending_actions(
        self,
        symbol: Optional[str] = None,
        as_of_date: Optional[date] = None,
    ) -> List[CorporateAction]:
        """Get pending actions, optionally filtered."""
        actions = self._pending_actions

        if symbol:
            actions = [a for a in actions if a.symbol == symbol]

        if as_of_date:
            actions = [a for a in actions if a.ex_date <= as_of_date]

        return actions

    def apply_split(
        self,
        position: Position,
        action: CorporateAction,
        current_price: Decimal,
    ) -> AdjustmentResult:
        """
        Apply stock split to position.

        Args:
            position: Position to adjust
            action: Split action
            current_price: Current stock price

        Returns:
            AdjustmentResult
        """
        if action.split_factor is None:
            raise ValueError("Split action missing split_ratio")

        factor = action.split_factor

        # Adjust quantity
        new_quantity_decimal = Decimal(position.quantity) * factor
        new_quantity = int(new_quantity_decimal.to_integral_value(ROUND_DOWN))
        fractional = new_quantity_decimal - Decimal(new_quantity)

        # Adjust average price (inverse of split)
        new_avg_price = position.average_price / factor

        # Cost basis stays the same (just spread across more shares)
        new_cost_basis = position.cost_basis

        # Handle fractional shares
        cash_for_fractional = Decimal("0")
        if fractional > 0:
            if self.handle_fractional == "cash":
                # Pay cash for fractional shares
                cash_for_fractional = fractional * current_price
            # Otherwise just round down (shares lost)

        adjusted_position = Position(
            symbol=position.symbol,
            quantity=new_quantity,
            average_price=new_avg_price,
            cost_basis=new_cost_basis,
        )

        result = AdjustmentResult(
            action=action,
            original_position=position,
            adjusted_position=adjusted_position,
            cash_generated=cash_for_fractional,
            fractional_shares=fractional,
            adjustment_details={
                "split_factor": float(factor),
                "original_qty": position.quantity,
                "new_qty": new_quantity,
                "original_price": float(position.average_price),
                "new_price": float(new_avg_price),
            },
        )

        logger.info(
            f"Applied {action.split_ratio[0]}:{action.split_ratio[1]} split to "
            f"{position.symbol}: {position.quantity} -> {new_quantity} shares"
        )

        return result

    def apply_reverse_split(
        self,
        position: Position,
        action: CorporateAction,
        current_price: Decimal,
    ) -> AdjustmentResult:
        """
        Apply reverse split to position.

        Args:
            position: Position to adjust
            action: Reverse split action
            current_price: Current stock price

        Returns:
            AdjustmentResult
        """
        # Reverse split is same as split but factor < 1
        return self.apply_split(position, action, current_price)

    def apply_cash_dividend(
        self,
        position: Position,
        action: CorporateAction,
        current_price: Decimal,
    ) -> AdjustmentResult:
        """
        Apply cash dividend to position.

        Args:
            position: Position to adjust
            action: Dividend action
            current_price: Current stock price

        Returns:
            AdjustmentResult
        """
        if action.dividend_amount is None:
            raise ValueError("Dividend action missing dividend_amount")

        # Calculate dividend payment
        dividend_payment = action.dividend_amount * position.quantity

        # Reinvest if DRIP enabled
        new_quantity = position.quantity
        shares_purchased = Decimal("0")
        cash_generated = dividend_payment

        if self.reinvest_dividends and current_price > 0:
            shares_purchased = dividend_payment / current_price
            whole_shares = int(shares_purchased.to_integral_value(ROUND_DOWN))
            new_quantity = position.quantity + whole_shares
            cash_generated = (shares_purchased - Decimal(whole_shares)) * current_price

        # Adjust cost basis for DRIP
        new_cost_basis = position.cost_basis
        if self.reinvest_dividends and shares_purchased > 0:
            new_cost_basis += shares_purchased * current_price

        adjusted_position = Position(
            symbol=position.symbol,
            quantity=new_quantity,
            average_price=position.average_price,  # Original cost basis preserved
            cost_basis=new_cost_basis,
        )

        result = AdjustmentResult(
            action=action,
            original_position=position,
            adjusted_position=adjusted_position,
            cash_generated=cash_generated,
            adjustment_details={
                "dividend_per_share": float(action.dividend_amount),
                "total_dividend": float(dividend_payment),
                "reinvested": self.reinvest_dividends,
                "shares_purchased": float(shares_purchased) if self.reinvest_dividends else 0,
            },
        )

        logger.info(
            f"Applied ${action.dividend_amount}/share dividend to "
            f"{position.symbol}: ${dividend_payment} total"
        )

        return result

    def apply_stock_dividend(
        self,
        position: Position,
        action: CorporateAction,
    ) -> AdjustmentResult:
        """
        Apply stock dividend to position.

        Args:
            position: Position to adjust
            action: Stock dividend action

        Returns:
            AdjustmentResult
        """
        if action.split_ratio is None:
            raise ValueError("Stock dividend action missing split_ratio")

        # Stock dividend is similar to split
        # e.g., 5% stock dividend = (105, 100) ratio
        factor = action.split_factor

        new_quantity_decimal = Decimal(position.quantity) * factor
        new_quantity = int(new_quantity_decimal.to_integral_value(ROUND_DOWN))
        fractional = new_quantity_decimal - Decimal(new_quantity)

        # Cost basis stays same, spread across more shares
        new_avg_price = position.cost_basis / Decimal(new_quantity)

        adjusted_position = Position(
            symbol=position.symbol,
            quantity=new_quantity,
            average_price=new_avg_price,
            cost_basis=position.cost_basis,
        )

        result = AdjustmentResult(
            action=action,
            original_position=position,
            adjusted_position=adjusted_position,
            fractional_shares=fractional,
            adjustment_details={
                "dividend_rate": float(factor - 1),
                "shares_received": new_quantity - position.quantity,
            },
        )

        logger.info(
            f"Applied stock dividend to {position.symbol}: "
            f"+{new_quantity - position.quantity} shares"
        )

        return result

    def apply_symbol_change(
        self,
        position: Position,
        action: CorporateAction,
    ) -> AdjustmentResult:
        """
        Apply symbol change to position.

        Args:
            position: Position to adjust
            action: Symbol change action

        Returns:
            AdjustmentResult
        """
        if action.new_symbol is None:
            raise ValueError("Symbol change action missing new_symbol")

        adjusted_position = Position(
            symbol=action.new_symbol,
            quantity=position.quantity,
            average_price=position.average_price,
            cost_basis=position.cost_basis,
        )

        result = AdjustmentResult(
            action=action,
            original_position=position,
            adjusted_position=adjusted_position,
            adjustment_details={
                "old_symbol": position.symbol,
                "new_symbol": action.new_symbol,
            },
        )

        logger.info(
            f"Applied symbol change: {position.symbol} -> {action.new_symbol}"
        )

        return result

    def apply_action(
        self,
        position: Position,
        action: CorporateAction,
        current_price: Decimal,
    ) -> AdjustmentResult:
        """
        Apply a corporate action to a position.

        Args:
            position: Position to adjust
            action: Corporate action to apply
            current_price: Current stock price

        Returns:
            AdjustmentResult
        """
        if action.action_type == CorporateActionType.STOCK_SPLIT:
            result = self.apply_split(position, action, current_price)
        elif action.action_type == CorporateActionType.REVERSE_SPLIT:
            result = self.apply_reverse_split(position, action, current_price)
        elif action.action_type == CorporateActionType.CASH_DIVIDEND:
            result = self.apply_cash_dividend(position, action, current_price)
        elif action.action_type == CorporateActionType.STOCK_DIVIDEND:
            result = self.apply_stock_dividend(position, action)
        elif action.action_type in (
            CorporateActionType.NAME_CHANGE,
            CorporateActionType.SYMBOL_CHANGE,
        ):
            result = self.apply_symbol_change(position, action)
        else:
            raise ValueError(f"Unsupported action type: {action.action_type}")

        # Mark as processed
        action.is_processed = True
        action.processed_at = datetime.now()
        self._processed_actions.append(action)

        if action in self._pending_actions:
            self._pending_actions.remove(action)

        self._notify_callbacks(action, result)

        return result

    def process_pending(
        self,
        positions: Dict[str, Position],
        prices: Dict[str, Decimal],
        as_of_date: date,
    ) -> List[AdjustmentResult]:
        """
        Process all pending corporate actions up to a date.

        Args:
            positions: Current positions
            prices: Current prices
            as_of_date: Process actions up to this date

        Returns:
            List of adjustment results
        """
        results = []

        pending = self.get_pending_actions(as_of_date=as_of_date)

        for action in pending:
            if action.symbol not in positions:
                continue

            position = positions[action.symbol]
            price = prices.get(action.symbol, position.average_price)

            try:
                result = self.apply_action(position, action, price)
                results.append(result)

                # Update position in dict
                positions[result.adjusted_position.symbol] = result.adjusted_position

                # Handle symbol change
                if result.adjusted_position.symbol != position.symbol:
                    del positions[position.symbol]

            except Exception as e:
                logger.error(
                    f"Error processing corporate action for {action.symbol}: {e}"
                )

        return results

    def adjust_historical_price(
        self,
        price: Decimal,
        symbol: str,
        price_date: date,
        actions: Optional[List[CorporateAction]] = None,
    ) -> Decimal:
        """
        Adjust a historical price for splits/dividends.

        Args:
            price: Original historical price
            symbol: Stock symbol
            price_date: Date of the price
            actions: Actions to apply (or use processed actions)

        Returns:
            Split-adjusted price
        """
        if actions is None:
            actions = [
                a for a in self._processed_actions
                if a.symbol == symbol and a.ex_date > price_date
            ]

        adjusted_price = price

        for action in sorted(actions, key=lambda a: a.ex_date):
            if action.action_type in (
                CorporateActionType.STOCK_SPLIT,
                CorporateActionType.REVERSE_SPLIT,
            ):
                if action.split_factor:
                    adjusted_price = adjusted_price / action.split_factor

            elif action.action_type == CorporateActionType.CASH_DIVIDEND:
                # Subtract dividend from price for adjustment
                if action.dividend_amount:
                    adjusted_price = adjusted_price - action.dividend_amount

        return adjusted_price

    def get_status(self) -> Dict[str, Any]:
        """Get corporate actions status."""
        return {
            "pending_count": len(self._pending_actions),
            "processed_count": len(self._processed_actions),
            "pending_by_symbol": {
                symbol: len([a for a in self._pending_actions if a.symbol == symbol])
                for symbol in set(a.symbol for a in self._pending_actions)
            },
            "reinvest_dividends": self.reinvest_dividends,
            "fractional_handling": self.handle_fractional,
        }

