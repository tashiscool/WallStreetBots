"""
Settlement Models.

Ported from QuantConnect/LEAN's settlement framework.
Handles T+1, T+2 settlement for different security types.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta, date
from decimal import Decimal
from enum import Enum
from typing import ClassVar, Dict, List, Optional, Set
import logging

logger = logging.getLogger(__name__)


class SettlementType(Enum):
    """Type of settlement."""
    IMMEDIATE = "immediate"  # T+0 (crypto, forex)
    NEXT_DAY = "next_day"    # T+1 (some equities, options)
    STANDARD = "standard"    # T+2 (US equities)
    EXTENDED = "extended"    # T+3 or more


@dataclass
class PendingSettlement:
    """Represents a pending settlement."""
    symbol: str
    trade_date: datetime
    settlement_date: datetime
    quantity: int
    amount: Decimal  # Positive = credit, negative = debit
    trade_id: str
    is_settled: bool = False

    @property
    def days_until_settlement(self) -> int:
        """Days until settlement."""
        if self.is_settled:
            return 0
        delta = self.settlement_date.date() - datetime.now().date()
        return max(0, delta.days)


@dataclass
class SettlementSummary:
    """Summary of settlement status."""
    pending_count: int
    pending_credits: Decimal
    pending_debits: Decimal
    net_pending: Decimal
    next_settlement_date: Optional[datetime]
    settled_today: List[PendingSettlement]


class ISettlementModel(ABC):
    """
    Abstract interface for settlement models.

    Tracks settlement timing for trades.
    """

    @abstractmethod
    def get_settlement_date(
        self,
        trade_date: datetime,
        symbol: str,
    ) -> datetime:
        """
        Get settlement date for a trade.

        Args:
            trade_date: Date of trade
            symbol: Security symbol

        Returns:
            Settlement date
        """
        pass

    @abstractmethod
    def get_unsettled_cash(self) -> Decimal:
        """
        Get total unsettled cash (pending).

        Returns:
            Net unsettled amount
        """
        pass

    @abstractmethod
    def get_settled_cash(self, total_cash: Decimal) -> Decimal:
        """
        Get settled (available) cash.

        Args:
            total_cash: Total cash balance

        Returns:
            Available settled cash
        """
        pass


class ImmediateSettlementModel(ISettlementModel):
    """
    Immediate settlement (T+0).

    Used for crypto, forex, and futures.
    """

    def __init__(self):
        """Initialize immediate settlement model."""
        self._pending: List[PendingSettlement] = []

    def get_settlement_date(
        self,
        trade_date: datetime,
        symbol: str,
    ) -> datetime:
        """Settlement is immediate."""
        return trade_date

    def get_unsettled_cash(self) -> Decimal:
        """No unsettled cash for immediate settlement."""
        return Decimal("0")

    def get_settled_cash(self, total_cash: Decimal) -> Decimal:
        """All cash is settled immediately."""
        return total_cash

    def record_trade(
        self,
        symbol: str,
        quantity: int,
        amount: Decimal,
        trade_id: str,
    ) -> PendingSettlement:
        """Record a trade (immediately settled)."""
        now = datetime.now()
        settlement = PendingSettlement(
            symbol=symbol,
            trade_date=now,
            settlement_date=now,
            quantity=quantity,
            amount=amount,
            trade_id=trade_id,
            is_settled=True,
        )
        return settlement


class DelayedSettlementModel(ISettlementModel):
    """
    Delayed settlement model (T+N).

    Handles T+1, T+2, or custom settlement periods.
    """

    # US Market holidays (simplified - should use proper calendar)
    US_HOLIDAYS: ClassVar[Set[date]] = set()

    def __init__(
        self,
        settlement_days: int = 2,
        use_business_days: bool = True,
    ):
        """
        Initialize delayed settlement model.

        Args:
            settlement_days: Days until settlement (T+N)
            use_business_days: Count only business days
        """
        self.settlement_days = settlement_days
        self.use_business_days = use_business_days
        self._pending: List[PendingSettlement] = []

    def _is_business_day(self, d: date) -> bool:
        """Check if date is a business day."""
        # Weekend check
        if d.weekday() >= 5:
            return False
        # Holiday check
        if d in self.US_HOLIDAYS:
            return False
        return True

    def _add_business_days(self, start: date, days: int) -> date:
        """Add N business days to a date."""
        current = start
        days_added = 0

        while days_added < days:
            current += timedelta(days=1)
            if self._is_business_day(current):
                days_added += 1

        return current

    def get_settlement_date(
        self,
        trade_date: datetime,
        symbol: str,
    ) -> datetime:
        """Calculate settlement date."""
        trade_day = trade_date.date()

        if self.use_business_days:
            settle_day = self._add_business_days(
                trade_day, self.settlement_days
            )
        else:
            settle_day = trade_day + timedelta(days=self.settlement_days)

        # Return as datetime at end of day
        return datetime.combine(
            settle_day,
            datetime.max.time().replace(microsecond=0)
        )

    def record_trade(
        self,
        symbol: str,
        quantity: int,
        amount: Decimal,
        trade_id: str,
    ) -> PendingSettlement:
        """
        Record a trade for settlement tracking.

        Args:
            symbol: Security symbol
            quantity: Trade quantity
            amount: Cash amount (+ credit, - debit)
            trade_id: Unique trade ID

        Returns:
            PendingSettlement record
        """
        now = datetime.now()
        settlement_date = self.get_settlement_date(now, symbol)

        pending = PendingSettlement(
            symbol=symbol,
            trade_date=now,
            settlement_date=settlement_date,
            quantity=quantity,
            amount=amount,
            trade_id=trade_id,
            is_settled=False,
        )

        self._pending.append(pending)
        logger.info(
            f"Trade recorded: {symbol} settles {settlement_date.date()}"
        )
        return pending

    def process_settlements(self) -> List[PendingSettlement]:
        """
        Process any settlements that are due.

        Returns:
            List of newly settled trades
        """
        now = datetime.now()
        settled = []

        for pending in self._pending:
            if not pending.is_settled and pending.settlement_date <= now:
                pending.is_settled = True
                settled.append(pending)
                logger.info(
                    f"Settlement complete: {pending.symbol} "
                    f"(${pending.amount})"
                )

        return settled

    def get_unsettled_cash(self) -> Decimal:
        """Get total unsettled cash."""
        self.process_settlements()
        return sum(
            p.amount for p in self._pending
            if not p.is_settled
        )

    def get_settled_cash(self, total_cash: Decimal) -> Decimal:
        """Get available settled cash."""
        unsettled = self.get_unsettled_cash()
        # Subtract pending debits from available cash
        pending_debits = sum(
            p.amount for p in self._pending
            if not p.is_settled and p.amount < 0
        )
        return total_cash + pending_debits  # Debits are negative

    def get_pending_settlements(self) -> List[PendingSettlement]:
        """Get all pending settlements."""
        self.process_settlements()
        return [p for p in self._pending if not p.is_settled]

    def get_settlement_summary(self) -> SettlementSummary:
        """Get summary of settlement status."""
        settled_today = self.process_settlements()

        pending = [p for p in self._pending if not p.is_settled]
        credits = sum(p.amount for p in pending if p.amount > 0)
        debits = sum(p.amount for p in pending if p.amount < 0)

        next_settlement = None
        if pending:
            next_settlement = min(p.settlement_date for p in pending)

        return SettlementSummary(
            pending_count=len(pending),
            pending_credits=credits,
            pending_debits=debits,
            net_pending=credits + debits,
            next_settlement_date=next_settlement,
            settled_today=settled_today,
        )

    def clear_settled(self) -> int:
        """
        Clear settled trades from tracking.

        Returns:
            Number of cleared trades
        """
        before = len(self._pending)
        self._pending = [p for p in self._pending if not p.is_settled]
        return before - len(self._pending)


class EquitySettlementModel(DelayedSettlementModel):
    """
    US Equity settlement model (T+2).

    Standard settlement for US stocks.
    """

    def __init__(self):
        """Initialize T+2 equity settlement."""
        super().__init__(
            settlement_days=2,
            use_business_days=True,
        )


class OptionsSettlementModel(DelayedSettlementModel):
    """
    Options settlement model (T+1).

    Options typically settle next business day.
    """

    def __init__(self):
        """Initialize T+1 options settlement."""
        super().__init__(
            settlement_days=1,
            use_business_days=True,
        )


class MultiAssetSettlementModel(ISettlementModel):
    """
    Multi-asset settlement model.

    Routes to appropriate settlement model based on security type.
    """

    def __init__(self):
        """Initialize multi-asset settlement."""
        self._equity_model = EquitySettlementModel()
        self._options_model = OptionsSettlementModel()
        self._immediate_model = ImmediateSettlementModel()

        # Symbol to model mapping
        self._symbol_models: Dict[str, DelayedSettlementModel] = {}

    def _get_model(self, symbol: str) -> ISettlementModel:
        """Get appropriate settlement model for symbol."""
        if symbol in self._symbol_models:
            return self._symbol_models[symbol]

        # Detect by symbol pattern
        if self._is_option_symbol(symbol):
            return self._options_model
        elif self._is_crypto_symbol(symbol):
            return self._immediate_model
        else:
            return self._equity_model

    def _is_option_symbol(self, symbol: str) -> bool:
        """Check if symbol is an option."""
        # Options typically have format: SYMBOL YYMMDD C/P STRIKE
        return len(symbol) > 10 and any(c in symbol for c in ['C', 'P'])

    def _is_crypto_symbol(self, symbol: str) -> bool:
        """Check if symbol is crypto."""
        crypto_suffixes = ['USD', 'USDT', 'BTC', 'ETH']
        return any(symbol.endswith(s) for s in crypto_suffixes)

    def set_symbol_settlement(
        self,
        symbol: str,
        settlement_days: int,
    ) -> None:
        """Set custom settlement for a symbol."""
        self._symbol_models[symbol] = DelayedSettlementModel(
            settlement_days=settlement_days,
            use_business_days=True,
        )

    def get_settlement_date(
        self,
        trade_date: datetime,
        symbol: str,
    ) -> datetime:
        """Get settlement date for symbol."""
        model = self._get_model(symbol)
        return model.get_settlement_date(trade_date, symbol)

    def record_trade(
        self,
        symbol: str,
        quantity: int,
        amount: Decimal,
        trade_id: str,
    ) -> PendingSettlement:
        """Record a trade."""
        model = self._get_model(symbol)
        if hasattr(model, 'record_trade'):
            return model.record_trade(symbol, quantity, amount, trade_id)
        else:
            # Immediate settlement
            now = datetime.now()
            return PendingSettlement(
                symbol=symbol,
                trade_date=now,
                settlement_date=now,
                quantity=quantity,
                amount=amount,
                trade_id=trade_id,
                is_settled=True,
            )

    def get_unsettled_cash(self) -> Decimal:
        """Get total unsettled cash across all models."""
        return (
            self._equity_model.get_unsettled_cash() +
            self._options_model.get_unsettled_cash()
        )

    def get_settled_cash(self, total_cash: Decimal) -> Decimal:
        """Get available settled cash."""
        unsettled_debits = Decimal("0")

        for model in [self._equity_model, self._options_model]:
            for p in model._pending:
                if not p.is_settled and p.amount < 0:
                    unsettled_debits += p.amount

        return total_cash + unsettled_debits

    def get_all_pending(self) -> List[PendingSettlement]:
        """Get all pending settlements."""
        pending = []
        pending.extend(self._equity_model.get_pending_settlements())
        pending.extend(self._options_model.get_pending_settlements())
        return sorted(pending, key=lambda p: p.settlement_date)

    def process_all_settlements(self) -> List[PendingSettlement]:
        """Process all pending settlements."""
        settled = []
        settled.extend(self._equity_model.process_settlements())
        settled.extend(self._options_model.process_settlements())
        return settled


class SettlementModelFactory:
    """Factory for creating settlement models."""

    @staticmethod
    def create_for_security_type(security_type: str) -> ISettlementModel:
        """
        Create settlement model for security type.

        Args:
            security_type: Type of security

        Returns:
            Appropriate settlement model
        """
        security_type = security_type.lower()

        if security_type == "equity":
            return EquitySettlementModel()
        elif security_type == "option":
            return OptionsSettlementModel()
        elif security_type in ["crypto", "forex"]:
            return ImmediateSettlementModel()
        elif security_type == "future":
            return DelayedSettlementModel(settlement_days=1)
        else:
            return EquitySettlementModel()

    @staticmethod
    def create_multi_asset() -> MultiAssetSettlementModel:
        """Create multi-asset settlement model."""
        return MultiAssetSettlementModel()

