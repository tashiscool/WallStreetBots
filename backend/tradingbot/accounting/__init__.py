"""
Accounting Module.

Provides buying power models, settlement tracking, margin calculations,
and wash sale tracking.

Usage:
    from backend.tradingbot.accounting import (
        BuyingPowerModelFactory,
        PatternDayTradingMarginModel,
        EquitySettlementModel,
    )

    # Create PDT margin model
    model = BuyingPowerModelFactory.create(AccountType.MARGIN, is_pdt=True)

    # Check buying power
    result = model.has_sufficient_buying_power(order, positions, cash, account_value)

    # Track settlement
    settlement = EquitySettlementModel()
    settlement.record_trade("AAPL", 100, Decimal("-15000"), "trade_123")
"""

from .washsale import (
    WashSaleEngine,
    Fill,
    Lot,
)

from .buying_power import (
    AccountType,
    OrderDirection,
    SecurityType,
    Position,
    Order,
    BuyingPowerResult,
    DayTradeInfo,
    IBuyingPowerModel,
    CashBuyingPowerModel,
    MarginBuyingPowerModel,
    PatternDayTradingMarginModel,
    OptionBuyingPowerModel,
    BuyingPowerModelFactory,
)

from .settlement import (
    SettlementType,
    PendingSettlement,
    SettlementSummary,
    ISettlementModel,
    ImmediateSettlementModel,
    DelayedSettlementModel,
    EquitySettlementModel,
    OptionsSettlementModel,
    MultiAssetSettlementModel,
    SettlementModelFactory,
)

__all__ = [
    # Buying Power - Types
    "AccountType",
    "BuyingPowerModelFactory",
    "BuyingPowerResult",
    "CashBuyingPowerModel",
    "DayTradeInfo",
    "DelayedSettlementModel",
    "EquitySettlementModel",
    # Wash Sale
    "Fill",
    # Buying Power - Models
    "IBuyingPowerModel",
    # Settlement - Models
    "ISettlementModel",
    "ImmediateSettlementModel",
    "Lot",
    "MarginBuyingPowerModel",
    "MultiAssetSettlementModel",
    "OptionBuyingPowerModel",
    "OptionsSettlementModel",
    "Order",
    "OrderDirection",
    "PatternDayTradingMarginModel",
    "PendingSettlement",
    "Position",
    "SecurityType",
    "SettlementModelFactory",
    "SettlementSummary",
    # Settlement - Types
    "SettlementType",
    "WashSaleEngine",
]
