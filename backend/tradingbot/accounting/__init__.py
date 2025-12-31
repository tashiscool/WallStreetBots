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
    # Wash Sale
    "Fill",
    "Lot",
    "WashSaleEngine",
    # Buying Power - Types
    "AccountType",
    "OrderDirection",
    "SecurityType",
    "Position",
    "Order",
    "BuyingPowerResult",
    "DayTradeInfo",
    # Buying Power - Models
    "IBuyingPowerModel",
    "CashBuyingPowerModel",
    "MarginBuyingPowerModel",
    "PatternDayTradingMarginModel",
    "OptionBuyingPowerModel",
    "BuyingPowerModelFactory",
    # Settlement - Types
    "SettlementType",
    "PendingSettlement",
    "SettlementSummary",
    # Settlement - Models
    "ISettlementModel",
    "ImmediateSettlementModel",
    "DelayedSettlementModel",
    "EquitySettlementModel",
    "OptionsSettlementModel",
    "MultiAssetSettlementModel",
    "SettlementModelFactory",
]
