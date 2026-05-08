"""Production strategy class exports with lazy imports."""

from importlib import import_module
from typing import Any

_CLASS_TO_MODULE = {
    "ProductionDebitSpreads": "production_debit_spreads",
    "ProductionEarningsProtection": "production_earnings_protection",
    "ProductionIndexBaseline": "production_index_baseline",
    "ProductionLEAPSTracker": "production_leaps_tracker",
    "ProductionLottoScanner": "production_lotto_scanner",
    "ProductionMomentumWeeklies": "production_momentum_weeklies",
    "ProductionNarrativeRerateStrategy": "production_narrative_rerate",
    "ProductionSPXCreditSpreads": "production_spx_credit_spreads",
    "ProductionSwingTrading": "production_swing_trading",
    "ProductionWSBDipBot": "production_wsb_dip_bot",
    "ProductionWheelStrategy": "production_wheel_strategy",
}

__all__ = list(_CLASS_TO_MODULE)


def __getattr__(name: str) -> Any:
    module_name = _CLASS_TO_MODULE.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = import_module(f"backend.tradingbot.strategies.production.{module_name}")
    value = getattr(module, name)
    globals()[name] = value
    return value
